"""MCU baseline for MoE LLMs: full-model mode connectivity unlearning.

Every weight that differs between the original model and the pre-unlearned endpoint
(full-model NegGrad+ by default, i.e. every weight) is put on a quadratic Bezier curve;
the MCU tensor mask decides which control points are trained on E_t[L(D_r) - beta L(D_f)]
with MCU's adaptive beta (references: next-token accuracies of the endpoint).
Start, end and control point of the whole model are held at once (plus gradients and
momentum of the selected half), about 6-7x the bf16 model size, so spread it over the GPUs
of a node with ``--device-map auto``; ``--mid-dtype bfloat16`` halves the control point.

    python unlearn_mcu.py --model deepseek --forget-dataset wmdp
"""

import random

import torch
from tqdm import tqdm

from pleu_llm.common import (
    Timer, add_common_args, endpoint_tensors, new_parser, run_dir, save_delta, set_seed, write_results,
)
from pleu_llm.curve import ParamCurve
from pleu_llm.data import TextDataset, cycle, lm_loss_and_accuracy, make_loader, to_device
from pleu_llm.evaluation import add_eval_args, evaluate_fe_ut
from pleu_llm.masks import mcu_tensor_mask, mean_gradients
from pleu_llm.modeling import input_device, load_model_and_tokenizer, resolve_model_path
from pleu_llm.routing import select_t
from pleu_llm.training import Accumulator, current_beta
from unlearn_pleu import add_curve_args, beta_references, prepare_texts


@torch.no_grad()
def differing_tensors(model, endpoint):
    """Endpoint tensors (CPU) that differ from the original model."""
    params = dict(model.named_parameters())
    return {name: value for name, value in endpoint.items()
            if name in params and not torch.equal(params[name].detach().cpu(), value)}


def main():
    parser = add_common_args(new_parser(__doc__))
    add_eval_args(parser)
    add_curve_args(parser, lr=1e-3, epochs=1)
    parser.add_argument("--mid-dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    set_seed(args.seed)
    if args.endpoint is None:
        args.endpoint = str(run_dir("neggrad_plus", args))
    if args.mask_max_batches is None:
        args.mask_max_batches = 64
    out_dir = run_dir("mcu", args)

    base_path = resolve_model_path(args.model, args.model_root)
    forget, retain, forget_holdout, retain_holdout = prepare_texts(args)
    model, tokenizer = load_model_and_tokenizer(base_path, args.device_map)
    for param in model.parameters():
        param.requires_grad_(False)
    timer, device = Timer(), input_device(model)
    end = differing_tensors(model, endpoint_tensors(args.endpoint))
    print(f"{len(end)} tensors differ between the original model and the endpoint")

    forget_set = TextDataset(forget, tokenizer, args.cutoff_len)
    retain_set = TextDataset(retain, tokenizer, args.cutoff_len)
    refs = beta_references(args, model, base_path, tokenizer, forget_set, retain_set)
    curve_names = list(end)
    grad = lambda data, desc: mean_gradients(  # noqa: E731
        model, make_loader(data, tokenizer, args.batch_size, False), curve_names, args.mask_max_batches, desc,
        dtype=torch.bfloat16)
    retain_grads = grad(retain_set, "retain grad")
    forget_grads = grad(forget_set, "forget grad")
    mask, mask_stats = mcu_tensor_mask(retain_grads, forget_grads, args.k_f, args.k_r)
    del retain_grads, forget_grads
    print(f"MCU mask: {mask_stats}")

    curve = ParamCurve(model, end, trainable=[n for n, keep in mask.items() if keep],
                       mid_dtype=getattr(torch, args.mid_dtype))
    del end
    params = curve.trainable_parameters()
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum)
    acc = Accumulator(params, optimizer, None, args.grad_accum_steps, args.max_grad_norm)
    retain_iter = cycle(make_loader(retain_set, tokenizer, args.batch_size, True))
    forget_loader = make_loader(forget_set, tokenizer, args.batch_size, True)
    model.train()
    betas = []
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(tqdm(forget_loader, desc=f"MCU epoch {epoch + 1}"), start=1):
            curve.set_t(random.random())
            loss_retain, acc_retain = lm_loss_and_accuracy(model, to_device(next(retain_iter), device))
            loss_forget, acc_forget = lm_loss_and_accuracy(model, to_device(batch, device))
            betas.append(current_beta(args, acc_retain, acc_forget, refs))
            acc.backward(loss_retain.to(loss_forget.device) - betas[-1] * loss_forget, last=step == len(forget_loader))

    holdout = lambda texts: list(make_loader(TextDataset(texts, tokenizer, args.cutoff_len),  # noqa: E731
                                             tokenizer, args.batch_size, False))
    t_best, table = select_t(model, curve, [float(t) for t in args.t_grid.split(",")],
                             holdout(forget_holdout), holdout(retain_holdout), args.retain_tol)
    changed = curve.bake(t_best)
    minutes = timer.minutes()
    print(f"MCU t={t_best:.2f} | {minutes:.1f} min")
    save_delta(out_dir, base_path, changed, method="mcu", endpoint=args.endpoint, t=t_best)
    results = {"time_min": minutes, "t": t_best, "t_selection": table, "mask": mask_stats, "beta_references": refs,
               "mean_beta": sum(betas) / max(len(betas), 1), "checkpoint": str(out_dir)}
    if not args.skip_eval:
        results.update(evaluate_fe_ut(model, tokenizer, args))
    write_results(out_dir / "results.json", "mcu", args, results)


if __name__ == "__main__":
    main()
