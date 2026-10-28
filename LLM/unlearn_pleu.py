"""PLEU for MoE LLMs (DeepSeek-V2-Lite, Qwen1.5-MoE-A2.7B).

LTEU / IDFF / SEPO are computed once on the original model; the MoE layers are then
unlearned sequentially (shallow to deep). At layer l the target expert follows a masked
Bezier curve between its original weights and those of the pre-unlearned endpoint
(NegGrad+ by default), and the control point plus the router of layer l minimize
E_t[L(D_r) - beta L(D_match^(l))] + alpha L_dist. The point t^(l) is chosen on held-out
forget/retain text and written into the model before the next layer.

    python unlearn_baseline.py --model deepseek --forget-dataset wmdp --method neggrad_plus  # endpoint
    python unlearn_pleu.py --model deepseek --forget-dataset wmdp
    python unlearn_pleu.py ... --ablation no_lteu | no_idff | no_sepo
"""

import random

import torch
from tqdm import tqdm

from pleu_llm.common import (
    Timer, add_common_args, endpoint_tensors, new_parser, run_dir, save_delta, set_seed, write_results,
)
from pleu_llm.curve import ParamCurve
from pleu_llm.data import (
    IndexedSubset, TextDataset, cycle, forget_and_retain_texts, lm_loss, make_loader, split_holdout, to_device,
)
from pleu_llm.evaluation import add_eval_args, evaluate_fe_ut
from pleu_llm.masks import mean_gradients, sepo_mask
from pleu_llm.modeling import (
    RouterRecorder, expert_param_names, input_device, load_model_and_tokenizer, moe_layers,
    resolve_model_path, router_param_names, routing_distance,
)
from pleu_llm.routing import assign_samples, layer_targets, sample_routing_weights, select_t

ABLATIONS = ("none", "no_lteu", "no_idff", "no_sepo")


def add_curve_args(parser, lr, epochs):
    parser.add_argument("--endpoint", default=None, help="Pre-unlearned checkpoint dir (default: neggrad_plus run)")
    parser.add_argument("--learning-rate", type=float, default=lr)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-epochs", type=int, default=epochs)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5, help="Unlearning penalty in the curve objective")
    parser.add_argument("--retain-fraction", type=float, default=0.5, help="Share of the retain texts used")
    parser.add_argument("--k-r", type=float, default=0.1)
    parser.add_argument("--k-f", type=float, default=0.5)
    parser.add_argument("--mask-max-batches", type=int, default=None)
    parser.add_argument("--t-grid", default="0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--holdout", type=int, default=32, help="Held-out forget/retain texts for choosing t")
    parser.add_argument("--retain-tol", type=float, default=0.05,
                        help="Max relative increase of held-out retain loss when choosing t")


def prepare_texts(args):
    forget, retain = forget_and_retain_texts(args)
    forget, forget_holdout = split_holdout(forget, args.holdout, args.seed)
    retain, retain_holdout = split_holdout(retain, args.holdout, args.seed + 1)
    retain = random.Random(args.seed).sample(retain, max(1, int(len(retain) * args.retain_fraction)))
    return forget, retain, forget_holdout, retain_holdout


def main():
    parser = add_common_args(new_parser(__doc__))
    add_eval_args(parser)
    add_curve_args(parser, lr=1e-2, epochs=1)
    parser.add_argument("--alpha", type=float, default=1.0, help="Routing distance weight")
    parser.add_argument("--ablation", default="none", choices=ABLATIONS)
    parser.add_argument("--unassigned", default="drop", choices=["drop", "last"])
    args = parser.parse_args()
    set_seed(args.seed)
    if args.endpoint is None:
        args.endpoint = str(run_dir("neggrad_plus", args))
    name = "pleu" if args.ablation == "none" else f"pleu_{args.ablation}"
    out_dir = run_dir(name, args)

    base_path = resolve_model_path(args.model, args.model_root)
    forget, retain, forget_holdout, retain_holdout = prepare_texts(args)
    model, tokenizer = load_model_and_tokenizer(base_path, args.device_map)
    for param in model.parameters():
        param.requires_grad_(False)
    timer, device = Timer(), input_device(model)
    forget_set = TextDataset(forget, tokenizer, args.cutoff_len)
    retain_set = TextDataset(retain, tokenizer, args.cutoff_len)
    batches = lambda texts: list(make_loader(TextDataset(texts, tokenizer, args.cutoff_len),  # noqa: E731
                                             tokenizer, args.batch_size, False))
    holdout_f, holdout_r = batches(forget_holdout), batches(retain_holdout)

    # ---- LTEU + IDFF on the original model
    layers = {layer.layer_idx: layer for layer in moe_layers(model)}
    weights = sample_routing_weights(model, make_loader(forget_set, tokenizer, args.batch_size, False))
    targets, rel = layer_targets(weights, "random" if args.ablation == "no_lteu" else "relevance", args.seed)
    assignment, unassigned = assign_samples(weights, targets, {l: layers[l].top_k for l in targets},
                                            exclusive=args.ablation != "no_idff")
    if args.unassigned == "last" and unassigned:
        assignment[next(reversed(assignment))] += unassigned
    names = {l: expert_param_names(model, layers[l], e) for l, e in targets.items()}

    # ---- SEPO masks on the original model
    masks = {}
    if args.ablation != "no_sepo":
        all_names = [n for layer_names in names.values() for n in layer_names]
        imp_f = mean_gradients(model, make_loader(forget_set, tokenizer, args.batch_size, False), all_names,
                               args.mask_max_batches, "forget importance")
        imp_r = mean_gradients(model, make_loader(retain_set, tokenizer, args.batch_size, False), all_names,
                               args.mask_max_batches, "retain importance")
        for l, layer_names in names.items():
            masks[l] = sepo_mask({n: imp_r[n].abs() for n in layer_names},
                                 {n: imp_f[n].abs() for n in layer_names}, args.k_r, args.k_f)
        del imp_f, imp_r

    endpoint = endpoint_tensors(args.endpoint, [n for layer_names in names.values() for n in layer_names])
    stats = {}
    for l, e in targets.items():
        stats[l] = {"target_expert": e, "top_relevance": round(float(rel[l][e]), 4),
                    "num_assigned": len(assignment[l]), "mask": masks[l][1] if l in masks else None}
        print(f"layer {l}: expert {e} | rel {stats[l]['top_relevance']} | assigned {len(assignment[l])}")
    print(f"unassigned forget samples: {len(unassigned)} ({args.unassigned})")

    # ---- sequential layer-wise curve unlearning
    retain_iter = cycle(make_loader(retain_set, tokenizer, args.batch_size, True))
    t_grid = [float(t) for t in args.t_grid.split(",")]
    for l, e in targets.items():
        if not assignment[l]:
            stats[l]["skipped"] = True
            continue
        curve = ParamCurve(model, {n: endpoint[n] for n in names[l]},
                           masks={n: m for n, m in masks[l][0].items()} if l in masks else None)
        router_names = set(router_param_names(model, layers[l]))
        routers = [p for n, p in model.named_parameters() if n in router_names]
        for param in routers:
            param.requires_grad_(True)
        params = curve.trainable_parameters() + routers
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum)
        loader = make_loader(IndexedSubset(forget_set, assignment[l]), tokenizer, args.batch_size, True)
        model.train()
        for epoch in range(args.num_epochs):
            for batch in tqdm(loader, desc=f"layer {l} epoch {epoch + 1}", leave=False):
                curve.set_t(random.random())
                batch = to_device(batch, device)
                loss_retain = lm_loss(model, to_device(next(retain_iter), device))
                with RouterRecorder(model, [l]) as recorder:
                    loss_forget = lm_loss(model, batch)
                    loss_dist = routing_distance(recorder.probs[l], batch["attention_mask"], [e])
                loss = loss_retain - args.beta * loss_forget + args.alpha * loss_dist
                optimizer.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    optimizer.step()
        t_best, table = select_t(model, curve, t_grid, holdout_f, holdout_r, args.retain_tol)
        stats[l].update({"t": t_best, "t_selection": table, "num_trainable": curve.num_trainable()})
        curve.bake(t_best)
        for param in routers:
            param.requires_grad_(False)
        print(f"layer {l}: t={t_best:.2f}")

    minutes = timer.minutes()
    keep = {n for l in targets for n in names[l] + router_param_names(model, layers[l])}
    changed = {n: p for n, p in model.named_parameters() if n in keep}
    save_delta(out_dir, base_path, changed, method=name, endpoint=args.endpoint, layer_targets=dict(targets))
    results = {"time_min": minutes, "num_unassigned": len(unassigned), "layers": stats, "checkpoint": str(out_dir)}
    if not args.skip_eval:
        results.update(evaluate_fe_ut(model, tokenizer, args))
    write_results(out_dir / "results.json", name, args, results)


if __name__ == "__main__":
    main()
