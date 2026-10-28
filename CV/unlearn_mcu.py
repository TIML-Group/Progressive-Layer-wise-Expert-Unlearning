"""MCU baseline: full-model mode connectivity unlearning.

A quadratic Bezier curve connects the original model and a pre-unlearned endpoint
(NegGrad+ by default). The control point of the tensors selected by the MCU mask is
trained with  E_t[ L(D_r) - beta * L(D_f) ],  and t is picked on a grid.

    python unlearn_baseline.py --method neggrad_plus --split-mode class   # endpoint
    python unlearn_mcu.py --split-mode class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from pleu_cv.common import (
    Timer, add_common_args, add_selection_args, compute_metrics, format_metrics,
    get_device, load_model, model_path, new_parser, parse_args, save_model,
    selection_score, set_seed, write_results,
)
from pleu_cv.curve import BezierCurve, CurveModel, FixedPointModel, t_grid
from pleu_cv.data import (
    build_datasets, build_eval_loaders, build_subsets, fixed_fraction,
    load_split_indices, make_loader, split_dir,
)
from pleu_cv.masks import mcu_tensor_mask, mean_gradients
from pleu_cv.unlearn import (
    add_beta_args, add_curve_args, batch_accuracy, beta_references, current_beta,
    cycle, sample_t, select_curve_point, to_device,
)


def main():
    parser = add_common_args(new_parser(__doc__))
    add_selection_args(parser)
    add_curve_args(parser, lr=1e-4, epochs=5)
    add_beta_args(parser)
    args = parse_args(parser)
    if args.endpoint_path is None:
        args.endpoint_path = model_path("neggrad_plus", args)

    set_seed(args.seed)
    device = get_device()
    trainset, trainset_eval, testset, _ = build_datasets(args.dataset, args.data_root, args.output_root, args.image_size)
    indices = load_split_indices(split_dir(args.output_root, args.dataset), args.split_mode)
    subsets = build_subsets(trainset, trainset_eval, testset, indices)
    eval_loaders = build_eval_loaders(subsets, args.eval_batch_size, args.num_workers)
    retain_idx = fixed_fraction(indices["retain_train"], args.retain_fraction, args.seed)
    forget_loader = make_loader(subsets["forget_train"], args.batch_size, args.num_workers, shuffle=True)
    retain_loader = make_loader(Subset(trainset, retain_idx), args.batch_size, args.num_workers, shuffle=True)

    timer = Timer()
    model = load_model(args, args.origin_path, device)
    endpoint = load_model(args, args.endpoint_path, device)
    endpoint_metrics = compute_metrics(endpoint, eval_loaders, device, args.select_max_batches)
    refs = beta_references(args, endpoint_metrics)
    print(f"endpoint | {format_metrics(endpoint_metrics)}")

    names = [name for name, _ in model.named_parameters()]
    grad_loader = lambda data: make_loader(data, args.eval_batch_size, args.num_workers)  # noqa: E731
    retain_grads = mean_gradients(model, grad_loader(Subset(trainset_eval, retain_idx)), device, names,
                                  args.mask_max_batches, "retain grad")
    forget_grads = mean_gradients(model, grad_loader(subsets["forget_train_eval"]), device, names,
                                  args.mask_max_batches, "forget grad")
    mask, mask_stats = mcu_tensor_mask(retain_grads, forget_grads, args.k_f, args.k_r)
    del retain_grads, forget_grads
    print(f"MCU mask: {mask_stats}")

    for param in model.parameters():
        param.requires_grad_(False)
    start = {name: param for name, param in model.named_parameters()}
    end = {name: param for name, param in endpoint.named_parameters()}
    curve = BezierCurve(start, end, trainable=[name for name, keep in mask.items() if keep]).to(device)
    del endpoint
    curve_model = CurveModel(model, curve)
    params = curve.trainable_parameters()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    t_values = t_grid(args.eval_start_t, args.eval_end_t, args.eval_num_points)

    best = None
    for epoch in range(1, args.epochs + 1):
        model.eval()
        retain_iter = cycle(retain_loader)
        for batch_f in tqdm(forget_loader, desc=f"MCU epoch {epoch}", leave=False):
            inputs_f, targets_f = to_device(batch_f, device)
            inputs_r, targets_r = to_device(next(retain_iter), device)
            t = sample_t(args.train_t_min, args.train_t_max, device)
            logits_r = curve_model(inputs_r, t)[0]
            logits_f = curve_model(inputs_f, t)[0]
            beta = current_beta(args, batch_accuracy(logits_r, targets_r), batch_accuracy(logits_f, targets_f), refs)
            loss = F.cross_entropy(logits_r, targets_r) - beta * F.cross_entropy(logits_f, targets_f)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()

        point = select_curve_point(lambda t: FixedPointModel(curve_model, t), eval_loaders, device, args,
                                   t_values, compute_metrics, selection_score)
        print(f"epoch {epoch} | t={point['t']:.2f} | {format_metrics(point['metrics'])}")
        if best is None or point["score"] < best["score"]:
            best = {**point, "epoch": epoch, "state": {**dict(model.state_dict()),
                                                        **curve.export(torch.tensor(point["t"], device=device))}}

    minutes = timer.minutes()
    model.load_state_dict(best["state"])
    metrics = compute_metrics(model, eval_loaders, device)
    print(f"best epoch {best['epoch']} t={best['t']:.2f} | {format_metrics(metrics)} | {minutes:.2f} min")
    save_model(model_path("mcu", args), best["state"], epoch=best["epoch"], t=best["t"], metrics=metrics)
    write_results("mcu", args, {"metrics": metrics, "best_epoch": best["epoch"], "t": best["t"],
                                "time_min": minutes, "mask": mask_stats, "endpoint": str(args.endpoint_path)})


if __name__ == "__main__":
    main()
