"""PLEU: Progressive Layer-wise Expert Unlearning (Section 3).

1. LTEU   pick the target expert E_*^(l) of every MoE layer by its expected routing
          weight on the forget set (original model).
2. IDFF   assign every forget sample to the first layer whose router sends it to
          that layer's target expert (top-K of the token-averaged routing weights).
3. SEPO   per target expert, protect the top-k_r entries by retain-gradient magnitude
          and activate the top-k_f of the rest by forget-gradient magnitude.
4. MCUE   unlearn the layers sequentially, shallow to deep: a masked Bezier curve between
          the original and the pre-unlearned (NegGrad+) expert, with the control point and
          the layer's router trained on  E_t[L(D_r) - beta L(D_match^(l))] + alpha L_dist.
          t^(l) is chosen on a grid and baked into the model before the next layer.

    python unlearn_pleu.py --split-mode class
    python unlearn_pleu.py --split-mode class --ablation no_lteu   # random expert per layer
    python unlearn_pleu.py --split-mode class --ablation no_idff   # no inter-layer filtering
    python unlearn_pleu.py --split-mode class --ablation no_sepo   # update the whole expert
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from pleu_cv.common import (
    Timer, add_common_args, add_selection_args, clone_state, compute_metrics,
    format_metrics, get_device, load_model, model_path, new_parser, parse_args,
    save_model, selection_score, set_seed, write_results,
)
from pleu_cv.curve import BezierCurve, CurveModel, FixedPointModel, t_grid
from pleu_cv.data import (
    build_datasets, build_eval_loaders, build_subsets, fixed_fraction,
    load_split_indices, make_loader, split_dir,
)
from pleu_cv.masks import mean_gradients, sepo_mask
from pleu_cv.routing import (
    GateRecorder, assign_samples_to_layers, expert_param_prefix, moe_layers,
    router_param_prefix, routing_distance, sample_routing_weights, select_layer_targets,
)
from pleu_cv.unlearn import (
    add_beta_args, add_curve_args, batch_accuracy, beta_references, current_beta,
    cycle, sample_t, select_curve_point, to_device,
)

ABLATIONS = ("none", "no_lteu", "no_idff", "no_sepo")


def expert_names(model, layer_idx, expert_idx):
    prefix = expert_param_prefix(layer_idx, expert_idx)
    return [name for name, _ in model.named_parameters() if name.startswith(prefix)]


def router_params(model, layer_idx):
    prefix = router_param_prefix(layer_idx)
    return [param for name, param in model.named_parameters() if name.startswith(prefix)]


def localize(args, model, subsets, indices, eval_loaders, trainset_eval, retain_idx, device):
    """LTEU + IDFF + SEPO, all computed once on the original model."""
    weights = sample_routing_weights(model, eval_loaders["forget_train_eval"], device, "forget routing")
    strategy = "random" if args.ablation == "no_lteu" else "relevance"
    targets, relevance = select_layer_targets(weights, strategy, args.seed)
    top_k = {layer_idx: k for layer_idx, _, k in moe_layers(model)}
    assignment, unassigned = assign_samples_to_layers(weights, targets, top_k, exclusive=args.ablation != "no_idff")
    if args.unassigned == "last" and len(unassigned) > 0:
        last = next(reversed(assignment))
        assignment[last] = torch.cat([assignment[last], unassigned]).sort().values

    masks = OrderedDict()
    if args.ablation != "no_sepo":
        names = [n for layer_idx, expert_idx in targets.items() for n in expert_names(model, layer_idx, expert_idx)]
        loader = lambda data: make_loader(data, args.eval_batch_size, args.num_workers)  # noqa: E731
        retain_imp = mean_gradients(model, loader(Subset(trainset_eval, retain_idx)), device, names,
                                    args.mask_max_batches, "retain importance")
        forget_imp = mean_gradients(model, loader(subsets["forget_train_eval"]), device, names,
                                    args.mask_max_batches, "forget importance")
        for layer_idx, expert_idx in targets.items():
            layer_names = expert_names(model, layer_idx, expert_idx)
            masks[layer_idx] = sepo_mask(
                {n: retain_imp[n].abs() for n in layer_names},
                {n: forget_imp[n].abs() for n in layer_names},
                args.k_r, args.k_f,
            )

    stats = OrderedDict()
    for layer_idx, expert_idx in targets.items():
        stats[layer_idx] = {
            "target_expert": expert_idx,
            "relevance": [round(v, 4) for v in relevance[layer_idx].tolist()],
            "num_assigned": int(len(assignment[layer_idx])),
            "mask": masks[layer_idx][1] if layer_idx in masks else None,
        }
        print(f"layer {layer_idx}: expert {expert_idx} | rel {stats[layer_idx]['relevance']} | "
              f"assigned {stats[layer_idx]['num_assigned']}")
    print(f"unassigned forget samples: {len(unassigned)} ({args.unassigned})")
    forget_idx = indices["forget_train"]
    layer_data = OrderedDict((l, forget_idx[a.numpy()]) for l, a in assignment.items())
    return targets, layer_data, {l: m[0] for l, m in masks.items()}, stats, int(len(unassigned))


def unlearn_layer(args, model, endpoint_state, layer_idx, expert_idx, mask, forget_data, retain_loader,
                  eval_loaders, refs, device):
    names = expert_names(model, layer_idx, expert_idx)
    params = dict(model.named_parameters())
    curve = BezierCurve({n: params[n] for n in names}, {n: endpoint_state[n] for n in names},
                        entry_mask=mask).to(device)
    routers = router_params(model, layer_idx)
    for param in routers:
        param.requires_grad_(True)
    trainable = curve.trainable_parameters() + routers
    optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, -(-len(forget_data) // args.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    curve_model = CurveModel(model, curve)
    forget_loader = make_loader(forget_data, args.batch_size, args.num_workers, shuffle=True)
    retain_iter = cycle(retain_loader)
    t_values = t_grid(args.eval_start_t, args.eval_end_t, args.eval_num_points)

    best = None
    for epoch in range(1, args.epochs + 1):
        model.eval()
        for batch_f in tqdm(forget_loader, desc=f"layer {layer_idx} epoch {epoch}", leave=False):
            inputs_f, targets_f = to_device(batch_f, device)
            inputs_r, targets_r = to_device(next(retain_iter), device)
            t = sample_t(args.train_t_min, args.train_t_max, device)
            logits_r = curve_model(inputs_r, t)[0]
            with GateRecorder(model, [layer_idx]) as recorder:
                logits_f = curve_model(inputs_f, t)[0]
                loss_dist = routing_distance(recorder.probs[layer_idx], [expert_idx])
            beta = current_beta(args, batch_accuracy(logits_r, targets_r), batch_accuracy(logits_f, targets_f), refs)
            loss = (F.cross_entropy(logits_r, targets_r) - beta * F.cross_entropy(logits_f, targets_f)
                    + args.alpha * loss_dist)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

        point = select_curve_point(lambda t: FixedPointModel(curve_model, t), eval_loaders, device, args,
                                   t_values, compute_metrics, selection_score)
        print(f"layer {layer_idx} epoch {epoch} | t={point['t']:.2f} | {format_metrics(point['metrics'])}")
        if best is None or point["score"] < best["score"]:
            best = {**point, "epoch": epoch,
                    "expert": curve.export(torch.tensor(point["t"], device=device)),
                    "router": [param.detach().clone() for param in routers]}

    with torch.no_grad():
        for name, value in best["expert"].items():
            params[name].copy_(value)
        for param, value in zip(routers, best["router"]):
            param.copy_(value)
    for param in routers:
        param.requires_grad_(False)
    return {"t": best["t"], "epoch": best["epoch"], "metrics": best["metrics"],
            "num_trainable": curve.num_trainable()}


def main():
    parser = add_common_args(new_parser(__doc__))
    add_selection_args(parser)
    add_curve_args(parser, lr=1e-2, epochs=2, eval_start_t=0.5)
    add_beta_args(parser)
    parser.set_defaults(eval_num_points=6)
    parser.add_argument("--alpha", type=float, default=1.0, help="Routing distance weight")
    parser.add_argument("--ablation", default="none", choices=ABLATIONS)
    parser.add_argument("--unassigned", default="drop", choices=["drop", "last"],
                        help="Forget samples never routed to a target expert: unused (paper) or added to the last layer")
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
    retain_loader = make_loader(Subset(trainset, retain_idx), args.batch_size, args.num_workers, shuffle=True)

    timer = Timer()
    model = load_model(args, args.origin_path, device)
    endpoint = load_model(args, args.endpoint_path, device)
    refs = beta_references(args, compute_metrics(endpoint, eval_loaders, device, args.select_max_batches))
    endpoint_state = clone_state(endpoint)
    del endpoint
    for param in model.parameters():
        param.requires_grad_(False)

    targets, layer_data, masks, stats, num_unassigned = localize(
        args, model, subsets, indices, eval_loaders, trainset_eval, retain_idx, device)

    for layer_idx, expert_idx in targets.items():
        if len(layer_data[layer_idx]) == 0:
            print(f"layer {layer_idx}: no assigned forget sample, skipped")
            stats[layer_idx]["skipped"] = True
            continue
        stats[layer_idx].update(unlearn_layer(
            args, model, endpoint_state, layer_idx, expert_idx, masks.get(layer_idx),
            Subset(trainset, layer_data[layer_idx]), retain_loader, eval_loaders, refs, device))

    minutes = timer.minutes()
    metrics = compute_metrics(model, eval_loaders, device)
    name = "pleu" if args.ablation == "none" else f"pleu_{args.ablation}"
    print(f"{name} | {format_metrics(metrics)} | {minutes:.2f} min")
    save_model(model_path(name, args), model.state_dict(), metrics=metrics, layer_targets=dict(targets))
    write_results(name, args, {"metrics": metrics, "time_min": minutes, "num_unassigned": num_unassigned,
                               "layers": {int(k): v for k, v in stats.items()},
                               "endpoint": str(args.endpoint_path)})


if __name__ == "__main__":
    main()
