"""Gradient-based baselines: GA, NegGrad+, and their SEUF (selective expert) variants.

    python unlearn_baseline.py --method neggrad_plus --split-mode class
    python unlearn_baseline.py --method ga --seuf
    python unlearn_baseline.py --method neggrad_plus --seuf --seuf-experts 3     # Table 1 rows
    python unlearn_baseline.py --method neggrad_plus --seuf --seuf-layerwise

SEUF selects the expert(s) with the highest expected routing weight on the forget set
over the whole network, trains only those experts and their routers, and adds the
routing distance constraint that keeps forget samples routed to them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pleu_cv.common import (
    Timer, add_common_args, add_selection_args, clone_state, compute_metrics,
    format_metrics, get_device, load_model, model_path, new_parser, parse_args,
    save_model, selection_score, set_seed, write_results,
)
from pleu_cv.data import build_datasets, build_eval_loaders, build_subsets, load_split_indices, make_loader, split_dir
from pleu_cv.routing import (
    GateRecorder, expert_param_prefix, router_param_prefix, routing_distance,
    sample_routing_weights, select_global_targets, select_layer_targets,
)
from pleu_cv.unlearn import cycle, to_device

DEFAULTS = {  # (lr, epochs) of the original CIFAR-100 scripts
    ("ga", False): (5e-6, 5),
    ("neggrad_plus", False): (5e-4, 5),
    ("ga", True): (5e-4, 5),
    ("neggrad_plus", True): (5e-4, 5),
}


def freeze_to_experts(model, targets):
    """Train only the selected experts and the routers of their layers."""
    prefixes = []
    for layer_idx, experts in targets.items():
        prefixes.append(router_param_prefix(layer_idx))
        prefixes.extend(expert_param_prefix(layer_idx, expert_idx) for expert_idx in experts)
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)
        if param.requires_grad:
            trainable.append(param)
    return trainable


def method_name(args):
    if not args.seuf:
        return args.method
    name = f"seuf_{args.method}"
    if args.seuf_layerwise:
        return name + "_layerwise"
    return name if args.seuf_experts == 1 else f"{name}_{args.seuf_experts}experts"


def main():
    parser = add_common_args(new_parser(__doc__))
    add_selection_args(parser)
    parser.add_argument("--method", default="neggrad_plus", choices=["ga", "neggrad_plus"])
    parser.add_argument("--seuf", action="store_true", help="Selective-expert unlearning (SEUF)")
    parser.add_argument("--seuf-experts", type=int, default=1, help="Number of globally selected experts")
    parser.add_argument("--seuf-layerwise", action="store_true", help="One target expert in every MoE layer")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.5, help="NegGrad+ forget weight")
    parser.add_argument("--alpha", type=float, default=1.0, help="SEUF routing distance weight")
    args = parse_args(parser)
    default_lr, default_epochs = DEFAULTS[(args.method, args.seuf)]
    args.lr = default_lr if args.lr is None else args.lr
    args.epochs = default_epochs if args.epochs is None else args.epochs

    set_seed(args.seed)
    device = get_device()
    trainset, trainset_eval, testset, _ = build_datasets(args.dataset, args.data_root, args.output_root, args.image_size)
    subsets = build_subsets(trainset, trainset_eval, testset,
                            load_split_indices(split_dir(args.output_root, args.dataset), args.split_mode))
    eval_loaders = build_eval_loaders(subsets, args.eval_batch_size, args.num_workers)
    forget_loader = make_loader(subsets["forget_train"], args.batch_size, args.num_workers, shuffle=True)
    retain_loader = make_loader(subsets["retain_train"], args.batch_size, args.num_workers, shuffle=True)

    model = load_model(args, args.origin_path, device)
    targets = None
    if args.seuf:
        weights = sample_routing_weights(model, eval_loaders["forget_train_eval"], device)
        if args.seuf_layerwise:
            layer_targets, _ = select_layer_targets(weights)
            targets = {layer: [expert] for layer, expert in layer_targets.items()}
        else:
            targets, _ = select_global_targets(weights, args.seuf_experts)
        print("SEUF target experts:", dict(targets))
        trainable = freeze_to_experts(model, targets)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    else:
        trainable = list(model.parameters())
        optimizer = torch.optim.Adam(trainable, lr=args.lr)

    timer, best = Timer(), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        retain_iter = cycle(retain_loader)
        for batch_f in tqdm(forget_loader, desc=f"{method_name(args)} epoch {epoch}", leave=False):
            inputs_f, targets_f = to_device(batch_f, device)
            with GateRecorder(model, targets.keys() if targets else []) as recorder:
                loss_forget = F.cross_entropy(model(inputs_f)[0], targets_f)
                loss_dist = sum(
                    routing_distance(recorder.probs[layer], experts) for layer, experts in targets.items()
                ) / len(targets) if targets else 0.0
            if args.method == "neggrad_plus":
                inputs_r, targets_r = to_device(next(retain_iter), device)
                loss = F.cross_entropy(model(inputs_r)[0], targets_r) - args.beta * loss_forget
            else:
                loss = -loss_forget
            loss = loss + args.alpha * loss_dist
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.method == "neggrad_plus" or args.seuf:
                nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        metrics = compute_metrics(model, eval_loaders, device)
        score = selection_score(metrics, args)
        print(f"epoch {epoch} | {format_metrics(metrics)} | selection {score:.3f}")
        if best is None or score < best["score"]:
            best = {"epoch": epoch, "score": score, "metrics": metrics, "state": clone_state(model)}

    minutes = timer.minutes()
    name = method_name(args)
    save_model(model_path(name, args), best["state"], epoch=best["epoch"], metrics=best["metrics"],
               targets={int(k): v for k, v in (targets or {}).items()})
    print(f"best epoch {best['epoch']} | {format_metrics(best['metrics'])} | {minutes:.2f} min")
    write_results(name, args, {"metrics": best["metrics"], "best_epoch": best["epoch"], "time_min": minutes,
                               "targets": {int(k): v for k, v in (targets or {}).items()}})


if __name__ == "__main__":
    main()
