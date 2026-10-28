"""Unified relearning protocol (Section 4.1).

Fine-tunes an unlearned model for a few steps (10 by default) on benign ImageNet
classes that are disjoint from ImageNet-100, using a fresh classification head, then
restores the original head and reports FA/RA. ``--trace`` also records the train FA
after every step (Fig. 3).

    python relearn.py --method pleu --split-mode class --trace
    python relearn.py --model-path outputs/cifar100/models/mcu_class_s42.pth
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from pleu_cv.common import (
    Timer, add_common_args, compute_metrics, evaluate, format_metrics, get_device,
    load_model, model_path, new_parser, parse_args, set_seed, write_results,
)
from pleu_cv.data import (
    build_datasets, build_eval_loaders, build_relearn_dataset, build_subsets,
    load_split_indices, make_loader, split_dir,
)
from pleu_cv.unlearn import cycle, to_device
from train import mix_batch


def main():
    parser = add_common_args(new_parser(__doc__))
    parser.add_argument("--method", default=None, help="Run name of the unlearned model, e.g. pleu, mcu, seuf_ga")
    parser.add_argument("--model-path", default=None, help="Explicit checkpoint (overrides --method)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--trace", action="store_true", help="Evaluate train FA after every step")
    args = parse_args(parser)
    if args.model_path is None:
        if args.method is None:
            parser.error("give --method or --model-path")
        args.model_path = model_path(args.method, args)
    label = args.method or "custom"

    set_seed(args.seed)
    device = get_device()
    trainset, trainset_eval, testset, train_tf = build_datasets(args.dataset, args.data_root, args.output_root, args.image_size)
    subsets = build_subsets(trainset, trainset_eval, testset,
                            load_split_indices(split_dir(args.output_root, args.dataset), args.split_mode))
    eval_loaders = build_eval_loaders(subsets, args.eval_batch_size, args.num_workers)
    relearn_set = build_relearn_dataset(args.data_root, train_tf)
    relearn_iter = cycle(make_loader(relearn_set, args.batch_size, args.num_workers, shuffle=True, drop_last=True))
    print(f"relearning set: {len(relearn_set)} images, {len(relearn_set.classes)} classes")

    model = load_model(args, args.model_path, device)
    before = compute_metrics(model, eval_loaders, device)
    print(f"before relearning | {format_metrics(before)}")

    original_head = copy.deepcopy(model.head)
    model.head = nn.Linear(model.head.in_features, len(relearn_set.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.steps, 1))

    def forget_train_acc():
        relearn_head, model.head = model.head, original_head
        acc = evaluate(model, eval_loaders["forget_train_eval"], device, "FA train")
        model.head = relearn_head
        return acc

    timer, trace = Timer(), [before["forget_train_acc"]] if args.trace else None
    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets = to_device(next(relearn_iter), device)
        inputs, y_a, y_b, lam = mix_batch(inputs, targets, args.cutmix_alpha, 0.0)
        logits = model(inputs)[0]
        loss = lam * F.cross_entropy(logits, y_a, label_smoothing=args.label_smoothing)
        loss = loss + (1.0 - lam) * F.cross_entropy(logits, y_b, label_smoothing=args.label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if args.trace:
            trace.append(forget_train_acc())
            print(f"step {step} | loss {loss.item():.4f} | FA(train) {trace[-1]:.2f}")

    model.head = original_head
    after = compute_metrics(model, eval_loaders, device)
    print(f"after {args.steps} relearning steps | {format_metrics(after)}")
    write_results(f"relearn_{label}", args, {"model_path": str(args.model_path), "before": before,
                                             "after": after, "fa_train_trace": trace,
                                             "time_min": timer.minutes()})


if __name__ == "__main__":
    main()
