"""Train the original MoE-ViT (``origin``) or the retrain-from-scratch reference (``rt``).

    python train.py --dataset cifar100                      # -> models/origin.pth
    python train.py --dataset cifar100 --retain-only --split-mode class   # RT
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from pleu_cv.common import (
    Timer, add_common_args, build_model, clone_state, compute_metrics, evaluate,
    format_metrics, get_device, model_path, new_parser, parse_args, save_model,
    set_seed, write_results,
)
from pleu_cv.data import (
    DATASET_CONFIGS, SPLIT_MODES, build_datasets, build_eval_loaders, build_subsets,
    load_split_indices, make_loader, split_dir,
)


def rand_bbox(height, width, lam):
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    cy, cx = np.random.randint(height), np.random.randint(width)
    y1, y2 = np.clip(cy - cut_h // 2, 0, height), np.clip(cy + cut_h // 2, 0, height)
    x1, x2 = np.clip(cx - cut_w // 2, 0, width), np.clip(cx + cut_w // 2, 0, width)
    return y1, y2, x1, x2


def mix_batch(inputs, targets, cutmix_alpha, mixup_alpha):
    """CutMix (CIFAR-100 recipe) or MixUp (ImageNet-100 recipe); returns inputs, y_a, y_b, lam."""
    perm = torch.randperm(inputs.size(0), device=inputs.device)
    if cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        y1, y2, x1, x2 = rand_bbox(inputs.size(2), inputs.size(3), lam)
        inputs = inputs.clone()
        inputs[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(inputs.size(2) * inputs.size(3))
        return inputs, targets, targets[perm], lam
    if mixup_alpha > 0:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        return lam * inputs + (1.0 - lam) * inputs[perm], targets, targets[perm], lam
    return inputs, targets, targets, 1.0


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self, model):
        state = clone_state(model)
        state.update({k: v.clone() for k, v in self.shadow.items()})
        return state


def main():
    parser = add_common_args(new_parser(__doc__))
    parser.add_argument("--retain-only", action="store_true", help="Train on D_r only (RT baseline)")
    for key, value in DATASET_CONFIGS["cifar100"]["train"].items():
        flag = "--" + key.replace("_", "-")
        if flag == "--batch-size":
            continue
        if isinstance(value, bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            parser.add_argument(flag, type=type(value), default=None)
    args = parse_args(parser)
    for key, value in DATASET_CONFIGS[args.dataset]["train"].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    set_seed(args.seed)
    device = get_device()
    trainset, trainset_eval, testset, _ = build_datasets(args.dataset, args.data_root, args.output_root, args.image_size)
    eval_loaders, select_key = None, "full_test"
    train_data = trainset
    if args.retain_only:
        indices = load_split_indices(split_dir(args.output_root, args.dataset), args.split_mode)
        subsets = build_subsets(trainset, trainset_eval, testset, indices)
        eval_loaders = build_eval_loaders(subsets, args.eval_batch_size, args.num_workers)
        train_data, select_key = Subset(trainset, indices["retain_train"]), "retain_test"
        select_loader = eval_loaders["retain_test"]
    else:
        select_loader = make_loader(testset, args.eval_batch_size, args.num_workers)
    train_loader = make_loader(train_data, args.batch_size, args.num_workers, shuffle=True, drop_last=True)

    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    eval_model = build_model(args, device)

    timer, best_acc, best_state, best_epoch = Timer(), -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for inputs, targets in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            inputs, y_a, y_b, lam = mix_batch(inputs, targets, args.cutmix_alpha, args.mixup_alpha)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, aux_loss = model(inputs)
                ce = lam * F.cross_entropy(logits, y_a, label_smoothing=args.label_smoothing)
                ce = ce + (1.0 - lam) * F.cross_entropy(logits, y_b, label_smoothing=args.label_smoothing)
                loss = ce + args.aux_weight * aux_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            running += loss.item() * targets.size(0)
            seen += targets.size(0)
        scheduler.step()

        eval_model.load_state_dict(ema.state_dict(model) if ema is not None else model.state_dict())
        acc = evaluate(eval_model, select_loader, device, select_key)
        print(f"epoch {epoch} | loss {running / max(seen, 1):.4f} | {select_key} {acc:.2f} | {timer.minutes():.1f} min")
        if acc > best_acc:
            best_acc, best_epoch, best_state = acc, epoch, clone_state(eval_model)

    method = "rt" if args.retain_only else "origin"
    path = model_path(method, args) if args.retain_only else args.origin_path
    eval_model.load_state_dict(best_state)
    payload = {"best_epoch": best_epoch, f"best_{select_key}_acc": best_acc, "time_min": timer.minutes()}
    save_model(path, best_state, epoch=best_epoch, args={k: str(v) for k, v in vars(args).items()})
    if args.retain_only:
        payload["metrics"] = compute_metrics(eval_model, eval_loaders, device)
        print(format_metrics(payload["metrics"]))
        write_results(method, args, payload)
        return
    # Origin row of Table 2: forget/retain accuracies under every available split.
    for mode in SPLIT_MODES:
        try:
            indices = load_split_indices(split_dir(args.output_root, args.dataset), mode)
        except FileNotFoundError:
            continue
        loaders = build_eval_loaders(build_subsets(trainset, trainset_eval, testset, indices),
                                     args.eval_batch_size, args.num_workers)
        args.split_mode = mode
        metrics = compute_metrics(eval_model, loaders, device)
        print(f"origin [{mode}] | {format_metrics(metrics)}")
        write_results(method, args, {**payload, "metrics": metrics})


if __name__ == "__main__":
    main()
