"""Pieces shared by the unlearning scripts."""

import torch


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def batch_accuracy(logits, targets):
    return 100.0 * logits.argmax(dim=1).eq(targets).float().mean().item()


def relative_gap(current, base):
    if abs(base) < 1e-8:
        return 0.0 if current <= base else float("inf")
    return (current - base) / base


def adaptive_beta(retain_acc, forget_acc, retain_base, forget_base, beta_low, beta_high):
    """MCU's adaptive unlearning penalty: stop pushing once the forget accuracy reaches its
    reference, push weakly when retain accuracy lags more than forget accuracy exceeds."""
    if forget_acc <= forget_base:
        return 0.0
    if relative_gap(forget_acc, forget_base) < relative_gap(retain_acc, retain_base):
        return beta_low
    return beta_high


def add_beta_args(parser, beta=0.15):
    parser.add_argument("--beta-mode", default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--beta", type=float, default=beta, help="Fixed unlearning penalty")
    parser.add_argument("--beta-low", type=float, default=0.01)
    parser.add_argument("--beta-high", type=float, default=0.5)
    parser.add_argument("--retain-base", type=float, default=None,
                        help="Adaptive beta retain reference (default: endpoint model train RA)")
    parser.add_argument("--forget-base", type=float, default=None,
                        help="Adaptive beta forget reference (default: 0 class-wise, endpoint train FA random)")


def beta_references(args, endpoint_metrics):
    retain_base = args.retain_base if args.retain_base is not None else endpoint_metrics["retain_train_acc"]
    if args.forget_base is not None:
        forget_base = args.forget_base
    elif args.split_mode == "class":
        forget_base = 0.0
    else:
        forget_base = endpoint_metrics["forget_train_acc"]
    return retain_base, forget_base


def current_beta(args, retain_acc, forget_acc, refs):
    if args.beta_mode == "fixed":
        return args.beta
    return adaptive_beta(retain_acc, forget_acc, refs[0], refs[1], args.beta_low, args.beta_high)


def to_device(batch, device):
    inputs, targets = batch
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def sample_t(low, high, device):
    return torch.empty(1, device=device).uniform_(low, high)


def add_curve_args(parser, lr, epochs, eval_start_t=0.0):
    parser.add_argument("--endpoint-path", default=None,
                        help="Pre-unlearned endpoint theta_p (default: neggrad_plus model of this split/seed)")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--retain-fraction", type=float, default=0.5, help="Share of D_r used for curve training")
    parser.add_argument("--train-t-min", type=float, default=0.0)
    parser.add_argument("--train-t-max", type=float, default=1.0)
    parser.add_argument("--eval-start-t", type=float, default=eval_start_t)
    parser.add_argument("--eval-end-t", type=float, default=1.0)
    parser.add_argument("--eval-num-points", type=int, default=11)
    parser.add_argument("--select-max-batches", type=int, default=None,
                        help="Evaluate at most this many batches per split when choosing t")
    parser.add_argument("--k-r", type=float, default=0.1, help="Retain-protected fraction")
    parser.add_argument("--k-f", type=float, default=0.5, help="Forget-activated fraction")
    parser.add_argument("--mask-max-batches", type=int, default=None,
                        help="Batches per set used for gradient importance (default: all)")


def select_curve_point(model_at_t, eval_loaders, device, args, t_values, compute_metrics, selection_score):
    """MCU point selection: evaluate a grid of t and keep the best selection score.

    ``model_at_t(t)`` returns a module evaluating the curve at ``t``.
    """
    best = None
    for t in t_values:
        t_tensor = torch.tensor(float(t), device=device)
        metrics = compute_metrics(model_at_t(t_tensor), eval_loaders, device, args.select_max_batches)
        score = selection_score(metrics, args)
        if best is None or score < best["score"]:
            best = {"t": float(t), "score": score, "metrics": metrics}
    return best
