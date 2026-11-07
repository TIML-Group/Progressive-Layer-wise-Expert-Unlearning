"""Optimization helpers shared by the LLM unlearning scripts."""

import math

import torch
from transformers import get_linear_schedule_with_warmup


def add_optim_args(parser, lr, epochs=1):
    parser.add_argument("--learning-rate", type=float, default=lr)
    parser.add_argument("--num-epochs", type=int, default=epochs)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)


def add_memory_args(parser):
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Recompute activations in backward (full-model training on long inputs)")


def enable_full_training(model, args):
    """Train every parameter of the model (no adapters)."""
    for param in model.parameters():
        param.requires_grad_(True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()


def build_optimizer(params, args, steps_per_epoch):
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    # the Accumulator also steps on the last micro-batch of every epoch
    total = max(1, args.num_epochs * math.ceil(steps_per_epoch / args.grad_accum_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * args.warmup_ratio), total)
    return optimizer, scheduler


def add_beta_args(parser, beta=0.15):
    """MCU defaults: adaptive beta in {0, 0.01, 0.5}; 0.15 when fixed."""
    parser.add_argument("--beta-mode", default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--beta", type=float, default=beta, help="Fixed unlearning penalty")
    parser.add_argument("--beta-low", type=float, default=0.01)
    parser.add_argument("--beta-high", type=float, default=0.5)
    parser.add_argument("--retain-base", type=float, default=None,
                        help="Retain next-token accuracy reference (default: endpoint model on the retain texts)")
    parser.add_argument("--forget-base", type=float, default=None,
                        help="Forget next-token accuracy reference (default: endpoint model on the forget texts)")
    parser.add_argument("--ref-max-batches", type=int, default=None,
                        help="Batches per set used to measure the endpoint references (default: all)")


def relative_gap(current, base):
    if abs(base) < 1e-8:
        return 0.0 if current <= base else float("inf")
    return (current - base) / base


def adaptive_beta(retain_acc, forget_acc, retain_base, forget_base, beta_low, beta_high):
    """Adaptive unlearning penalty of MCU (``training_step_dynamic`` of the official code):
    0 once the batch forget accuracy reaches its reference, ``beta_low`` while the forget
    accuracy is relatively closer to its reference than the retain accuracy is to its own,
    ``beta_high`` otherwise."""
    if forget_acc <= forget_base:
        return 0.0
    if relative_gap(forget_acc, forget_base) < relative_gap(retain_acc, retain_base):
        return beta_low
    return beta_high


def current_beta(args, retain_acc, forget_acc, refs):
    if args.beta_mode == "fixed":
        return args.beta
    return adaptive_beta(retain_acc, forget_acc, refs[0], refs[1], args.beta_low, args.beta_high)


class Accumulator:
    """Gradient accumulation with clipping; skips non-finite losses."""

    def __init__(self, params, optimizer, scheduler, accum_steps, max_grad_norm):
        self.params, self.optimizer, self.scheduler = params, optimizer, scheduler
        self.accum_steps, self.max_grad_norm = accum_steps, max_grad_norm
        self.micro, self.skipped = 0, 0
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss, last=False):
        finite = bool(torch.isfinite(loss))
        if finite:
            (loss / self.accum_steps).backward()
            self.micro += 1
        else:
            self.skipped += 1
        if self.micro and (self.micro == self.accum_steps or last):
            self.step()
        return finite

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.micro = 0  # micro-batches accumulated since the last step
