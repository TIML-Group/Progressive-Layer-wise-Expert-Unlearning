"""Optimization helpers shared by the LLM unlearning scripts."""

import math

import torch
from transformers import get_linear_schedule_with_warmup

# Same target set as the original GA / NegGrad+ scripts.
LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}


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


def add_lora_args(parser):
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)


def lora_target_modules(model):
    suffixes = {name.split(".")[-1] for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head")}
    return sorted(LORA_TARGETS & suffixes) or sorted(suffixes)


def apply_lora(model, args):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        bias="none", task_type=TaskType.CAUSAL_LM, target_modules=lora_target_modules(model))
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def build_optimizer(params, args, steps_per_epoch):
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total = max(1, math.ceil(steps_per_epoch * args.num_epochs / args.grad_accum_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * args.warmup_ratio), total)
    return optimizer, scheduler


class Accumulator:
    """Gradient accumulation with clipping; skips non-finite losses."""

    def __init__(self, params, optimizer, scheduler, accum_steps, max_grad_norm):
        self.params, self.optimizer, self.scheduler = params, optimizer, scheduler
        self.accum_steps, self.max_grad_norm = accum_steps, max_grad_norm
        self.micro, self.skipped = 0, 0
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss, last=False):
        if not torch.isfinite(loss):
            self.skipped += 1
            return False
        (loss / self.accum_steps).backward()
        self.micro += 1
        if self.micro % self.accum_steps == 0 or last:
            self.step()
        return True

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
