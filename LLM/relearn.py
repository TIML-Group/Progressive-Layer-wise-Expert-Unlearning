"""Unified relearning protocol for LLMs: a few full-parameter fine-tuning steps (10 by
default) on benign text that is disjoint from the forget set, then FE/UT evaluation.
Like full-model NegGrad+, this needs the GPUs of a node (``--device-map auto``);
``--lora`` fine-tunes adapters instead.

    python relearn.py --model deepseek --forget-dataset wmdp --method pleu
    python relearn.py --checkpoint outputs/DeepSeek-V2-Lite-Chat/wmdp/mcu_s123 --forget-dataset wmdp
    python relearn.py --model deepseek --forget-dataset wmdp --method base    # original model
"""

from pathlib import Path

from tqdm import tqdm

from pleu_llm.common import Timer, add_common_args, load_any, new_parser, run_dir, set_seed, write_results
from pleu_llm.data import TextDataset, cycle, lm_loss, load_text_jsonl, make_loader, relearn_data_path, to_device
from pleu_llm.evaluation import add_eval_args, evaluate_fe_ut
from pleu_llm.modeling import input_device
from pleu_llm.training import (
    Accumulator, add_lora_args, add_memory_args, apply_lora, build_optimizer, enable_full_training,
)


def main():
    parser = add_common_args(new_parser(__doc__))
    add_eval_args(parser)
    add_lora_args(parser)
    add_memory_args(parser)
    parser.add_argument("--lora", action="store_true", help="Relearn with LoRA adapters instead of all parameters")
    parser.add_argument("--method", default=None, help="Run name under the output root (``base`` = original model)")
    parser.add_argument("--checkpoint", default=None, help="Unlearned checkpoint directory (overrides --method)")
    parser.add_argument("--relearn-data", type=Path, default=None,
                        help="JSONL with a 'text' field (default: <data-root>/relearn/<forget>_benign.jsonl)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None, help="Default 1e-5 (full) / 1e-4 (LoRA)")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.set_defaults(grad_accum_steps=1, batch_size=4)
    args = parser.parse_args()
    set_seed(args.seed)
    if args.checkpoint is None:
        if args.method is None:
            parser.error("give --method or --checkpoint")
        args.checkpoint = args.model if args.method == "base" else str(run_dir(args.method, args))
    label = args.method or Path(args.checkpoint).name
    data_path = args.relearn_data or relearn_data_path(args.data_root, args.forget_dataset)

    if args.learning_rate is None:
        args.learning_rate = 1e-4 if args.lora else 1e-5
    model, tokenizer = load_any(args.checkpoint, args)
    if args.lora:
        model = apply_lora(model, args)
    else:
        enable_full_training(model, args)
    texts = load_text_jsonl(data_path)
    print(f"relearning on {len(texts)} texts from {data_path}")
    loader = make_loader(TextDataset(texts, tokenizer, args.cutoff_len), tokenizer, args.batch_size, True)
    data_iter = cycle(loader)
    params = [p for p in model.parameters() if p.requires_grad]
    args.num_epochs = 1
    optimizer, scheduler = build_optimizer(params, args, args.steps * args.grad_accum_steps)
    acc = Accumulator(params, optimizer, scheduler, args.grad_accum_steps, args.max_grad_norm)

    timer, device = Timer(), input_device(model)
    model.train()
    for step in tqdm(range(args.steps * args.grad_accum_steps), desc="relearning"):
        acc.backward(lm_loss(model, to_device(next(data_iter), device)))
    minutes = timer.minutes()

    if args.lora:
        model = model.merge_and_unload()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_disable()
    results = {"checkpoint": str(args.checkpoint), "relearn_data": str(data_path), "steps": args.steps,
               "time_min": minutes}
    if not args.skip_eval:
        results.update(evaluate_fe_ut(model, tokenizer, args))
    out = Path(args.checkpoint) if Path(args.checkpoint).is_dir() and args.method != "base" else run_dir("base", args)
    write_results(out / "relearn_results.json", f"relearn_{label}", args, results)


if __name__ == "__main__":
    main()
