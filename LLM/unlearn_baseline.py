"""LLM baselines: GA and NegGrad+ on all parameters of the model, and their SEUF variants
that train only the selected expert(s) and router(s).

Full-model training keeps weights, gradients and AdamW states of the whole model
(about 4x its bf16 size, ~125 GB for DeepSeek-V2-Lite), so it is spread over the GPUs of
a node with ``--device-map auto``; ``--gradient-checkpointing`` trims the activations.

    python unlearn_baseline.py --model deepseek --forget-dataset wmdp --method neggrad_plus
    python unlearn_baseline.py --model qwen --forget-dataset rwku --method ga --seuf
    python unlearn_baseline.py --method neggrad_plus --seuf --seuf-experts 2      # Table 1
    python unlearn_baseline.py --method neggrad_plus --seuf --seuf-layerwise      # Table 1
"""

from tqdm import tqdm

from pleu_llm.common import Timer, add_common_args, new_parser, run_dir, save_delta, set_seed, write_results
from pleu_llm.data import TextDataset, cycle, forget_and_retain_texts, lm_loss, make_loader, to_device
from pleu_llm.evaluation import add_eval_args, evaluate_fe_ut
from pleu_llm.modeling import (
    RouterRecorder, expert_param_names, input_device, load_model_and_tokenizer, moe_layers,
    resolve_model_path, router_param_names, routing_distance,
)
from pleu_llm.routing import global_targets, layer_targets, sample_routing_weights
from pleu_llm.training import Accumulator, add_memory_args, add_optim_args, build_optimizer, enable_full_training

DEFAULT_LR = {False: 1e-5, True: 3e-4}  # full model / SEUF experts


def method_name(args):
    if not args.seuf:
        return args.method
    name = f"seuf_{args.method}"
    if args.seuf_layerwise:
        return name + "_layerwise"
    return name if args.seuf_experts == 1 else f"{name}_{args.seuf_experts}experts"


def select_seuf_targets(model, tokenizer, forget_texts, args):
    loader = make_loader(TextDataset(forget_texts, tokenizer, args.cutoff_len), tokenizer, args.batch_size, False)
    weights = sample_routing_weights(model, loader, "forget routing")
    if args.seuf_layerwise:
        targets, _ = layer_targets(weights)
        return {layer: [expert] for layer, expert in targets.items()}
    targets, _ = global_targets(weights, args.seuf_experts)
    return dict(targets)


def main():
    parser = add_common_args(new_parser(__doc__))
    add_eval_args(parser)
    add_optim_args(parser, lr=None)
    add_memory_args(parser)
    parser.add_argument("--method", default="neggrad_plus", choices=["ga", "neggrad_plus"])
    parser.add_argument("--beta", type=float, default=0.5, help="NegGrad+ forget weight")
    parser.add_argument("--seuf", action="store_true")
    parser.add_argument("--seuf-experts", type=int, default=1)
    parser.add_argument("--seuf-layerwise", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0, help="SEUF routing distance weight")
    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LR[args.seuf]
    set_seed(args.seed)

    name = method_name(args)
    out_dir = run_dir(name, args)
    base_path = resolve_model_path(args.model, args.model_root)
    forget_texts, retain_texts = forget_and_retain_texts(args)
    model, tokenizer = load_model_and_tokenizer(base_path, args.device_map)
    timer = Timer()

    targets = None
    if args.seuf:
        targets = select_seuf_targets(model, tokenizer, forget_texts, args)
        print("SEUF target experts:", targets)
        layers = {layer.layer_idx: layer for layer in moe_layers(model)}
        trainable_names = set()
        for layer_idx, experts in targets.items():
            trainable_names.update(router_param_names(model, layers[layer_idx]))
            for expert in experts:
                trainable_names.update(expert_param_names(model, layers[layer_idx], expert))
        for param_name, param in model.named_parameters():
            param.requires_grad_(param_name in trainable_names)
    else:
        enable_full_training(model, args)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    forget_loader = make_loader(TextDataset(forget_texts, tokenizer, args.cutoff_len), tokenizer,
                                args.batch_size, True)
    retain_iter = cycle(make_loader(TextDataset(retain_texts, tokenizer, args.cutoff_len), tokenizer,
                                    args.batch_size, True))
    optimizer, scheduler = build_optimizer(params, args, len(forget_loader))
    acc = Accumulator(params, optimizer, scheduler, args.grad_accum_steps, args.max_grad_norm)
    device = input_device(model)

    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(tqdm(forget_loader, desc=f"{name} epoch {epoch + 1}"), start=1):
            batch = to_device(batch, device)
            with RouterRecorder(model, list(targets) if targets else []) as recorder:
                loss_forget = lm_loss(model, batch)
                loss_dist = (sum(routing_distance(recorder.probs[l], batch["attention_mask"], e)
                                 for l, e in targets.items()) / len(targets)) if targets else 0.0
            if args.method == "neggrad_plus":
                loss = lm_loss(model, to_device(next(retain_iter), device)) - args.beta * loss_forget
            else:
                loss = -loss_forget
            acc.backward(loss + args.alpha * loss_dist, last=step == len(forget_loader))
    train_minutes = timer.minutes()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_disable()
    changed = {n: p for n, p in model.named_parameters() if p.requires_grad}
    save_delta(out_dir, base_path, changed, method=name, targets=targets)
    results = {"time_min": train_minutes, "targets": targets, "skipped_steps": acc.skipped,
               "num_trainable": sum(p.numel() for p in params), "checkpoint": str(out_dir)}
    if not args.skip_eval:
        results.update(evaluate_fe_ut(model, tokenizer, args))
    write_results(out_dir / "results.json", name, args, results)


if __name__ == "__main__":
    main()
