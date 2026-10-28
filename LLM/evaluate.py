"""Evaluate FE / UT of the original model or of an unlearned checkpoint.

    python evaluate.py --model deepseek --forget-dataset wmdp            # original model
    python evaluate.py --checkpoint outputs/DeepSeek-V2-Lite-Chat/rwku/pleu_s123 --forget-dataset rwku
"""

from pathlib import Path

from pleu_llm.common import add_common_args, load_any, new_parser, run_dir, set_seed, write_results
from pleu_llm.evaluation import add_eval_args, evaluate_fe_ut


def main():
    parser = add_common_args(new_parser(__doc__))
    add_eval_args(parser)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    target = args.checkpoint or args.model
    model, tokenizer = load_any(target, args)
    results = evaluate_fe_ut(model, tokenizer, args)
    out = Path(args.checkpoint) if args.checkpoint else run_dir("base", args)
    write_results(out / "eval_results.json", "eval", args, {"target": str(target), **results})


if __name__ == "__main__":
    main()
