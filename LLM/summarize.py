"""Collect FE / UT / time before and after relearning (Table 3 layout, mean ± std over seeds).

    python summarize.py --model deepseek --forget-dataset wmdp
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from pleu_llm.data import DEFAULT_OUTPUT_ROOT
from pleu_llm.modeling import MODEL_IDS

ORDER = ["base", "ga", "seuf_ga", "neggrad_plus", "seuf_neggrad_plus", "mcu", "pleu",
         "pleu_no_lteu", "pleu_no_idff", "pleu_no_sepo"]


def fmt(values, digits=4):
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}" if len(values) > 1 else f"{values[0]:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--forget-dataset", default="wmdp")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    key = Path(args.model).name if Path(args.model).is_dir() else MODEL_IDS.get(args.model, args.model).split("/")[-1]
    root = args.output_root / key / args.forget_dataset
    rows = defaultdict(lambda: defaultdict(list))
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        method = re.sub(r"_s\d+(_.*)?$", "", run.name)
        for filename, prefix in (("results.json", ""), ("eval_results.json", ""), ("relearn_results.json", "re_")):
            path = run / filename
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                for metric in ("fe", "ut", "time_min"):
                    if metric in payload and not (prefix and metric == "time_min"):
                        rows[method][prefix + metric].append(payload[metric])

    print(f"{key} / {args.forget_dataset}")
    print("| Method | FE | UT | Time (min) | FE (relearn) | UT (relearn) |")
    print("|---|---|---|---|---|---|")
    for method in [m for m in ORDER if m in rows] + sorted(set(rows) - set(ORDER)):
        r = rows[method]
        print(f"| {method} | {fmt(r['fe'])} | {fmt(r['ut'])} | {fmt(r['time_min'], 2)} | "
              f"{fmt(r['re_fe'])} | {fmt(r['re_ut'])} |")


if __name__ == "__main__":
    main()
