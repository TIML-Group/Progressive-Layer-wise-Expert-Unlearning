"""Aggregate result files into the Table 2 layout (mean ± std over seeds), and
optionally plot the relearning FA traces (Fig. 3).

    python summarize.py --dataset cifar100 --split-mode class
    python summarize.py --dataset cifar100 --split-mode class --plot fig_relearning.pdf
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pleu_cv.data import DEFAULT_OUTPUT_ROOT

METHODS = ["origin", "rt", "ga", "seuf_ga", "neggrad_plus", "seuf_neggrad_plus", "mcu", "pleu",
           "pleu_no_lteu", "pleu_no_idff", "pleu_no_sepo"]
COLUMNS = [("forget_train_acc", "FA train"), ("retain_train_acc", "RA train"),
           ("forget_test_acc", "FA test"), ("retain_test_acc", "RA test")]
LABELS = {"origin": "Origin", "rt": "RT", "ga": "GA", "seuf_ga": "GA (SEUF)", "neggrad_plus": "NegGrad+",
          "seuf_neggrad_plus": "NegGrad+ (SEUF)", "mcu": "MCU", "pleu": "PLEU", "pleu_no_lteu": "PLEU w/o LTEU",
          "pleu_no_idff": "PLEU w/o IDFF", "pleu_no_sepo": "PLEU w/o SEPO"}


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}" if len(values) > 1 else f"{values[0]:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--split-mode", default="class")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plot", default=None, help="Write the relearning trace figure to this path")
    args = parser.parse_args()

    results_dir = args.output_root / args.dataset / "results"
    unlearn, relearn = defaultdict(list), defaultdict(list)
    for path in sorted(results_dir.glob(f"*_{args.split_mode}_s*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = payload["method"]
        if method.startswith("relearn_"):
            relearn[method[len("relearn_"):]].append(payload)
        else:
            unlearn[method].append(payload)

    header = ["Method"] + [c for _, c in COLUMNS] + ["Time (min)"] + [f"Re-{c}" for _, c in COLUMNS]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    order = [m for m in METHODS if m in unlearn or m in relearn]
    order += sorted((set(unlearn) | set(relearn)) - set(order))
    for method in order:
        runs = unlearn.get(method, [])
        row = [LABELS.get(method, method)]
        row += [mean_std([r.get("metrics", {}).get(key) for r in runs]) for key, _ in COLUMNS]
        row.append(mean_std([r.get("time_min") for r in runs]) if method not in ("origin",) else "--")
        row += [mean_std([r["after"][key] for r in relearn.get(method, [])]) for key, _ in COLUMNS]
        print("| " + " | ".join(row) + " |")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4.5))
        for method in order:
            traces = [r["fa_train_trace"] for r in relearn.get(method, []) if r.get("fa_train_trace")]
            if not traces:
                continue
            curve = np.mean(np.asarray(traces), axis=0)
            plt.plot(range(len(curve)), curve, marker="o", label=LABELS.get(method, method),
                     linewidth=2.5 if method == "pleu" else 1.5)
        plt.xlabel("Relearning step")
        plt.ylabel("Forget accuracy on train set (%)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"Saved figure to {args.plot}")


if __name__ == "__main__":
    main()
