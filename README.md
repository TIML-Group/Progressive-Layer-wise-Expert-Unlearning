# Progressive Layer-wise Expert Unlearning (PLEU)

Official code for **Beyond Superficial Unlearning: Progressive Layer-wise Expert Erasure in MoE Models**
(Haoxin Guo, Yingdan Shi, Ren Wang).

Unlearning a single, globally selected expert of a Mixture-of-Experts model looks effective right
after unlearning, but the removed capability comes back after a few steps of fine-tuning on unrelated
data. PLEU localizes the target knowledge progressively — from layers to experts, samples and
parameters — and erases it with mode-connectivity optimization:

| Component | What it does |
|---|---|
| **LTEU** — Layer-wise Target Expert Unlearning | picks the expert with the highest expected routing weight on the forget set in *every* MoE layer |
| **IDFF** — Inter-layer Data Flow Filtering | assigns each forget sample to the first layer that routes it to that layer's target expert, so no sample drives several layers |
| **MCUE** — Mode Connectivity Unlearning Engine | moves each target expert along a quadratic Bézier curve between the original and a pre-unlearned (NegGrad+) model, layer by layer |
| **SEPO** — Selective Expert Parameter Optimization | inside each target expert, protects retain-critical entries (top k_r) and only updates forget-relevant ones (top k_f) |

## Repository

```
PLEU/
├── CV/     MoE-ViT on CIFAR-100 and ImageNet-100 (class-wise and random forgetting)
└── LLM/    DeepSeek-V2-Lite and Qwen1.5-MoE-A2.7B on WMDP and RWKU
```

Each part is self-contained (own package, scripts, tests and README) and implements PLEU together
with the same baselines: **GA**, **NegGrad+**, **GA (SEUF)**, **NegGrad+ (SEUF)**, **MCU**, retraining
(**RT**, CV only), the PLEU ablations, and the unified **relearning** protocol.

* [CV/README.md](CV/README.md) — data layout, commands, implementation notes for vision.
* [LLM/README.md](LLM/README.md) — models, metrics, GPU memory, implementation notes for language models.

## Quick start

```bash
# vision: every method on CIFAR-100, 3 seeds, both splits
cd CV && pip install -r requirements.txt
bash scripts/run_all.sh cifar100 "42 43 44"

# language: every method on DeepSeek-V2-Lite / WMDP, 3 seeds
cd LLM && pip install -r requirements.txt
python download_model.py --model deepseek
bash scripts/run_all.sh deepseek wmdp "123 124 125"
```

On a Slurm cluster, `sbatch scripts/run.sbatch ...` in either folder runs the same pipeline
(set `PLEU_ENV_SCRIPT` to a script that activates your environment).

## Reproducing the paper

| Paper | Folder | Command |
|---|---|---|
| Table 1 — more experts in SEUF | `LLM/` | `python unlearn_baseline.py --method neggrad_plus --seuf [--seuf-experts 2 \| 3 \| --seuf-layerwise]`, then `relearn.py --method <run>` |
| Table 2 — vision results | `CV/` | `bash scripts/run_all.sh {cifar100,imagenet100} "42 43 44"`, then `python summarize.py --dataset ... --split-mode {class,random}` |
| Fig. 3 — relearning dynamics | `CV/` | `python relearn.py --method <m> --trace` for each method, then `python summarize.py ... --plot fig.pdf` |
| Table 3 — LLM results | `LLM/` | `bash scripts/run_all.sh {deepseek,qwen} {wmdp,rwku} "123 124 125"`, then `python summarize.py --model ... --forget-dataset ...` |
| Table 4 — ablations | `LLM/` | `ABLATIONS=1 bash scripts/run_all.sh deepseek wmdp 123` (`unlearn_pleu.py --ablation no_lteu \| no_idff \| no_sepo`) |

PLEU and MCU start from the NegGrad+ model of the same split and seed, so run NegGrad+ first
(`run_all.sh` does this automatically).

## Tests

```bash
cd CV  && python -m pytest tests -q   # unit tests + end-to-end run on synthetic images (CPU)
cd LLM && python -m pytest tests -q   # end-to-end runs on tiny random Qwen2-MoE (and DeepSeek-V2) models
```
