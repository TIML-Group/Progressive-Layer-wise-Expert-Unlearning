# PLEU: Progressive Layer-wise Expert Unlearning

Code for *Beyond Superficial Unlearning: Progressive Layer-wise Expert Erasure in MoE Models*.

PLEU unlearns one target expert in every MoE layer (LTEU), assigns each forget sample to a single
layer (IDFF), moves the expert along a Bézier curve towards a NegGrad+ model (MCUE), and only
updates selected parameters inside the expert (SEPO).

```
CV/    MoE-ViT on CIFAR-100 / ImageNet-100
LLM/   DeepSeek-V2-Lite / Qwen1.5-MoE-A2.7B on WMDP / RWKU
```

Both folders contain PLEU and the baselines GA, NegGrad+, GA (SEUF), NegGrad+ (SEUF) and MCU,
plus the relearning attack. Install with `pip install -r requirements.txt` inside each folder.

## CV

```bash
cd CV
python prepare_splits.py --dataset cifar100          # class-wise and random 10% splits
python train.py --dataset cifar100                   # original model
python unlearn_baseline.py --split-mode class --method neggrad_plus   # also the PLEU endpoint
python unlearn_pleu.py --split-mode class
python relearn.py --split-mode class --method pleu --trace
python summarize.py --split-mode class
```

Other methods: `train.py --retain-only` (RT), `unlearn_baseline.py --method ga [--seuf]`,
`unlearn_mcu.py`, `unlearn_pleu.py --ablation no_lteu|no_idff|no_sepo`.
`bash scripts/run_all.sh cifar100 "42 43 44"` runs everything.
Data go to `CV/data` (`imagenet100/`, `data_re/` for relearning), results to `CV/outputs`.

## LLM

```bash
cd LLM
python download_model.py --model deepseek            # or qwen
python unlearn_baseline.py --model deepseek --forget-dataset wmdp --method neggrad_plus
python unlearn_pleu.py     --model deepseek --forget-dataset wmdp
python relearn.py          --model deepseek --forget-dataset wmdp --method pleu
python summarize.py        --model deepseek --forget-dataset wmdp
```

Other methods: `unlearn_baseline.py --method ga [--seuf]`, `unlearn_mcu.py`,
`unlearn_pleu.py --ablation ...`. `bash scripts/run_all.sh deepseek wmdp "123 124 125"` runs everything.
RWKU data go to `LLM/data/RWKU/{wiki_forget_set,forget_eval}`, relearning text to
`LLM/data/relearn/<dataset>_benign.jsonl` (see `generate_relearn_data.py`).

GPUs (bf16 model ≈ 30 GB):

| Step | GPUs |
|---|---|
| PLEU, SEUF, evaluation | 1 |
| GA / NegGrad+ / relearning (all parameters) | 2–4 |
| MCU | 4 |

PLEU only needs the NegGrad+ checkpoint as its endpoint; it reads the target experts from it and
trains one expert at a time, so it runs on a single GPU.

## Tests

`python -m pytest tests -q` in either folder runs the whole pipeline on tiny models on CPU.
