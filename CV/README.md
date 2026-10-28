# PLEU — vision experiments

MoE-ViT (M³ViT-style) on **CIFAR-100** and **ImageNet-100**, with class-wise forgetting (10 classes)
and random forgetting (10% of the training set).

The backbone ([pleu_cv/model.py](pleu_cv/model.py)) has 12 transformer blocks; every odd block replaces
its MLP with a top-1 mixture of 4 experts (6 MoE layers). Parameter names are identical to the original
`try_moevit.py`, so earlier checkpoints load unchanged.

| | CIFAR-100 | ImageNet-100 |
|---|---|---|
| input / patch | 32 / 4 | 224 / 16 |
| width / depth / heads | 256 / 12 / 8 | 256 / 12 / 8 |
| training recipe | 250 ep, AdamW 1e-3, CutMix | 200 ep, AdamW 8e-4, MixUp, EMA, AMP |

## Files

| File | Purpose |
|---|---|
| `prepare_splits.py` | class-wise and random forget/retain splits (ImageNet-100 is first split 90/10 per class) |
| `build_relearn_set.py` | relearning set: 100 random ImageNet-1k classes outside ImageNet-100, unrelated to the forget set |
| `train.py` | original model (`origin`); `--retain-only` trains the retrain baseline (RT) |
| `unlearn_baseline.py` | GA, NegGrad+ and their SEUF variants (`--seuf`, `--seuf-experts N`, `--seuf-layerwise`) |
| `unlearn_mcu.py` | MCU: full-model Bézier curve between origin and NegGrad+, MCU tensor mask |
| `unlearn_pleu.py` | PLEU; `--ablation no_lteu \| no_idff \| no_sepo` |
| `relearn.py` | 10 relearning steps on the relearning set; `--trace` records FA(train) after each step |
| `summarize.py` | Table 2 rows as mean ± std over seeds; `--plot` draws Fig. 3 |
| `scripts/run_all.sh`, `scripts/run.sbatch` | the whole pipeline, locally or on Slurm |
| `pleu_cv/` | `data` · `model` · `routing` (relevance, IDFF, routing distance) · `masks` (MCU, SEPO) · `curve` (Bézier) · `unlearn` (shared loop helpers) |

## Data and outputs

```
$PLEU_CV_DATA            (default: CV/data)
├── cifar-100-python/            downloaded automatically
├── imagenet100/<wnid>/*.JPEG
└── data_re/<wnid>/*             build_relearn_set.py

$PLEU_CV_OUTPUT/<dataset>/  (default: CV/outputs)
├── splits/      {forget,retain}_indices_{class,random}_{train,test}.npy, split_metadata.json
├── models/      origin.pth, <method>_<split>_s<seed>.pth
└── results/     <method>_<split>_s<seed>.json, relearn_<method>_<split>_s<seed>.json
```

Existing artifacts can be reused: `prepare_splits.py --dataset imagenet100 --from-json <imagenet100_classwise_split.json>`
for the old ImageNet-100 split and `--origin-path <ckpt>` for an already trained original model.

## Running

```bash
pip install -r requirements.txt
bash scripts/run_all.sh cifar100 "42 43 44"           # everything, both splits, 3 seeds
sbatch scripts/run.sbatch imagenet100 "42 43 44"      # same on Slurm
SPLITS=class SKIP_TRAIN=1 bash scripts/run_all.sh cifar100 42   # reuse splits and origin
```

Step by step (one split, one seed):

```bash
python prepare_splits.py --dataset cifar100
python build_relearn_set.py --imagenet-train-dir /path/to/imagenet/train     # or --hf-dataset imagenet-1k
python train.py --dataset cifar100                                           # origin
python train.py --dataset cifar100 --split-mode class --retain-only          # RT
python unlearn_baseline.py --dataset cifar100 --split-mode class --method neggrad_plus   # also the PLEU/MCU endpoint
python unlearn_baseline.py --dataset cifar100 --split-mode class --method ga --seuf
python unlearn_mcu.py      --dataset cifar100 --split-mode class
python unlearn_pleu.py     --dataset cifar100 --split-mode class
python relearn.py          --dataset cifar100 --split-mode class --method pleu --trace
python summarize.py        --dataset cifar100 --split-mode class --plot fig_relearning.pdf
```

Every script accepts `--seed`, `--tag` (suffix for output names) and architecture overrides
(`--image-size`, `--embed-dim`, …); `--help` lists the method-specific options.

## Implementation notes

**PLEU** ([unlearn_pleu.py](unlearn_pleu.py))

1. *Localization on the original model, once, over the whole forget set (eval transform).*
   Relevance of an expert is its routing weight averaged over tokens and forget samples (Eq. 3.2);
   the most relevant expert of each MoE layer is its target (LTEU). Each sample is assigned to the
   first layer whose top-K experts (K = 1) of its token-averaged weights contain the target (IDFF);
   samples never routed to a target are left out (`--unassigned last` gives them to the last layer).
   SEPO masks use |∂L/∂θ| on D_r and D_f: the top k_r = 0.1 by retain importance are protected,
   the top k_f = 0.5 of the rest by forget importance are trainable.
2. *Sequential unlearning, shallow to deep.* Only the target expert of layer l is on the masked Bézier
   curve towards the NegGrad+ expert. Its control point and the router of layer l minimize
   E_t[L(D_r) − β L(D_match^(l))] + α L_dist with α = 1, MCU's adaptive β and 50% of D_r.
   t^(l) is chosen on a grid with the MCU selection rule and written into the model before layer l+1.

**Checkpoint / curve-point selection** (all methods, as in the original scripts): class-wise maximizes
RA(train) + RA(test) − FA(train) − FA(test); random matches the train FA of RT
(`--target-forget-acc`, default 65.70 for CIFAR-100 and 88.01 for ImageNet-100). RT keeps the epoch with
the best retain test accuracy.

**Relearning** replaces the classifier head with a fresh one for the 100 relearning classes, trains all
parameters for 10 steps (AdamW 3e-4, CutMix), restores the original head and reports FA/RA.

## Tests

`python -m pytest tests -q` runs unit tests (Bézier curve, IDFF, SEPO, routing distance) and a CPU
end-to-end run of every script on a tiny synthetic ImageNet-style dataset.
