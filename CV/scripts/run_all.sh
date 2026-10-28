#!/usr/bin/env bash
# Full CV pipeline for one dataset: splits, origin, RT, baselines, MCU, PLEU, ablations, relearning.
#   bash scripts/run_all.sh cifar100 "42 43 44"
#   SPLITS=class SKIP_TRAIN=1 bash scripts/run_all.sh imagenet100 42
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${1:-cifar100}
SEEDS=${2:-"42 43 44"}
SPLITS=${SPLITS:-"class random"}
COMMON=(--dataset "$DATASET" ${EXTRA_ARGS:-})

if [[ -z "${SKIP_TRAIN:-}" ]]; then
  python prepare_splits.py --dataset "$DATASET"
  python train.py "${COMMON[@]}"
fi

for SPLIT in $SPLITS; do
  for SEED in $SEEDS; do
    ARGS=("${COMMON[@]}" --split-mode "$SPLIT" --seed "$SEED")
    [[ -n "${SKIP_RT:-}" ]] || python train.py "${ARGS[@]}" --retain-only
    python unlearn_baseline.py "${ARGS[@]}" --method ga
    python unlearn_baseline.py "${ARGS[@]}" --method neggrad_plus
    python unlearn_baseline.py "${ARGS[@]}" --method ga --seuf
    python unlearn_baseline.py "${ARGS[@]}" --method neggrad_plus --seuf
    python unlearn_mcu.py "${ARGS[@]}"
    python unlearn_pleu.py "${ARGS[@]}"
    for ABLATION in no_lteu no_idff no_sepo; do
      python unlearn_pleu.py "${ARGS[@]}" --ablation "$ABLATION"
    done
    for METHOD in ga seuf_ga neggrad_plus seuf_neggrad_plus mcu pleu; do
      python relearn.py "${ARGS[@]}" --method "$METHOD" --trace
    done
  done
  python summarize.py --dataset "$DATASET" --split-mode "$SPLIT"
done
