#!/usr/bin/env bash
# Full LLM pipeline for one model and forget set: original eval, baselines, MCU, PLEU,
# ablations (DeepSeek/WMDP in the paper), relearning of every method.
#   bash scripts/run_all.sh deepseek wmdp "123 124 125"
#   ABLATIONS=1 bash scripts/run_all.sh deepseek wmdp 123
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-deepseek}
FORGET=${2:-wmdp}
SEEDS=${3:-123}
COMMON=(--model "$MODEL" --forget-dataset "$FORGET" ${EXTRA_ARGS:-})

python evaluate.py "${COMMON[@]}"
python relearn.py "${COMMON[@]}" --method base
for SEED in $SEEDS; do
  ARGS=("${COMMON[@]}" --seed "$SEED")
  python unlearn_baseline.py "${ARGS[@]}" --method ga
  python unlearn_baseline.py "${ARGS[@]}" --method neggrad_plus
  python unlearn_baseline.py "${ARGS[@]}" --method ga --seuf
  python unlearn_baseline.py "${ARGS[@]}" --method neggrad_plus --seuf
  python unlearn_mcu.py "${ARGS[@]}"
  python unlearn_pleu.py "${ARGS[@]}"
  if [[ -n "${ABLATIONS:-}" ]]; then
    for ABLATION in no_lteu no_idff no_sepo; do
      python unlearn_pleu.py "${ARGS[@]}" --ablation "$ABLATION"
    done
    python unlearn_baseline.py "${ARGS[@]}" --method neggrad_plus --seuf --seuf-experts 2
    python unlearn_baseline.py "${ARGS[@]}" --method neggrad_plus --seuf --seuf-experts 3
    python unlearn_baseline.py "${ARGS[@]}" --method neggrad_plus --seuf --seuf-layerwise
  fi
  for METHOD in ga seuf_ga neggrad_plus seuf_neggrad_plus mcu pleu; do
    python relearn.py "${ARGS[@]}" --method "$METHOD"
  done
done
python summarize.py --model "$MODEL" --forget-dataset "$FORGET"
