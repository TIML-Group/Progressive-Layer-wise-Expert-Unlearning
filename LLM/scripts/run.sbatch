#!/usr/bin/env bash
#SBATCH --job-name=pleu-llm
#SBATCH --account=bfmd-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
# Full-model GA / NegGrad+, relearning and MCU need the GPUs of a whole node
# (device_map=auto spreads the model); SEUF and PLEU fit on one GPU.
# Usage: sbatch scripts/run.sbatch deepseek wmdp "123 124 125"
#        sbatch scripts/run.sbatch python unlearn_pleu.py --model qwen --forget-dataset rwku
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -n "${PLEU_ENV_SCRIPT:-}" ]] && source "$PLEU_ENV_SCRIPT"
mkdir -p logs
if [[ "${1:-}" == python ]]; then
  "$@"
else
  bash scripts/run_all.sh "$@"
fi
