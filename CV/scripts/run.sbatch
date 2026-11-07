#!/usr/bin/env bash
#SBATCH --job-name=pleu-cv
#SBATCH --account=bfmd-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch scripts/run.sbatch cifar100 "42 43 44"
#        sbatch scripts/run.sbatch python unlearn_pleu.py --dataset cifar100 --split-mode class
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -n "${PLEU_ENV_SCRIPT:-}" ]] && source "$PLEU_ENV_SCRIPT"
mkdir -p logs
if [[ "${1:-}" == python ]]; then
  "$@"
else
  bash scripts/run_all.sh "$@"
fi
