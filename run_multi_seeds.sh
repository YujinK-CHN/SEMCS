#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_multi_seeds.sh --seeds 1,4,8,10,40,50 [--gpus 0,1,2] [run_sbatch.sh args...]
#
# Example:
#   bash run_multi_seeds.sh --seeds 1,4,8,10,40,50 --gpus 0,1,2 --run-mode sequential no_wandb StarCraft 10000000 test
#
# Each seed launches in the background as a separate run_sbatch.sh process.
# GPU assignment is automatic (least-loaded GPU via lock files in main.py).

SEEDS=""
GPUS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      SEEDS="$2"; shift 2;;
    --gpus)
      GPUS="$2"; shift 2;;
    *)
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "$SEEDS" ]]; then
  echo "Usage: bash run_multi_seeds.sh --seeds 1,4,8,10,40,50 [--gpus 0,1,2] [run_sbatch.sh args...]"
  exit 1
fi

if [[ -n "$GPUS" ]]; then
  export SEMCS_GPUS="$GPUS"
fi

IFS=',' read -ra SEED_LIST <<< "$SEEDS"

echo "Launching ${#SEED_LIST[@]} seeds: ${SEED_LIST[*]}"
echo "Args: ${EXTRA_ARGS[*]}"

PIDS=()
for seed in "${SEED_LIST[@]}"; do
  echo ">> Starting seed=$seed"
  bash run_sbatch.sh --seed "$seed" "${EXTRA_ARGS[@]}" &
  PIDS+=($!)
  sleep 3
done

echo "All ${#SEED_LIST[@]} seeds launched. Waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  seed=${SEED_LIST[$i]}
  if wait "$pid"; then
    echo "Seed $seed finished OK."
  else
    echo "Seed $seed FAILED (exit $?)."
    FAILED=$((FAILED + 1))
  fi
done

if (( FAILED > 0 )); then
  echo "$FAILED seed(s) failed."
  exit 1
fi
echo "All seeds done."
