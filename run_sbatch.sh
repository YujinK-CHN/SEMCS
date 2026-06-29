#!/usr/bin/env bash
set -euo pipefail

# ── 1) Parse --run-mode ──────────────────────────────────────
RUN_MODE="parallel"
SEED="10"
while [[ $# -gt 0 ]]; do
  case "$1" in
      --run-mode)
        RUN_MODE="$2"; shift 2;;
      --seed)
        SEED="$2"; shift 2;;
      *)
      break;;
  esac
done
echo "Run mode: $RUN_MODE, Seed: $SEED"

# ── 2) If sequential, don’t exit on command failures ────────
if [[ "$RUN_MODE" == "sequential" ]]; then
  set +e   # disable “exit on error”
else
  set -e   # ensure parallel still fails fast
fi

# ── 3) run_job helper ───────────────────────────────────────
run_job(){
  if [[ "$RUN_MODE" == "parallel" ]]; then
    "$@" &
  else
    "$@"
  fi
}

#################### Common Paramters ###################
wandb_flag=$1
env_name=$2
model_dir="None"
num_env_steps=$3
platform=$4
project_name="Baselines"
key_name="compare"

# Default to 0 (False)
use_wandb=0
if [ "$wandb_flag" == "use_wandb" ]; then
  use_wandb=1
fi

# ─── Define a unique ID for this experiment run ─────────────
if [[ "$env_name" == "StarCraft" ]]; then
  PORT_ID="$$"
  PORT_FILE="results/used_ports.txt"
  touch "$PORT_FILE"
  export PORT_FILE
  echo "Created port file: $PORT_FILE"
fi

#################   StarCraft   #################
if [[ "$env_name" == "StarCraft" ]]; then
  TASKS=(
    "3m|5m_vs_6m|8m_vs_9m|10m_vs_11m"
    # "2s3z|3s5z|3s5z_vs_3s6z"
  )
elif [[ "$env_name" == "AliceBob" ]]; then
  TASKS=(
    # "233-0|233-1|233-2|233-3"
    "344-0|344-1|344-2|344-3"
  )
elif [[ "$env_name" == "Football" ]]; then
  TASKS=(
    "academy_3_vs_1_with_keeper|academy_pass_and_shoot_with_keeper"
  )
else
  echo "Unknown env_name: $env_name" >&2
  exit 1
fi

SCRIPTS=(
  #s_w_comm_w_pred.sh    # MCS
  s_dt2gs.sh            # DT2GS
  #s_sesil.sh            # SESiL
  #s_sft.sh              # SFT (sequential fine-tuning baseline)
)

METHODS=(
  #mcs_skill_GRU_Pre_Merge_CommMask    # MCS
  dt2gs_subtask_VAE_Merge             # DT2GS
  #sesil_enc_VAE_Merge                 # SESiL
  #sft_sequential                      # SFT
)

# ── Run multi‐task scripts ───────────────────────────────
for TASK in "${TASKS[@]}"; do
  for script in "${SCRIPTS[@]}"; do
    for method in "${METHODS[@]}"; do
      if [[ "$RUN_MODE" == "sequential" ]]; then
        echo ">> Run: $script $TASK $method"
        bash "$script" --run-mode "$RUN_MODE" --seed "$SEED" \
          "$env_name" "$TASK" "$TASK" "$model_dir" "$use_wandb" "$num_env_steps" "$platform" "$project_name" "$key_name" "$method"
        rc=$?
        if (( rc != 0 )); then
          echo "⚠️  $script failed for $TASK + $method (exit $rc), continuing..."
        fi
      else
        run_job bash "$script" --run-mode "$RUN_MODE" --seed "$SEED" \
          "$env_name" "$TASK" "$TASK" "$model_dir" "$use_wandb" "$num_env_steps" "$platform" "$project_name" "$key_name" "$method"
      fi
    done
  done
done

# ── Wait for background jobs (only in parallel) ──────────
if [[ "$RUN_MODE" == "parallel" ]]; then
  wait
fi

echo "All experiments dispatched."
