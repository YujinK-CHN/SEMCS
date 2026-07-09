#!/usr/bin/env bash
set -euo pipefail

# ── 1) Parse --run-mode ──────────────────────────────────────
RUN_MODE="parallel"
SEED="10"
RESUME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-mode)
      RUN_MODE="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    --resume)
      RESUME="--resume"; shift;;
    *)
      break;;
  esac
done

# ── 2) run_job helper ───────────────────────────────────────
run_job(){
  if [[ "$RUN_MODE" == "parallel" ]]; then
    "$@" &
  else
    "$@"
  fi
}


#################################################################
####################### Common Parameters #######################
#################################################################
env_name=$1
use_multi_envs=1
train_tasks=$2
eval_tasks=$3
output_dir=results
model_dir=$4
use_wandb=$5
num_env_steps=$6
platform=$7
project_name=$8
key_name=$9
method=${10}

# pi choice
pi_choice="CatTrans"
pi_use_obs=1
pi_use_latent=1
# skill parameters
use_latent_skills=1
skill_choice="UseTrans"
skill_type="Discrete"   # "Discrete", "Continuous"
share_tblocks=0
skill_kl_loss=0
# for communication
comm_use_active_masks=1  # whether to use active masks during communciation
# for using skill prediction
n_future_steps=${11:-1}
kl_gamma=${12:-1}
num_skills=${13:-10}
comm_threshold=${14:-0}
# for sim
use_similarity=0
sim_metrics="None"  # "Cosin", "Wasserstein", "Covariance", "None"

# Define which methods this script actually supports:
SUPPORTED_METHODS=(
    mcs_skill_GRU_Pre_Merge_CommMask     # pi(a|o, z)      s_w_comm_w_pred_wo_sim.sh
)

# If the requested method isn’t in that list, just exit quietly
if [[ ! " ${SUPPORTED_METHODS[*]} " =~ " ${method} " ]]; then
  exit 0
fi

#################################################################
####### \pi(a|o, z) policy with latent with obs with comm #######
#################################################################
if [[ "$method" == "mcs_skill_GRU_Pre_Merge_CommMask" ]]; then
    algorithm_name="mcs"
    settings="skill_GRU_Pre_MC_"$key_name
    op_aggregate="GRU"
    use_action_predictor=1
    skill_to_obs="merge"
    comm_channel="CommMask"
fi

run_job bash merge_scripts.sh --run-mode $RUN_MODE --seed $SEED --extra "$RESUME" $platform $project_name $env_name $use_multi_envs $train_tasks $eval_tasks $algorithm_name $output_dir $model_dir $use_wandb $num_env_steps \
    $settings $use_latent_skills $skill_choice $skill_type $num_skills $share_tblocks $skill_kl_loss $skill_to_obs \
    $comm_channel $comm_use_active_masks $op_aggregate $use_similarity $sim_metrics \
    $pi_choice $pi_use_obs $pi_use_latent $use_action_predictor $n_future_steps $kl_gamma $comm_threshold
wait
