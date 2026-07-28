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

algorithm_name="newskill"

# Defaults (MAPPO-based — no skills/communication)
use_entity_actor=1
pi_choice="CatTrans"
pi_use_obs=1
pi_use_latent=0
use_latent_skills=0
skill_choice="None"
skill_type="Continuous"
share_tblocks=0
skill_kl_loss=0
comm_use_active_masks=0
n_future_steps=0
kl_gamma=1
num_skills=4
comm_threshold=0
use_similarity=0
sim_metrics="None"
op_aggregate="None"
use_action_predictor=0
skill_to_obs="None"
comm_channel="None"

# Define which methods this script supports:
SUPPORTED_METHODS=(
    newskill_mappo
    newskill_sesil_mappo
)

# If the requested method isn't in that list, exit quietly
if [[ ! " ${SUPPORTED_METHODS[*]} " =~ " ${method} " ]]; then
  exit 0
fi

# Newskill budget defaults
NEWSKILL_PHASE1_BUDGET=3000000
NEWSKILL_PHASE2_BUDGET=2000000
NEWSKILL_KNOWN_TASKS=8
NEWSKILL_UNKNOWN_TASKS=2

#################################################################
########## newskill_mappo: MAPPO baseline #######################
#################################################################
if [[ "$method" == "newskill_mappo" ]]; then
    settings="newskill_mappo_"$key_name
    NEWSKILL_EXTRA="--newskill_variant mappo --newskill_phase1_budget $NEWSKILL_PHASE1_BUDGET --newskill_phase2_budget $NEWSKILL_PHASE2_BUDGET --newskill_known_tasks $NEWSKILL_KNOWN_TASKS --newskill_unknown_tasks $NEWSKILL_UNKNOWN_TASKS"
fi

#################################################################
########## newskill_sesil_mappo: SESiL + outlander ##############
#################################################################
if [[ "$method" == "newskill_sesil_mappo" ]]; then
    settings="newskill_sesil_mappo_"$key_name
    NEWSKILL_EXTRA="--newskill_variant sesil_mappo --newskill_phase1_budget $NEWSKILL_PHASE1_BUDGET --newskill_phase2_budget $NEWSKILL_PHASE2_BUDGET --newskill_known_tasks $NEWSKILL_KNOWN_TASKS --newskill_unknown_tasks $NEWSKILL_UNKNOWN_TASKS --evo_solver_algo mappo"
fi

extra_args="$RESUME $NEWSKILL_EXTRA"
run_job bash merge_scripts.sh --run-mode $RUN_MODE --seed $SEED --extra "$extra_args" $platform $project_name $env_name $use_multi_envs $train_tasks $eval_tasks $algorithm_name $output_dir $model_dir $use_wandb $num_env_steps \
    $settings $use_latent_skills $skill_choice $skill_type $num_skills $share_tblocks $skill_kl_loss $skill_to_obs \
    $comm_channel $comm_use_active_masks $op_aggregate $use_similarity $sim_metrics \
    $pi_choice $pi_use_obs $pi_use_latent $use_action_predictor $n_future_steps $kl_gamma $comm_threshold \
    $use_entity_actor
wait
