#!/usr/bin/env bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=results/Logs/run.%j.out
#SBATCH --error=results/Logs/run.%j.err
#SBATCH --distribution=cyclic
#SBATCH --spread-job

# decide background or not
set -euo pipefail

# ── 1) Parse --run-mode ──────────────────────────────────────
RUN_MODE="parallel"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-mode)
      RUN_MODE="$2"; shift 2;;
    *)
      break;;
  esac
done

platform=$1
project_name=$2

# arguments
python_dir=$3
output_dir=$4
model_dir=$5
use_wandb=$6
experiment_name=$7
env_name=$8
use_multi_envs=$9
train_tasks=${10}
eval_tasks=${11}
algorithm_name=${12}
num_env_steps=${13}

##################### Parameters ##################
use_unified_env=${14}
use_recurrent_policy=${15}
use_centralized_V=${16}
hidden_size=${17}
n_head=${18}
n_block=${19}
n_training_threads=${20}
n_rollout_threads=${21}
n_eval_rollout_threads=${22}
eval_episodes=${23}
num_mini_batch=${24}
ppo_epoch=${25}
use_recurrent_with_agents=${26}
use_actor_loss=${27}
use_latent_skills=${28}
skill_choice=${29}
skill_type=${30}
num_skills=${31}
share_tblocks=${32}
skill_kl_loss=${33}
skill_to_obs=${34}
comm_channel=${35}
comm_use_active_masks=${36}
op_aggregate=${37}
op_entity=${38}
use_similarity=${39}
sim_metrics=${40}
pi_choice=${41}
pi_use_obs=${42}
pi_use_latent=${43}
use_action_predictor=${44}
n_future_steps=${45}
kl_gamma=${46}
comm_threshold=${47}
##################### Parameters ##################
random_seed=(${@:48:55})


if [ "$env_name" == "StarCraft" ]; then
  SLURM_NTASKS=6
elif [ "$env_name" == "Football" ]; then
  SLURM_NTASKS=4
else
  SLURM_NTASKS=6
fi


# ── 2) run_job helper ───────────────────────────────────────
run_job(){
  if [[ "$RUN_MODE" == "parallel" ]] || [[ "$env_name" == "AliceBob" ]]; then
    "$@" &
  elif [[ "$RUN_MODE" == "parallel" ]] || [[ "$env_name" == "Football" ]]; then
    "$@" &
  elif [[ "$platform" != "train_main" && "$env_name" == "StarCraft" ]]; then
    "$@" &
  else
    "$@"
  fi
}


mkdir -p $output_dir
export PYMARL_RESULT_DIR=$output_dir
export WANDB_API_KEY=3b3f69eae02f0d4f0e57bba34d533a8fa2b9b373
# source ~/.bashrc

for i in `seq 1 $SLURM_NTASKS`; do
  # echo 'exp and seed: '${experiment_name}'_s'${random_seed[$(($i-1))]}
  run_job $python_dir/python -u main.py \
      --model_dir $model_dir --use_wandb $use_wandb --project_name $project_name --experiment_name $experiment_name --env_name $env_name --use_multi_envs $use_multi_envs --train_tasks $train_tasks --eval_tasks $eval_tasks --algorithm_name $algorithm_name --num_env_steps $num_env_steps \
      --use_unified_env $use_unified_env --use_recurrent_policy $use_recurrent_policy --use_centralized_V $use_centralized_V \
      --hidden_size $hidden_size --n_head $n_head --n_block $n_block \
      --n_training_threads $n_training_threads --n_rollout_threads $n_rollout_threads --n_eval_rollout_threads $n_eval_rollout_threads --eval_episodes $eval_episodes --num_mini_batch $num_mini_batch --ppo_epoch $ppo_epoch --use_recurrent_with_agents $use_recurrent_with_agents --use_actor_loss $use_actor_loss \
      --use_latent_skills $use_latent_skills --skill_choice $skill_choice --skill_type $skill_type --num_skills $num_skills --share_tblocks $share_tblocks --skill_kl_loss $skill_kl_loss --skill_to_obs $skill_to_obs \
      --comm_channel $comm_channel --comm_use_active_masks $comm_use_active_masks --op_aggregate $op_aggregate --op_entity $op_entity \
      --use_similarity $use_similarity --sim_metrics $sim_metrics \
      --pi_choice $pi_choice --pi_use_obs $pi_use_obs --pi_use_latent $pi_use_latent \
      --use_action_predictor $use_action_predictor --n_future_steps $n_future_steps --kl_gamma $kl_gamma --comm_threshold $comm_threshold \
      --seed ${random_seed[$(($i-1))]}
  sleep 10
done

wait
