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

# ── 2) run_job helper ───────────────────────────────────────
run_job(){
  if [[ "$RUN_MODE" == "parallel" ]]; then
    "$@" &
  else
    "$@"
  fi
}

#################### Start Script ##################
platform=$1
project_name=$2
env_name=$3
use_multi_envs=$4
train_tasks=$5   # multi-task problem set train_tasks to a form to "3m|8m|8m_vs_9m" otherwise like 3m
eval_tasks=$6
algorithm_name=$7
output_dir=$8
model_dir=$9
use_wandb=${10}
num_env_steps=${11}
settings=${12}

if [ "$platform" == "train_main" ]; then
    python_dir=~/anaconda3/envs/transfer/bin
    n_training_threads=18
else
    python_dir=.
    n_training_threads=10
fi

output_dir='./'$output_dir

mkdir -p $output_dir
export PYMARL_RESULT_DIR=$output_dir


###### decide the number of tasks
bars_only=${train_tasks//[^|]/}
count=${#bars_only}
if (( count == 2 )); then
  n_rollout_threads=15
elif (( count == 4 )); then
  n_rollout_threads=15
else
  n_rollout_threads=16
fi

if [ "$env_name" == "StarCraft" ]; then
  num_mini_batch=8
  n_eval_rollout_threads=1
  eval_episodes=32
  ppo_epoch=8
elif [ "$env_name" == "Football" ]; then
  num_mini_batch=2
  n_eval_rollout_threads=32
  eval_episodes=32
  ppo_epoch=10
elif [ "$env_name" == "AliceBob" ]; then
  num_mini_batch=10
  n_eval_rollout_threads=40
  eval_episodes=40
  ppo_epoch=8
else
  num_mini_batch=1
  n_eval_rollout_threads=32
  eval_episodes=32
  ppo_epoch=8
fi

##################### Parameters ##################
use_unified_env=1
use_recurrent_policy=0
use_centralized_V=1

# for transformers
hidden_size=64
n_head=3
n_block=1

# optimization parameters
use_recurrent_with_agents=0  # sample agents as a group or not
use_actor_loss=1

# set skills
use_latent_skills=${13}
skill_choice=${14}
skill_type=${15}   # "Discrete", "Continuous"
num_skills=${16}
share_tblocks=${17} # do not share blocks in skills generation
skill_kl_loss=${18}
skill_to_obs=${19}
# for communication
comm_channel=${20}   # "All", "ExcludeSelf", "None"
comm_use_active_masks=${21}  # whether to use active masks during communciation
op_aggregate=${22}    # "Mean", "Sum", "GRU", "TransMean", "TransSum"
# process entities
op_entity="Mean"  # "Mean", "Sum"
# for similarity
use_similarity=${23}
sim_metrics=${24}  # "Cosin", "Wasserstein", "Covariance", "None"

# integrate into policies
pi_choice=${25}
pi_use_obs=${26}
pi_use_latent=${27}

# for skill predictor
use_action_predictor=${28}
n_future_steps=${29}
kl_gamma=${30}
comm_threshold=${31}

# for online training
random_seed=(1 10 20 30 40 50)

##################### Parameters ##################

if [ "$algorithm_name" == "mcs" ]; then
    job_id="tE"
    use_recurrent_policy=0
    experiment_name="${settings}_ski{${skill_to_obs:0:1}-${skill_type:0:1}-${num_skills}}_com{${comm_channel:0:1}-m${comm_use_active_masks}-p${comm_threshold}-${op_aggregate}}-p{r${use_recurrent_policy}-${use_action_predictor}-${n_future_steps}-${kl_gamma}}"
    echo "env is ${env_name}, train tasks are ${train_tasks}, eval tasks are ${eval_tasks}, algorithm is ${algorithm_name}, exp key is ${experiment_name}"
elif [ "$algorithm_name" == "dt2gs" ]; then
    job_id="tE"
    use_recurrent_policy=0
    experiment_name="${settings}_sub{${skill_to_obs:0:1}-${skill_type:0:1}-${num_skills}}"
    echo "env is ${env_name}, train tasks are ${train_tasks}, eval tasks are ${eval_tasks}, algorithm is ${algorithm_name}, exp key is ${experiment_name}"
elif [ "$algorithm_name" == "sesil" ]; then
    job_id="tE"
    use_recurrent_policy=0
    experiment_name="${settings}_enc{${num_skills}}_evo{25}"
    echo "env is ${env_name}, train tasks are ${train_tasks}, eval tasks are ${eval_tasks}, algorithm is ${algorithm_name}, exp key is ${experiment_name}"
fi

##################################################################################
################## run experiments with corresponding parameters ##################
##################################################################################
if [ "$platform" == "train_main" ]; then
    run_job bash per_job.sh --run-mode $RUN_MODE $platform $project_name $python_dir $output_dir $model_dir $use_wandb $experiment_name $env_name $use_multi_envs $train_tasks $eval_tasks $algorithm_name $num_env_steps \
        $use_unified_env $use_recurrent_policy $use_centralized_V \
        $hidden_size $n_head $n_block \
        $n_training_threads $n_rollout_threads $n_eval_rollout_threads $eval_episodes $num_mini_batch $ppo_epoch $use_recurrent_with_agents $use_actor_loss \
        $use_latent_skills $skill_choice $skill_type $num_skills $share_tblocks $skill_kl_loss $skill_to_obs \
        $comm_channel $comm_use_active_masks $op_aggregate $op_entity \
        $use_similarity $sim_metrics \
        $pi_choice $pi_use_obs $pi_use_latent \
        $use_action_predictor $n_future_steps $kl_gamma $comm_threshold \
        "${random_seed[@]}"
else
    run_job python main.py \
        --model_dir $model_dir --use_wandb $use_wandb --project_name $project_name --experiment_name $experiment_name --env_name $env_name --use_multi_envs $use_multi_envs --train_tasks $train_tasks --eval_tasks $eval_tasks --algorithm_name $algorithm_name --num_env_steps $num_env_steps \
        --use_unified_env $use_unified_env --use_recurrent_policy $use_recurrent_policy --use_centralized_V $use_centralized_V \
        --hidden_size $hidden_size --n_head $n_head --n_block $n_block \
        --n_training_threads $n_training_threads --n_rollout_threads $n_rollout_threads --n_eval_rollout_threads $n_eval_rollout_threads --eval_episodes $eval_episodes --num_mini_batch $num_mini_batch --ppo_epoch $ppo_epoch --use_recurrent_with_agents $use_recurrent_with_agents --use_actor_loss $use_actor_loss \
        --use_latent_skills $use_latent_skills --skill_choice $skill_choice --skill_type $skill_type --num_skills $num_skills --share_tblocks $share_tblocks --skill_kl_loss $skill_kl_loss --skill_to_obs $skill_to_obs \
        --comm_channel $comm_channel --comm_use_active_masks $comm_use_active_masks --op_aggregate $op_aggregate --op_entity $op_entity \
        --use_similarity $use_similarity --sim_metrics $sim_metrics \
        --pi_choice $pi_choice --pi_use_obs $pi_use_obs --pi_use_latent $pi_use_latent \
        --use_action_predictor $use_action_predictor --n_future_steps $n_future_steps --kl_gamma $kl_gamma --comm_threshold $comm_threshold \
        --seed $SEED
fi

wait
