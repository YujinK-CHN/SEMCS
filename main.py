#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
import torch
import random
import argparse
from os.path import dirname, abspath
import traceback
import cProfile
import pstats

from utils.create_envs import make_train_env, make_eval_env
# sys.path.append("../../")
from config.config import get_common_config, get_smac_config, get_alicebob_config, get_football_config
from config.algo_policy_config import get_mcs_config, get_hmasd_config, get_DT2GS_config, get_ppo_config, get_mat_config, get_SESiL_config

from envs.alice_and_bob.alicebob_unified_maps import get_alicebob_params
from envs.football.football_maps import get_football_params
from envs.starcraft2.smac_maps import get_smac_params

results_path = os.environ.get("PYMARL_RESULT_DIR", dirname(dirname(abspath(__file__))))

# SEED_GPU_MAP = {1: 0, 10: 0, 20: 1, 30: 1, 40: 2, 50: 2, 60: 3, 70: 3}
SEED_GPU_MAP = {1: 0, 10: 0, 20: 1, 30: 1, 40: 2, 50: 2}
# SEED_GPU_MAP = {1: 2, 10: 2, 20: 2, 30: 3, 40: 3, 50: 3}

def main(args):
    parser = argparse.ArgumentParser(
        description='TransferMADRL', formatter_class=argparse.RawDescriptionHelpFormatter, conflict_handler="resolve")
    
    """
    parameters for transfer learning
        1. use_multi_envs: if multiple tasks are trained together
        2. train_tasks: specify a set of source tasks or one source task to be trained and evaluated
    """
    parser.add_argument("--use_wandb", type=int, default=False, help="[for wandb usage], by default False, will log date to wandb server. or else will use tensorboard to log data.")
    parser.add_argument("--use_norm_init", type=int, default=True, help="by default True, use normalized input layers.")
    parser.add_argument("--use_multi_envs", type=int, default=True, help="by default True, use multishared envs.")
    parser.add_argument("--task_id", type=str, default='0', help="specify the id of task")
    parser.add_argument("--train_tasks", type=str, default='v0', help="specify the set of tasks to train Transformer agent on multi envs, if use_multi_envs is true then set as 8m|3s5z|2s_vs_1sc")
    parser.add_argument("--eval_tasks", type=str, default='v0', help="specify the set of tasks to train Transformer agent on multi envs, if use_multi_envs is true then set as 8m|3s5z|2s_vs_1sc")
    parser.add_argument("--save_replay", type=int, default=False, help="whether save the replay when only evaluationg the model")

    # parameters for skill generation
    parser.add_argument("--use_latent_skills", type=int, default=True, help="by default True, use latent skills for training.")
    parser.add_argument("--num_skills", type=int, default=10, help="the number of skills")
    parser.add_argument("--skill_type", type=str, default="Continuous", choices=["Discrete", "Continuous", "None"])
    parser.add_argument("--skill_to_obs", type=str, default="merge", choices=["merge", "entity", "None"], help="decide how to store past skill in observation: merge (+num_skills), entity (+ num_entity*num_skills)")
    parser.add_argument("--skill_choice", type=str, default="UseVAE", choices=["UseVAE", "UseGRU", "UseTrans", "None"], help="the way to generate skills")
    parser.add_argument("--share_tblocks", type=int, default=False, help="decide if we need to share tblocks in Transformer.")
    parser.add_argument("--reduce_connection", type=int, default=True, help="decide if remove tprobs and directly use tblocks for later input.")
    parser.add_argument("--kl_gamma", type=float, default=0.95, help='discount factor for kl loss (default: 0.95)')
    parser.add_argument("--pred_type", type=str, default="CrossEntropy", choices=["KL", "CrossEntropy", "None"], help="which loss is used for the predictor")
    parser.add_argument("--record_grad_interval", type=int, default=20, help="interval to record gradients")
    parser.add_argument("--comm_threshold", type=float, default=0.5, help='threshold to decide whether to communicate or not')
    parser.add_argument("--use_mixed_percision", type=int, default=False, help="decide whether use mixed percision or not.")

    # parameters for training skills
    parser.add_argument("--use_similarity", type=int, default=0, help="whether to use skill training via enhancing similarites")
    parser.add_argument("--sim_metrics", type=str, default="Cosin", choices=["Cosin", "Wasserstein", "Covariance", "Contrastive", "None"], help="which similarity metrix is used for comparing data")
    parser.add_argument("--sim_coeffi", type=float, default=0.01, help="the coefficient of similarity metrix")
    parser.add_argument("--use_action_predictor", type=int, default=1, choices=[0, 1, 2, 3], help="whether to predict actions based on skills: 0 means do not predict; 1 means using for senders; 2 means using for receivers")
    parser.add_argument("--n_future_steps", type=int, default=1, help="how many steps is going to predict")
    parser.add_argument("--skill_loss_coeffi", type=float, default=0.1, help="the coefficient of using action predictor")
    parser.add_argument("--padded_values", type=int, default=-1, help="to pad future steps")

    # parameters for communication
    parser.add_argument("--comm_channel", type=str, default="All", choices=["All", "ExcludeSelf", "Direct", "CommMask", "None"], help="how to communicate")
    parser.add_argument("--op_entity", type=str, default="Mean", choices=["Mean", "Sum"], help="the way to operate entites when generating skills")
    parser.add_argument("--op_aggregate", type=str, default="Mean", choices=["Mean", "Sum", "GRU", "TransMean", "TransSum", "None"], help="the way to operate entites when generating skills")
    parser.add_argument("--comm_use_active_masks", type=int, default=False, help="whether to use active masks during communication")

    # parameters for integrating skills into policies
    parser.add_argument("--pi_choice", type=str, default="CatTrans", choices=["CatTrans", "CatInputs", "None"], help="the way to integrate observation and skills")
    parser.add_argument("--pi_use_obs", type=int, default=True, help="by default True, use obs in policy.")
    parser.add_argument("--pi_use_latent", type=int, default=True, help="by default True, use latent in policy.")

    # evaluation
    parser.add_argument("--eval_record_traj", type=int, default=True, help="whether to record trajctory during evaluation")
    parser.add_argument("--record_attention", type=int, default=False, help="whether save the attention when evalutating the model, only true when subtasks, used for mcs and DT2GS")

    # transfer learning
    parser.add_argument("--only_evaluate", type=int, default=False, help="without training, only evaluating the model from loading")
    parser.add_argument("--model_dir", type=str, default="None", help="by default 'None'. set the path to pretrained model.")
    parser.add_argument("--resume", action="store_true", default=False, help="resume training from the latest checkpoint in run_dir")
    parser.add_argument("--transfer_only_skill_generator", type=int, default=False, help="by default False, whether just restore skill generator without action policy")

    # commonly used parameters for all environments
    parser = get_common_config(parser)

    env_name = parser.parse_known_args(args)[0].env_name
    algo_name = parser.parse_known_args(args)[0].algorithm_name
    
    # set environments parameters
    if "StarCraft" in env_name:
        parser = get_smac_config(parser)
    elif "AliceBob" in env_name:
        parser = get_alicebob_config(parser)
    elif "Football" in env_name:
        parser = get_football_config(parser)
    else:
        raise NotImplementedError

    # set algorithm parameters
    if "dt2gs" in algo_name:
        parser = get_DT2GS_config(parser, env_name)
    elif "sesil" in algo_name or "newskill" in algo_name:
        parser = get_SESiL_config(parser, env_name)
    elif "sft" in algo_name or "mappo" in algo_name:
        parser = get_mcs_config(parser, env_name)
    elif "mcs" in algo_name:
        parser = get_mcs_config(parser, env_name)
    else:
        raise NotImplementedError

    all_args = parser.parse_known_args(args)[0]

    """
    parameters checking
    """
    if all_args.algorithm_name == "mcs":
        all_args.use_naive_recurrent_policy = False
        if "|" in all_args.train_tasks:
            assert all_args.use_multi_envs == 1, "Please use multi_envs when use multiple tasks!"
        assert all_args.pi_use_obs or all_args.pi_use_latent, "Policy should use either obs or latent!"
        if all_args.pi_use_obs and not all_args.pi_use_latent:  # if no skill is involved
            assert all_args.use_latent_skills == 0, "Do not use latent skills."
            assert all_args.comm_channel == "None", "Do not use communication channel when skill is not involved."
            assert all_args.use_similarity == 0, "Do not use similarity metrix when skill is not involved."
            assert all_args.skill_to_obs == "None", "Do not use skill in RNN state."
        # store skills in obs for later use
        if all_args.pi_use_latent:
            assert all_args.skill_to_obs != "None", "Do store skill to obs."
        # when use_action_predictor is not 0 or 1, communciation channel must be used
        if all_args.use_action_predictor != 0:
            assert all_args.pi_use_latent, "Action predictor needs latents."
        if all_args.use_action_predictor == 2:
            assert all_args.comm_channel != "None", "When predictor considers receiver's perspective, communciation channel must be used!"
        if all_args.use_action_predictor == 3:
            assert all_args.n_future_steps == 1, "When using SkillAllPredictor, only considering 1 step."
        # when use entity-based Transformer, use direct communication channel
        if all_args.skill_to_obs == "entity" and "Trans" in all_args.op_aggregate:
            assert all_args.comm_channel == "Direct", "When use entity-based Transformer, use direct communication channel!"
    elif all_args.algorithm_name == "dt2gs":
        all_args.use_naive_recurrent_policy = False
        if "|" in all_args.train_tasks:
            assert all_args.use_multi_envs == 1, "Please use multi_envs when use multiple tasks!"
        assert all_args.pi_use_obs or all_args.pi_use_latent, "Policy should use either obs or latent!"
        if all_args.pi_use_latent:
            assert all_args.skill_to_obs != "None", "Do store subtask latent to obs."
        all_args.use_latent_skills = bool(all_args.pi_use_latent)
        all_args.skill_choice = "UseVAE"
        all_args.num_skills = all_args.num_subtask
        all_args.use_action_predictor = 0
        all_args.use_similarity = 0
        all_args.comm_channel = "None"
    elif all_args.algorithm_name == "sesil":
        all_args.use_naive_recurrent_policy = False
        if "|" in all_args.train_tasks:
            assert all_args.use_multi_envs == 1, "Please use multi_envs when use multiple tasks!"
        all_args.use_action_predictor = 0
        all_args.use_similarity = 0
        if all_args.evo_solver_algo == "mappo":
            all_args.pi_use_obs = 1
            all_args.pi_use_latent = 0
            all_args.use_latent_skills = 0
            all_args.skill_choice = "None"
            all_args.comm_channel = "None"
            all_args.skill_to_obs = "None"
        elif all_args.evo_solver_algo == "mcs":
            all_args.pi_use_obs = 1
            if all_args.pi_use_latent:
                all_args.use_latent_skills = bool(all_args.pi_use_latent)
                assert all_args.skill_to_obs != "None", "Do store skill to obs."
            else:
                all_args.use_latent_skills = 0
                all_args.skill_to_obs = "None"
        elif all_args.evo_solver_algo == "dt2gs":
            all_args.use_latent_skills = bool(all_args.pi_use_latent)
            all_args.skill_choice = "UseVAE"
            all_args.num_skills = all_args.num_subtask
            all_args.comm_channel = "None"
    elif all_args.algorithm_name in ("sft", "mappo"):
        all_args.use_naive_recurrent_policy = False
        if "|" in all_args.train_tasks:
            assert all_args.use_multi_envs == 1, "Please use multi_envs when use multiple tasks!"
        all_args.pi_use_obs = 1
        all_args.pi_use_latent = 0
        all_args.use_latent_skills = 0
        all_args.skill_choice = "None"
        all_args.use_action_predictor = 0
        all_args.use_similarity = 0
        all_args.comm_channel = "None"
        all_args.skill_to_obs = "None"
    elif all_args.algorithm_name == "newskill":
        all_args.use_naive_recurrent_policy = False
        assert "|" in all_args.train_tasks, "newskill requires multiple tasks!"
        assert all_args.use_multi_envs == 1, "Please use multi_envs when use multiple tasks!"
        n_total = len(all_args.train_tasks.split("|"))
        assert all_args.newskill_known_tasks + all_args.newskill_unknown_tasks == n_total, \
            f"known ({all_args.newskill_known_tasks}) + unknown ({all_args.newskill_unknown_tasks}) must equal total tasks ({n_total})"
        if all_args.newskill_variant == "mappo":
            all_args.pi_use_obs = 1
            all_args.pi_use_latent = 0
            all_args.use_latent_skills = 0
            all_args.skill_choice = "None"
            all_args.use_action_predictor = 0
            all_args.use_similarity = 0
            all_args.comm_channel = "None"
            all_args.skill_to_obs = "None"
        elif all_args.newskill_variant == "sesil_mappo":
            all_args.pi_use_obs = 1
            all_args.pi_use_latent = 0
            all_args.use_latent_skills = 0
            all_args.skill_choice = "None"
            all_args.use_action_predictor = 0
            all_args.use_similarity = 0
            all_args.comm_channel = "None"
            all_args.skill_to_obs = "None"
    else:
        raise NotImplementedError
        
    # set episode length
    if "|" not in all_args.train_tasks:
        if all_args.env_name == "AliceBob":
            all_args.episode_length = get_alicebob_params(all_args.train_tasks)["limit"]
        elif all_args.env_name == "Football":
            all_args.episode_length = get_football_params(all_args.train_tasks)["limit"]
        elif all_args.env_name == "StarCraft":
            all_args.episode_length = get_smac_params(all_args.train_tasks)["limit"]

    print("episode_length:", all_args.use_multi_envs, all_args.train_tasks, all_args.episode_length)

    # cuda
    use_cuda = all_args.cuda and torch.cuda.is_available()
    torch.set_num_threads(all_args.n_training_threads)
    if use_cuda:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            gpu_id = 0
        else:
            # If you have more than 2 cards you could mod‐wrap or default:
            gpu_id = SEED_GPU_MAP.get(all_args.seed, 0) % num_gpus
        print(f"Choosing GPU {gpu_id} device (CUDA available: {num_gpus})")
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Choosing CPU")
        device = torch.device("cpu")

    # set folder and recording path name
    alg_setting_name = "{}_{}".format(all_args.algorithm_name, all_args.experiment_name)
    num_tasks = len(all_args.train_tasks.split("|"))
    if num_tasks <= 4:
        proj_env_tasks_name = "{}_train({})".format(all_args.env_name, all_args.train_tasks)
    else:
        proj_env_tasks_name = "{}_train_on_{}".format(all_args.env_name, num_tasks)
    alg_setting_seed_name = "{}_{}_s{}".format(all_args.algorithm_name, all_args.experiment_name, all_args.seed)
    run_dir = os.path.join(results_path, proj_env_tasks_name, alg_setting_seed_name)

    # specify where the model comes from: envs and algorithms
    if all_args.model_dir != "None":
        if "evaluation" in all_args.experiment_name or "transfer" in all_args.experiment_name:
            run_dir = os.path.join(run_dir, all_args.model_dir.split("/")[-3]+"_"+all_args.model_dir.split("/")[-2])

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    all_args.run_dir = run_dir
    all_args.replay_dir = os.path.join(os.getcwd(), results_path, proj_env_tasks_name)
    
    if "Football" in all_args.env_name:
        train_task = all_args.train_tasks
        train_task = train_task.split("|")
        train_task = [t.replace("academy_", "").replace("_with_keeper", "") for t in train_task]
        train_task = "|".join(train_task)
    else:
        train_task = all_args.train_tasks
        
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project="TransferComm_"+all_args.env_name+"_"+all_args.project_name,
                         notes=socket.gethostname(),
                         name=alg_setting_seed_name,
                         group=train_task+"_"+alg_setting_name,
                         dir=str(all_args.run_dir),
                         job_type="training",
                         reinit=True,
                         mode="online")
    
    # seed
    random.seed(all_args.seed)
    os.environ['PYTHONHASHSEED'] = str(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    """
    create a list of envs to be used for training and evaluation
    """
    # print(all_args)
    try:
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        #############################################################
        """
        Run with transfer learing:
            Runner: use multiple training and evaluation environments
            Policy: use one single policy for all environments
            Trainer: use samples from multuple environments to train the policy
            Buffer: store data from multiple environments
        """
        run_config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "device": device
        }
        if all_args.algorithm_name == "newskill":
            from runner.policy.newskill_runner import NewskillRunner as Runner
        elif "sesil" in all_args.algorithm_name:
            from runner.policy.sesil_runner import sesilETERunner as Runner
        elif "sft" in all_args.algorithm_name:
            from runner.policy.sft_runner import sftETERunner as Runner
        elif "mappo" in all_args.algorithm_name:
            from runner.policy.mappo_runner import mappoETERunner as Runner
        elif "dt2gs" in all_args.algorithm_name:
            from runner.policy.dt2gs_runner import dt2gsETERunner as Runner
        elif "mcs" in all_args.algorithm_name:
            from runner.policy.mcs_runner import mcsETERunner as Runner
        else:
            raise NotImplementedError
        runner = Runner(run_config)
        profiler = cProfile.Profile()
        profiler.enable()
        runner.run()
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumtime').print_stats(5)
        print("......Finish succeffully~~Start to close envs......")
    except Exception as e:
        traceback.print_exc()
        print("......Post Exeption, start to close envs......")

    ########## close envs ##########
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        print("Close eval envs.....")
        eval_envs.close()
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.close()
        print("Finishing writing...")
    print("Envs closed...")
    ########## close envs ##########


if __name__ == "__main__":
    main(sys.argv[1:])

