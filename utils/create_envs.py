#!/usr/bin/env python
import os
from os.path import dirname, abspath
from copy import deepcopy

# sys.path.append("../../")
from envs.alice_and_bob.alicebob_unified_maps import get_alicebob_params
from envs.football.football_maps import get_football_params
from envs.starcraft2.smac_maps import get_smac_params
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, MultiEnvShareSubprocVecEnv


"""
create train and testing envs
1. make_train_env: create training environments for agents based on source tasks
2. make_eval_env: create evaluation environments for agents based on source tasks
"""

def init_env(all_args, rank):
    if "AliceBob" == all_args.env_name:
        if all_args.use_unified_env:
            from envs.alice_and_bob.alice_and_bob_unified import AliceBob_Unified as AliceBob
        else:
            from envs.alice_and_bob.alice_and_bob import AliceBob
        env = AliceBob(all_args, seed=all_args.seed + rank * 1000)
    elif "Football" == all_args.env_name:
        from envs.football.football_unified import FootballUnifiedEnv
        env = FootballUnifiedEnv(all_args)
        env.seed(all_args.seed + rank * 1000)
    elif "StarCraft" == all_args.env_name:
        if all_args.use_unified_env:
            from envs.starcraft2.StarCraft2_UnifiedEnv import StarCraft2UnifiedEnv as StarCraft2Env
        else:
            from envs.starcraft2.StarCraft2_Env import StarCraft2Env
        env = StarCraft2Env(all_args)
        env.seed(all_args.seed + rank * 1000)
    else:
        print("Can not support the " + all_args.env_name + "environment.")
        raise NotImplementedError
    return env

def make_train_env(all_args):
    """
    @param use_multi_envs: decide whether to use multi-task or single-task
    @param train_tasks: decide which tasks are used for training
    """
    def get_env_fn(args, rank=0):
        def init_fn():
            return init_env(args, rank)
        return init_fn
    
    if not all_args.use_multi_envs:
        all_args.map_name = all_args.train_tasks
        all_args.task_id = "{}-{}".format(all_args.map_name, 0) if "AliceBob" in all_args.env_name and "-" not in all_args.map_name else all_args.map_name
        n_agents, n_enemies, n_entities = get_num_agents_entities(all_args)
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(all_args)], n_agents, n_enemies, n_entities)
        else:
            return ShareSubprocVecEnv([get_env_fn(all_args, i) for i in range(all_args.n_rollout_threads)], n_agents, n_enemies, n_entities)
    else:
        # multi_envs is concatenate from multi map_name by |
        if "|" in all_args.train_tasks:
            multi_envs = all_args.train_tasks.split('|')
        else:
            multi_envs = [all_args.train_tasks]
        assert all_args.n_rollout_threads % len(multi_envs) == 0, \
            f"n_rollout_threads {all_args.n_rollout_threads} should divided by num of multi_envs {len(multi_envs)}"
        num_threads_per_env = all_args.n_rollout_threads // len(multi_envs)
        # print(f"num_threads_per_env: {num_threads_per_env}")
        multi_env_list = []
        n_agent_list = []
        n_enemy_list = []
        n_entity_list = []
        current_thread = 0
        for idx, map_name in enumerate(multi_envs):
            # print(f"map_name: {map_name}")
            seed_thread = 0
            args_copy = deepcopy(all_args)
            args_copy.map_name = map_name
            args_copy.task_id = "{}-{}".format(map_name, idx) if "AliceBob" in all_args.env_name and "-" not in map_name else map_name
            if num_threads_per_env > 1:
                multi_env_list.append([get_env_fn(args_copy, seed_thread+i) for i in range(num_threads_per_env)])
            else:
                multi_env_list.append([get_env_fn(args_copy)])
            n_agents, n_enemies, n_entities = get_num_agents_entities(args_copy)
            n_agent_list.append(n_agents)
            n_enemy_list.append(n_enemies)
            n_entity_list.append(n_entities)
            current_thread += num_threads_per_env
        assert current_thread == all_args.n_rollout_threads  # use all threads for transfer learning     
        return MultiEnvShareSubprocVecEnv(multi_env_list, num_threads_per_env, multi_envs, n_agent_list, n_enemy_list, n_entity_list)


def make_eval_env(all_args):
    """
    @param use_multi_envs: decide whether to use multi-task or single-task
    @param eval_tasks: decide which tasks are used for evaluation
    """
    def get_env_fn(args, rank=0):
        def init_fn():
            return init_env(args, rank)
        return init_fn
    
    if not all_args.use_multi_envs:
        all_args.map_name = all_args.eval_tasks
        n_agents, n_enemies, n_entities = get_num_agents_entities(all_args)
        if all_args.n_eval_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(all_args)], n_agents, n_enemies, n_entities)
        else:
            return ShareSubprocVecEnv([get_env_fn(all_args, i) for i in range(all_args.n_eval_rollout_threads)], n_agents, n_enemies, n_entities)
    else:
        if "|" in all_args.eval_tasks:
            multi_envs = all_args.eval_tasks.split('|')
        else:
            multi_envs = [all_args.eval_tasks]
        if all_args.n_eval_rollout_threads > 1:
            assert all_args.n_eval_rollout_threads % len(multi_envs) == 0, \
                f"n_eval_rollout_threads {all_args.n_eval_rollout_threads} should divided by num of multi_envs {len(multi_envs)}"
            num_threads_per_env = all_args.n_eval_rollout_threads // len(multi_envs)
            # print(f"num_threads_per_env: {num_threads_per_env}")
            multi_env_list = []
            n_agent_list = []
            n_enemy_list = []
            n_entity_list = []
            current_thread = 0
            for idx, map_name in enumerate(multi_envs):
                seed_thread = 0
                args_copy = deepcopy(all_args)
                args_copy.map_name = map_name
                args_copy.task_id = "{}-{}".format(map_name, idx) if "AliceBob" in all_args.env_name and "-" not in map_name else map_name
                if num_threads_per_env > 1:
                    multi_env_list.append([get_env_fn(args_copy, seed_thread+i) for i in range(num_threads_per_env)])
                else:
                    multi_env_list.append([get_env_fn(args_copy)])
                n_agents, n_enemies, n_entities = get_num_agents_entities(args_copy)
                n_agent_list.append(n_agents)
                n_enemy_list.append(n_enemies)
                n_entity_list.append(n_entities)
                current_thread += num_threads_per_env
            assert current_thread == all_args.n_eval_rollout_threads  # use all threads for transfer learning     
            return MultiEnvShareSubprocVecEnv(multi_env_list, num_threads_per_env, multi_envs, n_agent_list, n_enemy_list, n_entity_list)
        else:
            multi_env_list = []
            n_agent_list = []
            n_enemy_list = []
            n_entity_list = []
            for idx, map_name in enumerate(multi_envs):
                args_copy = deepcopy(all_args)
                args_copy.map_name = map_name
                args_copy.task_id = "{}-{}".format(map_name, idx) if "-" not in map_name else map_name
                multi_env_list.append([get_env_fn(args_copy)])
                n_agents, n_enemies, n_entities = get_num_agents_entities(args_copy)
                n_agent_list.append(n_agents)
                n_enemy_list.append(n_enemies)
                n_entity_list.append(n_entities)
            return MultiEnvShareSubprocVecEnv(multi_env_list, 1, multi_envs, n_agent_list, n_enemy_list, n_entity_list)


def get_num_agents_entities(args):
    if "AliceBob" == args.env_name:
        num_agents = get_alicebob_params(args.map_name)["n_agents"]
        num_enemies = get_alicebob_params(args.map_name)["n_keys"] + get_alicebob_params(args.map_name)["n_goals"]
        if args.use_grids_obs:
            num_enemies += 8  # 8 grids
        num_entities = num_agents + num_enemies
    elif "StarCraft" == args.env_name:
        num_agents = get_smac_params(args.map_name)["n_agents"]
        num_enemies = get_smac_params(args.map_name)["n_enemies"]
        num_entities = num_agents + num_enemies
    elif "Football" == args.env_name:
        num_agents = get_football_params(args.map_name)["n_agents"]
        num_enemies = 0
        num_entities = 4
    else:
        raise NotImplementedError
    return num_agents, num_enemies, num_entities

