import time
from functools import reduce
import wandb
import os
import numpy as np
import torch
import pickle

import sys
sys.stdout.flush()

from runner.policy.base_runner import Runner

from base_policy.utils.util import _t2n, stack_over_steps_with_padding, print_cuda
from base_policy.utils.multi_envs_shared_buffer import MultiEnvSharedReplayBufferComm
from base_policy.algorithms.mcs.mcs_policy import mcsPolicy as Policy
from base_policy.algorithms.mcs.mcs_trainer import mcsTrainer as Trainer

class mcsETERunner(Runner):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):   
        super(mcsETERunner, self).__init__(config)

        # parameters for HMASD
        self.use_sparse_reward = self.all_args.use_sparse_reward

        # use for multiple environments
        self.multi_envs = self.envs.multi_envs
        self.eval_multi_envs = self.eval_envs.multi_envs
        self.num_multi_envs = len(self.multi_envs)
        self.num_thread_per_env = self.envs.num_thread_per_env
        self.num_eval_thread_per_env = self.eval_envs.num_thread_per_env
        self.obs_space_list = self.envs.observation_space
        self.cent_obs_space_list = self.envs.share_observation_space
        self.act_space_list = self.envs.action_space
        self.quo = self.all_args.eval_episodes // (self.num_multi_envs * self.num_eval_thread_per_env)
            
        # for predictor
        self.n_future_steps = self.all_args.n_future_steps
        self.use_action_predictor = self.all_args.use_action_predictor
        self.padded_values = self.all_args.padded_values            
        
        # parameters for evaluation
        self.eval_deterministic = self.all_args.eval_deterministic
        print("eval_deterministic:", self.eval_deterministic)
        
        # policy network
        self.policy = Policy(self.all_args,
                            self.multi_envs,
                            self.num_thread_per_env,
                            self.envs.observation_space,
                            self.share_observation_space,
                            self.envs.action_space,
                            device = self.device)

        
        # algorithm
        self.trainer = Trainer(self.all_args, self.policy, self.num_agents, self.num_enemies, self.num_entities, device = self.device)
                
        """
        Transfer learning: restore model 
        """
        if self.model_dir != "None":
            print("restore actors, critics, and vnorm...")
            self.restore()

        # buffer     
        self.buffer = MultiEnvSharedReplayBufferComm(
            self.all_args, self.num_agents, self.num_entities,
            self.obs_space_list, self.cent_obs_space_list, self.act_space_list, 
            self.num_thread_per_env
        )


    def run2(self):
        for episode in range(1):
            self.eval(episode)

    def run(self):
        # we only evaluation the model from skills
        if self.model_dir != "None":
            if self.all_args.only_evaluate or self.all_args.save_replay:
                print("start to replay...")
                self.evaluate4replay()
            return

        # training
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        start_episode = 0
        if getattr(self.all_args, 'resume', False):
            start_episode = self.restore_checkpoint()

        # record for rewards
        done_episodes_rewards = [[] for _ in range(self.num_multi_envs)]
        one_episode_rewards = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]
        train_episode_steps = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]

        # record for win rates
        battles_game_num = [0 for _ in range(self.num_multi_envs)]
        battles_won_num = [0 for _ in range(self.num_multi_envs)]
                
        ######################### parameters for different envs #########################
        # parameters for AliceBob
        if "AliceBob" in self.env_name:
            battles_goals = [0 for _ in range(self.num_multi_envs)]
            battles_step = [0 for _ in range(self.num_multi_envs)]
        if "StarCraft" in self.env_name:
            last_battles_game = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]
            last_battles_won = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]
        ######################### parameters for different envs #########################

        for episode in range(start_episode, episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            if self.use_action_predictor:
                rewards_episode, infos_episode, dones_episode = self.collect_data_preds(self.episode_length)
            else:
                rewards_episode, infos_episode, dones_episode = self.collect_data(self.episode_length)
            self.compute()     

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
                   
            train_infos = self.train(episode)

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode=episode)

            ######################### Record Information #########################
            for rewards_step, infos_step, dones_step in zip(rewards_episode, infos_episode, dones_episode):                
                for idx, (rewards_tuple, infos_tuple, dones_tuple) in enumerate(zip(rewards_step, infos_step, dones_step)):
                    dones_env = np.all(dones_tuple, axis=1)

                    reward_env = np.mean(rewards_tuple, axis=1).flatten()
                    one_episode_rewards[idx] += reward_env
                    for t in range(self.num_thread_per_env):
                        train_episode_steps[idx][t] += 1
                        if dones_env[t]:
                            # rewards
                            done_episodes_rewards[idx].append(one_episode_rewards[idx][t])         
                            ######################### Alice Bob ###################
                            # SMAC holds global varibles to record win rates
                            if "AliceBob" in self.env_name:
                                if infos_tuple[t][0]["battle_won"]:
                                    battles_won_num[idx] += 1
                                battles_goals[idx] += infos_tuple[t][0]["goals"]
                                battles_step[idx] += train_episode_steps[idx][t]
                                # count for finished env to avoid dead threads 
                                battles_game_num[idx] += 1
                            train_episode_steps[idx][t] = 0 # reset steps 
                            one_episode_rewards[idx][t] = 0 # reset episode rewards       
                        ######################### record info ##################                    

            # log information
            if episode % self.log_interval == 0:
                if len(done_episodes_rewards) > 0:
                    # avergae over all tasks
                    train_infos["train_episode_rewards"] = [np.mean(episode_reward) for episode_reward in done_episodes_rewards]
                    done_episodes_rewards = [[] for _ in range(self.num_multi_envs)]
                ######################### log extra info #########################
                if "AliceBob" in self.env_name:
                    train_win_rate = []
                    train_goals = []
                    train_battles_step = []
                    for b_won, b_goal, b_step, b_num in zip(battles_won_num, battles_goals, battles_step, battles_game_num):
                        train_win_rate.append(b_won / b_num)
                        train_goals.append(b_goal / b_num)
                        train_battles_step.append(b_step / b_num)
                    
                    train_infos["train_win_rate"] = train_win_rate
                    train_infos["train_goals"] = train_goals
                    train_infos["train_battles_step"] = train_battles_step
                    
                    # reset parameters
                    battles_game_num = [0 for _ in range(self.num_multi_envs)]
                    battles_won_num = [0 for _ in range(self.num_multi_envs)]
                    battles_goals = [0 for _ in range(self.num_multi_envs)]
                    battles_step = [0 for _ in range(self.num_multi_envs)]
                    
                if "StarCraft" in self.env_name:
                    train_incre_win_rate = []
                    train_win_rate = []
                    # only last step info is needed, which is accumulated over episodes
                    for idx, infos in enumerate(infos_episode[-1]):
                        battles_won = []
                        battles_game = []
                        incre_battles_won = []
                        incre_battles_game = []
                        for i, info in enumerate(infos):
                            if 'battles_won' in info[0].keys():
                                battles_won.append(info[0]['battles_won'])
                                incre_battles_won.append(info[0]['battles_won']-last_battles_won[idx][i])
                            if 'battles_game' in info[0].keys():
                                battles_game.append(info[0]['battles_game'])
                                incre_battles_game.append(info[0]['battles_game']-last_battles_game[idx][i])
                        train_incre_win_rate.append(np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0)
                        train_win_rate.append(np.sum(battles_won)/np.sum(battles_game) if np.sum(battles_game)>0 else 0.0)
                        last_battles_game[idx] = battles_game
                        last_battles_won[idx] = battles_won

                    train_infos['train_incre_win_rate'] = train_incre_win_rate
                    train_infos['train_win_rate'] = train_win_rate
                    train_dead_ratio = []
                    for idx in range(self.num_multi_envs):
                        train_dead_ratio.append(1 - self.buffer.buffer_lists[idx].active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.buffer_lists[idx].active_masks.shape)))
                    train_infos['train_dead_ratio'] = train_dead_ratio
                ######################### log extra info #########################
                self.log_train(train_infos, total_num_steps)

                # print info
                end = time.time()
                print("\n Env {}-{} Algo {} Exp {} updates {}/{} episodes, rewards {:.2f}, total num timesteps {}/{}, FPS {}, Time {:.2f} hour.\n"
                        .format(self.all_args.env_name,
                                self.all_args.seed,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                np.mean(train_infos["train_episode_rewards"]),
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start)), 
                                (end - start)/3600))

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                if self.n_eval_rollout_threads == 1 and "StarCraft" in self.env_name:
                    self.eval(total_num_steps, episode)
                else:
                    self.eval_parallel(total_num_steps, episode)


    @torch.no_grad()
    def eval(self, total_num_steps, episode):
        """
        this evaluation requires the environment restarts automatically; used for SMAC
        """
        assert self.num_eval_thread_per_env == 1, "Eval thread must be 1!"
        eval_obs_list, eval_share_obs_list, eval_available_actions_list, idxs_tuple = self.eval_envs.reset()
        eval_episode_rewards = [[] for _ in idxs_tuple]
        one_episode_rewards = [[] for _ in idxs_tuple]
        eval_episode = [0 for _ in idxs_tuple]
        eval_dones_all = [False for _ in idxs_tuple]
        one_episode_steps = [0 for _ in idxs_tuple]
        eval_battles_won = [0 for _ in idxs_tuple]
        
        eval_rnn_states_list = []
        eval_rnn_states_comm_list = []
        eval_masks_lists = []
        eval_active_masks_lists = []        
        for idx in idxs_tuple:
            eval_rnn_states_list.append(
                np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states.shape[3:]), dtype=np.float32)
            )
            eval_rnn_states_comm_list.append(
                np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states_comm.shape[3:]), dtype=np.float32)
            )
            eval_masks_lists.append(
                np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
            )
            eval_active_masks_lists.append(
                np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
            )
        # init subtask
        eval_skills = None
        
        ###### save data for the whole trajecotry ######
        save_entity_obs_lists = [[] for _ in idxs_tuple]
        save_skill_lists = [[] for _ in idxs_tuple]
        save_done_lists = [[] for _ in idxs_tuple]
        save_win_lists = [[] for _ in idxs_tuple]
        save_actions_lists = [[] for _ in idxs_tuple]
        save_skill_dot_lists = [[] for _ in idxs_tuple]
        save_comm_skill_dot_lists = [[] for _ in idxs_tuple]
        save_comm_weight_lists = [[] for _ in idxs_tuple]
        ###### save data for the whole trajecotry ######

        while True:
            self.trainer.prep_rollout()
            eval_obs_list = [np.concatenate(eval_obs_list[idx]) for idx in idxs_tuple]
            
            # view skills as additional observation
            if self.all_args.use_latent_skills and self.all_args.skill_to_obs != "None":
                # the first time when eval_skills is empty
                if eval_skills is None:
                    if self.all_args.skill_to_obs == "merge": 
                        eval_skills = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_skills, dtype=torch.float32) for eval_obs in eval_obs_list]
                    elif self.all_args.skill_to_obs == "entity": 
                        eval_skills = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_skills*n_en, dtype=torch.float32) for eval_obs, n_en in zip(eval_obs_list, self.num_entities)]
                eval_skill_list = [_t2n(item) for item in eval_skills]

                # reshape skills (this can be prevented by outputing skills for all entities)
                eval_skill_list = [np.reshape(skill, (np.shape(eval_obs)[0], -1)) for eval_obs, skill in zip(eval_obs_list, eval_skill_list)]  
                eval_obs_list = [np.concatenate([eval_obs, eval_skill], axis=-1) for eval_obs, eval_skill in zip(eval_obs_list, eval_skill_list)]
                                
                eval_skill_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_skills]

            eval_rnn_states_list = [np.concatenate(eval_rnn_states_list[idx]) for idx in idxs_tuple]
            eval_rnn_states_comm_list = [np.concatenate(eval_rnn_states_comm_list[idx]) for idx in idxs_tuple]
            eval_masks_lists = [np.concatenate(eval_masks_lists[idx]) for idx in idxs_tuple] 
            eval_active_masks_lists = [np.concatenate(eval_active_masks_lists[idx]) for idx in idxs_tuple]            
            eval_available_actions_list = [np.concatenate(eval_available_actions_list[idx]) for idx in idxs_tuple]
            
            eval_actions, eval_rnn_states, eval_rnn_states_comm, eval_entity_obs, eval_skills, record_info  = \
                self.trainer.policy.act(
                    eval_obs_list, eval_rnn_states_list, eval_rnn_states_comm_list, eval_masks_lists, eval_active_masks_lists, eval_available_actions_list,
                    deterministic=self.eval_deterministic, n_agents=self.eval_num_agents, n_enemies=self.eval_num_enemies, n_entities=self.eval_num_entities)

            eval_entity_obs_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_entity_obs]                
            th_na_list = [eval_ob.shape[0] for eval_ob in eval_obs_list]
            eval_actions_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in torch.split(eval_actions, th_na_list, dim=0)]

            eval_rnn_states_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_rnn_states]
            eval_rnn_states_comm_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_rnn_states_comm]

            # Obser reward and next obs
            eval_obs_list, eval_share_obs_list, eval_rewards_list, \
                eval_dones_list, eval_infos_list, eval_available_actions_list, idxs_tuple = self.eval_envs.step(eval_actions_list)
            # Win
            eval_win_flag = [np.array([False]) for _ in eval_dones_list]

            for eval_rewards, eval_dones, eval_infos, idx in zip(eval_rewards_list, eval_dones_list, eval_infos_list, idxs_tuple):
                one_episode_rewards[idx].append(eval_rewards)
                eval_dones_env = np.all(eval_dones, axis=1)
                eval_rnn_states_list[idx][eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states.shape[3:]), dtype=np.float32)
                eval_rnn_states_comm_list[idx][eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states_comm.shape[3:]), dtype=np.float32)
                
                eval_masks = np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_masks_lists[idx] = eval_masks
                
                eval_active_masks = np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
                
                # if all agents done then set 1 if some agents done set 0
                eval_active_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                eval_active_masks[eval_dones_env == True] = np.ones(((eval_dones_env == True).sum(), self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_active_masks_lists[idx] = eval_active_masks

                for eval_i in range(self.num_eval_thread_per_env):
                    one_episode_steps[idx] += 1
                    if eval_dones_env[eval_i]:
                        eval_episode[idx] += 1
                        eval_episode_rewards[idx].append(np.sum(one_episode_rewards[idx], axis=0))
                        one_episode_rewards[idx] = []
                        ######################### record info #########################
                        if "StarCraft" in self.env_name:
                            if eval_infos[eval_i][0].get("won", False):
                                eval_battles_won[idx] += 1  # this is not divided by num_eval_thread_per_env!!!
                                eval_win_flag[idx] = np.array([True])              
                        ######################### record info #########################

            for idx in idxs_tuple:
                if eval_episode[idx] >= self.all_args.eval_episodes and eval_dones_all[idx] == False:
                    eval_dones_all[idx] = True
                    eval_env_infos = {'eval_episode_rewards_{}'.format(self.eval_multi_envs[idx]): np.mean(eval_episode_rewards[idx])}                
                    ######################### log extra info #########################
                    if "StarCraft" in self.env_name:
                        eval_env_infos["eval_win_rate_{}".format(self.eval_multi_envs[idx])] = eval_battles_won[idx] / eval_episode[idx]
                    ######################### log extra info #########################
                    self.log_eval(eval_env_infos, total_num_steps)

            # store data, the tasks with shorter epsiodes will still be recorded for the simplicity
            for idx in idxs_tuple:  # only store the last 10 episodes
                if eval_episode[idx] >= self.all_args.eval_episodes-10 and eval_dones_all[idx] == False:
                    save_entity_obs_lists[idx].append(eval_entity_obs_list[idx])
                    if self.all_args.use_latent_skills and self.all_args.skill_to_obs != "None":
                        save_skill_lists[idx].append(eval_skill_list[idx])
                    save_done_lists[idx].append(np.all(eval_dones_list[idx], axis=1))
                    save_win_lists[idx].append(eval_win_flag[idx])
                    save_actions_lists[idx].append(eval_actions_list[idx])
                    # for recording attention
                    if self.all_args.record_attention:
                        save_skill_dot_lists[idx].append(np.array(np.split(_t2n(record_info["skill_dot"][idx]), self.num_eval_thread_per_env)))
                        save_comm_skill_dot_lists[idx].append(np.array(np.split(_t2n(record_info["comm_skill_dot"][idx]), self.num_eval_thread_per_env)))
                        save_comm_weight_lists[idx].append(np.array(np.split(_t2n(record_info["comm_weights"][idx]), self.num_eval_thread_per_env)))

            if np.all(eval_dones_all):
                if not self.all_args.save_replay and self.all_args.eval_record_traj and episode%(self.eval_interval*20)==0:
                    self.save_trajectory(total_num_steps, save_entity_obs_lists, save_skill_lists, save_done_lists, save_win_lists, save_actions_lists, 
                                         skill_dot_lists=save_skill_dot_lists, 
                                         comm_skill_dot_lists=save_comm_skill_dot_lists,
                                         comm_weight_lists=save_comm_weight_lists)
                break


    @torch.no_grad()
    def eval_parallel(self, total_num_steps, episode):
        eval_obs_list, eval_share_obs_list, eval_available_actions_list, idxs_tuple = self.eval_envs.reset()
        eval_episode_rewards = [[0]*self.num_eval_thread_per_env for _ in idxs_tuple]
        one_episode_rewards = [[] for _ in idxs_tuple] # with different thread in []
        one_episode_steps = [[0]*self.num_eval_thread_per_env for _ in idxs_tuple]
        eval_battles_won = [0 for _ in idxs_tuple]
        recorded_envs = [False for _ in idxs_tuple]
        
        # set unfinished threads
        done_episodes_per_thread = np.zeros((len(idxs_tuple), self.num_eval_thread_per_env), dtype=int)        
        eval_episodes_per_thread = done_episodes_per_thread + self.quo
        unfinished_thread = done_episodes_per_thread != eval_episodes_per_thread
        
        ######################### parameters for different envs #########################
        # eval parameters for AliceBob
        if "AliceBob" in self.env_name or "Football" in self.env_name: 
            eval_battles_goals = [0 for _ in idxs_tuple]
            eval_battles_step = [0 for _ in idxs_tuple]
        ######################### parameters for different envs #########################
        
        eval_rnn_states_list = []
        eval_rnn_states_comm_list = []
        eval_masks_lists = []
        eval_active_masks_lists = []        
        for idx in idxs_tuple:
            eval_rnn_states_list.append(
                np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states.shape[3:]), dtype=np.float32)
            )
            eval_rnn_states_comm_list.append(
                np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states_comm.shape[3:]), dtype=np.float32)
            )
            eval_masks_lists.append(
                np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
            )
            eval_active_masks_lists.append(
                np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
            )
        # init subtask
        eval_skills = None

        ###### save data for the whole trajecotry ######
        save_entity_obs_lists = [[] for _ in idxs_tuple]
        save_skill_lists = [[] for _ in idxs_tuple]
        save_done_lists = [[] for _ in idxs_tuple]
        save_win_lists = [[] for _ in idxs_tuple]
        save_actions_lists = [[] for _ in idxs_tuple]
        save_comm_weight_lists = [[] for _ in idxs_tuple]
        save_skill_dot_lists = [[] for _ in idxs_tuple]
        save_comm_skill_dot_lists = [[] for _ in idxs_tuple]
        ###### save data for the whole trajecotry ######
        
        while True:
            # render the first evaluation episode
            if "AliceBob" in self.env_name and self.use_render and episode%(self.eval_interval*self.render_episodes)==0:
                for idx in enumerate(idxs_tuple):
                    if unfinished_thread[idx][0]:  # only render the first thread
                        self.eval_envs.render(idx)

            self.trainer.prep_rollout()
            eval_obs_list = [np.concatenate(eval_obs_list[idx]) for idx in idxs_tuple]
            
            # view skills as additional observation
            if self.all_args.use_latent_skills and self.all_args.skill_to_obs != "None":
                # the first time when eval_skills is empty
                if eval_skills is None:
                    if self.all_args.skill_to_obs == "merge":
                        eval_skills = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_skills, dtype=torch.float32) for eval_obs in eval_obs_list]
                    elif self.all_args.skill_to_obs == "entity": 
                        eval_skills = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_skills*n_en, dtype=torch.float32) for eval_obs, n_en in zip(eval_obs_list, self.num_entities)]
                eval_skill_list = [_t2n(item) for item in eval_skills]

                # reshape skills (this can be prevented by outputing skills for all entities)
                eval_skill_list = [np.reshape(skill, (np.shape(eval_obs)[0], -1)) for eval_obs, skill in zip(eval_obs_list, eval_skill_list)]  
                eval_obs_list = [np.concatenate([eval_obs, eval_skill], axis=-1) for eval_obs, eval_skill in zip(eval_obs_list, eval_skill_list)]
                                
                eval_skill_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_skills]

            eval_rnn_states_list = [np.concatenate(eval_rnn_states_list[idx]) for idx in idxs_tuple]
            eval_rnn_states_comm_list = [np.concatenate(eval_rnn_states_comm_list[idx]) for idx in idxs_tuple]
            eval_masks_lists = [np.concatenate(eval_masks_lists[idx]) for idx in idxs_tuple] 
            eval_active_masks_lists = [np.concatenate(eval_active_masks_lists[idx]) for idx in idxs_tuple]            
            eval_available_actions_list = [np.concatenate(eval_available_actions_list[idx]) for idx in idxs_tuple]
            
            eval_actions, eval_rnn_states, eval_rnn_states_comm, eval_entity_obs, eval_skills, record_info  = \
                self.trainer.policy.act(
                    eval_obs_list, eval_rnn_states_list, eval_rnn_states_comm_list, eval_masks_lists, eval_active_masks_lists, eval_available_actions_list,
                    deterministic=self.eval_deterministic, n_agents=self.eval_num_agents, n_enemies=self.eval_num_enemies, n_entities=self.eval_num_entities)

            eval_entity_obs_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_entity_obs]                
            th_na_list = [eval_ob.shape[0] for eval_ob in eval_obs_list]
            eval_actions_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in torch.split(eval_actions, th_na_list, dim=0)]

            eval_rnn_states_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_rnn_states]
            eval_rnn_states_comm_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_rnn_states_comm]

            # Obser reward and next obs
            eval_obs_list, eval_share_obs_list, eval_rewards_list, \
                eval_dones_list, eval_infos_list, eval_available_actions_list, idxs_tuple = self.eval_envs.step(eval_actions_list)
            # Win
            eval_win_flag = [np.array([False]) for _ in eval_dones_list]

            for eval_rewards, eval_dones, eval_infos, idx in zip(eval_rewards_list, eval_dones_list, eval_infos_list, idxs_tuple):
                eval_reward_env = np.mean(eval_rewards, axis=1) # remove agent index
                one_episode_rewards[idx].append(eval_reward_env)                
                eval_dones_env = np.all(eval_dones, axis=1)
                eval_rnn_states_list[idx][eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states.shape[3:]), dtype=np.float32)
                eval_rnn_states_comm_list[idx][eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states_comm.shape[3:]), dtype=np.float32)
                
                eval_masks = np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_masks_lists[idx] = eval_masks
                
                eval_active_masks = np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
                
                # if all agents done then set 1 if some agents done set 0
                eval_active_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                eval_active_masks[eval_dones_env == True] = np.ones(((eval_dones_env == True).sum(), self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_active_masks_lists[idx] = eval_active_masks

                for eval_i in range(self.num_eval_thread_per_env):
                    one_episode_steps[idx][eval_i] += 1
                    if unfinished_thread[idx][eval_i] and eval_dones_env[eval_i]:
                        done_episodes_per_thread[idx][eval_i] += 1
                        per_thread_one_episode_rewards = np.array(one_episode_rewards[idx])[:, eval_i]
                        eval_episode_rewards[idx][eval_i] += np.sum(per_thread_one_episode_rewards)
                        for ep in range(len(per_thread_one_episode_rewards)):  # reset the corresponding task and threads but it is fine since only 1 episode is used
                            one_episode_rewards[idx][ep][eval_i] = 0.0
                        ######################### record info #########################
                        if "AliceBob" in self.env_name:
                            if eval_infos[eval_i][0]['battle_won']:
                                eval_battles_won[idx] += 1
                            eval_battles_goals[idx] += eval_infos[eval_i][0]['goals']
                            eval_battles_step[idx] += one_episode_steps[idx][eval_i]
                            one_episode_steps[idx][eval_i] = 0
                        if "Football" in self.env_name:
                            if eval_infos[eval_i][0]['score_reward'] > 0:
                                eval_battles_won[idx] += 1
                            eval_battles_goals[idx] += eval_infos[eval_i][0]['score_reward']
                            eval_battles_step[idx] += one_episode_steps[idx][eval_i]
                            one_episode_steps[idx][eval_i] = 0               
                        ######################### record info #########################

            unfinished_thread = done_episodes_per_thread != eval_episodes_per_thread
            
            for idx in idxs_tuple:
                if np.all(done_episodes_per_thread[idx] == eval_episodes_per_thread[idx]) and not recorded_envs[idx]:
                    recorded_envs[idx] = True
                    eval_episode = np.sum(eval_episodes_per_thread[idx])
                    eval_env_infos = {'eval_episode_rewards_{}'.format(self.eval_multi_envs[idx]): np.mean(eval_episode_rewards[idx])}                
                    ######################### log extra info #########################
                    if "AliceBob" in self.env_name or "Football" in self.env_name:
                        eval_env_infos["eval_win_rate_{}".format(self.eval_multi_envs[idx])] = eval_battles_won[idx]/eval_episode
                        eval_env_infos["eval_goals_{}".format(self.eval_multi_envs[idx])] = eval_battles_goals[idx]/eval_episode
                        eval_env_infos["eval_steps_{}".format(self.eval_multi_envs[idx])] = eval_battles_step[idx]/eval_episode
                    ######################### log extra info #########################
                    self.log_eval(eval_env_infos, total_num_steps)

                # save data
                if not self.all_args.save_replay and self.all_args.eval_record_traj and episode%(self.eval_interval*20)==0:
                    if np.all(unfinished_thread[idx]) and not recorded_envs[idx]:
                        save_entity_obs_lists[idx].append(eval_entity_obs_list[idx])
                        if self.all_args.use_latent_skills and self.all_args.skill_to_obs != "None":
                            save_skill_lists[idx].append(eval_skill_list[idx])
                        save_done_lists[idx].append(np.all(eval_dones_list[idx], axis=1))
                        save_win_lists[idx].append(eval_win_flag[idx])
                        save_actions_lists[idx].append(eval_actions_list[idx])
                        # for recording attention
                        if self.all_args.record_attention:
                            save_skill_dot_lists[idx].append(np.array(np.split(_t2n(record_info["skill_dot"][idx]), self.num_eval_thread_per_env)))
                            save_comm_skill_dot_lists[idx].append(np.array(np.split(_t2n(record_info["comm_skill_dot"][idx]), self.num_eval_thread_per_env)))
                            save_comm_weight_lists[idx].append(np.array(np.split(_t2n(record_info["comm_weights"][idx]), self.num_eval_thread_per_env)))

            if np.all(done_episodes_per_thread == eval_episodes_per_thread):
                if not self.all_args.save_replay and self.all_args.eval_record_traj and episode%(self.eval_interval*20)==0:
                    self.save_trajectory(total_num_steps, save_entity_obs_lists, save_skill_lists, save_done_lists, save_win_lists, save_actions_lists,
                                         skill_dot_lists=save_skill_dot_lists, 
                                         comm_skill_dot_lists=save_comm_skill_dot_lists,
                                         comm_weight_lists=save_comm_weight_lists)
                break


    def warmup(self):
        # reset env
        obs_list, share_obs_list, available_actions_list, idxs_tuple = self.envs.reset()

        for obs, share_obs, available_actions, n_entity, idx in zip(obs_list, share_obs_list, available_actions_list, self.num_entities, idxs_tuple):
            # replay buffer
            if not self.use_centralized_V:
                share_obs = obs

            # past skills as aditional observations
            if self.all_args.skill_to_obs == "merge":
                self.buffer.buffer_lists[idx].share_obs[0] = np.concatenate([share_obs, np.zeros([self.num_thread_per_env, self.num_agents[idx], self.all_args.num_skills])], axis=-1).copy()
                self.buffer.buffer_lists[idx].obs[0] = np.concatenate([obs, np.zeros([self.num_thread_per_env, self.num_agents[idx], self.all_args.num_skills])], axis=-1).copy()
                self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()
            elif self.all_args.skill_to_obs == "entity":
                self.buffer.buffer_lists[idx].share_obs[0] = np.concatenate([share_obs, np.zeros([self.num_thread_per_env, self.num_agents[idx], self.all_args.num_skills*n_entity])], axis=-1).copy()
                self.buffer.buffer_lists[idx].obs[0] = np.concatenate([obs, np.zeros([self.num_thread_per_env, self.num_agents[idx], self.all_args.num_skills*n_entity])], axis=-1).copy()
                self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()
            else:
                self.buffer.buffer_lists[idx].share_obs[0] = share_obs.copy()
                self.buffer.buffer_lists[idx].obs[0] = obs.copy()
                self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()

    def collect_data_preds(self, episode_length):
        rewards_episode = []
        dones_episode = []
        infos_episode = []
        available_actions_episode = []
        pi_probs_episode = []
        actions_episode = []
        idxs_eispode = []
        for step in range(episode_length):
            if self.all_args.use_latent_skills:
                # Sample actions
                values_list, actions_list, action_log_probs_list, pi_probs_list, \
                    rnn_states_list, rnn_states_comm_list, rnn_states_critic_list, latent_actor, latent_critic = self.collect_with_latent(step)
                
                # Obser reward and next obs
                obs_list, share_obs_list, rewards_list, dones_list, \
                    infos_list, available_actions_list, idxs_tuple = self.envs.step(actions_list)
                # concat obs/share_obs with past latents            
                ob_na_list = [np.shape(obs) for obs in obs_list]
                
                if self.all_args.skill_to_obs != "None":
                    latent_actor = [latent.reshape(ob_na[0], ob_na[1], -1) for latent, ob_na in zip(latent_actor, ob_na_list)]
                    latent_critic = [latent.reshape(ob_na[0], ob_na[1], -1) for latent, ob_na in zip(latent_critic, ob_na_list)]

                    obs_list = [np.concatenate([obs, latent_a], axis=-1) for obs, latent_a in zip(obs_list, latent_actor)]
                    # share_obs_list = [np.concatenate([share_obs, subtask_c], axis=-1) for share_obs, subtask_c in zip(share_obs_list, subtasks_actor)]
                    share_obs_list = [np.concatenate([share_obs, latent_c], axis=-1) for share_obs, latent_c in zip(share_obs_list, latent_critic)]                
            else:
                # Sample actions
                values_list, actions_list, action_log_probs_list, pi_probs_list, rnn_states_list, rnn_states_comm_list, rnn_states_critic_list = self.collect_without_latent(step)
                # Obser reward and next obs
                obs_list, share_obs_list, rewards_list, dones_list, \
                    infos_list, available_actions_list, idxs_tuple = self.envs.step(actions_list)
                # concat obs/share_obs with past latents            
                ob_na_list = [np.shape(obs) for obs in obs_list]
            
            for obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_comm, rnn_states_critic, idx in zip(
                    obs_list, share_obs_list, rewards_list, dones_list, infos_list, available_actions_list, \
                        values_list, actions_list, action_log_probs_list, rnn_states_list, rnn_states_comm_list, rnn_states_critic_list, idxs_tuple
                ):
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_comm, rnn_states_critic
                # insert data into buffer/enc_buffer
                self.insert(data, idx)
            # collect data
            rewards_episode.append(rewards_list)
            dones_episode.append(dones_list)
            infos_episode.append(infos_list)
            available_actions_episode.append(available_actions_list)
            pi_probs_episode.append(pi_probs_list)
            actions_episode.append(actions_list)
            idxs_eispode.append(idxs_tuple)
    
        # insert data
        data = actions_episode, pi_probs_episode, available_actions_episode, idxs_eispode
        self.insert_futures(data)
        return rewards_episode, infos_episode, dones_episode


    def collect_data(self, episode_length):
        rewards_episode = []
        dones_episode = []
        infos_episode = []
        for step in range(episode_length):
            if self.all_args.use_latent_skills:
                # Sample actions
                values_list, actions_list, action_log_probs_list, _, \
                    rnn_states_list, rnn_states_comm_list, rnn_states_critic_list, latent_actor, latent_critic = self.collect_with_latent(step)
                # Obser reward and next obs
                obs_list, share_obs_list, rewards_list, dones_list, \
                    infos_list, available_actions_list, idxs_tuple = self.envs.step(actions_list)
                # concat obs/share_obs with past latents            
                ob_na_list = [np.shape(obs) for obs in obs_list]

                if self.all_args.skill_to_obs != "None":
                    latent_actor = [latent.reshape(ob_na[0], ob_na[1], -1) for latent, ob_na in zip(latent_actor, ob_na_list)]
                    latent_critic = [latent.reshape(ob_na[0], ob_na[1], -1) for latent, ob_na in zip(latent_critic, ob_na_list)]
                    obs_list = [np.concatenate([obs, latent_a], axis=-1) for obs, latent_a in zip(obs_list, latent_actor)]
                    # share_obs_list = [np.concatenate([share_obs, subtask_c], axis=-1) for share_obs, subtask_c in zip(share_obs_list, subtasks_actor)]
                    share_obs_list = [np.concatenate([share_obs, latent_c], axis=-1) for share_obs, latent_c in zip(share_obs_list, latent_critic)]                
            else:
                # Sample actions
                values_list, actions_list, action_log_probs_list, _, rnn_states_list, rnn_states_comm_list, rnn_states_critic_list = self.collect_without_latent(step)
                # Obser reward and next obs
                obs_list, share_obs_list, rewards_list, dones_list, \
                    infos_list, available_actions_list, idxs_tuple = self.envs.step(actions_list)
                # concat obs/share_obs with past latents            
                ob_na_list = [np.shape(obs) for obs in obs_list]
            
            for obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_comm, rnn_states_critic, idx in zip(
                    obs_list, share_obs_list, rewards_list, dones_list, infos_list, available_actions_list, \
                        values_list, actions_list, action_log_probs_list, rnn_states_list, rnn_states_comm_list, rnn_states_critic_list, idxs_tuple
                ):
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_comm, rnn_states_critic
                # insert data into buffer/enc_buffer
                self.insert(data, idx)
            rewards_episode.append(rewards_list)
            dones_episode.append(dones_list)
            infos_episode.append(infos_list)
        return rewards_episode, infos_episode, dones_episode


    @torch.no_grad()
    def collect_without_latent(self, step):
        self.trainer.prep_rollout()
        # (threads*na, _size)
        cent_obs = [np.concatenate(self.buffer.buffer_lists[idx].share_obs[step]) for idx in range(self.num_multi_envs)]
        obs = [np.concatenate(self.buffer.buffer_lists[idx].obs[step]) for idx in range(self.num_multi_envs)]
        rnn_state = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states[step]) for idx in range(self.num_multi_envs)]
        rnn_state_comm = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_comm[step]) for idx in range(self.num_multi_envs)]
        rnn_state_critic = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_critic[step]) for idx in range(self.num_multi_envs)]
        masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[step]) for idx in range(self.num_multi_envs)]
        active_masks = [np.concatenate(self.buffer.buffer_lists[idx].active_masks[step]) for idx in range(self.num_multi_envs)]
        available_actions = [np.concatenate(self.buffer.buffer_lists[idx].available_actions[step]) for idx in range(self.num_multi_envs)]
        th_na_list = [cent_ob.shape[0] for cent_ob in cent_obs]

        value, action, action_log_prob, pi_prob, rnn_state, rnn_state_comm, rnn_state_critic, _, _, train_info, record_info = \
            self.trainer.policy.get_actions(
                cent_obs, obs, rnn_state, rnn_state_comm, rnn_state_critic, masks, active_masks, available_actions, 
                n_agents=self.num_agents, 
                n_enemies=self.num_enemies,
                n_entities=self.num_entities,
            )
        
        values = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(value, th_na_list, dim=0)]
        actions = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action, th_na_list, dim=0)]
        action_log_probs = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action_log_prob, th_na_list, dim=0)]
        pi_probs = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in pi_prob]

        rnn_states = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state]
        rnn_states_comm = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_comm]
        rnn_states_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_critic]

        return values, actions, action_log_probs, pi_probs, rnn_states, rnn_states_comm, rnn_states_critic
    
    
    
    @torch.no_grad()
    def collect_with_latent(self, step):
        self.trainer.prep_rollout()
        # (threads*na, _size)
        cent_obs = [np.concatenate(self.buffer.buffer_lists[idx].share_obs[step]) for idx in range(self.num_multi_envs)]
        obs = [np.concatenate(self.buffer.buffer_lists[idx].obs[step]) for idx in range(self.num_multi_envs)]
        rnn_state = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states[step]) for idx in range(self.num_multi_envs)]
        rnn_state_comm = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_comm[step]) for idx in range(self.num_multi_envs)]
        rnn_state_critic = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_critic[step]) for idx in range(self.num_multi_envs)]
        masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[step]) for idx in range(self.num_multi_envs)]
        active_masks = [np.concatenate(self.buffer.buffer_lists[idx].active_masks[step]) for idx in range(self.num_multi_envs)]
        available_actions = [np.concatenate(self.buffer.buffer_lists[idx].available_actions[step]) for idx in range(self.num_multi_envs)]
        th_na_list = [cent_ob.shape[0] for cent_ob in cent_obs]
                
        value, action, action_log_prob, pi_prob, rnn_state, rnn_state_comm, rnn_state_critic, latent_actor, latent_critic, _, _ = \
            self.trainer.policy.get_actions(
                cent_obs, obs, rnn_state, rnn_state_comm, rnn_state_critic, masks, active_masks, available_actions, 
                n_agents=self.num_agents, 
                n_enemies=self.num_enemies, 
                n_entities=self.num_entities,
            )
        
        values = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(value, th_na_list, dim=0)]
        actions = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action, th_na_list, dim=0)]
        pi_probs = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in pi_prob]
        action_log_probs = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action_log_prob, th_na_list, dim=0)]

        # rnn_state仍是tensor而不是若干个tensor组成的list
        rnn_states = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state]
        rnn_states_comm = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_comm]
        rnn_states_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_critic]
        latent_actor = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in latent_actor]
        latent_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in latent_critic]
    
        return values, actions, action_log_probs, pi_probs, rnn_states, rnn_states_comm, rnn_states_critic, latent_actor, latent_critic

    
    def insert_futures(self, data):
        actions_episode, pi_probs_episode, available_actions_episode, idxs_episode = data

        future_actions_episode = stack_over_steps_with_padding(actions_episode, self.n_future_steps, self.padded_values)
        future_pi_probs_episode = stack_over_steps_with_padding(pi_probs_episode, self.n_future_steps, self.padded_values)
        future_available_actions_episode = stack_over_steps_with_padding(available_actions_episode, self.n_future_steps, self.padded_values)

        step = 0
        for future_actions, future_pi, future_available_actions, idx_tuple in zip(future_actions_episode, future_pi_probs_episode, future_available_actions_episode, idxs_episode):
            for act, pi, available_acts, idx in zip(future_actions, future_pi, future_available_actions, idx_tuple):
                self.buffer.insert_futures(idx, step, act, pi, available_acts)
            step += 1

    def insert(self, data, idx):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_comm, rnn_states_critic  = data
        
        dones_env = np.all(dones, axis=1)
        n_agents = self.num_agents[idx]
        
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), n_agents, *self.buffer.buffer_lists[idx].rnn_states.shape[3:]), dtype=np.float32)
        rnn_states_comm[dones_env == True] = np.zeros(((dones_env == True).sum(), n_agents, *self.buffer.buffer_lists[idx].rnn_states_comm.shape[3:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), n_agents, *self.buffer.buffer_lists[idx].rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.num_thread_per_env, n_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), n_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.num_thread_per_env, n_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), n_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(n_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(idx, share_obs, obs, rnn_states, rnn_states_comm, rnn_states_critic, actions, action_log_probs, values, rewards, masks, 
                           bad_masks=bad_masks, active_masks=active_masks, available_actions=available_actions)
    
    '''
    Redefine some methods of base_runner.Runner
    '''
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        cent_obs = [np.concatenate(self.buffer.buffer_lists[idx].share_obs[-1]) for idx in range(self.num_multi_envs)]        
        rnn_state_critic = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_critic[-1]) for idx in range(self.num_multi_envs)]
        masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[-1]) for idx in range(self.num_multi_envs)]
        active_masks = [np.concatenate(self.buffer.buffer_lists[idx].active_masks[-1]) for idx in range(self.num_multi_envs)]

        th_na_list = [cent_ob.shape[0] for cent_ob in cent_obs]
        next_value = self.trainer.policy.get_values(
            cent_obs, rnn_state_critic, masks, active_masks,
            n_agents=self.num_agents,
            n_enemies=self.num_enemies,
            n_entities=self.num_entities)
        next_values = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(next_value, th_na_list, dim=0)]
                
        for idx, next_value in zip(range(self.num_multi_envs), next_values):
            self.buffer.compute_returns(idx, next_value, self.trainer.value_normalizer)
    
    
    def save_trajectory(self, total_num_steps, entity_obs_lists, skill_lists, done_lists, win_lists, actions_lists, obs_dot_lists=None, skill_dot_lists=None, comm_skill_dot_lists=None, comm_weight_lists=None):
        save_data = {
            "entity_obs": [],
            "mean_entity_obs": [],
            "skill": [],
            "done": [],
            "win_flag": [],
            "actions": [],
            "obs_dot": [],
            "skill_dot": [],
            "comm_skill_dot": [],
            "comm_weight": []
        }
        for idx, entity_ob in enumerate(entity_obs_lists):
            entity_ob = np.squeeze(entity_ob)   # num_tasks, time_step, n_process, num_agents, num_entity, feat_dim
            save_data["entity_obs"].append(entity_ob)            
        for idx, entity_ob in enumerate(entity_obs_lists):
            entity_ob = np.squeeze(entity_ob).mean(-2)  # num_tasks, time_step, n_process, num_agents, feat_dim
            save_data["mean_entity_obs"].append(entity_ob)
        if skill_lists and len(skill_lists) > 0:
            for idx, skill in enumerate(skill_lists):
                skill = np.squeeze(skill)
                save_data["skill"].append(skill)
        for idx, done in enumerate(done_lists):
            save_data["done"].append(np.squeeze(done))
        for idx, win_flag in enumerate(win_lists):
            save_data["win_flag"].append(np.squeeze(win_flag))
        for idx, eval_action in enumerate(actions_lists):
            eval_action = np.squeeze(eval_action)
            save_data["actions"].append(eval_action)
        if obs_dot_lists and len(obs_dot_lists) > 0:
            for idx, eval_attention in enumerate(obs_dot_lists):
                save_data["obs_dot"].append(np.squeeze(eval_attention))
        if skill_dot_lists and len(skill_dot_lists) > 0:
            for idx, eval_attention in enumerate(skill_dot_lists):
                save_data["skill_dot"].append(np.squeeze(eval_attention))
        if comm_skill_dot_lists and len(comm_skill_dot_lists) > 0:
            for idx, eval_attention in enumerate(comm_skill_dot_lists):
                save_data["comm_skill_dot"].append(np.squeeze(eval_attention))
        if comm_weight_lists and len(comm_weight_lists) > 0:
            for idx, eval_attention in enumerate(comm_weight_lists):
                save_data["comm_weight"].append(np.squeeze(eval_attention))

        flat_data = {}
        for k, arr_list in save_data.items():
            for i, arr in enumerate(arr_list):
                flat_data[f"{k}_{i}"] = arr
                # print(f"{k}_{i}", np.shape(arr))
        np.savez_compressed(os.path.join(self.trajectory_dir, f"trajectory_{total_num_steps}.npz"), **flat_data)
        

    def log_train(self, train_infos, total_num_steps):
        """
        Log train info.
        :param train_infos: (dict) information about train envs.
        :param total_num_steps: (int) total number of training env steps.
        """
        average_step_rewrads = [np.mean(self.buffer.buffer_lists[idx].rewards) for idx in range(self.num_multi_envs)]
        train_infos["train_step_rewards"] = average_step_rewrads
        for k, v in train_infos.items():
            if self.use_wandb:
                if isinstance(v, list):
                    for idx, env_name in enumerate(self.multi_envs):
                        wandb.log({"{}_{}".format(k, env_name): v[idx]}, total_num_steps)
                elif isinstance(v, dict):
                    for idx, (key, value) in enumerate(v.items()):
                        wandb.log({"{}_{}".format(k, key): value}, total_num_steps)
                else:
                    wandb.log({k: v}, step=total_num_steps)
            else:
                if isinstance(v, list):
                    for idx, env_name in enumerate(self.multi_envs):
                        self.writter.add_scalars(k, {"{}_{}".format(k, env_name): v[idx]}, total_num_steps)
                elif isinstance(v, dict):
                    for idx, (key, value) in enumerate(v.items()):
                        self.writter.add_scalars(k, {"{}_{}".format(k, key): value}, total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, total_num_steps)
    

    def restore_only_skill_encoder(self):
        '''
        just restore skill when transfer from a pearl mappo model
        '''
        policy_skill_encoder_state_dict = torch.load(str(self.model_dir) + "/skill.pt")
        self.policy.skill_encoder.load_state_dict(policy_skill_encoder_state_dict)      

