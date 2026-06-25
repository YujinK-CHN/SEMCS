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

from base_policy.utils.util import _t2n, stack_over_steps_with_padding
from base_policy.utils.multi_envs_shared_buffer import MultiEnvSharedReplayBufferComm
from base_policy.algorithms.sesil.sesil_policy import sesilPolicy as Policy
from base_policy.algorithms.sesil.sesil_trainer import sesilTrainer as Trainer


class sesilETERunner(Runner):
    def __init__(self, config):
        super(sesilETERunner, self).__init__(config)

        self.use_sparse_reward = self.all_args.use_sparse_reward
        self.multi_envs = self.envs.multi_envs
        self.eval_multi_envs = self.eval_envs.multi_envs
        self.num_multi_envs = len(self.multi_envs)
        self.num_thread_per_env = self.envs.num_thread_per_env
        self.num_eval_thread_per_env = self.eval_envs.num_thread_per_env
        self.obs_space_list = self.envs.observation_space
        self.cent_obs_space_list = self.envs.share_observation_space
        self.act_space_list = self.envs.action_space
        self.quo = self.all_args.eval_episodes // (self.num_multi_envs * self.num_eval_thread_per_env)

        self.eval_deterministic = self.all_args.eval_deterministic

        self.evo_interval = self.all_args.evo_interval
        self.num_encoders = self.all_args.num_encoders

        self.policy = Policy(self.all_args,
                             self.multi_envs,
                             self.num_thread_per_env,
                             self.envs.observation_space,
                             self.share_observation_space,
                             self.envs.action_space,
                             device=self.device)

        self.trainer = Trainer(self.all_args, self.policy, self.num_agents, self.num_enemies, self.num_entities, device=self.device)

        if self.model_dir != "None":
            self.restore()

        self.buffer = MultiEnvSharedReplayBufferComm(
            self.all_args, self.num_agents, self.num_entities,
            self.obs_space_list, self.cent_obs_space_list, self.act_space_list,
            self.num_thread_per_env
        )

    def run(self):
        if self.model_dir != "None":
            if self.all_args.only_evaluate or self.all_args.save_replay:
                self.evaluate4replay()
            return

        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        done_episodes_rewards = [[] for _ in range(self.num_multi_envs)]
        one_episode_rewards = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]
        train_episode_steps = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]

        battles_game_num = [0 for _ in range(self.num_multi_envs)]
        battles_won_num = [0 for _ in range(self.num_multi_envs)]

        if "AliceBob" in self.env_name:
            battles_goals = [0 for _ in range(self.num_multi_envs)]
            battles_step = [0 for _ in range(self.num_multi_envs)]
        if "StarCraft" in self.env_name:
            last_battles_game = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]
            last_battles_won = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            rewards_episode, infos_episode, dones_episode = self.collect_data(self.episode_length)
            self.compute()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            train_infos = self.train(episode)

            # Evolutionary step
            if self.evo_interval > 0 and episode > 0 and episode % self.evo_interval == 0:
                obs_batch = [torch.tensor(
                    np.concatenate(self.buffer.buffer_lists[idx].obs[-1]),
                    dtype=torch.float32, device=self.device
                ) for idx in range(self.num_multi_envs)]

                from base_policy.utils.entity_util import encode_entity
                entity_ob_batch, _, _ = encode_entity(self.all_args, obs_batch, self.num_agents, self.num_entities, self.all_args.actor_feat_dim)
                evo_info = self.trainer.evolutionary_step(entity_ob_batch)

                train_infos['Evolution/pairs_formed'] = evo_info.get("pairs", 0)
                train_infos['Evolution/loners_mutated'] = evo_info.get("loners", 0)

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # Record information
            for rewards_step, infos_step, dones_step in zip(rewards_episode, infos_episode, dones_episode):
                for idx, (rewards_tuple, infos_tuple, dones_tuple) in enumerate(zip(rewards_step, infos_step, dones_step)):
                    dones_env = np.all(dones_tuple, axis=1)
                    reward_env = np.mean(rewards_tuple, axis=1).flatten()
                    one_episode_rewards[idx] += reward_env

                    for t in range(self.num_thread_per_env):
                        train_episode_steps[idx][t] += 1
                        if dones_env[t]:
                            done_episodes_rewards[idx].append(one_episode_rewards[idx][t])

                            # Record fitness for evolutionary step
                            for agent_i in range(self.num_agents[idx]):
                                enc_id = self.policy.actor.encoder_population.get_encoder_id(agent_i)
                                self.policy.actor.encoder_population.record_fitness(enc_id, idx, one_episode_rewards[idx][t])

                            if "AliceBob" in self.env_name:
                                if infos_tuple[t][0]["battle_won"]:
                                    battles_won_num[idx] += 1
                                battles_goals[idx] += infos_tuple[t][0]["goals"]
                                battles_step[idx] += train_episode_steps[idx][t]
                                battles_game_num[idx] += 1

                            train_episode_steps[idx][t] = 0
                            one_episode_rewards[idx][t] = 0

            if episode % self.log_interval == 0:
                if len(done_episodes_rewards) > 0:
                    train_infos["train_episode_rewards"] = [np.mean(episode_reward) if episode_reward else 0.0 for episode_reward in done_episodes_rewards]
                    done_episodes_rewards = [[] for _ in range(self.num_multi_envs)]

                if "AliceBob" in self.env_name:
                    train_win_rate = []
                    train_goals = []
                    train_battles_step = []
                    for b_won, b_goal, b_step, b_num in zip(battles_won_num, battles_goals, battles_step, battles_game_num):
                        train_win_rate.append(b_won / b_num if b_num > 0 else 0.0)
                        train_goals.append(b_goal / b_num if b_num > 0 else 0.0)
                        train_battles_step.append(b_step / b_num if b_num > 0 else 0.0)
                    train_infos["train_win_rate"] = train_win_rate
                    train_infos["train_goals"] = train_goals
                    train_infos["train_battles_step"] = train_battles_step
                    battles_game_num = [0 for _ in range(self.num_multi_envs)]
                    battles_won_num = [0 for _ in range(self.num_multi_envs)]
                    battles_goals = [0 for _ in range(self.num_multi_envs)]
                    battles_step = [0 for _ in range(self.num_multi_envs)]

                if "StarCraft" in self.env_name:
                    train_incre_win_rate = []
                    train_win_rate = []
                    for idx, infos in enumerate(infos_episode[-1]):
                        battles_won = []
                        battles_game = []
                        incre_battles_won = []
                        incre_battles_game = []
                        for i, info in enumerate(infos):
                            if 'battles_won' in info[0].keys():
                                battles_won.append(info[0]['battles_won'])
                                incre_battles_won.append(info[0]['battles_won'] - last_battles_won[idx][i])
                            if 'battles_game' in info[0].keys():
                                battles_game.append(info[0]['battles_game'])
                                incre_battles_game.append(info[0]['battles_game'] - last_battles_game[idx][i])
                        train_incre_win_rate.append(np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(incre_battles_game) > 0 else 0.0)
                        train_win_rate.append(np.sum(battles_won) / np.sum(battles_game) if np.sum(battles_game) > 0 else 0.0)
                        last_battles_game[idx] = battles_game
                        last_battles_won[idx] = battles_won

                    train_infos['train_incre_win_rate'] = train_incre_win_rate
                    train_infos['train_win_rate'] = train_win_rate
                    train_dead_ratio = []
                    for idx in range(self.num_multi_envs):
                        train_dead_ratio.append(1 - self.buffer.buffer_lists[idx].active_masks.sum() / reduce(lambda x, y: x * y, list(self.buffer.buffer_lists[idx].active_masks.shape)))
                    train_infos['train_dead_ratio'] = train_dead_ratio

                self.log_train(train_infos, total_num_steps)

                end = time.time()
                print("\n Env {}-{} Algo {} Exp {} updates {}/{} episodes, rewards {:.2f}, total num timesteps {}/{}, FPS {}, Time {:.2f} hour.\n"
                      .format(self.all_args.env_name, self.all_args.seed, self.algorithm_name, self.experiment_name,
                              episode, episodes, np.mean(train_infos["train_episode_rewards"]),
                              total_num_steps, self.num_env_steps,
                              int(total_num_steps / (end - start)),
                              (end - start) / 3600))

            if episode % self.eval_interval == 0 and self.use_eval:
                if self.n_eval_rollout_threads == 1 and "StarCraft" in self.env_name:
                    self.eval(total_num_steps, episode)
                else:
                    self.eval_parallel(total_num_steps, episode)

    @torch.no_grad()
    def eval(self, total_num_steps, episode):
        assert self.num_eval_thread_per_env == 1
        eval_obs_list, eval_share_obs_list, eval_available_actions_list, idxs_tuple = self.eval_envs.reset()
        eval_episode_rewards = [[] for _ in idxs_tuple]
        one_episode_rewards = [[] for _ in idxs_tuple]
        eval_episode = [0 for _ in idxs_tuple]
        eval_dones_all = [False for _ in idxs_tuple]
        eval_battles_won = [0 for _ in idxs_tuple]

        eval_rnn_states_list = []
        eval_rnn_states_comm_list = []
        eval_masks_lists = []
        eval_active_masks_lists = []
        for idx in idxs_tuple:
            eval_rnn_states_list.append(np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states.shape[3:]), dtype=np.float32))
            eval_rnn_states_comm_list.append(np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states_comm.shape[3:]), dtype=np.float32))
            eval_masks_lists.append(np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32))
            eval_active_masks_lists.append(np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32))

        eval_skills = None
        save_entity_obs_lists = [[] for _ in idxs_tuple]
        save_skill_lists = [[] for _ in idxs_tuple]
        save_done_lists = [[] for _ in idxs_tuple]
        save_win_lists = [[] for _ in idxs_tuple]
        save_actions_lists = [[] for _ in idxs_tuple]

        while True:
            self.trainer.prep_rollout()
            eval_obs_list = [np.concatenate(eval_obs_list[idx]) for idx in idxs_tuple]

            if self.all_args.use_latent_skills and self.all_args.skill_to_obs != "None":
                if eval_skills is None:
                    if self.all_args.skill_to_obs == "merge":
                        eval_skills = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_skills, dtype=torch.float32) for eval_obs in eval_obs_list]
                    elif self.all_args.skill_to_obs == "entity":
                        eval_skills = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_skills * n_en, dtype=torch.float32) for eval_obs, n_en in zip(eval_obs_list, self.num_entities)]
                eval_skill_list = [_t2n(item) for item in eval_skills]
                eval_skill_list = [np.reshape(skill, (np.shape(eval_obs)[0], -1)) for eval_obs, skill in zip(eval_obs_list, eval_skill_list)]
                eval_obs_list = [np.concatenate([eval_obs, eval_skill], axis=-1) for eval_obs, eval_skill in zip(eval_obs_list, eval_skill_list)]
                eval_skill_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_skills]

            eval_rnn_states_list = [np.concatenate(eval_rnn_states_list[idx]) for idx in idxs_tuple]
            eval_rnn_states_comm_list = [np.concatenate(eval_rnn_states_comm_list[idx]) for idx in idxs_tuple]
            eval_masks_lists = [np.concatenate(eval_masks_lists[idx]) for idx in idxs_tuple]
            eval_active_masks_lists = [np.concatenate(eval_active_masks_lists[idx]) for idx in idxs_tuple]
            eval_available_actions_list = [np.concatenate(eval_available_actions_list[idx]) for idx in idxs_tuple]

            eval_actions, eval_rnn_states, eval_rnn_states_comm, eval_entity_obs, eval_skills, record_info = \
                self.trainer.policy.act(
                    eval_obs_list, eval_rnn_states_list, eval_rnn_states_comm_list, eval_masks_lists, eval_active_masks_lists, eval_available_actions_list,
                    deterministic=self.eval_deterministic, n_agents=self.eval_num_agents, n_enemies=self.eval_num_enemies, n_entities=self.eval_num_entities)

            eval_entity_obs_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_entity_obs]
            th_na_list = [eval_ob.shape[0] for eval_ob in eval_obs_list]
            eval_actions_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in torch.split(eval_actions, th_na_list, dim=0)]
            eval_rnn_states_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_rnn_states]
            eval_rnn_states_comm_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_rnn_states_comm]

            eval_obs_list, eval_share_obs_list, eval_rewards_list, eval_dones_list, eval_infos_list, eval_available_actions_list, idxs_tuple = self.eval_envs.step(eval_actions_list)
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
                eval_active_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                eval_active_masks[eval_dones_env == True] = np.ones(((eval_dones_env == True).sum(), self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_active_masks_lists[idx] = eval_active_masks

                for eval_i in range(self.num_eval_thread_per_env):
                    if eval_dones_env[eval_i]:
                        eval_episode[idx] += 1
                        eval_episode_rewards[idx].append(np.sum(one_episode_rewards[idx], axis=0))
                        one_episode_rewards[idx] = []
                        if "StarCraft" in self.env_name:
                            if eval_infos[eval_i][0].get("won", False):
                                eval_battles_won[idx] += 1
                                eval_win_flag[idx] = np.array([True])

            for idx in idxs_tuple:
                if eval_episode[idx] >= self.all_args.eval_episodes and eval_dones_all[idx] == False:
                    eval_dones_all[idx] = True
                    eval_env_infos = {'eval_episode_rewards_{}'.format(self.eval_multi_envs[idx]): np.mean(eval_episode_rewards[idx])}
                    if "StarCraft" in self.env_name:
                        eval_env_infos["eval_win_rate_{}".format(self.eval_multi_envs[idx])] = eval_battles_won[idx] / eval_episode[idx]
                    self.log_eval(eval_env_infos, total_num_steps)

            if np.all(eval_dones_all):
                break

    @torch.no_grad()
    def eval_parallel(self, total_num_steps, episode):
        eval_obs_list, eval_share_obs_list, eval_available_actions_list, idxs_tuple = self.eval_envs.reset()
        eval_episode_rewards = [[0] * self.num_eval_thread_per_env for _ in idxs_tuple]
        one_episode_rewards = [[] for _ in idxs_tuple]
        one_episode_steps = [[0] * self.num_eval_thread_per_env for _ in idxs_tuple]
        eval_battles_won = [0 for _ in idxs_tuple]
        recorded_envs = [False for _ in idxs_tuple]

        done_episodes_per_thread = np.zeros((len(idxs_tuple), self.num_eval_thread_per_env), dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + self.quo
        unfinished_thread = done_episodes_per_thread != eval_episodes_per_thread

        if "AliceBob" in self.env_name or "Football" in self.env_name:
            eval_battles_goals = [0 for _ in idxs_tuple]
            eval_battles_step = [0 for _ in idxs_tuple]

        eval_rnn_states_list = []
        eval_rnn_states_comm_list = []
        eval_masks_lists = []
        eval_active_masks_lists = []
        for idx in idxs_tuple:
            eval_rnn_states_list.append(np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states.shape[3:]), dtype=np.float32))
            eval_rnn_states_comm_list.append(np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states_comm.shape[3:]), dtype=np.float32))
            eval_masks_lists.append(np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32))
            eval_active_masks_lists.append(np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32))

        eval_skills = None

        while True:
            self.trainer.prep_rollout()
            eval_obs_list = [np.concatenate(eval_obs_list[idx]) for idx in idxs_tuple]

            if self.all_args.use_latent_skills and self.all_args.skill_to_obs != "None":
                if eval_skills is None:
                    if self.all_args.skill_to_obs == "merge":
                        eval_skills = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_skills, dtype=torch.float32) for eval_obs in eval_obs_list]
                    elif self.all_args.skill_to_obs == "entity":
                        eval_skills = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_skills * n_en, dtype=torch.float32) for eval_obs, n_en in zip(eval_obs_list, self.num_entities)]
                eval_skill_list = [_t2n(item) for item in eval_skills]
                eval_skill_list = [np.reshape(skill, (np.shape(eval_obs)[0], -1)) for eval_obs, skill in zip(eval_obs_list, eval_skill_list)]
                eval_obs_list = [np.concatenate([eval_obs, eval_skill], axis=-1) for eval_obs, eval_skill in zip(eval_obs_list, eval_skill_list)]
                eval_skill_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_skills]

            eval_rnn_states_list = [np.concatenate(eval_rnn_states_list[idx]) for idx in idxs_tuple]
            eval_rnn_states_comm_list = [np.concatenate(eval_rnn_states_comm_list[idx]) for idx in idxs_tuple]
            eval_masks_lists = [np.concatenate(eval_masks_lists[idx]) for idx in idxs_tuple]
            eval_active_masks_lists = [np.concatenate(eval_active_masks_lists[idx]) for idx in idxs_tuple]
            eval_available_actions_list = [np.concatenate(eval_available_actions_list[idx]) for idx in idxs_tuple]

            eval_actions, eval_rnn_states, eval_rnn_states_comm, eval_entity_obs, eval_skills, record_info = \
                self.trainer.policy.act(
                    eval_obs_list, eval_rnn_states_list, eval_rnn_states_comm_list, eval_masks_lists, eval_active_masks_lists, eval_available_actions_list,
                    deterministic=self.eval_deterministic, n_agents=self.eval_num_agents, n_enemies=self.eval_num_enemies, n_entities=self.eval_num_entities)

            th_na_list = [eval_ob.shape[0] for eval_ob in eval_obs_list]
            eval_actions_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in torch.split(eval_actions, th_na_list, dim=0)]
            eval_rnn_states_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_rnn_states]
            eval_rnn_states_comm_list = [np.array(np.split(_t2n(item), self.num_eval_thread_per_env)) for item in eval_rnn_states_comm]

            eval_obs_list, eval_share_obs_list, eval_rewards_list, eval_dones_list, eval_infos_list, eval_available_actions_list, idxs_tuple = self.eval_envs.step(eval_actions_list)

            for eval_rewards, eval_dones, eval_infos, idx in zip(eval_rewards_list, eval_dones_list, eval_infos_list, idxs_tuple):
                eval_reward_env = np.mean(eval_rewards, axis=1)
                one_episode_rewards[idx].append(eval_reward_env)
                eval_dones_env = np.all(eval_dones, axis=1)
                eval_rnn_states_list[idx][eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states.shape[3:]), dtype=np.float32)
                eval_rnn_states_comm_list[idx][eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states_comm.shape[3:]), dtype=np.float32)

                eval_masks = np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_masks_lists[idx] = eval_masks

                eval_active_masks = np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_active_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                eval_active_masks[eval_dones_env == True] = np.ones(((eval_dones_env == True).sum(), self.eval_num_agents[idx], 1), dtype=np.float32)
                eval_active_masks_lists[idx] = eval_active_masks

                for eval_i in range(self.num_eval_thread_per_env):
                    one_episode_steps[idx][eval_i] += 1
                    if unfinished_thread[idx][eval_i] and eval_dones_env[eval_i]:
                        done_episodes_per_thread[idx][eval_i] += 1
                        per_thread_one_episode_rewards = np.array(one_episode_rewards[idx])[:, eval_i]
                        eval_episode_rewards[idx][eval_i] += np.sum(per_thread_one_episode_rewards)
                        for ep in range(len(per_thread_one_episode_rewards)):
                            one_episode_rewards[idx][ep][eval_i] = 0.0
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

            unfinished_thread = done_episodes_per_thread != eval_episodes_per_thread

            for idx in idxs_tuple:
                if np.all(done_episodes_per_thread[idx] == eval_episodes_per_thread[idx]) and not recorded_envs[idx]:
                    recorded_envs[idx] = True
                    eval_episode = np.sum(eval_episodes_per_thread[idx])
                    eval_env_infos = {'eval_episode_rewards_{}'.format(self.eval_multi_envs[idx]): np.mean(eval_episode_rewards[idx])}
                    if "AliceBob" in self.env_name or "Football" in self.env_name:
                        eval_env_infos["eval_win_rate_{}".format(self.eval_multi_envs[idx])] = eval_battles_won[idx] / eval_episode
                        eval_env_infos["eval_goals_{}".format(self.eval_multi_envs[idx])] = eval_battles_goals[idx] / eval_episode
                        eval_env_infos["eval_steps_{}".format(self.eval_multi_envs[idx])] = eval_battles_step[idx] / eval_episode
                    self.log_eval(eval_env_infos, total_num_steps)

            if np.all(done_episodes_per_thread == eval_episodes_per_thread):
                break

    def warmup(self):
        obs_list, share_obs_list, available_actions_list, idxs_tuple = self.envs.reset()
        for obs, share_obs, available_actions, n_entity, idx in zip(obs_list, share_obs_list, available_actions_list, self.num_entities, idxs_tuple):
            if not self.use_centralized_V:
                share_obs = obs
            if self.all_args.skill_to_obs == "merge":
                self.buffer.buffer_lists[idx].share_obs[0] = np.concatenate([share_obs, np.zeros([self.num_thread_per_env, self.num_agents[idx], self.all_args.num_skills])], axis=-1).copy()
                self.buffer.buffer_lists[idx].obs[0] = np.concatenate([obs, np.zeros([self.num_thread_per_env, self.num_agents[idx], self.all_args.num_skills])], axis=-1).copy()
                self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()
            elif self.all_args.skill_to_obs == "entity":
                self.buffer.buffer_lists[idx].share_obs[0] = np.concatenate([share_obs, np.zeros([self.num_thread_per_env, self.num_agents[idx], self.all_args.num_skills * n_entity])], axis=-1).copy()
                self.buffer.buffer_lists[idx].obs[0] = np.concatenate([obs, np.zeros([self.num_thread_per_env, self.num_agents[idx], self.all_args.num_skills * n_entity])], axis=-1).copy()
                self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()
            else:
                self.buffer.buffer_lists[idx].share_obs[0] = share_obs.copy()
                self.buffer.buffer_lists[idx].obs[0] = obs.copy()
                self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()

    def collect_data(self, episode_length):
        rewards_episode = []
        dones_episode = []
        infos_episode = []
        for step in range(episode_length):
            if self.all_args.use_latent_skills:
                values_list, actions_list, action_log_probs_list, _, \
                    rnn_states_list, rnn_states_comm_list, rnn_states_critic_list, latent_actor, latent_critic = self.collect_with_latent(step)
                obs_list, share_obs_list, rewards_list, dones_list, \
                    infos_list, available_actions_list, idxs_tuple = self.envs.step(actions_list)
                ob_na_list = [np.shape(obs) for obs in obs_list]

                if self.all_args.skill_to_obs != "None":
                    latent_actor = [latent.reshape(ob_na[0], ob_na[1], -1) for latent, ob_na in zip(latent_actor, ob_na_list)]
                    latent_critic = [latent.reshape(ob_na[0], ob_na[1], -1) for latent, ob_na in zip(latent_critic, ob_na_list)]
                    obs_list = [np.concatenate([obs, latent_a], axis=-1) for obs, latent_a in zip(obs_list, latent_actor)]
                    share_obs_list = [np.concatenate([share_obs, latent_c], axis=-1) for share_obs, latent_c in zip(share_obs_list, latent_critic)]
            else:
                values_list, actions_list, action_log_probs_list, _, rnn_states_list, rnn_states_comm_list, rnn_states_critic_list = self.collect_without_latent(step)
                obs_list, share_obs_list, rewards_list, dones_list, \
                    infos_list, available_actions_list, idxs_tuple = self.envs.step(actions_list)

            for obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_comm, rnn_states_critic, idx in zip(
                    obs_list, share_obs_list, rewards_list, dones_list, infos_list, available_actions_list,
                    values_list, actions_list, action_log_probs_list, rnn_states_list, rnn_states_comm_list, rnn_states_critic_list, idxs_tuple):
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, rnn_states, rnn_states_comm, rnn_states_critic
                self.insert(data, idx)
            rewards_episode.append(rewards_list)
            dones_episode.append(dones_list)
            infos_episode.append(infos_list)
        return rewards_episode, infos_episode, dones_episode

    @torch.no_grad()
    def collect_without_latent(self, step):
        self.trainer.prep_rollout()
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
                n_agents=self.num_agents, n_enemies=self.num_enemies, n_entities=self.num_entities)

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
                n_agents=self.num_agents, n_enemies=self.num_enemies, n_entities=self.num_entities)

        values = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(value, th_na_list, dim=0)]
        actions = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action, th_na_list, dim=0)]
        pi_probs = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in pi_prob]
        action_log_probs = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action_log_prob, th_na_list, dim=0)]
        rnn_states = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state]
        rnn_states_comm = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_comm]
        rnn_states_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_critic]
        latent_actor = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in latent_actor]
        latent_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in latent_critic]

        return values, actions, action_log_probs, pi_probs, rnn_states, rnn_states_comm, rnn_states_critic, latent_actor, latent_critic

    def insert(self, data, idx):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_comm, rnn_states_critic = data

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

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        cent_obs = [np.concatenate(self.buffer.buffer_lists[idx].share_obs[-1]) for idx in range(self.num_multi_envs)]
        rnn_state_critic = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_critic[-1]) for idx in range(self.num_multi_envs)]
        masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[-1]) for idx in range(self.num_multi_envs)]
        active_masks = [np.concatenate(self.buffer.buffer_lists[idx].active_masks[-1]) for idx in range(self.num_multi_envs)]

        th_na_list = [cent_ob.shape[0] for cent_ob in cent_obs]
        next_value = self.trainer.policy.get_values(
            cent_obs, rnn_state_critic, masks, active_masks,
            n_agents=self.num_agents, n_enemies=self.num_enemies, n_entities=self.num_entities)
        next_values = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(next_value, th_na_list, dim=0)]

        for idx, next_value in zip(range(self.num_multi_envs), next_values):
            self.buffer.compute_returns(idx, next_value, self.trainer.value_normalizer)

    def log_train(self, train_infos, total_num_steps):
        average_step_rewards = [np.mean(self.buffer.buffer_lists[idx].rewards) for idx in range(self.num_multi_envs)]
        train_infos["train_step_rewards"] = average_step_rewards
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
