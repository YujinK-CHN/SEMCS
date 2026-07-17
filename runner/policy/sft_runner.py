"""
Sequential Fine-Tuning (SFT) baseline runner.

Trains on tasks one at a time (task 1 → task 2 → ... → task N), carrying
weights forward between phases.  Evaluates on ALL tasks at each eval
interval so results are directly comparable to multi-task methods.

Reuses flat-obs MAPPO policy (matching SESiL's actor architecture minus
encoder population) and MCS trainer.  The only difference from MAPPO
is the training schedule: each phase feeds only the active task's buffer
to the trainer.
"""
import time
from functools import reduce
import numpy as np
import torch

from runner.policy.mappo_runner import mappoETERunner


class _SingleTaskBufferView:
    """Thin wrapper that exposes one entry of a MultiEnvBuffer as if it
    were a full multi-env buffer with num_multi_envs == 1."""

    def __init__(self, full_buffer, task_idx):
        self._full = full_buffer
        self._idx = task_idx
        self.buffer_lists = [full_buffer.buffer_lists[task_idx]]
        self.num_multi_envs = 1
        self.num_thread_per_env = full_buffer.num_thread_per_env

    def after_update(self):
        self.buffer_lists[0].after_update()

    # --- generator delegates (yield exactly one generator) -----------------
    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        yield self.buffer_lists[0].feed_forward_generator(advantages[0], num_mini_batch, mini_batch_size)

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        yield self.buffer_lists[0].naive_recurrent_generator(advantages[0], num_mini_batch)

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        yield self.buffer_lists[0].recurrent_generator(advantages[0], num_mini_batch, data_chunk_length)

    def recurrent_generator_with_future_actions(self, advantages, num_mini_batch, data_chunk_length):
        yield self.buffer_lists[0].recurrent_generator_with_future_actions(advantages[0], num_mini_batch, data_chunk_length)

    def recurrent_generator_with_agents(self, advantages, num_mini_batch, data_chunk_length):
        yield self.buffer_lists[0].recurrent_generator_with_agents(advantages[0], num_mini_batch, data_chunk_length)

    def recurrent_generator_with_agents_with_future_actions(self, advantages, num_mini_batch, data_chunk_length):
        yield self.buffer_lists[0].recurrent_generator_with_agents_with_future_actions(advantages[0], num_mini_batch, data_chunk_length)

    def compute_returns(self, idx, next_value, value_normalizer):
        assert idx == 0
        self.buffer_lists[0].compute_returns(next_value, value_normalizer)


class sftETERunner(mappoETERunner):
    """Sequential Fine-Tuning runner.

    Inherits all collect / eval / logging from mappoETERunner (flat-obs MAPPO).
    Overrides ``run()`` to train on one task at a time.
    """

    def run(self):
        if self.model_dir != "None":
            if self.all_args.only_evaluate or self.all_args.save_replay:
                self.evaluate4replay()
            return

        self.warmup()
        start = time.time()

        total_episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episodes_per_task = total_episodes // self.num_multi_envs

        start_episode = 0
        if getattr(self.all_args, 'resume', False):
            start_episode = self.restore_checkpoint()

        # bookkeeping (same as mcs_runner)
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

        global_episode = 0
        start_phase = start_episode // episodes_per_task
        start_ep_in_phase = start_episode % episodes_per_task

        for phase, active_task in enumerate(range(self.num_multi_envs)):
            if phase < start_phase:
                global_episode += episodes_per_task
                continue
            task_name = self.multi_envs[active_task]
            print(f"\n{'='*60}")
            print(f"  [SFT] Phase {phase+1}/{self.num_multi_envs}: training on task '{task_name}'")
            print(f"  Episodes {global_episode}..{global_episode + episodes_per_task - 1} / {total_episodes}")
            print(f"{'='*60}\n")

            # Swap trainer's view so it only trains on the active task
            orig_multi_envs = self.trainer.multi_envs
            orig_num_multi_envs = self.trainer.num_multi_envs
            orig_n_agents = self.trainer.n_agents_list
            orig_n_enemies = self.trainer.n_enemies_list
            orig_n_entities = self.trainer.n_entities_list

            self.trainer.multi_envs = [self.multi_envs[active_task]]
            self.trainer.num_multi_envs = 1
            self.trainer.n_agents_list = [self.num_agents[active_task]]
            self.trainer.n_enemies_list = [self.num_enemies[active_task]]
            self.trainer.n_entities_list = [self.num_entities[active_task]]

            phase_start = start_ep_in_phase if phase == start_phase else 0
            for ep_in_phase in range(phase_start, episodes_per_task):
                episode = global_episode + ep_in_phase

                if self.use_linear_lr_decay:
                    self.trainer.policy.lr_decay(episode, total_episodes)

                # collect data from ALL envs (env wrapper requires it)
                rewards_episode, infos_episode, dones_episode = self.collect_data(self.episode_length)

                # compute returns for ALL tasks (needed for value targets)
                self.compute()

                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

                # train on ACTIVE task only
                single_buf = _SingleTaskBufferView(self.buffer, active_task)
                self.trainer.prep_training()
                train_infos = self.trainer.train(single_buf, episode)
                single_buf.after_update()

                # after_update for non-active buffers too (reset pointers)
                for idx in range(self.num_multi_envs):
                    if idx != active_task:
                        self.buffer.buffer_lists[idx].after_update()

                if (episode % self.save_interval == 0 or episode == total_episodes - 1):
                    self.save(episode=episode)

                # --- record info (same as mcs_runner) ---
                for rewards_step, infos_step, dones_step in zip(rewards_episode, infos_episode, dones_episode):
                    for idx, (rewards_tuple, infos_tuple, dones_tuple) in enumerate(zip(rewards_step, infos_step, dones_step)):
                        dones_env = np.all(dones_tuple, axis=1)
                        reward_env = np.mean(rewards_tuple, axis=1).flatten()
                        one_episode_rewards[idx] += reward_env
                        for t in range(self.num_thread_per_env):
                            train_episode_steps[idx][t] += 1
                            if dones_env[t]:
                                done_episodes_rewards[idx].append(one_episode_rewards[idx][t])
                                if "AliceBob" in self.env_name:
                                    if infos_tuple[t][0]["battle_won"]:
                                        battles_won_num[idx] += 1
                                    battles_goals[idx] += infos_tuple[t][0]["goals"]
                                    battles_step[idx] += train_episode_steps[idx][t]
                                    battles_game_num[idx] += 1
                                train_episode_steps[idx][t] = 0
                                one_episode_rewards[idx][t] = 0

                # --- log ---
                if episode % self.log_interval == 0:
                    if len(done_episodes_rewards) > 0:
                        train_infos["train_episode_rewards"] = [np.mean(r) if r else 0.0 for r in done_episodes_rewards]
                        done_episodes_rewards = [[] for _ in range(self.num_multi_envs)]

                    if "AliceBob" in self.env_name:
                        train_win_rate = []
                        train_goals = []
                        train_battles_step = []
                        for b_won, b_goal, b_step, b_num in zip(battles_won_num, battles_goals, battles_step, battles_game_num):
                            if b_num > 0:
                                train_win_rate.append(b_won / b_num)
                                train_goals.append(b_goal / b_num)
                                train_battles_step.append(b_step / b_num)
                            else:
                                train_win_rate.append(0.0)
                                train_goals.append(0.0)
                                train_battles_step.append(0.0)
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

                    # restore trainer view for logging (log_train expects all tasks)
                    self.trainer.multi_envs = orig_multi_envs
                    self.trainer.num_multi_envs = orig_num_multi_envs
                    self.log_train(train_infos, total_num_steps)
                    self.trainer.multi_envs = [self.multi_envs[active_task]]
                    self.trainer.num_multi_envs = 1

                    end = time.time()
                    print("\n Env {}-{} Algo {} Exp {} updates {}/{} episodes, rewards {:.2f}, total num timesteps {}/{}, FPS {}, Time {:.2f} hour."
                          .format(self.all_args.env_name, self.all_args.seed, self.algorithm_name,
                                  self.experiment_name, episode, total_episodes,
                                  np.mean(train_infos["train_episode_rewards"]),
                                  total_num_steps, self.num_env_steps,
                                  int(total_num_steps / (end - start)),
                                  (end - start) / 3600))
                    print(f"  [SFT] Phase {phase+1}/{self.num_multi_envs}, training on: {task_name}\n")

                # eval on ALL tasks
                if episode % self.eval_interval == 0 and self.use_eval:
                    # restore trainer view for eval
                    self.trainer.multi_envs = orig_multi_envs
                    self.trainer.num_multi_envs = orig_num_multi_envs
                    self.trainer.n_agents_list = orig_n_agents
                    self.trainer.n_enemies_list = orig_n_enemies
                    self.trainer.n_entities_list = orig_n_entities

                    if self.n_eval_rollout_threads == 1 and "StarCraft" in self.env_name:
                        self.eval(total_num_steps, episode)
                    else:
                        self.eval_parallel(total_num_steps, episode)

                    # restore single-task view for training
                    self.trainer.multi_envs = [self.multi_envs[active_task]]
                    self.trainer.num_multi_envs = 1
                    self.trainer.n_agents_list = [self.num_agents[active_task]]
                    self.trainer.n_enemies_list = [self.num_enemies[active_task]]
                    self.trainer.n_entities_list = [self.num_entities[active_task]]

            # restore trainer state at end of phase
            self.trainer.multi_envs = orig_multi_envs
            self.trainer.num_multi_envs = orig_num_multi_envs
            self.trainer.n_agents_list = orig_n_agents
            self.trainer.n_enemies_list = orig_n_enemies
            self.trainer.n_entities_list = orig_n_entities

            global_episode += episodes_per_task
