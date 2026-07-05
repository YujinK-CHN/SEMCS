"""
SESiL runner — generational evolutionary training.

Gen 0: train M independent solvers (one per task) sequentially.
Gen 1+: evaluate all solvers on all tasks → SEMFO selection → merge pairs → offspring
         inherits both parents' tasks → train → repeat.
Population shrinks each generation until one solver remains or budget exhausted.

Each solver trains one at a time. All envs are stepped every timestep, but only
the solver's assigned tasks contribute data for training. Unassigned tasks get
random valid actions sampled from available_actions.
"""
import copy
import math
import time
import os
import numpy as np
import torch

from runner.policy.base_runner import Runner
from base_policy.utils.util import _t2n
from base_policy.utils.multi_envs_shared_buffer import MultiEnvSharedReplayBufferComm
from base_policy.algorithms.sesil.sesil_policy import sesilPolicy as Policy
from base_policy.algorithms.mcs.mcs_trainer import mcsTrainer as Trainer
from base_policy.algorithms.sesil.evolution import (
    build_mating_scores, bidirectional_selection, merge_actors
)


class FilteredBuffer:
    """Wraps a MultiEnvSharedReplayBufferComm exposing only selected task indices."""
    def __init__(self, full_buffer, task_ids):
        self.buffer_lists = [full_buffer.buffer_lists[i] for i in task_ids]
        self.num_multi_envs = len(task_ids)
        self.num_thread_per_env = full_buffer.num_thread_per_env

    def insert(self, idx, *args, **kwargs):
        self.buffer_lists[idx].insert(*args, **kwargs)

    def compute_returns(self, idx, next_value, value_normalizer=None):
        self.buffer_lists[idx].compute_returns(next_value, value_normalizer)

    def after_update(self):
        for buf in self.buffer_lists:
            buf.after_update()

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].feed_forward_generator(env_advantages, num_mini_batch, mini_batch_size)

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].naive_recurrent_generator(env_advantages, num_mini_batch)

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator(env_advantages, num_mini_batch, data_chunk_length)

    def recurrent_generator_with_future_actions(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator_with_future_actions(env_advantages, num_mini_batch, data_chunk_length)

    def recurrent_generator_with_agents(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator_with_agents(env_advantages, num_mini_batch, data_chunk_length)

    def recurrent_generator_with_agents_with_future_actions(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator_with_agents_with_future_actions(env_advantages, num_mini_batch, data_chunk_length)


class Solver:
    """One member of the population: a policy + its assigned task indices."""
    def __init__(self, policy, trainer, task_ids, all_multi_envs, all_num_agents, all_num_enemies, all_num_entities, device):
        self.policy = policy
        self.trainer = trainer
        self.task_ids = list(task_ids)
        self.device = device
        self._sync_trainer(all_multi_envs, all_num_agents, all_num_enemies, all_num_entities)

    def _sync_trainer(self, all_multi_envs, all_num_agents, all_num_enemies, all_num_entities):
        """Align trainer's task view to this solver's assigned tasks."""
        self.trainer.num_multi_envs = len(self.task_ids)
        self.trainer.multi_envs = [all_multi_envs[i] for i in self.task_ids]
        self.trainer.n_agents_list = [all_num_agents[i] for i in self.task_ids]
        self.trainer.n_enemies_list = [all_num_enemies[i] for i in self.task_ids]
        self.trainer.n_entities_list = [all_num_entities[i] for i in self.task_ids]


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

        self.eval_steps_interval = self.eval_interval * self.episode_length * self.num_multi_envs * self.num_thread_per_env
        self.next_eval_step = self.eval_steps_interval

        self.evo_gen0_fraction = self.all_args.evo_gen0_fraction
        self.evo_eval_episodes = self.all_args.evo_eval_episodes
        self.evo_threshold = self.all_args.evo_threshold
        self.evo_weight_extra = self.all_args.evo_weight_extra
        self.evo_weight_common = self.all_args.evo_weight_common

        # Create one solver per task
        self.solvers = []
        for task_idx in range(self.num_multi_envs):
            policy = Policy(self.all_args,
                            self.multi_envs,
                            self.num_thread_per_env,
                            self.envs.observation_space,
                            self.share_observation_space,
                            self.envs.action_space,
                            device=self.device)
            trainer = Trainer(self.all_args, policy, self.num_agents, self.num_enemies, self.num_entities, device=self.device)
            self.solvers.append(Solver(policy, trainer, [task_idx],
                                       self.multi_envs, self.num_agents, self.num_enemies, self.num_entities, self.device))

        # Use first solver as the "active" policy/trainer for base class compatibility
        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

        self.buffer = MultiEnvSharedReplayBufferComm(
            self.all_args, self.num_agents, self.num_entities,
            self.obs_space_list, self.cent_obs_space_list, self.act_space_list,
            self.num_thread_per_env
        )

    # ─── Budget allocation ──────────────────────────────────────

    def _steps_per_episode(self, num_tasks=None):
        """Steps consumed by one episode. num_tasks defaults to all tasks."""
        if num_tasks is None:
            num_tasks = self.num_multi_envs
        return self.episode_length * num_tasks * self.num_thread_per_env

    def _compute_gen_budget(self, generation, cumulative_steps):
        if generation == 0:
            return int(self.evo_gen0_fraction * self.num_env_steps)
        remaining = self.num_env_steps - cumulative_steps
        if remaining <= 0:
            return 0
        num_solvers = len(self.solvers)
        remaining_gens = max(1, math.ceil(math.log2(num_solvers))) + 1
        return max(1, int(remaining / remaining_gens))

    # ─── Main loop ──────────────────────────────────────────────

    def run(self):
        start = time.time()
        self.cumulative_steps = 0
        generation = 0

        # === Evolutionary phase: train, evaluate, merge until 1 solver remains ===
        while self.cumulative_steps < self.num_env_steps and len(self.solvers) > 1:
            gen_budget = self._compute_gen_budget(generation, self.cumulative_steps)

            print(f"\n{'='*60}")
            print(f"Generation {generation}: {len(self.solvers)} solvers, "
                  f"gen budget={gen_budget} steps")
            print(f"  Solvers: {[s.task_ids for s in self.solvers]}")

            self._train_all_solvers(gen_budget)

            fitness_matrix = self._evaluate_population(self.cumulative_steps)
            print(f"  Fitness matrix (solvers x tasks):\n{np.array2string(fitness_matrix, precision=3)}")

            self._evolve(fitness_matrix, self.cumulative_steps)

            generation += 1
            print(f"  Cumulative steps: {self.cumulative_steps}/{self.num_env_steps}, "
                  f"Time: {(time.time() - start) / 3600:.2f}h")

        # === Final phase: train the merged solver with all remaining budget ===
        if self.solvers and self.cumulative_steps < self.num_env_steps:
            remaining = self.num_env_steps - self.cumulative_steps
            print(f"\n{'='*60}")
            print(f"Final training: 1 solver on all tasks, remaining budget={remaining} steps")

            self._train_all_solvers(remaining)

            print(f"  Cumulative steps: {self.cumulative_steps}/{self.num_env_steps}, "
                  f"Time: {(time.time() - start) / 3600:.2f}h")

        print(f"\nSESiL finished. Final solver has tasks: {self.solvers[0].task_ids if self.solvers else 'none'}")

    # ─── Sequential training ─────────────────────────────────────

    def _train_all_solvers(self, gen_budget):
        """Train each solver sequentially. Each solver only steps its own task envs."""
        budget_per_solver = max(1, gen_budget // len(self.solvers))

        for si, solver in enumerate(self.solvers):
            self.policy = solver.policy
            self.trainer = solver.trainer
            self.trainer.policy = solver.policy
            filtered_buf = FilteredBuffer(self.buffer, solver.task_ids)
            n_tasks = len(solver.task_ids)
            spe = self._steps_per_episode(n_tasks)
            episodes_per_solver = max(1, budget_per_solver // spe)

            print(f"  Training solver {si} (tasks {solver.task_ids}) for {episodes_per_solver} episodes "
                  f"({budget_per_solver} steps budget, {spe} steps/episode)")
            for episode in range(episodes_per_solver):
                self._warmup_tasks(solver.task_ids)
                self._collect_episode(solver)
                self._compute_filtered(solver, filtered_buf)
                solver.trainer.prep_training()
                solver.trainer.train(filtered_buf, episode)
                filtered_buf.after_update()
                self.cumulative_steps += spe
                if (episode + 1) % max(1, episodes_per_solver // 5) == 0 or episode == episodes_per_solver - 1:
                    print(f"    Solver {si} episode {episode+1}/{episodes_per_solver}, "
                          f"cumulative steps: {self.cumulative_steps}")

                if self.cumulative_steps >= self.next_eval_step:
                    self._evaluate_population(self.cumulative_steps)
                    self.next_eval_step += self.eval_steps_interval

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

    @torch.no_grad()
    def _collect_episode(self, solver):
        """Collect one episode stepping only the solver's assigned task envs."""
        task_ids = solver.task_ids
        n_agents_s = [self.num_agents[i] for i in task_ids]
        n_enemies_s = [self.num_enemies[i] for i in task_ids]
        n_entities_s = [self.num_entities[i] for i in task_ids]
        remotes = [self.envs.remotes[i] for i in task_ids]

        for step in range(self.episode_length):
            solver.trainer.prep_rollout()

            cent_obs = [np.concatenate(self.buffer.buffer_lists[idx].share_obs[step]) for idx in task_ids]
            obs = [np.concatenate(self.buffer.buffer_lists[idx].obs[step]) for idx in task_ids]
            rnn_s = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states[step]) for idx in task_ids]
            rnn_s_comm = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_comm[step]) for idx in task_ids]
            rnn_s_critic = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_critic[step]) for idx in task_ids]
            masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[step]) for idx in task_ids]
            active_masks = [np.concatenate(self.buffer.buffer_lists[idx].active_masks[step]) for idx in task_ids]
            avail_actions = [np.concatenate(self.buffer.buffer_lists[idx].available_actions[step]) for idx in task_ids]
            th_na_list = [co.shape[0] for co in cent_obs]

            value, action, action_log_prob, _, rnn_s_out, rnn_s_comm_out, rnn_s_critic_out, _, _, _, _ = \
                solver.policy.get_actions(
                    cent_obs, obs, rnn_s, rnn_s_comm, rnn_s_critic, masks, active_masks, avail_actions,
                    n_agents=n_agents_s, n_enemies=n_enemies_s, n_entities=n_entities_s)

            vals = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(value, th_na_list, dim=0)]
            acts = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action, th_na_list, dim=0)]
            alps = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action_log_prob, th_na_list, dim=0)]
            rnns = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_s_out]
            rnns_c = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_s_comm_out]
            rnns_cr = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_s_critic_out]

            # Step only assigned task envs
            for local, (remote, gid) in enumerate(zip(remotes, task_ids)):
                remote.send(('step', acts[local], gid))

            for local, (remote, gid) in enumerate(zip(remotes, task_ids)):
                res = remote.recv()
                if isinstance(res[0], str) and res[0] == "exception":
                    raise RuntimeError(f"Env worker {gid} exception: {res[1][0]}\n{res[1][1]}")
                obs_r, share_obs_r, rewards, dones, infos, available_actions, idx = res
                data = (obs_r, share_obs_r, rewards, dones, infos, available_actions,
                        vals[local], acts[local], alps[local],
                        rnns[local], rnns_c[local], rnns_cr[local])
                self._insert(data, idx)

    # ─── Buffer helpers ─────────────────────────────────────────

    def _insert(self, data, idx):
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
    def _compute_filtered(self, solver, filtered_buf):
        """Compute returns for a solver's assigned tasks only."""
        solver.trainer.prep_rollout()
        task_ids = solver.task_ids
        n_agents_assigned = [self.num_agents[i] for i in task_ids]
        n_enemies_assigned = [self.num_enemies[i] for i in task_ids]
        n_entities_assigned = [self.num_entities[i] for i in task_ids]

        cent_obs = [np.concatenate(self.buffer.buffer_lists[idx].share_obs[-1]) for idx in task_ids]
        rnn_state_critic = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_critic[-1]) for idx in task_ids]
        masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[-1]) for idx in task_ids]
        active_masks = [np.concatenate(self.buffer.buffer_lists[idx].active_masks[-1]) for idx in task_ids]

        th_na_list = [cent_ob.shape[0] for cent_ob in cent_obs]
        next_value = solver.policy.get_values(
            cent_obs, rnn_state_critic, masks, active_masks,
            n_agents=n_agents_assigned, n_enemies=n_enemies_assigned, n_entities=n_entities_assigned)
        next_values = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(next_value, th_na_list, dim=0)]

        for local_idx, next_val in enumerate(next_values):
            filtered_buf.compute_returns(local_idx, next_val, solver.trainer.value_normalizer)

    def warmup(self):
        obs_list, share_obs_list, available_actions_list, idxs_tuple = self.envs.reset()
        for obs, share_obs, available_actions, idx in zip(obs_list, share_obs_list, available_actions_list, idxs_tuple):
            if not self.use_centralized_V:
                share_obs = obs
            self.buffer.buffer_lists[idx].share_obs[0] = share_obs.copy()
            self.buffer.buffer_lists[idx].obs[0] = obs.copy()
            self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()

    def _warmup_tasks(self, task_ids):
        """Reset only the specified task envs."""
        remotes = [self.envs.remotes[i] for i in task_ids]
        for remote, gid in zip(remotes, task_ids):
            remote.send(('reset', None, gid))
        for remote, gid in zip(remotes, task_ids):
            res = remote.recv()
            if isinstance(res[0], str) and res[0] == "exception":
                raise RuntimeError(f"Env worker {gid} reset exception: {res[1][0]}\n{res[1][1]}")
            obs, share_obs, available_actions, idx = res
            if not self.use_centralized_V:
                share_obs = obs
            self.buffer.buffer_lists[idx].share_obs[0] = share_obs.copy()
            self.buffer.buffer_lists[idx].obs[0] = obs.copy()
            self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()

    # ─── Evaluation ─────────────────────────────────────────────

    def _evaluate_population(self, total_num_steps):
        """
        Evaluate every solver on every task.
        Returns fitness_matrix: (num_solvers, num_tasks), higher = better.
        """
        M = len(self.solvers)
        K = self.num_multi_envs
        fitness_matrix = np.zeros((M, K))

        win_rate_matrix = np.zeros((M, K))

        for solver_idx, solver in enumerate(self.solvers):
            self.policy = solver.policy
            self.trainer = solver.trainer
            self.trainer.policy = solver.policy

            task_rewards, task_win_rates = self._eval_solver_on_all_tasks()

            for task_idx in range(K):
                fitness_matrix[solver_idx, task_idx] = task_rewards[task_idx]
                win_rate_matrix[solver_idx, task_idx] = task_win_rates[task_idx]

        # Log population-average per-task performance (fair comparison with other algorithms)
        for task_idx in range(K):
            avg_reward = np.mean(fitness_matrix[:, task_idx])
            avg_win_rate = np.mean(win_rate_matrix[:, task_idx])
            task_name = self.eval_multi_envs[task_idx]
            eval_infos = {f'eval_episode_rewards_{task_name}': avg_reward}
            if "StarCraft" in self.env_name:
                eval_infos[f'eval_win_rate_{task_name}'] = avg_win_rate
            elif "AliceBob" in self.env_name or "Football" in self.env_name:
                eval_infos[f'eval_win_rate_{task_name}'] = avg_reward
            self.log_eval(eval_infos, total_num_steps)

        return fitness_matrix

    @torch.no_grad()
    def _eval_solver_on_all_tasks(self):
        """Run the current self.policy on all eval tasks. Returns (per-task rewards, per-task win rates)."""
        eval_obs_list, eval_share_obs_list, eval_available_actions_list, idxs_tuple = self.eval_envs.reset()
        eval_episode_rewards = [[0] * self.num_eval_thread_per_env for _ in idxs_tuple]
        one_episode_rewards = [[] for _ in idxs_tuple]
        eval_battles_won = [0 for _ in idxs_tuple]
        recorded_envs = [False for _ in idxs_tuple]

        done_episodes_per_thread = np.zeros((len(idxs_tuple), self.num_eval_thread_per_env), dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + self.quo

        eval_rnn_states_list = []
        eval_rnn_states_comm_list = []
        eval_masks_lists = []
        eval_active_masks_lists = []
        for idx in idxs_tuple:
            eval_rnn_states_list.append(np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states.shape[3:]), dtype=np.float32))
            eval_rnn_states_comm_list.append(np.zeros((self.num_eval_thread_per_env, self.eval_num_agents[idx], *self.buffer.buffer_lists[0].rnn_states_comm.shape[3:]), dtype=np.float32))
            eval_masks_lists.append(np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32))
            eval_active_masks_lists.append(np.ones((self.num_eval_thread_per_env, self.eval_num_agents[idx], 1), dtype=np.float32))

        task_returns = [0.0] * len(idxs_tuple)
        task_win_rates = [0.0] * len(idxs_tuple)

        while True:
            self.trainer.prep_rollout()
            eval_obs_list_flat = [np.concatenate(eval_obs_list[idx]) for idx in idxs_tuple]
            eval_rnn_states_list = [np.concatenate(eval_rnn_states_list[idx]) for idx in idxs_tuple]
            eval_rnn_states_comm_list = [np.concatenate(eval_rnn_states_comm_list[idx]) for idx in idxs_tuple]
            eval_masks_lists = [np.concatenate(eval_masks_lists[idx]) for idx in idxs_tuple]
            eval_active_masks_lists = [np.concatenate(eval_active_masks_lists[idx]) for idx in idxs_tuple]
            eval_available_actions_list = [np.concatenate(eval_available_actions_list[idx]) for idx in idxs_tuple]

            eval_actions, eval_rnn_states, eval_rnn_states_comm, _, _, _ = \
                self.trainer.policy.act(
                    eval_obs_list_flat, eval_rnn_states_list, eval_rnn_states_comm_list,
                    eval_masks_lists, eval_active_masks_lists, eval_available_actions_list,
                    deterministic=self.eval_deterministic, n_agents=self.eval_num_agents,
                    n_enemies=self.eval_num_enemies, n_entities=self.eval_num_entities)

            th_na_list = [eval_ob.shape[0] for eval_ob in eval_obs_list_flat]
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
                    if done_episodes_per_thread[idx][eval_i] < eval_episodes_per_thread[idx][eval_i] and eval_dones_env[eval_i]:
                        done_episodes_per_thread[idx][eval_i] += 1
                        per_thread_rewards = np.array(one_episode_rewards[idx])[:, eval_i]
                        eval_episode_rewards[idx][eval_i] += np.sum(per_thread_rewards)
                        for ep in range(len(per_thread_rewards)):
                            one_episode_rewards[idx][ep][eval_i] = 0.0
                        if "StarCraft" in self.env_name:
                            if eval_infos[eval_i][0].get("won", False):
                                eval_battles_won[idx] += 1
                        elif "AliceBob" in self.env_name:
                            if eval_infos[eval_i][0].get('battle_won', False):
                                eval_battles_won[idx] += 1
                        elif "Football" in self.env_name:
                            if eval_infos[eval_i][0].get('score_reward', 0) > 0:
                                eval_battles_won[idx] += 1

            for idx in idxs_tuple:
                if np.all(done_episodes_per_thread[idx] == eval_episodes_per_thread[idx]) and not recorded_envs[idx]:
                    recorded_envs[idx] = True
                    eval_episode = np.sum(eval_episodes_per_thread[idx])
                    task_returns[idx] = np.mean(eval_episode_rewards[idx])
                    task_win_rates[idx] = eval_battles_won[idx] / eval_episode if eval_episode > 0 else 0.0

            if np.all(done_episodes_per_thread == eval_episodes_per_thread):
                break

        return task_returns, task_win_rates

    # ─── Evolution ──────────────────────────────────────────────

    def _evolve(self, fitness_matrix, total_num_steps):
        M = len(self.solvers)
        if M <= 1:
            return

        factorial_cost = -fitness_matrix
        ranks = np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1
        rank_fitness = 1.0 / ranks

        scores = build_mating_scores(rank_fitness, self.evo_threshold, self.evo_weight_extra, self.evo_weight_common)
        pairs, loners = bidirectional_selection(scores)

        print(f"  [SESiL Evo] pairs: {pairs}, loners: {loners}")

        sample_obs = self.buffer.buffer_lists[0].obs[0]
        sample_obs_flat = torch.tensor(
            np.concatenate(sample_obs).reshape(-1, sample_obs.shape[-1]),
            dtype=torch.float32, device=self.device
        )

        new_solvers = []

        for (a, b) in pairs:
            solver_a = self.solvers[a]
            solver_b = self.solvers[b]

            offspring_actor = merge_actors(
                solver_a.policy.actor, solver_b.policy.actor,
                sample_obs_flat, self.device)

            merged_tasks = sorted(set(solver_a.task_ids + solver_b.task_ids))

            offspring_policy = Policy(self.all_args,
                                     self.multi_envs,
                                     self.num_thread_per_env,
                                     self.envs.observation_space,
                                     self.share_observation_space,
                                     self.envs.action_space,
                                     device=self.device)
            offspring_policy.actor = offspring_actor
            offspring_policy.actor_optimizer = torch.optim.Adam(
                offspring_policy.actor.parameters(),
                lr=self.all_args.lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)

            offspring_trainer = Trainer(self.all_args, offspring_policy, self.num_agents, self.num_enemies, self.num_entities, device=self.device)
            new_solvers.append(Solver(offspring_policy, offspring_trainer, merged_tasks,
                                       self.multi_envs, self.num_agents, self.num_enemies, self.num_entities, self.device))

            print(f"    Merged solver {a} (tasks {solver_a.task_ids}) + solver {b} (tasks {solver_b.task_ids}) -> offspring (tasks {merged_tasks})")

        for l in loners:
            new_solvers.append(self.solvers[l])

        self.solvers = new_solvers

        if self.solvers:
            self.policy = self.solvers[0].policy
            self.trainer = self.solvers[0].trainer

    # ─── Logging & saving ───────────────────────────────────────

    def log_eval(self, eval_infos, total_num_steps):
        for k, v in eval_infos.items():
            if self.use_wandb:
                import wandb
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def save(self):
        for i, solver in enumerate(self.solvers):
            actor_path = os.path.join(self.save_dir, f'actor_solver{i}.pt')
            torch.save(solver.policy.actor.state_dict(), actor_path)
            critic_path = os.path.join(self.save_dir, f'critic_solver{i}.pt')
            torch.save(solver.policy.critic.state_dict(), critic_path)
