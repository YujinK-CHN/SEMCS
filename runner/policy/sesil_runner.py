"""
SESiL runner — generational evolutionary training.

Gen 0: train M independent solvers (one per task) for evo_interval_init episodes.
Gen 1+: evaluate all solvers on all tasks → SEMFO selection → merge pairs → offspring
         inherits both parents' tasks → train for evo_interval episodes → repeat.
Population shrinks each generation until one solver remains or budget exhausted.

Logging uses the same TensorBoard format as MCS (eval_win_rate_{task}, etc.)
so plot_results.py works unchanged.
"""
import copy
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


class Solver:
    """One member of the population: a policy + its assigned task indices."""
    def __init__(self, policy, trainer, task_ids, device):
        self.policy = policy
        self.trainer = trainer
        self.task_ids = list(task_ids)
        self.device = device


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
        self.evo_interval_init = self.all_args.evo_interval_init if self.all_args.evo_interval_init else self.evo_interval
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
            self.solvers.append(Solver(policy, trainer, [task_idx], self.device))

        # Use first solver as the "active" policy/trainer for base class compatibility
        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

        self.buffer = MultiEnvSharedReplayBufferComm(
            self.all_args, self.num_agents, self.num_entities,
            self.obs_space_list, self.cent_obs_space_list, self.act_space_list,
            self.num_thread_per_env
        )

    def run(self):
        start = time.time()
        cumulative_steps = 0
        generation = 0

        while cumulative_steps < self.num_env_steps and len(self.solvers) > 0:
            evo_eps = self.evo_interval_init if generation == 0 else self.evo_interval
            print(f"\n{'='*60}")
            print(f"Generation {generation}: {len(self.solvers)} solvers, training {evo_eps} episodes each")
            print(f"  Solvers: {[s.task_ids for s in self.solvers]}")

            # === TRAIN each solver on its assigned tasks ===
            for solver_idx, solver in enumerate(self.solvers):
                steps_used = self._train_solver(solver, evo_eps)
                cumulative_steps += steps_used
                if cumulative_steps >= self.num_env_steps:
                    break

            # === EVALUATE all solvers on all tasks ===
            fitness_matrix = self._evaluate_population(cumulative_steps)
            print(f"  Fitness matrix (solvers x tasks):\n{np.array2string(fitness_matrix, precision=3)}")

            # Log per-task performance (best solver per task)
            self._log_population_performance(fitness_matrix, cumulative_steps)

            # === EVOLVE ===
            if len(self.solvers) > 1:
                self._evolve(fitness_matrix, cumulative_steps)

            generation += 1
            end = time.time()
            print(f"  Cumulative steps: {cumulative_steps}/{self.num_env_steps}, "
                  f"Time: {(end - start) / 3600:.2f}h")

        print(f"\nSESiL finished. Final solver has tasks: {self.solvers[0].task_ids if self.solvers else 'none'}")

    def _train_solver(self, solver, num_episodes):
        """Train a solver on its assigned tasks for num_episodes episodes. Returns steps consumed."""
        self.policy = solver.policy
        self.trainer = solver.trainer
        self.trainer.policy = solver.policy

        self.warmup()
        steps_used = 0

        done_episodes_rewards = [[] for _ in range(self.num_multi_envs)]
        one_episode_rewards = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]

        for episode in range(num_episodes):
            self.trainer.prep_rollout()

            # Collect one episode of data from ALL envs (framework requires it)
            # but only train on assigned tasks
            rewards_episode, infos_episode, dones_episode = self._collect_episode()
            self.compute()

            ep_steps = self.episode_length * self.n_rollout_threads
            steps_used += ep_steps

            # Train only on assigned tasks
            self._train_on_assigned_tasks(solver, episode)

            # Track rewards
            for rewards_step, dones_step in zip(rewards_episode, dones_episode):
                for idx, (rewards_tuple, dones_tuple) in enumerate(zip(rewards_step, dones_step)):
                    dones_env = np.all(dones_tuple, axis=1)
                    reward_env = np.mean(rewards_tuple, axis=1).flatten()
                    one_episode_rewards[idx] += reward_env
                    for t in range(self.num_thread_per_env):
                        if dones_env[t]:
                            done_episodes_rewards[idx].append(one_episode_rewards[idx][t])
                            one_episode_rewards[idx][t] = 0

        return steps_used

    def _collect_episode(self):
        """Collect one episode of data from all envs."""
        rewards_episode = []
        dones_episode = []
        infos_episode = []
        for step in range(self.episode_length):
            values_list, actions_list, action_log_probs_list, _, \
                rnn_states_list, rnn_states_comm_list, rnn_states_critic_list = self._collect_step(step)

            obs_list, share_obs_list, rewards_list, dones_list, \
                infos_list, available_actions_list, idxs_tuple = self.envs.step(actions_list)

            for obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_comm, rnn_states_critic, idx in zip(
                    obs_list, share_obs_list, rewards_list, dones_list, infos_list, available_actions_list,
                    values_list, actions_list, action_log_probs_list, rnn_states_list, rnn_states_comm_list, rnn_states_critic_list, idxs_tuple):
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, rnn_states, rnn_states_comm, rnn_states_critic
                self._insert(data, idx)

            rewards_episode.append(rewards_list)
            dones_episode.append(dones_list)
            infos_episode.append(infos_list)
        return rewards_episode, infos_episode, dones_episode

    @torch.no_grad()
    def _collect_step(self, step):
        """Collect actions for one step across all envs."""
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
        rnn_states = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state]
        rnn_states_comm = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_comm]
        rnn_states_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_critic]

        return values, actions, action_log_probs, None, rnn_states, rnn_states_comm, rnn_states_critic

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

    def _train_on_assigned_tasks(self, solver, episode):
        """Run PPO update but only on the solver's assigned tasks' buffers."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, task_ids=solver.task_ids)

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

    def warmup(self):
        obs_list, share_obs_list, available_actions_list, idxs_tuple = self.envs.reset()
        for obs, share_obs, available_actions, idx in zip(obs_list, share_obs_list, available_actions_list, idxs_tuple):
            if not self.use_centralized_V:
                share_obs = obs
            self.buffer.buffer_lists[idx].share_obs[0] = share_obs.copy()
            self.buffer.buffer_lists[idx].obs[0] = obs.copy()
            self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()

    def _evaluate_population(self, total_num_steps):
        """
        Evaluate every solver on every task.
        Returns fitness_matrix: (num_solvers, num_tasks), higher = better.
        Also logs metrics in TensorBoard format for plot_results.py.
        """
        M = len(self.solvers)
        K = self.num_multi_envs
        fitness_matrix = np.zeros((M, K))

        for solver_idx, solver in enumerate(self.solvers):
            # Temporarily set this solver as active
            self.policy = solver.policy
            self.trainer = solver.trainer
            self.trainer.policy = solver.policy

            task_rewards = self._eval_solver_on_all_tasks()

            for task_idx in range(K):
                fitness_matrix[solver_idx, task_idx] = task_rewards[task_idx]

        # Log best-per-task performance (what plot_results.py reads)
        for task_idx in range(K):
            best_reward = np.max(fitness_matrix[:, task_idx])
            task_name = self.eval_multi_envs[task_idx]
            eval_infos = {f'eval_episode_rewards_{task_name}': best_reward}
            if "AliceBob" in self.env_name or "Football" in self.env_name:
                eval_infos[f'eval_win_rate_{task_name}'] = best_reward
            self.log_eval(eval_infos, total_num_steps)

        return fitness_matrix

    @torch.no_grad()
    def _eval_solver_on_all_tasks(self):
        """
        Run the current self.policy on all eval tasks.
        Returns list of mean episode rewards per task.
        """
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
                        if "AliceBob" in self.env_name:
                            if eval_infos[eval_i][0].get('battle_won', False):
                                eval_battles_won[idx] += 1

            for idx in idxs_tuple:
                if np.all(done_episodes_per_thread[idx] == eval_episodes_per_thread[idx]) and not recorded_envs[idx]:
                    recorded_envs[idx] = True
                    eval_episode = np.sum(eval_episodes_per_thread[idx])
                    task_returns[idx] = np.mean(eval_episode_rewards[idx])
                    if "AliceBob" in self.env_name:
                        task_returns[idx] = eval_battles_won[idx] / eval_episode

            if np.all(done_episodes_per_thread == eval_episodes_per_thread):
                break

        return task_returns

    def _evolve(self, fitness_matrix, total_num_steps):
        """
        Run SEMFO-inspired evolution: build mating scores, select pairs,
        merge encoders, produce offspring. Population shrinks.
        """
        M = len(self.solvers)
        if M <= 1:
            return

        # Convert to rank-based fitness (higher = better)
        factorial_cost = -fitness_matrix  # negate so lower cost = higher fitness
        ranks = np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1
        rank_fitness = 1.0 / ranks

        scores = build_mating_scores(rank_fitness, self.evo_threshold, self.evo_weight_extra, self.evo_weight_common)
        pairs, loners = bidirectional_selection(scores)

        print(f"  [SESiL Evo] pairs: {pairs}, loners: {loners}")

        # Build sample obs for permutation alignment
        sample_obs = self.buffer.buffer_lists[0].obs[0]
        sample_obs_flat = torch.tensor(
            np.concatenate(sample_obs).reshape(-1, sample_obs.shape[-1]),
            dtype=torch.float32, device=self.device
        )

        new_solvers = []

        # Create offspring for each pair
        for (a, b) in pairs:
            solver_a = self.solvers[a]
            solver_b = self.solvers[b]

            offspring_actor = merge_actors(
                solver_a.policy.actor, solver_b.policy.actor,
                sample_obs_flat, self.device)

            # Offspring inherits both parents' task assignments
            merged_tasks = sorted(set(solver_a.task_ids + solver_b.task_ids))

            # Create new policy with merged actor
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
            new_solvers.append(Solver(offspring_policy, offspring_trainer, merged_tasks, self.device))

            print(f"    Merged solver {a} (tasks {solver_a.task_ids}) + solver {b} (tasks {solver_b.task_ids}) → offspring (tasks {merged_tasks})")

        # Keep loners as-is
        for l in loners:
            new_solvers.append(self.solvers[l])

        self.solvers = new_solvers

        # Update active policy reference
        if self.solvers:
            self.policy = self.solvers[0].policy
            self.trainer = self.solvers[0].trainer

    def log_eval(self, eval_infos, total_num_steps):
        for k, v in eval_infos.items():
            if self.use_wandb:
                import wandb
                wandb.log({k: np.mean(v)}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def save(self):
        for i, solver in enumerate(self.solvers):
            actor_path = os.path.join(self.save_dir, f'actor_solver{i}.pt')
            torch.save(solver.policy.actor.state_dict(), actor_path)
            critic_path = os.path.join(self.save_dir, f'critic_solver{i}.pt')
            torch.save(solver.policy.critic.state_dict(), critic_path)
