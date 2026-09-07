"""
SESiL runner — evolutionary multi-task MARL.

Each generation:
  1. Train M solvers (each on its assigned task subset) sequentially
  2. Evaluate all solvers on all tasks → fitness matrix
  3. Mate selection (bidirectional, complementarity-based)
  4. Merge paired solvers (permutation-aligned weight averaging for actor+critic)
  5. Loners survive unchanged
  6. Offspring inherit union of parents' tasks

Budget is split evenly across generations. Solvers within a generation share
their generation's budget proportional to their task count.
"""
import copy
import json
import random
import time
import os
import numpy as np
import torch


from runner.policy.base_runner import Runner
from base_policy.utils.util import _t2n
from base_policy.utils.multi_envs_shared_buffer import MultiEnvSharedReplayBufferComm
from base_policy.algorithms.sesil.sesil_policy import sesilPolicy as MappoPolicy
from base_policy.algorithms.mcs.mcs_policy import mcsPolicy as McsPolicy
from base_policy.algorithms.dt2gs.dt2gs_policy import dt2gsPolicy as Dt2gsPolicy
from base_policy.algorithms.mcs.mcs_trainer import mcsTrainer as McsTrainer
from base_policy.algorithms.dt2gs.dt2gs_trainer import dt2gsTrainer as Dt2gsTrainer
from base_policy.algorithms.sesil.evolution import (
    build_mating_scores, bidirectional_selection, merge_actors, merge_critics
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
        self.trainer.num_multi_envs = len(self.task_ids)
        self.trainer.multi_envs = [all_multi_envs[i] for i in self.task_ids]
        self.trainer.n_agents_list = [all_num_agents[i] for i in self.task_ids]
        self.trainer.n_enemies_list = [all_num_enemies[i] for i in self.task_ids]
        self.trainer.n_entities_list = [all_num_entities[i] for i in self.task_ids]


def _generate_task_assignments(num_tasks, num_solvers, tasks_per_solver, seed=42):
    """
    Randomly assign tasks to solvers, guaranteeing full coverage of all tasks.
    Returns list of lists, one per solver.
    """
    rng = random.Random(seed)
    all_tasks = list(range(num_tasks))

    # Phase 1: ensure every task is covered by at least one solver
    assignments = [[] for _ in range(num_solvers)]
    uncovered = list(all_tasks)
    rng.shuffle(uncovered)
    for i, t in enumerate(uncovered):
        assignments[i % num_solvers].append(t)

    # Phase 2: fill each solver up to tasks_per_solver
    for i in range(num_solvers):
        while len(assignments[i]) < tasks_per_solver:
            candidates = [t for t in all_tasks if t not in assignments[i]]
            if not candidates:
                break
            assignments[i].append(rng.choice(candidates))
        assignments[i] = sorted(assignments[i])

    return assignments


# Structured task assignment: tier-based grouping by unit-type knowledge
TASK_TIERS = {
    "easy": ["3m", "2m_vs_1z", "3s_vs_3z", "2s3z", "2s_vs_1sc"],
    "medium": ["8m", "5m_vs_6m", "3s_vs_4z", "3s5z", "1c3s5z", "6h_vs_8z", "MMM", "so_many_baneling"],
    "hard": ["3s_vs_5z", "8m_vs_9m", "3s5z_vs_3s6z", "MMM2", "corridor"],
}

# Hardcoded solver assignments by task name (8 solvers, 3 tasks each, 10 unique tasks)
STRUCTURED_ASSIGNMENTS = [
    ["3m", "8m", "8m_vs_9m"],           # marine specialist
    ["3m", "5m_vs_6m", "8m_vs_9m"],     # marine asymmetric
    ["3s_vs_3z", "3s5z", "3s_vs_5z"],   # stalker-zealot specialist
    ["3s_vs_3z", "3s5z", "3s5z_vs_3s6z"],  # stalker-zealot scaled
    ["2s3z", "3s5z", "3s5z_vs_3s6z"],   # protoss mixed
    ["2s3z", "1c3s5z", "3s_vs_5z"],     # protoss multi-unit
    ["3m", "1c3s5z", "3s5z_vs_3s6z"],   # cross-type
    ["3s_vs_3z", "5m_vs_6m", "8m_vs_9m"],  # cross-type
]


def _generate_structured_assignments(task_names, num_solvers):
    """
    Map STRUCTURED_ASSIGNMENTS to task indices based on task_names ordering.
    Returns (assignments, selected_task_names).
    """
    name_to_idx = {name: idx for idx, name in enumerate(task_names)}

    assignments = []
    for si in range(num_solvers):
        template = STRUCTURED_ASSIGNMENTS[si % len(STRUCTURED_ASSIGNMENTS)]
        solver_tasks = []
        for tname in template:
            if tname in name_to_idx:
                solver_tasks.append(name_to_idx[tname])
            else:
                print(f"  WARNING: structured assignment task '{tname}' not found in train_tasks, skipping")
        assert solver_tasks, f"Solver {si} has no valid tasks from {template}"
        assignments.append(sorted(solver_tasks))

    selected = sorted(set(t for a in assignments for t in a))
    selected_names = [task_names[i] for i in selected]
    return assignments, selected_names


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

        # Evolution parameters
        self.evo_solver_algo = self.all_args.evo_solver_algo
        self.evo_num_solvers = self.all_args.evo_num_solvers
        self.evo_tasks_per_solver = self.all_args.evo_tasks_per_solver
        self.evo_num_generations = self.all_args.evo_num_generations
        self.evo_pretrain_budget = self.all_args.evo_pretrain_budget
        self.evo_pretrain_phase_a_ratio = self.all_args.evo_pretrain_phase_a_ratio
        self.evo_pretrain_mode = self.all_args.evo_pretrain_mode

        self.evo_individual_budget = self.all_args.evo_individual_budget
        self.evo_keep_population = self.all_args.evo_keep_population
        self.evo_eval_episodes = self.all_args.evo_eval_episodes
        self.evo_threshold = self.all_args.evo_threshold
        self.evo_weight_extra = self.all_args.evo_weight_extra
        self.evo_weight_common = self.all_args.evo_weight_common

        # Generate task assignments
        self.evo_task_assignment = self.all_args.evo_task_assignment
        if self.evo_task_assignment == "structured":
            task_names = [env for env in self.multi_envs]
            task_assignments, selected_names = _generate_structured_assignments(
                task_names, self.evo_num_solvers)
            print(f"  Structured task assignment: {len(selected_names)} unique tasks")
            print(f"  Selected: {selected_names}")
        else:
            task_assignments = _generate_task_assignments(
                self.num_multi_envs, self.evo_num_solvers,
                self.evo_tasks_per_solver, seed=self.all_args.seed)

        # Create solvers
        self.solvers = []
        for si in range(self.evo_num_solvers):
            policy = self._create_policy()
            trainer = self._create_trainer(policy)
            self.solvers.append(Solver(policy, trainer, task_assignments[si],
                                       self.multi_envs, self.num_agents, self.num_enemies, self.num_entities, self.device))

        # Evolution log file
        self.evo_log_path = os.path.join(self.run_dir, 'evolution_log.txt')
        self._pretrain_done = False
        self._gen_steps_path = os.path.join(self.run_dir, 'generation_steps.json')
        self._gen_steps = {}  # {"pretrain_end": step, "gen_0": step, ...}

        # Use first solver as "active" for base class compatibility
        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

        self.buffer = MultiEnvSharedReplayBufferComm(
            self.all_args, self.num_agents, self.num_entities,
            self.obs_space_list, self.cent_obs_space_list, self.act_space_list,
            self.num_thread_per_env
        )

    # ─── Helpers ────────────────────────────────────────────────

    def _create_policy(self):
        policy_cls = {"mappo": MappoPolicy, "mcs": McsPolicy, "dt2gs": Dt2gsPolicy}[self.evo_solver_algo]
        return policy_cls(self.all_args,
                          self.multi_envs,
                          self.num_thread_per_env,
                          self.envs.observation_space,
                          self.share_observation_space,
                          self.envs.action_space,
                          device=self.device)

    def _create_trainer(self, policy):
        trainer_cls = {"mappo": McsTrainer, "mcs": McsTrainer, "dt2gs": Dt2gsTrainer}[self.evo_solver_algo]
        return trainer_cls(self.all_args, policy, self.num_agents, self.num_enemies, self.num_entities, device=self.device)

    def _steps_per_episode(self, num_tasks):
        return self.episode_length * num_tasks * self.num_thread_per_env

    def _save_gen_steps(self):
        with open(self._gen_steps_path, 'w') as f:
            json.dump(self._gen_steps, f, indent=2)

    # ─── Main loop ──────────────────────────────────────────────

    def run(self):
        start = time.time()
        self.cumulative_steps = 0

        print(f"  evo_individual_budget={self.evo_individual_budget}, "
              f"solvers={len(self.solvers)}, "
              f"evo_num_generations={self.evo_num_generations}")

        start_gen = 0

        # Resume support
        if getattr(self.all_args, 'resume', False):
            start_gen = self._restore_sesil_checkpoint()

        self.next_eval_step = self.cumulative_steps

        # Pretraining phase (no evolution, runs before gen 0)
        if self.evo_pretrain_budget > 0 and not self._pretrain_done:
            print(f"\n{'='*60}")
            print(f"Pretraining ({self.evo_pretrain_mode}): "
                  f"{len(self.solvers)} solvers, budget={self.evo_pretrain_budget} steps")
            if self.evo_pretrain_mode == "encoder":
                assert self.evo_solver_algo == "mappo", \
                    "Encoder pretrain mode is only supported for sesil_mappo."
                self._pretrain_encoder(self.evo_pretrain_budget)
            elif self.evo_pretrain_mode == "full":
                self._pretrain_full(self.evo_pretrain_budget)
            elif self.evo_pretrain_mode == "common":
                assert self.evo_solver_algo == "mappo", \
                    "Common pretrain mode is only supported for sesil_mappo."
                self._pretrain_common(self.evo_pretrain_budget)
            elif self.evo_pretrain_mode == "common_head":
                assert self.evo_solver_algo == "mappo", \
                    "Common head pretrain mode is only supported for sesil_mappo."
                self._pretrain_common_head(self.evo_pretrain_budget)
            elif self.evo_pretrain_mode == "apt":
                assert self.evo_solver_algo == "mappo", \
                    "APT pretrain mode is only supported for sesil_mappo."
                self._pretrain_apt(self.evo_pretrain_budget)
            else:
                self._train_all_solvers(self.evo_pretrain_budget)
            self._save_sesil_checkpoint(-1)
            self._gen_steps["pretrain_end"] = self.cumulative_steps
            self._save_gen_steps()
            print(f"  Pretraining done. Cumulative steps: {self.cumulative_steps}")

        for gen in range(start_gen, self.evo_num_generations):
            if self.cumulative_steps >= self.num_env_steps:
                print(f"\n  Budget exhausted ({self.cumulative_steps}/{self.num_env_steps}), stopping.")
                break

            budget_per_gen = len(self.solvers) * self.evo_individual_budget
            gen_budget = min(budget_per_gen, self.num_env_steps - self.cumulative_steps)

            print(f"\n{'='*60}")
            print(f"Generation {gen}/{self.evo_num_generations}: "
                  f"{len(self.solvers)} solvers, budget={gen_budget} steps")
            for si, s in enumerate(self.solvers):
                print(f"  Solver {si}: tasks {s.task_ids}")

            self._gen_steps[f"gen_{gen}"] = self.cumulative_steps
            self._save_gen_steps()

            # === Train all solvers ===
            self._train_all_solvers(gen_budget)

            # === Evaluate population ===
            fitness_matrix, win_rate_matrix = self._evaluate_population(self.cumulative_steps)
            print(f"  Fitness matrix (solvers x tasks):\n{np.array2string(fitness_matrix, precision=3)}")
            print(f"  Win rate matrix:\n{np.array2string(win_rate_matrix, precision=3)}")

            # === Evolve (mate selection + merge) — skip on last generation ===
            pre_evo_tasks = [list(s.task_ids) for s in self.solvers]
            pairs, loners = [], []
            if gen < self.evo_num_generations - 1 and len(self.solvers) > 1:
                pairs, loners = self._evolve(fitness_matrix, win_rate_matrix)

            # === Save checkpoint ===
            self._save_sesil_checkpoint(gen)

            elapsed_h = (time.time() - start) / 3600
            print(f"  Cumulative steps: {self.cumulative_steps}/{self.num_env_steps}, "
                  f"Time: {elapsed_h:.2f}h")

            self._log_generation(gen, fitness_matrix, win_rate_matrix, pairs, loners, pre_evo_tasks, elapsed_h)

        if self.cumulative_steps < self.num_env_steps:
            print(f"\n  WARNING: All {self.evo_num_generations} generations completed but only "
                  f"{self.cumulative_steps}/{self.num_env_steps} steps used "
                  f"({self.num_env_steps - self.cumulative_steps} steps unused). "
                  f"Consider increasing --evo_individual_budget.")

        print(f"\nSESiL finished. {len(self.solvers)} solver(s) remain.")
        for si, s in enumerate(self.solvers):
            print(f"  Solver {si}: tasks {s.task_ids}")

    # ─── Sequential training ─────────────────────────────────────

    def _train_all_solvers(self, gen_budget):
        """Train each solver sequentially — each gets exactly evo_individual_budget."""
        for si, solver in enumerate(self.solvers):
            self.policy = solver.policy
            self.trainer = solver.trainer
            self.trainer.policy = solver.policy
            filtered_buf = FilteredBuffer(self.buffer, solver.task_ids)
            n_tasks = len(solver.task_ids)
            spe = self._steps_per_episode(n_tasks)

            solver_budget = self.evo_individual_budget
            episodes_per_solver = max(1, solver_budget // spe)

            print(f"  Training solver {si} (tasks {solver.task_ids}) for "
                  f"{episodes_per_solver} episodes ({solver_budget} steps budget)")

            for episode in range(episodes_per_solver):
                self._warmup_tasks(solver.task_ids)
                self._collect_episode(solver)
                self._compute_filtered(solver, filtered_buf)
                solver.trainer.prep_training()
                solver.trainer.train(filtered_buf, episode)
                filtered_buf.after_update()
                self.cumulative_steps += spe

                if self.cumulative_steps >= self.next_eval_step:
                    self._evaluate_population(self.cumulative_steps)
                    self.next_eval_step += self.eval_steps_interval

                if (episode + 1) % max(1, episodes_per_solver // 5) == 0:
                    print(f"    Solver {si} episode {episode+1}/{episodes_per_solver}, "
                          f"cumulative: {self.cumulative_steps}")

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

    def _pretrain_full(self, pretrain_budget):
        """Pretrain each solver on ALL tasks (not just assigned)."""
        all_task_ids = list(range(self.num_multi_envs))
        original_task_ids = [list(s.task_ids) for s in self.solvers]

        for s in self.solvers:
            s.task_ids = list(all_task_ids)
            s._sync_trainer(self.multi_envs, self.num_agents, self.num_enemies, self.num_entities)

        for si, s in enumerate(self.solvers):
            print(f"  Solver {si}: pretraining on all tasks {all_task_ids}")

        self._train_all_solvers(pretrain_budget)

        for s, orig in zip(self.solvers, original_task_ids):
            s.task_ids = orig
            s._sync_trainer(self.multi_envs, self.num_agents, self.num_enemies, self.num_entities)

    def _pretrain_encoder(self, pretrain_budget):
        """Train one shared encoder on all tasks (50%), then copy+freeze encoder and finetune solvers on assigned tasks (50%)."""
        all_task_ids = list(range(self.num_multi_envs))
        phase_a_budget = int(pretrain_budget * self.evo_pretrain_phase_a_ratio)
        phase_b_budget = pretrain_budget - phase_a_budget

        # Phase A: train one temporary solver on all tasks
        tmp_policy = self._create_policy()
        tmp_trainer = self._create_trainer(tmp_policy)
        tmp_solver = Solver(tmp_policy, tmp_trainer, all_task_ids,
                            self.multi_envs, self.num_agents, self.num_enemies, self.num_entities, self.device)

        original_solvers = self.solvers
        self.solvers = [tmp_solver]

        print(f"  Encoder pretrain phase A: training one solver on all tasks, budget={phase_a_budget}")
        self._train_all_solvers(phase_a_budget)

        self.solvers = original_solvers

        # Copy encoder to all solvers
        encoder_attr = 'integration' if hasattr(tmp_policy.actor, 'integration') else 'encoder'
        actor_encoder_sd = getattr(tmp_policy.actor, encoder_attr).state_dict()
        critic_encoder_sd = getattr(tmp_policy.critic, encoder_attr).state_dict()

        for si, solver in enumerate(self.solvers):
            getattr(solver.policy.actor, encoder_attr).load_state_dict(actor_encoder_sd)
            getattr(solver.policy.critic, encoder_attr).load_state_dict(critic_encoder_sd)
            print(f"  Copied pretrained encoder to solver {si}")

        # Phase B: finetune solvers on assigned tasks with encoder frozen
        for solver in self.solvers:
            for param in getattr(solver.policy.actor, encoder_attr).parameters():
                param.requires_grad = False
            for param in getattr(solver.policy.critic, encoder_attr).parameters():
                param.requires_grad = False

        print(f"  Encoder pretrain phase B: finetuning solvers (encoder frozen), budget={phase_b_budget}")
        self._train_all_solvers(phase_b_budget)

        # Unfreeze encoder for all solvers
        for solver in self.solvers:
            for param in getattr(solver.policy.actor, encoder_attr).parameters():
                param.requires_grad = True
            for param in getattr(solver.policy.critic, encoder_attr).parameters():
                param.requires_grad = True

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

    def _pretrain_common(self, pretrain_budget):
        """Common pretrain: train on all tasks (50%), then copy+freeze actor and finetune critic (50%)."""
        all_task_ids = list(range(self.num_multi_envs))
        phase_a_budget = int(pretrain_budget * self.evo_pretrain_phase_a_ratio)
        phase_b_budget = pretrain_budget - phase_a_budget

        # Phase A: train one temporary solver on all tasks
        tmp_policy = self._create_policy()
        tmp_trainer = self._create_trainer(tmp_policy)
        tmp_solver = Solver(tmp_policy, tmp_trainer, all_task_ids,
                            self.multi_envs, self.num_agents, self.num_enemies,
                            self.num_entities, self.device)

        original_solvers = self.solvers
        self.solvers = [tmp_solver]

        print(f"  Common pretrain phase A: training one solver on all tasks, budget={phase_a_budget}")
        self._train_all_solvers(phase_a_budget)

        # Copy full actor to all solvers
        self.solvers = original_solvers
        actor_sd = tmp_solver.policy.actor.state_dict()
        for si, solver in enumerate(self.solvers):
            solver.policy.actor.load_state_dict(actor_sd)
            solver.policy.actor_optimizer = torch.optim.Adam(
                solver.policy.actor.parameters(),
                lr=self.all_args.lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)
            print(f"  Copied common actor to solver {si}")

        # Phase B: finetune solvers on assigned tasks with actor frozen
        for solver in self.solvers:
            for param in solver.policy.actor.parameters():
                param.requires_grad = False

        print(f"  Common pretrain phase B: finetuning solvers (actor frozen), budget={phase_b_budget}")
        self._train_all_solvers(phase_b_budget)

        # Unfreeze actor for all solvers
        for solver in self.solvers:
            for param in solver.policy.actor.parameters():
                param.requires_grad = True

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

    def _pretrain_common_head(self, pretrain_budget):
        """Common head pretrain: train on all tasks (50%), copy actor+critic, wipe+finetune last layer only (50%)."""
        import torch.nn as nn
        all_task_ids = list(range(self.num_multi_envs))
        phase_a_budget = int(pretrain_budget * self.evo_pretrain_phase_a_ratio)
        phase_b_budget = pretrain_budget - phase_a_budget

        # Phase A: train one temporary solver on all tasks
        tmp_policy = self._create_policy()
        tmp_trainer = self._create_trainer(tmp_policy)
        tmp_solver = Solver(tmp_policy, tmp_trainer, all_task_ids,
                            self.multi_envs, self.num_agents, self.num_enemies,
                            self.num_entities, self.device)

        original_solvers = self.solvers
        self.solvers = [tmp_solver]

        print(f"  Common head pretrain phase A: training one solver on all tasks, budget={phase_a_budget}")
        self._train_all_solvers(phase_a_budget)

        # Copy actor+critic to all solvers, wipe last layer, freeze everything else
        self.solvers = original_solvers
        actor_sd = tmp_solver.policy.actor.state_dict()
        critic_sd = tmp_solver.policy.critic.state_dict()
        for si, solver in enumerate(self.solvers):
            solver.policy.actor.load_state_dict(actor_sd)
            solver.policy.critic.load_state_dict(critic_sd)

            nn.init.orthogonal_(solver.policy.actor.act_layer.action_out.linear.weight, gain=0.01)
            nn.init.constant_(solver.policy.actor.act_layer.action_out.linear.bias, 0)
            nn.init.orthogonal_(solver.policy.critic.v_out.weight, gain=1.0)
            nn.init.constant_(solver.policy.critic.v_out.bias, 0)

            for param in solver.policy.actor.parameters():
                param.requires_grad = False
            for param in solver.policy.actor.act_layer.action_out.linear.parameters():
                param.requires_grad = True

            for param in solver.policy.critic.parameters():
                param.requires_grad = False
            for param in solver.policy.critic.v_out.parameters():
                param.requires_grad = True

            solver.policy.actor_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, solver.policy.actor.parameters()),
                lr=self.all_args.lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)
            solver.policy.critic_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, solver.policy.critic.parameters()),
                lr=self.all_args.critic_lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)

            print(f"  Copied common model to solver {si}, wiped & unfroze last layer")

        # Phase B: finetune last layer on assigned tasks
        print(f"  Common head pretrain phase B: finetuning last layer on assigned tasks, budget={phase_b_budget}")
        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer
        self._train_all_solvers(phase_b_budget)

        # Unfreeze everything
        for solver in self.solvers:
            for param in solver.policy.actor.parameters():
                param.requires_grad = True
            for param in solver.policy.critic.parameters():
                param.requires_grad = True
            solver.policy.actor_optimizer = torch.optim.Adam(
                solver.policy.actor.parameters(),
                lr=self.all_args.lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)
            solver.policy.critic_optimizer = torch.optim.Adam(
                solver.policy.critic.parameters(),
                lr=self.all_args.critic_lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

    def _pretrain_apt(self, pretrain_budget):
        """APT pretrain: Phase A trains with intrinsic entropy reward (50%), Phase B same as common_head (50%)."""
        import torch.nn as nn
        from base_policy.algorithms.sesil.apt_entropy import compute_apt_reward, RMS

        all_task_ids = list(range(self.num_multi_envs))
        phase_a_budget = int(pretrain_budget * self.evo_pretrain_phase_a_ratio)
        phase_b_budget = pretrain_budget - phase_a_budget

        # Phase A: train one solver on all tasks using APT intrinsic reward
        tmp_policy = self._create_policy()
        tmp_trainer = self._create_trainer(tmp_policy)
        tmp_solver = Solver(tmp_policy, tmp_trainer, all_task_ids,
                            self.multi_envs, self.num_agents, self.num_enemies,
                            self.num_entities, self.device)

        apt_rms = RMS(device=self.device)
        knn_k = self.all_args.apt_knn_k
        knn_avg = bool(self.all_args.apt_knn_avg)
        knn_rms = bool(self.all_args.apt_knn_rms)
        knn_clip = self.all_args.apt_knn_clip

        self.policy = tmp_solver.policy
        self.trainer = tmp_solver.trainer
        self.trainer.policy = tmp_solver.policy
        filtered_buf = FilteredBuffer(self.buffer, all_task_ids)
        n_tasks = len(all_task_ids)
        spe = self._steps_per_episode(n_tasks)
        episodes = max(1, phase_a_budget // spe)

        print(f"  APT pretrain phase A: training one solver on all tasks with intrinsic reward, "
              f"budget={phase_a_budget}, episodes={episodes}")

        for episode in range(episodes):
            self._warmup_tasks(all_task_ids)
            self._collect_episode(tmp_solver)

            # Replace env rewards with APT intrinsic rewards
            self._replace_rewards_with_apt(
                tmp_solver, all_task_ids, apt_rms, knn_k, knn_avg, knn_rms, knn_clip)

            self._compute_filtered(tmp_solver, filtered_buf)
            tmp_solver.trainer.prep_training()
            tmp_solver.trainer.train(filtered_buf, episode)
            filtered_buf.after_update()
            self.cumulative_steps += spe

            if self.cumulative_steps >= self.next_eval_step:
                self._evaluate_population(self.cumulative_steps)
                self.next_eval_step += self.eval_steps_interval

            if (episode + 1) % max(1, episodes // 5) == 0:
                print(f"    APT phase A episode {episode+1}/{episodes}, "
                      f"cumulative: {self.cumulative_steps}")

        # Phase B: copy actor+critic, wipe last layer, freeze, finetune (same as common_head)
        original_solvers = self.solvers
        actor_sd = tmp_solver.policy.actor.state_dict()
        critic_sd = tmp_solver.policy.critic.state_dict()
        for si, solver in enumerate(original_solvers):
            solver.policy.actor.load_state_dict(actor_sd)
            solver.policy.critic.load_state_dict(critic_sd)

            nn.init.orthogonal_(solver.policy.actor.act_layer.action_out.linear.weight, gain=0.01)
            nn.init.constant_(solver.policy.actor.act_layer.action_out.linear.bias, 0)
            nn.init.orthogonal_(solver.policy.critic.v_out.weight, gain=1.0)
            nn.init.constant_(solver.policy.critic.v_out.bias, 0)

            for param in solver.policy.actor.parameters():
                param.requires_grad = False
            for param in solver.policy.actor.act_layer.action_out.linear.parameters():
                param.requires_grad = True

            for param in solver.policy.critic.parameters():
                param.requires_grad = False
            for param in solver.policy.critic.v_out.parameters():
                param.requires_grad = True

            solver.policy.actor_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, solver.policy.actor.parameters()),
                lr=self.all_args.lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)
            solver.policy.critic_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, solver.policy.critic.parameters()),
                lr=self.all_args.critic_lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)

            print(f"  Copied APT model to solver {si}, wiped & unfroze last layer")

        print(f"  APT pretrain phase B: finetuning last layer on assigned tasks, budget={phase_b_budget}")
        self.solvers = original_solvers
        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer
        self._train_all_solvers(phase_b_budget)

        # Unfreeze everything
        for solver in self.solvers:
            for param in solver.policy.actor.parameters():
                param.requires_grad = True
            for param in solver.policy.critic.parameters():
                param.requires_grad = True
            solver.policy.actor_optimizer = torch.optim.Adam(
                solver.policy.actor.parameters(),
                lr=self.all_args.lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)
            solver.policy.critic_optimizer = torch.optim.Adam(
                solver.policy.critic.parameters(),
                lr=self.all_args.critic_lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

    @torch.no_grad()
    def _replace_rewards_with_apt(self, solver, task_ids, apt_rms,
                                  knn_k, knn_avg, knn_rms, knn_clip):
        """Replace buffer rewards with APT intrinsic rewards computed from encoder representations."""
        from base_policy.algorithms.sesil.apt_entropy import compute_apt_reward

        solver.trainer.prep_rollout()
        n_agents_s = [self.num_agents[i] for i in task_ids]
        n_enemies_s = [self.num_enemies[i] for i in task_ids]
        n_entities_s = [self.num_entities[i] for i in task_ids]

        for step in range(self.episode_length):
            obs = [np.concatenate(self.buffer.buffer_lists[idx].obs[step]) for idx in task_ids]
            masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[step]) for idx in task_ids]
            active_masks = [np.concatenate(self.buffer.buffer_lists[idx].active_masks[step]) for idx in task_ids]

            from base_policy.utils.util import check
            tpdv = dict(dtype=torch.float32, device=self.device)
            obs_t = check(obs, tpdv)

            # Extract representations from encoder
            fea_list = solver.policy.actor._encode(obs_t, n_agents_s, n_enemies_s, n_entities_s)

            # Mean-pool entity dimension to get per-agent representations
            representations = torch.cat(
                [torch.mean(fea, dim=1) for fea in fea_list], dim=0)

            # Compute APT reward
            reward, apt_rms = compute_apt_reward(
                representations, representations,
                knn_k=knn_k, knn_avg=knn_avg, knn_rms=knn_rms,
                knn_clip=knn_clip, rms=apt_rms)

            # Distribute rewards back to each task's buffer
            offset = 0
            for local_idx, idx in enumerate(task_ids):
                n_agents = self.num_agents[idx]
                n_total = self.num_thread_per_env * n_agents
                task_reward = reward[offset:offset + n_total]
                task_reward_np = task_reward.cpu().numpy().reshape(
                    self.num_thread_per_env, n_agents, 1)
                self.buffer.buffer_lists[idx].rewards[step] = task_reward_np
                offset += n_total

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
        Returns fitness_matrix: (num_solvers, num_tasks).
        Also logs population-average per-task performance for plotting.
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

        # Log best-per-task performance: for each task, pick the best solver on that task
        for task_idx in range(K):
            best_solver = int(np.argmax(fitness_matrix[:, task_idx]))
            task_name = self.eval_multi_envs[task_idx]
            eval_infos = {f'eval_episode_rewards_{task_name}': fitness_matrix[best_solver, task_idx]}
            if "StarCraft" in self.env_name:
                eval_infos[f'eval_win_rate_{task_name}'] = win_rate_matrix[best_solver, task_idx]
            elif "AliceBob" in self.env_name or "Football" in self.env_name:
                eval_infos[f'eval_win_rate_{task_name}'] = fitness_matrix[best_solver, task_idx]
            self.log_eval(eval_infos, total_num_steps)

        return fitness_matrix, win_rate_matrix

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
                    task_returns[idx] = np.sum(eval_episode_rewards[idx]) / eval_episode
                    task_win_rates[idx] = eval_battles_won[idx] / eval_episode if eval_episode > 0 else 0.0

            if np.all(done_episodes_per_thread == eval_episodes_per_thread):
                break

        return task_returns, task_win_rates

    # ─── Evolution ──────────────────────────────────────────────

    def _evolve(self, fitness_matrix, win_rate_matrix):
        """Mate selection + merge. Returns (pairs, loners)."""
        M = len(self.solvers)
        if M <= 1:
            return [], list(range(M))

        # Combined fitness for mate selection: reward × win_rate
        mating_fitness = fitness_matrix * win_rate_matrix
        scores = build_mating_scores(mating_fitness, self.evo_threshold,
                                     self.evo_weight_extra, self.evo_weight_common)
        pairs, loners = bidirectional_selection(scores)

        print(f"  [SESiL Evo] pairs: {pairs}, loners: {loners}")

        # Get sample obs for permutation alignment
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

            offspring_policy = self._create_policy()
            offspring_policy.actor = offspring_actor
            offspring_policy.actor_optimizer = torch.optim.Adam(
                offspring_policy.actor.parameters(),
                lr=self.all_args.lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)

            if self.evo_solver_algo == "mappo":
                # MAPPO: also merge critic
                offspring_critic = merge_critics(
                    solver_a.policy.critic, solver_b.policy.critic)
                offspring_policy.critic = offspring_critic
                offspring_policy.critic_optimizer = torch.optim.Adam(
                    offspring_policy.critic.parameters(),
                    lr=self.all_args.critic_lr, eps=self.all_args.opti_eps,
                    weight_decay=self.all_args.weight_decay)
            # MCS/DT2GS: critic + centralized components stay freshly initialized

            offspring_trainer = self._create_trainer(offspring_policy)
            new_solvers.append(Solver(offspring_policy, offspring_trainer, merged_tasks,
                                       self.multi_envs, self.num_agents, self.num_enemies, self.num_entities, self.device))

            if self.evo_keep_population:
                offspring_policy_2 = copy.deepcopy(offspring_policy)
                offspring_policy_2.actor_optimizer = torch.optim.Adam(
                    offspring_policy_2.actor.parameters(),
                    lr=self.all_args.lr, eps=self.all_args.opti_eps,
                    weight_decay=self.all_args.weight_decay)
                offspring_policy_2.critic_optimizer = torch.optim.Adam(
                    offspring_policy_2.critic.parameters(),
                    lr=self.all_args.critic_lr, eps=self.all_args.opti_eps,
                    weight_decay=self.all_args.weight_decay)
                offspring_trainer_2 = self._create_trainer(offspring_policy_2)
                new_solvers.append(Solver(offspring_policy_2, offspring_trainer_2, merged_tasks,
                                           self.multi_envs, self.num_agents, self.num_enemies, self.num_entities, self.device))

            n_offspring = 2 if self.evo_keep_population else 1
            print(f"    Merged solver {a} (tasks {solver_a.task_ids}) + "
                  f"solver {b} (tasks {solver_b.task_ids}) -> {n_offspring} offspring (tasks {merged_tasks})")

        for l in loners:
            new_solvers.append(self.solvers[l])
            print(f"    Loner solver {l} (tasks {self.solvers[l].task_ids}) survives")

        self.solvers = new_solvers

        if self.solvers:
            self.policy = self.solvers[0].policy
            self.trainer = self.solvers[0].trainer

        return pairs, loners

    # ─── Logging ────────────────────────────────────────────────

    def log_eval(self, eval_infos, total_num_steps):
        for k, v in eval_infos.items():
            if self.use_wandb:
                import wandb
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def _log_generation(self, gen, fitness_matrix, win_rate_matrix, pairs, loners, pre_evo_tasks, elapsed_h):
        """Append generation summary to evolution_log.txt."""
        task_names = list(self.eval_multi_envs)
        mating_fitness = fitness_matrix * win_rate_matrix
        num_solvers = fitness_matrix.shape[0]

        with open(self.evo_log_path, 'a') as f:
            f.write(f"{'='*70}\n")
            f.write(f"Generation {gen}/{self.evo_num_generations}  |  "
                    f"Solvers: {num_solvers}  |  "
                    f"Steps: {self.cumulative_steps}/{self.num_env_steps}  |  "
                    f"Time: {elapsed_h:.2f}h\n")
            f.write(f"{'='*70}\n\n")

            f.write("Task assignments:\n")
            for si in range(num_solvers):
                names = [task_names[t] for t in pre_evo_tasks[si]]
                f.write(f"  Solver {si}: {names}\n")

            f.write(f"\nFitness (reward):\n")
            header = "          " + "".join(f"{t:>14s}" for t in task_names)
            f.write(header + "\n")
            for si in range(num_solvers):
                row = f"  Solver {si}" + "".join(f"{fitness_matrix[si, t]:14.3f}" for t in range(len(task_names)))
                f.write(row + "\n")

            f.write(f"\nWin rate:\n")
            f.write(header + "\n")
            for si in range(num_solvers):
                row = f"  Solver {si}" + "".join(f"{win_rate_matrix[si, t]:14.3f}" for t in range(len(task_names)))
                f.write(row + "\n")

            f.write(f"\nMating fitness (reward x win_rate):\n")
            f.write(header + "\n")
            for si in range(num_solvers):
                row = f"  Solver {si}" + "".join(f"{mating_fitness[si, t]:14.3f}" for t in range(len(task_names)))
                f.write(row + "\n")

            if pairs or loners:
                f.write(f"\nEvolution:\n")
                for (a, b) in pairs:
                    merged = sorted(set(pre_evo_tasks[a] + pre_evo_tasks[b]))
                    merged_names = [task_names[t] for t in merged]
                    f.write(f"  Pair: solver {a} ({[task_names[t] for t in pre_evo_tasks[a]]}) "
                            f"+ solver {b} ({[task_names[t] for t in pre_evo_tasks[b]]}) "
                            f"-> offspring tasks: {merged_names}\n")
                for l in loners:
                    f.write(f"  Loner: solver {l} ({[task_names[t] for t in pre_evo_tasks[l]]}) survives\n")
                f.write(f"  Population after evolution: {len(self.solvers)} solvers\n")
            else:
                f.write(f"\nNo evolution (last generation).\n")

            f.write("\n\n")

    # ─── Checkpoint save/restore ────────────────────────────────

    def _save_sesil_checkpoint(self, generation):
        """Save full population state for resume."""
        ckpt = {
            'generation': generation,
            'cumulative_steps': self.cumulative_steps,
            'num_solvers': len(self.solvers),
            'task_assignments': [s.task_ids for s in self.solvers],
        }
        for i, solver in enumerate(self.solvers):
            ckpt[f'actor_{i}'] = solver.policy.actor.state_dict()
            ckpt[f'critic_{i}'] = solver.policy.critic.state_dict()
            ckpt[f'actor_optimizer_{i}'] = solver.policy.actor_optimizer.state_dict()
            ckpt[f'critic_optimizer_{i}'] = solver.policy.critic_optimizer.state_dict()

        torch.save(ckpt, os.path.join(self.save_dir, 'sesil_checkpoint.pt'))
        if generation == -1:
            torch.save(ckpt, os.path.join(self.save_dir, 'sesil_pretrain.pt'))
        print(f"  Saved SESiL checkpoint at generation {generation}")

    def _restore_sesil_checkpoint(self):
        """Restore population from checkpoint. Returns the generation to resume from."""
        ckpt_path = os.path.join(self.save_dir, 'sesil_checkpoint.pt')
        if not os.path.exists(ckpt_path):
            print("  No SESiL checkpoint found, starting from scratch")
            return 0

        ckpt = torch.load(ckpt_path, map_location=self.device)
        gen = ckpt['generation']
        self.cumulative_steps = ckpt['cumulative_steps']
        num_solvers = ckpt['num_solvers']
        task_assignments = ckpt['task_assignments']

        # Rebuild solvers with restored weights and task assignments
        self.solvers = []
        for i in range(num_solvers):
            policy = self._create_policy()
            policy.actor.load_state_dict(ckpt[f'actor_{i}'])
            policy.critic.load_state_dict(ckpt[f'critic_{i}'])
            policy.actor_optimizer.load_state_dict(ckpt[f'actor_optimizer_{i}'])
            policy.critic_optimizer.load_state_dict(ckpt[f'critic_optimizer_{i}'])

            trainer = self._create_trainer(policy)
            self.solvers.append(Solver(policy, trainer, task_assignments[i],
                                       self.multi_envs, self.num_agents, self.num_enemies, self.num_entities, self.device))

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

        self._pretrain_done = True  # pretraining already completed if any checkpoint exists
        if os.path.exists(self._gen_steps_path):
            with open(self._gen_steps_path) as f:
                self._gen_steps = json.load(f)
        start_gen = max(0, gen + 1)  # gen=-1 means pretraining done, start from gen 0
        print(f"  Resumed SESiL from generation {gen}, continuing from generation {start_gen}, "
              f"cumulative_steps={self.cumulative_steps}")
        return start_gen

    def save(self, episode=None):
        """Override base save — SESiL uses its own checkpoint format."""
        for i, solver in enumerate(self.solvers):
            actor_path = os.path.join(self.save_dir, f'actor_solver{i}.pt')
            torch.save(solver.policy.actor.state_dict(), actor_path)
            critic_path = os.path.join(self.save_dir, f'critic_solver{i}.pt')
            torch.save(solver.policy.critic.state_dict(), critic_path)
