"""
Newskill experiment runner — comparing MAPPO vs SESiL on integrating unknown tasks.

Two variants:
  - newskill_mappo: Phase 1 trains MAPPO on known tasks, Phase 2 trains on all tasks.
  - newskill_sesil_mappo: Phase 1 runs SESiL evolution on known tasks + trains
    outlander on unknown tasks. Phase 2 injects outlander into population,
    continues evolving on all tasks.

Both variants log a phase boundary marker for plotting.
"""
import json
import time
import os
import numpy as np

from runner.policy.base_runner import Runner
from runner.policy.sesil_runner import (
    sesilETERunner, FilteredBuffer, Solver
)
from base_policy.utils.multi_envs_shared_buffer import MultiEnvSharedReplayBufferComm
from base_policy.algorithms.sesil.sesil_policy import sesilPolicy as MappoPolicy
from base_policy.algorithms.mcs.mcs_trainer import mcsTrainer as McsTrainer


class NewskillRunner:
    """Dispatcher that picks the right runner based on newskill_variant."""
    def __new__(cls, config):
        variant = config["all_args"].newskill_variant
        if variant == "mappo":
            return NewskillMappoRunner(config)
        elif variant == "sesil_mappo":
            return NewskillSesilRunner(config)
        else:
            raise ValueError(f"Unknown newskill_variant: {variant}")


class NewskillMappoRunner(Runner):
    """MAPPO baseline for newskill: Phase 1 on known tasks, Phase 2 on all tasks."""

    def __init__(self, config):
        super().__init__(config)

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

        # Newskill params
        self.phase1_budget = self.all_args.newskill_phase1_budget
        self.phase2_budget = self.all_args.newskill_phase2_budget
        self.n_known = self.all_args.newskill_known_tasks
        self.n_unknown = self.all_args.newskill_unknown_tasks
        self.known_task_ids = list(range(self.n_known))
        self.unknown_task_ids = list(range(self.n_known, self.n_known + self.n_unknown))
        self.all_task_ids = list(range(self.num_multi_envs))

        # Policy (MAPPO with entity obs)
        self.policy = MappoPolicy(
            self.all_args, self.multi_envs, self.num_thread_per_env,
            self.envs.observation_space, self.share_observation_space,
            self.envs.action_space, device=self.device)

        self.trainer = McsTrainer(
            self.all_args, self.policy, self.num_agents, self.num_enemies,
            self.num_entities, device=self.device)

        self.buffer = MultiEnvSharedReplayBufferComm(
            self.all_args, self.num_agents, self.num_entities,
            self.obs_space_list, self.cent_obs_space_list, self.act_space_list,
            self.num_thread_per_env)

        self._phase_steps_path = os.path.join(self.run_dir, 'generation_steps.json')

    def run(self):
        start = time.time()
        self.cumulative_steps = 0
        self.next_eval_step = 0
        phase_steps = {}

        # Phase 1: train on known tasks only
        print(f"\n{'='*60}")
        print(f"Phase 1: MAPPO training on {self.n_known} known tasks, "
              f"budget={self.phase1_budget}")
        self._train_on_tasks(self.known_task_ids, self.phase1_budget)
        phase_steps["phase1_end"] = self.cumulative_steps

        # Phase 2: train on ALL tasks
        print(f"\n{'='*60}")
        print(f"Phase 2: MAPPO training on all {self.num_multi_envs} tasks, "
              f"budget={self.phase2_budget}")
        self._train_on_tasks(self.all_task_ids, self.phase2_budget)
        phase_steps["phase2_end"] = self.cumulative_steps

        with open(self._phase_steps_path, 'w') as f:
            json.dump(phase_steps, f, indent=2)

        elapsed_h = (time.time() - start) / 3600
        print(f"\nNewskill MAPPO finished. Total steps: {self.cumulative_steps}, "
              f"Time: {elapsed_h:.2f}h")

    def _train_on_tasks(self, task_ids, budget):
        """Train the single MAPPO policy on given task subset for budget steps."""
        n_tasks = len(task_ids)
        spe = self.episode_length * n_tasks * self.num_thread_per_env
        episodes = max(1, budget // spe)

        # Create a Solver-like wrapper to reuse sesil's collect/compute
        solver = _SimpleSolver(self.policy, self.trainer, task_ids,
                               self.multi_envs, self.num_agents,
                               self.num_enemies, self.num_entities)
        filtered_buf = FilteredBuffer(self.buffer, task_ids)

        for episode in range(episodes):
            self._warmup_tasks(task_ids)
            self._collect_episode(solver)
            self._compute_filtered(solver, filtered_buf)
            solver.trainer.prep_training()
            solver.trainer.train(filtered_buf, episode)
            filtered_buf.after_update()
            self.cumulative_steps += spe

            if self.cumulative_steps >= self.next_eval_step:
                self._eval_and_log(self.cumulative_steps)
                self.next_eval_step += self.eval_steps_interval

            if (episode + 1) % max(1, episodes // 10) == 0:
                print(f"  Episode {episode+1}/{episodes}, "
                      f"cumulative: {self.cumulative_steps}")

        self.save(episode=episode)

    # Reuse sesil_runner's low-level methods
    _warmup_tasks = sesilETERunner._warmup_tasks
    _collect_episode = sesilETERunner._collect_episode
    _compute_filtered = sesilETERunner._compute_filtered
    _insert = sesilETERunner._insert
    _eval_solver_on_all_tasks = sesilETERunner._eval_solver_on_all_tasks

    def _eval_and_log(self, total_num_steps):
        """Evaluate on all tasks and log."""
        task_rewards, task_win_rates = self._eval_solver_on_all_tasks()
        for task_idx in range(self.num_multi_envs):
            task_name = self.eval_multi_envs[task_idx]
            eval_infos = {f'eval_episode_rewards_{task_name}': task_rewards[task_idx]}
            if "StarCraft" in self.env_name:
                eval_infos[f'eval_win_rate_{task_name}'] = task_win_rates[task_idx]
            elif "AliceBob" in self.env_name or "Football" in self.env_name:
                eval_infos[f'eval_win_rate_{task_name}'] = task_rewards[task_idx]
            self.log_eval(eval_infos, total_num_steps)

    def log_eval(self, eval_infos, total_num_steps):
        for k, v in eval_infos.items():
            if self.use_wandb:
                import wandb
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)


class _SimpleSolver:
    """Lightweight solver wrapper for NewskillMappoRunner (no evolution needed)."""
    def __init__(self, policy, trainer, task_ids, all_multi_envs, all_num_agents, all_num_enemies, all_num_entities):
        self.policy = policy
        self.trainer = trainer
        self.task_ids = list(task_ids)
        self.trainer.num_multi_envs = len(task_ids)
        self.trainer.multi_envs = [all_multi_envs[i] for i in task_ids]
        self.trainer.n_agents_list = [all_num_agents[i] for i in task_ids]
        self.trainer.n_enemies_list = [all_num_enemies[i] for i in task_ids]
        self.trainer.n_entities_list = [all_num_entities[i] for i in task_ids]


class NewskillSesilRunner(sesilETERunner):
    """SESiL variant for newskill: evolve on known tasks, then inject outlander."""

    def __init__(self, config):
        super().__init__(config)

        self.phase1_budget = self.all_args.newskill_phase1_budget
        self.phase2_budget = self.all_args.newskill_phase2_budget
        self.n_known = self.all_args.newskill_known_tasks
        self.n_unknown = self.all_args.newskill_unknown_tasks
        self.known_task_ids = list(range(self.n_known))
        self.unknown_task_ids = list(range(self.n_known, self.n_known + self.n_unknown))
        self.all_task_ids = list(range(self.num_multi_envs))

    def _train_outlander(self, budget):
        """Train the outlander solver with outlander-specific logging."""
        solver = self.solvers[0]
        self.policy = solver.policy
        self.trainer = solver.trainer
        self.trainer.policy = solver.policy
        filtered_buf = FilteredBuffer(self.buffer, solver.task_ids)
        n_tasks = len(solver.task_ids)
        spe = self.episode_length * n_tasks * self.num_thread_per_env
        episodes = max(1, budget // spe)

        for episode in range(episodes):
            self._warmup_tasks(solver.task_ids)
            self._collect_episode(solver)
            self._compute_filtered(solver, filtered_buf)
            solver.trainer.prep_training()
            solver.trainer.train(filtered_buf, episode)
            filtered_buf.after_update()
            self.cumulative_steps += spe

            if (episode + 1) % max(1, episodes // 5) == 0:
                print(f"    Outlander episode {episode+1}/{episodes}, "
                      f"outlander_steps: {self.cumulative_steps}")

    def _pretrain_on_known_tasks(self, pretrain_budget):
        """Pretrain using only known tasks by temporarily narrowing num_multi_envs."""
        saved = self.num_multi_envs
        self.num_multi_envs = self.n_known
        if self.evo_pretrain_mode == "full":
            self._pretrain_full(pretrain_budget)
        elif self.evo_pretrain_mode == "encoder":
            self._pretrain_encoder(pretrain_budget)
        elif self.evo_pretrain_mode == "foundation":
            self._pretrain_foundation(pretrain_budget)
        self.num_multi_envs = saved

    def run(self):
        start = time.time()
        self.cumulative_steps = 0

        remaining_p1 = self.phase1_budget - self.evo_pretrain_budget
        if self.evo_gen_budget > 0:
            budget_per_gen = self.evo_gen_budget
            phase1_gens = max(1, remaining_p1 // budget_per_gen)
        else:
            phase1_gens = self.evo_num_generations // 2 or 1
            budget_per_gen = remaining_p1 // phase1_gens

        self.next_eval_step = 0

        # ── Assign each solver exactly tasks_per_solver known tasks, guaranteeing full coverage ──
        import random as _rng
        rng = _rng.Random(self.all_args.seed)
        tps = min(self.evo_tasks_per_solver, len(self.known_task_ids))
        assignments = [[] for _ in range(len(self.solvers))]
        # First pass: deal all known tasks round-robin to guarantee coverage
        deck = list(self.known_task_ids)
        rng.shuffle(deck)
        for i, t in enumerate(deck):
            assignments[i % len(self.solvers)].append(t)
        # Second pass: fill each solver up to tasks_per_solver with random picks
        for i in range(len(self.solvers)):
            while len(assignments[i]) < tps:
                candidates = [t for t in self.known_task_ids if t not in assignments[i]]
                if not candidates:
                    break
                assignments[i].append(rng.choice(candidates))
            assignments[i] = sorted(assignments[i])
        for solver, task_ids in zip(self.solvers, assignments):
            solver.task_ids = task_ids
            solver._sync_trainer(self.multi_envs, self.num_agents, self.num_enemies, self.num_entities)

        # ── Pretraining (on known tasks only) ──
        if self.evo_pretrain_budget > 0 and not self._pretrain_done:
            print(f"\n{'='*60}")
            print(f"Phase 1 Pretrain ({self.evo_pretrain_mode}): "
                  f"{len(self.solvers)} solvers, budget={self.evo_pretrain_budget}")
            if self.evo_pretrain_mode in ("encoder", "full", "foundation"):
                # Parent pretrain methods use all tasks — override to use known only
                self._pretrain_on_known_tasks(self.evo_pretrain_budget)
            else:
                self._train_all_solvers(self.evo_pretrain_budget)
            self._gen_steps["pretrain_end"] = self.cumulative_steps
            self._save_gen_steps()

        # ── Train outlander on unknown tasks (separate budget) ──
        print(f"\n{'='*60}")
        print(f"Outlander Training (separate from main budget, not counted in cumulative steps)")
        outlander_budget = self.evo_gen_budget if self.evo_gen_budget > 0 else (self.phase1_budget - self.evo_pretrain_budget) // max(1, self.evo_num_generations)
        print(f"  Tasks: {self.unknown_task_ids}, budget: {outlander_budget} steps")
        outlander_policy = self._create_policy()
        outlander_trainer = self._create_trainer(outlander_policy)
        outlander = Solver(outlander_policy, outlander_trainer, self.unknown_task_ids,
                           self.multi_envs, self.num_agents, self.num_enemies,
                           self.num_entities, self.device)

        original_solvers = self.solvers
        saved_steps = self.cumulative_steps
        saved_next_eval = self.next_eval_step
        self.cumulative_steps = 0
        self.next_eval_step = float('inf')
        self.solvers = [outlander]
        self._train_outlander(outlander_budget)
        self.cumulative_steps = saved_steps
        self.next_eval_step = saved_next_eval
        self.solvers = original_solvers
        print(f"  Outlander training done. Resuming main cumulative_steps={self.cumulative_steps}")

        # ── Phase 1: SESiL evolution on known tasks ──
        print(f"\n{'='*60}")
        print(f"Phase 1: SESiL evolution on known tasks, "
              f"{phase1_gens} generations, budget_per_gen={budget_per_gen}")

        for gen in range(phase1_gens):
            if self.cumulative_steps >= self.phase1_budget:
                break

            gen_budget = min(budget_per_gen,
                             self.phase1_budget - self.cumulative_steps)

            print(f"\n  Gen {gen}/{phase1_gens}: {len(self.solvers)} solvers, "
                  f"budget={gen_budget}")
            for si, s in enumerate(self.solvers):
                print(f"    Solver {si}: tasks {s.task_ids}")

            self._gen_steps[f"phase1_gen_{gen}"] = self.cumulative_steps
            self._save_gen_steps()

            self._train_all_solvers(gen_budget)

            fitness_matrix, win_rate_matrix = self._evaluate_population(self.cumulative_steps)
            print(f"  Fitness:\n{np.array2string(fitness_matrix, precision=3)}")

            if gen < phase1_gens - 1 and len(self.solvers) > 1:
                self._evolve(fitness_matrix, win_rate_matrix)

        phase1_end = self.cumulative_steps
        self._gen_steps["phase1_end"] = phase1_end
        self._save_gen_steps()

        # ── Phase 2: Inject outlander, evolve on ALL tasks ──
        print(f"\n{'='*60}")
        print(f"Phase 2: Injecting outlander, evolving on all {self.num_multi_envs} tasks")

        # Expand all existing solvers to cover all tasks
        for solver in self.solvers:
            solver.task_ids = list(self.all_task_ids)
            solver._sync_trainer(self.multi_envs, self.num_agents, self.num_enemies, self.num_entities)

        # Inject outlander into population with all tasks
        outlander.task_ids = list(self.all_task_ids)
        outlander._sync_trainer(self.multi_envs, self.num_agents, self.num_enemies, self.num_entities)
        if self.evo_keep_population:
            import random
            replace_idx = random.randrange(len(self.solvers))
            print(f"  Replacing solver {replace_idx} with outlander (keep_population=1)")
            self.solvers[replace_idx] = outlander
        else:
            self.solvers.append(outlander)
        print(f"  Population size after injection: {len(self.solvers)}")

        phase2_gens = max(1, self.phase2_budget // budget_per_gen) if budget_per_gen > 0 else 1

        for gen in range(phase2_gens):
            remaining = self.phase1_budget + self.phase2_budget - self.cumulative_steps
            if remaining <= 0:
                break

            gen_budget = min(budget_per_gen, remaining)

            print(f"\n  Phase 2 Gen {gen}/{phase2_gens}: {len(self.solvers)} solvers, "
                  f"budget={gen_budget}")

            self._gen_steps[f"phase2_gen_{gen}"] = self.cumulative_steps
            self._save_gen_steps()

            self._train_all_solvers(gen_budget)

            fitness_matrix, win_rate_matrix = self._evaluate_population(self.cumulative_steps)
            print(f"  Fitness:\n{np.array2string(fitness_matrix, precision=3)}")

            if gen < phase2_gens - 1 and len(self.solvers) > 1:
                self._evolve(fitness_matrix, win_rate_matrix)

        self._gen_steps["phase2_end"] = self.cumulative_steps
        self._save_gen_steps()

        elapsed_h = (time.time() - start) / 3600
        print(f"\nNewskill SESiL finished. {len(self.solvers)} solver(s) remain. "
              f"Total steps: {self.cumulative_steps}, Time: {elapsed_h:.2f}h")

