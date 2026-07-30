"""
SEBAL runner — Social Evolutionary Basis Analysis Learning.

SESiL foundation pretrain variant + GLOBA-based merging and mate selection.
Inherits sesilETERunner, overrides pretrain (copies actor+critic),
mate selection (weight-based GLOBA scores), and merging (GLOBA SVD).
"""
import copy
import time
import numpy as np
import torch

from runner.policy.sesil_runner import sesilETERunner, Solver
from base_policy.algorithms.sesil.evolution import bidirectional_selection, build_mating_scores
from base_policy.algorithms.sesil.globa_merge import globa_merge_state_dicts, globa_mating_scores


class SebalRunner(sesilETERunner):
    def __init__(self, config):
        super().__init__(config)
        self._foundation_base_actor = None
        self._foundation_base_critic = None
        self._best_generalist_idx = 0

    def run(self):
        start = time.time()
        self.cumulative_steps = 0

        remaining = self.num_env_steps - self.evo_pretrain_budget
        if self.evo_gen_budget > 0:
            budget_per_gen = self.evo_gen_budget
            self.evo_num_generations = max(1, remaining // budget_per_gen)
            print(f"  evo_gen_budget={budget_per_gen}, derived evo_num_generations={self.evo_num_generations}")
        else:
            budget_per_gen = remaining // self.evo_num_generations

        start_gen = 0

        if getattr(self.all_args, 'resume', False):
            start_gen = self._restore_sesil_checkpoint()
            base_path = self._get_base_path()
            if base_path is not None:
                import os
                base_ckpt = torch.load(base_path, map_location=self.device)
                self._foundation_base_actor = base_ckpt['base_actor']
                self._foundation_base_critic = base_ckpt['base_critic']

        self.next_eval_step = self.cumulative_steps

        if self.evo_pretrain_budget > 0 and not self._pretrain_done:
            print(f"\n{'='*60}")
            print(f"SEBAL Foundation Pretrain: "
                  f"{len(self.solvers)} solvers, budget={self.evo_pretrain_budget} steps")
            self._pretrain_foundation_sebal(self.evo_pretrain_budget)
            self._pretrain_done = True
            self._save_sesil_checkpoint(-1)
            self._save_base_model()
            self._gen_steps["pretrain_end"] = self.cumulative_steps
            self._save_gen_steps()
            print(f"  Pretraining done. Cumulative steps: {self.cumulative_steps}")

        for gen in range(start_gen, self.evo_num_generations):
            if self.cumulative_steps >= self.num_env_steps:
                print(f"\n  Budget exhausted ({self.cumulative_steps}/{self.num_env_steps}), stopping.")
                break

            gen_budget = min(budget_per_gen, self.num_env_steps - self.cumulative_steps)

            print(f"\n{'='*60}")
            print(f"Generation {gen}/{self.evo_num_generations}: "
                  f"{len(self.solvers)} solvers, budget={gen_budget} steps")
            for si, s in enumerate(self.solvers):
                print(f"  Solver {si}: tasks {s.task_ids}")

            self._gen_steps[f"gen_{gen}"] = self.cumulative_steps
            self._save_gen_steps()

            # Train all solvers (periodic eval happens inside, tracks best generalist)
            self._best_generalist_idx = 0
            self._best_generalist_fitness = -float('inf')
            self._train_all_solvers(gen_budget)

            # Evolve (GLOBA mate selection + merge) — skip on last generation
            pre_evo_tasks = [list(s.task_ids) for s in self.solvers]
            pairs, loners = [], []
            if gen < self.evo_num_generations - 1 and len(self.solvers) > 1:
                pairs, loners = self._evolve_globa()

            self._save_sesil_checkpoint(gen)

            elapsed_h = (time.time() - start) / 3600
            print(f"  Cumulative steps: {self.cumulative_steps}/{self.num_env_steps}, "
                  f"Time: {elapsed_h:.2f}h")

            self._log_generation_sebal(gen, pairs, loners, pre_evo_tasks, elapsed_h)

        if self.cumulative_steps < self.num_env_steps:
            print(f"\n  WARNING: All {self.evo_num_generations} generations completed but only "
                  f"{self.cumulative_steps}/{self.num_env_steps} steps used.")

        print(f"\nSEBAL finished. {len(self.solvers)} solver(s) remain.")
        for si, s in enumerate(self.solvers):
            print(f"  Solver {si}: tasks {s.task_ids}")

    def _pretrain_foundation_sebal(self, pretrain_budget):
        """Foundation pretrain that copies both actor AND critic to all solvers."""
        all_task_ids = list(range(self.num_multi_envs))

        tmp_policy = self._create_policy()
        tmp_trainer = self._create_trainer(tmp_policy)
        tmp_solver = Solver(tmp_policy, tmp_trainer, all_task_ids,
                            self.multi_envs, self.num_agents, self.num_enemies,
                            self.num_entities, self.device)

        original_solvers = self.solvers
        self.solvers = [tmp_solver]
        self.policy = tmp_policy
        self.trainer = tmp_trainer

        reward_running_mean = np.zeros(len(all_task_ids))
        reward_running_var = np.ones(len(all_task_ids))
        reward_count = np.zeros(len(all_task_ids))

        from runner.policy.sesil_runner import FilteredBuffer
        n_tasks = len(all_task_ids)
        spe = self._steps_per_episode(n_tasks)
        num_episodes = max(1, pretrain_budget // spe)
        filtered_buf = FilteredBuffer(self.buffer, all_task_ids)

        print(f"  Foundation pretrain: {num_episodes} episodes on all tasks {all_task_ids}")

        for episode in range(num_episodes):
            self._warmup_tasks(all_task_ids)
            self._collect_episode(tmp_solver)

            for tid in all_task_ids:
                raw = self.buffer.buffer_lists[tid].rewards
                flat = raw.flatten()
                batch_mean = flat.mean()
                batch_var = flat.var()
                batch_n = len(flat)
                old_count = reward_count[tid]
                new_count = old_count + batch_n
                delta = batch_mean - reward_running_mean[tid]
                reward_running_mean[tid] += delta * batch_n / max(new_count, 1)
                m_a = reward_running_var[tid] * old_count
                m_b = batch_var * batch_n
                M2 = m_a + m_b + delta ** 2 * old_count * batch_n / max(new_count, 1)
                reward_running_var[tid] = M2 / max(new_count, 1)
                reward_count[tid] = new_count

            normalized = []
            for tid in all_task_ids:
                raw = self.buffer.buffer_lists[tid].rewards
                std = np.sqrt(reward_running_var[tid] + 1e-8)
                normalized.append((raw - reward_running_mean[tid]) / std)

            blended = np.mean(normalized, axis=0) if len(set(
                n.shape for n in normalized)) == 1 else None

            if blended is not None:
                for tid in all_task_ids:
                    self.buffer.buffer_lists[tid].rewards = blended.copy()
            else:
                mean_normalized = np.mean([n.mean() for n in normalized])
                for tid in all_task_ids:
                    raw = self.buffer.buffer_lists[tid].rewards
                    self.buffer.buffer_lists[tid].rewards = np.full_like(raw, mean_normalized)

            self._compute_filtered(tmp_solver, filtered_buf)
            tmp_solver.trainer.prep_training()
            tmp_solver.trainer.train(filtered_buf, episode)
            filtered_buf.after_update()
            self.cumulative_steps += spe

            if self.cumulative_steps >= self.next_eval_step:
                self._eval_and_log_best_generalist(self.cumulative_steps)
                self.next_eval_step += self.eval_steps_interval

            if (episode + 1) % max(1, num_episodes // 10) == 0:
                print(f"    Episode {episode+1}/{num_episodes}, "
                      f"cumulative={self.cumulative_steps}")

        # Save foundation base (both actor and critic)
        self._foundation_base_actor = {k: v.clone() for k, v in tmp_policy.actor.state_dict().items()}
        self._foundation_base_critic = {k: v.clone() for k, v in tmp_policy.critic.state_dict().items()}

        # Restore solvers and copy full model (actor + critic)
        self.solvers = original_solvers
        actor_sd = tmp_policy.actor.state_dict()
        critic_sd = tmp_policy.critic.state_dict()
        for si, solver in enumerate(self.solvers):
            solver.policy.actor.load_state_dict(actor_sd)
            solver.policy.actor_optimizer = torch.optim.Adam(
                solver.policy.actor.parameters(),
                lr=self.all_args.lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)
            solver.policy.critic.load_state_dict(critic_sd)
            solver.policy.critic_optimizer = torch.optim.Adam(
                solver.policy.critic.parameters(),
                lr=self.all_args.critic_lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)
            print(f"  Copied foundation actor+critic to solver {si}")

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

    def _train_all_solvers(self, gen_budget):
        """Override to track best generalist during periodic evaluations."""
        total_tasks = sum(len(s.task_ids) for s in self.solvers)

        for si, solver in enumerate(self.solvers):
            self.policy = solver.policy
            self.trainer = solver.trainer
            self.trainer.policy = solver.policy
            from runner.policy.sesil_runner import FilteredBuffer
            filtered_buf = FilteredBuffer(self.buffer, solver.task_ids)
            n_tasks = len(solver.task_ids)
            spe = self._steps_per_episode(n_tasks)

            solver_budget = int(gen_budget * n_tasks / total_tasks)
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
                    self._eval_and_log_best_generalist(self.cumulative_steps)
                    self.next_eval_step += self.eval_steps_interval

                if (episode + 1) % max(1, episodes_per_solver // 5) == 0:
                    print(f"    Solver {si} episode {episode+1}/{episodes_per_solver}, "
                          f"cumulative: {self.cumulative_steps}")

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

    def _eval_and_log_best_generalist(self, total_num_steps):
        """Evaluate all solvers, log metrics, and track best generalist."""
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

        # Track best generalist
        avg_reward_per_solver = fitness_matrix.mean(axis=1)
        current_best = int(np.argmax(avg_reward_per_solver))
        if avg_reward_per_solver[current_best] > self._best_generalist_fitness:
            self._best_generalist_fitness = avg_reward_per_solver[current_best]
            self._best_generalist_idx = current_best

        # Log best generalist's per-task performance
        best = self._best_generalist_idx
        for task_idx in range(K):
            task_name = self.eval_multi_envs[task_idx]
            eval_infos = {f'eval_episode_rewards_{task_name}': fitness_matrix[best, task_idx]}
            if "StarCraft" in self.env_name:
                eval_infos[f'eval_win_rate_{task_name}'] = win_rate_matrix[best, task_idx]
            elif "AliceBob" in self.env_name or "Football" in self.env_name:
                eval_infos[f'eval_win_rate_{task_name}'] = fitness_matrix[best, task_idx]
            self.log_eval(eval_infos, total_num_steps)

        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer

    def _evaluate_population_no_log(self):
        """Evaluate all solvers on all tasks, return fitness/win_rate matrices without logging."""
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
        self.policy = self.solvers[0].policy
        self.trainer = self.solvers[0].trainer
        return fitness_matrix, win_rate_matrix

    def _evolve_globa(self):
        """GLOBA-based mate selection + merge."""
        M = len(self.solvers)
        if M <= 1:
            return [], list(range(M))

        assert self._foundation_base_actor is not None, "Foundation base not set — SEBAL requires foundation pretrain."

        base_actor_sd = self._foundation_base_actor
        base_critic_sd = self._foundation_base_critic

        # Compute GLOBA mating scores from actor weights
        solver_actor_sds = [s.policy.actor.state_dict() for s in self.solvers]
        globa_scores = globa_mating_scores(base_actor_sd, solver_actor_sds, self.all_args)

        if self.all_args.globa_use_task_scores:
            # Run evaluation to get fitness matrix for task-based scores (no logging — periodic eval already covers it)
            fitness_matrix, win_rate_matrix = self._evaluate_population_no_log()
            mating_fitness = fitness_matrix * win_rate_matrix
            task_scores = build_mating_scores(mating_fitness, self.evo_threshold,
                                              self.evo_weight_extra, self.evo_weight_common)

            # Normalize both score sets to [0,1] range before combining
            def _normalize_scores(raw_scores):
                all_vals = [v for inner in raw_scores.values() for v in inner.values()]
                if not all_vals:
                    return raw_scores
                max_v = max(all_vals)
                if max_v <= 0:
                    return raw_scores
                return {i: {j: v / max_v for j, v in inner.items()} for i, inner in raw_scores.items()}

            norm_task = _normalize_scores(task_scores)
            norm_globa = _normalize_scores(globa_scores)

            coef_t = self.all_args.globa_coef_task
            coef_w = self.all_args.globa_coef_weight
            scores = {}
            for i in norm_globa:
                scores[i] = {}
                for j in norm_globa[i]:
                    t_val = norm_task.get(i, {}).get(j, 0.0)
                    g_val = norm_globa[i][j]
                    scores[i][j] = max(coef_t * t_val + coef_w * g_val, 0.0)
            print(f"  [SEBAL] Combined mating scores (coef_task={coef_t}, coef_weight={coef_w})")
        else:
            scores = globa_scores

        pairs, loners = bidirectional_selection(scores)

        print(f"  [SEBAL Evo] pairs: {pairs}, loners: {loners}")

        new_solvers = []

        for (a, b) in pairs:
            solver_a = self.solvers[a]
            solver_b = self.solvers[b]
            merged_tasks = sorted(set(solver_a.task_ids + solver_b.task_ids))

            # Merge actor via GLOBA
            merged_actor_sd = globa_merge_state_dicts(
                base_actor_sd,
                solver_a.policy.actor.state_dict(),
                solver_b.policy.actor.state_dict(),
                self.all_args)

            # Merge critic via GLOBA
            merged_critic_sd = globa_merge_state_dicts(
                base_critic_sd,
                solver_a.policy.critic.state_dict(),
                solver_b.policy.critic.state_dict(),
                self.all_args)

            offspring_policy = self._create_policy()
            offspring_policy.actor.load_state_dict(merged_actor_sd)
            offspring_policy.actor_optimizer = torch.optim.Adam(
                offspring_policy.actor.parameters(),
                lr=self.all_args.lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)
            offspring_policy.critic.load_state_dict(merged_critic_sd)
            offspring_policy.critic_optimizer = torch.optim.Adam(
                offspring_policy.critic.parameters(),
                lr=self.all_args.critic_lr, eps=self.all_args.opti_eps,
                weight_decay=self.all_args.weight_decay)

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

        # Update base model if rolling mode
        if self.all_args.globa_base_mode == "rolling" and new_solvers:
            best_idx = self._best_generalist_idx
            if best_idx < len(self.solvers):
                self._foundation_base_actor = {k: v.clone() for k, v in self.solvers[best_idx].policy.actor.state_dict().items()}
                self._foundation_base_critic = {k: v.clone() for k, v in self.solvers[best_idx].policy.critic.state_dict().items()}
                self._save_base_model()
                print(f"    [Rolling] Updated base model to solver {best_idx} (best generalist)")

        self.solvers = new_solvers
        if self.solvers:
            self.policy = self.solvers[0].policy
            self.trainer = self.solvers[0].trainer

        return pairs, loners

    # ─── Base model persistence ────────────────────────────────

    def _get_base_path(self):
        import os
        path = os.path.join(self.save_dir, 'sebal_base.pt')
        return path if os.path.exists(path) else None

    def _save_base_model(self):
        import os
        torch.save({
            'base_actor': self._foundation_base_actor,
            'base_critic': self._foundation_base_critic,
        }, os.path.join(self.save_dir, 'sebal_base.pt'))

    # ─── Logging ───────────────────────────────────────────────

    def _log_generation_sebal(self, gen, pairs, loners, pre_evo_tasks, elapsed_h):
        """Append generation summary to evolution_log.txt."""
        task_names = list(self.eval_multi_envs)
        num_solvers = len(pre_evo_tasks)

        with open(self.evo_log_path, 'a') as f:
            f.write(f"{'='*70}\n")
            f.write(f"SEBAL Generation {gen}/{self.evo_num_generations}  |  "
                    f"Solvers: {num_solvers}  |  "
                    f"Steps: {self.cumulative_steps}/{self.num_env_steps}  |  "
                    f"Time: {elapsed_h:.2f}h\n")
            f.write(f"Base mode: {self.all_args.globa_base_mode}\n")
            f.write(f"{'='*70}\n\n")

            f.write("Task assignments:\n")
            for si in range(num_solvers):
                names = [task_names[t] for t in pre_evo_tasks[si]]
                f.write(f"  Solver {si}: {names}\n")

            if pairs or loners:
                f.write(f"\nEvolution (GLOBA merge):\n")
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
