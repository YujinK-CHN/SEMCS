"""
Plot average task performance across algorithms from TensorBoard logs.

Usage:
    python plot_results.py                          # plot all experiment groups
    python plot_results.py --env StarCraft          # filter by env name
    python plot_results.py --metric eval_win_rate   # default metric
    python plot_results.py --metric eval_episode_rewards
    python plot_results.py --output comparison.png  # save to file
"""

import json
import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--env", type=str, default=None,
                        help="Filter by env name substring (e.g. StarCraft, AliceBob)")
    parser.add_argument("--metric", type=str, default="eval_win_rate",
                        choices=["eval_win_rate", "eval_episode_rewards"])
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to file instead of showing")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="Exponential smoothing factor (0=none, 0.9=heavy)")
    parser.add_argument("--max_steps", type=float, default=None,
                        help="Max steps to display (e.g. 5e6 or 5000000)")
    return parser.parse_args()


def get_algorithm_name(run_folder):
    """Extract algorithm name from run folder like 'mcs_skill_..._s1'."""
    return run_folder.split("_")[0]


def get_seed(run_folder):
    """Extract seed from run folder like '..._s30'."""
    match = re.search(r"_s(\d+)$", run_folder)
    return int(match.group(1)) if match else None


def read_tb_scalar(event_dir):
    """Read scalar data from all TensorBoard events files in a directory."""
    if not os.path.isdir(event_dir):
        return None, None
    event_files = sorted(f for f in os.listdir(event_dir) if f.startswith("events.out.tfevents"))
    if not event_files:
        return None, None
    all_steps = []
    all_values = []
    for ef in event_files:
        ea = EventAccumulator(os.path.join(event_dir, ef))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if not tags:
            continue
        scalars = ea.Scalars(tags[0])
        all_steps.extend(s.step for s in scalars)
        all_values.extend(s.value for s in scalars)
    if not all_steps:
        return None, None
    order = np.argsort(all_steps)
    return np.array(all_steps)[order], np.array(all_values)[order]


def interpolate_to_common_steps(all_steps, all_values):
    """Interpolate multiple runs to a common step grid for averaging."""
    if not all_steps:
        return None, None
    min_step = max(s[0] for s in all_steps)
    max_step = min(s[-1] for s in all_steps)
    if min_step >= max_step:
        max_step = max(s[-1] for s in all_steps)
        min_step = min(s[0] for s in all_steps)
    n_points = max(len(s) for s in all_steps)
    common_steps = np.linspace(min_step, max_step, n_points)
    interpolated = []
    for steps, values in zip(all_steps, all_values):
        interp_vals = np.interp(common_steps, steps, values)
        interpolated.append(interp_vals)
    return common_steps, np.array(interpolated)


def smooth(values, weight):
    """Exponential moving average smoothing."""
    if weight <= 0:
        return values
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def main():
    args = parse_args()
    results_dir = args.results_dir

    experiment_groups = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d != "used_ports.txt"
    ]

    if args.env:
        experiment_groups = [g for g in experiment_groups if args.env in g]

    if not experiment_groups:
        print(f"No experiment groups found in {results_dir}" +
              (f" matching '{args.env}'" if args.env else ""))
        return

    for group in sorted(experiment_groups):
        group_dir = os.path.join(results_dir, group)
        runs = [d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))]

        # Extract task names from the group folder: train(task1|task2|...)
        task_match = re.search(r"train\((.+)\)", group)
        if task_match:
            tasks = task_match.group(1).split("|")
        else:
            # Folder format: {env}_train_on_{N} — discover tasks from logged metric dirs
            tasks = set()
            for run in runs:
                logs_dir = os.path.join(group_dir, run, "logs")
                if os.path.isdir(logs_dir):
                    for d in os.listdir(logs_dir):
                        m = re.match(rf"{args.metric}_(.+)", d)
                        if m:
                            tasks.add(m.group(1))
            tasks = sorted(tasks)
            if not tasks:
                print(f"Skipping {group}: can't discover tasks")
                continue

        # Group runs by algorithm
        algo_runs = defaultdict(list)
        for run in runs:
            algo = get_algorithm_name(run)
            algo_runs[algo].append(run)

        print(f"\n{'=' * 60}")
        print(f"Group: {group}")
        print(f"Tasks: {tasks}")
        print(f"Algorithms: {list(algo_runs.keys())}")
        for algo, r in algo_runs.items():
            print(f"  {algo}: {len(r)} seeds")

        n_tasks = len(tasks)
        colors = plt.cm.tab10.colors

        # ── Main figure: per-task + average ──────────────────────
        fig, axes = plt.subplots(1, n_tasks + 1, figsize=(5 * (n_tasks + 1), 4))
        if n_tasks + 1 == 1:
            axes = [axes]

        for algo_idx, (algo, run_list) in enumerate(sorted(algo_runs.items())):
            color = colors[algo_idx % len(colors)]
            per_task_interp = {}

            for task_idx, task in enumerate(tasks):
                metric_name = f"{args.metric}_{task}"
                all_steps = []
                all_values = []

                for run in run_list:
                    event_dir = os.path.join(group_dir, run, "logs", metric_name, metric_name)
                    steps, values = read_tb_scalar(event_dir)
                    if steps is not None and len(steps) > 1:
                        all_steps.append(steps)
                        all_values.append(values)

                if not all_steps:
                    continue

                if args.max_steps is not None:
                    for i in range(len(all_steps)):
                        mask = all_steps[i] <= args.max_steps
                        all_steps[i] = all_steps[i][mask]
                        all_values[i] = all_values[i][mask]
                    all_steps = [s for s in all_steps if len(s) > 1]
                    all_values = [v for v in all_values if len(v) > 1]
                    if not all_steps:
                        continue

                common_steps, interp_matrix = interpolate_to_common_steps(all_steps, all_values)
                if common_steps is None:
                    continue

                per_task_interp[task_idx] = (common_steps, interp_matrix)

                mean_vals = interp_matrix.mean(axis=0)
                std_vals = interp_matrix.std(axis=0)

                if args.smooth > 0:
                    mean_vals = smooth(mean_vals, args.smooth)

                ax = axes[task_idx]
                ax.plot(common_steps, mean_vals, color=color, label=algo, linewidth=1.5)
                ax.fill_between(common_steps, mean_vals - std_vals, mean_vals + std_vals,
                                color=color, alpha=0.15)
                ax.set_title(task, fontsize=11)
                ax.set_xlabel("Steps")
                if task_idx == 0:
                    ax.set_ylabel(args.metric.replace("_", " ").title())
                ax.grid(True, alpha=0.3)

            # Compute average across tasks
            if per_task_interp:
                min_len = min(v[1].shape[1] for v in per_task_interp.values())
                avg_per_seed = []
                for seed_idx in range(max(v[1].shape[0] for v in per_task_interp.values())):
                    task_vals = []
                    for t_idx in per_task_interp:
                        cs, im = per_task_interp[t_idx]
                        if seed_idx < im.shape[0]:
                            task_vals.append(im[seed_idx, :min_len])
                    if task_vals:
                        avg_per_seed.append(np.mean(task_vals, axis=0))

                if avg_per_seed:
                    avg_matrix = np.array(avg_per_seed)
                    ref_steps = list(per_task_interp.values())[0][0][:min_len]
                    mean_avg = avg_matrix.mean(axis=0)
                    std_avg = avg_matrix.std(axis=0)

                    if args.smooth > 0:
                        mean_avg = smooth(mean_avg, args.smooth)

                    ax = axes[-1]
                    ax.plot(ref_steps, mean_avg, color=color, label=algo, linewidth=2)
                    ax.fill_between(ref_steps, mean_avg - std_avg, mean_avg + std_avg,
                                    color=color, alpha=0.15)

        axes[-1].set_title("Average Across Tasks", fontsize=11, fontweight="bold")
        axes[-1].set_xlabel("Steps")
        axes[-1].set_ylabel(args.metric.replace("_", " ").title())
        axes[-1].grid(True, alpha=0.3)

        # Draw generation boundary lines from generation_steps.json (SESiL runs)
        for algo_idx, (algo, run_list) in enumerate(sorted(algo_runs.items())):
            color = colors[algo_idx % len(colors)]
            for run in run_list:
                gen_steps_path = os.path.join(group_dir, run, "generation_steps.json")
                if os.path.exists(gen_steps_path):
                    with open(gen_steps_path) as f:
                        gen_steps = json.load(f)
                    for key, step in sorted(gen_steps.items(), key=lambda x: x[1]):
                        if args.max_steps is not None and step > args.max_steps:
                            continue
                        is_pretrain = key == "pretrain_end"
                        linestyle = '--' if is_pretrain else ':'
                        for ax in axes:
                            ax.axvline(x=step, color=color, linestyle=linestyle, alpha=0.4, linewidth=0.8)
                        label = "pretrain end" if is_pretrain else key.replace("_", " ")
                        axes[0].text(step, axes[0].get_ylim()[1], f" {algo}:{label}",
                                     fontsize=6, color=color, alpha=0.6,
                                     rotation=90, va='top', ha='left')
                    break  # one run per algo is enough (same config across seeds)

        # Add legend to the last subplot
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend(loc="lower right", fontsize=9)

        fig.suptitle(group, fontsize=12, y=1.02)
        plt.tight_layout()

        if args.output:
            base, ext = os.path.splitext(args.output)
            fname = f"{base}_{group}{ext}" if len(experiment_groups) > 1 else args.output
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved to {fname}")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()
