"""
Evaluate a trained model on held-out WebArena-Lite tasks.

Runs the model on k attempts per task, computes success rate
with bootstrap 95% confidence intervals.

Usage:
    python evaluate.py \
        --model checkpoints/model_A \
        --tasks held_out_tasks.json \
        --k 3 \
        --output results/model_A.json
"""

import argparse
import json
import os
import sys
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime


def bootstrap_ci(successes: List[bool], n_bootstrap: int = 10000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute success rate with bootstrap confidence interval.

    Args:
        successes: List of booleans (True = task succeeded).
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (0.95 = 95% CI).

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if len(successes) == 0:
        return 0.0, 0.0, 0.0

    arr = np.array(successes, dtype=float)
    mean = float(arr.mean())

    if len(successes) < 2:
        return mean, mean, mean

    rng = np.random.default_rng(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(sample.mean())

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - ci
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return mean, ci_lower, ci_upper


def pass_at_k(attempts: List[bool], k: int) -> bool:
    """
    pass@k: Did at least one of k attempts succeed?
    """
    return any(attempts[:k])


def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute evaluation metrics from raw results.

    Args:
        results: List of {task_id, site, attempts: [bool], rewards: [float]}

    Returns:
        Dict with metrics.
    """
    # Per-task pass@k
    task_successes = [pass_at_k(r['attempts'], len(r['attempts'])) for r in results]
    mean, ci_low, ci_high = bootstrap_ci(task_successes)

    # Per-attempt success rate
    all_attempts = []
    for r in results:
        all_attempts.extend(r['attempts'])
    attempt_rate = float(np.mean(all_attempts)) if all_attempts else 0.0

    # Per-site breakdown
    sites = set(r.get('site', 'unknown') for r in results)
    per_site = {}
    for site in sorted(sites):
        site_results = [r for r in results if r.get('site', 'unknown') == site]
        site_successes = [pass_at_k(r['attempts'], len(r['attempts'])) for r in site_results]
        s_mean, s_low, s_high = bootstrap_ci(site_successes)
        per_site[site] = {
            'n_tasks': len(site_results),
            'pass_at_k': s_mean,
            'ci_95': [s_low, s_high],
        }

    return {
        'n_tasks': len(results),
        'n_attempts_per_task': len(results[0]['attempts']) if results else 0,
        'pass_at_k': mean,
        'ci_95': [ci_low, ci_high],
        'per_attempt_success_rate': attempt_rate,
        'per_site': per_site,
    }


def compute_metrics_with_training(results: List[Dict], training_meta: Dict = None) -> Dict:
    """
    Compute evaluation metrics plus training metadata.

    Args:
        results: List of task result dicts.
        training_meta: Optional dict with training info:
            - training_tasks: int (number of tasks used for training)
            - training_trajectories: int
            - training_steps: int (total gradient steps)
            - training_time_seconds: float
            - data_processing_time_seconds: float
            - condition: str ("A" or "C")

    Returns:
        Combined metrics dict.
    """
    metrics = compute_metrics(results)

    if training_meta:
        metrics['training'] = {
            'condition': training_meta.get('condition', 'unknown'),
            'n_training_tasks': training_meta.get('training_tasks', 0),
            'n_training_trajectories': training_meta.get('training_trajectories', 0),
            'n_training_steps': training_meta.get('training_steps', 0),
            'training_time_seconds': training_meta.get('training_time_seconds', 0),
            'training_time_human': _format_duration(
                training_meta.get('training_time_seconds', 0)),
            'data_processing_time_seconds': training_meta.get(
                'data_processing_time_seconds', 0),
            'data_processing_time_human': _format_duration(
                training_meta.get('data_processing_time_seconds', 0)),
            'total_time_seconds': (
                training_meta.get('training_time_seconds', 0) +
                training_meta.get('data_processing_time_seconds', 0)),
            'total_time_human': _format_duration(
                training_meta.get('training_time_seconds', 0) +
                training_meta.get('data_processing_time_seconds', 0)),
        }

    return metrics


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}hr"


def format_results_table(metrics: Dict) -> str:
    """Pretty-print metrics as a table."""
    lines = []
    lines.append(f"Tasks: {metrics['n_tasks']}, "
                 f"Attempts/task: {metrics['n_attempts_per_task']}")
    lines.append(f"pass@{metrics['n_attempts_per_task']}: "
                 f"{metrics['pass_at_k']:.1%} "
                 f"[{metrics['ci_95'][0]:.1%}, {metrics['ci_95'][1]:.1%}]")
    lines.append(f"Per-attempt success: {metrics['per_attempt_success_rate']:.1%}")

    if 'training' in metrics:
        t = metrics['training']
        lines.append("")
        lines.append("Training:")
        lines.append(f"  Tasks used: {t['n_training_tasks']}")
        lines.append(f"  Trajectories: {t['n_training_trajectories']}")
        lines.append(f"  Gradient steps: {t['n_training_steps']}")
        lines.append(f"  Data processing: {t['data_processing_time_human']}")
        lines.append(f"  Training time: {t['training_time_human']}")
        lines.append(f"  Total time: {t['total_time_human']}")

    lines.append("")
    lines.append("Per-site breakdown:")
    for site, data in metrics['per_site'].items():
        lines.append(f"  {site}: {data['pass_at_k']:.1%} "
                     f"[{data['ci_95'][0]:.1%}, {data['ci_95'][1]:.1%}] "
                     f"(n={data['n_tasks']})")
    return "\n".join(lines)


def compare_results(results_a: Dict, results_c: Dict) -> str:
    """Print a comparison table between two conditions."""
    lines = []
    lines.append(f"{'Metric':<30} {'A (WebRL)':>12} {'C (Ours)':>12} {'Delta':>10}")
    lines.append("-" * 66)

    delta = results_c['pass_at_k'] - results_a['pass_at_k']
    k = results_a['n_attempts_per_task']
    lines.append(
        f"{'pass@' + str(k):<30} "
        f"{results_a['pass_at_k']:>11.1%} "
        f"{results_c['pass_at_k']:>11.1%} "
        f"{delta:>+9.1%}"
    )

    delta_a = results_c['per_attempt_success_rate'] - results_a['per_attempt_success_rate']
    lines.append(
        f"{'Per-attempt success':<30} "
        f"{results_a['per_attempt_success_rate']:>11.1%} "
        f"{results_c['per_attempt_success_rate']:>11.1%} "
        f"{delta_a:>+9.1%}"
    )

    # Training stats (if available)
    t_a = results_a.get('training', {})
    t_c = results_c.get('training', {})
    if t_a or t_c:
        lines.append("")
        lines.append(f"{'Training':<30} {'A (WebRL)':>12} {'C (Ours)':>12} {'Delta':>10}")
        lines.append("-" * 66)

        a_tasks = t_a.get('n_training_tasks', 0)
        c_tasks = t_c.get('n_training_tasks', 0)
        lines.append(
            f"{'Tasks used':<30} {a_tasks:>12} {c_tasks:>12} {c_tasks - a_tasks:>+10}"
        )

        a_traj = t_a.get('n_training_trajectories', 0)
        c_traj = t_c.get('n_training_trajectories', 0)
        lines.append(
            f"{'Trajectories':<30} {a_traj:>12} {c_traj:>12} {c_traj - a_traj:>+10}"
        )

        a_steps = t_a.get('n_training_steps', 0)
        c_steps = t_c.get('n_training_steps', 0)
        lines.append(
            f"{'Gradient steps':<30} {a_steps:>12} {c_steps:>12} {c_steps - a_steps:>+10}"
        )

        a_time = t_a.get('training_time_human', 'N/A')
        c_time = t_c.get('training_time_human', 'N/A')
        lines.append(
            f"{'Training time':<30} {a_time:>12} {c_time:>12}"
        )

        a_total = t_a.get('total_time_human', 'N/A')
        c_total = t_c.get('total_time_human', 'N/A')
        lines.append(
            f"{'Total time (data+train)':<30} {a_total:>12} {c_total:>12}"
        )

    # Per-site
    lines.append("")
    lines.append("Per-site:")
    all_sites = sorted(set(list(results_a['per_site'].keys()) +
                           list(results_c['per_site'].keys())))
    for site in all_sites:
        a_val = results_a['per_site'].get(site, {}).get('pass_at_k', 0)
        c_val = results_c['per_site'].get(site, {}).get('pass_at_k', 0)
        lines.append(
            f"  {site:<28} {a_val:>11.1%} {c_val:>11.1%} {c_val - a_val:>+9.1%}"
        )

    return "\n".join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate models or compare results')
    subparsers = parser.add_subparsers(dest='command')

    # Compute metrics from raw results
    metrics_parser = subparsers.add_parser('metrics', help='Compute metrics from results JSON')
    metrics_parser.add_argument('--results', type=str, required=True,
                                help='Path to results JSON file')

    # Compare two conditions
    compare_parser = subparsers.add_parser('compare', help='Compare two result files')
    compare_parser.add_argument('--a', type=str, required=True, help='Condition A results')
    compare_parser.add_argument('--c', type=str, required=True, help='Condition C results')

    # Generate dummy results for testing
    dummy_parser = subparsers.add_parser('dummy', help='Generate dummy results for testing')
    dummy_parser.add_argument('--output', type=str, default='/tmp/dummy_results.json')
    dummy_parser.add_argument('--n_tasks', type=int, default=50)
    dummy_parser.add_argument('--k', type=int, default=3)
    dummy_parser.add_argument('--success_rate', type=float, default=0.15)

    args = parser.parse_args()

    if args.command == 'metrics':
        with open(args.results) as f:
            raw_results = json.load(f)
        metrics = compute_metrics(raw_results['task_results'])
        print(format_results_table(metrics))

    elif args.command == 'compare':
        with open(args.a) as f:
            raw_a = json.load(f)
        with open(args.c) as f:
            raw_c = json.load(f)
        metrics_a = compute_metrics(raw_a['task_results'])
        metrics_c = compute_metrics(raw_c['task_results'])
        print(compare_results(metrics_a, metrics_c))

    elif args.command == 'dummy':
        rng = np.random.default_rng(42)
        sites = ['shopping', 'reddit', 'gitlab']
        task_results = []
        for i in range(args.n_tasks):
            site = sites[i % len(sites)]
            attempts = [bool(rng.random() < args.success_rate) for _ in range(args.k)]
            task_results.append({
                'task_id': f'task_{i}',
                'site': site,
                'attempts': attempts,
            })
        output = {
            'model': 'dummy',
            'timestamp': datetime.now().isoformat(),
            'task_results': task_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved {args.n_tasks} dummy results to {args.output}")

        metrics = compute_metrics(task_results)
        print(format_results_table(metrics))

    else:
        parser.print_help()
