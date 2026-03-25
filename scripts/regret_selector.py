"""
Regret-based task selection for RL training data.

Selects tasks at the "learning frontier" — tasks where the agent sometimes
succeeds but not always. These have the most room for improvement.

Regret = pass@k - avg@k
  - High regret: inconsistent performance → most to learn
  - Zero regret: always succeeds (too easy) or always fails (too hard)
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_regret(task_rewards: Dict[str, List[float]], k: int = 3) -> Dict[str, float]:
    """
    Compute regret score for each task.

    Args:
        task_rewards: {task_id: [reward_attempt_1, ..., reward_attempt_k]}
                      Rewards are floats in [0, 1] (milestone-based).
        k: Number of attempts per task (used in pass@k formula).

    Returns:
        {task_id: regret_score}

    Formula:
        p = mean(rewards)              # average milestone reward across attempts
        pass@k = 1 - (1 - p)^k         # probability of at least one "success"
        regret = pass@k - p             # gap between best-case and average
    """
    regrets = {}
    for task_id, rewards in task_rewards.items():
        if len(rewards) == 0:
            regrets[task_id] = 0.0
            continue
        p = float(np.mean(rewards))
        p = np.clip(p, 0.0, 1.0)
        pass_at_k = 1.0 - (1.0 - p) ** k
        regrets[task_id] = pass_at_k - p
    return regrets


def select_tasks(
    task_rewards: Dict[str, List[float]],
    k: int = 3,
    top_fraction: float = 0.5,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Select tasks with highest regret.

    Args:
        task_rewards: {task_id: [reward_attempt_1, ..., reward_attempt_k]}
        k: Number of attempts per task.
        top_fraction: Fraction of tasks to select (0.0, 1.0].

    Returns:
        (selected_task_ids, regret_scores)
    """
    regrets = compute_regret(task_rewards, k=k)
    sorted_tasks = sorted(regrets.items(), key=lambda x: x[1], reverse=True)
    n = max(1, int(len(sorted_tasks) * top_fraction))
    selected = [task_id for task_id, _ in sorted_tasks[:n]]
    return selected, regrets
