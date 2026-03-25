"""Tests for regret_selector.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
from regret_selector import compute_regret, select_tasks


def test_all_succeed():
    """Tasks that always succeed have zero regret (p=1, pass@k=1)."""
    rewards = {"t1": [1.0, 1.0, 1.0], "t2": [1.0, 1.0, 1.0]}
    regrets = compute_regret(rewards, k=3)
    assert abs(regrets["t1"]) < 1e-9
    assert abs(regrets["t2"]) < 1e-9


def test_all_fail():
    """Tasks that always fail have zero regret (p=0, pass@k=0)."""
    rewards = {"t1": [0.0, 0.0, 0.0], "t2": [0.0, 0.0, 0.0]}
    regrets = compute_regret(rewards, k=3)
    assert abs(regrets["t1"]) < 1e-9
    assert abs(regrets["t2"]) < 1e-9


def test_mixed_has_positive_regret():
    """Tasks with mixed results have positive regret."""
    rewards = {"t1": [0.5, 0.5, 0.5]}  # p=0.5
    regrets = compute_regret(rewards, k=3)
    # pass@3 = 1 - (1-0.5)^3 = 1 - 0.125 = 0.875
    # regret = 0.875 - 0.5 = 0.375
    assert abs(regrets["t1"] - 0.375) < 1e-9


def test_regret_formula_known_values():
    """Verify regret against hand-computed values."""
    rewards = {"t1": [0.3, 0.3, 0.3]}  # p=0.3
    regrets = compute_regret(rewards, k=3)
    # pass@3 = 1 - (0.7)^3 = 1 - 0.343 = 0.657
    # regret = 0.657 - 0.3 = 0.357
    assert abs(regrets["t1"] - 0.357) < 1e-3


def test_higher_regret_for_frontier_tasks():
    """Tasks at the learning frontier (p~0.3-0.5) have highest regret."""
    rewards = {
        "easy": [0.9, 0.9, 0.9],        # p=0.9 → low regret
        "hard": [0.1, 0.1, 0.1],        # p=0.1 → low regret
        "frontier": [0.4, 0.4, 0.4],    # p=0.4 → high regret
    }
    regrets = compute_regret(rewards, k=3)
    assert regrets["frontier"] > regrets["easy"]
    assert regrets["frontier"] > regrets["hard"]


def test_select_top_fraction():
    """select_tasks returns the top fraction by regret."""
    rewards = {
        "easy": [1.0, 1.0, 1.0],
        "hard": [0.0, 0.0, 0.0],
        "frontier1": [0.4, 0.4, 0.4],
        "frontier2": [0.5, 0.5, 0.5],
    }
    selected, regrets = select_tasks(rewards, k=3, top_fraction=0.5)
    assert len(selected) == 2
    assert "frontier1" in selected
    assert "frontier2" in selected


def test_select_minimum_one():
    """Even with tiny fraction, at least 1 task is selected."""
    rewards = {"t1": [0.5, 0.5, 0.5]}
    selected, _ = select_tasks(rewards, k=3, top_fraction=0.01)
    assert len(selected) == 1


def test_empty_rewards():
    """Empty reward list gives zero regret."""
    rewards = {"t1": []}
    regrets = compute_regret(rewards, k=3)
    assert regrets["t1"] == 0.0


def test_single_attempt():
    """Works with k=1."""
    rewards = {"t1": [0.5]}
    regrets = compute_regret(rewards, k=1)
    # pass@1 = 1 - (1-0.5)^1 = 0.5
    # regret = 0.5 - 0.5 = 0
    assert abs(regrets["t1"]) < 1e-9


def test_variable_reward_across_attempts():
    """Handles different rewards per attempt (not all the same)."""
    rewards = {"t1": [0.0, 0.5, 1.0]}  # p = 0.5
    regrets = compute_regret(rewards, k=3)
    # Same as p=0.5: regret = 0.375
    assert abs(regrets["t1"] - 0.375) < 1e-9


def test_k_affects_regret():
    """Higher k increases regret for mid-range tasks."""
    rewards = {"t1": [0.3, 0.3, 0.3]}
    regret_k3 = compute_regret(rewards, k=3)["t1"]
    regret_k5 = compute_regret(rewards, k=5)["t1"]
    assert regret_k5 > regret_k3


def test_rewards_clipped():
    """Rewards outside [0,1] are clipped."""
    rewards = {"t1": [1.5, -0.5, 0.5]}  # mean = 0.5, but clipped to [0,1]
    regrets = compute_regret(rewards, k=3)
    # mean is 0.5, clipped to 0.5 → same as test_mixed
    assert abs(regrets["t1"] - 0.375) < 1e-9


if __name__ == "__main__":
    test_all_succeed()
    test_all_fail()
    test_mixed_has_positive_regret()
    test_regret_formula_known_values()
    test_higher_regret_for_frontier_tasks()
    test_select_top_fraction()
    test_select_minimum_one()
    test_empty_rewards()
    test_single_attempt()
    test_variable_reward_across_attempts()
    test_k_affects_regret()
    test_rewards_clipped()
    print("All tests passed!")
