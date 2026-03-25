"""
Test that .pt output files are compatible with WebRL's ReplayBuffer.

Validates the data contract between process_data / process_data_ours
and the training loop.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from webrl.data.utils import ReplayBuffer


REQUIRED_FIELDS = {'observation', 'action', 'reward', 'next_observation', 'done', 'mc_return'}


def validate_pt_file(path: str, verbose: bool = True):
    """
    Validate a .pt file can be loaded into ReplayBuffer.

    Checks:
    1. File loads as a list of trajectories (list of list of dicts)
       OR a flat list of step dicts (for _filter files)
    2. Each step has all required fields
    3. Fields have correct types
    4. Steps insert into ReplayBuffer without error
    """
    data = torch.load(path, weights_only=False)

    # Determine format: list of trajectories or flat list of steps
    if len(data) > 0 and isinstance(data[0], list):
        # List of trajectories
        steps = []
        for traj in data:
            steps.extend(traj)
        format_type = "trajectories"
    elif len(data) > 0 and isinstance(data[0], dict):
        steps = data
        format_type = "flat"
    else:
        raise ValueError(f"Unexpected data format: {type(data[0]) if data else 'empty'}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"File: {path}")
        print(f"Format: {format_type}")
        print(f"Total steps: {len(steps)}")
        if format_type == "trajectories":
            print(f"Trajectories: {len(data)}")

    # Check fields
    for i, step in enumerate(steps):
        missing = REQUIRED_FIELDS - set(step.keys())
        if missing:
            raise ValueError(f"Step {i} missing fields: {missing}")

        # Type checks
        assert isinstance(step['observation'], str), \
            f"Step {i}: observation must be str, got {type(step['observation'])}"
        assert isinstance(step['action'], str), \
            f"Step {i}: action must be str, got {type(step['action'])}"
        assert isinstance(step['reward'], (int, float, np.floating)), \
            f"Step {i}: reward must be numeric, got {type(step['reward'])}"
        assert isinstance(step['next_observation'], str), \
            f"Step {i}: next_observation must be str, got {type(step['next_observation'])}"
        assert isinstance(step['done'], (bool, int, np.bool_)), \
            f"Step {i}: done must be bool/int, got {type(step['done'])}"
        assert isinstance(step['mc_return'], (int, float, np.floating)), \
            f"Step {i}: mc_return must be numeric, got {type(step['mc_return'])}"

    # Test ReplayBuffer insertion
    buf = ReplayBuffer(batch_size=1, capacity=max(len(steps), 1))
    for step in steps:
        buf.insert(**step)

    if verbose:
        rewards = [s['reward'] for s in steps]
        mc_returns = [s['mc_return'] for s in steps]
        print(f"Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
        print(f"Reward mean: {np.mean(rewards):.3f}")
        print(f"MC return range: [{min(mc_returns):.3f}, {max(mc_returns):.3f}]")
        print(f"MC return mean: {np.mean(mc_returns):.3f}")
        print(f"Buffer insertion: OK ({buf.size} steps)")
        print(f"{'='*60}")

    return True


def create_dummy_pt(path: str, n_trajectories: int = 5, steps_per_traj: int = 3,
                    reward_type: str = "binary"):
    """Create a dummy .pt file for testing."""
    data = []
    for t in range(n_trajectories):
        traj = []
        if reward_type == "binary":
            reward = float(np.random.choice([0, 1]))
        else:
            reward = float(np.random.uniform(0, 1))

        for s in range(steps_per_traj):
            step = {
                'observation': f'Task: test task {t}\n\nRound {s}\n\nobs_{t}_{s}',
                'action': f'action_{t}_{s}<|eot_id|>',
                'reward': reward,
                'next_observation': f'Task: test task {t}\n\nRound {s+1}\n\nobs_{t}_{s+1}',
                'done': s == steps_per_traj - 1,
                'mc_return': reward * (0.9 ** (steps_per_traj - s - 1)),
            }
            traj.append(step)
        data.append(traj)
    torch.save(data, path)
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', help='.pt files to validate')
    parser.add_argument('--test-dummy', action='store_true',
                        help='Create and validate dummy files')
    args = parser.parse_args()

    if args.test_dummy:
        print("Creating dummy .pt files...")
        for rtype in ["binary", "milestone"]:
            path = f"/tmp/test_dummy_{rtype}.pt"
            create_dummy_pt(path, reward_type=rtype)
            validate_pt_file(path)
        print("\nDummy file tests passed!")

    for f in args.files:
        try:
            validate_pt_file(f)
            print(f"  PASS: {f}")
        except Exception as e:
            print(f"  FAIL: {f} — {e}")
