"""
Phase 0.8: End-to-end integration test.

Simulates the full pipeline with synthetic data:
1. Create fake trace files matching WebArena format
2. Run process_data.py (Condition A) → data_A.pt
3. Run process_data_ours.py (Condition C) → data_C.pt  (mocked LLM calls)
4. Verify both .pt files have identical schema
5. Verify both load into ReplayBuffer
6. Verify evaluate.py metrics computation works

Does NOT test actual training (requires GPU + DeepSpeed).
Does NOT call real LLM API (mocked).
"""

import sys
import os
import json
import torch
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from webrl.data.utils import ReplayBuffer
from webrl.environment.env_utils import add_mc_return
from regret_selector import compute_regret, select_tasks
from evaluate import compute_metrics, bootstrap_ci, format_results_table, compare_results


def create_fake_traces(dir_path, n_tasks=6, steps_per_task=3, k_attempts=2):
    """
    Create fake WebArena trace files matching the expected format.

    Creates:
      dir_path/traces/  — raw JSONL traces
      dir_path/actions/  — action score JSONs
    """
    os.makedirs(os.path.join(dir_path, 'traces'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'actions'), exist_ok=True)

    tasks = [f"Test task {i}: do something on website" for i in range(n_tasks)]
    trace_id = 0

    for task_idx, task in enumerate(tasks):
        for attempt in range(k_attempts):
            trace_name = f"trace_{trace_id}"
            # Alternate success/failure
            score = 1.0 if (task_idx + attempt) % 3 == 0 else 0.0

            # Build trace JSONL
            trace_content = []
            for step in range(steps_per_task):
                trace_content.append({
                    'target': task,
                    'prompt': f'html content for step {step}',
                    'html': f'<html>Page content step {step} task {task_idx}</html>',
                    'observation': [{'html': f'<html>obs {step}</html>', 'prompt': f'prompt {step}'}],
                    'response': f'click element_{step}',
                    'fixed_response': f'click element_{step}',
                    'score': score,
                })

            # Write trace JSONL
            trace_path = os.path.join(dir_path, 'traces', f'{trace_name}.jsonl')
            with open(trace_path, 'w') as f:
                for i, line in enumerate(trace_content):
                    if i == len(trace_content) - 1:
                        f.write(json.dumps(line))
                    else:
                        f.write(json.dumps(line) + '\n')

            # Write action score JSON
            action_path = os.path.join(dir_path, 'actions', f'{trace_name}.json')
            with open(action_path, 'w') as f:
                json.dump({'score': score}, f)

            trace_id += 1

    return tasks


def test_condition_a_pipeline(dir_path):
    """Test Condition A: trace_process + build_policy_data (WebRL's pipeline)."""
    from process_data import trace_process, build_policy_data

    print("Testing Condition A pipeline...")

    # Stage 1: trace_process
    trace_process(dir_path)
    assert os.path.exists(os.path.join(dir_path, 'fixed_traces')), \
        "fixed_traces/ not created"

    fixed_files = os.listdir(os.path.join(dir_path, 'fixed_traces'))
    assert len(fixed_files) > 0, "No fixed traces created"
    print(f"  trace_process: {len(fixed_files)} traces processed")

    # Stage 1 continued: build_policy_data
    output_path = os.path.join(dir_path, 'data_A.pt')
    build_policy_data(dir_path, output_path)

    assert os.path.exists(output_path), "data_A.pt not created"
    data = torch.load(output_path, weights_only=False)
    assert len(data) > 0, "data_A.pt is empty"
    print(f"  build_policy_data: {len(data)} trajectories, "
          f"{sum(len(t) for t in data)} steps")

    # Add MC returns (as the training loop would)
    data = [add_mc_return(t, gamma=0.9) for t in data]

    # Verify ReplayBuffer compatibility
    buf = ReplayBuffer(batch_size=1, capacity=10000)
    flat_steps = sum(data, [])
    for step in flat_steps:
        if not step['action'].endswith('<|eot_id|>'):
            step['action'] += '<|eot_id|>'
        buf.insert(**step)
    print(f"  ReplayBuffer: {buf.size} steps inserted OK")

    return data


def test_condition_c_pipeline(dir_path, tasks):
    """Test Condition C: milestone extraction + scoring + regret selection (mocked LLM)."""
    from process_data_ours import load_trajectories, _apply_prompt_template

    print("Testing Condition C pipeline...")

    # Load trajectories (requires fixed_traces from Condition A test)
    task_trajectories = load_trajectories(dir_path)
    print(f"  Loaded {len(task_trajectories)} unique tasks")

    # Mock milestone extraction
    task_milestones = {}
    for task in task_trajectories:
        task_milestones[task] = [
            "Navigate to the correct page",
            "Perform the action",
            "Verify the result",
        ]
    print(f"  Milestones: {len(task_milestones)} tasks (mocked)")

    # Mock milestone scoring — simulate varied rewards
    rng = np.random.default_rng(42)
    task_rewards = {}
    for task, trajectories in task_trajectories.items():
        rewards = []
        for traj in trajectories:
            # Mix of partial rewards
            reward = float(rng.uniform(0.0, 1.0))
            rewards.append(reward)
        task_rewards[task] = rewards
    print(f"  Scoring: {sum(len(v) for v in task_rewards.values())} trajectories (mocked)")

    # Regret selection
    selected_tasks, regrets = select_tasks(task_rewards, k=2, top_fraction=0.5)
    print(f"  Regret selection: {len(selected_tasks)}/{len(task_trajectories)} tasks selected")

    # Build training data from selected tasks
    all_trajectories = []
    for task in selected_tasks:
        trajectories = task_trajectories[task]
        rewards = task_rewards[task]
        for traj, reward in zip(trajectories, rewards):
            steps = []
            for i, step in enumerate(traj):
                if 'fixed_response' not in step:
                    continue
                steps.append({
                    'observation': traj[:i + 1],
                    'next_observation': traj[i + 1] if i < len(traj) - 1 else traj[i],
                    'task': task,
                    'reward': reward,
                    'done': i == len(traj) - 1,
                    'action': step['fixed_response'],
                    'trajectory_reward': reward,
                })
            if steps:
                all_trajectories.append(steps)

    # Apply prompt template
    all_trajectories = _apply_prompt_template(all_trajectories)

    # Add MC returns
    all_trajectories = [add_mc_return(t, gamma=0.9) for t in all_trajectories]

    # Save
    output_path = os.path.join(dir_path, 'data_C.pt')
    torch.save(all_trajectories, output_path)
    print(f"  Saved: {len(all_trajectories)} trajectories, "
          f"{sum(len(t) for t in all_trajectories)} steps")

    # Verify ReplayBuffer compatibility
    buf = ReplayBuffer(batch_size=1, capacity=10000)
    flat_steps = sum(all_trajectories, [])
    for step in flat_steps:
        if not step['action'].endswith('<|eot_id|>'):
            step['action'] += '<|eot_id|>'
        buf.insert(**step)
    print(f"  ReplayBuffer: {buf.size} steps inserted OK")

    return all_trajectories


def test_schema_match(dir_path):
    """Verify both .pt files have identical field schemas."""
    print("Testing schema match...")

    data_a = torch.load(os.path.join(dir_path, 'data_A.pt'), weights_only=False)
    data_c = torch.load(os.path.join(dir_path, 'data_C.pt'), weights_only=False)

    # Get fields from first step of each
    step_a = data_a[0][0] if isinstance(data_a[0], list) else data_a[0]
    step_c = data_c[0][0] if isinstance(data_c[0], list) else data_c[0]

    # Both must have the ReplayBuffer required fields
    required = {'observation', 'action', 'reward', 'next_observation', 'done', 'mc_return'}
    fields_a = set(step_a.keys())
    fields_c = set(step_c.keys())

    missing_a = required - fields_a
    missing_c = required - fields_c

    assert not missing_a, f"data_A missing fields: {missing_a}"
    assert not missing_c, f"data_C missing fields: {missing_c}"

    # Verify types match
    for field in required:
        type_a = type(step_a[field])
        type_c = type(step_c[field])
        assert type_a == type_c, \
            f"Type mismatch for '{field}': A={type_a}, C={type_c}"

    print(f"  data_A fields: {sorted(fields_a)}")
    print(f"  data_C fields: {sorted(fields_c)}")
    print(f"  Required fields present: OK")
    print(f"  Types match: OK")

    # Reward distribution check
    rewards_a = [s['reward'] for t in data_a for s in (t if isinstance(t, list) else [t])]
    rewards_c = [s['reward'] for t in data_c for s in (t if isinstance(t, list) else [t])]
    print(f"  Reward A: binary {set(rewards_a)} (should be {{0, 1}} or subset)")
    print(f"  Reward C: continuous [{min(rewards_c):.3f}, {max(rewards_c):.3f}]")


def test_evaluate_pipeline():
    """Test the evaluation metrics computation."""
    print("Testing evaluate pipeline...")

    # Simulate results
    rng = np.random.default_rng(42)
    results = []
    for i in range(20):
        site = ['shopping', 'reddit', 'gitlab'][i % 3]
        attempts = [bool(rng.random() < 0.3) for _ in range(3)]
        rewards = [float(rng.uniform(0, 1)) for _ in range(3)]
        results.append({
            'task_id': f'task_{i}',
            'site': site,
            'attempts': attempts,
            'rewards': rewards,
        })

    metrics = compute_metrics(results)
    assert 'pass_at_k' in metrics
    assert 'ci_95' in metrics
    assert 'per_site' in metrics
    assert 0 <= metrics['pass_at_k'] <= 1
    assert metrics['ci_95'][0] <= metrics['pass_at_k'] <= metrics['ci_95'][1]

    table = format_results_table(metrics)
    assert 'pass@3' in table
    print(f"  Metrics computed: pass@3 = {metrics['pass_at_k']:.1%}")

    # Test comparison
    results_b = []
    for i in range(20):
        site = ['shopping', 'reddit', 'gitlab'][i % 3]
        attempts = [bool(rng.random() < 0.5) for _ in range(3)]
        rewards = [float(rng.uniform(0, 1)) for _ in range(3)]
        results_b.append({
            'task_id': f'task_{i}',
            'site': site,
            'attempts': attempts,
            'rewards': rewards,
        })

    metrics_b = compute_metrics(results_b)
    comparison = compare_results(metrics, metrics_b)
    assert 'Delta' in comparison
    print(f"  Comparison table: OK")


def main():
    print("=" * 60)
    print("PHASE 0.8: End-to-End Integration Test")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp(prefix='webrl_test_')
    print(f"\nWorking directory: {tmp_dir}\n")

    try:
        # Step 1: Create fake traces
        print("Step 1: Creating fake traces...")
        tasks = create_fake_traces(tmp_dir, n_tasks=6, steps_per_task=3, k_attempts=2)
        trace_count = len(os.listdir(os.path.join(tmp_dir, 'traces')))
        print(f"  Created {trace_count} trace files for {len(tasks)} tasks\n")

        # Step 2: Condition A pipeline
        data_a = test_condition_a_pipeline(tmp_dir)
        print()

        # Step 3: Condition C pipeline (mocked LLM)
        data_c = test_condition_c_pipeline(tmp_dir, tasks)
        print()

        # Step 4: Schema match
        # Need to re-save data_A with mc_return
        data_a_with_mc = [add_mc_return(t, gamma=0.9) for t in torch.load(os.path.join(tmp_dir, 'data_A.pt'), weights_only=False)]
        torch.save(data_a_with_mc, os.path.join(tmp_dir, 'data_A.pt'))
        test_schema_match(tmp_dir)
        print()

        # Step 5: Evaluate pipeline
        test_evaluate_pipeline()
        print()

        print("=" * 60)
        print("ALL INTEGRATION TESTS PASSED")
        print("=" * 60)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
