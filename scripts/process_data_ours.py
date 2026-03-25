"""
Modified data processing pipeline for Condition C.

Replaces WebRL's binary ORM reward + perplexity filtering with:
  - Milestone-based partial reward (k/K) via Claude Sonnet
  - Regret-based task selection (pass@k - avg@k)

Produces a .pt file compatible with WebRL's ReplayBuffer and training loop.

Usage:
    python process_data_ours.py \
        --rollout_path traces/ \
        --output_path data_C.pt \
        --k_attempts 3 \
        --top_fraction 0.5
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Reuse WebRL's existing utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from webrl.environment.env_utils import add_mc_return

from milestone_extractor import extract_milestones_batch
from milestone_scorer import score_trajectories_for_task
from regret_selector import select_tasks

from process_data import read_jsonl, save_jsonl, trace_process, build_policy_data


def load_trajectories(dir_path: str) -> dict:
    """
    Load raw trajectories grouped by task.

    Returns:
        {task_id: [trajectory_1, trajectory_2, ...]}
        where each trajectory is a list of step dicts from the trace JSONL.
    """
    traces_dir = os.path.join(dir_path, 'fixed_traces')
    if not os.path.exists(traces_dir):
        raise FileNotFoundError(
            f"fixed_traces/ not found in {dir_path}. Run trace_process() first."
        )

    files = os.listdir(traces_dir)
    tasks = defaultdict(list)

    for f in files:
        if not f.endswith('.jsonl'):
            continue
        trace_content = read_jsonl(os.path.join(traces_dir, f))
        if len(trace_content) == 0:
            continue
        task_instruction = trace_content[-1].get('target', '')
        if not task_instruction:
            continue
        tasks[task_instruction].append(trace_content)

    return dict(tasks)


def process_condition_c(
    dir_path: str,
    output_path: str,
    k_attempts: int = 3,
    top_fraction: float = 0.5,
    gamma: float = 0.9,
    milestones_cache_path: str = None,
):
    """
    Full Condition C pipeline:
    1. trace_process() — clean raw traces (reuse WebRL's)
    2. Extract milestones for each unique task
    3. Score trajectories against milestones
    4. Compute regret, select top tasks
    5. Build policy data with milestone rewards
    6. Compute MC returns
    7. Save as .pt compatible with ReplayBuffer
    """
    # Step 1: Clean traces
    print("Step 1: Processing raw traces...")
    trace_process(dir_path)

    # Step 2: Load trajectories grouped by task
    print("Step 2: Loading trajectories...")
    task_trajectories = load_trajectories(dir_path)
    print(f"  Found {len(task_trajectories)} unique tasks, "
          f"{sum(len(v) for v in task_trajectories.values())} total trajectories")

    # Step 3: Extract milestones
    print("Step 3: Extracting milestones via Claude Sonnet...")
    task_instructions = {task: task for task in task_trajectories.keys()}

    if milestones_cache_path and os.path.exists(milestones_cache_path):
        print(f"  Loading cached milestones from {milestones_cache_path}")
        with open(milestones_cache_path, 'r') as f:
            task_milestones = json.load(f)
    else:
        task_milestones = extract_milestones_batch(task_instructions)
        if milestones_cache_path:
            with open(milestones_cache_path, 'w') as f:
                json.dump(task_milestones, f, indent=2)
            print(f"  Cached milestones to {milestones_cache_path}")

    # Step 4: Score trajectories
    print("Step 4: Scoring trajectories against milestones...")
    task_rewards = {}  # {task: [reward_per_attempt]}

    for task, trajectories in tqdm(task_trajectories.items(), desc="Scoring"):
        milestones = task_milestones.get(task, [])
        if not milestones:
            task_rewards[task] = [0.0] * len(trajectories)
            continue
        rewards = score_trajectories_for_task(milestones, trajectories)
        task_rewards[task] = rewards

    # Step 5: Regret selection
    print("Step 5: Computing regret and selecting tasks...")
    selected_tasks, regrets = select_tasks(
        task_rewards, k=k_attempts, top_fraction=top_fraction
    )
    print(f"  Selected {len(selected_tasks)}/{len(task_trajectories)} tasks")

    # Print regret distribution
    regret_values = list(regrets.values())
    print(f"  Regret stats: mean={np.mean(regret_values):.3f}, "
          f"max={np.max(regret_values):.3f}, "
          f"min={np.min(regret_values):.3f}")

    # Step 6: Build training data from selected tasks
    print("Step 6: Building training data...")
    all_trajectories = []

    for task in selected_tasks:
        trajectories = task_trajectories[task]
        rewards = task_rewards[task]

        for traj, reward in zip(trajectories, rewards):
            # Build step dicts matching WebRL's format
            steps = []
            for i, step in enumerate(traj):
                if 'fixed_response' not in step:
                    continue
                steps.append({
                    'observation': traj[:i + 1],
                    'next_observation': traj[i + 1] if i < len(traj) - 1 else traj[i],
                    'task': task,
                    'reward': reward,  # Milestone reward (same for all steps in trajectory)
                    'done': i == len(traj) - 1,
                    'action': step['fixed_response'],
                    'trajectory_reward': reward,
                })
            if steps:
                all_trajectories.append(steps)

    # Apply WebRL's prompt formatting (reuse build_policy_data's template logic)
    all_trajectories = _apply_prompt_template(all_trajectories)

    # Add MC returns
    all_trajectories = [add_mc_return(t, gamma=gamma) for t in all_trajectories]

    # Step 7: Save
    print(f"Step 7: Saving to {output_path}...")
    torch.save(all_trajectories, output_path)

    # Print stats
    total_steps = sum(len(t) for t in all_trajectories)
    print(f"  Saved {len(all_trajectories)} trajectories, {total_steps} total steps")

    return all_trajectories


def _apply_prompt_template(all_trajectories):
    """
    Apply WebRL's LLaMA-3 chat prompt formatting.
    Mirrors build_policy_data()'s template() function.
    """
    def format_history(contents, index):
        history = ""
        if index == 0:
            return history
        for i in range(index - 1, -1, -1):
            history = (
                f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{contents[i]['prompt']}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                f"{contents[i]['fixed_response']}\n\n"
            ) + history
        return history

    def format_prompt(instruction, index, html_text, contents):
        history = format_history(contents, index)
        if len(history) + len(html_text) > (16384 - 512):
            html_text = html_text[:(16384 - 512) - len(history)]
        current_turn = (
            f"Round {index}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{html_text}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        prompt = f"Task Instruction: {instruction}\n\n{history}{current_turn}"
        return prompt

    for traj in all_trajectories:
        for i, step in enumerate(traj):
            instruction = step['task']
            contents = step['observation']
            html_text = contents[-1].get('html', '') if isinstance(contents[-1], dict) else ''
            step['observation'] = format_prompt(instruction, i, html_text, contents)

            next_obs = step['next_observation']
            if isinstance(next_obs, dict):
                next_html = next_obs.get('html', '')
                next_contents = contents + [next_obs]
                step['next_observation'] = format_prompt(
                    instruction, i + 1, next_html, next_contents
                )
            elif isinstance(next_obs, str):
                pass  # Already formatted
            else:
                step['next_observation'] = step['observation']

    return all_trajectories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data for Condition C')
    parser.add_argument('--rollout_path', type=str, required=True,
                        help='Directory containing traces/ and actions/')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output .pt file path')
    parser.add_argument('--k_attempts', type=int, default=3,
                        help='Number of attempts per task for regret computation')
    parser.add_argument('--top_fraction', type=float, default=0.5,
                        help='Fraction of tasks to select by regret')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Discount factor for MC returns')
    parser.add_argument('--milestones_cache', type=str, default=None,
                        help='Path to cache milestone extractions (JSON)')
    args = parser.parse_args()

    process_condition_c(
        dir_path=args.rollout_path,
        output_path=args.output_path,
        k_attempts=args.k_attempts,
        top_fraction=args.top_fraction,
        gamma=args.gamma,
        milestones_cache_path=args.milestones_cache,
    )
