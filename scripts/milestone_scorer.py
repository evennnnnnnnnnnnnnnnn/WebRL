"""
Score a trajectory against milestones using Claude Sonnet.

Given a list of milestones and the agent's trajectory (sequence of
observations/actions), determines which milestones were achieved.
Returns a partial reward k/K where k = milestones achieved, K = total.
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from anthropic import Anthropic
from api_utils import get_client, api_call_with_retry

load_dotenv()


SCORING_PROMPT = """You are an expert at evaluating web agent trajectories against task milestones.

Given a list of milestones and an agent's trajectory (observations and actions), determine which milestones were achieved.

A milestone is "achieved" if the trajectory observations show evidence that the milestone's condition was met at any point during the trajectory. Be strict — only mark a milestone as achieved if there is clear evidence in the observations.

Milestones:
{milestones_json}

Agent Trajectory:
{trajectory_text}

Respond with ONLY a JSON array of booleans (true = achieved, false = not achieved), in the same order as the milestones. No explanation, no markdown, just the JSON array.

Example: [true, true, false, false]

Response:"""


def format_trajectory(trajectory: List[Dict]) -> str:
    """
    Format a trajectory into a readable text summary for the LLM.

    Each step has 'observation' (HTML-formatted prompt) and 'action' (agent response).
    We truncate observations to avoid hitting context limits.
    """
    parts = []
    for i, step in enumerate(trajectory):
        obs = step.get("observation", "")
        action = step.get("action", step.get("fixed_response", ""))

        # Truncate observation to last 2000 chars (most relevant part)
        if len(obs) > 2000:
            obs = "..." + obs[-2000:]

        parts.append(f"--- Step {i} ---")
        parts.append(f"Observation: {obs}")
        parts.append(f"Action: {action}")
        parts.append("")

    return "\n".join(parts)


def _call_score(client, model, prompt, num_milestones):
    """Single API call + parse. Called via api_call_with_retry."""
    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    if not text:
        raise ValueError("Empty response from API")

    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("["):
                text = cleaned
                break

    # Find JSON array in response (model sometimes adds explanation before/after)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    achieved = json.loads(text)

    if not isinstance(achieved, list) or len(achieved) != num_milestones:
        raise ValueError(
            f"Expected list of {num_milestones} booleans, got: {achieved}"
        )

    return [bool(a) for a in achieved]


def score_trajectory(
    milestones: List[str],
    trajectory: List[Dict],
    client: Optional[Anthropic] = None,
    model: str = "claude-sonnet-4-20250514",
) -> Tuple[float, List[bool]]:
    """
    Score a trajectory against milestones.

    Args:
        milestones: List of milestone descriptions.
        trajectory: List of step dicts with 'observation' and 'action' keys.
        client: Anthropic client.
        model: Claude model to use.

    Returns:
        (reward, achieved) where:
          - reward: float in [0, 1], k/K milestones achieved
          - achieved: list of booleans per milestone
    """
    if not milestones:
        return 0.0, []

    if client is None:
        client = get_client()

    trajectory_text = format_trajectory(trajectory)
    milestones_json = json.dumps(milestones, indent=2)

    prompt = SCORING_PROMPT.format(
        milestones_json=milestones_json,
        trajectory_text=trajectory_text,
    )

    achieved = api_call_with_retry(_call_score, client, model, prompt, len(milestones))
    reward = sum(achieved) / len(milestones)

    return reward, achieved


def score_trajectories_for_task(
    milestones: List[str],
    trajectories: List[List[Dict]],
    client: Optional[Anthropic] = None,
    model: str = "claude-sonnet-4-20250514",
) -> List[float]:
    """
    Score multiple trajectory attempts for one task.

    Args:
        milestones: Milestones for this task.
        trajectories: List of trajectories (each is list of step dicts).
        client: Anthropic client.
        model: Claude model to use.

    Returns:
        List of reward floats, one per trajectory.
    """
    if client is None:
        client = get_client()

    rewards = []
    for traj in trajectories:
        try:
            reward, _ = score_trajectory(milestones, traj, client=client, model=model)
            rewards.append(reward)
        except Exception as e:
            print(f"Warning: Failed to score trajectory: {e}")
            rewards.append(0.0)

    return rewards
