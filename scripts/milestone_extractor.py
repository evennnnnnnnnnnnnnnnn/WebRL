"""
Extract observable milestones from a web task using Claude Sonnet.

Given a task instruction (e.g., "Add a red shirt to the shopping cart"),
decomposes it into a sequence of observable milestones that can be
verified from the agent's trajectory.
"""

import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from anthropic import Anthropic
from api_utils import get_client, api_call_with_retry

load_dotenv()


EXTRACTION_PROMPT = """You are an expert at decomposing web navigation tasks into observable milestones.

Given a task instruction for a web agent, break it down into a sequence of milestones that:
1. Are independently verifiable from HTML page content
2. Are ordered from first to last in expected execution
3. Cover the full task — completing all milestones means the task is done
4. Are concrete and specific, not vague

Return a JSON array of milestone strings. Each milestone should describe an observable state change on the page.

Examples:

Task: "Add a red medium t-shirt to the shopping cart"
Milestones:
["Navigate to a product page with a red t-shirt", "Select size Medium", "Click Add to Cart", "Cart shows the red medium t-shirt"]

Task: "Post a comment saying 'Great post!' on the first post in r/gaming"
Milestones:
["Navigate to r/gaming subreddit", "Open the first post", "Enter comment text 'Great post!'", "Submit the comment", "Comment appears on the post"]

Task: "Find the total sales amount for last month in the admin dashboard"
Milestones:
["Navigate to admin dashboard", "Access sales/reports section", "Filter or select last month's date range", "Total sales amount is displayed"]

Now decompose this task:

Task: "{task_instruction}"
Milestones:"""


def _call_extract(client, model, prompt):
    """Single API call + parse. Called via api_call_with_retry."""
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Parse JSON array from response
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    milestones = json.loads(text)

    if not isinstance(milestones, list) or len(milestones) == 0:
        raise ValueError(f"Expected non-empty list of milestones, got: {milestones}")

    return milestones


def extract_milestones(
    task_instruction: str,
    client: Optional[Anthropic] = None,
    model: str = "claude-sonnet-4-20250514",
) -> List[str]:
    """
    Extract milestones from a task instruction using Claude Sonnet.

    Args:
        task_instruction: The web task description.
        client: Anthropic client. If None, creates one from ANTHROPIC_API_KEY env var.
        model: Claude model to use.

    Returns:
        List of milestone description strings.
    """
    if client is None:
        client = get_client()

    prompt = EXTRACTION_PROMPT.format(task_instruction=task_instruction)
    return api_call_with_retry(_call_extract, client, model, prompt)


def extract_milestones_batch(
    tasks: Dict[str, str],
    client: Optional[Anthropic] = None,
    model: str = "claude-sonnet-4-20250514",
) -> Dict[str, List[str]]:
    """
    Extract milestones for multiple tasks.

    Args:
        tasks: {task_id: task_instruction}
        client: Anthropic client.
        model: Claude model to use.

    Returns:
        {task_id: [milestone1, milestone2, ...]}
    """
    if client is None:
        client = get_client()

    results = {}
    for task_id, instruction in tasks.items():
        try:
            milestones = extract_milestones(instruction, client=client, model=model)
            results[task_id] = milestones
        except Exception as e:
            print(f"Warning: Failed to extract milestones for task {task_id}: {e}")
            results[task_id] = []

    return results
