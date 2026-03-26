#!/usr/bin/env python3
"""
Local test script for the full experiment pipeline.

Tests every phase of run_experiment.sh locally on CPU with:
  - GPT-2 (tiny model) instead of LLaMA-3.1-8B
  - Fake traces instead of real WebArena rollouts
  - No DeepSpeed/torchrun (single-process CPU)
  - No WebArena Docker containers
  - Skips actual Claude API calls (uses mock milestones)

Usage:
    cd web-rl
    python scripts/test_local.py

This catches import errors, data format issues, and pipeline bugs
before deploying to RunPod.
"""

import json
import os
import sys
import shutil
import traceback
import tempfile

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBRL_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, WEBRL_DIR)

# Use a temp directory for all test artifacts
TEST_DIR = os.path.join(WEBRL_DIR, "_local_test")


def log(msg):
    print(f"[TEST] {msg}")


def fail(msg, exc=None):
    print(f"\n[FAIL] {msg}")
    if exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    sys.exit(1)


def cleanup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# Step 1: Verify imports
# ═══════════════════════════════════════════════════════════════════
def step1_imports():
    log("Step 1: Checking imports...")

    errors = []

    # Core Python/ML
    for mod in ["torch", "numpy", "transformers", "accelerate", "tqdm",
                "hydra", "omegaconf", "click"]:
        try:
            __import__(mod)
        except ImportError as e:
            errors.append(f"  Missing: {mod} ({e})")

    # WebRL internal
    try:
        from webrl.data.utils import ReplayBuffer, DummyDataset
        log("  webrl.data: OK")
    except Exception as e:
        errors.append(f"  webrl.data: {e}")

    try:
        from webrl.environment.env_utils import add_mc_return
        log("  webrl.environment: OK")
    except Exception as e:
        errors.append(f"  webrl.environment: {e}")

    try:
        from webrl.misc import colorful_print
        log("  webrl.misc: OK")
    except Exception as e:
        errors.append(f"  webrl.misc: {e}")

    # Scripts
    try:
        from process_data import trace_process, build_policy_data, read_jsonl, save_jsonl
        log("  process_data: OK")
    except Exception as e:
        errors.append(f"  process_data: {e}")

    try:
        from process_data_ours import load_trajectories, _apply_prompt_template, process_condition_c
        log("  process_data_ours: OK")
    except Exception as e:
        errors.append(f"  process_data_ours: {e}")

    try:
        from evaluate import compute_metrics, compute_metrics_with_training, compare_results, format_results_table
        log("  evaluate: OK")
    except Exception as e:
        errors.append(f"  evaluate: {e}")

    try:
        from regret_selector import compute_regret, select_tasks
        log("  regret_selector: OK")
    except Exception as e:
        errors.append(f"  regret_selector: {e}")

    try:
        from milestone_extractor import extract_milestones, EXTRACTION_PROMPT
        log("  milestone_extractor: OK (import only, no API call)")
    except Exception as e:
        errors.append(f"  milestone_extractor: {e}")

    try:
        from milestone_scorer import score_trajectory, format_trajectory
        log("  milestone_scorer: OK (import only, no API call)")
    except Exception as e:
        errors.append(f"  milestone_scorer: {e}")

    # DeepSpeed — expected to fail on Windows, just warn
    try:
        import deepspeed
        log("  deepspeed: OK")
    except ImportError:
        log("  deepspeed: NOT INSTALLED (expected on Windows, required on RunPod)")

    # train_no_critic — imports deepspeed, so catch gracefully
    try:
        # Don't import directly as it imports deepspeed at module level
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_no_critic", os.path.join(SCRIPT_DIR, "train_no_critic.py")
        )
        log("  train_no_critic: file found (skipping import — needs deepspeed)")
    except Exception as e:
        errors.append(f"  train_no_critic: {e}")

    if errors:
        log("Import errors found:")
        for e in errors:
            print(e)
        fail("Fix import errors above before proceeding")

    log("Step 1 PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Step 2: Create fake traces (mimics WebArena rollout output)
# ═══════════════════════════════════════════════════════════════════
def step2_create_traces():
    log("Step 2: Creating fake traces...")

    traces_dir = os.path.join(TEST_DIR, "traces")
    actions_dir = os.path.join(TEST_DIR, "actions")
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(actions_dir, exist_ok=True)

    tasks = [
        "Find the price of the most expensive item in the Electronics category",
        "Post a comment on the first thread in the technology forum",
        "Check the total number of open issues in the project",
    ]

    trace_id = 0
    for t_idx, task in enumerate(tasks):
        for attempt in range(2):  # 2 attempts per task
            score = 1.0 if (t_idx + attempt) % 2 == 0 else 0.0
            trace = []
            for step in range(3):  # 3 steps per trajectory
                trace.append({
                    "target": task,
                    "prompt": f"Step {step} prompt for task {t_idx}",
                    "html": f"<html><body>Page content for step {step}, task {t_idx}, attempt {attempt}</body></html>",
                    "observation": [{"html": f"<html>obs {step}</html>", "prompt": f"p{step}"}],
                    "response": f"click element_{step}",
                    "fixed_response": f"click element_{step}",
                    "score": score,
                })

            name = f"trace_{trace_id}"
            # Write trace JSONL
            with open(os.path.join(traces_dir, f"{name}.jsonl"), "w") as f:
                for i, line in enumerate(trace):
                    f.write(json.dumps(line))
                    if i < len(trace) - 1:
                        f.write("\n")
            # Write action JSON
            with open(os.path.join(actions_dir, f"{name}.json"), "w") as f:
                json.dump({"score": score}, f)

            trace_id += 1

    log(f"  Created {trace_id} traces for {len(tasks)} tasks")
    log("Step 2 PASSED\n")
    return trace_id


# ═══════════════════════════════════════════════════════════════════
# Step 3: Condition A data processing (trace_process + build_policy_data)
# ═══════════════════════════════════════════════════════════════════
def step3_condition_a():
    log("Step 3: Condition A data processing...")

    import torch
    from process_data import trace_process, build_policy_data
    from webrl.environment.env_utils import add_mc_return
    from webrl.data.utils import ReplayBuffer

    # Step 3a: trace_process — creates fixed_traces/
    log("  3a: trace_process()...")
    trace_process(TEST_DIR)

    fixed_dir = os.path.join(TEST_DIR, "fixed_traces")
    if not os.path.exists(fixed_dir):
        fail("trace_process() did not create fixed_traces/")
    n_fixed = len([f for f in os.listdir(fixed_dir) if f.endswith(".jsonl")])
    log(f"  Created {n_fixed} fixed traces")

    # Step 3b: build_policy_data — creates .pt file
    log("  3b: build_policy_data()...")
    output_path = os.path.join(TEST_DIR, "data_A.pt")
    build_policy_data(TEST_DIR, output_path)

    data = torch.load(output_path, weights_only=False)
    log(f"  Loaded {len(data)} trajectories from data_A.pt")

    if len(data) == 0:
        fail("data_A.pt is empty — no trajectories produced")

    # Step 3c: add_mc_return
    log("  3c: add_mc_return()...")
    data = [add_mc_return(t, gamma=0.9) for t in data]

    # Verify mc_return field exists
    for traj in data:
        for step in traj:
            if "mc_return" not in step:
                fail("mc_return missing from step after add_mc_return()")

    # Step 3d: ReplayBuffer insertion
    log("  3d: ReplayBuffer insertion...")
    buf = ReplayBuffer(batch_size=1, capacity=1000)
    steps = sum(data, [])
    for s in steps:
        if not s["action"].endswith("<|eot_id|>"):
            s["action"] += "<|eot_id|>"
        buf.insert(**s)

    log(f"  ReplayBuffer: {buf.size} steps inserted")

    if buf.size == 0:
        fail("ReplayBuffer is empty after insertion")

    # Step 3e: Verify sample/get work
    sample = buf.sample(batch_size=1)
    assert "observation" in sample, "sample() missing 'observation'"
    assert "mc_return" in sample, "sample() missing 'mc_return'"

    item = buf.get(0)
    assert "observation" in item, "get() missing 'observation'"

    torch.save(data, output_path)
    log("Step 3 PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Step 4: Condition C data processing (milestones + regret, mocked)
# ═══════════════════════════════════════════════════════════════════
def step4_condition_c():
    log("Step 4: Condition C data processing (mocked API)...")

    import torch
    from process_data import trace_process
    from process_data_ours import load_trajectories, _apply_prompt_template
    from regret_selector import compute_regret, select_tasks
    from webrl.environment.env_utils import add_mc_return
    from webrl.data.utils import ReplayBuffer

    # Step 4a: load_trajectories (needs fixed_traces from step 3)
    log("  4a: load_trajectories()...")
    task_trajectories = load_trajectories(TEST_DIR)
    log(f"  Loaded {len(task_trajectories)} unique tasks")

    if len(task_trajectories) == 0:
        fail("No tasks loaded from fixed_traces/")

    # Step 4b: Mock milestones (skip API calls)
    log("  4b: Mock milestones (no API call)...")
    task_milestones = {}
    for task in task_trajectories:
        task_milestones[task] = [
            "Navigate to the correct page",
            "Identify the target element",
            "Perform the action",
            "Verify the result",
        ]

    # Step 4c: Mock scoring (skip API calls)
    log("  4c: Mock trajectory scoring (no API call)...")
    task_rewards = {}
    for task, trajectories in task_trajectories.items():
        rewards = []
        for traj in trajectories:
            # Use original score as mock reward
            score = traj[-1].get("score", 0.0)
            rewards.append(float(score))
        task_rewards[task] = rewards

    # Step 4d: Regret computation + selection
    log("  4d: Regret computation & task selection...")
    regrets = compute_regret(task_rewards, k=2)
    log(f"  Regrets: {regrets}")

    selected, all_regrets = select_tasks(task_rewards, k=2, top_fraction=0.5)
    log(f"  Selected {len(selected)}/{len(task_trajectories)} tasks")

    if len(selected) == 0:
        fail("No tasks selected by regret selector")

    # Step 4e: Build training data
    log("  4e: Build training data from selected tasks...")
    all_trajectories = []
    for task in selected:
        trajectories = task_trajectories[task]
        rewards = task_rewards[task]
        for traj, reward in zip(trajectories, rewards):
            steps = []
            for i, step in enumerate(traj):
                if "fixed_response" not in step:
                    continue
                steps.append({
                    "observation": traj[: i + 1],
                    "next_observation": traj[i + 1] if i < len(traj) - 1 else traj[i],
                    "task": task,
                    "reward": reward,
                    "done": i == len(traj) - 1,
                    "action": step["fixed_response"],
                    "trajectory_reward": reward,
                })
            if steps:
                all_trajectories.append(steps)

    log(f"  Built {len(all_trajectories)} trajectories")

    # Step 4f: Apply prompt template
    log("  4f: Apply prompt template...")
    all_trajectories = _apply_prompt_template(all_trajectories)

    # Verify observation is now a string (formatted prompt)
    for traj in all_trajectories:
        for step in traj:
            if not isinstance(step["observation"], str):
                fail(f"observation should be string after template, got {type(step['observation'])}")

    # Step 4g: MC returns + save
    log("  4g: MC returns + save...")
    all_trajectories = [add_mc_return(t, gamma=0.9) for t in all_trajectories]

    output_path = os.path.join(TEST_DIR, "data_C.pt")
    torch.save(all_trajectories, output_path)

    # Step 4h: ReplayBuffer insertion
    log("  4h: ReplayBuffer insertion...")
    buf = ReplayBuffer(batch_size=1, capacity=1000)
    flat = sum(all_trajectories, [])
    for s in flat:
        if not s["action"].endswith("<|eot_id|>"):
            s["action"] += "<|eot_id|>"
        buf.insert(**s)

    log(f"  ReplayBuffer: {buf.size} steps inserted")

    if buf.size == 0:
        fail("ReplayBuffer is empty after insertion for Condition C")

    log("Step 4 PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Step 5: Verify .pt file format compatibility (test_data_format.py)
# ═══════════════════════════════════════════════════════════════════
def step5_data_format():
    log("Step 5: Verifying .pt file format...")

    import torch
    from webrl.data.utils import ReplayBuffer

    for label, filename in [("A", "data_A.pt"), ("C", "data_C.pt")]:
        path = os.path.join(TEST_DIR, filename)
        data = torch.load(path, weights_only=False)
        log(f"  {label}: {len(data)} trajectories")

        required_keys = {"observation", "action", "reward", "done", "mc_return", "next_observation"}

        for i, traj in enumerate(data):
            if not isinstance(traj, list):
                fail(f"{label} traj {i} is not a list: {type(traj)}")
            for j, step in enumerate(traj):
                if not isinstance(step, dict):
                    fail(f"{label} traj {i} step {j} is not a dict: {type(step)}")
                missing = required_keys - set(step.keys())
                if missing:
                    fail(f"{label} traj {i} step {j} missing keys: {missing}")
                if not isinstance(step["observation"], str):
                    fail(f"{label} traj {i} step {j} observation not string: {type(step['observation'])}")
                if not isinstance(step["action"], str):
                    fail(f"{label} traj {i} step {j} action not string: {type(step['action'])}")

        # Verify ReplayBuffer compatibility
        buf = ReplayBuffer(batch_size=1, capacity=10000)
        flat = sum(data, [])
        for s in flat:
            if not s["action"].endswith("<|eot_id|>"):
                s["action"] += "<|eot_id|>"
            buf.insert(**s)

        log(f"  {label}: ReplayBuffer OK ({buf.size} steps)")

    log("Step 5 PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Step 6: Test model loading with GPT-2 (tiny, CPU)
# ═══════════════════════════════════════════════════════════════════
def step6_model_loading():
    log("Step 6: Testing model loading with GPT-2 (CPU)...")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"  # ~500MB, smallest possible

    log("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    log(f"  Tokenizer: vocab_size={tokenizer.vocab_size}")

    log("  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
    model.eval()
    log(f"  Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Test forward pass
    log("  Testing forward pass...")
    test_input = "Task Instruction: Click the search button\n\nRound 0\n\nSearch page content"
    inputs = tokenizer(test_input, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    log(f"  Forward pass OK, logits shape: {outputs.logits.shape}")

    # Test log-prob computation (mirrors LlamaAgent.get_log_prob)
    log("  Testing log-prob computation...")
    obs = "Task Instruction: test\n\nRound 0\n\npage"
    act = "click button_1"
    input_text = obs + act
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    obs_len = len(tokenizer(obs, return_tensors="pt")["input_ids"][0])

    with torch.no_grad():
        out = model(**input_ids)
    log_probs = out.logits.log_softmax(dim=-1)
    log(f"  Log-prob computation OK, shape: {log_probs.shape}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log("Step 6 PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Step 7: Test evaluation metrics
# ═══════════════════════════════════════════════════════════════════
def step7_evaluation():
    log("Step 7: Testing evaluation metrics...")

    from evaluate import compute_metrics, compute_metrics_with_training, compare_results, format_results_table

    results_a = [
        {"task_id": "t0", "site": "shopping", "attempts": [True, False, False]},
        {"task_id": "t1", "site": "reddit", "attempts": [False, False, False]},
        {"task_id": "t2", "site": "gitlab", "attempts": [True, True, False]},
    ]
    results_c = [
        {"task_id": "t0", "site": "shopping", "attempts": [True, True, False]},
        {"task_id": "t1", "site": "reddit", "attempts": [False, True, False]},
        {"task_id": "t2", "site": "gitlab", "attempts": [True, True, True]},
    ]

    meta_a = {
        "condition": "A", "training_tasks": 3, "training_trajectories": 6,
        "training_steps": 6, "training_time_seconds": 60,
        "data_processing_time_seconds": 30,
    }
    meta_c = {
        "condition": "C", "training_tasks": 2, "training_trajectories": 4,
        "training_steps": 4, "training_time_seconds": 40,
        "data_processing_time_seconds": 20,
    }

    m_a = compute_metrics_with_training(results_a, meta_a)
    m_c = compute_metrics_with_training(results_c, meta_c)

    log(f"  Condition A: pass@3={m_a['pass_at_k']:.1%}")
    log(f"  Condition C: pass@3={m_c['pass_at_k']:.1%}")

    comparison = compare_results(m_a, m_c)
    log("  Comparison table:")
    for line in comparison.split("\n"):
        print(f"    {line}")

    # Test format_results_table
    table = format_results_table(m_a)
    assert "pass@" in table, "format_results_table missing pass@ line"

    # Test JSON serialization (for saving results)
    import json
    json.dumps(m_a)  # Should not raise
    json.dumps(m_c)

    log("Step 7 PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Step 8: Test collect_results helper (from run_experiment.sh)
# ═══════════════════════════════════════════════════════════════════
def step8_collect_results():
    log("Step 8: Testing collect_results logic...")

    import json
    from collections import defaultdict

    # Create fake eval output (mimics what VAB-WebArena-Lite produces)
    eval_dir = os.path.join(TEST_DIR, "eval_test", "actions")
    os.makedirs(eval_dir, exist_ok=True)

    for i in range(5):
        score = 1.0 if i % 2 == 0 else 0.0
        with open(os.path.join(eval_dir, f"task_{i}_attempt1.json"), "w") as f:
            json.dump({"score": score}, f)
        with open(os.path.join(eval_dir, f"task_{i}_attempt2.json"), "w") as f:
            json.dump({"score": 0.5 if i == 1 else score}, f)

    # Run the collect logic (mirrors run_experiment.sh)
    import glob
    traces_dir = os.path.join(TEST_DIR, "eval_test")
    actions_dir = os.path.join(traces_dir, "actions")

    results = []
    for f_path in sorted(glob.glob(os.path.join(actions_dir, "*.json"))):
        with open(f_path) as fp:
            data = json.load(fp)
        task_id = os.path.basename(f_path).replace(".json", "")
        score = data.get("score", 0)
        results.append({"task_id": task_id, "site": "unknown", "score": score})

    task_attempts = defaultdict(list)
    for r in results:
        base_id = r["task_id"].rsplit("_attempt", 1)[0]
        task_attempts[base_id].append(r["score"] >= 0.5)

    task_results = [
        {"task_id": tid, "site": "unknown", "attempts": att}
        for tid, att in task_attempts.items()
    ]

    log(f"  Collected {len(task_results)} tasks, {len(results)} total attempts")
    assert len(task_results) == 5, f"Expected 5 tasks, got {len(task_results)}"
    assert len(results) == 10, f"Expected 10 attempts, got {len(results)}"

    log("Step 8 PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Step 9: Test VAB-WebArena-Lite imports
# ═══════════════════════════════════════════════════════════════════
def step9_vab_imports():
    log("Step 9: Testing VAB-WebArena-Lite imports...")

    vab_dir = os.path.normpath(os.path.join(
        WEBRL_DIR, "..", "visualagentbench", "VAB-WebArena-Lite"
    ))

    if not os.path.exists(vab_dir):
        log("  SKIPPED: VAB-WebArena-Lite not found at expected path")
        log(f"  Expected: {vab_dir}")
        log("Step 9 SKIPPED\n")
        return

    # Check key files exist
    key_files = [
        "run.py",
        "agent/agent.py",
        "agent/prompts/jsons/p_webrl.json",
        "browser_env/actions.py",
        "browser_env/envs.py",
        "evaluation_harness/evaluators.py",
        "llms/providers/openai_utils.py",
        "config_files/wa/test_webarena_lite.raw.json",
    ]

    missing = []
    for f in key_files:
        full_path = os.path.join(vab_dir, f)
        if os.path.exists(full_path):
            log(f"  {f}: OK")
        else:
            missing.append(f)
            log(f"  {f}: MISSING")

    if missing:
        fail(f"Missing VAB files: {missing}")

    # Check p_webrl.json is valid
    with open(os.path.join(vab_dir, "agent", "prompts", "jsons", "p_webrl.json")) as f:
        prompt_config = json.load(f)
    log(f"  p_webrl.json: valid JSON ({len(prompt_config)} keys)")

    # Check run.py has webrl observation type
    with open(os.path.join(vab_dir, "run.py"), encoding="utf-8", errors="replace") as f:
        run_content = f.read()
    if '"webrl"' in run_content:
        log("  run.py: 'webrl' observation type found")
    else:
        fail("run.py missing 'webrl' observation type")

    log("Step 9 PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Step 10: End-to-end data format cross-check
# ═══════════════════════════════════════════════════════════════════
def step10_cross_check():
    log("Step 10: Cross-checking data formats between A and C...")

    import torch
    from webrl.data.utils import ReplayBuffer

    data_a = torch.load(os.path.join(TEST_DIR, "data_A.pt"), weights_only=False)
    data_c = torch.load(os.path.join(TEST_DIR, "data_C.pt"), weights_only=False)

    # Both should produce flat steps compatible with the same ReplayBuffer
    buf = ReplayBuffer(batch_size=2, capacity=10000)

    steps_a = sum(data_a, [])
    steps_c = sum(data_c, [])

    for s in steps_a + steps_c:
        if not s["action"].endswith("<|eot_id|>"):
            s["action"] += "<|eot_id|>"
        buf.insert(**s)

    log(f"  Combined buffer: {buf.size} steps ({len(steps_a)} from A, {len(steps_c)} from C)")

    # Verify we can sample from the combined buffer
    sample = buf.sample(batch_size=min(2, buf.size))
    assert sample["observation"].shape[0] == min(2, buf.size)
    log("  Combined sampling: OK")

    # Verify DataLoader works
    from webrl.data.utils import DummyDataset
    from torch.utils.data import DataLoader

    data = [buf.get(i) for i in range(min(4, buf.size))]
    for d in data:
        for k, v in d.items():
            d[k] = v[0]

    dataset = DummyDataset(data)
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    assert "observation" in batch
    assert "mc_return" in batch
    log("  DataLoader iteration: OK")

    log("Step 10 PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("LOCAL PIPELINE TEST")
    print("=" * 60)
    print(f"WebRL dir: {WEBRL_DIR}")
    print(f"Test dir:  {TEST_DIR}")
    print()

    cleanup()
    os.makedirs(TEST_DIR, exist_ok=True)

    try:
        step1_imports()
        step2_create_traces()
        step3_condition_a()
        step4_condition_c()
        step5_data_format()
        step6_model_loading()
        step7_evaluation()
        step8_collect_results()
        step9_vab_imports()
        step10_cross_check()

        print()
        print("=" * 60)
        print("ALL STEPS PASSED")
        print("=" * 60)
        print()
        print("Verified:")
        print("  + All imports (webrl, scripts, packages)")
        print("  + Condition A data processing (trace_process + build_policy_data)")
        print("  + Condition C data processing (milestones + regret, mocked)")
        print("  + .pt file format compatibility")
        print("  + Model loading + forward pass (GPT-2 on CPU)")
        print("  + Evaluation metrics computation")
        print("  + Result collection logic")
        print("  + VAB-WebArena-Lite file structure")
        print("  + Cross-format compatibility (A ↔ C)")
        print()
        print("NOT tested (requires RunPod):")
        print("  - DeepSpeed / torchrun distributed training")
        print("  - Actual WebArena Docker rollouts")
        print("  - Real Claude API calls for milestones")
        print("  - vLLM model serving")
        print()

    finally:
        cleanup()
        log("Cleaned up test artifacts.")
