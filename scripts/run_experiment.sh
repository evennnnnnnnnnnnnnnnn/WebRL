#!/bin/bash
# =============================================================
# Full A vs C experiment runner for RunPod A100
#
# Prerequisites:
#   - Run setup_runpod.sh first
#   - source /tmp/webarena_env.sh
#   - vLLM server running on port 8000
#
# Usage: bash scripts/run_experiment.sh
# =============================================================

set -e

WEBRL_DIR="/WebRL"
VAB_DIR="/VisualAgentBench/VAB-WebArena-Lite"
cd "$WEBRL_DIR"

LOG_FILE="$WEBRL_DIR/experiment_log.txt"
RESULTS_DIR="$WEBRL_DIR/results"
CHECKPOINTS_DIR="$WEBRL_DIR/checkpoints"
TRACES_DIR="$WEBRL_DIR/traces_output"
VLLM_URL="http://localhost:8000"
VLLM_MODEL="THUDM/webrl-llama-3.1-8b"

mkdir -p "$RESULTS_DIR" "$CHECKPOINTS_DIR" "$TRACES_DIR"

# Load WebArena env vars
source /tmp/webarena_env.sh

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

time_phase() {
    local phase_name="$1"
    shift
    local start_time=$(date +%s)
    log "START: $phase_name"
    "$@"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log "DONE: $phase_name (${duration}s / $((duration/60))min)"
    echo "$phase_name=$duration" >> "$RESULTS_DIR/timings.txt"
}

# Helper: run VAB-WebArena-Lite rollout
run_rollout() {
    local result_dir="$1"
    local start_idx="$2"
    local end_idx="$3"
    local attempt_label="$4"

    cd "$VAB_DIR"
    python run.py \
        --instruction_path agent/prompts/jsons/p_webrl.json \
        --test_start_idx "$start_idx" \
        --test_end_idx "$end_idx" \
        --result_dir "$result_dir" \
        --test_config_base_dir config_files/wa/test_webarena_lite \
        --provider openai \
        --mode completion \
        --model "$VLLM_MODEL" \
        --planner_ip "localhost" \
        --stop_token "<|eot_id|>" \
        --max_obs_length 0 \
        --max_tokens 2048 \
        --viewport_width 1280 \
        --viewport_height 720 \
        --action_set_tag webrl_id \
        --observation_type webrl
    cd "$WEBRL_DIR"
}

# Helper: run rollout with a fine-tuned model served on vLLM
run_rollout_finetuned() {
    local model_path="$1"
    local result_dir="$2"
    local start_idx="$3"
    local end_idx="$4"

    # Kill existing vLLM, start with fine-tuned model
    kill $(cat /tmp/vllm_pid.txt) 2>/dev/null || true
    sleep 5

    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port 8000 \
        --tensor-parallel-size 1 \
        --max-model-len 16384 \
        --dtype bfloat16 \
        --trust-remote-code \
        > /tmp/vllm_server.log 2>&1 &
    echo $! > /tmp/vllm_pid.txt

    # Wait for vLLM ready
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/health 2>/dev/null | grep -q "ok\|200"; then
            break
        fi
        sleep 5
    done

    cd "$VAB_DIR"
    python run.py \
        --instruction_path agent/prompts/jsons/p_webrl.json \
        --test_start_idx "$start_idx" \
        --test_end_idx "$end_idx" \
        --result_dir "$result_dir" \
        --test_config_base_dir config_files/wa/test_webarena_lite \
        --provider openai \
        --mode completion \
        --model "$model_path" \
        --planner_ip "localhost" \
        --stop_token "<|eot_id|>" \
        --max_obs_length 0 \
        --max_tokens 2048 \
        --viewport_width 1280 \
        --viewport_height 720 \
        --action_set_tag webrl_id \
        --observation_type webrl
    cd "$WEBRL_DIR"
}

# Helper: refresh WebArena containers between attempts
# NOTE: Containers run on the LOCAL machine, not on this GPU server.
# To refresh, either:
#   1. SSH back to local machine and run docker exec, or
#   2. Skip refresh (acceptable for small-scale experiments)
# Set LOCAL_SSH to enable remote refresh, e.g.: LOCAL_SSH="user@your-ip"
LOCAL_SSH="${LOCAL_SSH:-}"

refresh_containers() {
    if [ -n "$LOCAL_SSH" ]; then
        log "  Refreshing WebArena containers via SSH ($LOCAL_SSH)..."
        for container in shopping shopping_admin reddit gitlab; do
            ssh "$LOCAL_SSH" "docker exec $container env-ctrl init" 2>/dev/null || true
        done
        sleep 10
    else
        log "  Skipping container refresh (containers are remote, LOCAL_SSH not set)"
        log "  Set LOCAL_SSH=user@host to enable remote refresh"
    fi
}

# Helper: collect results into JSON for evaluate.py
collect_results() {
    local traces_dir="$1"
    local output_json="$2"
    local model_name="$3"

    python -c "
import json, os, glob
from datetime import datetime

traces_dir = '${traces_dir}'
actions_dir = os.path.join(traces_dir, 'actions') if os.path.exists(os.path.join(traces_dir, 'actions')) else traces_dir

results = []
# Find all action score files
for f in sorted(glob.glob(os.path.join(actions_dir, '*.json'))):
    with open(f) as fp:
        data = json.load(fp)
    task_id = os.path.basename(f).replace('.json', '')
    score = data.get('score', 0)
    results.append({
        'task_id': task_id,
        'site': 'unknown',
        'score': score,
    })

# Group by task_id prefix for pass@k
from collections import defaultdict
task_attempts = defaultdict(list)
for r in results:
    # Extract base task id (remove attempt suffix if any)
    base_id = r['task_id'].rsplit('_attempt', 1)[0]
    task_attempts[base_id].append(r['score'] >= 0.5)

task_results = []
for task_id, attempts in task_attempts.items():
    task_results.append({
        'task_id': task_id,
        'site': 'unknown',
        'attempts': attempts,
    })

output = {
    'model': '${model_name}',
    'timestamp': datetime.now().isoformat(),
    'task_results': task_results,
}
with open('${output_json}', 'w') as f:
    json.dump(output, f, indent=2)
print(f'Collected {len(task_results)} tasks, {len(results)} total attempts')
"
}

# =============================================================
# Phase 1: Verify Setup
# =============================================================
phase1_verify() {
    log "Verifying setup..."

    # Check vLLM
    curl -s http://localhost:8000/health > /dev/null || {
        log "ERROR: vLLM not running. Run setup_runpod.sh first."
        exit 1
    }
    log "  ✓ vLLM server"

    # Check containers (local PC via Tailscale SOCKS5)
    for name_url in "shopping:$SHOPPING" "reddit:$REDDIT" "gitlab:$GITLAB"; do
        name="${name_url%%:*}"
        url="${name_url#*:}"
        curl -s --socks5 localhost:1055 --max-time 15 -o /dev/null "$url" || {
            log "ERROR: $name not reachable at $url. Check Tailscale + Docker on local PC."
            exit 1
        }
    done
    log "  ✓ WebArena containers (via Tailscale)"

    # Check env vars
    [ -n "$SHOPPING" ] || { log "ERROR: source /tmp/webarena_env.sh first"; exit 1; }
    log "  ✓ Environment variables"

    log "Phase 1 complete."
}

# =============================================================
# Phase 2: Sanity Check (10 tasks, 1 attempt)
# =============================================================
phase2_sanity() {
    log "Running sanity check: 10 tasks..."

    run_rollout "$TRACES_DIR/sanity" 0 10 "sanity"

    # Check scores
    python -c "
import json, glob, os
actions = glob.glob('$TRACES_DIR/sanity/actions/*.json')
scores = []
for f in actions:
    with open(f) as fp:
        scores.append(json.load(fp).get('score', 0))
successes = sum(1 for s in scores if s >= 0.5)
print(f'Sanity check: {successes}/{len(scores)} tasks succeeded')
if successes == 0:
    print('WARNING: No successes. Model may be too weak or setup broken.')
"

    log "Phase 2 complete."
}

# =============================================================
# Phase 3: Rollouts (training + eval tasks)
# =============================================================
phase3_rollouts() {
    log "Running training rollouts..."

    # WebArena-Lite has 165 tasks (indices 0-164)
    # Use 0-99 for training (100 tasks), 100-164 for eval (65 tasks)
    # Run k=3 attempts per task

    for attempt in 1 2 3; do
        log "  Training rollout attempt $attempt/3..."
        refresh_containers
        run_rollout "$TRACES_DIR/train_attempt_${attempt}" 0 100 "train_${attempt}"
    done

    log "Running evaluation rollouts (base model)..."
    for attempt in 1 2 3; do
        log "  Eval rollout attempt $attempt/3..."
        refresh_containers
        run_rollout "$TRACES_DIR/eval_base_attempt_${attempt}" 100 165 "eval_base_${attempt}"
    done

    log "Phase 3 complete."
}

# =============================================================
# Phase 4: Data Processing
# =============================================================
phase4_process() {
    local start_a=$(date +%s)

    # Merge training traces into single directory
    log "Merging training traces..."
    mkdir -p "$TRACES_DIR/train_merged/traces" "$TRACES_DIR/train_merged/actions"
    for attempt in 1 2 3; do
        src="$TRACES_DIR/train_attempt_${attempt}"
        if [ -d "$src/traces" ]; then
            for f in "$src/traces"/*.jsonl; do
                base=$(basename "$f" .jsonl)
                cp "$f" "$TRACES_DIR/train_merged/traces/${base}_attempt${attempt}.jsonl"
            done
        fi
        if [ -d "$src/actions" ]; then
            for f in "$src/actions"/*.json; do
                base=$(basename "$f" .json)
                cp "$f" "$TRACES_DIR/train_merged/actions/${base}_attempt${attempt}.json"
            done
        fi
    done

    # Condition A: WebRL pipeline
    log "Processing Condition A data..."
    python scripts/process_data.py \
        --stage 1 \
        --rollout_path "$TRACES_DIR/train_merged" \
        --add_reward \
        --orm_path THUDM/webrl-orm-llama-3.1-8b \
        --output_path "$RESULTS_DIR/data_A.pt"

    python scripts/process_data.py \
        --stage 2 \
        --experience_paths "$RESULTS_DIR/data_A.pt" \
        --actor_path THUDM/webrl-llama-3.1-8b \
        --output_path "$RESULTS_DIR/data_A.pt"

    local end_a=$(date +%s)
    echo "data_processing_A=$((end_a - start_a))" >> "$RESULTS_DIR/timings.txt"
    log "Condition A processing: $((end_a - start_a))s"

    local start_c=$(date +%s)

    # Condition C: Milestone + regret
    log "Processing Condition C data..."
    python scripts/process_data_ours.py \
        --rollout_path "$TRACES_DIR/train_merged" \
        --output_path "$RESULTS_DIR/data_C.pt" \
        --k_attempts 3 \
        --top_fraction 0.5 \
        --milestones_cache "$RESULTS_DIR/milestones_cache.json"

    local end_c=$(date +%s)
    echo "data_processing_C=$((end_c - start_c))" >> "$RESULTS_DIR/timings.txt"
    log "Condition C processing: $((end_c - start_c))s"

    # Validate
    log "Validating .pt files..."
    python tests/test_data_format.py "$RESULTS_DIR/data_A.pt" "$RESULTS_DIR/data_C.pt"

    log "Phase 4 complete."
}

# =============================================================
# Phase 5: Training
# =============================================================
phase5_train() {
    # Stop vLLM to free GPU memory for training
    log "Stopping vLLM to free GPU memory..."
    kill $(cat /tmp/vllm_pid.txt) 2>/dev/null || true
    sleep 10

    local start_a=$(date +%s)

    # Condition A: Full WebRL (actor + critic + reference)
    log "Training Condition A (full WebRL)..."
    cd scripts
    torchrun --nproc_per_node 1 run.py \
        --config_path config/main \
        --config_name webrl \
        --output_dir "$CHECKPOINTS_DIR/model_A"
    cd "$WEBRL_DIR"

    local end_a=$(date +%s)
    echo "training_A=$((end_a - start_a))" >> "$RESULTS_DIR/timings.txt"
    log "Condition A training: $((end_a - start_a))s"

    local start_c=$(date +%s)

    # Condition C: Critic-free (actor + reference only)
    log "Training Condition C (critic-free)..."
    cd scripts
    torchrun --nproc_per_node 1 train_no_critic.py \
        --config_path config/main \
        --config_name webrl \
        --output_dir "$CHECKPOINTS_DIR/model_C"
    cd "$WEBRL_DIR"

    local end_c=$(date +%s)
    echo "training_C=$((end_c - start_c))" >> "$RESULTS_DIR/timings.txt"
    log "Condition C training: $((end_c - start_c))s"

    log "Phase 5 complete."
}

# =============================================================
# Phase 6: Evaluation
# =============================================================
phase6_eval() {
    log "Evaluating both models..."

    # Eval Model A
    log "Evaluating Model A..."
    for attempt in 1 2 3; do
        refresh_containers
        run_rollout_finetuned \
            "$CHECKPOINTS_DIR/model_A/actor" \
            "$TRACES_DIR/eval_A_attempt_${attempt}" \
            100 165
    done

    # Collect Model A results
    mkdir -p "$TRACES_DIR/eval_A_merged/actions"
    for attempt in 1 2 3; do
        src="$TRACES_DIR/eval_A_attempt_${attempt}"
        if [ -d "$src/actions" ]; then
            for f in "$src/actions"/*.json; do
                base=$(basename "$f" .json)
                cp "$f" "$TRACES_DIR/eval_A_merged/actions/${base}_attempt${attempt}.json"
            done
        fi
    done
    collect_results "$TRACES_DIR/eval_A_merged" "$RESULTS_DIR/eval_A.json" "model_A"

    # Eval Model C
    log "Evaluating Model C..."
    for attempt in 1 2 3; do
        refresh_containers
        run_rollout_finetuned \
            "$CHECKPOINTS_DIR/model_C/actor" \
            "$TRACES_DIR/eval_C_attempt_${attempt}" \
            100 165
    done

    # Collect Model C results
    mkdir -p "$TRACES_DIR/eval_C_merged/actions"
    for attempt in 1 2 3; do
        src="$TRACES_DIR/eval_C_attempt_${attempt}"
        if [ -d "$src/actions" ]; then
            for f in "$src/actions"/*.json; do
                base=$(basename "$f" .json)
                cp "$f" "$TRACES_DIR/eval_C_merged/actions/${base}_attempt${attempt}.json"
            done
        fi
    done
    collect_results "$TRACES_DIR/eval_C_merged" "$RESULTS_DIR/eval_C.json" "model_C"

    # Compare
    log "Computing comparison..."
    python scripts/evaluate.py compare \
        --a "$RESULTS_DIR/eval_A.json" \
        --c "$RESULTS_DIR/eval_C.json" | tee "$RESULTS_DIR/comparison.txt"

    log "Phase 6 complete."
}

# =============================================================
# Phase 7: Package Results
# =============================================================
phase7_package() {
    log "Packaging results..."

    tar czf "$WEBRL_DIR/experiment_results.tar.gz" \
        -C "$WEBRL_DIR" \
        results/ \
        experiment_log.txt

    log "Results packaged: experiment_results.tar.gz"

    echo ""
    echo "=========================================="
    echo "EXPERIMENT COMPLETE"
    echo "=========================================="
    echo ""
    echo "Timings:"
    cat "$RESULTS_DIR/timings.txt"
    echo ""
    echo "Results:"
    cat "$RESULTS_DIR/comparison.txt" 2>/dev/null
    echo ""
    echo "Download: scp runpod:/WebRL/experiment_results.tar.gz ."
    echo "Then DESTROY the instance."
    echo "=========================================="
}

# =============================================================
# Main
# =============================================================
log "=========================================="
log "WebRL A vs C Experiment"
log "=========================================="

time_phase "Phase 1: Verify Setup" phase1_verify
time_phase "Phase 2: Sanity Check" phase2_sanity

echo ""
log ">>> Sanity check complete. Continuing automatically..."

time_phase "Phase 3: Rollouts" phase3_rollouts
time_phase "Phase 4: Data Processing" phase4_process
time_phase "Phase 5: Training" phase5_train
time_phase "Phase 6: Evaluation" phase6_eval
time_phase "Phase 7: Package Results" phase7_package
