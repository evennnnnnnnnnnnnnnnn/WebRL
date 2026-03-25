#!/bin/bash
# =============================================================
# Full A vs C experiment runner for RunPod A100
#
# Usage: bash scripts/run_experiment.sh
#
# Prerequisites:
#   - RunPod A100 80GB instance with Docker
#   - This repo cloned with our scripts + .env in place
#   - ANTHROPIC_API_KEY set in .env
#
# Each phase logs timing. If a phase fails, the script stops.
# =============================================================

set -e  # Stop on any error

WEBRL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WEBRL_DIR"

LOG_FILE="$WEBRL_DIR/experiment_log.txt"
RESULTS_DIR="$WEBRL_DIR/results"
CHECKPOINTS_DIR="$WEBRL_DIR/checkpoints"
TRACES_DIR="$WEBRL_DIR/traces_output"

mkdir -p "$RESULTS_DIR" "$CHECKPOINTS_DIR" "$TRACES_DIR"

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

# =============================================================
# Phase 1: Setup
# =============================================================
phase1_setup() {
    log "Installing dependencies..."
    pip install -e . --quiet
    pip install anthropic python-dotenv --quiet

    log "Downloading models..."
    # SFT base model (actor + reference)
    huggingface-cli download THUDM/webrl-llama-3.1-8b --quiet
    # ORM for Condition A
    huggingface-cli download THUDM/webrl-orm-llama-3.1-8b --quiet

    log "Verifying GPU..."
    nvidia-smi

    log "Phase 1 complete."
}

# =============================================================
# Phase 2: Sanity Check
# =============================================================
phase2_sanity() {
    log "Running sanity check: 10 tasks, k=3..."

    # TODO: Replace with actual VAB-WebArena-Lite rollout command
    # This is a placeholder — you need to:
    # 1. Clone VAB-WebArena-Lite
    # 2. Set up WebArena Docker containers
    # 3. Run the evaluation script
    #
    # Example:
    # cd /path/to/VAB-WebArena-Lite
    # python run_eval.py \
    #   --model THUDM/webrl-llama-3.1-8b \
    #   --tasks sample_10.json \
    #   --output_dir $TRACES_DIR/sanity \
    #   --max_tasks 10

    log "MANUAL CHECK: Verify pass@3 > 0 on at least 2-3 tasks before continuing."
    log "If pass@3 = 0 on everything, STOP and debug."
    log "Phase 2 complete."
}

# =============================================================
# Phase 3: Rollouts
# =============================================================
phase3_rollouts() {
    log "Running rollouts: 250 tasks x 3 attempts..."

    # TODO: Replace with actual VAB-WebArena-Lite rollout command
    # Parallelize across sites for speed:
    #
    # python run_eval.py --model THUDM/webrl-llama-3.1-8b \
    #   --tasks training_tasks_150.json \
    #   --output_dir $TRACES_DIR/train \
    #   --k_attempts 3 &
    #
    # python run_eval.py --model THUDM/webrl-llama-3.1-8b \
    #   --tasks heldout_tasks_100.json \
    #   --output_dir $TRACES_DIR/eval \
    #   --k_attempts 3 &
    #
    # wait

    log "Phase 3 complete. Traces in $TRACES_DIR"
}

# =============================================================
# Phase 4: Data Processing
# =============================================================
phase4_process() {
    local start_a=$(date +%s)

    # Condition A: WebRL pipeline (ORM binary reward + perplexity filter)
    log "Processing Condition A data..."
    python scripts/process_data.py \
        --stage 1 \
        --rollout_path "$TRACES_DIR/train" \
        --add_reward \
        --orm_path THUDM/webrl-orm-llama-3.1-8b \
        --output_path "$RESULTS_DIR/data_A.pt"

    # Stage 2: perplexity filtering
    python scripts/process_data.py \
        --stage 2 \
        --experience_paths "$RESULTS_DIR/data_A.pt" \
        --actor_path THUDM/webrl-llama-3.1-8b \
        --output_path "$RESULTS_DIR/data_A.pt"

    local end_a=$(date +%s)
    local duration_a=$((end_a - start_a))
    log "Condition A data processing: ${duration_a}s"
    echo "data_processing_A=$duration_a" >> "$RESULTS_DIR/timings.txt"

    local start_c=$(date +%s)

    # Condition C: Milestone rewards + regret selection
    log "Processing Condition C data..."
    python scripts/process_data_ours.py \
        --rollout_path "$TRACES_DIR/train" \
        --output_path "$RESULTS_DIR/data_C.pt" \
        --k_attempts 3 \
        --top_fraction 0.5 \
        --milestones_cache "$RESULTS_DIR/milestones_cache.json"

    local end_c=$(date +%s)
    local duration_c=$((end_c - start_c))
    log "Condition C data processing: ${duration_c}s"
    echo "data_processing_C=$duration_c" >> "$RESULTS_DIR/timings.txt"

    # Validate both outputs
    log "Validating .pt files..."
    python tests/test_data_format.py "$RESULTS_DIR/data_A.pt" "$RESULTS_DIR/data_C.pt"

    log "Phase 4 complete."
}

# =============================================================
# Phase 5: Training
# =============================================================
phase5_train() {
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
    local duration_a=$((end_a - start_a))
    log "Condition A training: ${duration_a}s"
    echo "training_A=$duration_a" >> "$RESULTS_DIR/timings.txt"

    local start_c=$(date +%s)

    # Condition C: Critic-free (actor + reference only)
    log "Training Condition C (critic-free)..."
    # TODO: Wire train_no_critic.py with proper Hydra config + entry point
    # torchrun --nproc_per_node 1 scripts/train_no_critic.py \
    #     --offline_data_path "$RESULTS_DIR/data_C.pt" \
    #     --save_path "$CHECKPOINTS_DIR/model_C"

    local end_c=$(date +%s)
    local duration_c=$((end_c - start_c))
    log "Condition C training: ${duration_c}s"
    echo "training_C=$duration_c" >> "$RESULTS_DIR/timings.txt"

    log "Phase 5 complete."
}

# =============================================================
# Phase 6: Evaluation
# =============================================================
phase6_eval() {
    log "Evaluating both models on 100 held-out tasks x 3 attempts..."

    # TODO: Replace with actual VAB-WebArena-Lite evaluation
    # for model in model_A model_C; do
    #   python run_eval.py \
    #     --model $CHECKPOINTS_DIR/$model/actor \
    #     --tasks heldout_tasks_100.json \
    #     --output_dir $TRACES_DIR/eval_$model \
    #     --k_attempts 3
    # done

    # Compute comparison (after collecting results into JSON)
    # python scripts/evaluate.py compare \
    #     --a $RESULTS_DIR/model_A_eval.json \
    #     --c $RESULTS_DIR/model_C_eval.json

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
    log "Download this file, then DESTROY the instance."

    # Print final summary
    echo ""
    echo "=========================================="
    echo "EXPERIMENT COMPLETE"
    echo "=========================================="
    cat "$RESULTS_DIR/timings.txt"
    echo ""
    log "Phase 7 complete."
}

# =============================================================
# Main
# =============================================================
log "=========================================="
log "WebRL A vs C Experiment"
log "=========================================="

time_phase "Phase 1: Setup" phase1_setup
time_phase "Phase 2: Sanity Check" phase2_sanity

echo ""
log ">>> GATE: Review sanity check results above."
log ">>> Press Enter to continue with full experiment, or Ctrl+C to abort."
read -r

time_phase "Phase 3: Rollouts" phase3_rollouts
time_phase "Phase 4: Data Processing" phase4_process
time_phase "Phase 5: Training" phase5_train
time_phase "Phase 6: Evaluation" phase6_eval
time_phase "Phase 7: Package Results" phase7_package
