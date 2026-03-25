#!/bin/bash
# =============================================================
# RunPod A100 Full Setup Script
#
# Sets up everything needed for the A vs C experiment:
#   1. WebArena Docker containers (shopping, reddit, gitlab)
#   2. VAB-WebArena-Lite (rollout/eval framework)
#   3. vLLM model server (for agent inference)
#   4. Environment variables
#
# Usage: bash scripts/setup_runpod.sh
# Expected time: ~20-30 min (mostly Docker pulls + model download)
# =============================================================

set -e

log() {
    echo "[SETUP $(date '+%H:%M:%S')] $1"
}

WEBRL_DIR="/WebRL"

# =============================================================
# WebArena Site URLs (local machine via Tailscale)
# Docker containers run on local PC, accessed over Tailscale mesh.
# =============================================================
LOCAL_IP="100.92.2.51"
TAILSCALE_SOCKS5="socks5://localhost:1055"
SHOPPING_URL="http://${LOCAL_IP}:7770"
SHOPPING_ADMIN_URL="http://${LOCAL_IP}:7780"
REDDIT_URL="http://${LOCAL_IP}:9999"
GITLAB_URL="http://${LOCAL_IP}:8023"

# Route traffic to local PC through Tailscale SOCKS5 proxy
# (userspace-networking mode requires this)
export ALL_PROXY="$TAILSCALE_SOCKS5"
export HTTP_PROXY="$TAILSCALE_SOCKS5"
export HTTPS_PROXY="$TAILSCALE_SOCKS5"
export NO_PROXY="localhost,127.0.0.1"

# =============================================================
# Step 1: Verify Remote WebArena Containers
# =============================================================
log "Step 1: Verifying remote WebArena containers (local PC via Tailscale)..."

for name_url in "shopping:${SHOPPING_URL}" "shopping_admin:${SHOPPING_ADMIN_URL}" "reddit:${REDDIT_URL}" "gitlab:${GITLAB_URL}"; do
    name="${name_url%%:*}"
    url="${name_url#*:}"
    if curl -s --socks5 localhost:1055 -o /dev/null -w "%{http_code}" --max-time 15 "$url" | grep -qE "200|302|301"; then
        log "    ✓ ${name}: OK ($url)"
    else
        log "    ✗ ${name}: NOT REACHABLE ($url)"
        log "      Check: Docker running on local PC? Tailscale connected? Firewall open?"
    fi
done

log "Step 1 complete."

# =============================================================
# Step 2: VAB-WebArena-Lite Setup
# =============================================================
log "Step 2: Setting up VAB-WebArena-Lite..."

cd /
if [ ! -d "VisualAgentBench" ]; then
    git clone https://github.com/THUDM/VisualAgentBench.git
fi

cd /VisualAgentBench/VAB-WebArena-Lite

# Clone visualwebarena base (specific commit)
if [ ! -d "visualwebarena" ]; then
    git clone https://github.com/web-arena-x/visualwebarena.git visualwebarena
    cd visualwebarena
    git reset --hard ad57aae4dad71531504726900b80db02e0526158
    cd ..
fi

# Apply VAB patches
if [ -f "replace.sh" ]; then
    bash replace.sh
fi

# Install dependencies
pip install -r requirements.txt --quiet 2>/dev/null
playwright install chromium --with-deps 2>/dev/null

# Set environment variables (tunnel URLs to local Docker containers)
export DATASET=webarena
export SHOPPING="$SHOPPING_URL"
export SHOPPING_ADMIN="${SHOPPING_ADMIN_URL}/admin"
export REDDIT="$REDDIT_URL"
export GITLAB="$GITLAB_URL"
export MAP=""
export WIKIPEDIA=""
export HOMEPAGE=""

# Save env vars for later use
cat > /tmp/webarena_env.sh << EOF
export DATASET=webarena
export SHOPPING="$SHOPPING_URL"
export SHOPPING_ADMIN="${SHOPPING_ADMIN_URL}/admin"
export REDDIT="$REDDIT_URL"
export GITLAB="$GITLAB_URL"
export MAP=""
export WIKIPEDIA=""
export HOMEPAGE=""
# Tailscale SOCKS5 proxy (userspace-networking mode)
export ALL_PROXY="socks5://localhost:1055"
export HTTP_PROXY="socks5://localhost:1055"
export HTTPS_PROXY="socks5://localhost:1055"
export NO_PROXY="localhost,127.0.0.1"
EOF

# Generate task configs
log "  Generating task configs..."
python scripts/generate_test_data.py 2>/dev/null || log "  Warning: generate_test_data.py may need manual config"

# Generate auth cookies
log "  Generating auth cookies..."
bash prepare.sh 2>/dev/null || log "  Warning: prepare.sh may need manual setup"

log "Step 2 complete."

# =============================================================
# Step 3: Install vLLM and serve model
# =============================================================
log "Step 3: Setting up vLLM model server..."

pip install vllm --quiet 2>/dev/null

# Start vLLM server in background
log "  Starting vLLM server for THUDM/webrl-llama-3.1-8b on port 8000..."
python -m vllm.entrypoints.openai.api_server \
    --model THUDM/webrl-llama-3.1-8b \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --trust-remote-code \
    > /tmp/vllm_server.log 2>&1 &

VLLM_PID=$!
echo $VLLM_PID > /tmp/vllm_pid.txt

# Wait for vLLM to be ready
log "  Waiting for vLLM to load model (this takes a few minutes)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health | grep -q "ok\|200" 2>/dev/null; then
        log "  ✓ vLLM server ready (PID: $VLLM_PID)"
        break
    fi
    if [ $i -eq 60 ]; then
        log "  ✗ vLLM server not ready after 5 min. Check /tmp/vllm_server.log"
        tail -20 /tmp/vllm_server.log
    fi
    sleep 5
done

# Save vLLM URL for experiment scripts
echo "http://localhost:8000" > /tmp/vllm_url.txt

log "Step 3 complete."

# =============================================================
# Step 4: WebRL repo setup
# =============================================================
log "Step 4: Finalizing WebRL setup..."

cd "$WEBRL_DIR"

# Install WebRL deps (no-deps to avoid the bloated setup.py)
pip install -e . --no-deps --quiet 2>/dev/null
pip install anthropic python-dotenv --quiet 2>/dev/null

# Create results directories
mkdir -p results checkpoints traces_output

log "Step 4 complete."

# =============================================================
# Step 5: Smoke test
# =============================================================
log "Step 5: Quick smoke test..."

# Test vLLM inference
python -c "
import requests
resp = requests.post('http://localhost:8000/v1/completions', json={
    'model': 'THUDM/webrl-llama-3.1-8b',
    'prompt': 'Task Instruction: Click on the search button.\n\nRound 0\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n<html>Search page</html>\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n',
    'max_tokens': 64,
    'temperature': 1.0,
})
result = resp.json()
print(f'vLLM inference test: {result[\"choices\"][0][\"text\"][:80]}...')
print('✓ Model inference OK')
" || log "  ✗ vLLM inference test failed"

# Test WebArena container access (via tunnel URLs)
python -c "
import requests
for name, url in [('shopping', '$SHOPPING_URL'), ('reddit', '$REDDIT_URL'), ('gitlab', '$GITLAB_URL')]:
    try:
        r = requests.get(url, timeout=10)
        print(f'✓ {name}: HTTP {r.status_code}')
    except Exception as e:
        print(f'✗ {name}: {e}')
" || log "  ✗ Container access test failed"

log "Step 5 complete."

# =============================================================
# Summary
# =============================================================
echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "WebArena containers (local machine via Tailscale):"
echo "  Shopping:       $SHOPPING_URL"
echo "  Shopping Admin: $SHOPPING_ADMIN_URL"
echo "  Reddit:         $REDDIT_URL"
echo "  GitLab:         $GITLAB_URL"
echo ""
echo "vLLM server:      http://localhost:8000"
echo "  PID: $(cat /tmp/vllm_pid.txt)"
echo "  Log: /tmp/vllm_server.log"
echo ""
echo "Environment:      source /tmp/webarena_env.sh"
echo ""
echo "Next steps:"
echo "  1. source /tmp/webarena_env.sh"
echo "  2. bash scripts/run_experiment.sh"
echo ""
echo "To stop vLLM:     kill \$(cat /tmp/vllm_pid.txt)"
echo "To check vLLM:    tail -f /tmp/vllm_server.log"
echo "=========================================="
