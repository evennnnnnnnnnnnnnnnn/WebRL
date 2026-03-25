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
HOSTNAME="localhost"

# =============================================================
# Step 1: WebArena Docker Containers
# =============================================================
log "Step 1: Starting WebArena Docker containers..."

# Shopping (Magento e-commerce)
log "  Starting shopping (port 7770)..."
docker run -d --name shopping -p 7770:80 am1n3e/webarena-verified-shopping 2>/dev/null || \
    docker start shopping

# Shopping Admin (CMS)
log "  Starting shopping_admin (port 7780)..."
docker run -d --name shopping_admin -p 7780:80 am1n3e/webarena-verified-shopping_admin 2>/dev/null || \
    docker start shopping_admin

# Reddit (Postmill forum)
log "  Starting reddit (port 9999)..."
docker run -d --name reddit -p 9999:80 am1n3e/webarena-verified-reddit 2>/dev/null || \
    docker start reddit

# GitLab
log "  Starting gitlab (port 8023)..."
docker run -d --name gitlab -p 8023:8023 am1n3e/webarena-verified-gitlab 2>/dev/null || \
    docker start gitlab

# Wait for containers to be ready
log "  Waiting for containers to start (60s)..."
sleep 60

# Verify containers
log "  Verifying containers..."
for port_name in "7770:shopping" "7780:shopping_admin" "9999:reddit" "8023:gitlab"; do
    port="${port_name%%:*}"
    name="${port_name##*:}"
    if curl -s -o /dev/null -w "%{http_code}" "http://${HOSTNAME}:${port}" | grep -qE "200|302|301"; then
        log "    ✓ ${name} (port ${port}): OK"
    else
        log "    ✗ ${name} (port ${port}): NOT READY (may need more time)"
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

# Set environment variables
export DATASET=webarena
export SHOPPING="http://${HOSTNAME}:7770"
export SHOPPING_ADMIN="http://${HOSTNAME}:7780/admin"
export REDDIT="http://${HOSTNAME}:9999"
export GITLAB="http://${HOSTNAME}:8023"
export MAP="http://${HOSTNAME}:3000"
export WIKIPEDIA="http://${HOSTNAME}:8888"
export HOMEPAGE="http://${HOSTNAME}:4399"

# Save env vars for later use
cat > /tmp/webarena_env.sh << EOF
export DATASET=webarena
export SHOPPING="http://${HOSTNAME}:7770"
export SHOPPING_ADMIN="http://${HOSTNAME}:7780/admin"
export REDDIT="http://${HOSTNAME}:9999"
export GITLAB="http://${HOSTNAME}:8023"
export MAP="http://${HOSTNAME}:3000"
export WIKIPEDIA="http://${HOSTNAME}:8888"
export HOMEPAGE="http://${HOSTNAME}:4399"
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

# Test WebArena container access
python -c "
import requests
for name, url in [('shopping', 'http://localhost:7770'), ('reddit', 'http://localhost:9999'), ('gitlab', 'http://localhost:8023')]:
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
echo "WebArena containers:"
echo "  Shopping:       http://${HOSTNAME}:7770"
echo "  Shopping Admin: http://${HOSTNAME}:7780"
echo "  Reddit:         http://${HOSTNAME}:9999"
echo "  GitLab:         http://${HOSTNAME}:8023"
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
