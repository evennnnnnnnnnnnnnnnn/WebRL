#!/bin/bash
# =============================================================
# RunPod A100 Full Setup Script
#
# Sets up everything needed for the A vs C experiment:
#   1. Tailscale SOCKS5 proxy for reaching local WebArena containers
#   2. Python dependencies (numpy, httpx[socks], dashscope, etc.)
#   3. VAB-WebArena-Lite (rollout/eval framework)
#   4. vLLM model server (for agent inference)
#   5. WebRL repo setup
#
# Prerequisites:
#   - Tailscale running: nohup tailscaled --tun=userspace-networking \
#       --socks5-server=localhost:1055 > /var/log/tailscaled.log 2>&1 &
#   - Tailscale logged in: tailscale up --authkey=YOUR_KEY
#   - Docker containers running on local PC with firewall ports open
#
# Usage: bash scripts/setup_runpod.sh
# Expected time: ~20-30 min
# =============================================================

set -e

log() {
    echo "[SETUP $(date '+%H:%M:%S')] $1"
}

WEBRL_DIR="/WebRL"
VAB_DIR="/VisualAgentBench/VAB-WebArena-Lite"

# =============================================================
# WebArena Site URLs (local machine via Tailscale SOCKS5)
# =============================================================
LOCAL_IP="100.92.2.51"
SOCKS5_PROXY="socks5://localhost:1055"

SHOPPING_URL="http://${LOCAL_IP}:7770"
SHOPPING_ADMIN_URL="http://${LOCAL_IP}:7780"
REDDIT_URL="http://${LOCAL_IP}:9999"
GITLAB_URL="http://${LOCAL_IP}:8023"
# Dummy URLs for sites we don't use (VAB asserts they're non-empty)
MAP_URL="http://${LOCAL_IP}:3000"
WIKIPEDIA_URL="http://${LOCAL_IP}:8888"
HOMEPAGE_URL="http://${LOCAL_IP}:4399"

# =============================================================
# Step 0: Verify Tailscale is running
# =============================================================
log "Step 0: Checking Tailscale SOCKS5 proxy..."

if ! curl -s --socks5 localhost:1055 --max-time 10 -o /dev/null "$SHOPPING_URL" 2>/dev/null; then
    log "  WARNING: Cannot reach shopping at $SHOPPING_URL via SOCKS5"
    log "  Ensure: tailscaled running, tailscale up, Docker + firewall on local PC"
fi

for name_url in "shopping:${SHOPPING_URL}" "reddit:${REDDIT_URL}" "gitlab:${GITLAB_URL}"; do
    name="${name_url%%:*}"
    url="${name_url#*:}"
    if curl -s --socks5 localhost:1055 -o /dev/null -w "%{http_code}" --max-time 15 "$url" 2>/dev/null | grep -qE "200|302|301"; then
        log "  ✓ ${name}: OK"
    else
        log "  ✗ ${name}: NOT REACHABLE ($url)"
    fi
done

log "Step 0 complete."

# =============================================================
# Step 1: Install Python dependencies (NO proxy — direct internet)
# =============================================================
log "Step 1: Installing Python dependencies..."

# Unset any proxy to avoid interfering with pip/git
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY

# Check existing torch — don't overwrite it (avoids NCCL mismatch)
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
log "  Existing PyTorch: $TORCH_VERSION"

if [ "$TORCH_VERSION" = "none" ]; then
    pip install torch --quiet
fi

# Pin numpy<2 (matplotlib/VAB compiled against numpy 1.x)
pip install "numpy<2" --quiet

# Install core deps (skip torch to keep pod's version)
pip install transformers==4.44.2 accelerate==0.32.1 deepspeed==0.15.1 \
    hydra-core omegaconf datasets peft openai anthropic python-dotenv \
    wandb beautifulsoup4 sentencepiece tenacity termcolor tqdm \
    httpx[socks] dashscope \
    --quiet

# Install vLLM
pip install vllm --quiet

log "Step 1 complete."

# =============================================================
# Step 2: VAB-WebArena-Lite Setup (NO proxy for git/pip/playwright)
# =============================================================
log "Step 2: Setting up VAB-WebArena-Lite..."

cd /
if [ ! -d "VisualAgentBench" ]; then
    git clone https://github.com/THUDM/VisualAgentBench.git
fi

cd "$VAB_DIR"

# Clone visualwebarena base (specific commit)
if [ ! -d "visualwebarena" ]; then
    git clone https://github.com/web-arena-x/visualwebarena.git visualwebarena
    cd visualwebarena
    git reset --hard ad57aae4dad71531504726900b80db02e0526158
    cd ..
fi

# Apply VAB patches (adds webrl observation type + action set)
if [ -f "replace.sh" ]; then
    bash replace.sh
fi

# Patch run.py to add 'webrl' to argparse choices (replace.sh doesn't do this)
if ! grep -q '"webrl"' run.py 2>/dev/null; then
    sed -i 's/"image_som"/"image_som", "webrl"/' run.py
    log "  Patched run.py: added 'webrl' to observation_type choices"
fi

# Patch agent.py to handle missing planner_ip gracefully
if grep -q 'args.planner_ip' agent/agent.py 2>/dev/null; then
    sed -i 's/planner_ip=args.planner_ip/planner_ip=getattr(args, "planner_ip", "")/' agent/agent.py
    log "  Patched agent.py: planner_ip defaults to empty string"
fi

# Install VAB dependencies
pip install -r requirements.txt --quiet 2>/dev/null

# Install Playwright (needs direct internet, no proxy)
playwright install chromium --with-deps 2>/dev/null

log "Step 2 complete."

# =============================================================
# Step 3: Enable SOCKS5 proxy + set environment variables
# =============================================================
log "Step 3: Configuring environment..."

export ALL_PROXY="$SOCKS5_PROXY"
export HTTP_PROXY="$SOCKS5_PROXY"
export HTTPS_PROXY="$SOCKS5_PROXY"
export NO_PROXY="localhost,127.0.0.1"

export DATASET=webarena
export SHOPPING="$SHOPPING_URL"
export SHOPPING_ADMIN="${SHOPPING_ADMIN_URL}/admin"
export REDDIT="$REDDIT_URL"
export GITLAB="$GITLAB_URL"
export MAP="$MAP_URL"
export WIKIPEDIA="$WIKIPEDIA_URL"
export HOMEPAGE="$HOMEPAGE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# Save env vars for later use (sourced by run_experiment.sh)
cat > /tmp/webarena_env.sh << 'ENVEOF'
export DATASET=webarena
export SHOPPING="http://100.92.2.51:7770"
export SHOPPING_ADMIN="http://100.92.2.51:7780/admin"
export REDDIT="http://100.92.2.51:9999"
export GITLAB="http://100.92.2.51:8023"
export MAP="http://100.92.2.51:3000"
export WIKIPEDIA="http://100.92.2.51:8888"
export HOMEPAGE="http://100.92.2.51:4399"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
# Tailscale SOCKS5 proxy
export ALL_PROXY="socks5://localhost:1055"
export HTTP_PROXY="socks5://localhost:1055"
export HTTPS_PROXY="socks5://localhost:1055"
export NO_PROXY="localhost,127.0.0.1"
ENVEOF

# Generate task configs (needs proxy to access sites for URL substitution)
cd "$VAB_DIR"
log "  Generating task configs..."
python scripts/generate_test_data.py 2>/dev/null || log "  Warning: generate_test_data.py failed — may need manual config"

# Verify configs were created
if [ -d "config_files/wa/test_webarena_lite" ]; then
    NCONFIGS=$(ls config_files/wa/test_webarena_lite/*.json 2>/dev/null | wc -l)
    log "  Generated $NCONFIGS task config files"
else
    log "  WARNING: config_files/wa/test_webarena_lite/ not created!"
    log "  You may need to run: cd $VAB_DIR && python scripts/generate_test_data.py"
fi

# Generate auth cookies (needs proxy to access sites for login)
log "  Generating auth cookies..."
bash prepare.sh 2>/dev/null || log "  Warning: prepare.sh failed — may need manual setup"

log "Step 3 complete."

# =============================================================
# Step 4: Start vLLM model server
# =============================================================
log "Step 4: Starting vLLM model server..."

# vLLM talks to localhost only — no proxy needed
# Limit GPU memory to 50% so evaluation captioning model fits
python -m vllm.entrypoints.openai.api_server \
    --model THUDM/webrl-llama-3.1-8b \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.5 \
    > /tmp/vllm_server.log 2>&1 &

VLLM_PID=$!
echo $VLLM_PID > /tmp/vllm_pid.txt

log "  Waiting for vLLM to load model (this takes a few minutes)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q "ok\|200"; then
        log "  ✓ vLLM server ready (PID: $VLLM_PID)"
        break
    fi
    if [ $i -eq 60 ]; then
        log "  ✗ vLLM server not ready after 5 min. Check /tmp/vllm_server.log"
        tail -20 /tmp/vllm_server.log
    fi
    sleep 5
done

log "Step 4 complete."

# =============================================================
# Step 5: WebRL repo setup
# =============================================================
log "Step 5: Finalizing WebRL setup..."

cd "$WEBRL_DIR"

pip install -e . --no-deps --quiet 2>/dev/null
mkdir -p results checkpoints traces_output

log "Step 5 complete."

# =============================================================
# Step 6: Smoke test
# =============================================================
log "Step 6: Quick smoke test..."

# Test vLLM inference (localhost — bypasses proxy)
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

# Test WebArena container access (via SOCKS5 proxy)
python -c "
import requests
for name, url in [('shopping', '$SHOPPING_URL'), ('reddit', '$REDDIT_URL'), ('gitlab', '$GITLAB_URL')]:
    try:
        r = requests.get(url, timeout=15)
        print(f'✓ {name}: HTTP {r.status_code}')
    except Exception as e:
        print(f'✗ {name}: {e}')
" || log "  ✗ Container access test failed"

log "Step 6 complete."

# =============================================================
# Summary
# =============================================================
echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "WebArena containers (local machine via Tailscale SOCKS5):"
echo "  Shopping:       $SHOPPING_URL"
echo "  Shopping Admin: $SHOPPING_ADMIN_URL"
echo "  Reddit:         $REDDIT_URL"
echo "  GitLab:         $GITLAB_URL"
echo ""
echo "vLLM server:      http://localhost:8000"
echo "  PID: $(cat /tmp/vllm_pid.txt)"
echo "  Log: /tmp/vllm_server.log"
echo "  GPU memory: 50%"
echo ""
echo "Environment:      source /tmp/webarena_env.sh"
echo ""
echo "Next steps:"
echo "  1. bash scripts/run_dryrun.sh"
echo "  2. bash scripts/run_experiment.sh"
echo ""
echo "To stop vLLM:     kill \$(cat /tmp/vllm_pid.txt)"
echo "To check vLLM:    tail -f /tmp/vllm_server.log"
echo "=========================================="
