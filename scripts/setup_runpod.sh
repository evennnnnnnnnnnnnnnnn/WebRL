#!/bin/bash
# =============================================================
# RunPod A100 Full Setup Script
#
# Sets up everything needed for the A vs C experiment:
#   1. Conda environment with correct PyTorch + vLLM
#   2. Tailscale SOCKS5 proxy verification
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
        log "  OK ${name}"
    else
        log "  FAIL ${name}: NOT REACHABLE ($url)"
    fi
done

log "Step 0 complete."

# =============================================================
# Step 1: Create conda env + install Python dependencies
# =============================================================
# IMPORTANT: Do NOT install into the RunPod system Python (3.11).
# The system has pre-installed PyTorch/pydantic/fastapi pinned to
# old versions that cause cascading conflicts with vLLM:
#   - torch 2.0.1+cu117 missing torch.library.infer_schema (needs 2.5+)
#   - pydantic missing IncEx (needs 2.x)
#   - typing_extensions missing TypeIs (needs >=4.10)
#   - duplicate libcudnn.so.8 + .so.9 causing AssertionError
# A fresh conda env avoids ALL of these.
# =============================================================
log "Step 1: Setting up conda environment..."

# Unset any proxy to avoid interfering with pip/git
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY

# Create conda env if it doesn't exist
if ! conda env list | grep -q "^webrl "; then
    log "  Creating conda env 'webrl' with Python 3.10..."
    conda create -n webrl python=3.10 -y
fi

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate webrl

log "  Python: $(python --version) at $(which python)"

# Install PyTorch with CUDA 12.4 wheels (compatible with CUDA 13.0 driver)
# MUST install torch BEFORE WebRL's setup.py to prevent it pulling in
# an ancient torch 2.0.1+cu117
# PINNED: torch 2.6.0 + vLLM 0.7.3 is a tested working combination.
#   - torch 2.10+ breaks vLLM 0.18 (FakeTensorMode AttributeError)
#   - torch 2.0.x breaks vLLM (missing infer_schema)
log "  Installing PyTorch 2.6.0 (CUDA 12.4)..."
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --quiet

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
log "  PyTorch version: $TORCH_VERSION"

# Install WebRL package (--no-deps to avoid it downgrading torch)
log "  Installing WebRL..."
cd "$WEBRL_DIR"
pip install -e . --no-deps --quiet

# Install vLLM (pinned to 0.7.3 — tested working with torch 2.6.0)
log "  Installing vLLM 0.7.3..."
pip install vllm==0.7.3 --quiet

# Fix duplicate cuDNN: vLLM installs both cudnn 8 and 9, but only 9 is needed.
# Two libcudnn.so.* files cause: AssertionError: Found 2 libcudnn.so.x
CUDNN_DIR=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null)/lib
if [ -f "$CUDNN_DIR/libcudnn.so.8" ] && [ -f "$CUDNN_DIR/libcudnn.so.9" ]; then
    log "  Removing duplicate libcudnn.so.8 (vLLM needs .so.9 only)..."
    rm "$CUDNN_DIR/libcudnn.so.8"
fi

# Install remaining dependencies
log "  Installing remaining dependencies..."
pip install "numpy<2" "typing_extensions>=4.10" --quiet
pip install transformers==4.44.2 accelerate==0.32.1 deepspeed==0.15.1 \
    hydra-core omegaconf datasets peft openai anthropic python-dotenv \
    wandb beautifulsoup4 sentencepiece tenacity termcolor tqdm \
    "httpx[socks]" dashscope pysocks "requests[socks]" \
    --quiet

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

# Patch openai_utils.py: fix unbound 'client' when using env var API key
# The function creates a local 'client' only when api_key is passed, but falls
# through to use it even when api_key is None, causing UnboundLocalError.
if grep -q 'if api_key is not None:' llms/providers/openai_utils.py 2>/dev/null; then
    python3 -c "
import re
with open('llms/providers/openai_utils.py', 'r') as f:
    content = f.read()
old = '''    if api_key is not None:
        client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.completions.create('''
new = '''    if api_key is not None:
        _client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        _client = client
    response = _client.completions.create('''
content = content.replace(old, new)
with open('llms/providers/openai_utils.py', 'w') as f:
    f.write(content)
"
    log "  Patched openai_utils.py: fixed unbound client variable"
fi

# Patch openai_utils.py: support OPENAI_API_BASE env var (vLLM uses this)
if grep -q 'OPENAI_API_URL' llms/providers/openai_utils.py 2>/dev/null; then
    sed -i 's/base_url = os.environ.get("OPENAI_API_URL")/base_url = os.environ.get("OPENAI_API_URL") or os.environ.get("OPENAI_API_BASE")/' llms/providers/openai_utils.py
    log "  Patched openai_utils.py: added OPENAI_API_BASE fallback"
fi

# Install VAB dependencies
pip install -r requirements.txt --quiet 2>/dev/null

# Install missing deps not in requirements.txt
pip install matplotlib text-generation aiolimiter dashscope google-auth \
    evaluate scikit-image --quiet 2>/dev/null

# Download NLTK data needed for evaluation
python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null

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
export no_proxy="localhost,127.0.0.1"

export DATASET=webarena
export SHOPPING="$SHOPPING_URL"
export SHOPPING_ADMIN="${SHOPPING_ADMIN_URL}/admin"
export REDDIT="$REDDIT_URL"
export GITLAB="$GITLAB_URL"
export MAP="$MAP_URL"
export WIKIPEDIA="$WIKIPEDIA_URL"
export HOMEPAGE="$HOMEPAGE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_API_BASE="http://localhost:8000/v1"

# Save env vars for later use (sourced by run_experiment.sh)
cat > /tmp/webarena_env.sh << 'ENVEOF'
# Activate conda env
eval "$(conda shell.bash hook)"
conda activate webrl

export DATASET=webarena
export SHOPPING="http://100.92.2.51:7770"
export SHOPPING_ADMIN="http://100.92.2.51:7780/admin"
export REDDIT="http://100.92.2.51:9999"
export GITLAB="http://100.92.2.51:8023"
export MAP="http://100.92.2.51:3000"
export WIKIPEDIA="http://100.92.2.51:8888"
export HOMEPAGE="http://100.92.2.51:4399"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_API_BASE="http://localhost:8000/v1"
# Tailscale SOCKS5 proxy
export ALL_PROXY="socks5://localhost:1055"
export HTTP_PROXY="socks5://localhost:1055"
export HTTPS_PROXY="socks5://localhost:1055"
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
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
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q "200"; then
        log "  OK vLLM server ready (PID: $VLLM_PID)"
        break
    fi
    # Check if process died
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        log "  FAIL vLLM process died. Check /tmp/vllm_server.log:"
        tail -20 /tmp/vllm_server.log
        exit 1
    fi
    if [ $i -eq 60 ]; then
        log "  FAIL vLLM server not ready after 5 min. Check /tmp/vllm_server.log"
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
print('OK Model inference')
" || log "  FAIL vLLM inference test"

# Test WebArena container access (via SOCKS5 proxy)
python -c "
import requests
for name, url in [('shopping', '$SHOPPING_URL'), ('reddit', '$REDDIT_URL'), ('gitlab', '$GITLAB_URL')]:
    try:
        r = requests.get(url, timeout=15)
        print(f'OK {name}: HTTP {r.status_code}')
    except Exception as e:
        print(f'FAIL {name}: {e}')
" || log "  FAIL Container access test"

# Verify Anthropic SDK
python -c "import anthropic; print('OK Anthropic SDK')" || log "  FAIL Anthropic SDK not installed"

log "Step 6 complete."

# =============================================================
# Summary
# =============================================================
echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Conda env:        webrl (Python 3.10)"
echo "PyTorch:          $(python -c 'import torch; print(torch.__version__)')"
echo "vLLM:             $(python -c 'import vllm; print(vllm.__version__)')"
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
echo "IMPORTANT: Always run 'conda activate webrl' before any commands."
echo ""
echo "Next steps:"
echo "  1. bash scripts/run_dryrun.sh"
echo "  2. bash scripts/run_experiment.sh"
echo ""
echo "To stop vLLM:     kill \$(cat /tmp/vllm_pid.txt)"
echo "To check vLLM:    tail -f /tmp/vllm_server.log"
echo "=========================================="
