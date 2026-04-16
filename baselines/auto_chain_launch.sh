#!/bin/bash
# Wait for pip install to finish, then launch proxy + AutoML-Agent on EC2.
#
# Steps:
#   1. Poll the install PID until it exits.
#   2. Verify required modules import in the amla env.
#   3. Start claude_cli_proxy.py in background (logs -> /tmp/proxy.log).
#   4. Wait for proxy /health to be 200.
#   5. Run run_amla_cifar100.py (logs -> /tmp/amla_run.log); training
#      happens on EC2 via remote_execute_ec2.py's patch.
#
# Fail-fast: if any step fails, stop and report.

set -u
LOG=/tmp/amla_auto_chain.log
INSTALL_PID="${INSTALL_PID:-}"
AMLA_ENV=/mnt/windows/LinuxData/amla_env
REPO_ROOT=/home/samsung/바탕화면/VP_Projects/alchemist
PROXY_LOG=/tmp/proxy.log
RUN_LOG=/tmp/amla_run.log

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# ---- Step 1: wait for install -------------------------------------------
if [ -n "$INSTALL_PID" ]; then
    log "waiting for pip install (PID=$INSTALL_PID)..."
    while kill -0 "$INSTALL_PID" 2>/dev/null; do
        sleep 30
    done
    log "install process exited"
fi

# ---- Step 2: verify install -------------------------------------------
log "verifying install..."
source /home/samsung/miniconda3/etc/profile.d/conda.sh
conda activate "$AMLA_ENV"

python - <<'PYEOF' 2>&1 | tee -a "$LOG"
import importlib, sys
required = ["openai", "langchain", "langchain_community", "torch", "transformers",
            "fastapi", "uvicorn", "peft", "accelerate"]
missing = []
for m in required:
    try:
        importlib.import_module(m)
        print(f"  ok: {m}")
    except Exception as e:
        missing.append((m, str(e)[:100]))
        print(f"  MISSING: {m} ({e})")
if missing:
    print(f"FAIL: {len(missing)} required modules missing")
    sys.exit(1)
print("import_check_ok")
PYEOF
if [ $? -ne 0 ]; then
    log "IMPORT CHECK FAILED — aborting chain"
    exit 1
fi

# ---- Step 3: start proxy in background --------------------------------
log "starting claude_cli_proxy on :8001 ..."
pkill -f 'claude_cli_proxy.py' 2>/dev/null || true
sleep 1
nohup python "$REPO_ROOT/baselines/claude_cli_proxy.py" --port 8001 --log-level info \
    > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
log "proxy PID=$PROXY_PID"

# ---- Step 4: wait for proxy /health ------------------------------------
for i in $(seq 1 30); do
    if curl -s -m 2 http://127.0.0.1:8001/health | grep -q '"ok":true'; then
        log "proxy healthy after ${i}s"
        break
    fi
    sleep 1
    if [ "$i" = "30" ]; then
        log "proxy failed to respond after 30s — aborting"
        tail -30 "$PROXY_LOG" | tee -a "$LOG"
        kill "$PROXY_PID" 2>/dev/null
        exit 1
    fi
done

# ---- Step 5: run AutoML-Agent on CIFAR-100 ----------------------------
log "launching run_amla_cifar100.py (training on EC2 via remote exec) ..."
export AMLA_EC2_HOST=3.90.166.150
export AMLA_EC2_KEY="$HOME/.ssh/alchemist-gpu-key-use1.pem"
export AMLA_EC2_USER=ubuntu
export AMLA_EC2_REMOTE=/home/ubuntu/amla_workspace
export AMLA_EC2_ACTIVATE="source /opt/pytorch/bin/activate"
export AMLA_EC2_TIMEOUT=28800   # 8 hours per prompt budget

cd "$REPO_ROOT"
python baselines/run_amla_cifar100.py 2>&1 | tee -a "$RUN_LOG"
RC=$?
log "run_amla_cifar100.py exited rc=$RC"

# ---- cleanup -----------------------------------------------------------
log "stopping proxy..."
kill "$PROXY_PID" 2>/dev/null
log "chain complete"
