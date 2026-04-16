#!/bin/bash
# Run AutoML-Agent on CIFAR-100 → Butterfly → Shopee sequentially.
# Proxy is started once at the top; all three experiments share it.
# Each experiment runs independently; a failure in one does not block others.

set -u
AMLA_ENV=/mnt/windows/LinuxData/amla_env
REPO=/home/samsung/바탕화면/VP_Projects/alchemist
LOG=/tmp/amla_chain_all.log

# AWS EC2 auto-stop config — set to "" to disable.
EC2_STOP_REGION="${EC2_STOP_REGION:-us-east-1}"
EC2_STOP_INSTANCE="${EC2_STOP_INSTANCE:-i-0e35db2d675713696}"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

stop_ec2() {
    if [ -z "$EC2_STOP_INSTANCE" ]; then
        log "EC2 auto-stop disabled (EC2_STOP_INSTANCE empty)"
        return 0
    fi
    log "stopping EC2 $EC2_STOP_INSTANCE in $EC2_STOP_REGION (state: stopped, EBS preserved)"
    aws ec2 stop-instances \
        --region "$EC2_STOP_REGION" \
        --instance-ids "$EC2_STOP_INSTANCE" \
        --query 'StoppingInstances[].[InstanceId,CurrentState.Name,PreviousState.Name]' \
        --output text 2>&1 | tee -a "$LOG"
}

source /home/samsung/miniconda3/etc/profile.d/conda.sh
conda activate "$AMLA_ENV"

# ---- Step 1: sanity imports ----------------------------------------------
log "verifying imports..."
python - <<'PY' 2>&1 | tee -a "$LOG"
import importlib, sys
need = ["openai", "langchain", "torch", "transformers", "fastapi", "uvicorn",
        "timm", "peft", "accelerate"]
miss = []
for m in need:
    try: importlib.import_module(m); print(f"  ok: {m}")
    except Exception as e: miss.append((m, str(e))); print(f"  MISSING: {m} ({e})")
sys.exit(1 if miss else 0)
PY
if [ $? -ne 0 ]; then log "IMPORT CHECK FAILED"; exit 1; fi

# ---- Step 2: start proxy (singleton) -------------------------------------
log "starting claude_cli_proxy on :8001 ..."
pkill -f 'claude_cli_proxy.py' 2>/dev/null || true
sleep 1
nohup python "$REPO/baselines/claude_cli_proxy.py" --port 8001 --log-level warning \
    > /tmp/proxy.log 2>&1 &
PROXY_PID=$!
log "proxy PID=$PROXY_PID"

cleanup() {
    log "cleanup: stopping proxy ${PROXY_PID:-?}"
    kill ${PROXY_PID:-} 2>/dev/null || true
    # Final S3 backup of any results (best-effort)
    if [ -n "${EC2_STOP_INSTANCE:-}" ]; then
        log "cleanup: rsyncing final results from EC2 (best effort)"
        rsync -az -e "ssh -i $HOME/.ssh/alchemist-gpu-key-use1.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
            ubuntu@3.90.166.150:/home/ubuntu/amla_workspace/agent_workspace/ \
            "$REPO/baselines/automl-agent/agent_workspace_remote/" 2>&1 | tail -3 | tee -a "$LOG" || true
    fi
    stop_ec2
}
trap cleanup INT TERM EXIT

# wait for proxy
for i in $(seq 1 30); do
    if curl -s -m 2 http://127.0.0.1:8001/health | grep -q '"ok":true'; then
        log "proxy healthy"
        break
    fi
    sleep 1
    [ "$i" = "30" ] && { log "PROXY START FAILED"; exit 1; }
done

# ---- Step 3: common EC2 env ----------------------------------------------
export AMLA_EC2_HOST=3.90.166.150
export AMLA_EC2_KEY="$HOME/.ssh/alchemist-gpu-key-use1.pem"
export AMLA_EC2_USER=ubuntu
export AMLA_EC2_REMOTE=/home/ubuntu/amla_workspace
export AMLA_EC2_ACTIVATE="source /opt/pytorch/bin/activate"

cd "$REPO"

# ---- Step 4: run three experiments ---------------------------------------
run_exp() {
    local name="$1"
    local script="$2"
    local budget_seconds="$3"
    local exp_log="/tmp/amla_${name}_run.log"
    log "========== START $name =========="
    log "   budget: $((budget_seconds/3600))h  log: $exp_log"
    export AMLA_EC2_TIMEOUT="$budget_seconds"
    timeout "$((budget_seconds + 1800))" python "$script" 2>&1 | tee "$exp_log"
    local rc=${PIPESTATUS[0]}
    log "========== END $name (rc=$rc) =========="
    return $rc
}

# CIFAR-100: 8h budget, safety cap 8h30m
run_exp cifar100 baselines/run_amla_cifar100.py 28800 || log "cifar100 FAILED, continuing"

# Butterfly: 4h budget, safety cap 4h30m
run_exp butterfly baselines/run_amla_butterfly.py 14400 || log "butterfly FAILED, continuing"

# Shopee-IET: 2h budget, safety cap 2h30m
run_exp shopee baselines/run_amla_shopee.py 7200 || log "shopee FAILED, continuing"

log "ALL EXPERIMENTS DONE"
