#!/bin/bash
# Run Alchemist on CIFAR-100 → Butterfly → Shopee-IET sequentially.
# Fair-comparison twin of auto_chain_all.sh (which runs AutoML-Agent).
#
# Budgets (match AutoML-Agent chain):
#   CIFAR-100: 8h, Butterfly: 4h, Shopee-IET: 2h (total safety cap ~14h)
# Each experiment runs independently; a failure in one does not block others.

set -u
REPO=/home/samsung/바탕화면/VP_Projects/alchemist
LOG=/tmp/alchemist_chain_all.log

EC2_STOP_REGION="${EC2_STOP_REGION:-us-east-1}"
EC2_STOP_INSTANCE="${EC2_STOP_INSTANCE:-}"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

stop_ec2() {
    if [ -z "$EC2_STOP_INSTANCE" ]; then
        log "EC2 auto-stop disabled (EC2_STOP_INSTANCE empty)"
        return 0
    fi
    log "stopping EC2 $EC2_STOP_INSTANCE in $EC2_STOP_REGION"
    aws ec2 stop-instances \
        --region "$EC2_STOP_REGION" \
        --instance-ids "$EC2_STOP_INSTANCE" \
        --query 'StoppingInstances[].[InstanceId,CurrentState.Name,PreviousState.Name]' \
        --output text 2>&1 | tee -a "$LOG"
}

cleanup() {
    log "cleanup: triggering EC2 auto-stop"
    stop_ec2
}
trap cleanup INT TERM EXIT

cd "$REPO"

run_exp() {
    local name="$1"
    local script="$2"
    local budget_seconds="$3"
    local exp_log="/tmp/alchemist_${name}_run.log"
    log "========== START alchemist-$name =========="
    log "   budget: $((budget_seconds/3600))h  log: $exp_log"
    timeout "$((budget_seconds + 1800))" bash "$script" 2>&1 | tee "$exp_log"
    local rc=${PIPESTATUS[0]}
    log "========== END alchemist-$name (rc=$rc) =========="
    return $rc
}

run_exp cifar100  baselines/run_alchemist_cifar100.sh  28800 || log "cifar100 FAILED, continuing"
run_exp butterfly baselines/run_alchemist_butterfly.sh 14400 || log "butterfly FAILED, continuing"
run_exp shopee    baselines/run_alchemist_shopee.sh     7200 || log "shopee FAILED, continuing"

log "ALL ALCHEMIST EXPERIMENTS DONE"
