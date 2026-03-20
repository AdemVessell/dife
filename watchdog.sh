#!/usr/bin/env bash
# Watchdog: monitors beta rerun processes and restarts them if they die or stall.
# A "stall" is defined as no new metrics.json files written in STALL_SECONDS.
# Usage: bash watchdog.sh  (run in background via nohup)

set -euo pipefail

STALL_SECONDS=600   # 10 min without a new result = stall
CHECK_INTERVAL=60   # check every 60s

LOG005="logs/beta005_rerun.log"
LOG010="logs/beta010_rerun.log"
WLOG="logs/watchdog.log"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$WLOG"; }

start_005() {
    log "Starting beta005 process..."
    nohup python run_beta_bound_rerun.py --beta-min 0.05 >> "$LOG005" 2>&1 &
    echo $! > /tmp/pid_beta005
    log "beta005 PID=$(cat /tmp/pid_beta005)"
}

start_010() {
    log "Starting beta010 process..."
    nohup python run_beta_bound_rerun.py --beta-min 0.10 >> "$LOG010" 2>&1 &
    echo $! > /tmp/pid_beta010
    log "beta010 PID=$(cat /tmp/pid_beta010)"
}

is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] || return 1
    local pid
    pid=$(cat "$pid_file")
    kill -0 "$pid" 2>/dev/null
}

result_count() {
    find "$1" -name "metrics.json" 2>/dev/null | wc -l
}

last_result_age() {
    # seconds since most recent metrics.json was written
    local newest
    newest=$(find "$1" -name "metrics.json" -printf "%T@\n" 2>/dev/null | sort -n | tail -1)
    if [[ -z "$newest" ]]; then
        echo 99999
    else
        echo $(( $(date +%s) - ${newest%.*} ))
    fi
}

is_complete() {
    # Complete = 30 metrics.json files present
    [[ $(result_count "$1") -ge 30 ]]
}

# ── Initial start ────────────────────────────────────────────────────────────
log "Watchdog started (stall_threshold=${STALL_SECONDS}s, check_interval=${CHECK_INTERVAL}s)"

OUT005="results/canonical_beta005"
OUT010="results/canonical_beta010"

if is_complete "$OUT005"; then
    log "beta005 already complete (30 results). Skipping."
else
    start_005
fi

if is_complete "$OUT010"; then
    log "beta010 already complete (30 results). Skipping."
else
    start_010
fi

# ── Watch loop ───────────────────────────────────────────────────────────────
while true; do
    sleep "$CHECK_INTERVAL"

    for beta in 005 010; do
        pid_file="/tmp/pid_beta${beta}"
        out_dir="results/canonical_beta${beta}"
        start_fn="start_${beta}"

        if is_complete "$out_dir"; then
            log "beta${beta} complete ($(result_count "$out_dir")/30 results). No action needed."
            continue
        fi

        running=false
        is_running "$pid_file" && running=true

        age=$(last_result_age "$out_dir")
        count=$(result_count "$out_dir")

        log "beta${beta}: running=$running results=${count}/30 last_result_age=${age}s"

        if $running && [[ "$age" -lt "$STALL_SECONDS" ]]; then
            # Healthy — do nothing
            continue
        fi

        if $running && [[ "$age" -ge "$STALL_SECONDS" ]]; then
            pid=$(cat "$pid_file")
            log "beta${beta} STALLED (age=${age}s). Killing PID $pid and restarting..."
            kill "$pid" 2>/dev/null || true
            sleep 3
        else
            log "beta${beta} DIED. Restarting..."
        fi

        $start_fn
    done

    # Exit watchdog when both are complete
    if is_complete "$OUT005" && is_complete "$OUT010"; then
        log "Both experiments complete! Watchdog exiting."
        exit 0
    fi
done
