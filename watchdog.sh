#!/usr/bin/env bash
# Watchdog: monitors beta rerun processes and restarts if dead or stalled.
# Stall = no new metrics.json written since process was last (re)started + STALL_GRACE.

STALL_GRACE=600   # seconds after start before we consider lack of results a stall
CHECK_INTERVAL=60

LOG005="logs/beta005_rerun.log"
LOG010="logs/beta010_rerun.log"
WLOG="logs/watchdog.log"
PID_FILE_005="/tmp/pid_beta005"
PID_FILE_010="/tmp/pid_beta010"
START_TIME_005="/tmp/start_beta005"
START_TIME_010="/tmp/start_beta010"
OUT005="results/canonical_beta005"
OUT010="results/canonical_beta010"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$WLOG"; echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

start_process() {
    local beta="$1"
    local pid_file start_file log_file beta_arg
    beta_arg="0.$(echo "$beta" | sed 's/^0//')"
    # map 005->0.05, 010->0.10
    if [[ "$beta" == "005" ]]; then
        beta_arg="0.05"; pid_file="$PID_FILE_005"; start_file="$START_TIME_005"; log_file="$LOG005"
    else
        beta_arg="0.10"; pid_file="$PID_FILE_010"; start_file="$START_TIME_010"; log_file="$LOG010"
    fi

    log "Starting beta${beta} (--beta-min ${beta_arg})..."
    nohup python run_beta_bound_rerun.py --beta-min "$beta_arg" >> "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    date +%s > "$start_file"
    log "beta${beta} PID=$pid"
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

newest_result_time() {
    # epoch of most recent metrics.json, or 0 if none
    local t
    t=$(find "$1" -name "metrics.json" -printf "%T@\n" 2>/dev/null | sort -n | tail -1)
    echo "${t%.*:-0}"
}

is_complete() {
    [[ $(result_count "$1") -ge 30 ]]
}

kill_process() {
    local pid_file="$1" beta="$2"
    if [[ -f "$pid_file" ]]; then
        local pid; pid=$(cat "$pid_file")
        log "Killing beta${beta} PID=$pid"
        kill "$pid" 2>/dev/null || true
        sleep 3
    fi
}

check_and_heal() {
    local beta="$1" out_dir pid_file start_file
    if [[ "$beta" == "005" ]]; then
        out_dir="$OUT005"; pid_file="$PID_FILE_005"; start_file="$START_TIME_005"
    else
        out_dir="$OUT010"; pid_file="$PID_FILE_010"; start_file="$START_TIME_010"
    fi

    is_complete "$out_dir" && log "beta${beta} complete (30/30). Done." && return

    local count; count=$(result_count "$out_dir")
    local running=false; is_running "$pid_file" && running=true

    local start_time=0
    [[ -f "$start_file" ]] && start_time=$(cat "$start_file")
    local now; now=$(date +%s)
    local uptime=$(( now - start_time ))

    local newest; newest=$(newest_result_time "$out_dir")
    local result_age=$(( now - newest ))
    # Only count result_age from process start if process started after newest result
    local age_since_start=$(( now - start_time ))

    log "beta${beta}: running=$running count=${count}/30 uptime=${uptime}s result_age=${result_age}s"

    if ! $running; then
        log "beta${beta} DIED — restarting"
        start_process "$beta"
        return
    fi

    # Stall: process has been running > STALL_GRACE and no new result since it started
    if [[ "$uptime" -gt "$STALL_GRACE" && "$newest" -lt "$start_time" ]]; then
        log "beta${beta} STALLED (running ${uptime}s, no results since restart) — killing and restarting"
        kill_process "$pid_file" "$beta"
        start_process "$beta"
        return
    fi

    # Also stall if result_age > STALL_GRACE AND process has been up > STALL_GRACE
    if [[ "$uptime" -gt "$STALL_GRACE" && "$result_age" -gt "$STALL_GRACE" ]]; then
        log "beta${beta} STALLED (last result ${result_age}s ago) — killing and restarting"
        kill_process "$pid_file" "$beta"
        start_process "$beta"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
log "=== Watchdog started (stall_grace=${STALL_GRACE}s, interval=${CHECK_INTERVAL}s) ==="

is_complete "$OUT005" || start_process "005"
is_complete "$OUT010" || start_process "010"

while true; do
    sleep "$CHECK_INTERVAL"
    check_and_heal "005"
    check_and_heal "010"
    is_complete "$OUT005" && is_complete "$OUT010" && { log "Both complete. Watchdog exiting."; exit 0; }
done
