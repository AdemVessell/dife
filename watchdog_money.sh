#!/bin/bash
# Watchdog for run_money_comparison.py — runs one seed/method at a time,
# restarts after env resets, skips completed jobs automatically.
cd /home/user/dife
LOG=/home/user/dife/money_comparison.log

METHODS="ConstReplay_0.3 MIR DIFE_only DIFE_MV"
SEEDS="0 1 2 3 4"

while true; do
    all_done=true
    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            OUT="results/money_comparison/split_cifar_rmax_0.30/${METHOD}/seed_${SEED}/metrics.json"
            if [ -f "$OUT" ]; then
                continue
            fi
            all_done=false
            echo "[$(date)] START ${METHOD} seed=${SEED}" >> $LOG
            python run_money_comparison.py --method "$METHOD" --seed "$SEED" >> $LOG 2>&1
            echo "[$(date)] END ${METHOD} seed=${SEED} (exit $?)" >> $LOG
        done
    done
    if [ "$all_done" = true ]; then
        echo "[$(date)] ALL 20 JOBS DONE" >> $LOG
        break
    fi
    sleep 10
done
