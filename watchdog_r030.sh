#!/bin/bash
cd /home/user/dife
LOG=/home/user/dife/sweep_r030.log

run_seed() {
    METHOD=$1; SEED=$2
    OUT="results/sweep_repaired/r_max_0.30/split_cifar/${METHOD}/seed_${SEED}/metrics.json"
    if [ -f "$OUT" ]; then
        echo "[$(date)] SKIP $METHOD seed=$SEED" >> $LOG
        return
    fi
    echo "[$(date)] START $METHOD seed=$SEED" >> $LOG
    python run_fast_track.py --bench split_cifar --seeds $SEED \
        --epochs-per-task 3 --methods $METHOD --r-max 0.3 \
        --output-root results/sweep_repaired/r_max_0.30 >> $LOG 2>&1
    echo "[$(date)] END $METHOD seed=$SEED (exit $?)" >> $LOG
}

while true; do
    for SEED in 2 3 4; do
        run_seed DIFE_only $SEED
        run_seed DIFE_MV $SEED
    done
    COUNT=$(find results/sweep_repaired/r_max_0.30 -name "metrics.json" | wc -l)
    if [ "$COUNT" -ge 10 ]; then
        echo "[$(date)] ALL DONE" >> $LOG
        break
    fi
    sleep 10
done
