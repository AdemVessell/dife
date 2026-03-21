#!/bin/bash
cd /home/user/dife
LOGFILE=logs/persistent_run.log
JOBFILE=logs/remaining_jobs.txt
DONEFILE=logs/done_jobs.txt

echo "[$(date)] Starting persistent run of $(wc -l < $JOBFILE) jobs" >> $LOGFILE

total=$(wc -l < $JOBFILE)
n=0
while IFS=' ' read -r beta method seed; do
    n=$((n+1))
    echo "[$(date)] [$n/$total] START beta=$beta method=$method seed=$seed" >> $LOGFILE
    python run_beta_bound_rerun.py --beta-min $beta --method $method --seed $seed >> $LOGFILE 2>&1
    ec=$?
    echo "[$(date)] [$n/$total] END   beta=$beta method=$method seed=$seed exit=$ec" >> $LOGFILE
    echo "$beta $method $seed" >> $DONEFILE
done < $JOBFILE

echo "[$(date)] ALL DONE" >> $LOGFILE
