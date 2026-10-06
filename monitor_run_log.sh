#!/bin/bash
# Summarise the health of the potential solve in a running B2.5 log, in bins of steps.
# Per bin: stalled solves (respo after solving > 1e-5), largest residual before/after the solve,
# number of "Applied po/Te ratio limit" messages per step, largest unclamped po.
# Healthy: stalled 0, respo_after ~1e-7, clamp msgs/step ~0.
# Usage: monitor_run_log.sh [logfile=run.log] [bin=100] [first_step=0] [last_step=1e9]
LOG=${1:-run.log}; BIN=${2:-100}; LO=${3:-0}; HI=${4:-1000000000}
awk -v bin="$BIN" -v lo="$LO" -v hi="$HI" '
/b2mndr_ok:itim/ {it=$2+0; if(it>maxit) maxit=it; next}
/respo before solving/ {rb[it]=$4+0}
/respo after solving/ {ra[it]=$4+0}
/Applied po\/Te ratio limit/ {nc[it]++; a=$(NF-1); if(a<0)a=-a; if(a>mp[it])mp[it]=a}
END {
  top=(maxit<hi)?maxit:hi
  for(b=lo;b<=top;b+=bin){ n=0; bad=0; mrb=0; mra=0; ncl=0; mpo=0
    for(i=b;i<b+bin;i++) if(i in ra){n++; if(ra[i]>1e-5)bad++; if(ra[i]>mra)mra=ra[i]; if(rb[i]>mrb)mrb=rb[i]; ncl+=nc[i]; if(mp[i]>mpo)mpo=mp[i]}
    if(n) printf "steps %6d-%6d: stalled %3d/%-3d  respo_before max %.2e  respo_after max %.2e  clamp/step %5.1f  max|po_preclamp| %.3g\n", b,b+bin-1,bad,n,mrb,mra,ncl/n,mpo }
  print "last step in log:", maxit }' "$LOG"
