#!/usr/bin/env bash
# usage: tools/wait_kernel.sh <slug> ; polls until the kernel finishes, then downloads its output to runs/<slug>/
slug=$1; out=runs/$slug
while true; do
  st=$(kaggle kernels status franciscoangulo/$slug 2>&1 | grep -v outdated | tail -1)
  case "$st" in
    *COMPLETE*|*ERROR*|*CANCEL*) break;;
  esac
  sleep 60
done
echo "$st"
mkdir -p "$out" && kaggle kernels output franciscoangulo/$slug -p "$out" 2>&1 | grep -v outdated | tail -2
python - "$out" <<'PY'
import json,sys,os
p=os.path.join(sys.argv[1],'results.json')
if os.path.exists(p):
    for r in json.load(open(p)):
        print(r['name'],r['seed'],'oof %.6f'%r['oof_auc'],'fold %.6f±%.6f'%(r['fold_mean'],r['fold_std']),'iters',r['iters'],'sec',r['sec'],'gpu',r['gpu'])
PY
