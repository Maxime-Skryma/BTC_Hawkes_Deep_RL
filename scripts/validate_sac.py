"""SAC check on an independent solvable Poisson task, never historical P&L."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from cjr.model import Params
from cjr.rl_algorithms import train
from cjr.reference import poisson_policy_iteration,policy_value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/publication.yaml')
    parser.add_argument('--output',default='experiences/reproduction_10seeds/sac_validation')
    args=parser.parse_args()
    spec=yaml.safe_load((ROOT/args.config).read_text());cfg=yaml.safe_load((ROOT/spec['market_config']).read_text())
    cal=json.loads((ROOT/spec['calibration']).read_text());folder=ROOT/args.output;folder.mkdir(parents=True,exist_ok=True)
    p=Params.from_calibration(cal,cfg,hawkes=False,alpha=False,fee_bps=0.0)
    optimum,_,_=poisson_policy_iteration(*cal['mean_rates'],p)
    results=[]
    # Declared before evaluation: a sanity check, not an optimality certificate.
    for seed in [11,23,37]:
        start=time.monotonic()
        ag,history=train(p,spec['settings'],'sac',seed,log=lambda _:None)
        y=torch.zeros(2*p.K+1,2+2*p.C);y[:,0]=torch.arange(-p.K,p.K+1)
        with torch.no_grad():d=ag.deltas(y).numpy()
        value=policy_value(*cal['mean_rates'],p,d[:,0],d[:,1])
        regret=float(np.max(optimum-value)/max(1.,np.max(np.abs(optimum))))
        row=dict(seed=seed,max_relative_value_regret=regret,
            zero_inventory_quotes=d[p.K].tolist(),seconds=time.monotonic()-start)
        results.append(row)
        (folder/f'seed{seed}.json').write_text(json.dumps(dict(result=row,history=history),indent=2))
        print(row,flush=True)
    passed=all(r['max_relative_value_regret']<.2 for r in results)
    (folder/'validation.json').write_text(json.dumps(dict(status='PASS' if passed else 'FAIL',
        threshold=.2,meaning='finite-budget sanity check; not a convergence or Hawkes certificate',
        selection='all three declared seeds; no historical P&L access',results=results),indent=2))
    if not passed:raise RuntimeError('Independent SAC control failed')


if __name__=='__main__':main()
