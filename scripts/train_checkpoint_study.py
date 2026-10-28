"""TD3/SAC synthetic validation study, isolated from historical test results."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from cjr import model as M, rl_algorithms as RL


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def save(path,data):
    path=Path(path);tmp=path.with_suffix('.pt.part');torch.save(data,tmp);tmp.replace(path)


@torch.no_grad()
def evaluate(agent,p,paths,events,seed):
    # Separate RNG and common random numbers; never consume training RNG.
    with torch.random.fork_rng():
        env=RL.EventEnvironment(p,paths,seed);env.y[:,0]=0
        discount=torch.ones(paths);returns=torch.zeros(paths);reward_sum=torch.zeros(paths);inventory=torch.zeros(paths)
        for _ in range(events):
            action,_,_=agent.actor.sample(agent.norm(env.y),RL.active_sides(env.y,p),deterministic=True)
            _,r,d=env.step(action)
            returns+=discount*r*env.value_scale;discount*=d;reward_sum+=r*env.value_scale;inventory+=env.y[:,0].abs()
        return dict(score=float(returns.mean()),score_se=float(returns.std(unbiased=True)/np.sqrt(paths)),
            mean_reward_per_event=float(reward_sum.mean()/events),mean_abs_inventory=float(inventory.mean()/events))


def job(args):
    spec,algo,fee,seed=args;torch.set_num_threads(spec.get('runtime_threads',spec['settings']['threads']))
    out=ROOT/spec['output']/algo/fee/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    protocol=json.loads((ROOT/spec['source']/'protocol.json').read_text());cal=json.loads((ROOT/spec['source']/'calibration.json').read_text())
    selection=json.loads((ROOT/spec['source']/'selection.json').read_text())['chosen'][fee][algo+'__full']
    settings=dict(spec['settings'],observation='full');settings['threads']=spec.get('runtime_threads',settings['threads']);factor=selection['parameters']['learning_rate_factor']
    for k in ('actor_lr','critic_lr'):settings[k]*=factor
    p=M.Params.from_calibration(cal,protocol['market_config'],fee_bps=protocol['market_config']['realism']['fees'][fee]['maker']*1e4)
    ev=spec['evaluation'];rows=json.loads((out/'evaluations.json').read_text()) if (out/'evaluations.json').exists() else []
    resume=torch.load(out/'resume.pt',weights_only=False,map_location='cpu') if (out/'resume.pt').exists() else None
    if resume is not None:rows=[r for r in rows if r['transitions']<=resume['completed_steps']*settings['envs']]
    best=max((r['score'] for r in rows),default=-float('inf'));best_at=min((r['transitions'] for r in rows if r['score']==best),default=0)
    start=time.monotonic();stop_reason='maximum_budget'
    def callback(agent,n,snapshot):
        nonlocal best,best_at,stop_reason
        if n%ev['every_transitions'] and n!=settings['transitions']:return False
        result=evaluate(agent,p,ev['paths'],ev['events'],ev['selection_seed']);result['transitions']=n;rows.append(result)
        state=agent.state_dict();state.update(training_seed=seed,study='checkpoint_study_v1',transitions=n,validation=result,validation_kind='synthetic discounted inventory-penalized reward')
        if result['score']>best:
            best=result['score'];best_at=n;save(out/'best.pt',state)
        save(out/'last.pt',state);save(out/'resume.pt',snapshot());write_json(out/'evaluations.json',rows)
        print(json.dumps(dict(algorithm=algo,fee=fee,seed=seed,**result)),flush=True)
        stop=n>=spec['minimum_transitions'] and n-best_at>=ev['patience_evaluations']*ev['every_transitions']
        if stop:stop_reason='validation_patience_reached_not_proof_of_convergence'
        return stop
    with (out/'training.jsonl').open('a') as log:
        ag,history=RL.train(p,settings,algo,seed,log=lambda row:log.write(json.dumps(row)+'\n'),checkpoint_callback=callback,resume_state=resume)
    sd=torch.load(out/'best.pt',map_location='cpu',weights_only=False);ag.load(sd)
    heldout=evaluate(ag,p,ev['paths']*2,ev['events'],ev['assessment_seed'])
    result=dict(algorithm=algo,fee=fee,seed=seed,learning_rate_factor=factor,best_transitions=best_at,last_transitions=rows[-1]['transitions'],best_validation_score=best,assessment=heldout,seconds_this_invocation=time.monotonic()-start,stop_reason=stop_reason,best_sha256=sha(out/'best.pt'),last_sha256=sha(out/'last.pt'))
    write_json(out/'summary.json',result);return result


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries=[json.loads(p.read_text()) for p in out.glob('*/*/*/summary.json')]
    pd.json_normalize(summaries).to_csv(out/'summary.csv',index=False)
    for fee in ('retail_bnb','vip3_bnb','vip6_bnb'):
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        for ax,algo in zip(axs,('td3','sac')):
            for p in sorted((out/algo/fee).glob('seed*/evaluations.json')):
                rows=json.loads(p.read_text());x=[r['transitions'] for r in rows];y=[r['score'] for r in rows];se=np.array([r['score_se'] for r in rows]);ax.plot(x,y,label=p.parent.name);ax.fill_between(x,np.array(y)-1.96*se,np.array(y)+1.96*se,alpha=.1)
            ax.set(title=algo.upper(),xlabel='Training transitions',ylabel='Mean discounted reward (bp-lots)');ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f'{fee}: synthetic validation from flat inventory\nShading: pointwise ±1.96 MC standard errors; not adjusted for checkpoint selection')
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(out/f'validation_{fee}.{ext}',dpi=150)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/checkpoint_study.json');parser.add_argument('--workers',type=int,default=8);parser.add_argument('--plan',action='store_true');parser.add_argument('--cpu-budget',type=int);parser.add_argument('--resume-code-update',action='store_true');args=parser.parse_args()
    spec=json.loads((ROOT/args.config).read_text());jobs=[(spec,a,f,s) for a in spec['algorithms'] for f in spec['fees'] for s in spec['seeds']]
    if args.plan:print(json.dumps(dict(models=len(jobs),**spec),indent=2));return
    total_jobs=len(jobs)
    jobs=[j for j in jobs if not (ROOT/spec['output']/j[1]/j[2]/f'seed{j[3]}'/'summary.json').exists()]
    initially_complete=total_jobs-len(jobs)
    if not jobs:
        report(ROOT/spec['output']);return
    workers=min(args.workers,len(jobs))
    if args.cpu_budget:
        import os
        available=sorted(os.sched_getaffinity(0))
        if not workers<=args.cpu_budget<=len(available):raise ValueError('CPU budget must cover workers and fit available CPUs')
        os.sched_setaffinity(0,available[:args.cpu_budget])
        base,extra=divmod(args.cpu_budget,workers)
        jobs=[(dict(j[0],runtime_threads=base+(i%workers<extra)),*j[1:]) for i,j in enumerate(jobs)]
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(spec=spec,inputs={n:sha(ROOT/spec['source']/n) for n in ('protocol.json','calibration.json','selection.json')},source={n:sha(ROOT/n) for n in ('cjr/rl_algorithms.py','scripts/train_checkpoint_study.py')},validation='Synthetic, not historical; no historical test data read',checkpoint_selection='Highest mean discounted reward on fixed independent simulated trajectories; earliest checkpoint wins ties',stopping='No validation improvement for patience checkpoints after minimum budget; not proof of convergence')
        if (out/'protocol.json').exists():
            previous=json.loads((out/'protocol.json').read_text())
            if previous!=manifest:
                if not args.resume_code_update or {k:v for k,v in previous.items() if k!='source'}!={k:v for k,v in manifest.items() if k!='source'}:
                    raise ValueError('Experiment inputs changed; use a new output directory')
        else:write_json(out/'protocol.json',manifest)
        write_json(out/f'execution_{time.time_ns()}.json',dict(manifest=manifest,workers=workers,cpu_budget=args.cpu_budget,runtime_threads=[j[0].get('runtime_threads',spec['settings']['threads']) for j in jobs],note='CPU allocation update; checkpoint objective, seeds and scientific inputs unchanged'))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(job,j) for j in jobs]
            for i,f in enumerate(as_completed(futures),initially_complete+1):
                row=f.result();write_json(out/'progress.json',dict(completed=i,total=total_jobs,latest=row));print(f'Completed {i}/{total_jobs}',flush=True)
        report(out)
if __name__=='__main__':main()
