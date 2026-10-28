"""Entraînement et évaluation communs, avec protocole écrit avant les scores.

python scripts/benchmark_suite.py prepare
python scripts/benchmark_suite.py train --algorithm td3
python scripts/benchmark_suite.py evaluate
python scripts/benchmark_suite.py report
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import os
import tempfile
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from cjr import model as M, discrete_value as DV, rl_algorithms as RL
from cjr import data, replay, simulate, vi, reference
from cjr.baselines import avellaneda_stoikov


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_scientific_sources(protocol,root=ROOT):
    for name,digest in protocol['source_sha256'].items():
        if sha(root/name)!=digest:
            raise RuntimeError(f'Source modifiée depuis le protocole : {name}')


def atomic_save(state,path):
    path=Path(path)
    handle,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.part',dir=path.parent)
    os.close(handle)
    try:
        torch.save(state,name)
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def setup(args):
    spec=yaml.safe_load((ROOT/args.config).read_text())
    cfg=yaml.safe_load((ROOT/spec['market_config']).read_text())
    cal=json.loads((ROOT/spec['calibration']).read_text())
    out=ROOT/spec['output']
    source=[*sorted((ROOT/'cjr').rglob('*.py')),*sorted((ROOT/'scripts').glob('*.py'))]
    protocol=dict(spec=spec,market_config=cfg,calibration_sha256=sha(ROOT/spec['calibration']),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source},
        training='Hawkes sum-exp simulator calibrated on train only; no historical validation/test access',
        selection='last fixed-budget checkpoint, all declared seeds retained; no PnL tuning',
        reward='CJR discounted inventory drift minus risk plus execution gain net maker fees',
        discount='exp(-r*tau), random physical waiting times; no constant discount per event',
        actions='a=kappa*delta >=0; no upper quote-distance cap; disabled sides masked',
        inventory='all critics output 21 discrete inventory heads; actors observe q',
        value_plus='all inventories per flow state; exact average over conditional order sides in Bellman',
        sac='v2: softplus Gaussian; temperature specified in native reward units then divided by value_scale like rewards/Q; automatic temperature; deterministic softplus(mean) evaluation',
        ppo='PPO-Clip, variable-discount GAE, KL early stop on actor only, bootstrap at rollout truncations',
        td3='two Q critics, clipped target noise with nonnegative action projection, delayed actor/target updates',
        limitations=['diagnostic previously viewed test days','no queue','no spot wallet/financing constraints',
                     'fixed fee scenarios','finite training budget; no claim of convergence or global optimality'],
        primary_sources=dict(td3='https://arxiv.org/abs/1802.09477',sac='https://arxiv.org/abs/1801.01290',
                             ppo='https://arxiv.org/abs/1707.06347'))
    protocol=json.loads(json.dumps(protocol,default=str))
    if args.stage=='prepare':
        out.mkdir(parents=True,exist_ok=True)
        p=out/'protocol.json'
        if p.exists() and json.loads(p.read_text())!=protocol:raise RuntimeError('Protocole différent : nouveau dossier requis')
        p.write_text(json.dumps(protocol,indent=2));(out/'calibration.json').write_text(json.dumps(cal,indent=2))
        (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    elif not (out/'protocol.json').exists() or json.loads((out/'protocol.json').read_text())!=protocol:
        raise RuntimeError('Préparer un protocole identique avant entraînement ou évaluation')
    return spec,cfg,cal,out


def checkpoint_path(out,algorithm,fee,seed):return out/'agents'/algorithm/f'{fee}_seed{seed}.pt'


def load_agent(path,p,spec):
    sd=torch.load(path,map_location='cpu',weights_only=False)
    if sd['algorithm_version']==DV.ALGORITHM_VERSION:
        ag=DV.Agent(p,{'agent':{'hidden':sd['hidden'],'observation':sd.get('observation','full')}},'cpu',sd['alpha_scale'],sd['value_scale'])
    else:ag=RL.Agent(p,sd['settings'],sd['algorithm'],sd['alpha_scale'],sd['value_scale'])
    ag.load(sd);return ag


def train_models(spec,cfg,cal,out,algorithm,fee_filter=None,seed_filter=None):
    torch.set_num_threads(spec['settings']['threads'])
    folder=out/'agents'/algorithm;folder.mkdir(parents=True,exist_ok=True)
    for fee,fees in cfg['realism']['fees'].items():
        if fee_filter is not None and fee!=fee_filter:continue
        p=M.Params.from_calibration(cal,cfg,fee_bps=fees['maker']*1e4)
        for seed in spec['seeds']:
            if seed_filter is not None and seed!=seed_filter:continue
            path=checkpoint_path(out,algorithm,fee,seed)
            if path.exists():
                load_agent(path,p,spec);print('Checkpoint vérifié, repris',path.name,flush=True);continue
            start=time.monotonic()
            def log(row):
                line=f'{algorithm} {fee} seed{seed} {row}'
                print(line,flush=True)
                with (folder/'training.log').open('a') as f:f.write(line+'\n')
            if algorithm=='value_plus':
                c=copy.deepcopy(cfg)
                c['agent']=dict(hidden=spec['settings']['hidden'],seed=seed,batch=spec['settings']['envs'],
                    iterations=spec['settings']['transitions']//spec['settings']['envs'],
                    lr_critic=spec['settings']['critic_lr'],polyak=spec['settings']['tau'])
                ag,hist=DV.train(p,c,'cpu',log=log)
            else:ag,hist=RL.train(p,spec['settings'],algorithm,seed,log=log)
            sd=ag.state_dict();sd['training_seed']=seed;sd['protocol_sha256']=sha(out/'protocol.json')
            atomic_save(sd,path)
            path.with_suffix('.json').write_text(json.dumps(dict(history=hist,seconds=time.monotonic()-start,
                transitions=spec['settings']['transitions'],checkpoint='final fixed budget'),indent=2))
            del ag;gc.collect()


def controls(spec,cfg,cal,out):
    torch.set_num_threads(spec['settings']['threads'])
    folder=out/'controls';folder.mkdir(exist_ok=True);results=[]
    fees=cfg['realism']['fees'][cfg['realism']['central']['fee']]
    p=M.Params.from_calibration(cal,cfg,hawkes=False,alpha=False,fee_bps=fees['maker']*1e4)
    for seed in spec['seeds']:
        path=folder/f'value_plus_poisson_seed{seed}.pt'
        if path.exists():ag=load_agent(path,p,spec)
        else:
            c=copy.deepcopy(cfg);c['agent']=dict(hidden=spec['settings']['hidden'],seed=seed,
                batch=spec['settings']['envs'],iterations=spec['settings']['transitions']//spec['settings']['envs'],
                lr_critic=spec['settings']['critic_lr'],polyak=spec['settings']['tau'])
            ag,h=DV.train(p,c,'cpu',log=lambda x:print('Contrôle discret',seed,x,flush=True) if x['it']%1024==0 else None)
            atomic_save(ag.state_dict(),path)
        y=torch.zeros(2*p.K+1,M.dim(p));y[:,0]=torch.arange(-p.K,p.K+1)
        with torch.no_grad():d=ag.deltas(y).numpy();v=(ag.V(y)*ag.value_scale).numpy()
        gate=reference.poisson_policy_metrics(*cal['mean_rates'],p,v,d[:,0],d[:,1]);gate['seed']=seed
        results.append(gate);print(gate,flush=True)
    (folder/'poisson_discrete.json').write_text(json.dumps(dict(results=results,passed=all(r['passed'] for r in results),
        thresholds=dict(value_fraction=.01,policy_regret_fraction=.01,probability_error=.02),
        scope='Poisson diagnostic of inventory heads, not global Hawkes certification'),indent=2))


def constant_policy(p):
    def policy(y):
        d=torch.full((len(y),2),1/p.kappa+p.fee,device=y.device)
        return torch.where(RL.active_sides(y,p).bool(),d,torch.inf)
    return policy


def replay_shard(spec,cfg,cal,out,fee,day):
    """Independent fee/day shard; final evaluate reconciles the complete grid."""
    if fee not in cfg['realism']['fees'] or day not in [str(d) for d in cfg['splits'][spec['evaluation']['historical_split']]]:
        raise ValueError('Replay shard outside declared protocol')
    torch.set_num_threads(spec['settings']['threads'])
    folder=out/'evaluation/intraday';folder.mkdir(parents=True,exist_ok=True)
    recordpath=folder/f'{fee}_{day}.json';path=recordpath.with_suffix('.npz')
    if recordpath.exists() and path.exists():return
    fees=cfg['realism']['fees'][fee]
    p=M.Params.from_calibration(cal,cfg,fee_bps=fees['maker']*1e4)
    policies={};agents=[]
    for algorithm in spec['algorithms']:
        for seed in spec['seeds']:
            checkpoint=checkpoint_path(out,algorithm,fee,seed)
            sd=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if sd['training_seed']!=seed or sd['protocol_sha256']!=sha(out/'protocol.json'):
                raise ValueError('Shard checkpoint differs from protocol')
            ag=load_agent(checkpoint,p,spec);agents.append(ag);policies[f'{algorithm}__{seed}']=ag.deltas
    pp=M.Params.from_calibration(cal,cfg,hawkes=False,alpha=False,fee_bps=fees['maker']*1e4)
    _,da,db=vi.poisson_vi(*cal['mean_rates'],pp)
    policies.update(as_=avellaneda_stoikov(p,cfg['benchmark']['as_horizon_s']),
        poisson=simulate.table_policy(torch.tensor(da,dtype=torch.float32),torch.tensor(db,dtype=torch.float32),p.K),
        constant=constant_policy(p))
    book,orders=data.load_day(cfg,day)
    records,paths=replay.replay_day(book,orders,cal,cfg,policies,'cpu',day,1.,fees,[11],every=spec['evaluation']['path_sampling'])
    for row in records:
        key=row['strategy'];algorithm,seed=(key.rsplit('__',1) if '__' in key else (key,0))
        row.update(fee=fee,algorithm=algorithm,train_seed=int(seed))
        if not np.isclose(row['pnl_net'],row['pnl_gross']-row['maker_cost']-row['taker_cost'],rtol=0,atol=1e-7):
            raise ValueError('Shard costs do not reconcile')
        if not np.isclose(paths[f'{key}|pnl'][-1],row['pnl_net'],rtol=0,atol=1e-7):
            raise ValueError('Shard path does not reconcile')
        if np.max(np.abs(paths[f'{key}|q']))>p.K:raise ValueError('Shard inventory outside bounds')
    np.savez_compressed(path,**paths)
    recordpath.write_text(json.dumps(records,indent=2))
    print('Shard complete',day,fee,flush=True)


def evaluate(spec,cfg,cal,out):
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    folder=out/'evaluation';folder.mkdir(exist_ok=True);(folder/'intraday').mkdir(exist_ok=True)
    declared_checkpoints={str(checkpoint_path(out,a,f,s).relative_to(out)):sha(checkpoint_path(out,a,f,s))
        for a in spec['algorithms'] for f in cfg['realism']['fees'] for s in spec['seeds']}
    manifest=folder/'checkpoints.json'
    if manifest.exists() and json.loads(manifest.read_text())!=declared_checkpoints:
        raise RuntimeError('Checkpoints modifiés depuis le début de l’évaluation')
    manifest.write_text(json.dumps(declared_checkpoints,indent=2))
    policy_sets={};agents=[];checkpoint_hashes={};sims=[]
    for fee,fees in cfg['realism']['fees'].items():
        p=M.Params.from_calibration(cal,cfg,fee_bps=fees['maker']*1e4)
        policies={}
        for algorithm in spec['algorithms']:
            for seed in spec['seeds']:
                path=checkpoint_path(out,algorithm,fee,seed)
                ag=load_agent(path,p,spec);checkpoint_hashes[str(path.relative_to(out))]=sha(path)
                sd=torch.load(path,map_location='cpu',weights_only=False)
                assert sd['training_seed']==seed and sd['protocol_sha256']==sha(out/'protocol.json')
                key=f'{algorithm}__{seed}';policies[key]=ag.deltas;agents.append(ag)
        pp=M.Params.from_calibration(cal,cfg,hawkes=False,alpha=False,fee_bps=fees['maker']*1e4)
        _,da,db=vi.poisson_vi(*cal['mean_rates'],pp)
        policies.update(as_=avellaneda_stoikov(p,cfg['benchmark']['as_horizon_s']),
                        poisson=simulate.table_policy(torch.tensor(da,dtype=torch.float32),torch.tensor(db,dtype=torch.float32),p.K),
                        constant=constant_policy(p))
        policy_sets[fee]=policies
        # Simulation indépendante, sans recherche d'hyperparamètres ou choix de seed.
        sp=folder/f'simulation_{fee}.json'
        if sp.exists():sims.extend(json.loads(sp.read_text()))
        else:
            records=[]
            for name,pol in policies.items():
                r=simulate.evaluate(pol,p,spec['evaluation']['simulation_seconds'],spec['evaluation']['simulation_paths'],'cpu',seed=spec['evaluation']['simulation_seed'])
                scale=cfg['model']['lot_usd']/1e4*3600/spec['evaluation']['simulation_seconds']
                algorithm,seed=(name.rsplit('__',1) if '__' in name else (name,0))
                record=dict(fee=fee,algorithm=algorithm,train_seed=int(seed),pnl_usdt_hour=r['pnl'][0]*scale,
                    pnl_se_usdt_hour=r['pnl'][1]*scale,criterion=r['criterion'][0],criterion_se=r['criterion'][1])
                records.append(record);print('Simulation',fee,name,record['pnl_usdt_hour'],flush=True)
            sp.write_text(json.dumps(records,indent=2));sims.extend(records)
    pd.DataFrame(sims).to_csv(folder/'simulation.csv',index=False)
    for ag in agents:
        if isinstance(ag,RL.Agent):ag.to(device)
        else:ag.critic.to(device);ag.target.to(device);ag.device=device
    rows=[]
    days=[str(d) for d in cfg['splits'][spec['evaluation']['historical_split']]]
    for day in days:
        book,orders=data.load_day(cfg,day)
        for fee,fees in cfg['realism']['fees'].items():
            recordpath=folder/'intraday'/f'{fee}_{day}.json';path=recordpath.with_suffix('.npz')
            if recordpath.exists():records=json.loads(recordpath.read_text())
            else:
                records,paths=replay.replay_day(book,orders,cal,cfg,policy_sets[fee],device,day,1.,fees,[11],every=spec['evaluation']['path_sampling'])
                for row in records:
                    key=row['strategy'];algorithm,seed=(key.rsplit('__',1) if '__' in key else (key,0))
                    row.update(fee=fee,algorithm=algorithm,train_seed=int(seed))
                    if not np.isclose(row['pnl_net'],row['pnl_gross']-row['maker_cost']-row['taker_cost'],atol=1e-7):raise RuntimeError('Frais non réconciliés')
                    if not np.isclose(paths[f'{key}|pnl'][-1],row['pnl_net'],atol=1e-7):raise RuntimeError('Trajectoire non réconciliée')
                    if np.max(np.abs(paths[f'{key}|q']))>cfg['model']['K']:raise RuntimeError('Inventaire hors bornes')
                np.savez_compressed(path,**paths);recordpath.write_text(json.dumps(records,indent=2))
            rows.extend(records)
            pd.DataFrame(rows).to_csv(folder/'historique_detail.csv',index=False)
            print('Rejeu terminé',day,fee,len(rows),'lignes',flush=True)
        del book,orders;gc.collect()
    table=pd.DataFrame(rows)
    expected=len(days)*len(cfg['realism']['fees'])*(len(spec['algorithms'])*len(spec['seeds'])+3)
    assert len(table)==expected and np.isfinite(table.select_dtypes('number')).all().all()
    assert checkpoint_hashes==declared_checkpoints
    (folder/'verification.json').write_text(json.dumps(dict(status='PASS',rows=expected,days=len(days),
        algorithms=spec['algorithms'],seeds=spec['seeds'],cash_fees_paths_inventory=True,
        precision_certified=False,checkpoint_sha256=checkpoint_hashes),indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','controls','evaluate','report'])
    ap.add_argument('--config',default='configs/benchmark_algorithms.yaml')
    ap.add_argument('--algorithm',choices=['td3','sac','ppo','value_plus'])
    ap.add_argument('--fee')
    ap.add_argument('--seed',type=int)
    ap.add_argument('--day')
    args=ap.parse_args();spec,cfg,cal,out=setup(args)
    if args.stage=='prepare':print(out)
    elif args.stage=='train':
        if not args.algorithm:ap.error('--algorithm requis pour train')
        if args.fee is not None and args.fee not in cfg['realism']['fees']:ap.error('Unknown fee')
        if args.seed is not None and args.seed not in spec['seeds']:ap.error('Undeclared seed')
        train_models(spec,cfg,cal,out,args.algorithm,args.fee,args.seed)
    elif args.stage=='controls':controls(spec,cfg,cal,out)
    elif args.stage=='evaluate':
        if args.day:
            if not args.fee:ap.error('--fee required with --day')
            replay_shard(spec,cfg,cal,out,args.fee,args.day)
        else:evaluate(spec,cfg,cal,out)
    else:
        from report_suite import report
        report(out)


if __name__=='__main__':main()
