"""Fine-tuning TD3/SAC depuis checkpoint, sélection sur les journées de validation historiques 2024.

Étapes (chaque étape fige ses entrées par empreinte) :
  train   : pour chaque bras/algorithme/frais/seed, choisit le point de départ parmi l'ancien modèle
            (524k) et best.pt v1 sur la validation 2024, puis fine-tune ; chaque checkpoint évalué
            (acteur courant et moyenné) est conservé.
  select  : par algorithme/frais/seed, retient le candidat de meilleur P&L net moyen de validation
            parmi les points de départ et tous les checkpoints fine-tunés. Aucune date test lue.
  test    : rejoue une seule fois les modèles retenus sur les 22 journées test (déjà consultées :
            résultat exploratoire), avec le même moteur et la même fenêtre que le test final v1.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from cjr import model as M, rl_algorithms as RL, finetune as FT, historical_validation as HV
from scripts.train_checkpoint_study import sha,write_json,save,evaluate as synthetic_evaluate

SOURCE=ROOT/'experiences/robust_study_v1'
# États de reprise (≈67 Mo chacun, buffer inclus) hors du volume persistant, trop petit : CJR_RESUME_ROOT.
RESUME_ROOT=Path(os.environ['CJR_RESUME_ROOT']) if os.environ.get('CJR_RESUME_ROOT') else None


def resume_path(out):
    if RESUME_ROOT is None:return out/'resume.pt'
    path=RESUME_ROOT/out.relative_to(ROOT)/'resume.pt';path.parent.mkdir(parents=True,exist_ok=True);return path
VALIDATION_CACHE=Path(os.environ.get('CJR_VALIDATION_CACHE', str(ROOT/'data/cache')))
TEST_CACHE=Path(os.environ.get('CJR_TEST_CACHE', '/tmp/cjr_final_test_cache'))
TEST_RAW=os.environ.get('CJR_TEST_RAW', 'data/raw/dot').split(os.pathsep)


def market():
    protocol=json.loads((SOURCE/'protocol.json').read_text());cal=json.loads((SOURCE/'calibration.json').read_text())
    cfg=copy.deepcopy(protocol['market_config'])
    cfg['data']['raw']=os.environ.get('CJR_TEST_RAW', 'data/raw/dot').split(os.pathsep)
    return cfg,cal


def params(cfg,cal,fee):
    return M.Params.from_calibration(cal,cfg,fee_bps=cfg['realism']['fees'][fee]['maker']*1e4)


def load_agent(path,p,algo,device='cpu'):
    sd=torch.load(path,map_location='cpu',weights_only=False)
    ag=RL.Agent(p,sd['settings'],algo,sd['alpha_scale'],sd['value_scale']);ag.load(sd);return ag.to(device),sd


def initial_paths(spec,algo,fee,seed):
    old=sorted((SOURCE/'final/models'/f'{algo}__full'/fee).glob(f'candidate*_seed{seed}.pt'))
    if len(old)!=1:raise ValueError(f'Ancien modèle ambigu ou absent : {algo} {fee} {seed}')
    return {'old_524k':old[0],'v1_best':ROOT/spec['source_v1']/algo/fee/f'seed{seed}'/'best.pt'}


def validator(spec,cfg,cal,fee,device):
    c=copy.deepcopy(cfg);c['evaluation'].update(warmup_s=spec['window_s'][0],stop_s=spec['window_s'][1])
    return HV.HistoricalValidator(c,cal,cfg['splits']['validation'],cfg['realism']['fees'][fee],VALIDATION_CACHE,
                                  device,seed=spec['replay_seed'],forbidden_days=cfg['splits']['test'],
                                  extra_through_ticks=spec.get('extra_through_ticks',()))


def compact(score):return {k:v for k,v in score.items() if k!='days'}|{'daily_pnl':[d['pnl_net'] for d in score['days']]}


def job(args):
    spec,arm,algo,fee,seed=args
    torch.set_num_threads(spec.get('runtime_threads',1))
    out=ROOT/spec['output']/arm/algo/fee/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    cfg,cal=market();p=params(cfg,cal,fee);device=spec['device'] if torch.cuda.is_available() else 'cpu'
    val=validator(spec,cfg,cal,fee,device);ev=spec['evaluation']
    # 1. Point de départ choisi sur validation historique (pas sur le test).
    starts={}
    for name,path in initial_paths(spec,algo,fee,seed).items():
        ag,sd=load_agent(path,p,algo,device)
        starts[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),validation=compact(val(ag.deltas)))
    init_name=max(starts,key=lambda k:(starts[k]['validation']['mean_daily_pnl'],k=='v1_best'))
    _,init_sd=load_agent(ROOT/starts[init_name]['path'],p,algo)
    write_json(out/'initial.json',dict(candidates=starts,chosen=init_name,rule='max mean daily net P&L, 2024 validation days; tie -> v1_best'))
    # 2. Réglages : architecture du checkpoint, taux d'apprentissage réduits, options du bras.
    settings=dict(init_sd['settings']);settings.update(spec['settings']);settings.update(spec['arms'][arm])
    for k in ('actor_lr','critic_lr'):settings[k]=init_sd['settings'][k]*spec['learning_rate_scale']
    settings['threads']=spec.get('runtime_threads',1)
    rows=json.loads((out/'evaluations.json').read_text()) if (out/'evaluations.json').exists() else []
    rpath=resume_path(out)
    resume=torch.load(rpath,weights_only=False,map_location='cpu') if rpath.exists() else None
    # Sans état de reprise, le run repart de zéro : les évaluations précédentes ne correspondent plus.
    rows=[r for r in rows if r['transitions']<=resume['completed_steps']*settings['envs']] if resume is not None else []
    start=time.monotonic()
    def callback(agent,n,snapshot):
        if n%ev['every_transitions'] and n!=settings['transitions']:return False
        for kind,candidate in (('current',agent),('ema',FT.averaged(agent))):
            eval_agent=copy.copy(candidate).to('cpu')
            synthetic=synthetic_evaluate(eval_agent,p,ev['synthetic_paths'],ev['synthetic_events'],ev['synthetic_seed'])
            state=eval_agent.state_dict();state.update(training_seed=seed,study=spec['name'],arm=arm,transitions=n,actor_kind=kind,
                initial=init_name,initial_sha256=starts[init_name]['sha256'])
            path=out/f'ckpt_{n:08d}_{kind}.pt';save(path,state)
            gpu,_=load_agent(path,p,algo,device)
            rows.append(dict(transitions=n,actor=kind,path=str(path.relative_to(ROOT)),sha256=sha(path),synthetic=synthetic,validation=compact(val(gpu.deltas))))
        save(rpath,snapshot());write_json(out/'evaluations.json',rows)
        last=rows[-2:];print(json.dumps(dict(arm=arm,algorithm=algo,fee=fee,seed=seed,transitions=n,**{r['actor']:round(r['validation']['mean_daily_pnl'],2) for r in last})),flush=True)
        return False
    with (out/'training.jsonl').open('a') as log:
        FT.train(p,settings,algo,seed,init_sd,log=lambda row:log.write(json.dumps(row)+'\n'),checkpoint_callback=callback,resume_state=resume)
    best=max(rows,key=lambda r:r['validation']['mean_daily_pnl'])
    result=dict(arm=arm,algorithm=algo,fee=fee,seed=seed,initial=init_name,initial_validation=starts[init_name]['validation']['mean_daily_pnl'],
        best_finetuned=dict(path=best['path'],transitions=best['transitions'],actor=best['actor'],validation=best['validation']['mean_daily_pnl']),
        settings=settings,seconds_this_invocation=time.monotonic()-start)
    write_json(out/'summary.json',result);rpath.unlink(missing_ok=True);return result


def select(spec):
    out=ROOT/spec['output'];chosen=[];rows=[]
    for algo in spec['algorithms']:
        for fee in spec['fees']:
            for seed in spec['seeds']:
                cands=[]
                for arm in spec['arms']:
                    d=out/arm/algo/fee/f'seed{seed}'
                    if not (d/'summary.json').exists():raise ValueError(f'Entraînement incomplet : {d}')
                    init=json.loads((d/'initial.json').read_text())
                    for name,c in init['candidates'].items():cands.append(dict(candidate=name,arm='start',transitions=0,actor='current',path=c['path'],sha256=c['sha256'],validation=c['validation']['mean_daily_pnl']))
                    cands+=[dict(candidate='finetuned',arm=arm,transitions=r['transitions'],actor=r['actor'],path=r['path'],sha256=r['sha256'],validation=r['validation']['mean_daily_pnl'],synthetic=r['synthetic']['score']) for r in json.loads((d/'evaluations.json').read_text())]
                unique={c['path']:c for c in cands}.values()
                for c in unique:
                    if sha(ROOT/c['path'])!=c['sha256']:raise ValueError(f'Checkpoint modifié : {c["path"]}')
                    rows.append(dict(algorithm=algo,fee=fee,seed=seed,**{k:v for k,v in c.items() if k!='sha256'}))
                # Égalité : le point de départ (moins de degrés de liberté) puis le plus court.
                best=max(unique,key=lambda c:(c['validation'],c['arm']=='start',-c['transitions']))
                chosen.append(dict(algorithm=algo,fee=fee,seed=seed,**best))
    pd.DataFrame(rows).to_csv(out/'validation_candidates.csv',index=False)
    write_json(out/'selection.json',dict(rule='max mean daily net P&L on 2024 validation days, liquidation included',
        validation_days=market()[0]['splits']['validation'],window_s=spec['window_s'],candidates=len(rows),chosen=chosen,
        caveat='Maximum over many candidates is optimistic on validation; the test is the only unbiased comparison and its days were already consulted'))
    frame=pd.DataFrame(chosen);print(frame[['algorithm','fee','seed','candidate','arm','transitions','actor','validation']].to_string())


def test_day(args):
    spec,day=args;torch.set_num_threads(1)
    cfg,cal=market();cfg=copy.deepcopy(cfg);cfg['data']['raw']=TEST_RAW
    sel=json.loads((ROOT/spec['output']/'selection.json').read_text());rows=[]
    if day not in cfg['splits']['test']:raise ValueError('Date hors test')
    d=HV.prepare_day(cfg,cal,day,TEST_CACHE,11)
    for c in sel['chosen']:
        if sha(ROOT/c['path'])!=c['sha256']:raise ValueError('Checkpoint retenu modifié')
        ag,_=load_agent(ROOT/c['path'],params(cfg,cal,c['fee']),c['algorithm'])
        r=HV.evaluate_day(ag.deltas,d,cfg,cfg['realism']['fees'][c['fee']],'cpu')
        rows.append(dict(r,algorithm=c['algorithm'],fee=c['fee'],train_seed=c['seed'],candidate=c['candidate'],arm=c['arm'],transitions=c['transitions'],actor=c['actor']))
    return rows


def test(spec,workers):
    out=ROOT/spec['output'];dates=market()[0]['splits']['test']
    if (out/'test_results.csv').exists():raise ValueError('Test déjà exécuté : résultats figés')
    write_json(out/'test_protocol.json',dict(selection_sha256=sha(out/'selection.json'),dates=dates,window_s=[600,86340],device='cpu',replay_seed=11,
        note='Single post-selection replay; test days were already consulted by earlier studies, results are exploratory'))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        frame=pd.DataFrame([r for rows in pool.map(test_day,[(spec,d) for d in dates]) for r in rows])
    frame.to_csv(out/'test_results.csv',index=False)
    summary=frame.groupby(['fee','algorithm']).agg(mean_daily_pnl=('pnl_net','mean'),median_daily_pnl=('pnl_net','median'),
        mean_abs_inventory=('mean_abs_q','mean'),mean_fills=('n_fills','mean')).reset_index()
    summary.to_csv(out/'test_summary.csv',index=False);print(summary.to_string())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','train','select','test'])
    parser.add_argument('--config',default='configs/finetune_study.json');parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--cpu-budget',type=int);args=parser.parse_args()
    spec=json.loads((ROOT/args.config).read_text())
    jobs=[(spec,arm,a,f,s) for arm in spec['arms'] for a in spec['algorithms'] for f in spec['fees'] for s in spec['seeds']]
    if args.stage=='plan':print(json.dumps(dict(models=len(jobs),**spec),indent=2));return
    if args.stage=='select':return select(spec)
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    if args.cpu_budget:
        import os
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:args.cpu_budget])
    if args.stage=='test':return test(spec,args.workers)
    import fcntl
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(spec=spec,inputs={n:sha(SOURCE/n) for n in ('protocol.json','calibration.json','selection.json')},
            source={n:sha(ROOT/n) for n in ('cjr/finetune.py','cjr/historical_validation.py','cjr/rl_algorithms.py','cjr/replay.py','scripts/finetune_checkpoint_study.py')})
        stored=json.loads((out/'protocol.json').read_text()) if (out/'protocol.json').exists() else None
        amendments=stored.pop('amendments',[]) if stored else []
        if stored is not None and stored!=manifest:
            raise ValueError('Entrées ou code modifiés : utiliser un nouveau dossier de sortie')
        write_json(out/'protocol.json',manifest|({'amendments':amendments} if amendments else {}))
        todo=[j for j in jobs if not (out/j[1]/j[2]/j[3]/f'seed{j[4]}'/'summary.json').exists()]
        with ProcessPoolExecutor(max_workers=min(args.workers,max(len(todo),1)),mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,f in enumerate(as_completed([pool.submit(job,j) for j in todo]),1):
                row=f.result();print(f'Completed {i}/{len(todo)} {row["arm"]} {row["algorithm"]} {row["fee"]} {row["seed"]}',flush=True)
if __name__=='__main__':main()
