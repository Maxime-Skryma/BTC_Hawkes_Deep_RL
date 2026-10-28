"""Réglage des hyperparamètres du fine-tuning TD3/SAC sur les journées de validation historiques 2024.

Réutilise tel quel scripts/finetune_checkpoint_study.py (job, select) : même point de départ, mêmes
évaluations, mêmes checkpoints conservés. Seule différence : chaque bras est une combinaison de la
grille (facteur de learning rate, updates par pas, poids d'ancrage), une seule seed.
  train  : entraîne toutes les combinaisons.
  table  : par algorithme/frais, meilleur P&L net moyen de validation de chaque combinaison et
           combinaison retenue. Aucune date test lue. Le maximum sur la grille est optimiste.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts import finetune_checkpoint_study as FS
from scripts.train_checkpoint_study import sha,write_json


def arm_name(lr,ups,anchor):return f'lr{lr:g}_ups{ups}_anchor{anchor:g}'


def arms(spec):
    g=spec['grid']
    return {arm_name(lr,ups,a):dict(learning_rate_scale=lr,updates_per_step=ups,anchor_weight=a)
            for lr,ups,a in itertools.product(g['learning_rate_scale'],g['updates_per_step'],g['anchor_weight'])}


def job(args):
    spec,arm,algo,fee,seed=args
    # Le facteur de learning rate n'est pas un réglage d'agent : il passe par la clé globale du job.
    s=copy.deepcopy(spec);s['arms']={arm:dict(arms(spec)[arm])};s['learning_rate_scale']=s['arms'][arm].pop('learning_rate_scale')
    return FS.job((s,arm,algo,fee,seed))


def table(spec):
    out=ROOT/spec['output'];rows=[]
    for arm,algo,fee,seed in itertools.product(arms(spec),spec['algorithms'],spec['fees'],spec['seeds']):
        d=out/arm/algo/fee/f'seed{seed}'
        if not (d/'summary.json').exists():raise ValueError(f'Entraînement incomplet : {d}')
        ev=json.loads((d/'evaluations.json').read_text());best=max(ev,key=lambda r:r['validation']['mean_daily_pnl'])
        init=json.loads((d/'initial.json').read_text())
        rows.append(dict(algorithm=algo,fee=fee,seed=seed,arm=arm,**arms(spec)[arm],initial=init['chosen'],
            initial_validation=init['candidates'][init['chosen']]['validation']['mean_daily_pnl'],
            best_validation=best['validation']['mean_daily_pnl'],best_transitions=best['transitions'],best_actor=best['actor'],
            last_validation=ev[-1]['validation']['mean_daily_pnl']))
    frame=pd.DataFrame(rows);frame.to_csv(out/'grid_validation.csv',index=False)
    chosen=frame.loc[frame.groupby(['algorithm','fee']).best_validation.idxmax()]
    write_json(out/'grid_selection.json',dict(rule='per algorithm/fee: arm with max best-checkpoint mean daily net P&L, 2024 validation, seed '+','.join(map(str,spec['seeds'])),
        chosen=chosen.to_dict('records'),caveat='Max over grid arms and checkpoints is optimistic; confirm on other seeds'))
    print(frame.pivot_table(index='arm',columns=['algorithm','fee'],values='best_validation').round(1).to_string())
    print(chosen[['algorithm','fee','arm','initial_validation','best_validation','best_transitions']].to_string())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','train','table'])
    parser.add_argument('--config',default='configs/finetune_grid.json');parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--cpus',help='affinité, ex. 48-103');args=parser.parse_args()
    spec=json.loads((ROOT/args.config).read_text())
    jobs=[(spec,arm,a,f,s) for arm in arms(spec) for a in spec['algorithms'] for f in spec['fees'] for s in spec['seeds']]
    if args.stage=='plan':print(json.dumps(dict(models=len(jobs),arms=arms(spec)),indent=2));return
    if args.stage=='table':return table(spec)
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    if args.cpus:
        lo,hi=map(int,args.cpus.split('-'));os.sched_setaffinity(0,range(lo,hi+1))
    import fcntl
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(spec=spec,inputs={n:sha(FS.SOURCE/n) for n in ('protocol.json','calibration.json','selection.json')},
            source={n:sha(ROOT/n) for n in ('cjr/finetune.py','cjr/historical_validation.py','cjr/rl_algorithms.py','cjr/replay.py',
                                            'scripts/finetune_checkpoint_study.py','scripts/finetune_grid_study.py')})
        stored=json.loads((out/'protocol.json').read_text()) if (out/'protocol.json').exists() else None
        amendments=stored.pop('amendments',[]) if stored else []
        if stored is not None and stored!=manifest:
            raise ValueError('Entrées ou code modifiés : utiliser un nouveau dossier de sortie')
        write_json(out/'protocol.json',manifest|({'amendments':amendments} if amendments else {}))
        todo=[j for j in jobs if not (out/j[1]/j[2]/j[3]/f'seed{j[4]}'/'summary.json').exists()]
        # Combinaisons les plus coûteuses (8 updates/pas) d'abord, pour équilibrer la fin.
        todo.sort(key=lambda j:-arms(spec)[j[1]]['updates_per_step'])
        with ProcessPoolExecutor(max_workers=min(args.workers,max(len(todo),1)),mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,f in enumerate(as_completed([pool.submit(job,j) for j in todo]),1):
                row=f.result();print(f'Completed {i}/{len(todo)} {row["arm"]} {row["algorithm"]} {row["fee"]} {row["seed"]}',flush=True)
if __name__=='__main__':main()
