"""Étape 7 de PROGRAMME_APRES_RESULTATS.txt : rejeu unique sur les 22 journées de test.

Modèles TD3/SAC retenus (experiences/final_selection/selection.json) et contrôles classiques re-réglés
(classical_selection.json), règles d'exécution 0, 1 et 2 ticks dans le même passage (tables δ
partagées). Refuse de s'exécuter une seconde fois. Résultat exploratoire : ces journées ont déjà
été consultées par les études précédentes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import torch
from cjr import historical_validation as HV
from scripts import finetune_checkpoint_study as FS
from scripts.robust_study import classical_policy
from scripts.train_checkpoint_study import sha,write_json

SEL=ROOT/'experiences/final_selection'
OUT=ROOT/'experiences/final_test'
TICKS=[0,1,2]


def day_job(day):
    torch.set_num_threads(1)
    cfg,cal=FS.market();cfg=copy.deepcopy(cfg);cfg['data']['raw']=FS.TEST_RAW
    cfg['evaluation'].update(warmup_s=600,stop_s=86340)
    if day not in cfg['splits']['test']:raise ValueError('Date hors test')
    grid=json.loads((FS.SOURCE/'protocol.json').read_text())['spec']['candidates']
    neural=json.loads((SEL/'selection.json').read_text())['chosen'];classical=json.loads((SEL/'classical_selection.json').read_text())['chosen']
    d=HV.prepare_day(cfg,cal,day,FS.TEST_CACHE,11);rows=[]
    for c in neural+classical:
        p=FS.params(cfg,cal,c['fee']);fee=cfg['realism']['fees'][c['fee']]
        if 'method' in c:
            pol=classical_policy(c['method'],int(c['candidate']),p,cfg,cal,grid);meta=dict(method=c['method'],seed=0,candidate=int(c['candidate']))
        else:
            if sha(ROOT/c['path'])!=c['sha256']:raise ValueError(f'Checkpoint modifié : {c["path"]}')
            pol=FS.load_agent(ROOT/c['path'],p,c['algorithm'])[0].deltas
            meta=dict(method=c['algorithm'],seed=int(c['seed']),run=c['run'],arm=c['arm'],path=c['path'],transitions=int(c['transitions']),actor=c['actor'])
        for m,r in HV.evaluate_day_rules(pol,d,cfg,fee,'cpu',TICKS).items():
            rows.append(dict(r,fee=c['fee'],through_ticks=m,**meta))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=22);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'results.csv').exists():raise ValueError('Test déjà exécuté : résultats figés')
    dates=FS.market()[0]['splits']['test']
    write_json(OUT/'protocol.json',dict(dates=dates,through_ticks=TICKS,primary_rule='1 tick',window_s=[600,86340],replay_seed=11,
        selection_sha256=sha(SEL/'selection.json'),classical_selection_sha256=sha(SEL/'classical_selection.json'),
        note='Single post-selection replay; test days were already consulted by earlier studies, results are exploratory'))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        frame=pd.DataFrame([r for rows in pool.map(day_job,dates) for r in rows])
    frame.to_csv(OUT/'results.csv',index=False)
    summary=frame.groupby(['through_ticks','fee','method']).agg(mean_daily_pnl=('pnl_net','mean'),median_daily_pnl=('pnl_net','median'),
        mean_fills=('n_fills','mean'),mean_abs_inventory=('mean_abs_q','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    print(summary[summary.through_ticks==1].pivot_table(index='method',columns='fee',values='mean_daily_pnl').round(1).to_string())
if __name__=='__main__':main()
