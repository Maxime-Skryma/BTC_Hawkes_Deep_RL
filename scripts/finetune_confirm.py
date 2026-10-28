"""Étape 5 de PROGRAMME_APRES_RESULTATS.txt : entraîne, sur les seeds manquantes, la combinaison
d'hyperparamètres retenue par algorithme/frais (experiences/final_selection/grid_selection.json).

Une combinaison de la référence (anchored, anchored_dr) a déjà ses trois seeds : rien à entraîner.
Une combinaison de la grille est entraînée avec exactement le job de scripts/finetune_grid_study.py.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import finetune_grid_study as G
from scripts import finetune_checkpoint_study as FS
from scripts.train_checkpoint_study import sha,write_json

SEEDS=[41,53,67]
OUT='experiences/finetune_confirm_v1'


def plan():
    sel=json.loads((ROOT/'experiences/final_selection/grid_selection.json').read_text())
    spec=json.loads((ROOT/'configs/finetune_grid.json').read_text())|dict(name='finetune_confirm_v1',output=OUT)
    jobs=[]
    for c in sel['chosen']:
        if c['run']=='reference':continue
        for seed in SEEDS:
            if seed in sel['seeds']:continue
            jobs.append((spec,c['arm'],c['algorithm'],c['fee'],seed))
    return spec,sel,jobs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','train'])
    parser.add_argument('--workers',type=int,default=26)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    if args.workers<1 or args.threads<1:parser.error('workers et threads doivent être positifs')
    spec,sel,jobs=plan()
    spec['runtime_threads']=args.threads
    if args.stage=='plan':
        for j in jobs:print(j[1:])
        print(f'{len(jobs)} jobs');return
    out=ROOT/OUT;out.mkdir(parents=True,exist_ok=True)
    write_json(out/'protocol.json',dict(spec=spec,grid_selection_sha256=sha(ROOT/'experiences/final_selection/grid_selection.json'),
        jobs=[j[1:] for j in jobs],source={n:sha(ROOT/n) for n in ('cjr/finetune.py','cjr/historical_validation.py','cjr/rl_algorithms.py',
        'cjr/replay.py','scripts/finetune_checkpoint_study.py','scripts/finetune_grid_study.py','scripts/finetune_confirm.py')}))
    todo=[j for j in jobs if not (out/j[1]/j[2]/j[3]/f'seed{j[4]}'/'summary.json').exists()]
    with ProcessPoolExecutor(max_workers=min(args.workers,max(len(todo),1)),mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,f in enumerate(as_completed([pool.submit(G.job,j) for j in todo]),1):
            row=f.result();print(f'Completed {i}/{len(todo)} {row["arm"]} {row["algorithm"]} {row["fee"]} {row["seed"]}',flush=True)
if __name__=='__main__':main()
