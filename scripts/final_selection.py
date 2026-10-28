"""Étapes 2 à 6 de PROGRAMME_APRES_RESULTATS.txt : table des scores, choix des hyperparamètres,
contrôles classiques re-réglés, sélection finale. Validation 2024 uniquement, règle principale 1 tick.

  candidates : fusionne initial.json, evaluations.json et rescore_strict.json de chaque job terminé
               en une table (un candidat par ligne, scores 0/1/2 ticks, fills, inventaire).
  classical  : re-règle les six contrôles classiques à 1 tick (3 candidats chacun, garde-fous).
  grid       : choisit la combinaison d'hyperparamètres par algorithme/frais.
  select     : choisit le checkpoint final par algorithme/frais/seed dans la combinaison retenue.
Garde-fous (fixés le 27/10, avant le test) : au moins 5 fills/jour et |inventaire| moyen < 5 lots,
mesurés à 1 tick sur la validation.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.train_checkpoint_study import sha,write_json

RULE='1'
MIN_FILLS,MAX_ABS_Q=5.,5.
OUT=ROOT/'experiences/final_selection'
RUNS={'reference':'experiences/finetune_study_v1','grid41':'experiences/finetune_grid_v1','grid53':'experiences/finetune_grid_v1_seed53',
      'confirm':'experiences/finetune_confirm_v1'}
CLASSICAL=['constant','as','glft','poisson','local_hawkes','local_hawkes_alpha']


def flat(v):
    """Score de validation à plat : règle 0 (champ principal) puis règles strictes."""
    row={'pnl_0':v['mean_daily_pnl'],'fills_0':v['mean_fills'],'absq_0':v['mean_abs_inventory'],'median_0':v['median_daily_pnl']}
    for m,s in v.get('rules',{}).items():
        row|={f'pnl_{m}':s['mean_daily_pnl'],f'fills_{m}':s['mean_fills'],f'absq_{m}':s['mean_abs_inventory'],
              f'median_{m}':s['median_daily_pnl'],f'se_{m}':float(np.std(s['daily_pnl'],ddof=1)/np.sqrt(len(s['daily_pnl'])))}
    return row


def eligible(frame,rule=RULE):
    return (frame[f'fills_{rule}']>=MIN_FILLS)&(frame[f'absq_{rule}']<MAX_ABS_Q)


def candidates(require_complete=True):
    rows=[]
    for run,out in RUNS.items():
        for summary in sorted((ROOT/out).glob('*/*/*/seed*/summary.json')):
            job=summary.parent;arm,algo,fee,seed=job.parts[-4],job.parts[-3],job.parts[-2],int(job.parts[-1][4:])
            init=json.loads((job/'initial.json').read_text());ev=json.loads((job/'evaluations.json').read_text())
            rescore=json.loads((job/'rescore_strict.json').read_text()) if (job/'rescore_strict.json').exists() else {}
            items=[dict(kind='start',candidate=name,transitions=0,actor='current',path=c['path'],sha256=c['sha256'],validation=c['validation'])
                   for name,c in init['candidates'].items()]
            items+=[dict(kind='finetuned',candidate='finetuned',transitions=r['transitions'],actor=r['actor'],path=r['path'],
                         sha256=r['sha256'],validation=r['validation']) for r in ev]
            for it in items:
                v=it.pop('validation')
                if 'rules' not in v and it['path'] in rescore:
                    if rescore[it['path']]['sha256']!=it['sha256']:raise ValueError(f'Empreinte réévaluée différente : {it["path"]}')
                    v=rescore[it['path']]['validation']
                rows.append(dict(run=run,arm=arm,algorithm=algo,fee=fee,seed=seed,**it,**flat(v)))
    frame=pd.DataFrame(rows)
    missing=frame[f'pnl_{RULE}'].isna() if f'pnl_{RULE}' in frame else pd.Series(True,index=frame.index)
    if require_complete and missing.any():
        raise ValueError(f'{int(missing.sum())} candidats sans score {RULE} tick (réévaluation non terminée)')
    return frame


def best_per_job(frame):
    """Meilleur candidat éligible de chaque job ; à défaut, meilleur sans garde-fou (signalé)."""
    out=[]
    for key,g in frame.groupby(['run','arm','algorithm','fee','seed']):
        ok=g[eligible(g)]
        pick=(ok if len(ok) else g).sort_values([f'pnl_{RULE}','transitions'],ascending=[False,True]).iloc[0]
        out.append(pick.to_dict()|{'eligible':bool(len(ok)),'n_candidates':len(g),'n_eligible':len(ok)})
    return pd.DataFrame(out)


def grid(frame):
    """Combinaison retenue par algorithme/frais : moyenne, sur les seeds de la grille, du meilleur
    candidat éligible à 1 tick. Les bras de référence entrent sur les mêmes seeds."""
    best=best_per_job(frame);seeds=sorted(best[best.run.str.startswith('grid')].seed.unique())
    arms=best[best.seed.isin(seeds)&best.run.isin(['reference','grid41','grid53'])].copy()
    # Les deux dossiers de grille sont les seeds d'une même combinaison.
    arms.loc[arms.run.isin(['grid41','grid53']), 'run']='grid41'
    score=arms.groupby(['algorithm','fee','run','arm']).agg(score=(f'pnl_{RULE}','mean'),seeds=('seed','nunique'),
        all_eligible=('eligible','all')).reset_index()
    score=score[score.seeds==len(seeds)]
    # Égalité : moins d'updates par pas, puis plus petit learning rate (référence : 8 updates, ×0,25).
    parsed=score.arm.str.extract(r'lr(?P<lr>[\d.]+)_ups(?P<ups>\d+)_anchor')
    score['ups']=parsed.ups.astype(float).fillna(8.);score['lr']=parsed.lr.astype(float).fillna(.25)
    chosen=score.sort_values(['score','ups','lr'],ascending=[False,True,True]).groupby(['algorithm','fee']).head(1)
    return score,chosen,seeds


def select(frame):
    """Checkpoint final par algorithme/frais/seed, dans la combinaison retenue par grid()."""
    sel=json.loads((OUT/'grid_selection.json').read_text());best=best_per_job(frame);chosen=[]
    for c in sel['chosen']:
        runs=['reference'] if c['run']=='reference' else ['grid41','grid53','confirm']
        g=best[(best.algorithm==c['algorithm'])&(best.fee==c['fee'])&(best.arm==c['arm'])&best.run.isin(runs)]
        if sorted(g.seed)!=[41,53,67]:raise ValueError(f'Seeds incomplètes pour {c["algorithm"]} {c["fee"]} {c["arm"]} : {sorted(g.seed)}')
        for r in g.to_dict('records'):
            if sha(ROOT/r['path'])!=r['sha256']:raise ValueError(f'Checkpoint modifié : {r["path"]}')
            chosen.append(r)
    frame=pd.DataFrame(chosen)
    write_json(OUT/'selection.json',dict(rule=f'{RULE} tick, max mean daily net P&L on 2024 validation, guards applied',
        guards=dict(min_fills=MIN_FILLS,max_abs_inventory=MAX_ABS_Q),grid_selection_sha256=sha(OUT/'grid_selection.json'),
        caveat='Max over many candidates is optimistic on validation; test days were already consulted',chosen=chosen))
    print(frame[['algorithm','fee','seed','run','arm','kind','transitions','actor','pnl_0','pnl_1','pnl_2','se_1','fills_1','absq_1','eligible']].round(1).to_string())


def classical():
    """Six contrôles × 3 candidats de réglage de l'étude 524k, évalués sur la validation (0/1/2 ticks).
    Retenu : meilleur P&L à 1 tick parmi les candidats éligibles (à défaut, le meilleur, signalé)."""
    import torch
    from scripts import finetune_checkpoint_study as FS
    from scripts.robust_study import classical_policy
    torch.set_num_threads(1)
    spec=json.loads((ROOT/'configs/finetune_study.json').read_text());cfg,cal=FS.market()
    grid=json.loads((FS.SOURCE/'protocol.json').read_text())['spec']['candidates'];rows=[]
    for fee in spec['fees']:
        p=FS.params(cfg,cal,fee);val=FS.validator(spec,cfg,cal,fee,'cpu')
        for method in CLASSICAL:
            for c in range(len(grid)):
                rows.append(dict(fee=fee,method=method,candidate=c,parameters=grid[c],**flat(FS.compact(val(classical_policy(method,c,p,cfg,cal,grid))))))
                print(fee,method,c,{k:round(rows[-1][k],1) for k in ('pnl_0','pnl_1','pnl_2','fills_1','absq_1')},flush=True)
    frame=pd.DataFrame(rows);frame['eligible']=eligible(frame)
    chosen=[]
    for (fee,method),g in frame.groupby(['fee','method']):
        ok=g[g.eligible];pick=(ok if len(ok) else g).sort_values([f'pnl_{RULE}','candidate'],ascending=[False,True]).iloc[0]
        chosen.append(pick.to_dict()|{'any_eligible':bool(len(ok))})
    frame.to_csv(OUT/'classical_validation.csv',index=False)
    write_json(OUT/'classical_selection.json',dict(rule=f'{RULE} tick',guards=dict(min_fills=MIN_FILLS,max_abs_inventory=MAX_ABS_Q),chosen=chosen))
    print(pd.DataFrame(chosen)[['fee','method','candidate','pnl_0','pnl_1','pnl_2','fills_1','absq_1','any_eligible']].round(1).to_string())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['candidates','grid','classical','select'])
    parser.add_argument('--partial',action='store_true',help='autorise des candidats sans score 1 tick (aperçu)')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='classical':return classical()
    frame=candidates(require_complete=not args.partial)
    frame=frame.dropna(subset=[f'pnl_{RULE}']) if args.partial else frame
    if args.stage=='candidates':
        frame.to_csv(OUT/('candidates_preview.csv' if args.partial else 'candidates.csv'),index=False)
        best=best_per_job(frame)
        print(best[['run','arm','algorithm','fee','seed','kind','transitions','actor','pnl_0','pnl_1','pnl_2','fills_1','absq_1','eligible']].round(1).to_string())
        return
    if args.stage=='select':return select(frame)
    score,chosen,seeds=grid(frame)
    if not args.partial:
        score.to_csv(OUT/'grid_scores.csv',index=False)
        write_json(OUT/'grid_selection.json',dict(rule=f'{RULE} tick',seeds=[int(s) for s in seeds],guards=dict(min_fills=MIN_FILLS,max_abs_inventory=MAX_ABS_Q),
            chosen=chosen.to_dict('records')))
    print(score.pivot_table(index=['run','arm'],columns=['algorithm','fee'],values='score').round(1).to_string());print(chosen.round(1).to_string())
if __name__=='__main__':main()
