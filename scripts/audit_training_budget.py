"""Audit completed transition budgets and validation trajectories; no test-based tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUNS={'grid41':'finetune_grid_v1','grid53':'finetune_grid_v1_seed53','confirm':'finetune_confirm_v1','reference':'finetune_study_v1'}
FEES=['retail_bnb','vip3_bnb','vip6_bnb']


def main():
    exp=ROOT/'experiences';out=ROOT/'resultats';table=out/'tableaux';table.mkdir(exist_ok=True,parents=True)
    selected=json.loads((exp/'final_selection/selection.json').read_text())['chosen']
    budgets=[]
    for run in RUNS.values():
        for path in (exp/run).rglob('summary.json'):
            summary=json.loads(path.read_text());ev=json.loads((path.parent/'evaluations.json').read_text())
            steps=sorted({r['transitions'] for r in ev});target=summary['settings']['transitions']
            assert steps==list(range(131072,target+1,131072)),str(path)
            budgets.append(dict(run=run,algorithm=summary['algorithm'],fee=summary['fee'],seed=summary['seed'],
                transitions=steps[-1],evaluations=len(ev),updates_per_step=summary['settings']['updates_per_step']))
    records=[];curves=[]
    for c in selected:
        job=exp/RUNS[c['run']]/c['arm']/c['algorithm']/c['fee']/f"seed{c['seed']}"
        initial=json.loads((job/'initial.json').read_text());ev=json.loads((job/'evaluations.json').read_text())
        base=ROOT/initial['candidates'][initial['chosen']]['path']
        sd=torch.load(base,map_location='cpu',weights_only=False)
        pretrain=int(sd.get('transitions',sd['settings']['transitions']))
        def add(step,v):
            strict=v.get('rules',{}).get('1')
            if strict is not None:
                curves.append(dict(algorithm=c['algorithm'],fee=c['fee'],seed=c['seed'],transitions=step,
                    score=strict['mean_daily_pnl'],eligible=strict['mean_fills']>=5 and strict['mean_abs_inventory']<5))
        # Starting candidate chosen for training; the final selection may instead retain the other start.
        for it in initial['candidates'].values():add(0,it['validation'])
        for it in ev:add(it['transitions'],it['validation'])
        chosen_sd=torch.load(ROOT/c['path'],map_location='cpu',weights_only=False)
        actual_selected=int(chosen_sd.get('transitions',chosen_sd['settings']['transitions']))
        last=[r['validation']['rules']['1']['mean_daily_pnl'] for r in ev if r['transitions']==1572864]
        records.append(dict(algorithm=c['algorithm'],fee=c['fee'],seed=c['seed'],base_transitions=pretrain,
            finetune_completed=max(r['transitions'] for r in ev),selected_kind=c['kind'],selected_extra_transitions=c['transitions'],
            selected_total_transitions=(pretrain+c['transitions']) if c['kind']=='finetuned' else actual_selected,
            selected_validation_pnl=c['pnl_1'],best_last_validation_pnl=max(last),eligible=c['eligible']))
    synthetic=[json.loads(p.read_text()) for p in (exp/'checkpoint_study_v1').rglob('summary.json')]
    report=dict(finetune_jobs=len(budgets),completed_finetune_transitions=sum(r['transitions'] for r in budgets),
        synthetic_jobs=len(synthetic),synthetic_last_min=min(s['last_transitions'] for s in synthetic),
        synthetic_last_max=max(s['last_transitions'] for s in synthetic),
        synthetic_best_min=min(s['best_transitions'] for s in synthetic),synthetic_best_max=max(s['best_transitions'] for s in synthetic),
        selected_start_count=sum(r['selected_kind']=='start' for r in records),
        selected_final_finetune_checkpoint_count=sum(r['selected_extra_transitions']==1572864 for r in records),
        conclusion='Declared budgets completed. This does not establish convergence or budget optimality; no isolated longer-budget control was run.')
    pd.DataFrame(records).to_csv(table/'audit_training_selected.csv',index=False)
    pd.DataFrame(budgets).to_csv(table/'audit_training_budgets.csv',index=False)
    (table/'audit_training.json').write_text(json.dumps(report,indent=2)+'\n')
    curves=pd.DataFrame(curves);curves.to_csv(table/'audit_training_curves.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    fig.suptitle('Historical validation during fine-tuning · selected configurations',fontsize=19,fontweight='bold')
    for row,algo in enumerate(['td3','sac']):
        for col,(fee,rate) in enumerate(zip(FEES,[7.5,3,1.5])):
            ax=axes[row,col];ax.set_title(f'({chr(97+row*3+col)}) {algo.upper()} · maker fee {rate:g} bps',loc='left',fontsize=13,fontweight='bold')
            ax.axhline(0,color='#555',ls='--',lw=1);ax.grid(axis='y',color='#e5e5e5')
            for seed,color in zip([41,53,67],['#2f7de1','#fc6228','#209e80']):
                g=curves[(curves.algorithm==algo)&(curves.fee==fee)&(curves.seed==seed)]
                full=g.groupby('transitions').score.max()
                eligible=g[g.eligible].groupby('transitions').score.max().reindex(full.index)
                ax.plot(full.index/1e6,full,color=color,lw=1,alpha=.22)
                ax.plot(eligible.index/1e6,eligible,color=color,lw=2,label=f'Seed {seed}')
            if row==1:ax.set_xlabel('Additional transitions (millions)')
            if col==0:ax.set_ylabel('Validation net P&L (USDT / session)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,frameon=False)
    for ext in ['png']:fig.savefig(out/'figures'/f'06_training_validation.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(report,indent=2))
    print(pd.DataFrame(records)[['algorithm','fee','seed','base_transitions','selected_total_transitions','selected_extra_transitions']].to_string(index=False))

if __name__=='__main__':main()
