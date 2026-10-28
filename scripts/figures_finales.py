"""Reconstruit les figures et tableaux depuis les résultats figés, sans rejeu."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'resultats'
FEES = ['retail_bnb', 'vip3_bnb', 'vip6_bnb']
TITLES = ['Maker fee: 7.5 bps', 'Maker fee: 3 bps', 'Maker fee: 1.5 bps']
COLORS = {'td3':'#2f7de1', 'sac':'#fc6228', 'as':'#555555'}
LABELS = {'td3':'TD3', 'sac':'SAC', 'as':'Avellaneda–Stoikov'}


def save(fig, name):
    for ext in ['png']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def panels(title, ylabel):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    fig.suptitle(title, fontsize=19, fontweight='bold')
    for i, (ax, label) in enumerate(zip(axes, TITLES)):
        ax.set_title(f'({chr(97+i)}) {label}', loc='left', fontsize=15, fontweight='bold', pad=13)
        ax.axhline(0, color='#555555', ls='--', lw=1.4)
        ax.grid(axis='y', color='#e5e5e5', lw=.8)
    axes[0].set_ylabel(ylabel)
    return fig, axes


def legend(fig, axes):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False, fontsize=11)


def bootstrap(daily, rng):
    # Blocs circulaires de deux séances observées, sans rééchantillonnage des seeds.
    n = len(daily)
    starts = rng.integers(n, size=(10000, (n+1)//2))
    idx = ((starts[:,:,None]+np.arange(2)) % n).reshape(10000, -1)[:,:n]
    return np.quantile(np.asarray(daily)[idx].mean(axis=1), [.025, .975])


def main():
    (OUT/'figures').mkdir(parents=True, exist_ok=True)
    (OUT/'tableaux').mkdir(exist_ok=True)
    raw = ROOT/'experiences/final_test/results.csv'
    if not raw.exists():
        raise SystemExit('Local results are missing. Follow docs/reproduction.md to run the study first.')
    f = pd.read_csv(raw)
    assert len(f)==2376 and f.date.nunique()==22
    assert not f.duplicated(['date','fee','method','seed','through_ticks']).any()
    assert np.isfinite(f[['pnl_net','pnl_gross','maker_cost','taker_cost','mean_abs_q']]).all().all()
    assert np.allclose(f.pnl_net, f.pnl_gross-f.maker_cost-f.taker_cost)
    for fee in FEES:
        for tick in [0,1,2]:
            for method in f.method.unique():
                g=f[(f.fee==fee)&(f.through_ticks==tick)&(f.method==method)]
                assert len(g)==22*(3 if method in ['td3','sac'] else 1)
    f.to_csv(OUT/'tableaux/resultats_detailles.csv', index=False)
    mainf=f[f.through_ticks==1]
    summary=[]
    for (tick,fee,method),g in f.groupby(['through_ticks','fee','method']):
        seeds=g.groupby('seed').pnl_net.mean()
        summary.append(dict(ticks=tick,frais=fee,methode=method,pnl_moyen=seeds.mean(),seed_min=seeds.min(),seed_max=seeds.max(),
            pnl_median=g.groupby('date').pnl_net.mean().median(),fills=g.n_fills.mean(),inventaire=g.mean_abs_q.mean(),
            pnl_brut=g.pnl_gross.mean(),frais_maker=g.maker_cost.mean(),frais_taker=g.taker_cost.mean()))
    pd.DataFrame(summary).to_csv(OUT/'tableaux/synthese.csv', index=False)
    sel=json.loads((ROOT/'experiences/final_selection/selection.json').read_text())
    selected=pd.DataFrame(sel['chosen'])
    selected.to_csv(OUT/'tableaux/checkpoints_retenus.csv', index=False)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':1.2,'axes.axisbelow':True,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=panels('Net performance across 22 test sessions', 'Mean net P&L (USDT / session)')
    methods=['td3','sac','as']
    for ax,fee in zip(axes,FEES):
        for i,m in enumerate(methods):
            v=mainf[(mainf.fee==fee)&(mainf.method==m)].groupby('seed').pnl_net.mean()
            ax.plot([i,i],[v.min(),v.max()],color=COLORS[m],lw=2,alpha=.5)
            ax.scatter(np.full(len(v),i),v,s=24,color=COLORS[m],alpha=.5,zorder=3)
            ax.scatter(i,v.mean(),s=80,color=COLORS[m],zorder=4)
        ax.set_xticks(range(3),['TD3','SAC','AS'],fontsize=11)
    save(fig,'01_net_performance')
    fig,axes=panels('Sensitivity to the execution rule', 'Mean net P&L (USDT / session)')
    for ax,fee in zip(axes,FEES):
        for m in methods:
            g=f[(f.fee==fee)&(f.method==m)].groupby(['through_ticks','seed']).pnl_net.mean().unstack()
            ax.plot(g.index,g.mean(axis=1),marker='o',lw=2.5,color=COLORS[m],label=LABELS[m])
            if m in ['td3','sac']:ax.fill_between(g.index,g.min(axis=1),g.max(axis=1),color=COLORS[m],alpha=.09)
        ax.set_xticks([0,1,2]);ax.set_xlabel('Required price penetration (ticks)')
    legend(fig,axes);save(fig,'02_execution_sensitivity')
    fig,axes=panels('Difference from Avellaneda–Stoikov · 1-tick rule','P&L difference (USDT / session)')
    rng=np.random.default_rng(20261028);contrasts=[]
    for ax,fee in zip(axes,FEES):
        base=mainf[(mainf.fee==fee)&(mainf.method=='as')].set_index('date').pnl_net.sort_index()
        for i,m in enumerate(['td3','sac']):
            g=mainf[(mainf.fee==fee)&(mainf.method==m)].groupby('date').pnl_net.mean().sort_index()
            diff=g-base;lo,hi=bootstrap(diff,rng)
            ax.errorbar(i,diff.mean(),yerr=[[diff.mean()-lo],[hi-diff.mean()]],fmt='o',color=COLORS[m],capsize=5,ms=9,lw=2)
            contrasts.append(dict(fee=fee,method=m,mean=diff.mean(),low95=lo,high95=hi,positive_days=int((diff>0).sum())))
        ax.set_xticks([0,1],['TD3','SAC']);ax.set_xlim(-.5,1.5)
    save(fig,'03_difference_from_as');pd.DataFrame(contrasts).to_csv(OUT/'tableaux/ecarts_as_bootstrap.csv',index=False)
    fig,axes=panels('Validation and test · 1-tick rule','Mean net P&L (USDT / session)')
    for ax,fee in zip(axes,FEES):
        for m in ['td3','sac']:
            a=selected[(selected.fee==fee)&(selected.algorithm==m)].pnl_1.mean()
            b=mainf[(mainf.fee==fee)&(mainf.method==m)].pnl_net.mean()
            ax.plot([0,1],[a,b],marker='o',ms=8,lw=2.5,color=COLORS[m],label=LABELS[m])
        ax.set_xticks([0,1],['2024 validation','2025–2026 test'],fontsize=10);ax.set_xlim(-.2,1.2)
    legend(fig,axes);save(fig,'04_validation_vs_test')
    fig,axes=panels('Cumulative P&L over observed sessions · 1-tick rule','Cumulative net P&L (USDT)')
    for ax,fee in zip(axes,FEES):
        for m in methods:
            g=mainf[(mainf.fee==fee)&(mainf.method==m)].pivot(index='date',columns='seed',values='pnl_net').sort_index().cumsum()
            x=np.arange(1,len(g)+1)
            ax.plot(x,g.mean(axis=1),lw=2,color=COLORS[m],label=LABELS[m])
            if m in ['td3','sac']:ax.fill_between(x,g.min(axis=1),g.max(axis=1),color=COLORS[m],alpha=.09)
        ax.set_xlabel('Observed monthly session');ax.set_xticks([1,6,12,18,22])
    legend(fig,axes);save(fig,'05_cumulative_session_pnl')
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='verification.json'}
    (OUT/'verification.json').write_text(json.dumps(dict(status='PASS',rows=len(f),days=22,checkpoints=18,source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),files=files),indent=2)+'\n')
    print('PASS: PNG figures, tables and verification')

if __name__=='__main__':main()
