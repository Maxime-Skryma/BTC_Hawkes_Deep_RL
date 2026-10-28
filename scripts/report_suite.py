"""Rapport scientifique autonome hors ligne, figures PNG/PDF et tableaux CSV."""
from pathlib import Path
import json
from html import escape
import numpy as np
import pandas as pd
from cjr.fees import fee_label
from cjr.statistics import build_statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'td3':'TD3 Hawkes','sac':'SAC Hawkes','ppo':'PPO Hawkes','value_plus':'Inventory-head value network',
        'as':'Avellaneda–Stoikov','poisson':'Poisson optimal','constant':'Constant quotes'}
COLORS={'td3':'#2563eb','sac':'#7c3aed','ppo':'#e11d48','value_plus':'#059669',
        'as':'#d97706','poisson':'#334155','constant':'#94a3b8'}


def report(out):
    out=Path(out);ev=out/'evaluation';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    protocol=json.loads((out/'protocol.json').read_text());spec=protocol['spec'];cfg=protocol['market_config']
    d=pd.read_csv(ev/'historique_detail.csv');d['algorithm']=d.algorithm.replace({'as_':'as'})
    expected=len(cfg['splits'][spec['evaluation']['historical_split']])*len(cfg['realism']['fees'])*(len(spec['algorithms'])*len(spec['seeds'])+3)
    assert len(d)==expected and d.groupby(['fee','strategy','date']).size().eq(1).all()
    seeds=d.groupby(['fee','algorithm','train_seed'],as_index=False).agg(pnl=('pnl_net','sum'),
        fills=('n_fills','sum'),maker=('maker_cost','sum'),taker=('taker_cost','sum'),
        turnover=('turnover','sum'),drawdown=('max_drawdown','max'))
    summary=seeds.groupby(['fee','algorithm'],as_index=False).agg(pnl_mean=('pnl','mean'),
        pnl_std=('pnl','std'),pnl_min=('pnl','min'),pnl_max=('pnl','max'),seeds=('train_seed','size'),
        fills=('fills','mean'),maker=('maker','mean'),taker=('taker','mean'),turnover=('turnover','mean'),
        daily_drawdown=('drawdown','mean')).fillna(0)
    daily=d.groupby(['fee','algorithm','date'],as_index=False).pnl_net.mean()
    positives=daily.assign(positive=daily.pnl_net>0).groupby(['fee','algorithm']).positive.sum()
    summary['positive_days']=[int(positives.loc[(r.fee,r.algorithm)]) for r in summary.itertuples()]
    summary.to_csv(ev/'resume.csv',index=False);seeds.to_csv(ev/'totaux_par_seed.csv',index=False)
    daily.to_csv(ev/'moyennes_journalieres.csv',index=False)
    dates=sorted(d.date.unique());fees=list(cfg['realism']['fees']);learned=spec['algorithms'];names=list(LABELS)
    build_statistics(out)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
                         'grid.alpha':.16,'figure.facecolor':'white'})
    note=f'{cfg["data"]["symbol"]} · {len(dates)} observed days · {len(spec["seeds"])} seeds · no order queue model · exploratory study'
    def fee_panels(size):
        cols=min(2,len(fees));rows=(len(fees)+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=size,squeeze=False)
        for ax in axes.flat[len(fees):]:ax.set_visible(False)
        return fig,axes
    def save(fig,name):
        fig.text(.5,.012,note,ha='center',fontsize=8,color='#475569')
        fig.tight_layout(rect=(0,.045,1,.94))
        for ext in ('png','pdf'):fig.savefig(figdir/f'{name}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    curves={}
    fig,axes=fee_panels((14,9))
    for ax,fee in zip(axes.flat,fees):
        curves[fee]={}
        for name in names:
            g=d[(d.fee==fee)&(d.algorithm==name)].pivot(index='date',columns='train_seed',values='pnl_net').loc[dates]
            trajectories=np.vstack((np.zeros((1,g.shape[1])),g.cumsum().to_numpy()))
            avg=trajectories.mean(1);lo=trajectories.min(1);hi=trajectories.max(1)
            curves[fee][name]=dict(mean=avg.tolist(),low=lo.tolist(),high=hi.tolist())
            if name=='none':continue
            ax.plot(range(len(dates)+1),avg,label=LABELS[name],color=COLORS[name],lw=1.6 if name in learned else 1.3,
                    ls='-' if name in learned else '--')
            if name in learned:ax.fill_between(range(len(dates)+1),lo,hi,color=COLORS[name],alpha=.07)
        ax.axhline(0,color='#94a3b8',lw=.6)
        ax.set_xticks(range(len(dates)+1),['Start']+[x[:7] for x in dates],rotation=25)
        ax.set_title(f'Maker {cfg["realism"]["fees"][fee]["maker"]*1e4:g} bps',loc='left',fontweight='bold')
        ax.set_ylabel('Cumulative net P&L (USDT)')
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Hawkes Deep RL: historical comparison',fontsize=17,fontweight='bold')
    save(fig,'pnl_historique')
    fig,axes=fee_panels((14,8))
    for ax,fee in zip(axes.flat,fees):
        s=summary[summary.fee==fee].set_index('algorithm').loc[names]
        ax.bar(range(len(names)),s.pnl_mean,yerr=s.pnl_std,color=[COLORS[n] for n in names],capsize=2)
        ax.set_xticks(range(len(names)),[LABELS[n] for n in names],rotation=35,ha='right',fontsize=7)
        ax.set_title(f'Maker {cfg["realism"]["fees"][fee]["maker"]*1e4:g} bps',loc='left')
        ax.set_ylabel('Total net P&L (USDT)');ax.axhline(0,color='grey',lw=.5)
    fig.suptitle('Historical net P&L; error bars: ±1 standard deviation across training seeds',fontweight='bold')
    save(fig,'pnl_total')
    central=cfg['realism']['central']['fee'];intraday_names=learned+['as','poisson']
    for metric,figname,ylabel in [('pnl','pnl_intrajournalier','P&L net (USDT)'),('q','inventaire','Inventory (lots)')]:
        cols=min(3,len(dates));rows=(len(dates)+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(5*cols,3.3*rows),squeeze=False)
        for extra in axes.flat[len(dates):]:extra.set_visible(False)
        for ax,day in zip(axes.flat,dates):
            with np.load(ev/'intraday'/f'{central}_{day}.npz') as paths:
                for name in intraday_names:
                    keys=([f'{name}__{seed}|{metric}' for seed in spec['seeds']] if name in learned else
                          [f'{"as_" if name=="as" else name}|{metric}'])
                    vals=np.mean([paths[k] for k in keys],axis=0)
                    ax.plot(paths['hour'],vals,color=COLORS[name],label=LABELS[name],lw=.8,ls='-' if name in learned else '--')
            ax.set(title=day,xlabel='Time (UTC hour)',ylabel=ylabel)
        axes[0,0].legend(fontsize=6,ncol=2)
        fig.suptitle(f'Intraday comparison; maker fee {cfg["realism"]["fees"][central]["maker"]*1e4:g} bps',fontweight='bold')
        save(fig,figname)
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for ax,name in zip(axes.flat,learned):
        for seed in spec['seeds']:
            hist=json.loads((out/'agents'/name/f'{central}_seed{seed}.json').read_text())['history']
            xx=[r.get('transitions',r.get('flow_transitions')) for r in hist]
            yy=[r.get('critic_loss',r.get('loss_critic')) for r in hist]
            ax.semilogy(xx,np.maximum(yy,1e-14),label=f'Seed {seed}',lw=1)
        ax.set(title=LABELS[name],xlabel='Flow transitions',ylabel='Critic loss');ax.legend(fontsize=7)
    fig.suptitle('Training diagnostics; loss scales differ across algorithms',fontweight='bold')
    save(fig,'apprentissage')
    sim=pd.read_csv(ev/'simulation.csv');sim.algorithm=sim.algorithm.replace({'as_':'as'})
    simsum=sim.groupby(['fee','algorithm'],as_index=False).agg(mean=('pnl_usdt_hour','mean'),std=('pnl_usdt_hour','std')).fillna(0)
    fig,axes=fee_panels((14,8))
    for ax,fee in zip(axes.flat,fees):
        s=simsum[simsum.fee==fee].set_index('algorithm').loc[names]
        ax.bar(range(len(names)),s['mean'],yerr=s['std'],color=[COLORS[n] for n in names],capsize=2)
        ax.set_xticks(range(len(names)),[LABELS[n] for n in names],rotation=35,ha='right',fontsize=7)
        ax.set_title(f'Maker {cfg["realism"]["fees"][fee]["maker"]*1e4:g} bps');ax.set_ylabel('Net P&L (USDT/h; maker fees included)')
    fig.suptitle('Synthetic evaluation; error bars: standard deviation across seed means',fontweight='bold')
    save(fig,'pnl_simule')
    md='# Comparaison Hawkes Deep RL\n\nTD3, SAC, PPO et critique amélioré ; toutes les seeds déclarées, aucune sélection sur le P&L.\n\n'
    md+='[Rapport interactif hors ligne](rapport.html) · [Courbes historiques](figures/pnl_historique.png) · [Totaux](figures/pnl_total.png) · [Intrajournalier](figures/pnl_intrajournalier.png) · [CSV](evaluation/resume.csv)\n\n'
    for fee in fees:
        md+=f'## Maker {cfg["realism"]["fees"][fee]["maker"]*1e4:g} bps\n\n| Méthode | P&L moyen (USDT) | Min–max seeds | Jours positifs de la courbe moyenne |\n|---|---:|---:|---:|\n'
        for r in summary[summary.fee==fee].sort_values('pnl_mean',ascending=False).itertuples():
            md+=f'| {LABELS[r.algorithm]} | {r.pnl_mean:.2f} | {r.pnl_min:.2f} à {r.pnl_max:.2f} | {r.positive_days}/{len(dates)} |\n'
        md+='\n'
    md+='Le scénario maker 7,5 bps est un test de frais élevés. Les scénarios 0, 1,5 et 3 bps constituent la comparaison principale ; les tarifs sont des hypothèses fixes, sans présumer de l’éligibilité du compte.\n\n'
    md+='[Analyse de robustesse, risque et incertitude](statistiques.html) : différences appariées à AS, stabilité entre journées et seeds, risque et intervalles exploratoires corrigés des comparaisons multiples.\n\n'
    md+='## Protocole et portée\n\n'
    md+=f'Hawkes bivarié somme de deux exponentielles (betas 2 et 0,05 s⁻¹), calibration train conservée. {spec["settings"]["transitions"]:,} transitions par méthode/frais/seed, réseaux de {spec["settings"]["hidden"]} neurones. Même actualisation physique exp(−rτ), frais, inventaire, tick, latence et moteur causal de rejeu. Les critiques ont 21 sorties d’inventaire ; le critique amélioré couvre les 21 inventaires par état de flux et moyenne exactement les côtés dans Bellman. Cela lui donne davantage de cibles contrefactuelles, pas davantage d’interactions de flux.\n\n'
    md+='TD3 : deux Q, politique et cibles différées, bruit cible. SAC : deux Q, politique Gaussienne transformée par softplus et température automatique ; son objectif inclut une entropie sur a=κδ. PPO : clipping, GAE avec actualisation variable, arrêt KL sur acteur et bootstrap des troncatures. Les actions sont positives sans plafond supérieur ; évaluations déterministes.\n\n'
    md+='Les dates test ont déjà été consultées. Ce sont des journées échantillonnées, sans trading entre dates. Les bandes min–max représentent la dispersion des seeds, pas un intervalle de confiance sur la rentabilité. Les budgets ne garantissent ni convergence ni optimalité. Aucun modèle de file réelle ni financement spot. Les frais sont des scénarios fixes.\n\n'
    md+='[Protocole complet et références primaires](protocol.json) · [Contrôles internes](controls/poisson_discrete.json) · [Vérifications](evaluation/verification.json)\n'
    (out/'RESULTATS.md').write_text(md)
    payload=dict(curves=curves,summary=summary.to_dict('records'),labels=LABELS,colors=COLORS,fees=fees,
                 feeLabels={f:fee_label(cfg["realism"]["fees"][f]) for f in fees},
                 dates=['Start']+[x[:7] for x in dates],central=central)
    page='''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DOTUSDT · Hawkes Deep RL</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#142238;font:15px system-ui,sans-serif}header{background:#101e33;color:white;padding:40px max(5vw,20px)}header small{color:#77ddc4;letter-spacing:2px}h1{font-size:34px;margin:12px 0}header p{color:#c4cedb;max-width:850px;line-height:1.6}main{max-width:1250px;margin:25px auto;padding:0 20px}.card{background:white;border:1px solid #e0e7ef;border-radius:12px;padding:24px;margin-bottom:20px}.bar{display:flex;align-items:center;justify-content:space-between;gap:15px}select{padding:9px;border:1px solid #cbd5e1;border-radius:6px;font:inherit}.legend{display:flex;flex-wrap:wrap;gap:14px;margin:15px 0}.legend label{font-size:13px;cursor:pointer}.note{font-size:13px;color:#5b6b80;line-height:1.6}svg{width:100%;height:auto}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:13px 10px;border-bottom:1px solid #edf0f5;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{color:#64748b;font-size:12px}a{color:#1d4ed8}img{max-width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.badge{display:inline-block;padding:5px 9px;background:#20344e;border-radius:5px;font-size:12px;margin:4px}.foot{padding:20px;color:#64748b;font-size:12px}@media(max-width:700px){h1{font-size:26px}.grid{grid-template-columns:1fr}.card{padding:15px}}
</style><header><small>RECHERCHE · MARKET MAKING</small><h1>Hawkes Deep RL sur DOTUSDT</h1><p>TD3, SAC et PPO, face à un critique amélioré, Avellaneda–Stoikov et Poisson optimal. Même moteur historique, mêmes coûts, les seeds déclarées.</p><span class="badge">__N_DAYS__ journées échantillonnées</span><span class="badge">__N_FEES__ scénarios de frais</span><span class="badge">Hawkes sum-exp</span><span class="badge">Étude exploratoire</span></header>
<main><section class="card"><div class="bar"><div><h2>P&L historique cumulé</h2><span class="note">Moyenne des seeds · bandes min–max · USDT nets après liquidation</span></div><select id="fee"></select></div><div class="legend" id="legend"></div><svg id="chart" viewBox="0 0 1100 420" role="img" aria-label="P&L cumulé par stratégie"></svg><p class="note">Chaque point ajoute une journée observée. Aucun trading n’est simulé entre les dates. Les bandes décrivent les entraînements, pas un intervalle de confiance sur le profit futur.</p></section>
<section class="card"><div class="bar"><h2>Résultats du scénario sélectionné</h2><a href="evaluation/resume.csv" download>CSV complet ↗</a></div><div class="tablewrap"><table><thead><tr><th>Méthode</th><th>Net moyen</th><th>Min seed</th><th>Max seed</th><th>Jours +</th><th>Fills moyens</th><th>Frais maker</th><th>Frais taker</th></tr></thead><tbody id="table"></tbody></table></div></section>
<section class="grid"><article class="card"><h2>À l’intérieur des journées</h2><a href="figures/pnl_intrajournalier.png"><img src="figures/pnl_intrajournalier.png" alt="Trajectoires intrajournalières"></a><a href="figures/pnl_intrajournalier.pdf">PDF</a> · <a href="figures/inventaire.png">Inventaires</a></article><article class="card"><h2>Apprentissage</h2><a href="figures/apprentissage.png"><img src="figures/apprentissage.png" alt="Courbes d’apprentissage"></a><p class="note">Les pertes ont des définitions différentes selon l’algorithme. Une perte basse ne certifie pas une politique optimale.</p></article></section>
<section class="card"><h2>Ce qui est comparé</h2><p class="note">Frais indiqués en points de base : 1 bp = 0,01 %. Le scénario maker 7,5 bps sert de test de frais élevés. Les scénarios maker 0, 1,5 et 3 bps constituent la comparaison principale. Les tarifs sont des hypothèses fixes, sans présumer de l’éligibilité du compte.</p><p class="note">Les critiques représentent les 21 inventaires par des sorties distinctes. Le critique amélioré moyenne exactement achat et vente dans Bellman. TD3 et SAC utilisent deux Q ; PPO conserve son optimisation sur trajectoires. SAC ajoute un bonus d’entropie à l’entraînement, puis utilise une politique déterministe à l’évaluation.</p><p class="note">Hawkes à deux exponentielles calibré sur train ; seuils et budgets fixés avant ces scores. Actions sans plafond supérieur de distance. Exécution sans file, inventaire signé et absence de financement spot explicite : ces résultats ne démontrent pas une rentabilité exécutable. Les journées test ont déjà été examinées.</p><a href="protocol.json">Protocole</a> · <a href="RESULTATS.md">Détail des résultats</a> · <a href="evaluation/verification.json">Vérifications</a> · <a href="figures/pnl_historique.pdf">Figure historique PDF</a> · <a href="figures/pnl_simule.png">Simulation synthétique séparée</a></section><div class="foot">Rapport autonome hors ligne · Toutes les seeds déclarées sont conservées · Aucun checkpoint choisi sur son P&L</div></main>
<script>const D=__DATA__;
const fee=document.getElementById('fee'),legend=document.getElementById('legend'),chart=document.getElementById('chart');const visible=new Set(['td3','sac','ppo','value_plus','as','poisson']);
D.fees.forEach(f=>{let o=document.createElement('option');o.value=f;o.textContent=D.feeLabels[f];fee.appendChild(o)});fee.value=D.central;
Object.keys(D.labels).forEach(n=>{let l=document.createElement('label');let c=document.createElement('input');c.type='checkbox';c.checked=visible.has(n);c.onchange=()=>{c.checked?visible.add(n):visible.delete(n);draw()};l.appendChild(c);let s=document.createElement('span');s.style.color=D.colors[n];s.textContent=' '+D.labels[n];l.appendChild(s);legend.appendChild(l)});
const ns='http://www.w3.org/2000/svg',fmt=x=>x.toLocaleString('fr-FR',{maximumFractionDigits:0});function add(tag,attrs,text){let e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;chart.appendChild(e);return e}
function draw(){chart.replaceChildren();let f=fee.value,keys=[...visible],v=[0];keys.forEach(k=>v.push(...D.curves[f][k].low,...D.curves[f][k].high));let lo=Math.min(...v),hi=Math.max(...v);let pad=Math.max((hi-lo)*.1,1);lo-=pad;hi+=pad;const X=i=>90+i*972/Math.max(1,D.dates.length-1),Y=v=>355-(v-lo)/(hi-lo)*310;
for(let j=0;j<5;j++){let val=lo+(hi-lo)*j/4,y=Y(val);add('line',{x1:85,x2:1070,y1:y,y2:y,stroke:'#e2e8f0'});add('text',{x:76,y:y+4,'text-anchor':'end','font-size':12,fill:'#64748b'},fmt(val))}D.dates.forEach((s,i)=>add('text',{x:X(i),y:385,'text-anchor':'middle','font-size':12,fill:'#64748b'},s));
keys.forEach(k=>{const c=D.curves[f][k],line=a=>a.map((v,i)=>(i?'L':'M')+X(i)+','+Y(v)).join(' ');let polygon=c.high.map((v,i)=>[X(i),Y(v)]).concat(c.low.map((v,i)=>[X(i),Y(v)]).reverse());add('polygon',{points:polygon.map(p=>p.join(',')).join(' '),fill:D.colors[k],opacity:.09});add('path',{d:line(c.mean),fill:'none',stroke:D.colors[k],'stroke-width':2,'stroke-dasharray':['as','poisson','constant','none'].includes(k)?'6 4':'none'});c.mean.forEach((v,i)=>{let p=add('circle',{cx:X(i),cy:Y(v),r:3,fill:D.colors[k]});let title=document.createElementNS(ns,'title');title.textContent=D.labels[k]+' · '+D.dates[i]+' : '+fmt(v)+' USDT';p.appendChild(title)})});
const body=document.getElementById('table');body.replaceChildren();D.summary.filter(r=>r.fee===f).sort((a,b)=>b.pnl_mean-a.pnl_mean).forEach(r=>{let tr=document.createElement('tr');[D.labels[r.algorithm],fmt(r.pnl_mean),fmt(r.pnl_min),fmt(r.pnl_max),r.positive_days+'/'+(D.dates.length-1),fmt(r.fills),fmt(r.maker),fmt(r.taker)].forEach((v,i)=>{let td=document.createElement('td');td.textContent=v;if(i===0)td.style.color=D.colors[r.algorithm];tr.appendChild(td)});body.appendChild(tr)})}
fee.onchange=draw;draw();</script></html>'''
    page=page.replace('DOTUSDT',escape(cfg['data']['symbol'])).replace('__N_FEES__',str(len(fees)))
    page=page.replace('<main>', '<main><section class="card"><h2>Robustesse, risque et incertitude</h2><p>Le meilleur P&amp;L moyen ne suffit pas à établir un gagnant. Analyse exploratoire des différences à AS, des journées et seeds, du risque et des intervalles corrigés.</p><a href="statistiques.html">Ouvrir l’analyse statistique complète →</a><a href="figures/comparaison_statistique.png"><img src="figures/comparaison_statistique.png" alt="Intervalles exploratoires des différences à AS"></a></section>',1)
    (out/'rapport.html').write_text(page.replace('__DATA__',json.dumps(payload,allow_nan=False)).replace('__N_DAYS__',str(len(dates))))
    print(out/'rapport.html',flush=True)
