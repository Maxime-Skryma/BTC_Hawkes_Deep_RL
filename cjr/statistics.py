"""Analyse exploratoire appariée : jours communs, seeds croisées, blocs temporels."""
from pathlib import Path
from itertools import combinations
from html import escape
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
from cjr.fees import fee_label

LABELS = {'td3':'TD3 Hawkes','sac':'SAC Hawkes','ppo':'PPO Hawkes',
          'value_plus':'Critique amélioré','as':'Avellaneda–Stoikov'}


def block_indices(n, length, repetitions, rng):
    """Blocs mobiles non circulaires sur l'ordre des dates observées."""
    if not 1 <= length <= n:
        raise ValueError('Longueur de bloc invalide')
    starts = rng.integers(0, n-length+1, size=(repetitions, math.ceil(n/length)))
    return (starts[..., None] + np.arange(length)).reshape(repetitions, -1)[:, :n]


def bootstrap_means(matrix, indices, rng):
    """Une seed est rééchantillonnée comme une trajectoire entière sur les dates.

    Le même rééchantillonnage des dates s'applique aux méthodes et à AS.
    Les seeds des différentes méthodes sont rééchantillonnées indépendamment.
    """
    seeds = rng.integers(0, matrix.shape[1], size=(len(indices), matrix.shape[1]))
    return matrix[indices[:, :, None], seeds[:, None, :]].mean(axis=(1, 2))


def interval(samples, alpha):
    return np.quantile(samples, [alpha/2, 1-alpha/2]).tolist()


def closing_drawdown(pnl):
    equity = np.r_[0., np.cumsum(pnl)]
    return float(np.max(np.maximum.accumulate(equity)-equity))


def analyse(frame, algorithms, repetitions=50000, random_seed=20261025):
    if repetitions < 1000:
        raise ValueError('Au moins 1000 réplications requises')
    d = frame.copy();d['algorithm'] = d.algorithm.replace({'as_':'as'})
    fees = sorted(d.fee.unique()); methods = [*algorithms, 'as']
    # Famille annoncée : vs AS et toutes les paires de méthodes apprises.
    family_size = len(fees)*(len(algorithms)+math.comb(len(algorithms), 2))
    rng = np.random.default_rng(random_seed)
    results=[];daily_rows=[];seed_rows=[];ci_rows=[];pair_rows=[]
    for fee in fees:
        part=d[d.fee==fee];dates=sorted(part.date.unique());n=len(dates)
        matrices={};groups={}
        for method in methods:
            g=part[part.algorithm==method]
            if g.empty or g.duplicated(['date','train_seed']).any():
                raise ValueError(f'Résultats manquants ou dupliqués : {fee}/{method}')
            matrix=g.pivot(index='date',columns='train_seed',values='pnl_net').reindex(dates)
            if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
                raise ValueError(f'Grille dates/seeds incomplète : {fee}/{method}')
            matrices[method]=matrix.to_numpy();groups[method]=g
        if matrices['as'].shape[1]!=1:
            raise ValueError('AS doit avoir une trajectoire de référence par date')
        base=matrices['as'][:,0]
        for method in methods:
            mat=matrices[method];g=groups[method];mean=mat.mean(axis=1);delta=mean-base
            totals=mat.sum(axis=0);delta_totals=(mat-base[:,None]).sum(axis=0)
            cut=max(1,n//2)
            loo=(delta.sum()-delta)/(n-1) if n>1 else np.array([np.nan])
            for date,value,diff in zip(dates,mean,delta):
                daily_rows.append(dict(fee=fee,algorithm=method,date=date,pnl_mean=value,as_pnl=base[dates.index(date)],delta_as=diff))
            for seed,gg in g.groupby('train_seed'):
                gg=gg.set_index('date').loc[dates]
                seed_rows.append(dict(fee=fee,algorithm=method,train_seed=int(seed),
                    pnl_total=float(gg.pnl_net.sum()),delta_as_total=float((gg.pnl_net.to_numpy()-base).sum()),
                    worst_day=float(gg.pnl_net.min()),max_daily_drawdown=float(gg.max_drawdown.max()),
                    closing_drawdown=closing_drawdown(gg.pnl_net.to_numpy()),mean_abs_inventory=float(gg.mean_abs_q.mean())))
            results.append(dict(fee=fee,algorithm=method,days=n,seeds=mat.shape[1],pnl_total_mean=totals.mean(),
                pnl_total_seed_min=totals.min(),pnl_total_seed_max=totals.max(),pnl_daily_mean=mean.mean(),
                delta_as_daily_mean=delta.mean(),delta_as_daily_median=np.median(delta),
                days_beating_as=int((delta>1e-9).sum()),day_win_rate=float((delta>1e-9).mean()),
                positive_days=int((mean>0).sum()),seeds_positive=int((totals>0).sum()),
                seeds_beating_as=int((delta_totals>0).sum()),
                delta_as_first_half_mean=float(delta[:cut].mean()),
                delta_as_second_half_mean=float(delta[cut:].mean()) if n>1 else np.nan,
                leave_one_day_out_delta_min=float(loo.min()),leave_one_day_out_delta_max=float(loo.max()),
                worst_day_mean=float(mean.min()),worst_seed_day=float(mat.min()),
                worst_20pct_days_mean=float(np.sort(mean)[:max(1,math.ceil(.2*n))].mean()),
                max_daily_drawdown=float(g.max_drawdown.max()),
                max_closing_drawdown=max(closing_drawdown(mat[:,i]) for i in range(mat.shape[1])),
                mean_abs_inventory=float(g.mean_abs_q.mean())))
        # <5 dates ou <3 seeds : ne pas fabriquer un IC dégénéré de validation rapide.
        eligible=n>=5 and all(matrices[a].shape[1]>=3 for a in algorithms)
        if not eligible:continue
        for length in (1,2,3):
            idx=block_indices(n,length,repetitions,rng)
            samples={a:bootstrap_means(matrices[a],idx,rng) for a in methods}
            for a in algorithms:
                for target,values in [('as',samples[a]-samples['as'])]:
                    lo,hi=interval(values,.05);flo,fhi=interval(values,.05/family_size)
                    ci_rows.append(dict(fee=fee,algorithm=a,reference=target,block_length=length,
                        mean_daily=float((matrices[a]-matrices['as']).mean()) if target=='as' else float(matrices[a].mean()),
                        ci95_low=lo,ci95_high=hi,family_low=flo,family_high=fhi))
            for a,b in combinations(algorithms,2):
                vals=samples[a]-samples[b];lo,hi=interval(vals,.05);flo,fhi=interval(vals,.05/family_size)
                pair_rows.append(dict(fee=fee,algorithm=a,reference=b,block_length=length,
                    mean_daily=float(matrices[a].mean()-matrices[b].mean()),ci95_low=lo,ci95_high=hi,
                    family_low=flo,family_high=fhi))
    summary=pd.DataFrame(results)
    ci=pd.DataFrame(ci_rows,columns=['fee','algorithm','reference','block_length','mean_daily','ci95_low','ci95_high','family_low','family_high'])
    pairs=pd.DataFrame(pair_rows,columns=ci.columns)
    for i,r in summary.iterrows():
        for target in ('as',):
            rows=ci[(ci.fee==r.fee)&(ci.algorithm==r.algorithm)&(ci.reference==target)]
            summary.loc[i,f'signal_{target}']='non évalué' if rows.empty else (
                'positif exploratoire' if (rows.family_low>0).all() else
                'négatif exploratoire' if (rows.family_high<0).all() else 'incertain')
    return dict(summary=summary,daily=pd.DataFrame(daily_rows),seeds=pd.DataFrame(seed_rows),intervals=ci,pairs=pairs),family_size


METHOD = '''Analyse exploratoire, ajoutée après consultation des résultats : aucune preuve confirmatoire ni rentabilité exécutable démontrée.
Unité de marché : une date observée, pas une paire date/seed indépendante. Les différences à AS sont appariées sur les mêmes dates.
Bootstrap croisé dates/seeds (50 000 réplications par défaut) : blocs mobiles non circulaires de 1, 2 et 3 observations consécutives dans l’ordre des dates ; les mêmes dates sont tirées pour toutes les méthodes. Chaque seed est tirée comme une trajectoire entière ; tirages indépendants entre méthodes. AS est conservé avec ces mêmes dates.
Les blocs portent sur des journées échantillonnées parfois espacées d’un mois, pas sur des jours calendaires consécutifs : sensibilité descriptive à une dépendance locale, sans garantie de stationnarité ni de couverture avec si peu de données.
IC percentiles à 95 % et intervalles élargis par Bonferroni sur toutes les comparaisons algorithme/frais vs AS et entre algorithmes. Cette correction est nominale et ne corrige ni les choix antérieurs, ni la sélection des dates, ni les hypothèses du simulateur. Les trois longueurs de blocs sont toutes publiées, sans choisir la plus favorable.
« Positif exploratoire » exige une borne inférieure corrigée > 0 pour les trois longueurs ; « incertain » ne prouve ni égalité ni absence d’effet. Aucun IC si moins de 5 dates ou 3 seeds par algorithme.
Risque : pire journée de la politique moyenne, pire journée toutes seeds, moyenne des 20 % pires journées (échantillon, pas une prévision de risque extrême), drawdown intrajournalier maximal mesuré par le moteur complet, drawdown aux clôtures sur la concaténation des seules journées observées, inventaire absolu moyen en lots. Les deux drawdowns sont différents ; aucun trading entre dates. Aucun seuil de risque n’a été fixé avant ces scores : pas de gagnant automatique ajusté du risque.
Les seeds mesurent ici les entraînements disponibles avec le protocole d’exécution conservé ; elles n’isolent pas toute l’incertitude d’exécution. Les dates déjà examinées doivent être remplacées par de nouvelles périodes pour une validation indépendante.'''


def build_statistics(out,repetitions=50000):
    out=Path(out);folder=out/'evaluation/statistics';folder.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((out/'protocol.json').read_text());cfg=protocol['market_config'];algorithms=protocol['spec']['algorithms']
    source=out/'evaluation/historique_detail.csv'
    frame=pd.read_csv(source)
    expected_dates={str(d) for d in cfg['splits'][protocol['spec']['evaluation']['historical_split']]}
    expected_seeds=set(protocol['spec']['seeds'])
    if set(frame.fee)!=set(cfg['realism']['fees']):raise ValueError('Scénarios de frais incomplets')
    for fee in cfg['realism']['fees']:
        for algorithm in algorithms:
            g=frame[(frame.fee==fee)&(frame.algorithm==algorithm)]
            if set(g.date)!=expected_dates or set(g.train_seed)!=expected_seeds:
                raise ValueError('Dates ou seeds différentes du protocole')
    tables,family=analyse(frame,algorithms,repetitions)
    for name,table in tables.items():table.to_csv(folder/f'{name}.csv',index=False)
    metadata=dict(status='exploratory',repetitions=repetitions,random_seed=20261025,block_lengths=[1,2,3],
        confidence=.95,multiple_comparison_family_size=family,correction='Bonferroni nominal percentile intervals',
        historical_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),method=METHOD.replace('50 000 réplications par défaut', f'{repetitions:,} réplications'),
        references=['https://arxiv.org/abs/2108.13264','https://www.stat.cmu.edu/~cshalizi/uADA/16/lectures/26.pdf'])
    (folder/'method.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    render(out,tables,cfg,METHOD.replace("50 000 réplications par défaut", f"{repetitions:,} réplications"))
    return tables


def render(out,tables,cfg,method):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fees=list(cfg['realism']['fees']);fig,axes=plt.subplots(len(fees),1,figsize=(11,3.3*len(fees)),squeeze=False)
    for ax,fee in zip(axes.flat,fees):
        g=tables['intervals'];g=g[(g.fee==fee)&(g.reference=='as')&(g.block_length==2)]
        for j,r in enumerate(g.itertuples()):
            ax.plot([r.family_low,r.family_high],[j,j],color='#94a3b8',lw=5)
            ax.plot([r.ci95_low,r.ci95_high],[j,j],color='#2563eb',lw=2)
            ax.plot(r.mean_daily,j,'o',color='#142238')
        ax.set_yticks(range(len(g)),[LABELS.get(a,a) for a in g.algorithm]);ax.axvline(0,color='#d97706',ls='--')
        ax.set_title(fee_label(cfg['realism']['fees'][fee]),fontsize=10)
        ax.set_xlabel('Avantage moyen sur AS (USDT / journée observée)')
        if g.empty:ax.text(.5,.5,'Échantillon insuffisant pour estimer les intervalles',ha='center',transform=ax.transAxes)
    fig.suptitle('Comparaison exploratoire à AS · blocs de 2 observations\nBleu : IC 95 % · gris : correction des comparaisons multiples',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'figures/comparaison_statistique.png',dpi=150);plt.close(fig)
    columns={'reference':'Référence','block_length':'Bloc (observations)','mean_daily':'Écart moyen / jour','ci95_low':'IC 95 % bas','ci95_high':'IC 95 % haut','family_low':'IC corrigé bas','family_high':'IC corrigé haut','algorithm':'Méthode','pnl_total_mean':'P&L total moyen','delta_as_daily_mean':'Écart AS / jour',
        'delta_as_daily_median':'Écart AS médian / jour','days_beating_as':'Jours > AS','day_win_rate':'Fréquence > AS',
        'seeds_positive':'Seeds rentables','seeds_beating_as':'Seeds > AS','signal_as':'Signal vs AS',
        'pnl_total_seed_min':'P&L min seed','pnl_total_seed_max':'P&L max seed',
        'leave_one_day_out_delta_min':'Écart AS min sans 1 jour','leave_one_day_out_delta_max':'Écart AS max sans 1 jour',
        'delta_as_first_half_mean':'Écart AS première moitié','delta_as_second_half_mean':'Écart AS seconde moitié',
        'worst_day_mean':'Pire jour moyen','worst_seed_day':'Pire jour toutes seeds',
        'worst_20pct_days_mean':'Moyenne 20 % pires jours','max_daily_drawdown':'Drawdown intrajournalier max',
        'max_closing_drawdown':'Drawdown clôtures max','mean_abs_inventory':'Inventaire absolu moyen (lots)'}
    def table(frame,cols=None):
        frame=frame.copy()
        if 'algorithm' in frame:frame['algorithm']=frame.algorithm.map(lambda a:LABELS.get(a,a))
        if 'reference' in frame:frame['reference']=frame.reference.map(lambda a:LABELS.get(a,a))
        return '<div class="scroll">'+frame[cols if cols else frame.columns].rename(columns=columns).to_html(index=False,float_format=lambda v:f'{v:.2f}',na_rep='non évalué',border=0)+'</div>'
    sections=[]
    for fee in fees:
        s=tables['summary'];s=s[s.fee==fee].sort_values('pnl_total_mean',ascending=False)
        learned=s[s.algorithm!='as'];best=learned.iloc[0]
        rows=tables['pairs'];rows=rows[rows.fee==fee]
        candidates=[]
        for a in learned.algorithm:
            ahead=True
            for b in learned.algorithm:
                if a==b:continue
                ab=rows[(rows.algorithm==a)&(rows.reference==b)]
                ba=rows[(rows.algorithm==b)&(rows.reference==a)]
                ahead &= (len(ab)==3 and (ab.family_low>0).all()) or (len(ba)==3 and (ba.family_high<0).all())
            if ahead and len(learned)>1:candidates.append(LABELS.get(a,a))
        verdict='Avantage entre méthodes soutenu dans ces rééchantillonnages : '+', '.join(candidates) if candidates else 'Pas de gagnant clairement établi entre les méthodes apprises.'
        text=f'Meilleur P&L moyen observé parmi les méthodes apprises : {LABELS.get(best.algorithm,best.algorithm)} ({best.pnl_total_mean:.2f} USDT). {verdict} Cela ne constitue pas une validation indépendante ni une recommandation de déploiement.'
        sections.append('<section><h2>'+escape(fee_label(cfg['realism']['fees'][fee]))+'</h2><p>'+escape(text)+'</p><h3>Performance et comparaison à AS</h3>'+table(s,['algorithm','pnl_total_mean','delta_as_daily_mean','delta_as_daily_median','days_beating_as','day_win_rate','seeds_positive','seeds_beating_as','signal_as'])+
            '<p>Jours et seeds sont des effectifs observés, pas des probabilités de profit futur. Le scénario compte '+str(int(best.days))+' dates et '+str(int(best.seeds))+' seeds par méthode apprise.</p><h3>Stabilité</h3>'+table(s,['algorithm','pnl_total_seed_min','pnl_total_seed_max','delta_as_first_half_mean','delta_as_second_half_mean','leave_one_day_out_delta_min','leave_one_day_out_delta_max'])+
            '<h3>Risque observé</h3>'+table(s,['algorithm','worst_day_mean','worst_seed_day','worst_20pct_days_mean','max_daily_drawdown','max_closing_drawdown','mean_abs_inventory'])+
            '<details><summary>Intervalles vs AS — toutes les longueurs de blocs (USDT/jour)</summary>'+table(tables['intervals'][tables['intervals'].fee==fee].drop(columns='fee'))+'</details><details><summary>Comparaisons entre méthodes apprises (USDT/jour)</summary>'+table(rows.drop(columns='fee'))+'</details></section>')
    links=' · '.join(f'<a href="evaluation/statistics/{name}.csv">{name}.csv</a>' for name in tables)
    page='<!doctype html><html lang="fr"><meta charset="utf-8"><title>Robustesse statistique · Hawkes RL</title><style>body{font:15px system-ui;margin:30px auto;max-width:1300px;padding:15px;color:#142238;background:#f3f6fa}section{background:white;padding:20px;margin:20px 0;border-radius:10px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}p{line-height:1.6}img{max-width:100%}summary{cursor:pointer;margin:15px 0}a{color:#2563eb}</style><h1>Robustesse et risque — analyse exploratoire</h1><a href="rapport.html">Rapport principal</a><p>'+escape(method).replace('\n','</p><p>')+'</p><p>'+links+'</p><a href="evaluation/statistics/method.json">Méthode, paramètres et empreintes des entrées</a><p><a href="https://arxiv.org/abs/2108.13264">Évaluation du deep RL (Agarwal et al.)</a> · <a href="https://www.stat.cmu.edu/~cshalizi/uADA/16/lectures/26.pdf">Bootstrap temporel (Shalizi)</a></p><img src="figures/comparaison_statistique.png" alt="Intervalles des différences à AS">'+''.join(sections)+'</html>'
    (out/'statistiques.html').write_text(page)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    build_statistics(args.output)
    print(Path(args.output)/'statistiques.html')
