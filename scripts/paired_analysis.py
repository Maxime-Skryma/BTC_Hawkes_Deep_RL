"""Reanalyse the existing publication without training or modifying its artifacts."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from cjr.study_statistics import analyse_study


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='experiences/publication/comparaison/evaluation/historique_detail.csv')
    parser.add_argument('--output',default='experiences/robust_published_statistics')
    parser.add_argument('--repetitions',type=int,default=50000)
    args=parser.parse_args()
    source=ROOT/args.input;out=ROOT/args.output
    if out.resolve().is_relative_to((ROOT/'experiences/publication').resolve()):
        raise ValueError('Publication is read-only')
    original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    frame=pd.read_csv(source)
    neural=['td3','sac','ppo','value_plus']
    classical=['as','constant','glft','poisson','local_hawkes','local_hawkes_alpha']
    frame.algorithm=frame.algorithm.replace({'as_':'as',**{a:a+'__full' for a in neural}})
    intervals,risk,metadata=analyse_study(frame,[a+'__full' for a in neural],classical,args.repetitions)
    metadata.update(input_sha256=original_hash,input=str(source),
        note='Original fixed-budget checkpoints, no validation tuning or information ablations. These require the separate robust study.',
        analysis_source_sha256=hashlib.sha256((ROOT/'cjr/study_statistics.py').read_bytes()).hexdigest())
    out.mkdir(parents=True,exist_ok=True)
    intervals.to_csv(out/'paired_intervals.csv',index=False)
    risk.to_csv(out/'risk_by_seed.csv',index=False)
    (out/'method.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    summary=[]
    for (fee,a,b,metric),group in intervals.groupby(['fee','algorithm','reference','metric']):
        summary.append(dict(fee=fee,algorithm=a,reference=b,metric=metric,
            mean_daily_difference=group.mean_daily_difference.iloc[0],
            signal='positif exploratoire' if (group.family_low>0).all() else
                   'négatif exploratoire' if (group.family_high<0).all() else 'incertain'))
    summary=pd.DataFrame(summary)
    summary.to_csv(out/'summary.csv',index=False)
    page=('<!doctype html><html lang="fr"><meta charset="utf-8"><title>Comparaisons appariées</title>'
        '<style>body{font:15px system-ui;margin:30px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}.scroll{overflow:auto}</style>'
        '<h1>Nouvelle analyse appariée des résultats existants</h1><p>Étude exploratoire sur neuf dates déjà consultées. '
        'Les modèles sont ceux de la publication initiale : le réglage sur validation et les ablations appartiennent à la nouvelle expérience.</p>'
        '<p><a href="paired_intervals.csv">Tous les intervalles CSV</a> · <a href="risk_by_seed.csv">Risque par seed</a> · '
        '<a href="method.json">Méthode et empreintes</a> · <a href="summary.csv">Résumé CSV</a></p>'
        '<p>Chaque réseau est comparé aux six contrôles classiques, sur les mêmes dates. Les intervalles simultanés '
        'couvrent ensemble les frais, les paires, les quatre métriques et les trois longueurs de blocs. '
        'Les seeds sont des trajectoires entières ; les contrôles déterministes ne sont pas dupliqués. '
        'Un écart positif de P&amp;L est favorable ; pour le drawdown et l’inventaire, un écart négatif est préférable. '
        'La correction est nominale et ne résout pas le faible nombre de journées.</p>'
        '<h2>Différences de P&amp;L net (USDT / journée observée)</h2><div class="scroll">'+
        summary[summary.metric=='pnl_net'].to_html(index=False,float_format=lambda v:f'{v:.2f}')+'</div>'
        '<h2>Risque et dispersion par entraînement</h2><div class="scroll">'+risk.to_html(index=False,float_format=lambda v:f'{v:.2f}')+
        '</div><h2>Méthode</h2><pre>'+escape(json.dumps(metadata,ensure_ascii=False,indent=2))+'</pre></html>')
    (out/'rapport.html').write_text(page)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original_hash
    print(out/'rapport.html')


if __name__=='__main__':main()
