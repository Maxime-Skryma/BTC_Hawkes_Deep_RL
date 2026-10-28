"""Non-neural controls vs deep RL; the original benchmark is read-only."""
import argparse
import copy
import hashlib
from html import escape
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numba import set_num_threads

from cjr import data, replay
from cjr.baselines import avellaneda_stoikov
from cjr.comparisons.glft import GLFT
from cjr.comparisons.local_control import LocalControl
from cjr.fees import fee_label
from cjr.model import Params
from cjr.statistics import block_indices, bootstrap_means

ROOT = Path(__file__).resolve().parents[2]
NEW_METHODS = ['glft', 'local_hawkes', 'local_hawkes_alpha']
EXISTING_CONTROLS = ['constant', 'poisson']
LABELS = {'as': 'Avellaneda–Stoikov', 'as_': 'Avellaneda–Stoikov',
          'constant': 'Cotation constante', 'poisson': 'Poisson optimal',
          'glft': 'GLFT', 'local_hawkes': 'Contrôle local Hawkes',
          'local_hawkes_alpha': 'Contrôle local Hawkes + alpha',
          'sac': 'SAC', 'td3': 'TD3', 'ppo': 'PPO',
          'value_plus': 'Critique neuronal amélioré'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconcile_as(actual, expected):
    """A new policy cannot be merged unless the common engine reproduces AS."""
    if len(expected) != 1:
        raise ValueError('Exactly one original AS record required per fee/date')
    expected = expected.iloc[0]
    for name in ('pnl_net', 'pnl_gross', 'maker_cost', 'taker_cost', 'turnover',
                 'n_fills', 'q_end', 'mean_abs_q', 'max_drawdown'):
        if not np.isclose(actual[name], expected[name], rtol=0, atol=1e-7):
            raise ValueError(f'AS replay differs from original: {name}')


def verify_replay(rows, path, K):
    names = [row['strategy'] for row in rows]
    if len(names) != len(set(names)) or set(names) != {*NEW_METHODS, 'as_'}:
        raise ValueError('Incomplete or duplicate control replay')
    with np.load(path) as traces:
        for row in rows:
            if not np.isclose(row['pnl_net'], row['pnl_gross']-row['maker_cost']-row['taker_cost'], rtol=0, atol=1e-7):
                raise ValueError('Control costs do not reconcile')
            if not np.isclose(traces[f'{row["strategy"]}|pnl'][-1], row['pnl_net'], rtol=0, atol=1e-7):
                raise ValueError('Control trace does not reconcile')
            if np.max(np.abs(traces[f'{row["strategy"]}|q'])) > K or abs(row['q_end']) > K:
                raise ValueError('Control inventory outside limits')


def compare(frame, methods, repetitions=50000, contrasts=None):
    """Paired dates, whole training trajectories; deterministic policies stay single."""
    if repetitions < 1000:
        raise ValueError('At least 1000 bootstrap repetitions required')
    d = frame.copy()
    d.algorithm = d.algorithm.replace({'as_': 'as'})
    fees = sorted(d.fee.unique())
    contrasts = contrasts if contrasts is not None else [(a, 'as') for a in methods]
    if len(contrasts) != len(set(contrasts)) or any(a == b for a, b in contrasts):
        raise ValueError('Contrasts must be unique and compare distinct methods')
    family = len(fees) * len(contrasts)
    rng = np.random.default_rng(20261025)
    rows = []
    for fee in fees:
        part = d[d.fee == fee]
        dates = sorted(part.date.unique())
        matrices = {}
        for name in [*methods, 'as']:
            group = part[part.algorithm == name]
            if group.empty or group.duplicated(['date', 'train_seed']).any():
                raise ValueError(f'Missing or duplicate results: {fee}/{name}')
            mat = group.pivot(index='date', columns='train_seed', values='pnl_net').reindex(dates)
            if not np.isfinite(mat.to_numpy()).all():
                raise ValueError(f'Incomplete date/seed matrix: {fee}/{name}')
            matrices[name] = mat.to_numpy()
        if matrices['as'].shape[1] != 1:
            raise ValueError('AS must have one deterministic trajectory')
        for length in (1, 2, 3):
            if len(dates) < 5:
                continue
            idx = block_indices(len(dates), length, repetitions, rng)
            samples = {name: bootstrap_means(mat, idx, rng) for name, mat in matrices.items()}
            for name, reference in contrasts:
                mat = matrices[name]
                if mat.shape[1] == 2 or matrices[reference].shape[1] == 2:
                    continue
                draws = samples[name] - samples[reference]
                lo, hi = np.quantile(draws, [.025 / family, 1 - .025 / family])
                ci = np.quantile(draws, [.025, .975])
                rows.append(dict(fee=fee, algorithm=name, reference=reference, block_length=length,
                    delta_as_daily_mean=float(mat.mean()-matrices[reference].mean()),
                    ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                    family_low=float(lo), family_high=float(hi)))
    return pd.DataFrame(rows, columns=['fee', 'algorithm', 'reference', 'block_length',
        'delta_as_daily_mean', 'ci95_low', 'ci95_high', 'family_low', 'family_high'])


def report(out, frame, methods, repetitions):
    learned = [m for m in methods if m in ('sac', 'td3', 'ppo', 'value_plus')]
    controls = [m for m in methods if m not in learned]
    contrasts = [(a, 'as') for a in methods] + [(a, b) for a in learned for b in controls]
    intervals = compare(frame, methods, repetitions, contrasts)
    intervals.rename(columns={'delta_as_daily_mean': 'mean_daily_difference'}).to_csv(
        out/'evaluation/intervals.csv', index=False)
    intervals[intervals.reference == 'as'].to_csv(out/'evaluation/intervals_vs_as.csv', index=False)
    totals = frame.groupby(['fee', 'algorithm', 'train_seed']).pnl_net.sum().reset_index()
    summary = totals.groupby(['fee', 'algorithm']).pnl_net.agg(['mean', 'min', 'max']).reset_index()
    cfg = json.loads((out/'protocol.json').read_text())['market_config']
    seed_count=int(frame[frame.algorithm.isin(learned)].groupby('algorithm').train_seed.nunique().max())
    sections, summaries = [], []
    for fee in cfg['realism']['fees']:
        s = summary[summary.fee == fee].copy()
        base = s[s.algorithm.isin(['as', 'as_'])]['mean'].iloc[0]
        s['delta_as_total'] = s['mean'] - base
        s['signal_vs_as'] = 'référence'
        for i, row in s.iterrows():
            if row.algorithm in ('as', 'as_'):
                continue
            ci = intervals[(intervals.fee == fee) & (intervals.algorithm == row.algorithm) &
                           (intervals.reference == 'as')]
            s.loc[i, 'signal_vs_as'] = ('non évalué' if len(ci) != 3 else
                'positif exploratoire' if (ci.family_low > 0).all() else
                'négatif exploratoire' if (ci.family_high < 0).all() else 'incertain')
        if 'mean_abs_q' in frame:
            risk = frame[frame.fee == fee].groupby('algorithm').agg(
                inventory_mean=('mean_abs_q', 'mean'), worst_day=('pnl_net', 'min'),
                max_intraday_drawdown=('max_drawdown', 'max'))
            s = s.join(risk, on='algorithm')
        s['family'] = np.where(s.algorithm.isin(learned), 'neuronal', 'sans réseau')
        summaries.append(s)
        display = s.drop(columns='fee').sort_values('mean', ascending=False).copy()
        display.algorithm = display.algorithm.map(LABELS)
        display=display.rename(columns={'algorithm':'Méthode','mean':'P&L moyen (USDT)',
            'min':'Min seeds','max':'Max seeds','delta_as_total':'Écart total à AS',
            'signal_vs_as':'Signal vs AS','inventory_mean':'Inventaire |q| moyen',
            'worst_day':'Pire jour / seed','max_intraday_drawdown':'Drawdown intrajournalier max',
            'family':'Famille'})
        pair_table = intervals[(intervals.fee == fee) & (intervals.reference != 'as')].copy()
        pair_table.algorithm = pair_table.algorithm.map(LABELS)
        pair_table.reference = pair_table.reference.map(LABELS)
        sections.append('<section id="'+escape(fee)+'"><h2>'+escape(fee_label(cfg['realism']['fees'][fee]))+'</h2>'+
                        display.to_html(index=False, float_format=lambda v: f'{v:.2f}')+
                        '<details><summary>Deep RL / critique neuronal contre les contrôles sans réseau</summary>'+
                        pair_table.rename(columns={'delta_as_daily_mean': 'mean_daily_difference'}).drop(columns='fee').to_html(
                            index=False, float_format=lambda v: f'{v:.2f}')+'</details></section>')
    pd.concat(summaries).to_csv(out/'evaluation/resume.csv', index=False)
    make_figures(out, frame, intervals, methods, cfg)
    page = '''<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Deep RL contre contrôle classique — CJR/Hawkes</title><style>
:root{color-scheme:light}body{font:16px system-ui;max-width:1440px;margin:0 auto;padding:28px;color:#17263c;background:#f3f6fb}
h1{font-size:clamp(26px,4vw,44px);max-width:1000px;line-height:1.12}h2{color:#173858}p{line-height:1.7;max-width:1050px}section{background:white;margin:24px 0;padding:24px;border-radius:14px;overflow:auto;border:1px solid #e3e9f1}
td,th{padding:10px;text-align:right;border-bottom:1px solid #e3e9f1}th{background:#eaf0f8}td:first-child,th:first-child{text-align:left}table{font-size:13px;border-collapse:collapse;width:100%;white-space:nowrap}details{overflow:auto;padding:16px 0}summary{cursor:pointer;font-weight:600}img{max-width:100%;background:white;border-radius:12px}a{color:#1558ac}.badge{display:inline-block;padding:8px 14px;background:#e2ebf8;border-radius:24px;margin:4px;font-weight:600}.note{background:#fff5db;padding:16px;border-radius:10px}nav{display:flex;gap:15px;flex-wrap:wrap;margin:24px 0}</style>
<h1>Deep RL contre contrôle classique — même rejeu CJR/Hawkes</h1>
__BADGES__
<p><strong>Question de recherche :</strong> l'apprentissage neuronal apporte-t-il un avantage sur des contrôles classiques qui observent le même marché ? Les écarts sont mesurés sur les mêmes journées, frais et règles d'exécution.</p>
<p>SAC, TD3, PPO et critique neuronal sont comparés à AS, à la cote constante,
au Poisson optimal et à trois contrôles supplémentaires sans réseau.
Les contrôles locaux résolvent exactement un petit HJB Poisson à chaque observation,
en figeant les intensités Hawkes courantes, avec ou sans alpha constant.
Ce sont des approximations locales, pas la solution optimale du marché Hawkes.
Les sauts, la décroissance et les exécutions du véritable marché restent inchangés.</p>
<p>Critère : avantage de P&amp;L net sur AS. GLFT est une politique déterministe,
sans seeds d'entraînement. Son objectif CARA/Poisson diffère de celui de CJR.
Approximation fermée stationnaire, frais intégrés par translation de la distance,
distances non négatives et mêmes limites d'inventaire que AS. Aucun réglage sur le test.</p>
<p class="note">Intervalles exploratoires par bootstrap croisé dates/trajectoires,
blocs de 1, 2 et 3 observations, correction de Bonferroni sur les méthodes et frais
comparés à AS et sur les comparaisons neuronales contre les contrôles classiques.
Le signal positif exige les trois bornes corrigées positives.
Les dates sont déjà vues et espacées : aucune validation indépendante.</p>
<p><a href="evaluation/historique_detail.csv">Résultats fusionnés</a> ·
<a href="evaluation/intervals_vs_as.csv">Tous les intervalles vs AS</a> ·
<a href="evaluation/intervals.csv">Tous les contrastes</a> ·
<a href="protocol.json">Protocole et empreintes</a></p>'''
    badges=f'<span class="badge">{escape(cfg["data"].get("symbol","Test"))}</span><span class="badge">{frame.date.nunique()} journées</span><span class="badge">{seed_count} seeds par méthode neuronale</span><span class="badge">{len(methods)+1} méthodes</span>' if 'data' in cfg else ''
    navigation='<nav>'+''.join(f'<a href="#{escape(f)}">{escape(fee_label(v))}</a>' for f,v in cfg['realism']['fees'].items())+'</nav>'
    (out/'rapport.html').write_text(page.replace('__BADGES__',badges) + navigation+
        '<h2>P&L historique</h2><p>Les bandes indiquent le minimum et le maximum des entraînements, pas un intervalle de confiance. Les journées sont échantillonnées ; aucun trading entre dates.</p><img src="figures/pnl_historique.png" alt="P&amp;L cumulé">'+
        '<h2>Avantage sur AS et incertitude</h2><p>Couleur : intervalle à 95 %. Gris : correction des comparaisons multiples. La figure montre le bloc 2 ; les trois longueurs sont publiées dans les CSV.</p><img src="figures/avantage_as.png" alt="Avantage quotidien sur AS">'+
        '<h2>Stabilité entre entraînements</h2><img src="figures/dispersion_seeds.png" alt="P&amp;L total de chaque entraînement et contrôles déterministes">'+''.join(sections) + '</html>')


def make_figures(out, frame, intervals, methods, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out/'figures';folder.mkdir(exist_ok=True)
    fees = list(cfg['realism']['fees'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
    colors = {'sac':'#7c3aed','td3':'#2563eb','ppo':'#e11d48','value_plus':'#059669',
              'as_':'#d97706','constant':'#94a3b8','poisson':'#334155','glft':'#a16207',
              'local_hawkes':'#0891b2','local_hawkes_alpha':'#c026d3'}
    for name in [*methods, 'as_']:
        colors.setdefault(name, '#333333')
    for kind in ('pnl_historique', 'avantage_as'):
        fig, axes = plt.subplots((len(fees)+1)//2, 2, figsize=(17, 6*((len(fees)+1)//2)), squeeze=False)
        for ax, fee in zip(axes.flat, fees):
            if kind == 'pnl_historique':
                for name in [*methods, 'as_']:
                    d = frame[(frame.fee == fee) & frame.algorithm.isin([name, 'as'] if name == 'as_' else [name])]
                    matrix = d.pivot(index='date', columns='train_seed', values='pnl_net').sort_index()
                    paths = np.vstack([np.zeros((1, matrix.shape[1])), matrix.cumsum().to_numpy()])
                    ax.plot(paths.mean(1), label=LABELS[name], color=colors[name],
                            ls='-' if name in ('sac','td3','ppo','value_plus') else '--')
                    if paths.shape[1] > 1:
                        ax.fill_between(np.arange(len(paths)), paths.min(1), paths.max(1), color=colors[name], alpha=.1)
                ax.set_ylabel('P&L net cumulé (USDT)')
                dates=sorted(frame.date.unique())
                ax.set_xticks(range(len(dates)+1), ['Départ']+[d[5:7]+'/'+d[2:4] for d in dates], rotation=35)
                ax.set_xlabel('Journées observées — aucun trading entre dates')
            else:
                d = intervals[(intervals.fee == fee) & (intervals.reference == 'as') & (intervals.block_length == 2)]
                for j, row in enumerate(d.itertuples()):
                    ax.plot([row.family_low, row.family_high], [j, j], lw=5, color='#aaaaaa')
                    ax.plot([row.ci95_low, row.ci95_high], [j, j], lw=2, color=colors[row.algorithm])
                    ax.plot(row.delta_as_daily_mean, j, 'o', color=colors[row.algorithm])
                ax.set_yticks(np.arange(len(d)), [LABELS[a] for a in d.algorithm])
                ax.axvline(0, color='black', lw=.7);ax.set_xlabel('Avantage moyen / jour sur AS (USDT) — bloc 2')
            ax.set_title(fee_label(cfg['realism']['fees'][fee]));ax.grid(alpha=.15)
        if kind == 'pnl_historique':
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
                       bbox_to_anchor=(.5,.96), ncol=5, fontsize=9, frameon=False)
        for ax in axes.flat[len(fees):]:ax.set_visible(False)
        title = 'P&L historique net — comparaison de toutes les méthodes' if kind == 'pnl_historique' else 'Avantage quotidien sur Avellaneda–Stoikov'
        note = 'Moyenne et min–max des seeds · journées échantillonnées · étude exploratoire' if kind == 'pnl_historique' else 'Couleur : IC à 95 % · gris : correction des comparaisons multiples · bootstrap bloc 2 · étude exploratoire'
        fig.suptitle(title, fontsize=18, fontweight='bold', y=.995)
        fig.text(.5,.012,note,ha='center',fontsize=10,color='#475569')
        fig.tight_layout(rect=(0,.04,1,.91 if kind == 'pnl_historique' else .95))
        for ext in ('png','pdf'):fig.savefig(folder/f'{kind}.{ext}', dpi=170)
        plt.close(fig)
    fig, axes=plt.subplots((len(fees)+1)//2,2,figsize=(17,6*((len(fees)+1)//2)),squeeze=False)
    for ax,fee in zip(axes.flat,fees):
        for j,name in enumerate([*methods,'as_']):
            d=frame[(frame.fee==fee)&frame.algorithm.isin([name,'as'] if name=='as_' else [name])]
            totals=d.groupby('train_seed').pnl_net.sum().to_numpy()
            jitter=np.linspace(-.12,.12,len(totals)) if len(totals)>1 else np.array([0.])
            ax.scatter(totals,j+jitter,s=25,color=colors[name],alpha=.8)
            ax.plot([totals.mean()],[j],marker='D',color='black',ms=4)
        ax.set_yticks(range(len(methods)+1),[LABELS[n] for n in [*methods,'as_']])
        ax.set_title(fee_label(cfg['realism']['fees'][fee]));ax.set_xlabel('P&L total (USDT) — un point par seed ; losange : moyenne')
        ax.axvline(0,color='#999999',lw=.7);ax.grid(axis='x',alpha=.2)
    for ax in axes.flat[len(fees):]:ax.set_visible(False)
    fig.suptitle('Stabilité des résultats entre entraînements', fontsize=18, fontweight='bold')
    fig.text(.5,.012,'Un point par seed · losange : moyenne · une trajectoire par contrôle déterministe',
             ha='center',fontsize=10,color='#475569')
    fig.tight_layout(rect=(0,.04,1,.95))
    for ext in ('png','pdf'):fig.savefig(folder/f'dispersion_seeds.{ext}',dpi=170)
    plt.close(fig)


def run(base, out, download=False, repetitions=50000):
    base, out = Path(base).resolve(), Path(out).resolve()
    if out == base or out.is_relative_to(base) or base.is_relative_to(out):
        raise ValueError('Output must be separate from the original benchmark')
    original_protocol = base/'protocol.json'
    original_csv = base/'evaluation/historique_detail.csv'
    protocol = json.loads(original_protocol.read_text())
    cfg = copy.deepcopy(protocol['market_config'])
    # Old published protocols contain an absolute path from before reorganisation.
    symbol = cfg['data']['symbol']
    cfg['data']['raw'] = [str(ROOT/'data/raw'/cfg['name']), *cfg['data']['raw']]
    cal = json.loads((base/'calibration.json').read_text())
    source_files = [Path(__file__), Path(__file__).with_name('glft.py'), Path(__file__).with_name('local_control.py'),
                    ROOT/'cjr/replay.py', ROOT/'cjr/data.py', ROOT/'cjr/baselines.py',
                    ROOT/'cjr/model.py', ROOT/'cjr/hawkes.py', ROOT/'cjr/calibrate.py',
                    ROOT/'cjr/statistics.py', ROOT/'cjr/download.py']
    manifest = dict(base_protocol_sha256=digest(original_protocol),
        base_results_sha256=digest(original_csv), calibration_sha256=digest(base/'calibration.json'),
        market_config=cfg, methods=[*protocol['spec']['algorithms'], *EXISTING_CONTROLS, *NEW_METHODS],
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_files},
        repetitions=repetitions, reference='as', deterministic_methods=[*NEW_METHODS, *EXISTING_CONTROLS, 'as'],
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__),
        gamma='same numeric gamma as AS, no test tuning; objectives differ',
        glft='stationary closed-form approximation; symmetric train mean arrival intensity; fee translation',
        local_control='Howard, tol=1e-10 scaled HJB residual, max_iter=100; exact per observation, no interpolation; frozen intensities and optionally constant alpha',
        bootstrap='same dates for all policies; whole training seed trajectories; blocks 1/2/3; Bonferroni across all declared vs-AS and neuronal-vs-classical contrasts')
    manifest = json.loads(json.dumps(manifest, default=str))
    out.mkdir(parents=True, exist_ok=True)
    target = out/'protocol.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Comparison protocol changed; choose a new output directory')
    target.write_text(json.dumps(manifest, indent=2))
    folder = out/'evaluation';folder.mkdir(exist_ok=True)
    (folder/'intraday').mkdir(exist_ok=True)
    original = pd.read_csv(original_csv)
    days = [str(d) for d in cfg['splits'][protocol['spec']['evaluation']['historical_split']]]
    methods = manifest['methods']
    original = original[original.algorithm.isin([*methods, 'as', 'as_'])].copy()
    for fee in cfg['realism']['fees']:
        for name in [*protocol['spec']['algorithms'], *EXISTING_CONTROLS, 'as_']:
            g = original[(original.fee == fee) & (original.algorithm.replace({'as': 'as_'}) == name)]
            seeds = {0} if name in ('as_', *EXISTING_CONTROLS) else set(protocol['spec']['seeds'])
            if set(g.date) != set(days) or set(g.train_seed) != seeds or len(g) != len(days)*len(seeds):
                raise ValueError('Original date/seed grid differs from its protocol')
    if download:
        from cjr.download import download as fetch
        selection = copy.deepcopy(cfg);selection['splits'] = {'test': days}
        fetch(selection, workers=2)
    torch.set_num_threads(2)
    set_num_threads(2)
    new = []
    for day in days:
        book, orders = data.load_day(cfg, day)
        inputs = dict(book=hashlib.sha256(pd.util.hash_pandas_object(book, index=True).values.tobytes()).hexdigest(),
                      orders=hashlib.sha256(pd.util.hash_pandas_object(orders, index=True).values.tobytes()).hexdigest())
        for fee, costs in cfg['realism']['fees'].items():
            path = folder/f'{fee}_{day}.json'
            trace_path = folder/'intraday'/f'{fee}_{day}.npz'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['inputs'] != inputs:
                    raise ValueError('Market data changed since comparison replay')
                if digest(trace_path) != saved['trace_sha256']:
                    raise ValueError('Control trace changed since comparison replay')
                rows = saved['rows']
            else:
                p = Params.from_calibration(cal, cfg, fee_bps=costs['maker']*1e4)
                rows, paths = replay.replay_day(book, orders, cal, cfg,
                    {'glft': GLFT(p, cal['mean_rates']).deltas,
                     'local_hawkes': LocalControl(p).deltas,
                     'local_hawkes_alpha': LocalControl(p, use_alpha=True).deltas,
                     'as_': avellaneda_stoikov(p, cfg['benchmark']['as_horizon_s'])},
                    'cpu', day, 1., costs, [11], every=protocol['spec']['evaluation']['path_sampling'])
                as_row = next(r for r in rows if r['strategy'] == 'as_')
                reconcile_as(as_row, original[(original.date == day) & (original.fee == fee) &
                                             original.algorithm.isin(['as', 'as_'])])
                np.savez_compressed(trace_path, **paths)
                verify_replay(rows, trace_path, cfg['model']['K'])
                path.write_text(json.dumps(dict(inputs=inputs, rows=rows, trace_sha256=digest(trace_path)), indent=2))
            verify_replay(rows, trace_path, cfg['model']['K'])
            as_row = next(r for r in rows if r['strategy'] == 'as_')
            reconcile_as(as_row, original[(original.date == day) & (original.fee == fee) &
                                         original.algorithm.isin(['as', 'as_'])])
            for row in rows:
                if row['strategy'] in NEW_METHODS:
                    new.append(dict(row, fee=fee, algorithm=row['strategy'], train_seed=0))
            print(symbol, day, fee, 'AS réconcilié; contrôles évalués', flush=True)
    merged = pd.concat([original, pd.DataFrame(new)], ignore_index=True)
    expected = len(days)*len(cfg['realism']['fees'])*(len(protocol['spec']['algorithms'])*len(protocol['spec']['seeds'])+
                                                   len(EXISTING_CONTROLS)+len(NEW_METHODS)+1)
    if len(merged) != expected or merged.duplicated(['fee', 'date', 'algorithm', 'train_seed']).any():
        raise ValueError('Merged benchmark grid is incomplete or duplicated')
    if not np.isfinite(merged.select_dtypes('number')).all().all():
        raise ValueError('Nonfinite merged benchmark scores')
    if not np.allclose(merged.pnl_net, merged.pnl_gross-merged.maker_cost-merged.taker_cost, rtol=0, atol=1e-7):
        raise ValueError('Merged costs do not reconcile')
    if digest(original_csv) != manifest['base_results_sha256'] or digest(original_protocol) != manifest['base_protocol_sha256']:
        raise ValueError('Original benchmark changed during comparison')
    merged.to_csv(folder/'historique_detail.csv', index=False)
    report(out, merged, methods, repetitions)
    (out/'verification.json').write_text(json.dumps(dict(status='PASS', rows=len(merged), dates=len(days),
        original_benchmark_unchanged=True, as_reconciled=True, costs_and_paths_reconciled=True,
        source_sha256=manifest['source_sha256'],
        results_sha256=digest(folder/'historique_detail.csv')), indent=2))
    for source in source_files:
        target = out/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(out/'rapport.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=ROOT/'experiences/publication/training')
    parser.add_argument('--output', type=Path, default=ROOT/'experiences/publication/comparaison')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--repetitions', type=int, default=50000)
    args = parser.parse_args()
    run(args.base, args.output, args.download, args.repetitions)
