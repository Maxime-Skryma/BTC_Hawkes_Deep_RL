"""Validation tuning, locked test and observation ablations, separate from publication.

python scripts/robust_study.py plan --config configs/robust_study.yaml
python scripts/robust_study.py all --config configs/robust_study.yaml --workers 4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import fcntl
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from cjr import data, replay, model as M, rl_algorithms as RL, discrete_value as DV, vi, simulate
from cjr.baselines import avellaneda_stoikov
from cjr.comparisons.glft import GLFT
from cjr.comparisons.local_control import LocalControl
from cjr.study_statistics import analyse_study


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, content, immutable=False):
    path = Path(path)
    content = json.loads(json.dumps(content, default=str))
    if immutable and path.exists():
        if json.loads(path.read_text()) != content:
            raise ValueError(f'Frozen artifact differs: {path}; use a new output directory')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2))
    temporary.replace(path)


def market_config(spec):
    cfg = yaml.safe_load((ROOT/spec['market_config']).read_text())
    if 'monthly_splits' in spec:
        cfg['splits'] = {name: [d.strftime('%Y-%m-%d') for d in pd.date_range(*bounds, freq='MS')]
                         for name, bounds in spec['monthly_splits'].items()}
    cfg['splits'] = {k: [str(d) for d in v] for k,v in cfg['splits'].items()}
    for name in ('train','validation','test'):
        dates = cfg['splits'][name]
        if not dates or dates != sorted(set(dates)) or any(pd.Timestamp(d).day != 1 for d in dates):
            raise ValueError('Splits require unique sorted first-of-month dates')
    train, val, test = (cfg['splits'][s] for s in ('train','validation','test'))
    if not max(train) < min(val) or not max(val) < min(test):
        raise ValueError('Require chronological disjoint train < validation < test')
    if max(test) >= datetime.now(timezone.utc).strftime('%Y-%m-%d'):
        raise ValueError('Only completed historical dates allowed')
    if spec.get('raw_directory'):
        cfg['data']['raw'] = [str(Path(spec['raw_directory']).expanduser()), *cfg['data']['raw']]
    return cfg


def learned_methods(spec):
    return [f'{a}__{o}' for a in spec['algorithms'] for o in spec['observations']]


def validation_winner(frame, candidates, seeds):
    """Equal dates and all declared seeds; stable tie break by declared order."""
    scores = []
    expected_dates = set(frame.date)
    if not expected_dates or set(frame.candidate)!=set(range(candidates)):
        raise ValueError('Incomplete validation candidate/date/seed grid')
    for candidate in range(candidates):
        g = frame[frame.candidate == candidate]
        if (g.duplicated(['date','train_seed']).any() or set(g.train_seed) != set(seeds)
            or set(g.date) != expected_dates or len(g) != len(expected_dates)*len(seeds)
            or not np.isfinite(g.pnl_net).all()):
            raise ValueError('Incomplete validation candidate/date/seed grid')
        scores.append(float(g.pnl_net.mean()))
    return int(np.argmax(scores)), scores


def classical_policy(method, candidate, p, cfg, cal, grid):
    """Tune decision surrogates only: the replay market, risk and costs stay fixed."""
    choice = grid[candidate]
    if method == 'as':
        return avellaneda_stoikov(p, choice['as_horizon_s'])
    if method == 'constant':
        distance = choice['constant_distance_factor']*(1/p.kappa+p.fee)
        def policy(y):
            return torch.where(RL.active_sides(y,p).bool(), torch.full((len(y),2),distance,device=y.device), torch.inf)
        return policy
    surrogate = copy.deepcopy(p)
    surrogate.gamma *= choice['risk_factor']
    if method == 'glft':
        return GLFT(surrogate, cal['mean_rates']).deltas
    if method.startswith('local_'):
        return LocalControl(surrogate, use_alpha=method=='local_hawkes_alpha').deltas
    if method == 'poisson':
        surrogate.mu = cal['mean_rates']
        surrogate.a = np.zeros_like(p.a).tolist()
        surrogate.eps_plus = surrogate.eps_minus = 0.
        _, ask, bid = vi.poisson_vi(*cal['mean_rates'],surrogate)
        return simulate.table_policy(torch.tensor(ask,dtype=torch.float32), torch.tensor(bid,dtype=torch.float32),p.K)
    raise ValueError(method)


def load_agent(path, p):
    sd = torch.load(path,map_location='cpu',weights_only=False)
    if sd['algorithm_version'] == DV.ALGORITHM_VERSION:
        ag = DV.Agent(p,{'agent':{'hidden':sd['hidden'],'observation':sd.get('observation','full')}},'cpu',sd['alpha_scale'],sd['value_scale'])
    else:
        ag = RL.Agent(p,sd['settings'],sd['algorithm'],sd['alpha_scale'],sd['value_scale'])
    ag.load(sd)
    return ag


def train_job(job):
    out, cfg, cal, settings, method, fee, candidate, seed, lock_hash = job
    path = Path(out)/f'{method}/{fee}/candidate{candidate}_seed{seed}.pt'
    p = M.Params.from_calibration(cal,cfg,fee_bps=cfg['realism']['fees'][fee]['maker']*1e4)
    if path.exists():
        saved = torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('study_lock') != lock_hash or saved.get('training_seed') != seed:
            raise ValueError('Checkpoint provenance differs')
        load_agent(path,p)
        return str(path)
    algorithm, observation = method.split('__')
    settings = dict(settings,observation=observation)
    torch.set_num_threads(settings['threads'])
    from numba import set_num_threads
    set_num_threads(settings['threads'])
    path.parent.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with path.with_suffix('.log').open('w') as log:
        if algorithm == 'value_plus':
            agent_cfg = {'agent':dict(hidden=settings['hidden'],seed=seed,batch=settings['envs'],
                iterations=settings['transitions']//settings['envs'],lr_critic=settings['critic_lr'],
                polyak=settings['tau'],observation=observation)}
            ag, history = DV.train(p,agent_cfg,'cpu',log=lambda row: print(row,file=log,flush=True))
        else:
            ag, history = RL.train(p,settings,algorithm,seed,log=lambda row: print(row,file=log,flush=True))
    state = ag.state_dict()
    state.update(training_seed=seed, study_lock=lock_hash)
    temporary = path.with_suffix('.pt.part')
    torch.save(state,temporary);temporary.replace(path)
    write_json(path.with_suffix('.json'),dict(seconds=time.monotonic()-start,settings=settings,
        history=history,candidate=candidate,training_seed=seed,method=method,study_lock=lock_hash))
    return str(path)


def load_day(cfg, day, cache_root):
    existing = ROOT/'data/cache'
    if (existing/'causal_v4'/cfg['name']/day/'orders.parquet').exists():
        return data.load_day(cfg,day,cache_root=existing)
    return data.load_day(cfg,day,cache_root=cache_root)


def replay_job(job):
    out, stage, day, fee, cfg, cal, spec, chosen, lock_hash = job
    torch.set_num_threads(spec['settings']['threads'])
    from numba import set_num_threads
    set_num_threads(spec['settings']['threads'])
    folder = Path(out)/stage/'shards'
    record = folder/f'{fee}_{day}.json'
    costs = cfg['realism']['fees'][fee]
    p = M.Params.from_calibration(cal,cfg,fee_bps=costs['maker']*1e4)
    agents, policies, checkpoint_hashes = [], {}, {}
    candidates = range(len(spec['candidates']))
    for method in learned_methods(spec):
        for c in (candidates if stage=='validation' else [chosen[fee][method]['candidate']]):
            for seed in spec['tuning_seeds'] if stage=='validation' else spec['final_seeds']:
                checkpoint = Path(out)/('tuning' if stage=='validation' else 'final')/'models'/method/fee/f'candidate{c}_seed{seed}.pt'
                checkpoint_hashes[str(checkpoint.relative_to(out))] = digest(checkpoint)
                state = torch.load(checkpoint,map_location='cpu',weights_only=False)
                if state.get('study_lock')!=lock_hash or state.get('training_seed')!=seed:
                    raise ValueError('Replay checkpoint provenance differs')
                ag = load_agent(checkpoint,p);agents.append(ag)
                policies[f'{method}|{c}|{seed}'] = ag.deltas
    for method in spec['classical']:
        for c in (candidates if stage=='validation' else [chosen[fee][method]['candidate']]):
            policies[f'{method}|{c}|0'] = classical_policy(method,c,p,cfg,cal,spec['candidates'])
    provenance = dict(study_lock=lock_hash, checkpoints=checkpoint_hashes,
        raw={kind:digest(data.raw_file(cfg,kind,pd.Timestamp(day))) for kind in ('trades','book_ticker')})
    if record.exists():
        saved = json.loads(record.read_text())
        if saved['provenance'] != provenance or saved['traces_sha256'] != digest(record.with_suffix('.npz')):
            raise ValueError('Replay inputs/checkpoints/traces changed')
        return saved['rows']
    book, orders = load_day(cfg,day,str(Path(spec['cache_directory']).expanduser()))
    rows, paths = replay.replay_day(book,orders,cal,cfg,policies,'cpu',day,1.,costs,[11],every=spec['path_sampling'])
    for row in rows:
        method,c,seed = row['strategy'].split('|')
        row.update(algorithm=method,candidate=int(c),train_seed=int(seed),fee=fee)
        if (not np.isclose(row['pnl_net'],row['pnl_gross']-row['maker_cost']-row['taker_cost'],rtol=0,atol=1e-7)
            or not np.isclose(paths[row['strategy']+'|pnl'][-1],row['pnl_net'],rtol=0,atol=1e-7)
            or abs(row['q_end'])>p.K or np.max(np.abs(paths[row['strategy']+'|q']))>p.K):
            raise ValueError('Replay costs, traces or inventory do not reconcile')
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(record.with_suffix('.npz'),**paths)
    write_json(record,dict(provenance=provenance,rows=rows,traces_sha256=digest(record.with_suffix('.npz'))))
    return rows


class Study:
    def __init__(self, config, workers=4):
        self.path = Path(config)
        if not self.path.is_absolute(): self.path = ROOT/self.path
        self.spec = yaml.safe_load(self.path.read_text())
        self.cfg = market_config(self.spec)
        self.out = ROOT/self.spec['output']
        if self.out.resolve().is_relative_to((ROOT/'experiences/publication').resolve()):
            raise ValueError('Published artifacts are read-only')
        self.workers = workers
        if not 1 <= workers <= 12: raise ValueError('workers must be 1..12')
        if len(self.spec['candidates']) < 2: raise ValueError('At least two tuning candidates required')
        if set(self.spec['tuning_seeds']) & set(self.spec['final_seeds']):
            raise ValueError('Final seeds must be independent from tuning seeds')
        for seeds in (self.spec['tuning_seeds'], self.spec['final_seeds']):
            if len(seeds)!=len(set(seeds)):raise ValueError('Duplicate training seeds')

    def plan(self):
        s = self.spec
        n, f, c = len(learned_methods(s)), len(self.cfg['realism']['fees']), len(s['candidates'])
        return dict(output=str(self.out),splits={k:dict(days=len(v),first=v[0],last=v[-1]) for k,v in self.cfg['splits'].items()},
            neural=learned_methods(s),classical=s['classical'],candidates=c,
            tuning_trainings=n*f*c*len(s['tuning_seeds']),final_trainings=n*f*len(s['final_seeds']),
            tuning_transitions_per_model=s['tuning_transitions'],final_transitions_per_model=s['settings']['transitions'],
            selection='mean net PnL per validation date, averaged over ALL tuning seeds; fixed candidate-order tie break; same candidate count per method/fee',
            scope='expanded historical exploratory study; previously consulted dates stay exploratory; monthly sampling only')

    def prepare(self):
        sources = [*sorted((ROOT/'cjr').rglob('*.py')),Path(__file__)]
        protocol = dict(spec=self.spec,market_config=self.cfg,plan=self.plan(),
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
            primary_metric='mean daily net PnL in USDT at fixed lot size; drawdown/inventory/turnover separately',
            tuning='same number of declared candidates per method/fee; same seeds, interactions and final checkpoint budget for each neural variant; no seed selection',
            ablation='mask network inputs during training AND evaluation; identical full Hawkes market, reward and execution; no parameter removal from market; partial observations give a reactive policy, not a POMDP-optimality claim',
            value_plus='analytical decision with learned continuation value; all inventory targets per flow state, more counterfactual targets than actor methods; report cost separately',
            test='read only after selection.json frozen and final checkpoint manifest frozen; earlier publication consultation remains a limitation',
            data_policy='first day of month only; do not delete any existing dataset; new raw/cache stored outside small workspace volume')
        write_json(self.out/'protocol.json',protocol,immutable=True)
        write_json(self.out/'plan.json',self.plan(),immutable=True)
        publication = ROOT/'experiences/publication'
        published = {str(p.relative_to(ROOT)):digest(p) for p in sorted(publication.rglob('*')) if p.is_file()}
        write_json(self.out/'publication_inputs.json',published,immutable=True)
        for source in sources:
            target = self.out/'source_snapshot'/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and digest(target)!=digest(source):raise ValueError('Source snapshot changed')
            if not target.exists():target.write_bytes(source.read_bytes())

    def check(self):
        self.prepare()
        return digest(self.out/'protocol.json')

    def progress(self,stage,completed,total,detail=None):
        result = dict(stage=stage,completed=completed,total=total,detail=detail,
                      updated_at=datetime.now(timezone.utc).isoformat())
        write_json(self.out/'progress.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)

    def download(self):
        from cjr.download import dataset_url, fetch_file
        self.check()
        dates = sorted({d for values in self.cfg['splits'].values() for d in values})
        jobs = [(kind,day) for day in dates for kind in ('trades','book_ticker')]
        for i,(kind,day) in enumerate(jobs):
            try: data.raw_file(self.cfg,kind,pd.Timestamp(day))
            except FileNotFoundError:
                path = Path(self.cfg['data']['raw'][0])/kind/f'{self.cfg["data"]["symbol"]}_{day}.csv.gz'
                fetch_file(dataset_url('binance',kind,self.cfg['data']['symbol'],day),path)
            self.progress('download',i+1,len(jobs),f'{kind} {day}')

    def calibrate(self):
        self.check()
        from cjr.calibrate import calibrate
        torch.set_num_threads(self.spec['settings']['threads'])
        from numba import set_num_threads
        set_num_threads(self.spec['settings']['threads'])
        days = self.cfg['splits']['train']
        inputs = {str(data.raw_file(self.cfg,kind,pd.Timestamp(d))):digest(data.raw_file(self.cfg,kind,pd.Timestamp(d)))
                  for d in days for kind in ('trades','book_ticker')}
        metadata = dict(training_dates=days,raw_sha256=inputs,protocol_sha256=digest(self.out/'protocol.json'))
        if (self.out/'calibration.json').exists():
            write_json(self.out/'calibration_inputs.json',metadata,immutable=True)
            return
        self.progress('calibration',0,len(days))
        loaded = [load_day(self.cfg,d,str(Path(self.spec['cache_directory']).expanduser())) for d in days]
        result = calibrate(self.cfg,loaded)
        write_json(self.out/'calibration_inputs.json',metadata,immutable=True)
        write_json(self.out/'calibration.json',result,immutable=True)
        self.progress('calibration',len(days),len(days))

    def pool(self, function, jobs, stage):
        if not jobs:return []
        results = []
        self.progress(stage,0,len(jobs))
        with ProcessPoolExecutor(max_workers=self.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {pool.submit(function,job):i for i,job in enumerate(jobs)}
            for future in as_completed(pending):
                results.append(future.result())
                self.progress(stage,len(results),len(jobs),pending[future])
        return results

    def selection(self):
        selection = json.loads((self.out/'selection.json').read_text())
        validation = self.out/'validation/results.csv'
        if selection['validation_sha256']!=digest(validation) or selection['protocol_sha256']!=self.check():
            raise ValueError('Selection inputs changed')
        if selection['calibration_sha256']!=digest(self.out/'calibration.json'):
            raise ValueError('Calibration changed')
        if selection['validation_checkpoint_manifest_sha256']!=digest(self.out/'validation/checkpoint_manifest.json'):
            raise ValueError('Validation model manifest changed')
        return selection

    def train(self, final=False):
        lock = self.check()
        chosen = self.selection()['chosen'] if final else None
        if final:lock = digest(self.out/'selection.json')
        cal = json.loads((self.out/'calibration.json').read_text())
        cal_hash = digest(self.out/'calibration.json')
        write_json(self.out/'calibration_lock.json',dict(sha256=cal_hash),immutable=True)
        jobs = []
        for method in learned_methods(self.spec):
            for fee in self.cfg['realism']['fees']:
                candidates = [chosen[fee][method]['candidate']] if final else range(len(self.spec['candidates']))
                for c in candidates:
                    settings = copy.deepcopy(self.spec['settings'])
                    factor = self.spec['candidates'][c]['learning_rate_factor']
                    for name in ('actor_lr','critic_lr','sac_alpha_lr'):settings[name] *= factor
                    if not final:settings['transitions'] = self.spec['tuning_transitions']
                    for seed in self.spec['final_seeds'] if final else self.spec['tuning_seeds']:
                        jobs.append((str(self.out/('final' if final else 'tuning')/'models'),self.cfg,cal,settings,method,fee,c,seed,lock))
        self.pool(train_job,jobs,'final_training' if final else 'tuning_training')

    def evaluate(self, final=False):
        lock = self.check()
        chosen = self.selection()['chosen'] if final else None
        stage = 'test' if final else 'validation'
        if final:lock = digest(self.out/'selection.json')
        cal = json.loads((self.out/'calibration.json').read_text())
        write_json(self.out/'calibration_lock.json',dict(sha256=digest(self.out/'calibration.json')),immutable=True)
        # Freeze every evaluated model BEFORE reading a single test observation.
        checkpoint_root = self.out/('final' if final else 'tuning')/'models'
        manifest = {str(p.relative_to(self.out)):digest(p) for p in sorted(checkpoint_root.rglob('*.pt'))}
        expected = len(learned_methods(self.spec))*len(self.cfg['realism']['fees'])*(
            len(self.spec['final_seeds']) if final else len(self.spec['candidates'])*len(self.spec['tuning_seeds']))
        if len(manifest)!=expected:raise ValueError('Incomplete checkpoint grid')
        write_json(self.out/stage/'checkpoint_manifest.json',dict(study_lock=lock,checkpoints=manifest),immutable=True)
        jobs = [(str(self.out),stage,d,fee,self.cfg,cal,self.spec,chosen,lock)
                for d in self.cfg['splits'][stage] for fee in self.cfg['realism']['fees']]
        groups = self.pool(replay_job,jobs,stage+'_replay')
        if manifest != {str(p.relative_to(self.out)):digest(p) for p in sorted(checkpoint_root.rglob('*.pt'))}:
            raise ValueError('Checkpoints changed during replay')
        frame = pd.DataFrame([row for group in groups for row in group]).sort_values(['fee','date','algorithm','candidate','train_seed'])
        count = len(self.cfg['splits'][stage])*len(self.cfg['realism']['fees'])*(
            len(learned_methods(self.spec))*(len(self.spec['final_seeds']) if final else len(self.spec['tuning_seeds']))+len(self.spec['classical']))
        if not final:count *= len(self.spec['candidates'])
        if len(frame)!=count or frame.duplicated(['fee','date','algorithm','candidate','train_seed']).any():
            raise ValueError('Incomplete or duplicate replay grid')
        frame.to_csv(self.out/stage/'results.csv',index=False)

    def select(self):
        self.check()
        path = self.out/'validation/results.csv'
        frame = pd.read_csv(path)
        if set(frame.date)!=set(self.cfg['splits']['validation']):raise ValueError('Selection requires validation dates ONLY')
        chosen, primary = {}, {}
        for fee in self.cfg['realism']['fees']:
            chosen[fee] = {}
            for method in [*learned_methods(self.spec),*self.spec['classical']]:
                group = frame[(frame.fee==fee)&(frame.algorithm==method)]
                if set(group.date)!=set(self.cfg['splits']['validation']):
                    raise ValueError('Incomplete validation dates for method/fee')
                seeds = self.spec['tuning_seeds'] if method in learned_methods(self.spec) else [0]
                winner,scores = validation_winner(group,len(self.spec['candidates']),seeds)
                chosen[fee][method] = dict(candidate=winner,validation_mean_daily_pnl=scores,
                                         parameters=self.spec['candidates'][winner])
            def score(method):
                entry=chosen[fee][method]
                return entry['validation_mean_daily_pnl'][entry['candidate']]
            primary[fee]=dict(neural=max([a+'__full' for a in self.spec['algorithms']],key=score),
                              classical=max(self.spec['classical'],key=score))
        write_json(self.out/'selection.json',dict(chosen=chosen,primary=primary,validation_sha256=digest(path),
            protocol_sha256=digest(self.out/'protocol.json'),calibration_sha256=digest(self.out/'calibration.json'),
            validation_checkpoint_manifest_sha256=digest(self.out/'validation/checkpoint_manifest.json'),
            selection_rule=self.plan()['selection']),immutable=True)

    def report(self):
        selection=self.selection()
        path = self.out/'test/results.csv'
        frame = pd.read_csv(path)
        if set(frame.date)!=set(self.cfg['splits']['test']):raise ValueError('Unexpected test dates')
        intervals,risk,metadata = analyse_study(frame,learned_methods(self.spec),self.spec['classical'],self.spec['bootstrap_repetitions'])
        metadata.update(test_sha256=digest(path),selection_sha256=digest(self.out/'selection.json'),
                        protocol_sha256=digest(self.out/'protocol.json'))
        folder = self.out/'analysis';folder.mkdir(exist_ok=True)
        intervals.to_csv(folder/'paired_intervals.csv',index=False)
        selected=[]
        for fee,methods in selection['primary'].items():
            selected.append(intervals[(intervals.fee==fee)&(intervals.algorithm==methods['neural'])&
                (intervals.reference==methods['classical'])])
        primary=pd.concat(selected,ignore_index=True)
        primary.to_csv(folder/'primary_comparison.csv',index=False)
        risk.to_csv(folder/'risk_by_seed.csv',index=False)
        frame.groupby(['fee','date','algorithm']).pnl_net.mean().reset_index().to_csv(folder/'daily_pnl.csv',index=False)
        summary = risk.groupby(['fee','algorithm']).agg(pnl_total_mean=('pnl_total','mean'),
            pnl_seed_min=('pnl_total','min'),pnl_seed_max=('pnl_total','max'),
            worst_seed_day=('worst_day','min'),max_intraday_drawdown=('max_intraday_drawdown','max'),
            mean_abs_inventory=('mean_abs_inventory','mean')).reset_index()
        summary.to_csv(folder/'summary.csv',index=False)
        write_json(folder/'method.json',metadata)
        costs = []
        for stage in ('tuning','final'):
            for p in (self.out/stage/'models').rglob('*.json'):
                row = json.loads(p.read_text())
                costs.append(dict(stage=stage,method=row['method'],candidate=row['candidate'],seed=row['training_seed'],
                    fee=p.parent.name,seconds=row['seconds'],transitions=row['settings']['transitions']))
        pd.DataFrame(costs).to_csv(folder/'training_costs.csv',index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures=self.out/'figures';figures.mkdir(exist_ok=True)
        fig,axes=plt.subplots(len(self.cfg['realism']['fees']),1,figsize=(12,4*len(self.cfg['realism']['fees'])),squeeze=False)
        for ax,fee in zip(axes.flat,self.cfg['realism']['fees']):
            g=frame[frame.fee==fee].groupby(['date','algorithm']).pnl_net.mean().unstack()
            for method in g.columns:
                if method in self.spec['classical'] or method.endswith('__full'):
                    ax.plot(np.arange(len(g)),g[method].cumsum(),label=method)
            ax.set_title(fee);ax.set_ylabel('Cumulative net P&L (USDT)');ax.set_xlabel('Observed day index (no trading between dates)')
            ax.legend(ncol=2,fontsize=7);ax.axhline(0,color='grey',lw=.5)
        fig.tight_layout();fig.savefig(figures/'pnl.png',dpi=140);plt.close(fig)
        page = ('<!doctype html><html lang="fr"><meta charset="utf-8"><title>Deep RL vs contrôles classiques</title>'
            '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}.scroll{overflow:auto}</style>'
            '<h1>Comparaison après réglage sur validation et ablations</h1>'
            '<p>Étude historique exploratoire : certaines dates ont déjà été consultées dans la publication précédente. '
            'Aucune preuve de rentabilité déployable. Les journées sont échantillonnées mensuellement ; aucun trading entre elles.</p>'
            '<p>Le réglage utilise uniquement la validation, toutes les seeds déclarées et le même nombre de candidats par méthode. '
            'Les modèles finaux utilisent de nouvelles seeds et un budget fixe. Les quatre observations sont inventaire seul, alpha seul avec inventaire, '
            'Hawkes avec inventaire, et état complet. Le marché reste inchangé.</p>'
            '<p><a href="selection.json">Sélection figée</a> · <a href="protocol.json">Protocole</a> · '
            '<a href="analysis/summary.csv">Résumé CSV</a> · <a href="analysis/paired_intervals.csv">Tous les intervalles appariés</a> · '
            '<a href="analysis/risk_by_seed.csv">Risque par seed</a> · <a href="analysis/training_costs.csv">Coût des entraînements</a></p>'
            '<img src="figures/pnl.png" style="max-width:100%" alt="P&L cumulé sur les seules dates observées">'
            '<h2>Comparaison principale (deux méthodes choisies sur validation)</h2><p><a href="analysis/primary_comparison.csv">Intervalles de la comparaison principale</a></p><div class="scroll">'+primary[(primary.metric=='pnl_net')&(primary.block_length==2)].to_html(index=False,float_format=lambda x:f'{x:.2f}')+'</div>'
            '<h2>P&amp;L total sur les dates test, dispersion et risque</h2><div class="scroll">'+summary.to_html(index=False,float_format=lambda x:f'{x:.2f}')+'</div>'
            '<h2>Méthode statistique</h2><pre>'+escape(json.dumps(metadata,ensure_ascii=False,indent=2))+'</pre>'
            '<h2>Différences de P&amp;L : blocs de 2 observations</h2><p>Consulter le CSV pour les blocs 1 et 3, '
            'ainsi que les différences de drawdown, inventaire et turnover. Les intervalles corrigés sont nominaux ; '
            'aucun gagnant automatique sur une seule longueur de bloc.</p><div class="scroll">'+
            intervals[(intervals.metric=='pnl_net')&(intervals.block_length==2)].to_html(index=False,float_format=lambda x:f'{x:.2f}')+'</div></html>')
        (self.out/'rapport.html').write_text(page)
        for filename,expected in json.loads((self.out/'publication_inputs.json').read_text()).items():
            if digest(ROOT/filename)!=expected:raise ValueError('Published artifact changed')
        write_json(self.out/'verification.json',dict(status='PASS',test_rows=len(frame),
            selected_on_validation_only=True,published_results_unchanged=True,analysis_sha256=digest(folder/'method.json')))
        self.progress('complete',1,1,str(self.out/'rapport.html'))

    def all(self):
        self.prepare();self.download();self.calibrate();self.train();self.evaluate();self.select()
        self.train(final=True);self.evaluate(final=True);self.report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['plan','status','prepare','download','calibrate','tune','validate','select','train-final','test','report','all'])
    parser.add_argument('--config',default='configs/robust_study.yaml')
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    study = Study(args.config,args.workers)
    if args.stage=='plan':print(json.dumps(study.plan(),ensure_ascii=False,indent=2));return
    if args.stage=='status':
        p=study.out/'progress.json'
        print(p.read_text() if p.exists() else 'Study not started')
        return
    stages = {'tune':lambda:study.train(), 'validate':lambda:study.evaluate(),
              'train-final':lambda:study.train(final=True), 'test':lambda:study.evaluate(final=True)}
    study.out.mkdir(parents=True,exist_ok=True)
    with (study.out/'runner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another runner is already active for this study')
        write_json(study.out/'runner.json',dict(pid=os.getpid(),config=str(study.path),stage=args.stage,
            started_at=datetime.now(timezone.utc).isoformat()))
        try:
            if args.stage in stages:stages[args.stage]()
            else:getattr(study,args.stage)()
        except Exception as error:
            study.progress('failed',0,1,f'{type(error).__name__}: {error}')
            raise


if __name__=='__main__':main()
