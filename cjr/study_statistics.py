"""Declared paired contrasts for validation-selected methods and information ablations."""
import math
import numpy as np
import pandas as pd
from cjr.statistics import block_indices, bootstrap_means, closing_drawdown

METRICS = ('pnl_net', 'max_drawdown', 'mean_abs_q', 'turnover')


def analyse_study(frame, neural, classical, repetitions=50000, seed=20261026):
    if repetitions < 1000:
        raise ValueError('At least 1000 bootstrap replicates required')
    methods = [*neural, *classical]
    if len(methods) != len(set(methods)) or not neural or not classical:
        raise ValueError('Distinct neural and classical methods required')
    # Every learned policy vs every classical policy; same-algorithm information
    # effects vs inventory, plus marginal alpha/Hawkes effects at the full state.
    contrasts = [(a, b, False) for a in neural for b in classical]
    for a in neural:
        algorithm, obs = a.split('__')
        if obs != 'inventory' and f'{algorithm}__inventory' in neural:
            contrasts.append((a, f'{algorithm}__inventory', True))
        if obs == 'full':
            for reduced in ('alpha', 'hawkes'):
                if f'{algorithm}__{reduced}' in neural:
                    contrasts.append((a, f'{algorithm}__{reduced}', True))
    fees = sorted(frame.fee.unique())
    # Include all metrics AND all reported block lengths in the declared family.
    family = len(fees) * len(contrasts) * len(METRICS) * 3
    rng = np.random.default_rng(seed)
    rows, risk = [], []
    maximum_standardized_error = np.zeros(repetitions)
    for fee in fees:
        part = frame[frame.fee == fee]
        dates = sorted(part.date.unique())
        matrices, seed_ids = {}, {}
        for method in methods:
            group = part[part.algorithm == method]
            if group.empty or group.duplicated(['date', 'train_seed']).any():
                raise ValueError(f'Missing/duplicate grid: {fee}/{method}')
            seed_ids[method] = sorted(group.train_seed.unique())
            if method in classical and len(seed_ids[method]) != 1:
                raise ValueError('Classical controls must remain deterministic')
            matrices[method] = {}
            for metric in METRICS:
                mat = group.pivot(index='date', columns='train_seed', values=metric).reindex(dates)
                if not np.isfinite(mat.to_numpy()).all():
                    raise ValueError(f'Incomplete date/seed grid: {fee}/{method}')
                matrices[method][metric] = mat.to_numpy()
            pnl = matrices[method]['pnl_net']
            for j, train_seed in enumerate(seed_ids[method]):
                values = pnl[:, j]
                risk.append(dict(fee=fee, algorithm=method, train_seed=int(train_seed), days=len(dates),
                    pnl_total=float(values.sum()), pnl_daily_mean=float(values.mean()),
                    worst_day=float(values.min()), worst_20pct_days_mean=float(np.sort(values)[:max(1, math.ceil(.2*len(dates)))].mean()),
                    closing_drawdown=closing_drawdown(values),
                    max_intraday_drawdown=float(matrices[method]['max_drawdown'][:, j].max()),
                    mean_abs_inventory=float(matrices[method]['mean_abs_q'][:, j].mean()),
                    turnover=float(matrices[method]['turnover'][:, j].sum())))
        for a, b, paired in contrasts:
            if paired and seed_ids[a] != seed_ids[b]:
                raise ValueError('Ablation contrasts require matching declared seed identities')
        if len(dates) < 5 or any(len(seed_ids[a]) < 3 for a in neural):
            continue
        for length in (1, 2, 3):
            idx = block_indices(len(dates), length, repetitions, rng)
            # Same seed resampling for variants of one algorithm and for every
            # metric, so simultaneous intervals retain their joint dependence.
            seed_draws = {}
            for method in methods:
                key = method.split('__')[0] if method in neural else method
                if key not in seed_draws:
                    nseed = len(seed_ids[method])
                    seed_draws[key] = rng.integers(0,nseed,size=(repetitions,nseed))
            for metric in METRICS:
                samples = {}
                for a in methods:
                    key = a.split('__')[0] if a in neural else a
                    seeds = seed_draws[key]
                    samples[a] = matrices[a][metric][idx[:,:,None], seeds[:,None,:]].mean(axis=(1,2))
                for a, b, paired in contrasts:
                    delta = matrices[a][metric].mean(1) - matrices[b][metric].mean(1)
                    # For ablations, match both date and training seed, preserving
                    # the entire difference trajectory across dates.
                    draws = samples[a]-samples[b]
                    ci = np.quantile(draws, [.025, .975])
                    se = float(draws.std(ddof=1))
                    if se > 1e-12:
                        maximum_standardized_error = np.maximum(maximum_standardized_error, np.abs(draws-delta.mean())/se)
                    loo = (delta.sum()-delta)/(len(delta)-1)
                    rows.append(dict(fee=fee, algorithm=a, reference=b, metric=metric,
                        block_length=length, paired_training_seeds=paired, days=len(dates),
                        mean_daily_difference=float(delta.mean()), median_daily_difference=float(np.median(delta)),
                        days_positive_difference=int((delta > 0).sum()),
                        leave_one_day_out_min=float(loo.min()), leave_one_day_out_max=float(loo.max()),
                        ci95_low=ci[0], ci95_high=ci[1], standard_error=se))
    columns = ['fee','algorithm','reference','metric','block_length','paired_training_seeds','days',
               'mean_daily_difference','median_daily_difference','days_positive_difference',
               'leave_one_day_out_min','leave_one_day_out_max','ci95_low','ci95_high','family_low','family_high']
    critical = float(np.quantile(maximum_standardized_error,.95)) if rows else None
    for row in rows:
        se = row.pop('standard_error')
        row['family_low'] = row['mean_daily_difference']-critical*se
        row['family_high'] = row['mean_daily_difference']+critical*se
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(risk), dict(
        status='exploratory', random_seed=seed, repetitions=repetitions,
        block_lengths=[1,2,3], family_size=family, metrics=list(METRICS),
        contrasts=[dict(algorithm=a, reference=b, paired_training_seeds=p) for a,b,p in contrasts],
        correction='Simultaneous centered standardized bootstrap maximum across fees, contrasts, metrics AND block lengths; fixed marginal bootstrap standard errors (not nested bootstrap-t)',
        simultaneous_critical_value=critical,
        interpretation='Positive PnL difference is better; smaller inventory/drawdown is better. No composite risk winner.',
        limitations='Monthly sampled dates, not consecutive trading days. No annualized Sharpe. No trading between dates. Finite-data bootstrap; nominal coverage only; prior test consultation not corrected.')
