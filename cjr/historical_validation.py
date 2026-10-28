"""Validation historique par rejeu causal, journées pré-calculées une fois.

Même calcul que replay.replay_day (grille, états α/E, exécution CJR, frais maker, liquidation
taker en fin de fenêtre), découpé en préparation (indépendante de la politique) et évaluation.
replay.py n'est pas modifié : son empreinte est figée dans les protocoles existants.
Usage prévu : sélection de checkpoints sur les journées de validation, jamais sur le test.
"""
import numpy as np

from cjr import data, replay
from cjr.calibrate import reference_mid


def prepare_day(cfg, cal, day, cache_root, seed):
    ev = cfg['evaluation']
    book, orders = data.load_day(cfg, day, cache_root=cache_root)
    grid = np.arange(ev['warmup_s'], ev['stop_s'], ev['decision_interval_s'])
    bt = book.t.to_numpy()
    mid = ((book.bid + book.ask) / 2).to_numpy()
    i = np.searchsorted(bt, grid, side='left') - 1
    alpha, E = replay.states(orders, cal, grid)
    ref = reference_mid(book)
    return dict(day=day, grid=grid, mid_g=np.where(i >= 0, mid[np.clip(i, 0, None)], np.nan), alpha=alpha, E=E,
                bt=bt, bid=book.bid.to_numpy(), ask=book.ask.to_numpy(), ot=orders.t.to_numpy(),
                oside=orders.side.to_numpy(), oext=orders.price_ext.to_numpy(), ref=ref,
                units=cfg['model']['lot_usd'] / ref, last=np.searchsorted(bt, ev['stop_s'], side='left') - 1,
                u=np.random.default_rng([seed, int(day.replace('-', ''))]).random(len(orders)), seed=seed)


def evaluate_day(policy, d, cfg, fee, device, rho=1.):
    K = int(cfg['model']['K'])
    da, db = replay.delta_tables(policy, d['alpha'], d['E'], K, device)
    return replay_tables(da, db, d, cfg, fee, rho)


def evaluate_day_rules(policy, d, cfg, fee, device, through_ticks, rho=1.):
    """Mêmes tables δ, rejouées pour chaque règle d'exécution : m=0 est la règle de replay.run
    (contact du prix suffit) ; m>0 exige que le prix extrême traverse notre niveau de m ticks."""
    K = int(cfg['model']['K'])
    da, db = replay.delta_tables(policy, d['alpha'], d['E'], K, device)
    return {m: replay_tables(da, db, d, cfg, fee, rho, m) for m in through_ticks}


def replay_tables(da, db, d, cfg, fee, rho=1., through_ticks=0):
    ev = cfg['evaluation']
    K, tick = int(cfg['model']['K']), float(cfg['data']['tick'])
    oext = d['oext'] - d['oside'] * through_ticks * tick if through_ticks else d['oext']
    q, cash, notional, nfill, mtm, mean_q, _, _ = replay.run(
        d['grid'], d['mid_g'], da, db, d['bt'], d['bid'], d['ask'], d['ot'], d['oside'], oext,
        ev['latency_s'], K, tick, d['u'], rho, d['ref'])
    liq = d['bid'][d['last']] if q > 0 else d['ask'][d['last']]
    units = d['units']
    return dict(date=d['day'], seed=d['seed'], pnl_gross=units * (cash + q * liq),
                pnl_net=units * (cash + q * liq) - fee['maker'] * units * notional - fee['taker'] * units * abs(q) * liq,
                n_fills=nfill, q_end=q, mean_abs_q=mean_q, turnover=units * notional,
                maker_cost=fee['maker'] * units * notional, taker_cost=fee['taker'] * units * abs(q) * liq)


class HistoricalValidator:
    """Score = P&L net moyen par journée (USDT), liquidation incluse, sur des journées fixes."""

    def __init__(self, cfg, cal, days, fee, cache_root, device='cpu', seed=11, forbidden_days=(), extra_through_ticks=()):
        leaked = sorted(set(days) & set(forbidden_days))
        if leaked:
            raise ValueError(f'Journées interdites pour la sélection : {leaked}')
        self.cfg, self.fee, self.device = cfg, fee, device
        self.extra = [int(m) for m in extra_through_ticks if int(m) != 0]
        self.days = [prepare_day(cfg, cal, day, cache_root, seed) for day in days]

    def __call__(self, policy):
        per_day = [evaluate_day_rules(policy, d, self.cfg, self.fee, self.device, [0] + self.extra) for d in self.days]
        score = summarize([r[0] for r in per_day])
        # Règles plus strictes : mêmes clés, sous 'rules' ; le score principal reste la règle m=0.
        if self.extra:
            score['rules'] = {str(m): {k: v for k, v in summarize([r[m] for r in per_day]).items() if k != 'days'}
                              | {'daily_pnl': [r[m]['pnl_net'] for r in per_day]} for m in self.extra}
        return score


def summarize(rows):
    pnl = np.array([r['pnl_net'] for r in rows])
    return dict(mean_daily_pnl=float(pnl.mean()), median_daily_pnl=float(np.median(pnl)),
                positive_days=int((pnl > 0).sum()), mean_fills=float(np.mean([r['n_fills'] for r in rows])),
                mean_abs_inventory=float(np.mean([r['mean_abs_q'] for r in rows])), days=rows)
