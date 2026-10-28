"""Rejeu sur données réelles, avec la règle d'exécution du modèle (pas de file).

Décisions sur une grille de 0,1 s (horloge de réception) ; l'état (α, E) n'utilise que les ordres reçus
strictement avant la décision, q est notre inventaire. Cotes : ask = mid (1 + δ_a/10⁴) arrondi au tick
supérieur, bid = mid (1 − δ_b/10⁴) arrondi au tick inférieur, actives après la latence jusqu'à la
décision suivante (+ latence). Post-only : pas de cote qui croiserait le meilleur prix opposé.
Exécution, comme dans le modèle : un market order dont la portée atteint notre prix (achat : prix
extrême ≥ ask) nous exécute avec probabilité ρ (part des exécutions qui nous revient, h = ρ A e^{−κδ}).
Tirages uniformes par ordre, communs à toutes les stratégies, sur plusieurs seeds.
Un fill par côté et par fenêtre. Frais maker c sur chaque exécution, liquidation taker en fin de journée.
"""
import numpy as np
import torch
from numba import njit

from cjr import hawkes
from cjr.calibrate import reference_mid


def states(orders, cal, grid):
    """(α, E) aux instants de grille, à partir des ordres reçus strictement avant."""
    t, s = orders.t.to_numpy(), orders.side.to_numpy()
    E = hawkes.decayed_counts(t, s, np.asarray(cal["betas"], float), grid)
    al = cal["alpha"]
    Ez = hawkes.decayed_counts(t, s, np.array([al["zeta"]]), grid)
    alpha = al["eps_plus"] * Ez[:, 0] - al["eps_minus"] * Ez[:, 1]
    return alpha, E


@torch.no_grad()
def delta_tables(policy, alpha, E, K, device, chunk=20_000):
    """δ (n, 2K+1) pour chaque instant et chaque inventaire, en float32."""
    owner=getattr(policy,'__self__',None)
    if owner is not None and hasattr(owner,'grid_tables'):
        return owner.grid_tables(alpha,E,K,device,chunk)
    n = len(alpha)
    qs = torch.arange(-K, K + 1, device=device, dtype=torch.float32)
    da = np.empty((n, 2 * K + 1), np.float32)
    db = np.empty((n, 2 * K + 1), np.float32)
    for i in range(0, n, chunk):
        a = torch.tensor(alpha[i:i + chunk], device=device, dtype=torch.float32)
        e = torch.tensor(E[i:i + chunk], device=device, dtype=torch.float32)
        m = len(a)
        y = torch.cat([qs.repeat(m)[:, None], a.repeat_interleave(2 * K + 1)[:, None],
                       e.repeat_interleave(2 * K + 1, 0)], 1)
        d = policy(y).view(m, 2 * K + 1, 2)
        da[i:i + m], db[i:i + m] = d[..., 0].cpu().numpy(), d[..., 1].cpu().numpy()
    return da, db


@njit(cache=True)
def run(g, mid_g, da, db, bt, bbid, bask, ot, oside, oext, latency, K, tick, u, rho, price_reference=0.0):
    n = len(g)
    q, cash, notional = 0, 0.0, 0.0
    nfill, abs_q = 0, 0.0
    mtm = np.empty(n)
    qpath = np.empty(n, np.int64)
    npath = np.empty(n)
    # Historique des fills : une décision ne peut utiliser que l'inventaire connu
    # à g[k], même si l'ancien ordre reste actif pendant la latence.
    fill_times = np.empty(2 * n)
    fill_inventory = np.empty(2 * n, np.int64)
    j, jb = 0, 0
    for k in range(n):
        start = g[k] + latency
        end = (g[k + 1] if k + 1 < n else g[k] + 1e-9) + latency
        while jb < len(bt) and bt[jb] < start:
            jb += 1
        while j < len(ot) and ot[j] < start:
            j += 1
        if jb > 0 and not np.isnan(mid_g[k]):
            bb, ba = bbid[jb - 1], bask[jb - 1]
            decision_fill = np.searchsorted(fill_times[:nfill], g[k], side="left") - 1
            decision_q = fill_inventory[decision_fill] if decision_fill >= 0 else 0
            a_d, b_d = da[k, decision_q + K], db[k, decision_q + K]
            a_on = np.isfinite(a_d) and decision_q > -K and q > -K
            b_on = np.isfinite(b_d) and decision_q < K and q < K
            ref = price_reference if price_reference > 0 else mid_g[k]
            a_px = np.ceil((mid_g[k] + ref * a_d / 1e4) / tick - 1e-9) * tick if a_on else 0.0
            b_px = np.floor((mid_g[k] - ref * b_d / 1e4) / tick + 1e-9) * tick if b_on else 0.0
            a_on = a_on and a_px > bb + 1e-12
            b_on = b_on and b_px < ba - 1e-12
            jj = j
            while jj < len(ot) and ot[jj] < end and (a_on or b_on):
                x = oext[jj]
                if u[jj] >= rho:
                    jj += 1
                    continue
                if oside[jj] == 1 and a_on and x >= a_px - 1e-12:
                    q -= 1
                    cash += a_px
                    notional += a_px
                    fill_times[nfill] = ot[jj]
                    fill_inventory[nfill] = q
                    nfill += 1
                    a_on = False
                elif oside[jj] == -1 and b_on and x <= b_px + 1e-12:
                    q += 1
                    cash -= b_px
                    notional += b_px
                    fill_times[nfill] = ot[jj]
                    fill_inventory[nfill] = q
                    nfill += 1
                    b_on = False
                jj += 1
        # Les fills appartiennent à [start,end[ : marquer au mid disponible à end,
        # pas à celui de la décision antérieure (qui antidaterait la trajectoire).
        jm = jb
        while jm < len(bt) and bt[jm] < end:
            jm += 1
        mark = (bbid[jm-1] + bask[jm-1]) / 2 if jm > 0 else mid_g[k]
        mtm[k] = cash + q * (mark if not np.isnan(mark) else 0.0)
        qpath[k] = q
        npath[k] = notional
        abs_q += abs(q)
    return q, cash, notional, nfill, mtm, abs_q / n, qpath, npath


def replay_day(book, orders, cal, cfg, policies, device, day, rho, fee, seeds, every=100):
    """Lignes de résultats (une par stratégie et par seed) et trajectoires de la première seed
    (heure, P&L net marqué au mid en $, inventaire), une décision sur `every`."""
    ev = cfg["evaluation"]
    K, tick = int(cfg["model"]["K"]), float(cfg["data"]["tick"])
    grid = np.arange(ev["warmup_s"], ev["stop_s"], ev["decision_interval_s"])
    bt = book.t.to_numpy()
    mid = ((book.bid + book.ask) / 2).to_numpy()
    i = np.searchsorted(bt, grid, side="left") - 1
    mid_g = np.where(i >= 0, mid[np.clip(i, 0, None)], np.nan)
    alpha, E = states(orders, cal, grid)
    ref = reference_mid(book)
    units = cfg["model"]["lot_usd"] / ref   # unités de l'actif par lot
    last = np.searchsorted(bt, ev["stop_s"], side="left") - 1
    sample = np.unique(np.r_[np.arange(0, len(grid), every), len(grid)-1])
    ends = np.r_[grid[1:] + ev["latency_s"], grid[-1] + ev["latency_s"] + 1e-9]
    rows, paths = [], {"hour": ends[sample] / 3600}
    for name, pol in policies.items():
        da, db = delta_tables(pol, alpha, E, K, device)
        for seed in seeds:
            u = np.random.default_rng([seed, int(day.replace("-", ""))]).random(len(orders))
            q, cash, notional, nfill, mtm, mean_q, qpath, npath = run(
                grid, mid_g, da, db, bt, book.bid.to_numpy(), book.ask.to_numpy(), orders.t.to_numpy(),
                orders.side.to_numpy(), orders.price_ext.to_numpy(), ev["latency_s"], K, tick, u, rho, ref)
            liq = book.bid.iloc[last] if q > 0 else book.ask.iloc[last]
            gross = units * (cash + q * liq)
            net = gross - fee["maker"] * units * notional - fee["taker"] * units * abs(q) * liq
            net_path = units * (mtm - fee["maker"] * npath)
            net_path[-1] = net  # dernier point = liquidation effective, réconcilié au CSV
            if seed == seeds[0]:
                paths[f"{name}|pnl"] = net_path[sample].copy()
                paths[f"{name}|q"] = qpath[sample].copy()
                paths[f"{name}|q"][-1] = 0  # liquidation finale
            rows.append(dict(date=day, strategy=name, seed=seed, pnl_gross=gross, pnl_net=net, n_fills=nfill,
                             q_end=q, mean_abs_q=mean_q, turnover=units * notional,
                             maker_cost=fee["maker"] * units * notional,
                             taker_cost=fee["taker"] * units * abs(q) * liq,
                             liquidation_notional=units * abs(q) * liq,
                             max_drawdown=float(np.max(np.maximum.accumulate(np.r_[0., net_path])[1:] - net_path))))
    return rows, paths
