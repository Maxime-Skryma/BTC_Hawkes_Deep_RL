"""Calibration train : Hawkes MLE, alpha par moments conditionnels incluant les arrivées
futures attendues, sigma sur résidus après drift, survie exponentielle contrainte A=1.
Prix et distances en bps d'un mid journalier fixe connu au début de la fenêtre cotée.
"""
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import nnls

from cjr import hawkes


def mid_at(book, times):
    """Mid reçu strictement avant chaque instant (bps relatifs non appliqués : prix bruts)."""
    i = np.searchsorted(book.t.to_numpy(), times, side="left") - 1
    mid = ((book.bid + book.ask) / 2).to_numpy()
    return np.where(i >= 0, mid[np.clip(i, 0, None)], np.nan)


def alpha_moment_map(theta, betas, zeta, H):
    """E[∫ Z_ζ^s(t+u)du | F_t], affine en E_Hawkes(t), Z_ζ(t).

    Inclut les ordres FUTURS en espérance conditionnelle, jamais leurs réalisations.
    ODE : m_E' = (-B + J a)m_E + J mu, m_Z' = a m_E + mu - ζ m_Z.
    """
    betas = np.asarray(betas, float)
    d = 2 * len(betas)
    mu, a = theta[:, 0], theta[:, 1:]
    J = np.tile(np.eye(2), (len(betas), 1))
    G = np.zeros((d + 5, d + 5))
    G[:d, :d] = J @ a - np.diag(np.repeat(betas, 2))
    G[:d, -1] = J @ mu
    G[d:d+2, :d] = a
    G[d:d+2, d:d+2] = -zeta * np.eye(2)
    G[d:d+2, -1] = mu
    G[d+2:d+4, d:d+2] = np.eye(2)
    return expm(G * H)[d+2:d+4]


def reference_mid(book):
    """Numéraire journalier connu au début de la fenêtre cotée (600 s)."""
    ref = mid_at(book, np.array([600.]))[0]
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError("Mid de référence indisponible au début de la fenêtre")
    return ref


def alpha_design(book, orders, zeta, H, grid, theta, betas):
    t, side = orders.t.to_numpy(), orders.side.to_numpy()
    Ez = hawkes.decayed_counts(t, side, np.array([zeta]), grid)
    E = hawkes.decayed_counts(t, side, np.asarray(betas, float), grid)
    moment_map = alpha_moment_map(theta, betas, zeta, H)
    states = np.column_stack([E, Ez, np.zeros((len(grid), 2)), np.ones(len(grid))])
    integrated_Z = states @ moment_map.T
    m0, m1 = mid_at(book, grid), mid_at(book, grid + H)
    y = (m1 - m0) / reference_mid(book) * 1e4
    return np.column_stack([integrated_Z[:, 0], -integrated_Z[:, 1]]), y


def realized_alpha_integral(orders, alpha, grid):
    """Primitive du drift réalisé, en bps, calcul exact des exponentielles observées.

    ∫_0^t Z_s(u)du = (N_s(t)-Z_s(t))/ζ, sans approximation de grille.
    """
    t, side = orders.t.to_numpy(), orders.side.to_numpy()
    zeta = alpha["zeta"]
    E = hawkes.decayed_counts(t, side, np.array([zeta]), grid)
    N = np.column_stack([np.searchsorted(t[side == s], grid, side="left") for s in (1, -1)])
    integrals = (N - E) / zeta
    return integrals @ np.array([alpha["eps_plus"], -alpha["eps_minus"]])


def calibrate(cfg, days):
    """days : liste de (book, orders) d'entraînement."""
    m = cfg["model"]
    betas = np.asarray(m["betas"], float)
    theta = hawkes.fit([o for _, o in days], betas)

    H = float(m["alpha_horizon_s"])
    best = None
    for zeta in m["zetas"]:
        X, Y = [], []
        for book, orders in days:
            grid = np.arange(600.0, 86400.0 - H - 1, 1.0)
            x, y = alpha_design(book, orders, zeta, H, grid, theta, betas)
            ok = np.isfinite(y)
            X.append(x[ok])
            Y.append(y[ok])
        X, Y = np.vstack(X), np.concatenate(Y)
        eps = nnls(X, Y)[0]  # amplitudes non négatives : achat +eps, vente -eps
        r2 = 1 - ((Y - X @ eps) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
        if best is None or r2 > best["r2"]:
            best = dict(zeta=float(zeta), eps_plus=float(eps[0]), eps_minus=float(eps[1]), r2=float(r2))

    Hs = float(m["sigma_horizon_s"])
    inc = []
    for book, orders in days:
        g = np.arange(600.0, 86400.0 - Hs, Hs)
        mm = mid_at(book, g)
        drift = np.diff(realized_alpha_integral(orders, best, g))
        inc.append(np.diff(mm) / reference_mid(book) * 1e4 - drift)
    inc = np.concatenate(inc)
    sigma2 = float(np.nanvar(inc) / Hs)

    D = np.concatenate([(o.side * (o.price_ext - o.mid) / reference_mid(book) * 1e4)
                        .loc[o.t >= 600.].dropna().to_numpy() for book, o in days])
    grid = np.unique(np.quantile(D, np.linspace(0.5, 0.99, 50)))
    surv = np.array([(D >= d).mean() for d in grid])
    if m.get("execution") == "cjr":
        # Le modèle impose log h(δ)=-κδ : ajuster avec intercept nul, pas écraser A après le fit.
        valid = grid > 0
        kappa = float(-np.dot(grid[valid], np.log(surv[valid])) / np.dot(grid[valid], grid[valid]))
        A = 1.0
    else:
        slope, intercept = np.polyfit(grid, np.log(surv), 1)
        kappa, A = float(-slope), float(min(np.exp(intercept), 1.0))
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("Calibration de kappa non identifiable")

    lam_bar = stationary_rates(theta, betas)
    return dict(
        calibration_version="cjr-v2-conditional-moments-fixed-numeraire",
        mu=theta[:, 0].tolist(), a=theta[:, 1:].reshape(2, len(betas), 2).tolist(), betas=betas.tolist(),
        branching_ratio=hawkes.branching_ratio(theta, betas), mean_rates=lam_bar.tolist(),
        alpha=best, sigma2=sigma2, A=A, kappa=kappa,
        depth_quantiles={f"q{int(q * 100)}": float(np.quantile(D, q)) for q in (0.1, 0.5, 0.9, 0.99)},
        fill_fit=dict(delta=grid.tolist(), survival=surv.tolist()),
        orders_per_day=float(np.mean([len(o) for _, o in days])))


def stationary_rates(theta, betas):
    a = theta[:, 1:].reshape(2, len(betas), 2)          # (cible, c, source)
    G = (a / np.asarray(betas)[None, :, None]).sum(1)   # normes du noyau
    return np.linalg.solve(np.eye(2) - G, theta[:, 0])
