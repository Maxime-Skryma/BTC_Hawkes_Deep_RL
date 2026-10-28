"""Hawkes bivarié achats / ventes, noyau somme d'exponentielles à décroissances fixes.

λ_s(t) = μ_s + Σ_{s', c} a[s, s', c] E_{s', c}(t),  E_{s', c}(t) = Σ_{t_j < t, côté s'} exp(-β_c (t - t_j)).
À β fixés, la log-vraisemblance de chaque côté est concave en (μ_s, a[s]) : fit MLE par L-BFGS-B.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize


@njit(cache=True)
def decayed_counts(ev_t, ev_side, betas, query):
    """E (n_query, 2·C) aux instants query (triés), événements reçus strictement avant.

    Colonnes : (achat, β_0), (vente, β_0), (achat, β_1), (vente, β_1), ...
    """
    C = len(betas)
    out = np.zeros((len(query), 2 * C))
    e = np.zeros(2 * C)
    last, j = 0.0, 0
    for i in range(len(query)):
        q = query[i]
        while j < len(ev_t) and ev_t[j] < q:
            dt = ev_t[j] - last
            for c in range(C):
                f = np.exp(-betas[c] * dt)
                e[2 * c] *= f
                e[2 * c + 1] *= f
            last = ev_t[j]
            for c in range(C):
                e[2 * c + (0 if ev_side[j] == 1 else 1)] += 1.0
            j += 1
        dt = q - last
        for c in range(C):
            f = np.exp(-betas[c] * dt)
            out[i, 2 * c] = e[2 * c] * f
            out[i, 2 * c + 1] = e[2 * c + 1] * f
    return out


def _day_terms(orders, betas, T):
    t, side = orders.t.to_numpy(), orders.side.to_numpy()
    X = np.hstack([np.ones((len(t), 1)), decayed_counts(t, side, betas, t)])
    comp = [T]
    for b in betas:
        g = (1 - np.exp(-b * (T - t))) / b
        comp += [g[side == 1].sum(), g[side == -1].sum()]
    return X, side, np.array(comp)


def fit(days, betas, T=86400.0):
    """days : liste de DataFrames orders. Renvoie theta (2, 1 + 2C) : ligne 0 achats, ligne 1 ventes."""
    betas = np.asarray(betas, float)
    terms = [_day_terms(o, betas, T) for o in days]
    theta = []
    for s in (1, -1):
        X = np.vstack([x[sd == s] for x, sd, _ in terms])
        comp = sum(c for _, _, c in terms)

        def nll(th):
            lam = X @ th
            return -np.log(lam).sum() + th @ comp, -(X / lam[:, None]).sum(0) + comp

        x0 = np.r_[len(X) / comp[0] * 0.5, np.full(X.shape[1] - 1, 0.05)]
        bounds = [(1e-8, None)] + [(0.0, None)] * (X.shape[1] - 1)
        res = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=bounds)
        if not res.success or not np.isfinite(res.fun) or not np.isfinite(res.x).all():
            raise RuntimeError(f"MLE Hawkes non convergée (côté {s}) : {res.message}")
        theta.append(res.x)
    theta = np.array(theta)
    if branching_ratio(theta, betas) >= 1:
        raise ValueError("MLE Hawkes non stationnaire : calibration refusée, ne pas simuler")
    return theta


def intensities(theta, orders, betas, query):
    E = decayed_counts(orders.t.to_numpy(), orders.side.to_numpy(), np.asarray(betas, float), query)
    X = np.hstack([np.ones((len(query), 1)), E])
    return X @ theta[0], X @ theta[1]


def branching_ratio(theta, betas):
    """Rayon spectral de la matrice des normes du noyau ∫φ = Σ_c a / β_c."""
    betas = np.asarray(betas, float)
    a = theta[:, 1:].reshape(2, len(betas), 2)  # (cible, c, source)
    G = (a / betas[None, :, None]).sum(1)
    return float(np.max(np.abs(np.linalg.eigvals(G))))
