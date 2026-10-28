"""Référence exacte du cas dégénéré (Guéant) : Poisson, κ constant, α = 0, état q seul.

V(q) = [−ψ(q) + λ⁺ θ⁺(q) + λ⁻ θ⁻(q)] / (r + λ⁺ + λ⁻),
θ⁺(q) = V(q) + max_δ h(δ) (δ − c + V(q−1) − V(q))  si q > −K (ask touché par un achat), sinon V(q),
δ* = max(1/κ + c − (V(q−1) − V(q)), 0) ; symétrique pour le bid. Contraction de rapport Λ / (r + Λ).
"""
import numpy as np


def poisson_vi(lam_buy, lam_sell, p, tol=1e-10, max_iter=1_000_000):
    K = p.K
    q = np.arange(-K, K + 1)
    V = np.zeros(2 * K + 1)
    psi = p.psi(q.astype(float))
    for _ in range(max_iter):
        da, db, gain_a, gain_b = quotes(V, p)
        new = (-psi + lam_buy * (V + gain_a) + lam_sell * (V + gain_b)) / (p.r + lam_buy + lam_sell)
        if np.max(np.abs(new - V)) < tol:
            V = new
            break
        V = new
    da, db, _, _ = quotes(V, p)
    return V, da, db


def quotes(V, p):
    """δ optimaux et gains h(δ)(δ − c + ΔV) pour chaque q ; inf et 0 aux bornes d'inventaire."""
    n = len(V)
    da, db = np.full(n, np.inf), np.full(n, np.inf)
    ga, gb = np.zeros(n), np.zeros(n)
    Da = V[:-1] - V[1:]          # V(q−1) − V(q), pour q > −K
    Db = V[1:] - V[:-1]          # V(q+1) − V(q), pour q < K
    da[1:] = np.maximum(1 / p.kappa + p.fee - Da, 0)
    db[:-1] = np.maximum(1 / p.kappa + p.fee - Db, 0)
    ga[1:] = p.A * np.exp(-p.kappa * da[1:]) * (da[1:] - p.fee + Da)
    gb[:-1] = p.A * np.exp(-p.kappa * db[:-1]) * (db[:-1] - p.fee + Db)
    return da, db, ga, gb
