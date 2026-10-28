"""Évaluation dans le simulateur calibré : PnL net du frais maker (spread − c + inventaire sous
dS = α dt + σ dW) et critère.

Une stratégie est une fonction y⁻ → (δ_ask, δ_bid) en bps (inf : pas de cote), évaluée juste avant
chaque market order (les cotes sont continûment ajustées à l'état, comme dans le modèle).
"""
import numpy as np
import torch

from cjr import model as M


@torch.no_grad()
def evaluate(policy, p, seconds, paths, device, seed=0):
    T = p.tensors(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    y = torch.zeros(paths, M.dim(p), device=device)
    for _ in range(300):  # Préchauffage événementiel approché, q = 0
        tau, side = M.next_event(y, p, T, gen)
        y = M.jump(M.decay(y, tau, p, T), side, p)
    t = torch.zeros(paths, device=device)
    pnl, crit, spread = (torch.zeros(paths, device=device) for _ in range(3))
    fills = torch.zeros(paths, device=device)
    abs_q = torch.zeros(paths, device=device)
    sigma = float(np.sqrt(p.sigma2))
    while bool((t < seconds).any()):
        live = t < seconds
        tau, side = M.next_event(y, p, T, gen)
        dt = torch.minimum(tau, (seconds - t).clamp_min(0))
        q, al, _ = M.unpack(y, p)
        dS = al * (-torch.expm1(-p.zeta * dt)) / p.zeta \
            + sigma * dt.sqrt() * torch.randn(paths, device=device, generator=gen)
        disc = torch.exp(-p.r * t)
        crit += live * disc * M.running_reward(y, dt, p)
        pnl += live * q * dS
        abs_q += live * q.abs() * dt
        yd = M.decay(y, tau, p, T)
        d = policy(yd)
        d_side = d.gather(1, side[:, None])[:, 0]
        allowed = torch.where(side == 0, q > -p.K, q < p.K)
        ok = torch.isfinite(d_side) & allowed
        if bool((torch.isfinite(d_side) & (d_side < 0)).any()):
            raise ValueError("La politique doit fournir des distances positives ou inf")
        h = torch.where(ok, M.fill_prob(torch.where(ok, d_side, 0.0), p), 0.0)
        hit = live & (tau <= seconds - t) & (torch.rand(paths, device=device, generator=gen) < h)
        gain = torch.where(hit, d_side - p.fee, 0.0)
        pnl += gain
        spread += gain
        crit += disc * torch.exp(-p.r * tau) * gain
        fills += hit
        y = M.jump(yd, side, p)
        y[:, 0] += torch.where(hit, torch.where(side == 0, -1.0, 1.0), 0.0)
        t = t + tau
    out = dict(pnl=pnl, criterion=crit, spread=spread, fills=fills, mean_abs_q=abs_q / seconds)
    return {k: (float(v.mean()), float(v.std() / np.sqrt(paths))) for k, v in out.items()}


def table_policy(da, db, K):
    """Stratégie ne dépendant que de q (tables indexées par q + K)."""
    def f(y):
        i = (y[:, 0] + K).long()
        return torch.stack([da.to(y.device)[i], db.to(y.device)[i]], 1)
    return f
