"""Modèle événementiel de Cartea–Jaimungal–Ricci en torch (lots parallèles).

État y = (q, α, E) avec E[c, s'] = Σ exp(−β_c (t − t_j)) sur les market orders passés de source s'
(0 achat, 1 vente). Intensités λ_s = μ_s + Σ_{c,s'} a[s, c, s'] E[c, s'].
Entre deux ordres : E_c décroît au taux β_c, α au taux ζ ; dS = α dt + σ dW.
À un ordre de côté s : E[:, s] += 1, α += ε⁺ (achat) ou −ε⁻ (vente) ; notre cote du côté touché
(ask pour un achat) à δ est exécutée avec probabilité h(δ) = ρ A e^{−κδ}, q ∓= 1, gain δ − c.
ρ : part des exécutions qui nous revient (priorité dans la file) ; c : frais maker (bps).
Critère : E[∫ e^{−rt} (q α − ψ(q)) dt + Σ e^{−rτ_i} (δ_i − c)], ψ(q) = ½ γ σ² q². Unité : 1 bp d'un lot.
"""
from dataclasses import dataclass, asdict
import hashlib
import json

import numpy as np
import torch


@dataclass
class Params:
    mu: list
    a: list            # (cible, c, source)
    betas: list
    eps_plus: float
    eps_minus: float
    zeta: float
    A: float
    kappa: float
    sigma2: float
    gamma: float
    r: float
    K: int
    fee: float = 0.0   # frais maker c (bps), retranché de chaque exécution

    def __post_init__(self):
        mu, a, b = np.asarray(self.mu), np.asarray(self.a), np.asarray(self.betas)
        if mu.shape != (2,) or b.ndim != 1 or not len(b) or a.shape != (2, len(b), 2):
            raise ValueError("Dimensions Hawkes invalides : mu (2), a (2,C,2), betas (C)")
        numbers = np.r_[mu.ravel(), a.ravel(), b, self.eps_plus, self.eps_minus,
                        self.zeta, self.A, self.kappa, self.sigma2, self.gamma, self.r, self.fee]
        if not np.isfinite(numbers).all():
            raise ValueError("Paramètres non finis")
        if (mu <= 0).any() or (a < 0).any() or (b <= 0).any():
            raise ValueError("Hawkes excitant : mu et beta positifs, a non négatif")
        radius = np.max(np.abs(np.linalg.eigvals((a / b[None, :, None]).sum(1))))
        if radius >= 1:
            raise ValueError("Hawkes non stationnaire : rayon spectral >= 1")
        if not (0 < self.A <= 1 and self.kappa > 0 and self.zeta > 0 and self.r > 0):
            raise ValueError("A dans ]0,1], kappa, zeta et r strictement positifs")
        if self.sigma2 < 0 or self.gamma < 0 or self.K < 1 or int(self.K) != self.K:
            raise ValueError("sigma2 et gamma non négatifs, K entier positif")

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_calibration(cls, cal, cfg, hawkes=True, alpha=True, rho=1.0, fee_bps=0.0):
        m = cfg["model"]
        cjr_execution = m.get("execution") == "cjr"
        if cjr_execution and rho != 1.0:
            raise ValueError("Exécution CJR : aucun facteur de partage, rho doit être égal à 1")
        mu, a = cal["mu"], cal["a"]
        if not hawkes:  # Poisson aux intensités moyennes du Hawkes
            mu, a = cal["mean_rates"], np.zeros_like(np.array(a)).tolist()
        al = cal["alpha"]
        return cls(mu=mu, a=a, betas=cal["betas"], eps_plus=al["eps_plus"] if alpha else 0.0,
                   eps_minus=al["eps_minus"] if alpha else 0.0, zeta=al["zeta"], A=1.0 if cjr_execution else rho * cal["A"],
                   kappa=cal["kappa"], sigma2=cal["sigma2"], gamma=m["gamma"], r=m["r"], K=int(m["K"]), fee=fee_bps)

    def tensors(self, device):
        return dict(mu=torch.tensor(self.mu, device=device, dtype=torch.float32), a=torch.tensor(self.a, device=device, dtype=torch.float32),
                    betas=torch.tensor(self.betas, device=device, dtype=torch.float32))

    @property
    def C(self):
        return len(self.betas)

    def psi(self, q):
        return 0.5 * self.gamma * self.sigma2 * q ** 2


def dim(p):
    return 2 + 2 * p.C


def unpack(y, p):
    return y[:, 0], y[:, 1], y[:, 2:].view(-1, p.C, 2)


def intensities(y, p, T):
    _, _, E = unpack(y, p)
    lam = T["mu"][None] + torch.einsum("sck,nck->ns", T["a"], E)
    return lam  # (n, 2) : achat, vente


def decay(y, tau, p, T):
    q, al, E = unpack(y, p)
    E = E * torch.exp(-T["betas"][None, :, None] * tau[:, None, None])
    return torch.cat([q[:, None], (al * torch.exp(-p.zeta * tau))[:, None], E.reshape(len(y), -1)], 1)


def running_reward(y, tau, p):
    """∫_0^τ e^{−rt} (q α_t − ψ(q)) dt avec α_t = α e^{−ζt}, q constant."""
    q, al, _ = unpack(y, p)
    return (q * al * (-torch.expm1(-(p.r + p.zeta) * tau)) / (p.r + p.zeta)
            - p.psi(q) * (-torch.expm1(-p.r * tau)) / p.r)


def next_event(y, p, T, gen):
    """Temps jusqu'au prochain market order (amincissement, intensité décroissante entre ordres) et côté."""
    n = len(y)
    tau = torch.zeros(n, device=y.device)
    done = torch.zeros(n, dtype=torch.bool, device=y.device)
    side = torch.zeros(n, dtype=torch.long, device=y.device)
    while not bool(done.all()):
        lam_bar = intensities(decay(y, tau, p, T), p, T).sum(1)
        w = -torch.log(torch.rand(n, device=y.device, generator=gen).clamp_min(
            torch.finfo(y.dtype).tiny)) / lam_bar
        cand = tau + w
        lam = intensities(decay(y, cand, p, T), p, T)
        u = torch.rand(n, device=y.device, generator=gen)
        accept = (~done) & (u * lam_bar <= lam.sum(1))
        buy = torch.rand(n, device=y.device, generator=gen) * lam.sum(1) < lam[:, 0]
        side = torch.where(accept, torch.where(buy, 0, 1), side)
        tau = torch.where(done, tau, cand)
        done = done | accept
    return tau, side


def jump(y, side, p):
    """Tous les ordres sont influents : saut d'excitation et d'alpha, exécutés ou non."""
    y = y.clone()
    E = y[:, 2:].view(-1, p.C, 2)
    E[torch.arange(len(y)), :, side] += 1.0
    y[:, 1] += torch.where(side == 0, p.eps_plus, -p.eps_minus)
    return y


def fill_prob(delta, p):
    return p.A * torch.exp(-p.kappa * delta.clamp_min(0))
