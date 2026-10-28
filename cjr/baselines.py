"""Avellaneda–Stoikov approché, en bps d'un lot, à horizon glissant explicite.

Formules AS 2008, section 3.2 ; adaptations : frais maker, delta>=0 et bornes q.
L'horizon glissant rend la règle stationnaire ; ce n'est pas l'horizon terminal
de la journée. Le coefficient gamma est exprimé dans les unités normalisées
du projet et repris sans ajustement du coefficient de risque CJR.
"""
import math
import torch


def avellaneda_stoikov(p, horizon_s):
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError('Horizon AS strictement positif et fini requis')
    gamma = p.gamma
    liquidity = math.log1p(gamma / p.kappa) / gamma if gamma > 0 else 1 / p.kappa
    inventory_risk = gamma * p.sigma2 * horizon_s
    half = liquidity + .5 * inventory_risk + p.fee

    def policy(y):
        q = y[:, 0]
        ask = (half - q * inventory_risk).clamp_min(0)
        bid = (half + q * inventory_risk).clamp_min(0)
        ask = torch.where(q > -p.K, ask, torch.inf)
        bid = torch.where(q < p.K, bid, torch.inf)
        return torch.stack((ask, bid), 1)
    return policy
