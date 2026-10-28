"""GLFT stationary closed-form approximation, in the project's bps/lot units.

Guéant, Lehalle, Fernandez-Tapia, arXiv:1105.3115, section 4.
This is a CARA/Poisson baseline evaluated in CJR, not a CJR-optimal policy.
Symmetric intensity A_GLFT = A_fill * mean(train mean_rates), in events/second.
Maker fee c: write x=delta-c, hence A_net=A_GLFT*exp(-kappa*c).
The published formula is applied to x, then c is added back to delta.
Nonnegative distances and inventory masks are the same projections as AS.
"""
import math

import numpy as np
import torch


class GLFT:
    def __init__(self, p, mean_rates):
        rates = np.asarray(mean_rates, dtype=float)
        if rates.shape != (2,) or not np.isfinite(rates).all() or (rates <= 0).any():
            raise ValueError('Two positive finite train mean rates required')
        self.p = p
        self.arrival_rate = float(rates.mean()) * p.A
        g, k = p.gamma, p.kappa
        # Stable limits as gamma -> 0 (including the liquidity-power term).
        self.liquidity = math.log1p(g / k) / g if g else 1 / k
        log_power = (1 + k / g) * math.log1p(g / k) if g else 1.
        self.skew = (math.sqrt(p.sigma2 * g / (2 * k * self.arrival_rate))
                     * math.exp(.5 * (log_power + k * p.fee))) if g else 0.
        if not math.isfinite(self.skew):
            raise ValueError('Nonfinite GLFT coefficient')

    def deltas(self, y):
        q = y[:, 0]
        ask = (self.p.fee + self.liquidity + (.5 - q) * self.skew).clamp_min(0)
        bid = (self.p.fee + self.liquidity + (.5 + q) * self.skew).clamp_min(0)
        return torch.stack((torch.where(q > -self.p.K, ask, torch.inf),
                            torch.where(q < self.p.K, bid, torch.inf)), dim=1)

    def grid_tables(self, alpha, E, K, device, chunk):
        """Same policy, evaluated without repeating a network over the whole day."""
        if K != self.p.K:
            raise ValueError('Inventory limit differs from GLFT parameters')
        y = torch.zeros(2 * K + 1, 1)
        y[:, 0] = torch.arange(-K, K + 1)
        d = self.deltas(y).numpy()
        return (np.broadcast_to(d[:, 0], (len(alpha), len(d))),
                np.broadcast_to(d[:, 1], (len(alpha), len(d))))
