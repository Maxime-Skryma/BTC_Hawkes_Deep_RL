"""Non-neural local Poisson control evaluated in the unchanged CJR market.

At each query freeze the two current Hawkes intensities and optionally alpha.
Solve the discounted, infinite-horizon birth/death HJB over all inventory states:

 r V(q) = q*alpha - psi(q)
          + lambda_buy max_d h(d)[d-fee+V(q-1)-V(q)]
          + lambda_sell max_d h(d)[d-fee+V(q+1)-V(q)].

Howard iteration uses an exact tridiagonal policy evaluation and the analytic
nonnegative quote maximizer. No neural network, training, fitted parameters,
feature discretization or interpolation. Coefficients are frozen ONLY in the
decision surrogate: historical replay and synthetic CJR dynamics are untouched.
This is not a solution of the full Hawkes HJB (which includes alpha/memory jumps).
"""
import numpy as np
import torch
from numba import njit, prange


@njit(cache=True)
def _quotes(value, K, kappa, fee):
    n = 2*K+1
    ask, bid = np.full(n, np.inf), np.full(n, np.inf)
    for i in range(1, n):
        ask[i] = max(1/kappa+fee-value[i-1]+value[i], 0.)
    for i in range(n-1):
        bid[i] = max(1/kappa+fee-value[i+1]+value[i], 0.)
    return ask, bid


@njit(cache=True)
def _solve(lam_buy, lam_sell, alpha, K, A, kappa, fee, r, risk, tol, max_iter):
    n = 2*K+1
    value = np.zeros(n)
    residual = np.inf
    for iteration in range(max_iter):
        ask, bid = _quotes(value, K, kappa, fee)
        lower = np.zeros(n)
        upper = np.zeros(n)
        diagonal = np.zeros(n)
        rhs = np.zeros(n)
        for i in range(n):
            qa = lam_buy*A*np.exp(-kappa*ask[i]) if i > 0 else 0.
            qb = lam_sell*A*np.exp(-kappa*bid[i]) if i < n-1 else 0.
            lower[i], upper[i] = -qa, -qb
            diagonal[i] = r+qa+qb
            q = i-K
            rhs[i] = q*alpha-risk*q*q
            if i > 0:
                rhs[i] += qa*(ask[i]-fee)
            if i < n-1:
                rhs[i] += qb*(bid[i]-fee)
        # Thomas elimination: r>0 makes every policy matrix strictly diagonally dominant.
        for i in range(1, n):
            factor = lower[i]/diagonal[i-1]
            diagonal[i] -= factor*upper[i-1]
            rhs[i] -= factor*rhs[i-1]
        value[-1] = rhs[-1]/diagonal[-1]
        for i in range(n-2, -1, -1):
            value[i] = (rhs[i]-upper[i]*value[i+1])/diagonal[i]
        ask, bid = _quotes(value, K, kappa, fee)
        residual = 0.
        scale = 1.
        for i in range(n):
            q = i-K
            reward = q*alpha-risk*q*q
            if i > 0:
                reward += lam_buy*A*np.exp(-kappa*ask[i])*(ask[i]-fee+value[i-1]-value[i])
            if i < n-1:
                reward += lam_sell*A*np.exp(-kappa*bid[i])*(bid[i]-fee+value[i+1]-value[i])
            residual = max(residual, abs(r*value[i]-reward))
            scale = max(scale, abs(reward), abs(r*value[i]))
        if residual <= tol*scale:
            return value, ask, bid, iteration+1, residual
    raise RuntimeError('Local-control Howard iteration did not converge')


@njit(cache=True, parallel=True)
def _tables(rates, alpha, K, A, kappa, fee, r, risk, tol, max_iter):
    ask = np.empty((len(rates), 2*K+1), np.float32)
    bid = np.empty_like(ask)
    for i in prange(len(rates)):
        _, da, db, _, _ = _solve(rates[i, 0], rates[i, 1], alpha[i],
                                K, A, kappa, fee, r, risk, tol, max_iter)
        ask[i], bid[i] = da, db
    return ask, bid


def solve_local(p, rates, alpha=0., tol=1e-10, max_iter=100):
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (2,) or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError('Two positive finite intensities required')
    if not np.isfinite(alpha) or tol <= 0 or max_iter < 1:
        raise ValueError('Invalid local-control solver parameters')
    return _solve(*rates, float(alpha), p.K, p.A, p.kappa, p.fee, p.r,
                  .5*p.gamma*p.sigma2, tol, max_iter)


class LocalControl:
    def __init__(self, p, use_alpha=False, tol=1e-10, max_iter=100):
        self.p, self.use_alpha = p, use_alpha
        self.tol, self.max_iter = tol, max_iter

    def grid_tables(self, alpha, E, K, device, chunk=20000):
        if K != self.p.K:
            raise ValueError('Inventory limit differs from local-control parameters')
        p = self.p
        rates = np.asarray(p.mu)[None] + np.einsum('sck,nck->ns', np.asarray(p.a),
                                                     np.asarray(E).reshape(-1, p.C, 2))
        drift = np.asarray(alpha, dtype=float) if self.use_alpha else np.zeros(len(rates))
        if not np.isfinite(rates).all() or (rates <= 0).any() or not np.isfinite(drift).all():
            raise ValueError('Nonfinite local-control observations')
        return _tables(np.ascontiguousarray(rates), np.ascontiguousarray(drift), K,
                       p.A, p.kappa, p.fee, p.r, .5*p.gamma*p.sigma2, self.tol, self.max_iter)

    @torch.no_grad()
    def deltas(self, y):
        obs = y.detach().cpu().numpy()
        da, db = self.grid_tables(obs[:, 1], obs[:, 2:], self.p.K, 'cpu')
        indices = np.rint(obs[:, 0]).astype(int)+self.p.K
        if (indices < 0).any() or (indices > 2*self.p.K).any():
            raise ValueError('Inventory outside limits')
        rows = np.arange(len(obs))
        return torch.as_tensor(np.column_stack((da[rows, indices], db[rows, indices])),
                               dtype=y.dtype, device=y.device)
