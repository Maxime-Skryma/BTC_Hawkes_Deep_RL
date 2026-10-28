"""Références indépendantes de la boucle TD : générateur et système linéaire Poisson."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def policy_value(lam_buy, lam_sell, p, da, db):
    """(r I - Q)V = g, générateur des seuls changements d'inventaire.

    Les ordres sans exécution ne changent pas q et ne figurent donc pas dans Q.
    Les distances infinies représentent une absence de cotation.
    """
    q = np.arange(-p.K, p.K + 1)
    n = len(q)
    Q = np.zeros((n, n))
    g = -p.psi(q.astype(float))
    for d, lam, shift in ((da, lam_buy, -1), (db, lam_sell, 1)):
        for i, delta in enumerate(d):
            j = i + shift
            if 0 <= j < n and np.isfinite(delta):
                if delta < 0:
                    raise ValueError("Distance négative")
                rate = lam * p.A * np.exp(-p.kappa * delta)
                Q[i, j] += rate
                Q[i, i] -= rate
                g[i] += rate * (delta - p.fee)
    return np.linalg.solve(p.r * np.eye(n) - Q, g)


def poisson_policy_iteration(lam_buy, lam_sell, p, tol=1e-11):
    """Howard + maximisation scalaire numérique, sans appel à vi.quotes.

    Référence volontairement différente de la formule fermée et de la VI.
    """
    n = 2 * p.K + 1
    da, db = np.full(n, 1 / p.kappa + max(p.fee, 0)), np.full(n, 1 / p.kappa + max(p.fee, 0))
    da[0], db[-1] = np.inf, np.inf
    V = policy_value(lam_buy, lam_sell, p, da, db)
    for _ in range(100):
        for d, shift in ((da, -1), (db, 1)):
            for i in range(n):
                j = i + shift
                if not 0 <= j < n:
                    continue
                diff = V[j] - V[i]
                upper = max(10 / p.kappa, abs(diff) + abs(p.fee) + 10 / p.kappa)
                def objective(x):
                    return -p.A * np.exp(-p.kappa * x) * (x - p.fee + diff)
                result = minimize_scalar(objective, bounds=(0, upper), method="bounded",
                                         options={"xatol": 1e-12})
                d[i] = min((0., result.x, upper), key=objective)
        new = policy_value(lam_buy, lam_sell, p, da, db)
        if np.max(np.abs(new - V)) < tol:
            return new, da, db
        V = new
    raise RuntimeError("Référence Howard non convergée")


def integrated_hazard(tau, mu, a, betas, E):
    """∫ Λ(t)dt, calcul numpy indépendant du simulateur torch."""
    mu, a, betas, E = map(np.asarray, (mu, a, betas, E))
    return (mu.sum() * tau +
            np.sum(a.sum(axis=0) * E * (-np.expm1(-betas * tau) / betas)[:, None]))


def event_bellman(y, p, value, policy):
    """Bellman semi-MDP par quadrature déterministe, sans tirage ni code torch.

    value accepte (q, alpha, E). policy fournit les deux cotes pré-événement.
    Permet de contrôler la transition complète avec valeurs tests non triviales.
    """
    q, alpha, E = y
    mu, a, betas = map(np.asarray, (p.mu, p.a, p.betas))
    def integrand(t):
        Et, at = E * np.exp(-betas * t)[:, None], alpha * np.exp(-p.zeta * t)
        survival_discount = np.exp(-p.r * t - integrated_hazard(t, mu, a, betas, E))
        lam = mu + np.einsum("sck,ck->s", a, Et)
        da, db = policy(q, at, Et)
        result = q * at - p.psi(q)
        for s, delta in enumerate((da, db)):
            Ej = Et.copy()
            Ej[:, s] += 1
            aj = at + (p.eps_plus if s == 0 else -p.eps_minus)
            v0 = value(q, aj, Ej)
            allowed = q > -p.K if s == 0 else q < p.K
            h = p.A * np.exp(-p.kappa * delta) if allowed and np.isfinite(delta) else 0.
            gain = h * (delta - p.fee + value(q + (-1 if s == 0 else 1), aj, Ej) - v0) if h else 0.
            result += lam[s] * (v0 + gain)
        return survival_discount * result
    return quad(integrand, 0, np.inf, epsabs=1e-9, epsrel=1e-9)[0]


def poisson_policy_metrics(lam_buy, lam_sell, p, values, da, db,
                           max_regret_fraction=.01, max_value_error_fraction=.01,
                           max_probability_error=.02):
    """Critères bloquants réutilisables avant une évaluation de marché."""
    optimum, oa, ob = poisson_policy_iteration(lam_buy, lam_sell, p)
    scale = max(1., float(np.max(np.abs(optimum))))
    evaluated = policy_value(lam_buy, lam_sell, p, da, db)
    prob_error = float(np.max(np.abs(np.exp(-p.kappa*np.column_stack([da, db])) -
                                       np.exp(-p.kappa*np.column_stack([oa, ob])))))
    regret = float(np.max(optimum-evaluated))/scale
    value_error = float(np.max(np.abs(np.asarray(values)-optimum)))/scale
    return dict(max_policy_regret_fraction=regret, max_value_error_fraction=value_error,
                max_fill_probability_error=prob_error,
                passed=bool(regret <= max_regret_fraction and value_error <= max_value_error_fraction
                            and prob_error <= max_probability_error))
