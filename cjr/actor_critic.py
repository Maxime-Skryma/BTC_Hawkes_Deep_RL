"""Acteur-critique de Guéant–Manziuk en temps événementiel.

Critique V(y) : valeur à un état quelconque entre deux ordres. Cible sur une transition simulée
(y → prochain ordre à τ, côté s, état décru y⁻) :
    V(y) ← ∫_0^τ e^{−rt}(qα − ψ) dt + e^{−rτ} θ_s(y⁻),
    θ_s(y⁻) = V'(J_s y⁻) + 1{côté autorisé} h(δ_s) (δ_s − c + V'(J_s y⁻, q ∓ 1) − V'(J_s y⁻)),
l'espérance sur l'exécution étant calculée exactement (seul le côté et τ sont tirés).
Acteur : δ_s(y⁻) maximise h(δ)(δ − c + ΔV) avec le critique figé. Il sort la probabilité d'exécution
p = h(δ)/A ∈ (0, 1] via -log(p) projeté ; d'où δ = −ln(p)/κ ≥ 0.
Référence de contrôle : δ* = max(1/κ + c − ΔV, 0) en forme fermée à partir du critique.
"""
import copy

ALGORITHM_VERSION = "cjr-event-ac-v4"

import numpy as np
import torch
from torch import nn

from cjr import model as M


class Normalizer:
    def __init__(self, p, alpha_scale, observation='full'):
        if observation not in ('full', 'inventory', 'alpha', 'hawkes'):
            raise ValueError('Unknown observation ablation')
        self.observation = observation
        self.K, self.alpha_scale = p.K, alpha_scale
        # Une mémoire dont tous les coefficients sont nuls est sans effet sur le contrôle.
        self.memory_mask = np.any(np.asarray(p.a) != 0, axis=0).ravel() & (observation in ('full', 'hawkes'))

    def __call__(self, y):
        mask = torch.as_tensor(self.memory_mask, device=y.device)
        alpha = y[:, 1:2] / self.alpha_scale if self.observation in ('full', 'alpha') else torch.zeros_like(y[:, 1:2])
        return torch.cat([y[:, :1] / self.K, alpha,
                          torch.log1p(y[:, 2:]) * mask], 1)


def mlp(n_in, n_out, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, n_out))


class Agent:
    def __init__(self, p, cfg, device, alpha_scale, value_scale):
        a = cfg["agent"]
        self.p, self.device, self.value_scale = p, device, value_scale
        self.norm = Normalizer(p, alpha_scale)
        self.critic = mlp(M.dim(p), 1, a["hidden"]).to(device)
        self.actor = mlp(M.dim(p), 2, a["hidden"]).to(device)   # -log(p_ask), -log(p_bid), projetés sur [0,+inf[
        self.target = copy.deepcopy(self.critic)

    def V(self, y, net=None):
        return (net or self.critic)(self.norm(y))[:, 0]

    def log_probs(self, y):
        return -self.actor(self.norm(y)).clamp_min(0)

    def probs(self, y):
        return self.log_probs(y).exp()

    def deltas(self, y):
        """(δ_ask, δ_bid) en bps ; inf du côté interdit par la limite d'inventaire."""
        d = -self.log_probs(y) / self.p.kappa
        q = y[:, 0]
        d[:, 0] = torch.where(q > -self.p.K, d[:, 0], torch.inf)
        d[:, 1] = torch.where(q < self.p.K, d[:, 1], torch.inf)
        return d

    def state_dict(self):
        return dict(algorithm_version=ALGORITHM_VERSION, params_fingerprint=self.p.fingerprint(),
                    critic=self.critic.state_dict(), actor=self.actor.state_dict(),
                    alpha_scale=self.norm.alpha_scale, value_scale=self.value_scale)

    def load(self, sd):
        if sd.get("algorithm_version") != ALGORITHM_VERSION or sd.get("params_fingerprint") != self.p.fingerprint():
            raise ValueError("Checkpoint ancien ou paramètres différents : réentraînement requis (nouveau chemin)")
        self.critic.load_state_dict(sd["critic"])
        self.actor.load_state_dict(sd["actor"])


def side_terms(agent, yd, p, net):
    """Pour chaque côté (0 : ask touché par un achat, 1 : bid touché par une vente) : V sans exécution,
    gain d'exécution δ − c + ΔV (en unités normalisées), probabilité d'exécution h, masque d'autorisation."""
    log_pr = agent.log_probs(yd)
    out = []
    for s in (0, 1):
        side = torch.full((len(yd),), s, device=yd.device)
        ys = M.jump(yd, side, p)
        yf = ys.clone()
        yf[:, 0] = (yf[:, 0] + (-1.0 if s == 0 else 1.0)).clamp(-p.K, p.K)
        allowed = (yd[:, 0] > -p.K) if s == 0 else (yd[:, 0] < p.K)
        v0, v1 = agent.V(ys, net), agent.V(yf, net)
        delta = -log_pr[:, s] / p.kappa
        h = p.A * log_pr[:, s].exp()
        out.append((v0, (delta - p.fee) / agent.value_scale + v1 - v0, h, allowed.float()))
    return out


def train(p, cfg, device, log=print):
    a = cfg["agent"]
    T = p.tensors(device)
    gen = torch.Generator(device=device).manual_seed(a["seed"])
    torch.manual_seed(a["seed"])
    exact_poisson = not np.any(p.a) and p.eps_plus == 0 and p.eps_minus == 0
    B = 2*p.K+1 if exact_poisson else a["batch"]
    y = torch.zeros(B, M.dim(p), device=device)
    y[:, 0] = (torch.arange(-p.K,p.K+1,device=device).float() if exact_poisson else
               torch.randint(-p.K, p.K + 1, (B,), device=device, generator=gen).float())
    # Préchauffage en nombre d’événements (approximation, pas une preuve de stationnarité).
    for _ in range(0 if exact_poisson else 300):
        tau, side = M.next_event(y, p, T, gen)
        y = M.jump(M.decay(y, tau, p, T), side, p)
    lam_bar = float(M.intensities(y, p, T).sum(1).mean())
    alpha_scale = float(y[:, 1].std().clamp_min(1e-6)) if (p.eps_plus or p.eps_minus) else 1.0
    value_scale = p.A / p.kappa * np.exp(-1) * lam_bar / p.r
    agent = Agent(p, cfg, device, alpha_scale, value_scale)
    opt_c = torch.optim.Adam(agent.critic.parameters(), lr=a["lr_critic"])
    opt_a = torch.optim.Adam(agent.actor.parameters(), lr=a["lr_actor"])
    # Décroissance cosinus des pas : précision finale du critique (les ΔV donnent les δ).
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, a["iterations"], eta_min=0.05 * o.param_groups[0]["lr"])
              for o in (opt_c, opt_a)]
    hist = []
    for it in range(a["iterations"]):
        if exact_poisson:
            yd = y  # tous les inventaires ; attente intégrée analytiquement ci-dessous
        else:
            tau, side = M.next_event(y, p, T, gen)
            yd = M.decay(y, tau, p, T)
            R = M.running_reward(y, tau, p) / value_scale
        with torch.no_grad():
            terms = side_terms(agent, yd, p, agent.target)
            theta = torch.stack([v0 + m * h * g for v0, g, h, m in terms], 1)
            if exact_poisson:
                # Cas dégénéré : intégration exacte du temps d'attente et du côté (GM, eq. 6–9).
                # Aucun recours à la solution VI ni à des cibles supervisées.
                rates = T["mu"]
                target = (-p.psi(y[:, 0]) / value_scale + (theta * rates).sum(1)) / (p.r + rates.sum())
            else:
                target = R + torch.exp(-p.r * tau) * theta.gather(1, side[:, None])[:, 0]
        loss_c = ((agent.V(y) - target) ** 2).mean()
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()

        for prm in agent.critic.parameters():
            prm.requires_grad_(False)
        terms = side_terms(agent, yd, p, agent.critic)
        # Espérance sur le côté conditionnellement à l’état pré-événement.
        # Le poids uniforme précédent surpondérait le côté rare dans un marché asymétrique.
        lam = M.intensities(yd, p, T).detach()
        weights = lam / lam.sum(1, keepdim=True)
        if a.get("actor_update", "hamiltonian") == "closed_form":
            # Amélioration exacte spécifique à h exponentielle, puis projection dans l'acteur.
            # Le critique fournit ΔV ; aucune cible issue de VI ni des données de test.
            log_probs = agent.log_probs(yd)
            distances = -log_probs / p.kappa
            differences = torch.stack([g for _, g, _, _ in terms], 1) - (distances-p.fee)/value_scale
            target_d = (1/p.kappa+p.fee-value_scale*differences.detach()).clamp_min(0)
            masks = torch.stack([m for _, _, _, m in terms], 1)
            loss_a = (weights*masks*(agent.actor(agent.norm(yd))-p.kappa*target_d)**2).mean()
        else:
            loss_a = -sum((weights[:, s] * m * h * g).mean()
                          for s, (_, g, h, m) in enumerate(terms))
        opt_a.zero_grad()
        loss_a.backward()
        opt_a.step()
        for prm in agent.critic.parameters():
            prm.requires_grad_(True)

        for sc in scheds:
            sc.step()
        with torch.no_grad():
            for t_, s_ in zip(agent.target.parameters(), agent.critic.parameters()):
                t_.mul_(1 - a["polyak"]).add_(a["polyak"] * s_)
            if not exact_poisson:
                v0, g, h, m = (torch.stack(x, 1) for x in zip(*terms))
                hs = h.gather(1, side[:, None])[:, 0] * m.gather(1, side[:, None])[:, 0]
                fill = torch.rand(B, device=device, generator=gen) < hs
                y = M.jump(yd, side, p)
                y[:, 0] += torch.where(fill, torch.where(side == 0, -1.0, 1.0), 0.0)
                reset = torch.rand(B, device=device, generator=gen) < a["reset_q"]
                y[:, 0] = torch.where(reset, torch.randint(-p.K, p.K + 1, (B,), device=device,
                                                           generator=gen).float(), y[:, 0])
        if it % 1000 == 0 or it == a["iterations"] - 1:
            hist.append(dict(it=it, loss_critic=loss_c.item(), objective_actor=-loss_a.item(),
                             V_mean=float(agent.V(y).mean().detach() * value_scale)))
            log(hist[-1])
    return agent, hist
