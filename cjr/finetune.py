"""Fine-tuning TD3/SAC à partir d'un checkpoint existant.

Module séparé : rl_algorithms.py reste identique à l'empreinte figée des études v1.
Composants (tous explicites dans les réglages, aucun n'est activé implicitement) :
- initialisation acteur + deux critiques + cibles depuis le checkpoint, mêmes unités de
  normalisation (alpha_scale, value_scale) que le checkpoint ;
- échauffement du critique, acteur figé (critic_warmup transitions) ;
- ancrage de type TD3+BC vers l'acteur initial, poids normalisé par |Q| moyen et décroissant
  linéairement jusqu'à zéro (anchor_weight, anchor_decay) : limite l'oubli du checkpoint ;
- plusieurs mises à jour par pas vectoriel (updates_per_step), comptées explicitement ;
- moyenne exponentielle des poids de l'acteur (actor_ema), évaluée à côté de l'acteur courant ;
- randomisation optionnelle de la dynamique Hawkes/alpha par groupes d'environnements ;
- plancher configurable de la température SAC (log_alpha_min, unités natives).
Évaluation : actions déterministes, comme dans rl_algorithms.
"""
import copy
import hashlib
import math
import numpy as np
import torch
from cjr import model as M, rl_algorithms as RL

FINETUNE_VERSION='cjr-hawkes-finetune-v1'


def actor_fingerprint(state):
    h=hashlib.sha256()
    for k,v in sorted(state['actor'].items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def perturbed_params(p,rng,ranges):
    """Facteurs log-uniformes sur l'intensité de base, l'excitation, l'impact et la persistance de alpha.
    Frais, kappa, A, K, sigma2, gamma et r restent nominaux : les actions gardent le même sens."""
    f={k:float(np.exp(rng.uniform(np.log(lo),np.log(hi)))) for k,(lo,hi) in ranges.items()}
    a=np.asarray(p.a)*f.get('excitation',1.);b=np.asarray(p.betas)
    radius=np.max(np.abs(np.linalg.eigvals((a/b[None,:,None]).sum(1))))
    if radius>=.99:a*=.99/radius
    return M.Params(**{**p.__dict__,'mu':(np.asarray(p.mu)*f.get('baseline',1.)).tolist(),'a':a.tolist(),
        'eps_plus':p.eps_plus*f.get('alpha_impact',1.),'eps_minus':p.eps_minus*f.get('alpha_impact',1.),
        'zeta':p.zeta*f.get('alpha_decay',1.)}),f


def randomized_params(p,groups,seed,ranges):
    """Groupe 0 nominal, puis groups-1 dynamiques perturbées, tirées une fois par seed."""
    rng=np.random.default_rng([seed,20261027]);out=[(p,{})]
    for _ in range(groups-1):out.append(perturbed_params(p,rng,ranges or {}))
    return out


class GroupedEnvironment:
    """Concatène des EventEnvironment ; récompenses dans les unités du groupe nominal."""
    def __init__(self,params,n,seed):
        sizes=[n//len(params)+(g<n%len(params)) for g in range(len(params))]
        if min(sizes)<1:raise ValueError('Plus de groupes que d\'environnements')
        self.envs=[RL.EventEnvironment(pg,m,seed+1000003*g) for g,(pg,m) in enumerate(zip(params,sizes))]
        self.alpha_scale=self.envs[0].alpha_scale;self.value_scale=self.envs[0].value_scale
        for env in self.envs[1:]:env.value_scale=self.value_scale
        self.sizes=sizes

    @property
    def y(self):return torch.cat([e.y for e in self.envs])

    @y.setter
    def y(self,value):
        for e,part in zip(self.envs,value.split(self.sizes)):e.y=part.clone()

    def step(self,action):
        out=[e.step(a) for e,a in zip(self.envs,action.split(self.sizes))]
        return tuple(torch.cat(x) for x in zip(*out))

    def rng_state(self):return [e.gen.get_state() for e in self.envs]

    def set_rng_state(self,states):
        for e,s in zip(self.envs,states):e.gen.set_state(s)


def averaged(ag):
    """Agent d'évaluation utilisant l'acteur moyenné ; critiques partagés (non utilisés en évaluation)."""
    out=copy.copy(ag);out.actor=ag.actor_ema;return out


def train(p,settings,algorithm,seed,init_state,log=print,checkpoint_callback=None,resume_state=None):
    if algorithm not in ('td3','sac'):raise ValueError('Fine-tuning TD3/SAC uniquement')
    torch.manual_seed(seed);torch.set_num_threads(settings['threads'])
    groups=randomized_params(p,settings.get('dr_groups',1),seed,settings.get('domain_randomization'))
    env=GroupedEnvironment([g for g,_ in groups],settings['envs'],seed)
    if not math.isclose(env.value_scale,init_state['value_scale'],rel_tol=1e-9):
        raise ValueError('Unités de récompense différentes de celles du checkpoint')
    # Normaliseur du checkpoint : alpha_scale dépend de la seed de son propre préchauffage.
    ag=RL.Agent(p,settings,algorithm,init_state['alpha_scale'],init_state['value_scale']);ag.load(init_state)
    reference=copy.deepcopy(ag.actor).requires_grad_(False);ag.actor_ema=copy.deepcopy(ag.actor).requires_grad_(False)
    init_id=actor_fingerprint(init_state)
    opt_actor=torch.optim.Adam(ag.actor.parameters(),lr=settings['actor_lr'])
    critic_params=list(ag.q1.parameters())+list(ag.q2.parameters())
    opt_critic=torch.optim.Adam(critic_params,lr=settings['critic_lr'])
    log_alpha_native=torch.tensor(math.log(settings['sac_initial_alpha']),requires_grad=True)
    opt_alpha=torch.optim.Adam([log_alpha_native],lr=settings['sac_alpha_lr'])
    buffer=RL.ReplayBuffer(settings['buffer'],M.dim(p),seed)
    envs,ups,delay=settings['envs'],settings['updates_per_step'],settings['policy_delay']
    steps=settings['transitions']//envs;history=[];first_step=0
    def record(row):
        if not all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float))):raise RuntimeError('Fine-tuning non fini')
        history.append(row);log(row)
    if resume_state is not None:
        if (resume_state['training_seed'],resume_state['algorithm'],resume_state['init_actor'])!=(seed,algorithm,init_id):
            raise ValueError('Resume seed/algorithm/initial checkpoint mismatch')
        if any(resume_state['settings'].get(k)!=v for k,v in settings.items() if k!='threads'):
            raise ValueError('Resume settings mismatch')
        for name in ('actor','q1','q2','t1','t2','target_actor','actor_ema'):
            net=getattr(ag,name)
            if net is not None:net.load_state_dict(resume_state['networks'][name])
        opt_actor.load_state_dict(resume_state['opt_actor']);opt_critic.load_state_dict(resume_state['opt_critic'])
        opt_alpha.load_state_dict(resume_state['opt_alpha'])
        with torch.no_grad():log_alpha_native.copy_(resume_state['log_alpha_native'])
        env.y=resume_state['env_y'];env.set_rng_state(resume_state['env_rng'])
        buffer.data=resume_state['buffer_data'];buffer.size=resume_state['buffer_size'];buffer.cursor=resume_state['buffer_cursor']
        buffer.rng.bit_generator.state=resume_state['buffer_rng']
        history=copy.deepcopy(resume_state['history']);first_step=resume_state['completed_steps']
        torch.set_rng_state(resume_state['torch_rng'])
    def snapshot(completed):
        return dict(finetune_version=FINETUNE_VERSION,algorithm=algorithm,training_seed=seed,settings=dict(settings),
            init_actor=init_id,completed_steps=completed,
            networks={n:getattr(ag,n).state_dict() for n in ('actor','q1','q2','t1','t2','target_actor','actor_ema') if getattr(ag,n) is not None},
            opt_actor=opt_actor.state_dict(),opt_critic=opt_critic.state_dict(),opt_alpha=opt_alpha.state_dict(),
            log_alpha_native=log_alpha_native.detach().clone(),env_y=env.y.clone(),env_rng=env.rng_state(),
            buffer_data=buffer.data,buffer_size=buffer.size,buffer_cursor=buffer.cursor,
            buffer_rng=buffer.rng.bit_generator.state,torch_rng=torch.get_rng_state(),history=history)
    updates=actor_updates=0
    for it in range(first_step,steps):
        y=env.y.clone()
        with torch.no_grad():
            mask=RL.active_sides(y,p)
            # Pas d'actions uniformes initiales : le tampon est rempli par la politique pré-entraînée.
            a,_,_=ag.actor.sample(ag.norm(y),mask)
            if algorithm=='td3':a=(a+settings['exploration_noise']*torch.randn_like(a)).clamp_min(0)*mask
            yn,r,d=env.step(a);buffer.add(y,a,r,yn,d)
        n=(it+1)*envs
        if buffer.size<settings['start_steps']:
            if checkpoint_callback is not None and checkpoint_callback(ag,n,lambda:snapshot(it+1)):break
            continue
        frozen=n<=settings['critic_warmup']
        anchor=settings['anchor_weight']*max(0.,1-n/settings['anchor_decay']) if settings['anchor_decay']>0 else 0.
        for k in range(ups):
            u=it*ups+k
            yb,ab,rb,ynb,db=buffer.sample(settings['batch']);x=ag.norm(yb);xn=ag.norm(ynb);maskn=RL.active_sides(ynb,p)
            temperature=RL.normalized_temperature(log_alpha_native.exp().detach(),ag.value_scale)
            with torch.no_grad():
                if algorithm=='td3':
                    an,_,_=ag.target_actor.sample(xn,maskn,True)
                    noise=(torch.randn_like(an)*settings['target_noise']).clamp(-settings['noise_clip'],settings['noise_clip'])
                    an=(an+noise).clamp_min(0)*maskn
                    future=torch.minimum(ag.t1(xn,an),ag.t2(xn,an))
                else:
                    an,lpn,_=ag.actor.sample(xn,maskn)
                    future=torch.minimum(ag.t1(xn,an),ag.t2(xn,an))-temperature*lpn
                target=rb+db*future
            loss_critic=(ag.q1(x,ab)-target).square().mean()+(ag.q2(x,ab)-target).square().mean()
            opt_critic.zero_grad();loss_critic.backward();torch.nn.utils.clip_grad_norm_(critic_params,10.);opt_critic.step();updates+=1
            delayed=algorithm=='sac' or u%delay==0
            loss_actor=bc=torch.tensor(0.)
            if delayed and not frozen:
                for param in critic_params:param.requires_grad_(False)
                maskb=RL.active_sides(yb,p);ac,lp,_=ag.actor.sample(x,maskb)
                if algorithm=='td3':q=ag.q1(x,ac);rl_loss=-q.mean();deterministic=ac
                else:
                    q=torch.minimum(ag.q1(x,ac),ag.q2(x,ac));rl_loss=(temperature*lp-q).mean()
                    deterministic=ag.actor.sample(x,maskb,True)[0]
                with torch.no_grad():ref=reference.sample(x,maskb,True)[0]
                bc=(deterministic-ref).square().sum(1).mean()
                # TD3+BC (Fujimoto & Gu, 2021) : lambda relatif à |Q| pour rester indépendant de l'échelle.
                loss_actor=rl_loss+anchor*q.abs().mean().detach()*bc
                opt_actor.zero_grad();loss_actor.backward();torch.nn.utils.clip_grad_norm_(ag.actor.parameters(),10.);opt_actor.step()
                for param in critic_params:param.requires_grad_(True)
                actor_updates+=1
                if algorithm=='sac':
                    alpha_loss=-(log_alpha_native*(lp+(-maskb.sum(1))).detach()).mean()
                    opt_alpha.zero_grad();alpha_loss.backward();opt_alpha.step()
                    with torch.no_grad():log_alpha_native.clamp_(settings['log_alpha_min'],1)
                else:RL.polyak(ag.target_actor,ag.actor,settings['tau'])
                RL.polyak(ag.actor_ema,ag.actor,1-settings['actor_ema'])
            if delayed:RL.polyak(ag.t1,ag.q1,settings['tau']);RL.polyak(ag.t2,ag.q2,settings['tau'])
        if it%256==0 or it==steps-1:
            record(dict(transitions=n,critic_loss=float(loss_critic.detach()),actor_loss=float(loss_actor.detach()),
                        anchor_mse=float(bc.detach()),anchor_weight=anchor,actor_frozen=int(frozen),
                        alpha_native=float(log_alpha_native.exp().detach()) if algorithm=='sac' else 0.,
                        critic_updates_this_invocation=updates,actor_updates_this_invocation=actor_updates))
        if checkpoint_callback is not None and checkpoint_callback(ag,n,lambda:snapshot(it+1)):break
    return ag,history
