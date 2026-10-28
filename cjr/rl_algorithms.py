"""TD3, SAC et PPO pour le semi-MDP Hawkes événementiel.

État avant ordre ; action a=kappa*delta >=0, sans plafond supérieur.
Après l'ordre : gain immédiat, attente exacte Hawkes, récompense intégrée,
actualisation exp(-r*tau). Le Brownien est intégré en espérance à l'entraînement.
Critiques : une sortie par inventaire entier. Évaluation : actions déterministes.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cjr import model as M
from cjr.actor_critic import Normalizer, mlp

ALGORITHM_VERSION='cjr-hawkes-td3-sac-ppo-v2'


def normalized_temperature(native_alpha, value_scale):
    """Entropy and rewards must undergo the same change of value units."""
    if native_alpha <= 0 or value_scale <= 0:
        raise ValueError('Temperature and value scale must be positive')
    return native_alpha/value_scale


def active_sides(y,p):return torch.stack((y[:,0]>-p.K,y[:,0]<p.K),1).to(y.dtype)


def action_deltas(action,y,p):
    return torch.where(active_sides(y,p).bool(),action/p.kappa,torch.inf)


def softplus_log_prob(mean,logstd,z,mask):
    # Jacobien da/dz = sigmoid(z), a=softplus(z). Pas de borne haute artificielle.
    normal=-.5*((z-mean)/logstd.exp()).square()-logstd-.5*math.log(2*math.pi)
    return ((normal-F.logsigmoid(z))*mask).sum(1)


class Policy(nn.Module):
    def __init__(self,n_in,hidden,base,stochastic=False,global_std=False):
        super().__init__();self.stochastic=stochastic;self.global_std=global_std
        self.net=mlp(n_in,2 if global_std or not stochastic else 4,hidden)
        nn.init.uniform_(self.net[-1].weight,-.003,.003)
        with torch.no_grad():
            self.net[-1].bias[:2].fill_(base+math.log(-math.expm1(-base)))
            if stochastic and not global_std:self.net[-1].bias[2:].fill_(-.5)
        if global_std:self.logstd=nn.Parameter(torch.full((2,),-.5))

    def distribution(self,x):
        out=self.net(x);mean=out[:,:2]
        if self.global_std:logstd=self.logstd.expand_as(mean)
        elif self.stochastic:logstd=out[:,2:]
        else:logstd=torch.zeros_like(mean)
        return mean,logstd.clamp(-5,1)

    def sample(self,x,mask,deterministic=False):
        mean,logstd=self.distribution(x)
        z=mean if deterministic or not self.stochastic else mean+logstd.exp()*torch.randn_like(mean)
        action=F.softplus(z)*mask
        logp=softplus_log_prob(mean,logstd,z,mask) if self.stochastic else torch.zeros(len(x),device=x.device)
        return action,logp,z


class Critic(nn.Module):
    def __init__(self,n_in,K,hidden,q_function=True):
        super().__init__();self.K=K;self.q_function=q_function
        self.net=mlp(n_in-1+(2 if q_function else 0),2*K+1,hidden)

    def forward(self,x,action=None):
        features=x[:,1:]
        if self.q_function:features=torch.cat((features,torch.log1p(action)),1)
        values=self.net(features)
        index=torch.round(x[:,0]*self.K).long()+self.K
        return values.gather(1,index[:,None])[:,0]


class Agent:
    def __init__(self,p,cfg,algorithm,alpha_scale,value_scale,device='cpu'):
        self.p,self.algorithm,self.cfg=p,algorithm,cfg
        self.norm=Normalizer(p,alpha_scale,cfg.get('observation','full'));self.value_scale=value_scale;self.device=device
        self.actor=Policy(M.dim(p),cfg['hidden'],1+p.kappa*p.fee,algorithm!='td3',algorithm=='ppo').to(device)
        self.q1=Critic(M.dim(p),p.K,cfg['hidden'],algorithm!='ppo').to(device)
        self.q2=Critic(M.dim(p),p.K,cfg['hidden']).to(device) if algorithm!='ppo' else None
        self.target_actor=copy.deepcopy(self.actor) if algorithm=='td3' else None
        self.t1=copy.deepcopy(self.q1) if algorithm!='ppo' else None
        self.t2=copy.deepcopy(self.q2) if algorithm!='ppo' else None

    def deltas(self,y):
        action,_,_=self.actor.sample(self.norm(y),active_sides(y,self.p),deterministic=True)
        return action_deltas(action,y,self.p)

    def to(self,device):
        for name in ('actor','q1','q2','target_actor','t1','t2'):
            module=getattr(self,name)
            if module is not None:module.to(device)
        self.device=device;return self

    def state_dict(self):
        return dict(algorithm_version=ALGORITHM_VERSION,algorithm=self.algorithm,
                    params_fingerprint=self.p.fingerprint(),settings=self.cfg,
                    alpha_scale=self.norm.alpha_scale,value_scale=self.value_scale,
                    actor=self.actor.state_dict(),q1=self.q1.state_dict(),
                    q2=None if self.q2 is None else self.q2.state_dict())

    def load(self,sd):
        if (sd.get('algorithm_version')!=ALGORITHM_VERSION or sd.get('algorithm')!=self.algorithm
            or sd.get('params_fingerprint')!=self.p.fingerprint()
            or sd['settings'].get('observation','full')!=self.cfg.get('observation','full')):raise ValueError('Checkpoint RL incompatible')
        self.actor.load_state_dict(sd['actor']);self.q1.load_state_dict(sd['q1'])
        if self.q2 is not None:self.q2.load_state_dict(sd['q2'])
        for target,source in ((self.target_actor,self.actor),(self.t1,self.q1),(self.t2,self.q2)):
            if target is not None:target.load_state_dict(source.state_dict())


class EventEnvironment:
    """Transition d'une décision pré-ordre à la décision pré-ordre suivante."""
    def __init__(self,p,n,seed,device='cpu'):
        self.p=p;self.T=p.tensors(device);self.gen=torch.Generator(device=device).manual_seed(seed)
        self.y=torch.zeros(n,M.dim(p),device=device)
        for _ in range(300):
            tau,side=M.next_event(self.y,p,self.T,self.gen)
            self.y=M.jump(M.decay(self.y,tau,p,self.T),side,p)
        tau,_=M.next_event(self.y,p,self.T,self.gen);self.y=M.decay(self.y,tau,p,self.T)
        self.y[:,0]=torch.randint(-p.K,p.K+1,(n,),generator=self.gen,device=device).float()
        self.alpha_scale=float(self.y[:,1].std().clamp_min(1e-6)) if p.eps_plus or p.eps_minus else 1.
        rates=np.linalg.solve(np.eye(2)-(np.asarray(p.a)/np.asarray(p.betas)[None,:,None]).sum(1),p.mu)
        self.value_scale=max(float(p.A/p.kappa*np.exp(-1)*rates.sum()/p.r),1e-6)

    @torch.no_grad()
    def step(self,action):
        p=self.p;y=self.y;mask=active_sides(y,p)
        lam=M.intensities(y,p,self.T)
        buy=torch.rand(len(y),generator=self.gen,device=y.device)<lam[:,0]/lam.sum(1)
        side=torch.where(buy,0,1)
        chosen=action.gather(1,side[:,None])[:,0]
        allowed=mask.gather(1,side[:,None])[:,0]
        fill=torch.rand(len(y),generator=self.gen,device=y.device)<p.A*torch.exp(-chosen)*allowed
        post=M.jump(y,side,p)
        post[:,0]+=torch.where(fill,torch.where(side==0,-1.,1.),0.)
        tau,_=M.next_event(post,p,self.T,self.gen)
        reward=(torch.where(fill,chosen/p.kappa-p.fee,0.)+M.running_reward(post,tau,p))/self.value_scale
        discount=torch.exp(-p.r*tau)
        self.y=M.decay(post,tau,p,self.T)
        return self.y.clone(),reward,discount


class ReplayBuffer:
    def __init__(self,capacity,dim,seed):
        self.capacity=capacity;self.size=0;self.cursor=0;self.rng=np.random.default_rng(seed)
        self.data=[torch.empty((capacity,d)) for d in (dim,2,1,dim,1)]

    def add(self,y,a,r,yn,discount):
        n=len(y);idx=(torch.arange(n)+self.cursor)%self.capacity
        for dst,src in zip(self.data,(y,a,r[:,None],yn,discount[:,None])):dst[idx]=src.detach()
        self.cursor=(self.cursor+n)%self.capacity;self.size=min(self.capacity,self.size+n)

    def sample(self,n):
        index=torch.from_numpy(self.rng.integers(0,self.size,n))
        y,a,r,yn,d=[t[index] for t in self.data]
        return y,a,r[:,0],yn,d[:,0]


def polyak(target,source,tau):
    with torch.no_grad():
        for t,s in zip(target.parameters(),source.parameters()):t.mul_(1-tau).add_(s,alpha=tau)


def clipped_ppo_objective(logp,old_logp,adv,clip):
    ratio=(logp-old_logp).exp()
    return torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv)


def gae(rewards,discounts,values,bootstrap,lam):
    advantage=torch.zeros_like(rewards);carry=torch.zeros_like(bootstrap)
    nxt=bootstrap
    for t in range(len(rewards)-1,-1,-1):
        delta=rewards[t]+discounts[t]*nxt-values[t]
        carry=delta+discounts[t]*lam*carry;advantage[t]=carry;nxt=values[t]
    return advantage,advantage+values


def train(p,settings,algorithm,seed,log=print,checkpoint_callback=None,resume_state=None):
    torch.manual_seed(seed);torch.set_num_threads(settings['threads'])
    env=EventEnvironment(p,settings['envs'],seed)
    ag=Agent(p,settings,algorithm,env.alpha_scale,env.value_scale)
    opt_actor=torch.optim.Adam(ag.actor.parameters(),lr=settings['actor_lr'])
    critic_params=list(ag.q1.parameters())+([] if ag.q2 is None else list(ag.q2.parameters()))
    opt_critic=torch.optim.Adam(critic_params,lr=settings['critic_lr'])
    history=[];steps=settings['transitions']//settings['envs']
    def record(row):
        if not all(np.isfinite(v) for v in row.values()):raise RuntimeError('Entraînement RL non fini')
        history.append(row);log(row)
    if algorithm=='ppo':
        if checkpoint_callback is not None or resume_state is not None:
            raise ValueError('Checkpoint training hooks currently support TD3/SAC only')
        horizon=settings['rollout'];total=0
        for epoch in range(steps//horizon):
            ys=[];zs=[];oldps=[];rewards=[];discounts=[];values=[]
            for _ in range(horizon):
                y=env.y.clone()
                with torch.no_grad():
                    x=ag.norm(y);a,lp,z=ag.actor.sample(x,active_sides(y,p));v=ag.q1(x)
                    _,r,d=env.step(a)
                ys.append(y);zs.append(z);oldps.append(lp);rewards.append(r);discounts.append(d);values.append(v)
            with torch.no_grad():bootstrap=ag.q1(ag.norm(env.y))
            advantage,returns=gae(torch.stack(rewards),torch.stack(discounts),torch.stack(values),bootstrap,settings['gae_lambda'])
            y=torch.cat(ys);z=torch.cat(zs);oldp=torch.cat(oldps);ret=returns.reshape(-1)
            adv=advantage.reshape(-1);adv=(adv-adv.mean())/adv.std().clamp_min(1e-6)
            stop=False;kl=0.
            for _ in range(settings['ppo_epochs']):
                for idx in torch.randperm(len(y)).split(settings['ppo_batch']):
                    x=ag.norm(y[idx]);mask=active_sides(y[idx],p)
                    mean,ls=ag.actor.distribution(x);lp=softplus_log_prob(mean,ls,z[idx],mask)
                    kl=float((oldp[idx]-lp).mean().detach())
                    if kl>settings['target_kl']:stop=True;break
                    loss_actor=-clipped_ppo_objective(lp,oldp[idx],adv[idx],settings['ppo_clip']).mean()
                    opt_actor.zero_grad();loss_actor.backward();torch.nn.utils.clip_grad_norm_(ag.actor.parameters(),.5);opt_actor.step()
                if stop:break
            # L'arrêt KL ne doit pas supprimer l'ajustement du critique.
            for _ in range(settings['ppo_epochs']):
                for idx in torch.randperm(len(y)).split(settings['ppo_batch']):
                    loss_critic=(ag.q1(ag.norm(y[idx]))-ret[idx]).square().mean()
                    opt_critic.zero_grad();loss_critic.backward();torch.nn.utils.clip_grad_norm_(critic_params,10.);opt_critic.step()
            total+=len(y)
            if epoch%8==0 or epoch==steps//horizon-1:
                record(dict(transitions=total,critic_loss=float(loss_critic.detach()),actor_loss=float(loss_actor.detach()),kl=kl))
        return ag,history
    buffer=ReplayBuffer(settings['buffer'],M.dim(p),seed)
    # Rewards and Q are divided by value_scale. Alpha is specified in the native
    # reward units, and must be divided by the SAME scale (v1 omitted this).
    log_alpha_native=torch.tensor(math.log(settings['sac_initial_alpha']),requires_grad=True)
    opt_alpha=torch.optim.Adam([log_alpha_native],lr=settings.get('sac_alpha_lr',settings['actor_lr']))
    first_step=0
    if resume_state is not None:
        if resume_state['training_seed']!=seed or resume_state['algorithm']!=algorithm:
            raise ValueError('Resume seed/algorithm mismatch')
        original=resume_state['settings']
        if any(original[k]!=v for k,v in settings.items() if k not in ('transitions','threads')):
            raise ValueError('Resume settings mismatch')
        ag.load(resume_state['agent'])
        for name in ('target_actor','t1','t2'):
            net=getattr(ag,name)
            if net is not None:net.load_state_dict(resume_state['targets'][name])
        opt_actor.load_state_dict(resume_state['opt_actor']);opt_critic.load_state_dict(resume_state['opt_critic'])
        with torch.no_grad():log_alpha_native.copy_(resume_state['log_alpha_native'])
        opt_alpha.load_state_dict(resume_state['opt_alpha'])
        env.y=resume_state['env_y'].clone();env.gen.set_state(resume_state['env_rng'])
        buffer.data=resume_state['buffer_data'];buffer.size=resume_state['buffer_size'];buffer.cursor=resume_state['buffer_cursor']
        buffer.rng.bit_generator.state=resume_state['buffer_rng']
        history=copy.deepcopy(resume_state['history']);first_step=resume_state['completed_steps']
        torch.set_rng_state(resume_state['torch_rng'])
    def snapshot(completed):
        return dict(algorithm=algorithm,training_seed=seed,settings=dict(settings),completed_steps=completed,
            agent=ag.state_dict(),targets={n:getattr(ag,n).state_dict() for n in ('target_actor','t1','t2') if getattr(ag,n) is not None},
            opt_actor=opt_actor.state_dict(),opt_critic=opt_critic.state_dict(),opt_alpha=opt_alpha.state_dict(),
            log_alpha_native=log_alpha_native.detach().clone(),env_y=env.y.clone(),env_rng=env.gen.get_state(),
            buffer_data=buffer.data,buffer_size=buffer.size,buffer_cursor=buffer.cursor,
            buffer_rng=buffer.rng.bit_generator.state,torch_rng=torch.get_rng_state(),history=history)
    for it in range(first_step,steps):
        y=env.y.clone()
        with torch.no_grad():
            mask=active_sides(y,p)
            if (it+1)*len(y)<=settings['start_steps']:
                a=torch.rand(len(y),2)*(2*(1+p.kappa*p.fee)+2)*mask
            else:
                a,_,_=ag.actor.sample(ag.norm(y),mask)
                if algorithm=='td3':a=(a+settings['exploration_noise']*torch.randn_like(a)).clamp_min(0)*mask
            yn,r,d=env.step(a);buffer.add(y,a,r,yn,d)
        if buffer.size<settings['start_steps']:continue
        y,a,r,yn,d=buffer.sample(settings['batch']);x=ag.norm(y);xn=ag.norm(yn);maskn=active_sides(yn,p)
        with torch.no_grad():
            if algorithm=='td3':
                an,_,_=ag.target_actor.sample(xn,maskn,True)
                noise=(torch.randn_like(an)*settings['target_noise']).clamp(-settings['noise_clip'],settings['noise_clip'])
                an=(an+noise).clamp_min(0)*maskn
                future=torch.minimum(ag.t1(xn,an),ag.t2(xn,an))
            else:
                an,lpn,_=ag.actor.sample(xn,maskn)
                future=torch.minimum(ag.t1(xn,an),ag.t2(xn,an))-normalized_temperature(log_alpha_native.exp(),ag.value_scale)*lpn
            target=r+d*future
        loss_critic=(ag.q1(x,a)-target).square().mean()+(ag.q2(x,a)-target).square().mean()
        opt_critic.zero_grad();loss_critic.backward();torch.nn.utils.clip_grad_norm_(critic_params,10.);opt_critic.step()
        loss_actor=torch.tensor(0.);alpha_loss=torch.tensor(0.)
        if algorithm=='sac' or it%settings['policy_delay']==0:
            for param in critic_params:param.requires_grad_(False)
            ac,lp,_=ag.actor.sample(x,active_sides(y,p))
            if algorithm=='td3':loss_actor=-ag.q1(x,ac).mean()
            else:loss_actor=(normalized_temperature(log_alpha_native.exp().detach(),ag.value_scale)*lp-torch.minimum(ag.q1(x,ac),ag.q2(x,ac))).mean()
            opt_actor.zero_grad();loss_actor.backward();torch.nn.utils.clip_grad_norm_(ag.actor.parameters(),10.);opt_actor.step()
            for param in critic_params:param.requires_grad_(True)
            if algorithm=='sac':
                target_entropy=-active_sides(y,p).sum(1)
                alpha_loss=-(log_alpha_native*(lp+target_entropy).detach()).mean()
                opt_alpha.zero_grad();alpha_loss.backward();opt_alpha.step()
                with torch.no_grad():log_alpha_native.clamp_(-10,1)
            if algorithm=='td3':polyak(ag.target_actor,ag.actor,settings['tau'])
            polyak(ag.t1,ag.q1,settings['tau']);polyak(ag.t2,ag.q2,settings['tau'])
        if it%256==0 or it==steps-1:
            record(dict(transitions=(it+1)*settings['envs'],critic_loss=float(loss_critic.detach()),
                        actor_loss=float(loss_actor.detach()),
                        alpha=float(normalized_temperature(log_alpha_native.exp().detach(),ag.value_scale)) if algorithm=='sac' else 0.,
                        alpha_native=float(log_alpha_native.exp().detach()) if algorithm=='sac' else 0.,
                        gradient_updates=it+1-settings['start_steps']//settings['envs']+1))
        if checkpoint_callback is not None:
            if checkpoint_callback(ag,(it+1)*settings['envs'],lambda:snapshot(it+1)):
                break
    return ag,history
