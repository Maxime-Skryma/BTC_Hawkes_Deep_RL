"""Deep RL par valeur, sans acteur : Bellman optimal et cotations analytiques."""
import copy
import numpy as np
import torch
from cjr import model as M
from cjr.actor_critic import Normalizer, mlp

ALGORITHM_VERSION = 'cjr-event-value-v5'


class Agent:
    def __init__(self,p,cfg,device,alpha_scale,value_scale):
        self.p,self.device,self.value_scale=p,device,value_scale
        self.norm=Normalizer(p,alpha_scale)
        self.critic=mlp(M.dim(p),1,cfg['agent']['hidden']).to(device)
        self.target=copy.deepcopy(self.critic)

    def V(self,y,net=None):
        return (self.critic if net is None else net)(self.norm(y))[:,0]

    def deltas(self,y,net=None):
        return greedy_deltas(self,y,self.p,net)

    def log_probs(self,y):
        return -self.p.kappa*self.deltas(y)

    def probs(self,y):
        return self.log_probs(y).exp()

    def state_dict(self):
        return dict(algorithm_version=ALGORITHM_VERSION,params_fingerprint=self.p.fingerprint(),
                    critic=self.critic.state_dict(),alpha_scale=self.norm.alpha_scale,
                    value_scale=self.value_scale)

    def load(self,sd):
        if sd.get('algorithm_version')!=ALGORITHM_VERSION or sd.get('params_fingerprint')!=self.p.fingerprint() or 'actor' in sd:
            raise ValueError('Checkpoint ancien ou paramètres différents : réentraînement requis')
        self.critic.load_state_dict(sd['critic'])
        self.target.load_state_dict(sd['critic'])


def continuation_values(agent,y,p,net):
    rows=[]
    for s in (0,1):
        side=torch.full((len(y),),s,device=y.device)
        ys=M.jump(y,side,p)
        yf=ys.clone();yf[:,0]=(yf[:,0]+(-1. if s==0 else 1.)).clamp(-p.K,p.K)
        allowed=(y[:,0]>-p.K) if s==0 else (y[:,0]<p.K)
        rows.append((agent.V(ys,net),agent.V(yf,net),allowed))
    return rows


def greedy_deltas(agent,y,p,net=None):
    return torch.stack([torch.where(m,(1/p.kappa+p.fee-agent.value_scale*(v1-v0)).clamp_min(0),torch.inf)
                        for v0,v1,m in continuation_values(agent,y,p,net)],1)


def side_terms(agent,yd,p,net):
    out=[]
    for v0,v1,allowed in continuation_values(agent,yd,p,net):
        # Valeur finie du côté interdit : pas de inf*0 dans Bellman.
        delta=(1/p.kappa+p.fee-agent.value_scale*(v1-v0)).clamp_min(0)
        h=p.A*torch.exp(-p.kappa*delta)
        out.append((v0,(delta-p.fee)/agent.value_scale+v1-v0,h,allowed.to(yd.dtype)))
    return out


def train(p,cfg,device,log=print):
    a=cfg['agent'];T=p.tensors(device)
    gen=torch.Generator(device=device).manual_seed(a['seed']);torch.manual_seed(a['seed'])
    exact_poisson=not np.any(p.a) and p.eps_plus==0 and p.eps_minus==0
    B=2*p.K+1 if exact_poisson else a['batch']
    y=torch.zeros(B,M.dim(p),device=device)
    y[:,0]=(torch.arange(-p.K,p.K+1,device=device).float() if exact_poisson else
            torch.randint(-p.K,p.K+1,(B,),device=device,generator=gen).float())
    for _ in range(0 if exact_poisson else 300):
        tau,side=M.next_event(y,p,T,gen);y=M.jump(M.decay(y,tau,p,T),side,p)
    lam_bar=float(M.intensities(y,p,T).sum(1).mean())
    alpha_scale=float(y[:,1].std().clamp_min(1e-6)) if (p.eps_plus or p.eps_minus) else 1.
    value_scale=p.A/p.kappa*np.exp(-1)*lam_bar/p.r
    agent=Agent(p,cfg,device,alpha_scale,value_scale)
    opt=torch.optim.Adam(agent.critic.parameters(),lr=a['lr_critic'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a['iterations'],eta_min=.05*a['lr_critic'])
    hist=[]
    for it in range(a['iterations']):
        if exact_poisson:yd=y
        else:
            tau,side=M.next_event(y,p,T,gen);yd=M.decay(y,tau,p,T)
            R=M.running_reward(y,tau,p)/value_scale
        with torch.no_grad():
            terms=side_terms(agent,yd,p,agent.target)
            theta=torch.stack([v0+m*h*g for v0,g,h,m in terms],1)
            if exact_poisson:
                target=(-p.psi(y[:,0])/value_scale+(theta*T['mu']).sum(1))/(p.r+T['mu'].sum())
            else:target=R+torch.exp(-p.r*tau)*theta.gather(1,side[:,None])[:,0]
        loss=((agent.V(y)-target)**2).mean()
        if not torch.isfinite(loss):raise RuntimeError('Critique non fini')
        opt.zero_grad();loss.backward();opt.step();sched.step()
        with torch.no_grad():
            for t,s in zip(agent.target.parameters(),agent.critic.parameters()):t.mul_(1-a['polyak']).add_(a['polyak']*s)
            if not exact_poisson:
                # Politique du critique courant, sans approximation par un acteur.
                d=agent.deltas(yd);chosen=d.gather(1,side[:,None])[:,0]
                fill=torch.rand(B,device=device,generator=gen)<p.A*torch.exp(-p.kappa*chosen)
                y=M.jump(yd,side,p)
                y[:,0]+=torch.where(fill,torch.where(side==0,-1.,1.),0.)
                reset=torch.rand(B,device=device,generator=gen)<a['reset_q']
                y[:,0]=torch.where(reset,torch.randint(-p.K,p.K+1,(B,),device=device,generator=gen).float(),y[:,0])
        if it%1000==0 or it==a['iterations']-1:
            hist.append(dict(it=it,loss_critic=loss.item(),V_mean=float(agent.V(y).mean().detach()*value_scale)))
            log(hist[-1])
    return agent,hist


def load_saved(p,cfg,device,sd):
    """Lecture explicite des anciens agents pour reproduire leurs expériences."""
    if sd.get('algorithm_version')=='cjr-event-ac-v4':
        from cjr.actor_critic import Agent as LegacyAgent
        agent=LegacyAgent(p,cfg,device,sd['alpha_scale'],sd['value_scale'])
    else:agent=Agent(p,cfg,device,sd['alpha_scale'],sd['value_scale'])
    agent.load(sd)
    return agent


def move_saved(agent,device):
    for name in ('actor','critic','target'):
        module=getattr(agent,name,None)
        if module is not None:module.to(device)
    agent.device=device
    return agent
