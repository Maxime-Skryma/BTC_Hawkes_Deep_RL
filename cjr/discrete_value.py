"""Critique à têtes d'inventaire et Bellman moyenné sur les deux côtés.

L'inventaire ne rentre pas dans le réseau : chaque sortie représente un entier q.
L'excitation est exogène à nos fills ; un lot d'états de flux couvre tous les q.
"""
import copy
import numpy as np
import torch
from torch import nn
from cjr import model as M
from cjr.actor_critic import Normalizer, mlp
from cjr.value_agent import greedy_deltas

ALGORITHM_VERSION='cjr-discrete-value-rb-v6'


class InventoryHead(nn.Module):
    def __init__(self,n_in,K,hidden=64):
        super().__init__();self.K=K
        self.net=mlp(n_in,2*K+1,hidden)

    def forward(self,x):return self.net(x)


class Agent:
    def __init__(self,p,cfg,device,alpha_scale,value_scale):
        self.p,self.device,self.value_scale=p,device,value_scale
        self.norm=Normalizer(p,alpha_scale,cfg['agent'].get('observation','full'))
        self.critic=InventoryHead(M.dim(p)-1,p.K,cfg['agent']['hidden']).to(device)
        self.target=copy.deepcopy(self.critic)

    def all_values(self,y,net=None):
        return (self.critic if net is None else net)(self.norm(y)[:,1:])

    def V(self,y,net=None):
        return self.all_values(y,net).gather(1,(y[:,0].long()+self.p.K)[:,None])[:,0]

    def deltas(self,y):return greedy_deltas(self,y,self.p)

    @torch.no_grad()
    def grid_tables(self,alpha,E,K,device,chunk=20000):
        if K!=self.p.K:raise ValueError('Inventaire de grille incompatible')
        n=len(alpha);tables=[np.empty((n,2*K+1),np.float32) for _ in range(2)]
        index=torch.arange(2*K+1,device=device);qs=index-K
        for start in range(0,n,chunk):
            stop=min(start+chunk,n)
            y=torch.tensor(np.column_stack((np.zeros(stop-start),alpha[start:stop],E[start:stop])),dtype=torch.float32,device=device)
            for side,shift in ((0,-1),(1,1)):
                post=M.jump(y,torch.full((len(y),),side,device=device),self.p)
                v=self.all_values(post)
                delta=(1/self.p.kappa+self.p.fee-self.value_scale*(v[:,(index+shift).clamp(0,2*K)]-v)).clamp_min(0)
                allowed=(qs>-K) if side==0 else (qs<K)
                tables[side][start:stop]=torch.where(allowed,delta,torch.inf).cpu().numpy()
        return tuple(tables)

    def state_dict(self):
        return dict(algorithm_version=ALGORITHM_VERSION,params_fingerprint=self.p.fingerprint(),
                    critic=self.critic.state_dict(),alpha_scale=self.norm.alpha_scale,
                    value_scale=self.value_scale,hidden=self.critic.net[0].out_features,
                    observation=self.norm.observation)

    def load(self,sd):
        if (sd.get('algorithm_version')!=ALGORITHM_VERSION or sd.get('params_fingerprint')!=self.p.fingerprint()
            or sd.get('observation','full')!=self.norm.observation):
            raise ValueError('Checkpoint critique discret incompatible')
        self.critic.load_state_dict(sd['critic']);self.target.load_state_dict(sd['critic'])


def conditional_side_mean(theta,y,p,T):
    lam=M.intensities(y,p,T)
    return (theta*lam/lam.sum(1,keepdim=True)).sum(1)


def inventory_targets(ag,flow,tau,p,T,poisson=False):
    """Bellman des 21 q : seulement deux passes du réseau cible par état de flux."""
    yd=flow if poisson else M.decay(flow,tau,p,T)
    qs=torch.arange(-p.K,p.K+1,device=flow.device,dtype=flow.dtype)
    index=torch.arange(2*p.K+1,device=flow.device)
    theta=[]
    for s,shift in ((0,-1),(1,1)):
        post=M.jump(yd,torch.full((len(flow),),s,device=flow.device),p)
        v0=ag.all_values(post,ag.target)
        v1=v0[:,(index+shift).clamp(0,2*p.K)]
        allowed=(qs>-p.K) if s==0 else (qs<p.K)
        delta=(1/p.kappa+p.fee-ag.value_scale*(v1-v0)).clamp_min(0)
        gain=p.A*torch.exp(-p.kappa*delta)*((delta-p.fee)/ag.value_scale+v1-v0)
        theta.append(v0+allowed*gain)
    theta=torch.stack(theta,2)
    if poisson:
        return (-p.psi(qs)[None,:]/ag.value_scale+(theta*T['mu']).sum(2))/(p.r+T['mu'].sum())
    lam=M.intensities(yd,p,T);weights=lam/lam.sum(1,keepdim=True)
    running=(qs[None,:]*flow[:,1,None]*(-torch.expm1(-(p.r+p.zeta)*tau[:,None]))/(p.r+p.zeta)
             -p.psi(qs)[None,:]*(-torch.expm1(-p.r*tau[:,None]))/p.r)/ag.value_scale
    return running+torch.exp(-p.r*tau[:,None])*(theta*weights[:,None,:]).sum(2)


def train(p,cfg,device,log=print):
    a=cfg['agent'];torch.manual_seed(a['seed'])
    gen=torch.Generator(device=device).manual_seed(a['seed']);T=p.tensors(device)
    B=a['batch'];nq=2*p.K+1
    flow=torch.zeros(B,M.dim(p),device=device)
    for _ in range(300):
        tau,side=M.next_event(flow,p,T,gen);flow=M.jump(M.decay(flow,tau,p,T),side,p)
    alpha_scale=float(flow[:,1].std().clamp_min(1e-6)) if (p.eps_plus or p.eps_minus) else 1.
    rates=np.linalg.solve(np.eye(2)-(np.asarray(p.a)/np.asarray(p.betas)[None,:,None]).sum(1),p.mu)
    scale=max(float(p.A/p.kappa*np.exp(-1)*rates.sum()/p.r),1e-6)
    ag=Agent(p,cfg,device,alpha_scale,scale)
    opt=torch.optim.Adam(ag.critic.parameters(),lr=a['lr_critic'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a['iterations'],eta_min=.05*a['lr_critic'])
    exact_poisson=not np.any(p.a) and p.eps_plus==0 and p.eps_minus==0
    history=[]
    for it in range(a['iterations']):
        with torch.no_grad():
            if exact_poisson:tau=torch.zeros(B,device=device)
            else:tau,side=M.next_event(flow,p,T,gen)
            target=inventory_targets(ag,flow,tau,p,T,exact_poisson).reshape(-1)
        # Une seule passe du critique pour les 21 inventaires de chaque état de flux.
        prediction=ag.all_values(flow).reshape(-1)
        loss=(prediction-target).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Critique discret non fini')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ag.critic.parameters(),10.);opt.step();scheduler.step()
        with torch.no_grad():
            for t,s in zip(ag.target.parameters(),ag.critic.parameters()):t.mul_(1-a['polyak']).add_(a['polyak']*s)
            if not exact_poisson:flow=M.jump(M.decay(flow,tau,p,T),side,p)
        if it%256==0 or it==a['iterations']-1:
            row=dict(it=it,loss_critic=float(loss.detach()),flow_transitions=(it+1)*B,
                     inventory_targets=(it+1)*B*nq)
            history.append(row);log(row)
    return ag,history
