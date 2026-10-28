from dataclasses import replace
import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar
from cjr import agent as AG, actor_critic as LEGACY, model as M


def params():
    return M.Params(mu=[.7,.3],a=[[[.24,.05]],[[.04,.18]]],betas=[1.5],
                    eps_plus=.3,eps_minus=.2,zeta=.7,A=1.,kappa=1.3,
                    sigma2=.8,gamma=.1,r=.2,K=3,fee=.4)


def test_value_only_post_jump_policy_matches_independent_scalar_maximization():
    p=params()
    class Oracle:
        value_scale=7.
        def V(self,y,net=None):
            q,al,e=M.unpack(y,p)
            return (.7*q*q+.2*q*al+.1*(e*e).sum((1,2)))/self.value_scale
    y=torch.tensor([[1.,.4,1.,2.]],dtype=torch.float64)
    oracle=Oracle();d=AG.greedy_deltas(oracle,y,p).numpy()[0]
    terms=AG.side_terms(oracle,y,p,None)
    for s in (0,1):
        # Oracle indépendant en NumPy, avec alpha et mémoire après le saut.
        q=1.;al=.4+(p.eps_plus if s==0 else -p.eps_minus)
        e=np.array([[1.,2.]]);e[:,s]+=1
        def value(q):return .7*q*q+.2*q*al+.1*np.sum(e*e)
        diff=value(q+(-1 if s==0 else 1))-value(q)
        objective=lambda delta:np.exp(-p.kappa*delta)*(delta-p.fee+diff)
        optimum=minimize_scalar(lambda delta:-objective(delta),bounds=(0,100),method='bounded')
        best=max(objective(0),objective(optimum.x))
        assert objective(d[s])==pytest.approx(best,abs=1e-11)
        v0,g,h,m=terms[s]
        assert v0.item()==pytest.approx(value(q)/oracle.value_scale)
        # Les amplitudes de saut de M.jump sont construites en float32.
        assert (h*g*oracle.value_scale).item()==pytest.approx(best,abs=1e-8)
        assert m.item()==1


def test_value_policy_boundary_and_no_artificial_distance_cap():
    p=params()
    class Linear:
        value_scale=1.
        def V(self,y,net=None):return 1000*y[:,0]
    y=torch.zeros(7,M.dim(p));y[:,0]=torch.arange(-3,4)
    d=AG.greedy_deltas(Linear(),y,p)
    assert torch.isinf(d[0,0]) and torch.isinf(d[-1,1])
    assert d[3,0]>1000 and d[3,1]==0
    terms=AG.side_terms(Linear(),y,p,None)
    assert all(torch.isfinite(x).all() for term in terms for x in term)


def test_checkpoint_has_no_actor_and_cannot_silently_accept_legacy_weights():
    p=params();cfg={'agent':{'hidden':8}}
    ag=AG.Agent(p,cfg,'cpu',1.,1.)
    assert not hasattr(ag,'actor') and 'actor' not in ag.state_dict()
    loaded=AG.Agent(p,cfg,'cpu',1.,1.);loaded.load(ag.state_dict())
    for x,z in zip(loaded.critic.parameters(),loaded.target.parameters()):assert torch.equal(x,z)
    legacy=LEGACY.Agent(p,cfg,'cpu',1.,1.)
    with pytest.raises(ValueError,match='Checkpoint'):ag.load(legacy.state_dict())
    with pytest.raises(ValueError,match='Checkpoint'):
        AG.Agent(replace(p,fee=2.),cfg,'cpu',1.,1.).load(ag.state_dict())
    assert hasattr(AG.load_saved(p,cfg,'cpu',legacy.state_dict()),'actor')


def test_bellman_greedy_policy_uses_target_values_when_target_is_requested():
    p=params();ag=AG.Agent(p,{'agent':{'hidden':8}},'cpu',1.,7.)
    class LinearValue(torch.nn.Module):
        def __init__(self,slope):super().__init__();self.slope=slope
        def forward(self,x):return self.slope*x[:,:1]
    ag.critic=LinearValue(0.)
    ag.target=LinearValue(5*p.K/ag.value_scale)  # valeur non normalisée = 5q
    y=torch.zeros(1,M.dim(p))
    base=1/p.kappa+p.fee
    assert ag.deltas(y)[0].tolist()==pytest.approx([base,base])
    assert ag.deltas(y,ag.target)[0].tolist()==pytest.approx([base+5,0.])
    terms=AG.side_terms(ag,y,p,ag.target)
    assert terms[0][2].item()==pytest.approx(np.exp(-p.kappa*(base+5)))
    assert terms[1][2].item()==1.


@pytest.mark.parametrize('poisson',[True,False])
def test_training_updates_critic_without_actor_or_nonfinite_boundaries(poisson):
    p=params()
    if poisson:p=replace(p,a=np.zeros_like(p.a).tolist(),eps_plus=0.,eps_minus=0.)
    cfg={'agent':dict(hidden=8,batch=16,iterations=4,seed=11,lr_critic=.001,polyak=.05,reset_q=.1)}
    ag,hist=AG.train(p,cfg,'cpu',log=lambda _:None)
    assert len(hist)==2 and all(np.isfinite(h['loss_critic']) for h in hist)
    assert not hasattr(ag,'actor')
    assert all(torch.isfinite(v).all() for v in ag.critic.parameters())
