from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from cjr import model as M, discrete_value as DV, rl_algorithms as RL, value_agent as VA
from cjr import replay


def params():
    return M.Params(mu=[.7,.3],a=[[[.2,.03]],[[.02,.15]]],betas=[1.],
                    eps_plus=.1,eps_minus=.2,zeta=.7,A=1.,kappa=1.3,
                    sigma2=.8,gamma=.1,r=.2,K=2,fee=.4)


def test_sac_temperature_preserves_objective_under_reward_unit_changes():
    # Both Q and the entropy bonus must scale by the same factor, hence the
    # actor objective and its gradient change only by that common factor.
    x=torch.tensor([.2,.7],requires_grad=True)
    alpha=.02
    native=(x.square()+alpha*x.log()).sum()
    scale=19.888665394135682
    normalized=(x.square()/scale+RL.normalized_temperature(alpha,scale)*x.log()).sum()
    g1=torch.autograd.grad(native,x,retain_graph=True)[0]
    g2=torch.autograd.grad(normalized,x)[0]
    assert torch.allclose(g1/scale,g2)
    assert normalized.item()==pytest.approx(native.item()/scale)
    with pytest.raises(ValueError):RL.normalized_temperature(alpha,0.)


def test_vector_inventory_targets_equal_scalar_bellman_with_exact_side_average():
    torch.manual_seed(1);p=params();T=p.tensors('cpu')
    ag=DV.Agent(p,{'agent':{'hidden':8}},'cpu',1.,7.)
    flow=torch.tensor([[0.,.4,1.,2.],[0.,-.3,2.,.5]])
    tau=torch.tensor([.2,.7]);nq=2*p.K+1
    actual=DV.inventory_targets(ag,flow,tau,p,T).reshape(-1)
    y=flow.repeat_interleave(nq,0);y[:,0]=torch.arange(-p.K,p.K+1).repeat(2)
    yd=M.decay(y,tau.repeat_interleave(nq),p,T)
    terms=VA.side_terms(ag,yd,p,ag.target)
    theta=torch.stack([v+m*h*g for v,g,h,m in terms],1)
    expected=M.running_reward(y,tau.repeat_interleave(nq),p)/7+torch.exp(-p.r*tau.repeat_interleave(nq))*DV.conditional_side_mean(theta,yd,p,T)
    assert torch.allclose(actual,expected,atol=2e-7)
    # La moyenne exacte a une variance nulle conditionnellement à theta et lambda.
    probability=M.intensities(yd,p,T);probability/=probability.sum(1,keepdim=True)
    assert torch.allclose(DV.conditional_side_mean(theta,yd,p,T),(theta*probability).sum(1))


def test_inventory_head_is_discrete_not_an_interpolated_q_input():
    p=params();ag=DV.Agent(p,{'agent':{'hidden':8}},'cpu',1.,1.)
    y=torch.zeros(5,M.dim(p));y[:,0]=torch.arange(-2,3)
    values=ag.all_values(y)
    assert torch.equal(values,values[:1].expand_as(values))
    assert torch.allclose(ag.V(y),values[0])


def test_discrete_grid_inference_preserves_scalar_quotes():
    p=params();ag=DV.Agent(p,{'agent':{'hidden':8}},'cpu',1.,7.)
    alpha=np.array([.2,-.1]);E=np.array([[1.,2.],[3.,1.]])
    y=torch.tensor(np.column_stack((np.tile(np.arange(-p.K,p.K+1),2),
        np.repeat(alpha,5),np.repeat(E,5,axis=0))),dtype=torch.float32)
    with torch.no_grad():expected=ag.deltas(y).view(2,5,2).numpy()
    ask,bid=replay.delta_tables(ag.deltas,alpha,E,p.K,'cpu')
    assert np.allclose(ask,expected[:,:,0],atol=1e-6)
    assert np.allclose(bid,expected[:,:,1],atol=1e-6)


def test_softplus_gaussian_density_integrates_to_one_and_handles_boundaries():
    mean,ls=.3,-.5
    def density(a):
        z=a+math.log(-math.expm1(-a))
        lp=RL.softplus_log_prob(torch.tensor([[mean]],dtype=torch.float64),torch.tensor([[ls]],dtype=torch.float64),torch.tensor([[z]],dtype=torch.float64),torch.ones(1,1))
        return math.exp(lp.item())
    assert quad(density,1e-12,20.,epsabs=1e-8)[0]==pytest.approx(1.,abs=1e-7)
    p=params();y=torch.zeros(3,M.dim(p));y[:,0]=torch.tensor([-2.,0.,2.])
    action=torch.tensor([[1.,2.],[1000.,0.],[3.,4.]])
    d=RL.action_deltas(action,y,p)
    assert torch.isinf(d[0,0]) and torch.isinf(d[-1,1])
    assert d[1,0]>700 and d[1,1]==0


def test_variable_time_gae_and_ppo_clipping():
    r=torch.tensor([[1.],[2.]]);discount=torch.tensor([[.5],[.25]])
    adv,ret=RL.gae(r,discount,torch.zeros_like(r),torch.tensor([4.]),1.)
    assert ret[:,0].tolist()==pytest.approx([2.5,3.])
    result=RL.clipped_ppo_objective(torch.log(torch.tensor([2.,2.])),torch.zeros(2),torch.tensor([1.,-1.]),.2)
    assert result.tolist()==pytest.approx([1.2,-2.])


def test_event_environment_reward_time_discount_and_inventory_are_causal(monkeypatch):
    p=replace(params(),a=[[[0.,0.]],[[0.,0.]]],eps_plus=0.,eps_minus=0.,gamma=0.)
    env=RL.EventEnvironment(p,20000,11);env.y.zero_()
    monkeypatch.setattr(M,'next_event',lambda y,p,T,gen:(torch.ones(len(y))*.7,torch.zeros(len(y),dtype=torch.long)))
    action=torch.tensor([1.,2.]).repeat(len(env.y),1)
    yn,r,d=env.step(action)
    expected=(.7*math.exp(-1)*(1/p.kappa-p.fee)+.3*math.exp(-2)*(2/p.kappa-p.fee))/env.value_scale
    second=(.7*math.exp(-1)*((1/p.kappa-p.fee)/env.value_scale)**2
            +.3*math.exp(-2)*((2/p.kappa-p.fee)/env.value_scale)**2)
    assert r.mean().item()==pytest.approx(expected,abs=5*math.sqrt((second-expected**2)/len(r)))
    assert torch.allclose(d,torch.full_like(d,math.exp(-p.r*.7)))
    assert (yn[:,0].abs()<=1).all()
    assert (yn[:,2:].sum(1)>0).all()  # Les ordres non exécutés influencent aussi Hawkes.


@pytest.mark.parametrize('algorithm',['td3','sac','ppo'])
def test_algorithm_updates_and_roundtrips_real_checkpoint(algorithm):
    p=params()
    s=dict(hidden=8,threads=1,envs=8,transitions=64,batch=8,buffer=64,start_steps=8,
           actor_lr=.001,critic_lr=.001,tau=.05,exploration_noise=.2,target_noise=.2,
           noise_clip=.5,policy_delay=2,sac_initial_alpha=.02,rollout=4,ppo_epochs=2,
           ppo_batch=16,ppo_clip=.2,target_kl=.03,gae_lambda=.95)
    ag,h=RL.train(p,s,algorithm,11,log=lambda _:None)
    assert h and all(np.isfinite(row['critic_loss']) for row in h)
    sd=ag.state_dict();loaded=RL.Agent(p,s,algorithm,sd['alpha_scale'],sd['value_scale']);loaded.load(sd)
    y=torch.zeros(3,M.dim(p));y[:,0]=torch.tensor([-2.,0.,2.])
    assert torch.allclose(ag.deltas(y),loaded.deltas(y))
    assert all(torch.isfinite(x).all() for x in ag.actor.parameters())
    with pytest.raises(ValueError):loaded.load({**sd,'algorithm':'wrong'})
