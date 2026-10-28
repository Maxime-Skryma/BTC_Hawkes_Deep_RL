"""Tests mathématiques : oracles indépendants, pas de données de marché ni de P&L sélectionné."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import kstest
from cjr import model as M, vi, reference, agent as AG
from cjr import actor_critic as LEGACY


def params():
    return M.Params(mu=[.7, .3], a=[[[.24,.05],[.015,.01]],[[.04,.18],[.005,.02]]],
                    betas=[1.5,.25], eps_plus=.3, eps_minus=.2, zeta=.7,
                    A=1., kappa=1.3, sigma2=.8, gamma=.1, r=.2, K=3, fee=.4)


@pytest.mark.parametrize('fee', [0.,1.5,3.,7.5])
def test_vi_against_independent_howard_and_linear_policy_evaluation(fee):
    p=replace(params(), fee=fee)
    V,da,db=vi.poisson_vi(.7,.3,p,tol=1e-12)
    W,xa,xb=reference.poisson_policy_iteration(.7,.3,p)
    assert np.max(np.abs(V-W)) < 1e-8
    assert np.max(np.abs(V-reference.policy_value(.7,.3,p,da,db))) < 1e-8
    for x,z in ((da,xa),(db,xb)):
        valid=np.isfinite(x)
        assert np.max(np.abs(x[valid]-z[valid])) < 2e-6


@pytest.mark.parametrize('field,value', [('A',1.01),('r',0.),('kappa',0.),('mu',[0.,1.]),
                                        ('betas',[-1.,.2]),('K',1.5),('sigma2',-1.)])
def test_invalid_models_fail(field,value):
    with pytest.raises(ValueError): replace(params(), **{field:value})


def test_nonstationary_model_fails():
    with pytest.raises(ValueError,match='stationnaire'):
        replace(params(),a=(np.asarray(params().a)*100).tolist())


def test_discounted_running_reward_independent_quadrature():
    p=params(); y=torch.tensor([[2.,-.6,1.,2.,3.,4.]],dtype=torch.float64)
    for t in [1e-10,.01,1.,100.]:
        exact=quad(lambda u: np.exp(-p.r*u)*(2*(-.6)*np.exp(-p.zeta*u)-p.psi(2)),0,t)[0]
        got=M.running_reward(y,torch.tensor([t],dtype=torch.float64),p).item()
        assert got == pytest.approx(exact,rel=1e-11,abs=1e-14)


def test_hawkes_next_event_compensator_and_conditional_side():
    p=params(); torch.set_num_threads(2)
    n=40000; y=torch.tensor([[1.,.4,1.,2.,3.,4.]],dtype=torch.float64).repeat(n,1)
    T={k:v.double() for k,v in p.tensors('cpu').items()}
    tau,side=M.next_event(y,p,T,torch.Generator().manual_seed(112))
    ts=tau.numpy(); E=y[0,2:].numpy().reshape(p.C,2)
    z=np.array([reference.integrated_hazard(t,p.mu,p.a,p.betas,E) for t in ts])
    # DKW tol conservatrice fixée avant exécution ; teste la loi, pas la moyenne seule.
    assert kstest(z,'expon').statistic < .015
    et=E[None]*np.exp(-ts[:,None,None]*np.asarray(p.betas)[None,:,None])
    lam=np.asarray(p.mu)[None]+np.einsum('sck,nck->ns',np.asarray(p.a),et)
    prob=lam[:,0]/lam.sum(1)
    for idx in np.array_split(np.argsort(ts),5):
        residual=(side.numpy()[idx]==0)-prob[idx]
        se=np.sqrt(np.sum(prob[idx]*(1-prob[idx])))/len(idx)
        assert abs(residual.mean()) < 5*se


def test_complete_event_bellman_against_independent_quadrature():
    p=params(); n=60000; torch.set_num_threads(2)
    y=torch.tensor([[1.,.4,1.,2.,3.,4.]],dtype=torch.float64).repeat(n,1)
    T={k:v.double() for k,v in p.tensors('cpu').items()}
    def val(q,al,e): return .7*q*q + .2*q*al + .1*np.sum(e*e)
    def policy(q,al,e): return (1.1+.1*al, .9-.1*al)
    exact=reference.event_bellman((1.,.4,np.array([[1.,2.],[3.,4.]])),p,val,policy)
    tau,s=M.next_event(y,p,T,torch.Generator().manual_seed(99))
    yd=M.decay(y,tau,p,T); ys=M.jump(yd,s,p)
    q,al,e=M.unpack(ys,p)
    v0=.7*q*q+.2*q*al+.1*(e*e).sum((1,2))
    qf=q+torch.where(s==0,-1.,1.)
    vf=.7*qf*qf+.2*qf*al+.1*(e*e).sum((1,2))
    delta=torch.where(s==0,1.1+.1*yd[:,1],.9-.1*yd[:,1])
    target=M.running_reward(y,tau,p)+torch.exp(-p.r*tau)*(v0+M.fill_prob(delta,p)*(delta-p.fee+vf-v0))
    assert abs(target.mean().item()-exact)<5*target.std().item()/np.sqrt(n)


def test_actor_no_artificial_upper_distance_and_checkpoint_identity():
    p=params(); cfg={'agent':{'hidden':8}}
    ag=LEGACY.Agent(p,cfg,'cpu',1.,1.)
    with torch.no_grad():
        for x in ag.actor.parameters(): x.zero_()
        ag.actor[-1].bias.fill_(100.)
    y=torch.zeros(1,M.dim(p))
    assert ag.deltas(y)[0,0].item() == pytest.approx(100/p.kappa)
    ag.load(ag.state_dict())
    other=LEGACY.Agent(replace(p,fee=2.),cfg,'cpu',1.,1.)
    with pytest.raises(ValueError,match='Checkpoint'): other.load(ag.state_dict())
    with pytest.raises(ValueError,match='Checkpoint'): ag.load({})


def test_production_td_side_terms_use_post_jump_value_for_fill_and_no_fill():
    p=params(); scale=7.
    class Oracle:
        value_scale=scale
        def log_probs(self,y):
            return torch.log(torch.tensor([.4,.7],dtype=y.dtype)).repeat(len(y),1)
        def V(self,y,net=None):
            q,al,e=M.unpack(y,p)
            return (.7*q*q+.2*q*al+.1*(e*e).sum((1,2)))/scale
    y=torch.tensor([[1.,.4,1.,2.,3.,4.]],dtype=torch.float64)
    terms=LEGACY.side_terms(Oracle(),y,p,None)
    for s,(v0,g,h,m) in enumerate(terms):
        e=np.array([[1.,2.],[3.,4.]])
        e[:,s]+=1
        al=.4+(p.eps_plus if s==0 else -p.eps_minus)
        qf=1+(-1 if s==0 else 1)
        base=.7+.2*al+.1*(e*e).sum()
        filled=.7*qf*qf+.2*qf*al+.1*(e*e).sum()
        prob=[.4,.7][s]
        assert v0.item()==pytest.approx(base/scale)
        assert g.item()==pytest.approx((-np.log(prob)/p.kappa-p.fee+filled-base)/scale)
        assert h.item()==pytest.approx(prob)
        assert m.item()==1


def test_hawkes_counts_against_exact_transient_first_moments():
    from scipy.linalg import expm
    p=params(); C=p.C; dim=2*C
    # Système indépendant : dE_ck/dt=-beta_c E_ck+lambda_k ; dC_k/dt=lambda_k.
    A=np.asarray(p.a).reshape(2,dim)
    B=np.zeros((dim,2))
    for c in range(C): B[2*c:2*c+2]=np.eye(2)
    H=np.zeros((dim+3,dim+3))
    H[:dim,:dim]=B@A-np.diag(np.repeat(p.betas,2))
    H[:dim,-1]=B@p.mu
    H[dim:dim+2,:dim]=A
    H[dim:dim+2,-1]=p.mu
    init=np.zeros(dim+3); init[-1]=1
    horizon=30.
    expected=(expm(H*horizon)@init)[dim:dim+2]
    n=8192; torch.set_num_threads(2)
    y=torch.zeros(n,M.dim(p),dtype=torch.float64)
    T={k:v.double() for k,v in p.tensors('cpu').items()}
    gen=torch.Generator().manual_seed(808)
    clock=torch.zeros(n,dtype=torch.float64)
    counts=torch.zeros(n,2,dtype=torch.float64)
    while bool((clock<horizon).any()):
        tau,side=M.next_event(y,p,T,gen)
        hit=(clock+tau<=horizon)&(clock<horizon)
        counts[torch.arange(n),side]+=hit
        y=M.jump(M.decay(y,tau,p,T),side,p)
        clock+=tau
    for s in (0,1):
        assert abs(counts[:,s].mean().item()-expected[s]) < 5*counts[:,s].std().item()/np.sqrt(n)


def test_alpha_conditional_design_includes_future_order_flow():
    from cjr.calibrate import alpha_moment_map
    # Poisson : m_Z_s(t)=z0 exp(-ζt)+mu_s(1-exp(-ζt))/ζ.
    mu=np.array([.7,.3]); zeta=.8; H=10.
    theta=np.column_stack([mu,np.zeros((2,2))])
    F=alpha_moment_map(theta,[1.],zeta,H)
    state=np.array([2.,3.,4.,5.,0.,0.,1.])
    g=-np.expm1(-zeta*H)/zeta
    expected=state[2:4]*g+mu/zeta*(H-g)
    assert F@state==pytest.approx(expected,rel=1e-12)
    # Le terme futur est strictement positif même si l'état initial est nul.
    assert np.all(F[:,-1]>0)


def test_realized_alpha_integral_against_direct_sum():
    import pandas as pd
    from cjr.calibrate import realized_alpha_integral
    orders=pd.DataFrame({'t':[.1,.8,1.,2.], 'side':[1,-1,1,-1]})
    al=dict(zeta=.7,eps_plus=.3,eps_minus=.2)
    grid=np.array([0.,.1,.5,1.,3.,10.])
    result=realized_alpha_integral(orders,al,grid)
    for i,t in enumerate(grid):
        expected=sum((.3 if s==1 else -.2)*(-np.expm1(-.7*(t-u)))/.7
                     for u,s in zip(orders.t,orders.side) if u<t)
        assert result[i]==pytest.approx(expected,abs=1e-14)


def test_received_batches_do_not_reveal_later_fills():
    import pandas as pd
    from cjr.data import aggregate_orders
    tr=pd.DataFrame({'timestamp':[10,10,11], 'local_timestamp':[100,900000,1000000],
                     'side':['buy','buy','sell'], 'price':[100.,101.,100.]})
    orders=aggregate_orders(tr,0)
    prefix=aggregate_orders(tr.iloc[:1],0)
    assert orders.t.iloc[0] == .0001
    assert orders.price_ext.iloc[0] == 100.
    assert orders.price_ext.iloc[1] == 101.
    pd.testing.assert_frame_equal(orders.loc[orders.t<.5].reset_index(drop=True),prefix)


def test_poisson_grid_reduces_to_exact_inventory_solution():
    from cjr.grid_reference import solve
    p=replace(params(),a=np.zeros((2,1,2)).tolist(),betas=[1.],eps_plus=0.,eps_minus=0.)
    grid=solve(p,L=2.,h=.5)
    V,da,db=vi.poisson_vi(*p.mu,p,tol=1e-12)
    assert np.max(np.abs(grid['V']-V[:,None,None])) < 1e-9


def test_actor_can_quote_exactly_at_mid_boundary():
    ag=LEGACY.Agent(params(),{'agent':{'hidden':8}},'cpu',1.,1.)
    with torch.no_grad():
        for x in ag.actor.parameters(): x.zero_()
        ag.actor[-1].bias.fill_(-1.)
    y=torch.zeros(1,M.dim(params()))
    assert ag.probs(y).tolist()==[[1.,1.]]
    assert ag.deltas(y).tolist()==[[0.,0.]]


def test_poisson_market_gate_accepts_optimal_policy_and_rejects_bad_critic():
    p=params()
    V,da,db=vi.poisson_vi(*p.mu,p)
    good=reference.poisson_policy_metrics(*p.mu,p,V,da,db)
    assert good['passed']
    bad=reference.poisson_policy_metrics(*p.mu,p,V+100,da,db)
    assert not bad['passed']
