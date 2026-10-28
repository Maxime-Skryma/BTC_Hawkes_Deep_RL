from dataclasses import replace
import numpy as np
import pytest
import torch
from cjr.baselines import avellaneda_stoikov
from cjr.model import Params


def params():
    return Params(mu=[1.,1.], a=[[[0.,0.]],[[0.,0.]]],betas=[1.],
                  eps_plus=0.,eps_minus=0.,zeta=1.,A=1.,kappa=2.,sigma2=.5,
                  gamma=.1,r=1/60,K=3,fee=.2)


def test_as_matches_published_reservation_and_spread_formula():
    p=params();tau=2.
    y=torch.tensor([[-1.,0.,0.,0.],[0.,0.,0.,0.],[1.,0.,0.,0.]],dtype=torch.float64)
    d=avellaneda_stoikov(p,tau)(y).numpy()
    half=np.log1p(p.gamma/p.kappa)/p.gamma+.5*p.gamma*p.sigma2*tau+p.fee
    assert np.allclose(d.sum(1),2*half)
    # Reservation shift from mid is (bid distance - ask distance)/2 with opposite sign.
    assert np.allclose((d[:,1]-d[:,0])/2,y[:,0].numpy()*p.gamma*p.sigma2*tau)
    assert np.allclose(d[:,0],half-y[:,0].numpy()*p.gamma*p.sigma2*tau)


def test_as_constraints_information_set_and_risk_neutral_limit():
    p=params();pol=avellaneda_stoikov(p,60.)
    y=torch.zeros(7,4);y[:,0]=torch.arange(-3,4)
    d=pol(y)
    assert torch.isinf(d[0,0]) and torch.isinf(d[-1,1])
    assert torch.all(d>=0)
    y[:,1:]=1e8
    assert torch.equal(d,pol(y))
    neutral=avellaneda_stoikov(replace(p,gamma=0.),60.)(y)
    assert neutral[3].tolist()==pytest.approx([1/p.kappa+p.fee]*2)
    more_fee=avellaneda_stoikov(replace(p,fee=p.fee+1),60.)(torch.zeros(1,4))
    assert torch.allclose(more_fee,pol(torch.zeros(1,4))+1)


@pytest.mark.parametrize('tau',[0.,-1.,float('inf'),float('nan')])
def test_invalid_as_horizon(tau):
    with pytest.raises(ValueError):avellaneda_stoikov(params(),tau)
