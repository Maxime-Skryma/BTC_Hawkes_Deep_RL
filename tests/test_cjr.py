import numpy as np
import pandas as pd
import pytest
import torch

from cjr import model as M
from cjr import replay, simulate, vi

CFG = {"model": {"gamma": 0.01, "r": 1 / 60, "K": 5}}
CAL = dict(mu=[0.5, 0.5], a=[[[0.0, 0.0]], [[0.0, 0.0]]], betas=[1.0], mean_rates=[0.5, 0.5],
           alpha=dict(zeta=0.5, eps_plus=0.2, eps_minus=0.2), A=0.8, kappa=2.0, sigma2=1.0)


def test_vi_matches_monte_carlo_criterion():
    p = M.Params.from_calibration(CAL, CFG, hawkes=False, alpha=False)
    V, da, db = vi.poisson_vi(0.5, 0.5, p)
    pol = simulate.table_policy(torch.tensor(da, dtype=torch.float32), torch.tensor(db, dtype=torch.float32), p.K)
    r = simulate.evaluate(pol, p, 600, 8192, "cpu")
    # horizon 600 s >> 1/r : le critère MC approche V(0)
    assert r["criterion"][0] == pytest.approx(V[p.K], abs=4 * r["criterion"][1] + 0.02 * abs(V[p.K]))


def test_vi_closed_form_quotes():
    p = M.Params.from_calibration(CAL, CFG, hawkes=False, alpha=False)
    V, da, db = vi.poisson_vi(0.5, 0.5, p)
    k = p.K
    assert da[k] == pytest.approx(max(1 / p.kappa - (V[k - 1] - V[k]), 0))
    assert np.isinf(da[0]) and np.isinf(db[-1])
    assert da[k] == pytest.approx(db[k])  # symétrie à q = 0


def test_jump_and_decay():
    p = M.Params.from_calibration(CAL, CFG)
    T = p.tensors("cpu")
    y = torch.zeros(2, M.dim(p))
    y = M.jump(y, torch.tensor([0, 1]), p)
    assert y[0, 1] == pytest.approx(0.2) and y[1, 1] == pytest.approx(-0.2)
    assert y[0, 2:].tolist() == [1.0, 0.0] and y[1, 2:].tolist() == [0.0, 1.0]
    yd = M.decay(y, torch.tensor([1.0, 1.0]), p, T)
    assert yd[0, 1] == pytest.approx(0.2 * np.exp(-0.5)) and yd[0, 2] == pytest.approx(np.exp(-1.0))


def test_cjr_execution_has_no_extra_thinning():
    cfg = {"model": dict(CFG["model"], execution="cjr")}
    p = M.Params.from_calibration(CAL, cfg)
    delta = torch.tensor([-1., 0., .5])
    assert M.fill_prob(delta, p).tolist() == pytest.approx([1., 1., np.exp(-1.)])
    with pytest.raises(ValueError, match="rho"):
        M.Params.from_calibration(CAL, cfg, rho=.1)


def test_replay_states_ignore_future():
    rng = np.random.default_rng(0)
    orders = pd.DataFrame({"t": np.sort(rng.uniform(0, 100, 300)), "side": rng.choice([1, -1], 300)})
    grid = np.arange(1, 50, 0.1)
    a1, e1 = replay.states(orders, CAL, grid)
    o2 = orders.copy()
    o2.loc[o2.t >= 50, "side"] *= -1
    a2, e2 = replay.states(o2, CAL, grid)
    assert np.allclose(a1, a2) and np.allclose(e1, e2)


def test_replay_fill_rule():
    g = np.array([0.0, 1.0])
    mid = np.array([100.0, 100.0])
    da = np.full((2, 3), 10.0, np.float32)   # ask à 100,10 (10 bps), K = 1
    db = np.full((2, 3), np.inf, np.float32)
    bt, bb, ba = np.array([-1.0]), np.array([99.99]), np.array([100.01])
    # achat qui atteint 100,10 -> exécuté ; achat qui s'arrête à 100,05 -> non
    for ext, n in ((100.10, 1), (100.05, 0)):
        r = replay.run(g, mid, da, db, bt, bb, ba, np.array([0.5]), np.array([1]), np.array([ext]),
                       0.1, 1, 0.01, np.array([0.0]), 1.0)
        assert r[3] == n
    # ρ : l'exécution n'a lieu que si le tirage de l'ordre est < ρ
    for u, n in ((0.05, 1), (0.5, 0)):
        r = replay.run(g, mid, da, db, bt, bb, ba, np.array([0.5]), np.array([1]), np.array([100.10]),
                       0.1, 1, 0.01, np.array([u]), 0.1)
        assert r[3] == n


def test_fee_and_rho_in_vi():
    p0 = M.Params.from_calibration(CAL, CFG, hawkes=False, alpha=False)
    p1 = M.Params.from_calibration(CAL, CFG, hawkes=False, alpha=False, rho=0.1, fee_bps=1.0)
    V0, da0, _ = vi.poisson_vi(0.5, 0.5, p0)
    V1, da1, _ = vi.poisson_vi(0.5, 0.5, p1)
    assert p1.A == pytest.approx(0.1 * p0.A)
    assert da1[p1.K] > da0[p1.K]          # le frais élargit les cotes
    assert V1[p1.K] < V0[p0.K]
    D = V1[p1.K - 1] - V1[p1.K]
    assert da1[p1.K] == pytest.approx(max(1 / p1.kappa + 1.0 - D, 0))


def test_replay_marks_after_fill_window_without_antidating_pnl():
    g=np.array([0.,1.]);mid=np.array([100.,110.])
    da=np.full((2,3),10.,np.float32);db=np.full((2,3),np.inf,np.float32)
    r=replay.run(g,mid,da,db,np.array([-1.,.9]),np.array([99.,109.]),np.array([101.,111.]),
                 np.array([.5]),np.array([1]),np.array([100.2]),.1,1,.01,np.array([0.]),1.,100.)
    assert r[3]==1
    assert r[4][0]==pytest.approx(100.1-110.)


def test_replay_day_final_path_matches_liquidation_and_costs():
    book=pd.DataFrame({'t':[599.,600.9,601.9],'bid':[99.99,109.99,109.99],
                       'ask':[100.01,110.01,110.01]})
    orders=pd.DataFrame({'t':[600.5],'side':[1],'price_ext':[100.2]})
    cfg={'model':{'K':1,'lot_usd':1000.},'data':{'tick':.01},
         'evaluation':{'warmup_s':600.,'stop_s':602.,'decision_interval_s':1.,'latency_s':.1}}
    def pol(y):
        d=torch.full((len(y),2),torch.inf)
        d[:,0]=10.
        return d
    rows,paths=replay.replay_day(book,orders,CAL,cfg,{'constant':pol},'cpu','2026-01-01',1.,
                                {'maker':.001,'taker':.002},[11],every=100)
    r=rows[0]
    assert r['n_fills']==1 and r['q_end']==-1
    assert r['pnl_gross']==pytest.approx(10*(100.1-110.01))
    assert r['pnl_net']==pytest.approx(r['pnl_gross']-r['maker_cost']-r['taker_cost'])
    assert paths['constant|pnl'][-1]==pytest.approx(r['pnl_net'])
    assert paths['constant|q'][-1]==0
    assert r['max_drawdown']==pytest.approx(-r['pnl_net'])
