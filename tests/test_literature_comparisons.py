from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from cjr.comparisons.glft import GLFT
from cjr.comparisons.run import compare, reconcile_as, report, run
from cjr.model import Params
from cjr.comparisons.local_control import LocalControl, solve_local


def params():
    return Params(mu=[1., 1.], a=[[[0., 0.]], [[0., 0.]]], betas=[1.],
                  eps_plus=0., eps_minus=0., zeta=1., A=1., kappa=2., sigma2=.5,
                  gamma=.1, r=1/60, K=3, fee=0.)


def test_glft_matches_published_closed_form_and_inventory_direction():
    p = params(); agent = GLFT(p, [2., 2.])
    y = torch.zeros(3, 4, dtype=torch.float64); y[:, 0] = torch.tensor([-1., 0., 1.])
    c1 = np.log1p(p.gamma/p.kappa)/p.gamma
    c2 = np.sqrt(p.sigma2*p.gamma/(2*p.kappa*2.) * (1+p.gamma/p.kappa)**(1+p.kappa/p.gamma))
    expected = np.column_stack((c1+(.5-y[:, 0].numpy())*c2,
                                c1+(.5+y[:, 0].numpy())*c2))
    assert np.allclose(agent.deltas(y).numpy(), expected)
    assert agent.deltas(y)[2, 0] < agent.deltas(y)[0, 0]
    assert agent.deltas(y)[2, 1] > agent.deltas(y)[0, 1]


def test_fee_translation_risk_neutral_limit_and_project_constraints():
    p = params(); fee = GLFT(replace(p, fee=.4), [2., 2.])
    plain = GLFT(p, [2., 2.])
    assert fee.skew == pytest.approx(plain.skew*np.exp(p.kappa*.4/2))
    y = torch.zeros(7, 4); y[:, 0] = torch.arange(-3, 4)
    d = fee.deltas(y)
    assert torch.isinf(d[0, 0]) and torch.isinf(d[-1, 1])
    assert (d >= 0).all()
    y[:, 1:] = 1e6
    assert torch.equal(d, fee.deltas(y))
    neutral = GLFT(replace(p, gamma=0., fee=.4), [2., 2.])
    assert neutral.deltas(torch.zeros(1, 4)).tolist()[0] == pytest.approx([.9, .9])
    a, b = fee.grid_tables(np.zeros(10), None, 3, 'cpu', 100)
    assert a.shape == b.shape == (10, 7)
    assert np.allclose(a[0], d[:, 0]) and np.allclose(b[-1], d[:, 1])


def frame():
    return pd.DataFrame([dict(fee='f', date=f'2026-01-{day+1:02}', algorithm=name,
                             train_seed=seed, pnl_net=day+gain)
        for day in range(6)
        for name, seeds, gain in [('as_', [0], 0), ('sac', [11, 23, 37], 5), ('glft', [0], 3)]
        for seed in seeds])


def test_paired_bootstrap_handles_deterministic_policy_without_fake_seeds():
    d = frame(); result = compare(d, ['sac', 'glft'], 1000)
    assert set(result.reference) == {'as'}
    assert len(result) == 6
    for row in result.itertuples():
        expected = 5 if row.algorithm == 'sac' else 3
        assert row.family_low == pytest.approx(expected)
        assert row.family_high == pytest.approx(expected)
    with pytest.raises(ValueError, match='Incomplete'):
        compare(d.drop(index=0), ['sac', 'glft'], 1000)


def test_reconciliation_fails_on_execution_change():
    names = ['pnl_net', 'pnl_gross', 'maker_cost', 'taker_cost', 'turnover',
             'n_fills', 'q_end', 'mean_abs_q', 'max_drawdown']
    row = dict.fromkeys(names, 1.)
    reconcile_as(row, pd.DataFrame([row]))
    with pytest.raises(ValueError, match='pnl_net'):
        reconcile_as(dict(row, pnl_net=2.), pd.DataFrame([row]))


def test_report_and_output_isolation(tmp_path):
    (tmp_path/'evaluation').mkdir()
    (tmp_path/'protocol.json').write_text(json.dumps({'market_config': {
        'realism': {'fees': {'f': {'maker': 0., 'taker': 0.}}}}}))
    report(tmp_path, frame(), ['sac', 'glft'], 1000)
    assert (tmp_path/'rapport.html').exists()
    assert set(pd.read_csv(tmp_path/'evaluation/intervals_vs_as.csv').reference) == {'as'}
    with pytest.raises(ValueError, match='separate'):
        run(tmp_path, tmp_path/'child')


@pytest.mark.parametrize('fee', [0., 1.5, 7.5])
def test_local_control_recovers_independent_poisson_optimum(fee):
    from cjr.reference import poisson_policy_iteration
    p = replace(params(), fee=fee)
    v, da, db, _, residual = solve_local(p, [1.2, .8])
    ref, ra, rb = poisson_policy_iteration(1.2, .8, p)
    assert np.allclose(v, ref, atol=1e-7)
    assert np.allclose(da, ra, atol=1e-6)
    assert np.allclose(db, rb, atol=1e-6)
    assert residual < 1e-8


def test_local_drift_hjb_and_scalar_quote_optimum():
    from scipy.optimize import minimize_scalar
    p = params(); alpha = .07; rates = [1.2, .8]
    v, ask, bid, _, _ = solve_local(p, rates, alpha)
    for i, q in enumerate(range(-p.K, p.K+1)):
        rhs = q*alpha-p.psi(q)
        for delta, shift, rate in [(ask[i], -1, rates[0]), (bid[i], 1, rates[1])]:
            j = i+shift
            if not 0 <= j < len(v):
                assert np.isinf(delta)
                continue
            def loss(d):
                return -p.A*np.exp(-p.kappa*d)*(d-p.fee+v[j]-v[i])
            optimum = minimize_scalar(loss, bounds=(0, 100), method='bounded')
            assert loss(delta) <= min(loss(0.), optimum.fun)+1e-9
            rhs -= rate*loss(delta)
        assert p.r*v[i] == pytest.approx(rhs, abs=1e-8)
    _, zero_ask, zero_bid, _, _ = solve_local(p, rates)
    # Positive alpha shifts both quotes upward: ask farther away, bid closer.
    assert ask[p.K] > zero_ask[p.K]
    assert bid[p.K] < zero_bid[p.K]


def test_local_policy_same_observation_same_quote_and_true_hawkes_intensities():
    from numba import set_num_threads
    set_num_threads(2)
    p = replace(params(), a=[[[.3, .1]], [[.1, .2]]])
    y = torch.tensor([[0., .03, 2., 1.], [0., -.04, 1., 3.]], dtype=torch.float64)
    policy = LocalControl(p, use_alpha=True)
    d = policy.deltas(y).numpy()
    for i, row in enumerate(y.numpy()):
        rates = np.array(p.mu)+np.einsum('sck,ck->s', p.a, row[2:].reshape(p.C, 2))
        _, da, db, _, _ = solve_local(p, rates, row[1])
        assert d[i] == pytest.approx([da[p.K], db[p.K]], abs=1e-6)
    reverse = policy.deltas(y.flip(0)).flip(0).numpy()
    assert np.array_equal(d, reverse)
    without = LocalControl(p)
    changed = y.clone();changed[:, 1] *= 100
    assert torch.equal(without.deltas(y), without.deltas(changed))
    with pytest.raises(RuntimeError, match='converge'):
        solve_local(p, [1.2, .8], .03, max_iter=1)


def test_secondary_contrasts_compare_neural_against_classical_not_zero():
    result = compare(frame(), ['sac', 'glft'], 1000,
                     contrasts=[('sac', 'as'), ('glft', 'as'), ('sac', 'glft')])
    assert set(result.reference) == {'as', 'glft'}
    pair = result[result.reference == 'glft']
    assert len(pair) == 3
    assert np.allclose(pair.delta_as_daily_mean, 2.)
    assert np.allclose(pair.family_low, 2.)


def test_full_extension_real_replay_resume_and_original_preservation(tmp_path, monkeypatch):
    from cjr import replay
    from cjr.baselines import avellaneda_stoikov
    from cjr.comparisons.run import digest
    import importlib
    module = importlib.import_module('cjr.comparisons.run')
    p = params()
    cfg = dict(name='fixture', data=dict(symbol='TEST', raw=['unused'], tick=.01),
        model=dict(execution='cjr', K=p.K, gamma=p.gamma, r=p.r, lot_usd=1000),
        realism=dict(fees={'f': dict(maker=0., taker=0.)}), benchmark=dict(as_horizon_s=60.),
        splits=dict(test=['2026-01-01']), evaluation=dict(warmup_s=.5, stop_s=4.,
            decision_interval_s=.1, latency_s=.1))
    cal = dict(mu=p.mu, a=p.a, betas=p.betas, mean_rates=p.mu, A=p.A,
        kappa=p.kappa, sigma2=p.sigma2, alpha=dict(zeta=p.zeta,
        eps_plus=p.eps_plus, eps_minus=p.eps_minus))
    book = pd.DataFrame(dict(t=np.arange(0, 5., .1), bid=99.99, ask=100.01))
    orders = pd.DataFrame(dict(t=np.arange(.05, 5., .2), side=np.tile([1, -1], 13)[:25],
                               price_ext=np.tile([100.1, 99.9], 13)[:25]))
    rows, _ = replay.replay_day(book, orders, cal, cfg,
        {'as_': avellaneda_stoikov(p, 60)}, 'cpu', '2026-01-01', 1., {'maker': 0., 'taker': 0.}, [11])
    original = [dict(rows[0], fee='f', algorithm=name, strategy=name, train_seed=seed)
        for name, seeds in [('as_', [0]), ('constant', [0]), ('poisson', [0]), ('sac', [11, 23, 37])]
        for seed in seeds]
    base = tmp_path/'base';(base/'evaluation').mkdir(parents=True)
    pd.DataFrame(original).to_csv(base/'evaluation/historique_detail.csv', index=False)
    (base/'calibration.json').write_text(json.dumps(cal))
    (base/'protocol.json').write_text(json.dumps(dict(market_config=cfg,
        spec=dict(algorithms=['sac'], seeds=[11, 23, 37], evaluation=dict(historical_split='test', path_sampling=1)))))
    before = {p.name: digest(p) for p in base.rglob('*') if p.is_file()}
    monkeypatch.setattr(module.data, 'load_day', lambda cfg, day: (book, orders))
    out = tmp_path/'new'
    run(base, out, repetitions=1000)
    merged = pd.read_csv(out/'evaluation/historique_detail.csv')
    assert len(merged) == 9
    assert {'glft', 'local_hawkes', 'local_hawkes_alpha'} <= set(merged.algorithm)
    assert json.loads((out/'verification.json').read_text())['status'] == 'PASS'
    first = digest(out/'evaluation/historique_detail.csv')
    run(base, out, repetitions=1000)
    assert first == digest(out/'evaluation/historique_detail.csv')
    assert before == {p.name: digest(p) for p in base.rglob('*') if p.is_file()}
    traces = out/'evaluation/intraday/f_2026-01-01.npz'
    traces.write_bytes(traces.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='trace changed'):
        run(base, out, repetitions=1000)
