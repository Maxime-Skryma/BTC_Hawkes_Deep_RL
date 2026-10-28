"""Contre-exemples causaux du rejeu et contrôles des références simples."""
import numpy as np
import pandas as pd
import torch
import yaml

from cjr import replay, simulate, vi, model as M
from cjr.calibrate import reference_mid
from pathlib import Path


def test_quote_inventory_is_frozen_at_decision_before_latency():
    g = np.array([0., 1., 2.])
    da = np.full((3, 5), 10., np.float32)
    db = np.full((3, 5), np.inf, np.float32)
    da[1, 1] = 100.  # quote if q=-1; q is still 0 at decision t=1
    result = replay.run(
        g, np.full(3, 100.), da, db, np.array([-1.]), np.array([99.99]), np.array([100.01]),
        np.array([1.1, 1.3]), np.ones(2, dtype=np.int64), np.array([100.2, 100.2]),
        .2, 2, .01, np.zeros(2), 1., 100.)
    # First fill arrives during latency. The quote decided at t=1 stays 100.1.
    assert result[3] == 2
    assert result[0] == -2


def test_reference_numeraire_ignores_future_book():
    book = pd.DataFrame({'t': [599., 600., 601.], 'bid': [99., 109., 119.],
                         'ask': [101., 111., 121.]})
    assert reference_mid(book) == 100.
    book.loc[book.t >= 600., ['bid', 'ask']] *= 100
    assert reference_mid(book) == 100.


def test_saved_poisson_baseline_uses_train_calibration_and_inventory_only():
    root = Path(__file__).resolve().parents[1]
    import json
    cal = json.loads((root / 'tests/fixtures/dot_train_calibration.json').read_text())
    cfg = yaml.safe_load((root / 'configs/dot.yaml').read_text())
    assert set(cfg['splits']['train']).isdisjoint(cfg['splits']['test'])
    for fee in cfg['realism']['fees'].values():
        p = M.Params.from_calibration(cal, cfg, hawkes=False, alpha=False, fee_bps=fee['maker']*1e4)
        _, da, db = vi.poisson_vi(*cal['mean_rates'], p)
        assert np.isinf(da[0]) and np.isinf(db[-1])
        assert np.all(da[1:] >= 0) and np.all(db[:-1] >= 0)
        policy = simulate.table_policy(torch.tensor(da), torch.tensor(db), p.K)
        y = torch.zeros(2*p.K+1, M.dim(p)); y[:, 0] = torch.arange(-p.K, p.K+1)
        a = policy(y)
        y[:, 1:] = 123456.  # neither contemporaneous nor future alpha/flow is consulted
        assert torch.equal(a, policy(y))
