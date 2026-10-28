"""Shard evaluation must preserve the serial benchmark, including common randomness."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cjr.model import Params
from cjr.rl_algorithms import Agent


def test_parallel_replay_shard_equals_serial_evaluation(tmp_path, monkeypatch):
    root=Path(__file__).resolve().parents[1]
    loader=importlib.util.spec_from_file_location('benchmark_shard_test',root/'scripts/benchmark_suite.py')
    suite=importlib.util.module_from_spec(loader);loader.loader.exec_module(suite)
    cfg=dict(name='fixture',data=dict(symbol='TEST',raw=['unused'],tick=.01),
        model=dict(execution='cjr',K=2,gamma=.1,r=.2,lot_usd=1000),
        realism=dict(fees={'f':dict(maker=0.,taker=0.)}),benchmark=dict(as_horizon_s=5.),
        splits=dict(test=['2026-01-01']),evaluation=dict(warmup_s=.5,stop_s=2.,
            decision_interval_s=.1,latency_s=.1))
    cal=dict(mu=[.7,.3],a=[[[0.,0.]],[[0.,0.]]],betas=[1.],mean_rates=[.7,.3],
        A=1.,kappa=1.3,sigma2=.8,alpha=dict(zeta=.7,eps_plus=0.,eps_minus=0.))
    spec=dict(algorithms=['sac'],seeds=[11],settings={'threads':1},evaluation=dict(
        historical_split='test',path_sampling=1,simulation_seconds=.1,simulation_paths=2,simulation_seed=2510))
    book=pd.DataFrame(dict(t=np.arange(0,3.,.1),bid=99.99,ask=100.01))
    orders=pd.DataFrame(dict(t=np.arange(.05,3.,.2),side=np.tile([1,-1],8)[:15],
                             price_ext=np.tile([100.1,99.9],8)[:15]))
    monkeypatch.setattr(suite.data,'load_day',lambda cfg,day:(book,orders))
    outputs=[]
    for mode in ['serial','shard']:
        out=tmp_path/mode;out.mkdir()
        (out/'protocol.json').write_text(json.dumps(dict(spec=spec,market_config=cfg)))
        p=Params.from_calibration(cal,cfg,fee_bps=0.0)
        torch.manual_seed(11)
        agent=Agent(p,{'hidden':8},'sac',1.,1.)
        sd=agent.state_dict();sd.update(training_seed=11,protocol_sha256=suite.sha(out/'protocol.json'))
        checkpoint=suite.checkpoint_path(out,'sac','f',11);checkpoint.parent.mkdir(parents=True)
        torch.save(sd,checkpoint)
        if mode=='shard':suite.replay_shard(spec,cfg,cal,out,'f','2026-01-01')
        suite.evaluate(spec,cfg,cal,out)
        outputs.append(pd.read_csv(out/'evaluation/historique_detail.csv'))
    pd.testing.assert_frame_equal(outputs[0],outputs[1])
    assert len(outputs[0])==4  # SAC, AS, Poisson, constant; no no-quote criterion.
