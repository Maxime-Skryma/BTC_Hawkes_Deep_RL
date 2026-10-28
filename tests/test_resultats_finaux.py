"""Checks of published evidence and regression for pooling grid seeds."""
import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT/'experiences/final_test/results.csv').exists(), reason='Run the study to validate local artifacts')
def test_final_results_complete_and_reconciled():
    f=pd.read_csv(ROOT/'resultats/tableaux/resultats_detailles.csv')
    assert len(f)==2376 and f.date.nunique()==22
    assert not f.duplicated(['date','fee','method','seed','through_ticks']).any()
    assert np.allclose(f.pnl_net,f.pnl_gross-f.maker_cost-f.taker_cost)
    for _,g in f.groupby(['fee','method','through_ticks']):
        expected={41,53,67} if g.method.iloc[0] in ['td3','sac'] else {0}
        assert set(g.seed)==expected
        assert g.groupby('seed').size().eq(22).all()


@pytest.mark.skipif(not (ROOT/'experiences/final_selection/grid_selection.json').exists(), reason='Run the study to validate local artifacts')
def test_grid_pools_both_seeds_and_retains_all_arms():
    from scripts.final_selection import candidates,grid
    scores,chosen,seeds=grid(candidates())
    assert list(seeds)==[41,53]
    assert len(scores)==84  # 6 algorithm/fee pairs × (12 grid arms + 2 references)
    assert scores.seeds.eq(2).all()
    assert len(chosen)==6
    for _,c in chosen.iterrows():
        alternatives=scores[(scores.algorithm==c.algorithm)&(scores.fee==c.fee)]
        assert c.score==alternatives.score.max()


@pytest.mark.skipif(not (ROOT/'experiences/final_test/protocol.json').exists(), reason='Run the study to validate local artifacts')
def test_retained_models_and_frozen_selection_hashes():
    sel=ROOT/'experiences/final_selection/selection.json'
    protocol=json.loads((ROOT/'experiences/final_test/protocol.json').read_text())
    assert hashlib.sha256(sel.read_bytes()).hexdigest()==protocol['selection_sha256']
    chosen=json.loads(sel.read_text())['chosen']
    assert len(chosen)==18
    for c in chosen:
        assert hashlib.sha256((ROOT/c['path']).read_bytes()).hexdigest()==c['sha256']
    exceptions=[c for c in chosen if not c['eligible']]
    assert [(c['algorithm'],c['fee'],c['seed']) for c in exceptions]==[('td3','retail_bnb',67)]


def test_grid_seed_pooling_without_generated_artifacts(monkeypatch):
    from scripts import final_selection as selection
    rows=[]
    for algo in ['td3','sac']:
        for fee in ['f1','f2','f3']:
            for seed,run in [(41,'grid41'),(53,'grid53')]:
                rows.append(dict(algorithm=algo,fee=fee,seed=seed,run=run,arm='lr0.1_ups2_anchor0',pnl_1=10.,eligible=True))
                rows.append(dict(algorithm=algo,fee=fee,seed=seed,run='reference',arm='anchored',pnl_1=1.,eligible=True))
    monkeypatch.setattr(selection,'best_per_job',lambda frame:frame)
    scores,chosen,seeds=selection.grid(pd.DataFrame(rows))
    assert list(seeds)==[41,53] and len(scores)==12 and len(chosen)==6
    assert chosen.run.eq('grid41').all() and chosen.score.eq(10.).all()
