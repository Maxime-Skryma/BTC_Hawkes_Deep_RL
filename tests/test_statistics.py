import numpy as np
import pandas as pd
import pytest
from cjr.statistics import analyse, block_indices, bootstrap_means, closing_drawdown


def fixture_frame(days=6):
    rows=[]
    for fee in ['fee']:
        for day in range(days):
            for algorithm, seeds in [('as_', [0]), ('sac', [11,23,37]), ('td3', [11,23,37])]:
                for seed in seeds:
                    gain={'as_':0,'sac':5,'td3':-5}[algorithm]
                    rows.append(dict(fee=fee,date=f'2026-01-{day+1:02}',algorithm=algorithm,
                        train_seed=seed,pnl_net=day+gain,max_drawdown=day+2,mean_abs_q=2))
    return pd.DataFrame(rows)


def test_paired_differences_risk_and_uncertainty():
    tables,family=analyse(fixture_frame(),['sac','td3'],repetitions=1000)
    s=tables['summary'].set_index('algorithm')
    assert family==3
    assert s.loc['sac','delta_as_daily_mean']==5
    assert s.loc['sac','delta_as_daily_median']==5
    assert s.loc['sac','days_beating_as']==6
    assert s.loc['sac','seeds_beating_as']==3
    assert s.loc['sac','leave_one_day_out_delta_min']==5
    assert s.loc['sac','signal_as']=='positif exploratoire'
    assert s.loc['td3','signal_as']=='négatif exploratoire'
    assert s.loc['sac','worst_20pct_days_mean']==5.5
    assert s.loc['sac','max_daily_drawdown']==7
    p=tables['pairs']
    assert (p.mean_daily==10).all()
    assert (p.family_low==10).all()


def test_whole_seed_resampling_preserves_cross_date_structure():
    matrix=np.array([[100., -100.],[-100.,100.]])
    idx=np.tile([0,1],(1000,1))
    assert np.all(bootstrap_means(matrix,idx,np.random.default_rng(1))==0)


def test_blocks_preserve_local_order_without_wrapping():
    idx=block_indices(9,3,500,np.random.default_rng(2))
    assert idx.shape==(500,9)
    assert (idx>=0).all() and (idx<9).all()
    assert (np.diff(idx.reshape(500,3,3),axis=2)==1).all()


def test_incomplete_or_duplicated_grid_is_rejected():
    f=fixture_frame()
    with pytest.raises(ValueError,match='incomplète'):analyse(f.drop(index=1),['sac','td3'],1000)
    with pytest.raises(ValueError,match='dupliqués'):analyse(pd.concat([f,f.iloc[[0]]]),['sac','td3'],1000)


def test_small_samples_have_no_confidence_claims():
    tables,_=analyse(fixture_frame(1),['sac','td3'],1000)
    assert tables['intervals'].empty
    assert (tables['summary'].signal_as=='non évalué').all()


def test_closing_drawdown_starts_at_zero():
    assert closing_drawdown([-5,2,-3,10])==6


def test_bootstrap_is_reproducible():
    a,_=analyse(fixture_frame(),['sac','td3'],1000)
    b,_=analyse(fixture_frame(),['sac','td3'],1000)
    pd.testing.assert_frame_equal(a['intervals'],b['intervals'])


def test_generated_report_reconciles_inputs_and_handles_quick_profile(tmp_path):
    import json
    from cjr.statistics import build_statistics
    frame=fixture_frame(1)
    (tmp_path/'evaluation').mkdir()
    (tmp_path/'figures').mkdir()
    frame.to_csv(tmp_path/'evaluation/historique_detail.csv',index=False)
    protocol={'spec':{'algorithms':['sac','td3'],'seeds':[11,23,37],
                      'evaluation':{'historical_split':'test'}},
              'market_config':{'splits':{'test':['2026-01-01']},
                               'realism':{'fees':{'fee':{'maker':0.,'taker':.0002}}}}}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    original=(tmp_path/'evaluation/historique_detail.csv').read_bytes()
    build_statistics(tmp_path,repetitions=1000)
    assert (tmp_path/'evaluation/historique_detail.csv').read_bytes()==original
    assert (tmp_path/'figures/comparaison_statistique.png').exists()
    page=(tmp_path/'statistiques.html').read_text()
    assert 'Pas de gagnant clairement établi' in page
    assert 'non évalué' in page
    metadata=json.loads((tmp_path/'evaluation/statistics/method.json').read_text())
    assert metadata['repetitions']==1000
    # La méthode doit décrire le nombre de réplications effectivement demandé.
    assert metadata['status']=='exploratory'
    frame=frame[~((frame.algorithm=='sac')&(frame.train_seed==11))]
    frame.to_csv(tmp_path/'evaluation/historique_detail.csv',index=False)
    with pytest.raises(ValueError,match='protocole'):build_statistics(tmp_path,1000)
