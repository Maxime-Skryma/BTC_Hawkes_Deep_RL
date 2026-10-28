import copy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cjr import model as M, rl_algorithms as RL, discrete_value as DV, replay
from cjr.actor_critic import Normalizer
from cjr.study_statistics import analyse_study

ROOT = Path(__file__).resolve().parents[1]
loader = spec_from_file_location('robust_study',ROOT/'scripts/robust_study.py')
study = module_from_spec(loader)
loader.loader.exec_module(study)


def params():
    return M.Params(mu=[.7,.3],a=[[[.2,.03]],[[.02,.15]]],betas=[1.],
        eps_plus=.1,eps_minus=.2,zeta=.7,A=1.,kappa=1.3,sigma2=.8,gamma=.1,r=.2,K=2,fee=.4)


@pytest.mark.parametrize('observation,hidden', [('inventory',[1,2,3]),('alpha',[2,3]),('hawkes',[1]),('full',[])])
@pytest.mark.parametrize('algorithm', ['td3','sac','ppo','value_plus'])
def test_hidden_information_cannot_change_quotes_or_critic_and_survives_reload(tmp_path,observation,hidden,algorithm):
    p=params();torch.manual_seed(4)
    settings={'hidden':8,'observation':observation}
    ag = (DV.Agent(p,{'agent':settings},'cpu',1.,7.) if algorithm=='value_plus'
          else RL.Agent(p,settings,algorithm,1.,7.))
    y=torch.tensor([[0.,.4,1.,2.],[1.,-.3,2.,.5]])
    perturbed=y.clone()
    perturbed[:,hidden]+=100.
    with torch.no_grad():
        assert torch.equal(ag.deltas(y),ag.deltas(perturbed))
        x=ag.norm(y);xp=ag.norm(perturbed)
        if algorithm=='value_plus':assert torch.equal(ag.V(y),ag.V(perturbed))
        elif algorithm=='ppo':assert torch.equal(ag.q1(x),ag.q1(xp))
        else:assert torch.equal(ag.q1(x,torch.ones(2,2)),ag.q1(xp,torch.ones(2,2)))
    path=tmp_path/'agent.pt';torch.save(ag.state_dict(),path)
    restored=study.load_agent(path,p)
    with torch.no_grad():assert torch.equal(ag.deltas(y),restored.deltas(y))
    wrong = (DV.Agent(p,{'agent':{'hidden':8,'observation':'full' if observation!='full' else 'inventory'}},'cpu',1.,7.)
             if algorithm=='value_plus' else RL.Agent(p,dict(settings,observation='full' if observation!='full' else 'inventory'),algorithm,1.,7.))
    with pytest.raises(ValueError):wrong.load(ag.state_dict())
    if algorithm=='value_plus':
        ask,bid=replay.delta_tables(ag.deltas,y[:,1].numpy(),y[:,2:].numpy(),p.K,'cpu')
        index=y[:,0].long().numpy()+p.K
        with torch.no_grad():assert np.allclose(np.column_stack((ask[np.arange(2),index],bid[np.arange(2),index])),ag.deltas(y).numpy(),atol=1e-6)


def test_ablation_does_not_change_simulated_market_or_reward():
    p=params()
    first=RL.EventEnvironment(p,16,11);second=RL.EventEnvironment(p,16,11)
    Normalizer(p,1.,'inventory')(first.y)
    Normalizer(p,1.,'full')(second.y)
    a=torch.ones(16,2)
    for left,right in zip(first.step(a),second.step(a)):assert torch.equal(left,right)


def test_selection_uses_all_dates_and_seeds_and_deterministic_ties():
    rows=[dict(date=d,train_seed=s,candidate=c,pnl_net=(10 if c==0 and s==11 else 0) if c==0 else 4)
          for d in ['2024-01-01','2024-02-01'] for s in [11,23,37] for c in [0,1]]
    frame=pd.DataFrame(rows)
    winner,scores=study.validation_winner(frame,2,[11,23,37])
    assert winner==1 and scores==pytest.approx([10/3,4])
    frame.pnl_net=1
    assert study.validation_winner(frame,2,[11,23,37])[0]==0
    with pytest.raises(ValueError):study.validation_winner(frame.iloc[1:],2,[11,23,37])
    with pytest.raises(ValueError):study.validation_winner(frame[frame.candidate==0],2,[11,23,37])


def test_study_rejects_overlap_nonmonthly_or_shared_final_seeds():
    spec=copy.deepcopy(study.Study('configs/robust_study_smoke.yaml').spec)
    spec['monthly_splits']['validation']=['2024-01-01','2025-01-01']
    with pytest.raises(ValueError,match='chronological'):study.market_config(spec)


def fixture_frame(n=6,seeds=(41,53,67)):
    rows=[]
    for i in range(n):
        for method,ss in [('sac__inventory',seeds),('sac__full',seeds),('as',[0]),('constant',[0])]:
            for s in ss:
                # Large correlated training effects cancel in the paired
                # ablation, but not in independent neural-vs-classical tests.
                gain=(s*10 if method.startswith('sac') else 0)+(5 if method.endswith('full') else 0)
                rows.append(dict(fee='fee',date=f'2025-{i+1:02}-01',algorithm=method,train_seed=s,
                    pnl_net=i+gain,max_drawdown=1+i,mean_abs_q=2.,turnover=1000.+i))
    return pd.DataFrame(rows)


def test_statistics_pair_dates_and_ablation_seeds_and_report_risk():
    intervals,risk,meta=analyse_study(fixture_frame(),['sac__inventory','sac__full'],['as','constant'],1000)
    ablation=intervals[(intervals.algorithm=='sac__full')&(intervals.reference=='sac__inventory')&(intervals.metric=='pnl_net')]
    assert len(ablation)==3
    assert np.allclose(ablation.ci95_low,5) and np.allclose(ablation.ci95_high,5)
    assert np.allclose(ablation.family_low,5) and np.allclose(ablation.family_high,5)
    assert ablation.paired_training_seeds.all()
    assert len(risk[risk.algorithm=='as'])==1
    assert meta['family_size']==1*5*4*3
    assert meta['simultaneous_critical_value']>0
    assert len(intervals)==meta['family_size']
    assert set(intervals.metric)=={'pnl_net','max_drawdown','mean_abs_q','turnover'}
    again,_,_=analyse_study(fixture_frame(),['sac__inventory','sac__full'],['as','constant'],1000)
    pd.testing.assert_frame_equal(intervals,again)


def test_no_confidence_claims_for_smoke_and_missing_grid_rejected():
    neural=['sac__inventory','sac__full'];classical=['as','constant']
    intervals,_,_=analyse_study(fixture_frame(1,(41,)),neural,classical,1000)
    assert intervals.empty
    with pytest.raises(ValueError,match='Incomplete'):analyse_study(fixture_frame().iloc[1:],neural,classical,1000)
    f=fixture_frame();f.loc[(f.algorithm=='sac__full')&(f.train_seed==41),'train_seed']=42
    with pytest.raises(ValueError,match='matching'):analyse_study(f,neural,classical,1000)


def test_frozen_selection_artifact_cannot_be_overwritten(tmp_path):
    p=tmp_path/'selection.json'
    study.write_json(p,{'candidate':0},immutable=True)
    with pytest.raises(ValueError,match='Frozen'):study.write_json(p,{'candidate':1},immutable=True)
    assert p.read_text().find('"candidate": 0')>=0
