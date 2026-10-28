"""Checkpoint training must preserve exact optimizer, replay and RNG state."""
import copy
import json
from pathlib import Path
import yaml
import torch
import pytest
from cjr import rl_algorithms as RL
from test_rl_algorithms import params
from scripts.train_checkpoint_study import evaluate



@pytest.fixture
def checkpoint_source(tmp_path):
    """Small frozen inputs for unit tests; no generated experiment required."""
    root=Path(__file__).resolve().parents[1]
    source=tmp_path/'source';source.mkdir()
    cfg=yaml.safe_load((root/'configs/dot.yaml').read_text())
    (source/'protocol.json').write_text(json.dumps(dict(market_config=cfg),default=str))
    (source/'calibration.json').write_bytes((root/'tests/fixtures/dot_train_calibration.json').read_bytes())
    (source/'selection.json').write_text(json.dumps(dict(chosen={'retail_bnb':{'td3__full':{'parameters':{'learning_rate_factor':1.}}}})))
    return str(source)


def settings():
    return dict(hidden=8,threads=1,envs=8,transitions=128,batch=8,buffer=64,start_steps=8,
        actor_lr=.001,critic_lr=.001,tau=.05,exploration_noise=.2,target_noise=.2,
        noise_clip=.5,policy_delay=2,sac_initial_alpha=.02)


@pytest.mark.parametrize('algo',['td3','sac'])
def test_resume_matches_uninterrupted_training_and_validation_does_not_change_rng(algo):
    p=params();s=settings();saved=[]
    def checkpoint(ag,n,snapshot):
        if n==64:
            saved.append(copy.deepcopy(snapshot()));return True
        return False
    full,_=RL.train(p,s,algo,41,log=lambda _:None)
    RL.train(p,s,algo,41,log=lambda _:None,checkpoint_callback=checkpoint)
    resumed,_=RL.train(p,s,algo,41,log=lambda _:None,resume_state=saved[0])
    for name in ('actor','q1','q2','t1','t2','target_actor'):
        net=getattr(full,name)
        if net is not None:
            for a,b in zip(net.parameters(),getattr(resumed,name).parameters()):assert torch.equal(a,b)
    rng=torch.get_rng_state().clone()
    first=evaluate(resumed,p,16,16,91823);second=evaluate(resumed,p,16,16,91823)
    assert first==second
    assert torch.equal(torch.get_rng_state(),rng)
    with pytest.raises(ValueError,match='seed/algorithm'):
        RL.train(p,s,algo,53,log=lambda _:None,resume_state=saved[0])


def test_best_checkpoint_is_selected_by_validation_not_last_loss(tmp_path,monkeypatch,checkpoint_source):
    from scripts import train_checkpoint_study as study
    scores=iter([2.,1.,3.])
    monkeypatch.setattr(study,'evaluate',lambda *args:dict(score=next(scores),score_se=.1,mean_reward_per_event=.2,mean_abs_inventory=.3))
    s=settings()
    spec=dict(source=checkpoint_source,output=str(tmp_path),settings=s,minimum_transitions=128,
        evaluation=dict(every_transitions=64,paths=8,events=8,selection_seed=123,assessment_seed=456,patience_evaluations=8))
    summary=study.job((spec,'td3','retail_bnb',41))
    assert summary['best_transitions']==64
    assert summary['last_transitions']==128
    folder=tmp_path/'td3/retail_bnb/seed41'
    assert torch.load(folder/'best.pt',weights_only=False)['validation']['score']==2.
    assert torch.load(folder/'last.pt',weights_only=False)['validation']['score']==1.
    resume=torch.load(folder/'resume.pt',weights_only=False)
    assert resume['completed_steps']==16 and resume['buffer_size']==64
    assert summary['assessment']['score']==3.


def test_resume_keeps_earliest_checkpoint_when_validation_scores_tie(tmp_path,monkeypatch,checkpoint_source):
    from scripts import train_checkpoint_study as study
    scores=iter([2.,2.,3.,1.,1.,3.])
    monkeypatch.setattr(study,'evaluate',lambda *args:dict(score=next(scores),score_se=.1,mean_reward_per_event=.2,mean_abs_inventory=.3))
    spec=dict(source=checkpoint_source,output=str(tmp_path),settings=settings(),minimum_transitions=128,
        evaluation=dict(every_transitions=64,paths=8,events=8,selection_seed=123,assessment_seed=456,patience_evaluations=8))
    first=study.job((spec,'td3','retail_bnb',41));assert first['best_transitions']==64
    folder=tmp_path/'td3/retail_bnb/seed41';(folder/'summary.json').unlink()
    spec['settings']['transitions']=256;spec['minimum_transitions']=256
    resumed=study.job((spec,'td3','retail_bnb',41));assert resumed['best_transitions']==64
    assert torch.load(folder/'best.pt',weights_only=False)['transitions']==64
