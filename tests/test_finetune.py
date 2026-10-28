"""Fine-tuning depuis checkpoint : reprise exacte, ancrage, randomisation, validation historique."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from cjr import finetune as FT, historical_validation as HV, model as M, replay, rl_algorithms as RL
from test_rl_algorithms import params

ROOT=Path(__file__).resolve().parents[1]


def base():
    return dict(hidden=8,threads=1,envs=8,transitions=64,batch=8,buffer=64,start_steps=8,actor_lr=.001,critic_lr=.001,
        tau=.05,exploration_noise=.2,target_noise=.2,noise_clip=.5,policy_delay=2,sac_initial_alpha=.02,sac_alpha_lr=.001,observation='full')


def finetune_settings(**kw):
    s=dict(base(),transitions=128,updates_per_step=3,critic_warmup=32,anchor_weight=.4,anchor_decay=96,actor_ema=.9,
           log_alpha_min=-9.,dr_groups=2,domain_randomization=dict(baseline=[.7,1.4],alpha_impact=[.5,4.]))
    s.update(kw);return s


def pretrained(algo,seed=7):
    ag,_=RL.train(params(),base(),algo,seed,log=lambda _:None);return ag.state_dict()


@pytest.mark.parametrize('algo',['td3','sac'])
def test_finetune_resume_matches_uninterrupted(algo):
    p=params();init=pretrained(algo);s=finetune_settings();saved=[]
    full,_=FT.train(p,s,algo,41,init,log=lambda _:None)
    def stop(ag,n,snapshot):
        if n==64:saved.append(copy.deepcopy(snapshot()));return True
        return False
    FT.train(p,s,algo,41,init,log=lambda _:None,checkpoint_callback=stop)
    resumed,_=FT.train(p,s,algo,41,init,log=lambda _:None,resume_state=saved[0])
    for name in ('actor','q1','q2','t1','t2','target_actor','actor_ema'):
        net=getattr(full,name)
        if net is not None:
            for a,b in zip(net.parameters(),getattr(resumed,name).parameters()):assert torch.equal(a,b)
    with pytest.raises(ValueError,match='initial checkpoint'):FT.train(p,s,algo,41,pretrained(algo,8),log=lambda _:None,resume_state=saved[0])


def test_starts_from_checkpoint_and_frozen_actor_during_critic_warmup():
    p=params();init=pretrained('td3');s=finetune_settings(transitions=32,critic_warmup=32)
    ag,_=FT.train(p,s,'td3',41,init,log=lambda _:None)
    for k,v in init['actor'].items():assert torch.equal(ag.actor.state_dict()[k],v)
    assert ag.norm.alpha_scale==init['alpha_scale'] and ag.value_scale==init['value_scale']
    # Critique bien mis à jour pendant l'échauffement.
    assert any(not torch.equal(ag.q1.state_dict()[k],v) for k,v in init['q1'].items())


def test_strong_anchor_keeps_policy_closer_to_checkpoint():
    p=params();init=pretrained('td3');ref=RL.Agent(p,base(),'td3',init['alpha_scale'],init['value_scale']);ref.load(init)
    y=RL.EventEnvironment(p,64,3).y
    def drift(weight):
        ag,_=FT.train(p,finetune_settings(critic_warmup=0,anchor_weight=weight,anchor_decay=10**9,dr_groups=1),'td3',41,init,log=lambda _:None)
        with torch.no_grad():
            m=RL.active_sides(y,p);return float((ag.actor.sample(ag.norm(y),m,True)[0]-ref.actor.sample(ref.norm(y),m,True)[0]).square().mean())
    assert drift(100.)<drift(0.)


def test_domain_randomization_keeps_action_semantics_and_reward_units():
    p=params();groups=FT.randomized_params(p,4,41,dict(baseline=[.5,2.],excitation=[.9,1.5],alpha_impact=[.5,4.]))
    assert groups[0][0]==p
    for g,_ in groups[1:]:
        assert (g.fee,g.kappa,g.A,g.K,g.gamma,g.sigma2,g.r)==(p.fee,p.kappa,p.A,p.K,p.gamma,p.sigma2,p.r)
    env=FT.GroupedEnvironment([g for g,_ in groups],10,5)
    assert sorted(env.sizes,reverse=True)==env.sizes and sum(env.sizes)==10
    assert all(e.value_scale==env.value_scale for e in env.envs)
    assert FT.randomized_params(p,4,41,dict(baseline=[.5,2.]))==FT.randomized_params(p,4,41,dict(baseline=[.5,2.]))


def test_historical_validator_matches_replay_day_and_rejects_test_days():
    cache=ROOT.parent/'.cache/BTC_Hawkes_Deep_RL/data/cache'
    if not (cache/'causal_v4/dot/2024-03-01/orders.parquet').exists() or not (ROOT/'experiences/checkpoint_study_v1/td3/vip3_bnb/seed41/best.pt').exists():pytest.skip('Local historical data/model unavailable')
    protocol=json.loads((ROOT/'experiences/robust_study_v1/protocol.json').read_text());cfg=protocol['market_config']
    cal=json.loads((ROOT/'experiences/robust_study_v1/calibration.json').read_text());cfg['evaluation'].update(warmup_s=600,stop_s=2400)
    fee=cfg['realism']['fees']['vip3_bnb'];p=M.Params.from_calibration(cal,cfg,fee_bps=fee['maker']*1e4)
    sd=torch.load(ROOT/'experiences/checkpoint_study_v1/td3/vip3_bnb/seed41/best.pt',weights_only=False)
    ag=RL.Agent(p,sd['settings'],'td3',sd['alpha_scale'],sd['value_scale']);ag.load(sd)
    from cjr import data
    book,orders=data.load_day(cfg,'2024-03-01',cache_root=cache)
    expected,_=replay.replay_day(book,orders,cal,cfg,{'x':ag.deltas},'cpu','2024-03-01',1.,fee,[11])
    got=HV.HistoricalValidator(cfg,cal,['2024-03-01'],fee,cache,forbidden_days=cfg['splits']['test'])(ag.deltas)['days'][0]
    for k in ('pnl_gross','pnl_net','n_fills','q_end','mean_abs_q','maker_cost','taker_cost'):assert got[k]==pytest.approx(expected[0][k],abs=1e-9)
    with pytest.raises(ValueError,match='interdites'):HV.HistoricalValidator(cfg,cal,['2025-01-01'],fee,cache,forbidden_days=cfg['splits']['test'])
