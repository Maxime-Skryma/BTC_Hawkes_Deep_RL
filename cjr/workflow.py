"""Étapes reproductibles appelées par le notebook et la ligne de commande."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Pipeline:
    @classmethod
    def for_asset(cls,symbol='DOTUSDT',profile='full',tick=None):
        if profile not in ('full','quick'):raise ValueError('Profil full ou quick requis')
        symbol=symbol.upper().replace('/','')
        if not re.fullmatch(r'[A-Z0-9_-]+',symbol):raise ValueError('Symbole invalide')
        assets=yaml.safe_load((ROOT/'configs/assets.yaml').read_text())
        asset=assets.get(symbol)
        if asset is None:
            if tick is None:raise ValueError('Actif inconnu : fournir son tick ou le définir dans configs/assets.yaml')
            asset=dict(name=symbol.lower(),raw=f'data/raw/{symbol.lower()}',tick=tick)
        else:asset=dict(asset)
        if tick is not None:asset['tick']=tick
        if not math.isfinite(float(asset['tick'])) or float(asset['tick'])<=0:raise ValueError('Tick positif et fini requis')
        source='configs/benchmark_algorithms.yaml' if profile=='full' else 'configs/reproduce_quick.yaml'
        spec=yaml.safe_load((ROOT/source).read_text());cfg=yaml.safe_load((ROOT/spec['market_config']).read_text())
        cfg['name']=asset['name'];cfg['data'].update(symbol=symbol,tick=asset['tick'],raw=[asset['raw']])
        output=f'experiences/{"reproduction_v2" if profile=="full" else "reproduction_quick_v2"}/{asset["name"]}'
        spec.update(output=output,calibration=f'{output}/calibration.json',market_config=f'{output}/market_config.yaml')
        destination=ROOT/output;destination.mkdir(parents=True,exist_ok=True)
        for path,content in [(destination/'market_config.yaml',cfg),(destination/'benchmark_config.yaml',spec)]:
            if path.exists() and yaml.safe_load(path.read_text())!=content:
                raise RuntimeError('Configuration de cet actif différente : choisir un nouveau dossier de sortie')
            path.write_text(yaml.safe_dump(content,sort_keys=False))
        return cls(destination/'benchmark_config.yaml')

    def __init__(self,config='configs/benchmark_algorithms.yaml'):
        self.config=Path(config)
        if not self.config.is_absolute():self.config=ROOT/self.config
        self.spec=yaml.safe_load(self.config.read_text())
        self.market_config=ROOT/self.spec['market_config']
        self.cfg=yaml.safe_load(self.market_config.read_text())
        self.out=ROOT/self.spec['output']
        self.calibration=ROOT/self.spec['calibration']
        if self.out.resolve() in {(ROOT/'experiences/comparaison_complete_2026-10-25').resolve(),
                                  (ROOT/'experiences/publication/training').resolve()}:
            raise ValueError('Les résultats publiés sont conservés ; choisir un autre dossier de sortie')

    def download(self):
        from cjr.download import download
        return download(self.cfg)

    def clean_other_data(self):
        """Libère seulement les données et caches des autres actifs, jamais experiences/."""
        assets=yaml.safe_load((ROOT/'configs/assets.yaml').read_text())
        removed=[];size=0;data_root=(ROOT/'data').resolve()
        for symbol,asset in assets.items():
            if symbol==self.cfg['data']['symbol']:continue
            for path in [ROOT/asset['raw'],ROOT/'data/cache/causal_v4'/asset['name']]:
                if not path.exists():continue
                if path.is_symlink() or not path.resolve().is_relative_to(data_root):
                    raise ValueError(f'Refus de supprimer des données hors du dossier data : {path}')
                size+=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
                shutil.rmtree(path);removed.append(str(path.relative_to(ROOT)))
        result=dict(removed=removed,freed_mib=round(size/1024**2,1),models_and_results_preserved=True)
        print(json.dumps(result,indent=2));return result

    def activate(self,clean_other=True):
        self.download()  # Une erreur réseau conserve toutes les anciennes données.
        return self.clean_other_data() if clean_other else None

    def calibrate(self):
        import pandas as pd
        import torch
        from cjr import data,calibrate
        torch.set_num_threads(self.spec['settings']['threads'])
        days=[str(d) for d in self.cfg['splits']['train']]
        files=[data.raw_file(self.cfg,kind,pd.Timestamp(day)) for day in days for kind in ('trades','book_ticker')]
        signature=dict(training_days=days,model=self.cfg['model'],symbol=self.cfg['data']['symbol'],
            inputs={str(p.relative_to(ROOT)):sha(p) for p in files},
            source={n:sha(ROOT/'cjr'/n) for n in ('calibrate.py','hawkes.py','data.py')})
        signature=json.loads(json.dumps(signature,default=str))
        meta=self.calibration.with_name('calibration_inputs.json')
        if self.calibration.exists():
            if not meta.exists() or json.loads(meta.read_text())!=signature:
                raise RuntimeError('Calibration existante avec d’autres entrées : nouveau dossier de sortie requis')
            print('Calibration train vérifiée et reprise',flush=True)
            return json.loads(self.calibration.read_text())
        with ThreadPoolExecutor(max_workers=2) as pool:
            loaded=list(pool.map(lambda day:data.load_day(self.cfg,day),days))
        result=calibrate.calibrate(self.cfg,loaded)
        self.calibration.parent.mkdir(parents=True,exist_ok=True)
        self.calibration.write_text(json.dumps(result,indent=2));meta.write_text(json.dumps(signature,indent=2))
        print('Calibration terminée sur',len(days),'jours train uniquement',flush=True)
        return result

    def run_stage(self,stage,algorithm=None):
        logs=self.out/'logs';logs.mkdir(parents=True,exist_ok=True)
        path=logs/f'{stage}{"_"+algorithm if algorithm else ""}.log'
        command=[sys.executable,str(ROOT/'scripts/benchmark_suite.py'),stage,'--config',str(self.config)]
        if algorithm:command+=['--algorithm',algorithm]
        with path.open('a') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            tail='\n'.join(path.read_text().splitlines()[-25:])
            raise RuntimeError(f'Étape {stage} échouée ; voir {path}\n{tail}')
        print(stage,algorithm or '',': terminé —',path.relative_to(ROOT),flush=True)
        return path

    def prepare(self):return self.run_stage('prepare')
    def controls(self):return self.run_stage('controls')
    def train(self,algorithm):return self.run_stage('train',algorithm)
    def train_all(self):
        with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(self.train,self.spec['algorithms']))
    def evaluate(self):return self.run_stage('evaluate')
    def report(self):return self.run_stage('report')

    def compare(self):
        from cjr.comparisons.run import run
        destination=self.out.with_name(self.out.name+'_comparison')
        run(self.out,destination)
        return destination/'rapport.html'

    def verify(self):
        import numpy as np
        import pandas as pd
        import re
        protocol=json.loads((self.out/'protocol.json').read_text())
        from importlib.util import module_from_spec,spec_from_file_location
        module_spec=spec_from_file_location('suite_verify',ROOT/'scripts/benchmark_suite.py')
        suite=module_from_spec(module_spec);module_spec.loader.exec_module(suite)
        suite.validate_scientific_sources(protocol,ROOT)
        frame=pd.read_csv(self.out/'evaluation/historique_detail.csv')
        expected=len(self.cfg['splits'][self.spec['evaluation']['historical_split']])*len(self.cfg['realism']['fees'])*(len(self.spec['algorithms'])*len(self.spec['seeds'])+3)
        assert len(frame)==expected
        assert np.isfinite(frame.select_dtypes('number')).all().all()
        assert np.allclose(frame.pnl_net,frame.pnl_gross-frame.maker_cost-frame.taker_cost,rtol=0,atol=1e-7)
        assert (frame.q_end.abs()<=self.cfg['model']['K']).all()
        for record in frame.itertuples():
            with np.load(self.out/'evaluation/intraday'/f'{record.fee}_{record.date}.npz') as paths:
                assert abs(paths[f'{record.strategy}|pnl'][-1]-record.pnl_net)<1e-7
                assert np.max(np.abs(paths[f'{record.strategy}|q']))<=self.cfg['model']['K']
        report=self.out/'rapport.html'
        for link in re.findall(r'(?:href|src)="([^"]+)"',report.read_text()):
            if not link.startswith(('http:','https:','#')):assert (self.out/link).exists(),link
        result=dict(status='PASS',rows=len(frame),seed_selection='all declared seeds',
                    calibration_train_only=True,sources_match=True,paths_and_costs_reconciled=True,
                    report=str(report.relative_to(ROOT)))
        (self.out/'reproduction_verified.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2));return result

    def all(self):
        self.download();self.calibrate();self.prepare();self.controls();self.train_all()
        self.evaluate();self.report();self.verify();return self.compare()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['download','calibrate','prepare','controls','train','evaluate','report','compare','verify','all'])
    parser.add_argument('--config')
    parser.add_argument('--asset',default='DOTUSDT')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--tick',type=float)
    parser.add_argument('--clean-other-data',action='store_true')
    parser.add_argument('--algorithm',choices=['td3','sac','ppo','value_plus'])
    args=parser.parse_args()
    pipeline=Pipeline(args.config) if args.config else Pipeline.for_asset(args.asset,'quick' if args.quick else 'full',args.tick)
    if args.stage=='download' and args.clean_other_data:pipeline.activate();return
    if args.stage=='train':pipeline.train(args.algorithm) if args.algorithm else pipeline.train_all()
    else:getattr(pipeline,args.stage)()


if __name__=='__main__':main()
