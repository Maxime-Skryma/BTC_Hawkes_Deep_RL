"""Reproduce the ten-seed publication without selecting seeds/checkpoints by P&L."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cjr.workflow import Pipeline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/reproduce_publication.yaml')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.workers<=16:parser.error('workers must be between 1 and 16')
    pipeline=Pipeline(args.config)
    pipeline.download();pipeline.calibrate()
    subprocess.run([sys.executable,str(ROOT/'scripts/validate_sac.py'),'--config',str(pipeline.config),
        '--output',str(pipeline.out.parent/'sac_validation')],cwd=ROOT,check=True)
    pipeline.prepare()
    spec=pipeline.spec;logs=pipeline.out/'logs';logs.mkdir(exist_ok=True)
    jobs=[(algorithm,fee,seed) for algorithm in spec['algorithms']
          for fee in pipeline.cfg['realism']['fees'] for seed in spec['seeds']]
    def train(job):
        algorithm,fee,seed=job
        command=[sys.executable,str(ROOT/'scripts/benchmark_suite.py'),'train',
                 '--config',str(pipeline.config),'--algorithm',algorithm,'--fee',fee,'--seed',str(seed)]
        path=logs/f'{algorithm}_{fee}_seed{seed}.log'
        with path.open('a') as stream:
            subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,
                env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMBA_NUM_THREADS':'2'})
        return job
    done=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(train,job):job for job in jobs}
        for future in as_completed(futures):
            done.append(future.result())
            (logs/'training_progress.json').write_text(json.dumps(dict(total=len(jobs),completed=done),indent=2))
            print(f'Training {len(done)}/{len(jobs)}:',*done[-1],flush=True)
    pipeline.controls()
    shards=[(fee,str(day)) for fee in pipeline.cfg['realism']['fees']
            for day in pipeline.cfg['splits'][spec['evaluation']['historical_split']]]
    def replay(job):
        fee,day=job
        command=[sys.executable,str(ROOT/'scripts/benchmark_suite.py'),'evaluate',
            '--config',str(pipeline.config),'--fee',fee,'--day',day]
        with (logs/f'replay_{fee}_{day}.log').open('a') as stream:
            subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,
                env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMBA_NUM_THREADS':'2'})
        return job
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,job in enumerate(pool.map(replay,shards),1):
            print(f'Replay {i}/{len(shards)}:',*job,flush=True)
    pipeline.evaluate();pipeline.report();pipeline.verify()
    from cjr.comparisons.run import run
    destination=pipeline.out.parent/'comparaison'
    run(pipeline.out,destination)
    protocol=json.loads((pipeline.out/'protocol.json').read_text())
    for name in protocol['source_sha256']:
        target=pipeline.out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    print('PUBLICATION COMPLETE',destination/'rapport.html',flush=True)


if __name__=='__main__':main()
