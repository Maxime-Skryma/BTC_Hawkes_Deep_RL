"""Téléchargement des journées publiques Tardis, sans dépendance à un autre dépôt."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip
import os
import time
import urllib.request
from cjr.data import PROJECT_ROOT
import pandas as pd


def dataset_url(exchange,kind,symbol,day):
    day=pd.Timestamp(day)
    return f'https://datasets.tardis.dev/v1/{exchange}/{kind}/{day:%Y/%m/%d}/{symbol}.csv.gz'


def fetch_file(url,path,api_key=None,attempts=3):
    path=Path(path)
    if path.exists():
        with gzip.open(path,'rb') as f:
            if not f.read(1):raise ValueError(f'Fichier vide : {path}')
        return 'présent'
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    for attempt in range(attempts):
        try:
            headers={'User-Agent':'hawkes-deep-rl/0.2'}
            if api_key:headers['Authorization']=f'Bearer {api_key}'
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response,temporary.open('wb') as f:
                while chunk:=response.read(1<<20):f.write(chunk)
            # Lire jusqu'à EOF contrôle aussi l'intégrité CRC du gzip.
            with gzip.open(temporary,'rb') as f:
                while f.read(1<<20):pass
            temporary.replace(path)
            return 'téléchargé'
        except Exception:
            if attempt==attempts-1:raise
            time.sleep(attempt+1)


def download(cfg,workers=4):
    dates=sorted({str(d) for split in cfg['splits'].values() for d in split})
    key=os.environ.get('TARDIS_API_KEY')
    if any(pd.Timestamp(d).day!=1 for d in dates) and not key:
        raise ValueError('Les jours autres que le 1er du mois nécessitent TARDIS_API_KEY')
    directory=Path(cfg['data']['raw'][0]).expanduser()
    if not directory.is_absolute():directory=PROJECT_ROOT/directory
    jobs=[(kind,day) for day in dates for kind in ('trades','book_ticker')]
    def job(args):
        kind,day=args;path=directory/kind/f'{cfg["data"]["symbol"]}_{day}.csv.gz'
        url=dataset_url(cfg['data'].get('exchange','binance'),kind,cfg['data']['symbol'],day)
        status=fetch_file(url,path,key)
        print(status,kind,day,flush=True)
        return path
    with ThreadPoolExecutor(max_workers=workers) as pool:return list(pool.map(job,jobs))
