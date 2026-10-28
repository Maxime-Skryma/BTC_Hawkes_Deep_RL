from pathlib import Path
import gzip
import io
import json
import pytest
import yaml
from cjr import data
from cjr.download import fetch_file, dataset_url
from cjr.workflow import Pipeline, ROOT


def test_raw_data_paths_are_relative_to_project_not_current_directory(tmp_path,monkeypatch):
    import pandas as pd
    monkeypatch.setattr(data,'PROJECT_ROOT',tmp_path/'clone')
    p=tmp_path/'clone/data/raw/dot/trades/DOTUSDT_2024-01-01.csv.gz'
    p.parent.mkdir(parents=True);p.write_bytes(b'test')
    elsewhere=tmp_path/'other';elsewhere.mkdir();monkeypatch.chdir(elsewhere)
    cfg={'data':{'raw':['data/raw/dot'],'symbol':'DOTUSDT'}}
    assert data.raw_file(cfg,'trades',pd.Timestamp('2024-01-01'))==p


def test_download_is_validated_before_atomic_publication(tmp_path,monkeypatch):
    import urllib.request
    content=gzip.compress(b'timestamp,price\n1,100\n')
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:io.BytesIO(content))
    path=tmp_path/'data.csv.gz'
    assert fetch_file('https://example.invalid/file',path,attempts=1)=='téléchargé'
    assert gzip.decompress(path.read_bytes())==b'timestamp,price\n1,100\n'
    assert not path.with_suffix('.gz.part').exists()
    assert fetch_file('https://example.invalid/file',path,attempts=1)=='présent'


def test_invalid_download_does_not_publish_corrupt_file(tmp_path,monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'not gzip'))
    path=tmp_path/'bad.csv.gz'
    with pytest.raises(gzip.BadGzipFile):fetch_file('https://example.invalid/file',path,attempts=1)
    assert not path.exists()


@pytest.mark.parametrize('output', ['experiences/comparaison_complete_2026-10-25', 'experiences/publication/training'])
def test_reproduction_cannot_overwrite_published_benchmark(tmp_path, output):
    spec=yaml.safe_load((ROOT/'configs/benchmark_algorithms.yaml').read_text())
    spec['output']=output
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError,match='conservés'):Pipeline(path)
