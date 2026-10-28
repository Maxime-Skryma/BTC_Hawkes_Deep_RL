import shutil
from pathlib import Path
import pytest
from cjr import workflow


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    shutil.copytree(workflow.ROOT/'configs', tmp_path/'configs')
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    return tmp_path


def test_assets_have_separate_outputs(workspace):
    dot = workflow.Pipeline.for_asset('DOTUSDT', 'quick')
    btc = workflow.Pipeline.for_asset('BTCUSDT', 'quick')
    assert dot.out != btc.out
    assert btc.cfg['data']['symbol'] == 'BTCUSDT'
    assert btc.cfg['data']['raw'] == ['data/raw/btc']


def test_switch_cleans_only_other_data_after_download(workspace, monkeypatch):
    pipeline = workflow.Pipeline.for_asset('BTCUSDT', 'quick')
    for relative in ['data/raw/dot', 'data/cache/causal_v4/dot', 'data/raw/btc', 'experiences/reproduction/dot']:
        folder = workspace/relative
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'keep').write_bytes(b'x'*16)
    monkeypatch.setattr(pipeline, 'download', lambda: None)
    result = pipeline.activate()
    assert len(result['removed']) == 2
    assert (workspace/'data/raw/btc/keep').exists()
    assert (workspace/'experiences/reproduction/dot/keep').exists()


def test_failed_download_keeps_old_data(workspace, monkeypatch):
    pipeline = workflow.Pipeline.for_asset('BTCUSDT', 'quick')
    old = workspace/'data/raw/dot'
    old.mkdir(parents=True)
    def fail(): raise OSError('download failed')
    monkeypatch.setattr(pipeline, 'download', fail)
    with pytest.raises(OSError): pipeline.activate()
    assert old.exists()
