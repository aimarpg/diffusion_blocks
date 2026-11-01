"""Release downloads preserve checkpoints and reject incomplete or altered assets."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
import yaml

from scripts import download_models as downloads


def test_download_verifies_and_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install correct bytes once and skip further network requests for verified files."""
    monkeypatch.chdir(tmp_path)
    data = b'checkpoint fixture'
    asset = {'name': 'model.pt', 'path': 'runs/model.pt', 'size_bytes': len(data),
             'sha256': hashlib.sha256(data).hexdigest()}
    calls = []

    def response(url: str, timeout: int) -> io.BytesIO:
        """Return a download fixture without accessing the network."""
        calls.append(url)
        return io.BytesIO(data)

    monkeypatch.setattr(downloads, 'urlopen', response)
    downloads.download_asset(asset, 'https://example.com')
    downloads.download_asset(asset, 'https://example.com')
    assert Path(asset['path']).read_bytes() == data
    assert calls == ['https://example.com/model.pt']


@pytest.mark.parametrize('data', [b'short', b'bad bytes', b'too many bytes'])
def test_bad_download_preserves_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         data: bytes) -> None:
    """Truncation, checksum failures and oversized responses cannot replace a local file."""
    monkeypatch.chdir(tmp_path)
    Path('model.pt').write_bytes(b'original')
    expected = b'new bytes'
    asset = {'name': 'model.pt', 'path': 'model.pt', 'size_bytes': len(expected),
             'sha256': hashlib.sha256(expected).hexdigest()}
    monkeypatch.setattr(downloads, 'urlopen', lambda *args, **kwargs: io.BytesIO(data))
    with pytest.raises(ValueError, match='existing file differs'):
        downloads.download_asset(asset, 'https://example.com')
    with pytest.raises(ValueError, match='mismatch|exceeds'):
        downloads.download_asset(asset, 'https://example.com', force=True)
    assert Path('model.pt').read_bytes() == b'original'
    assert not list(Path('.').glob('*.part'))


def test_download_selection_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Selected models include the tokenizer; invalid aliases and escaping paths fail."""
    monkeypatch.chdir(tmp_path)
    manifest = {'repository': 'owner/repo', 'tag': 'models-v1', 'assets': [
        {'model': 'a', 'name': 'a.pt', 'path': 'runs/a.pt'},
        {'model': 'b', 'name': 'b.pt', 'path': 'runs/b.pt'},
        {'name': 'tokenizer.json', 'path': 'data/tokenizer.json'}]}
    Path('manifest.json').write_text(json.dumps(manifest))
    selected = []
    monkeypatch.setattr(downloads, 'download_asset', lambda asset, *args: selected.append(asset['name']))
    assert downloads.main(['--manifest', 'manifest.json', '--models', 'b']) == 0
    assert selected == ['b.pt', 'tokenizer.json']
    assert downloads.main(['--manifest', 'manifest.json', '--models', 'missing']) == 1
    with pytest.raises(ValueError):
        downloads.local_path('../outside.pt')
    Path('escape').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        downloads.local_path('escape/outside.pt')


def test_release_manifest_matches_demo() -> None:
    """All configured model and tokenizer paths are covered by the committed manifest."""
    manifest = json.loads(Path('configs/model_assets.json').read_text())
    demo = yaml.safe_load(Path('configs/demo.yaml').read_text())
    assets = manifest['assets']
    assert {a['model']: a['path'] for a in assets if 'model' in a} == {
        name: model['checkpoint'] for name, model in demo['models'].items()}
    assert len({a['name'] for a in assets}) == len(assets)
    for asset in assets:
        assert 0 < asset['size_bytes'] < 2 * 1024**3
        assert len(asset['sha256']) == 64
        downloads.local_path(asset['path'])
        if 'model' not in asset:
            assert Path(asset['path']).parent == Path(demo['tokenizer_path'])
