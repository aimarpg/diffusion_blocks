#!/usr/bin/env python3
"""Prepare named release assets and a checksum manifest from the local demo files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_models import digest, local_path


def prepare(settings: dict) -> dict:
    """Stage checkpoints/tokenizer using hard links when possible and hash every asset."""
    demo = yaml.safe_load(local_path(settings['demo_config']).read_text())
    output = local_path(settings['output'])
    manifest_path = local_path(settings['manifest'])
    sources = [(f'{alias}.pt', local_path(model['checkpoint']), alias)
               for alias, model in demo['models'].items()]
    tokenizer = local_path(demo['tokenizer_path'])
    if not tokenizer.is_dir():
        raise ValueError(f'Missing tokenizer: {tokenizer}')
    sources += [(f'tokenizer-{path.name}', path, None)
                for path in sorted(tokenizer.iterdir()) if path.is_file()]
    if not sources or len({name for name, _, _ in sources}) != len(sources):
        raise ValueError('Release asset names must be unique')
    for name, path, _ in sources:
        if not path.is_file():
            raise ValueError(f'Missing release source: {path}')
        if path.stat().st_size >= 2 * 1024**3:
            raise ValueError(f'{name} exceeds the GitHub release asset size limit')
        if Path(name).name != name:
            raise ValueError(f'Invalid asset name: {name}')
    output.mkdir(parents=True, exist_ok=True)
    assets = []
    for name, source, model in sources:
        destination = output / name
        checksum = digest(source)
        if destination.exists():
            if digest(destination) != checksum:
                raise ValueError(f'{destination} differs; choose a fresh output directory')
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        asset = {'name': name, 'path': source.as_posix(),
                 'size_bytes': source.stat().st_size, 'sha256': checksum}
        if model is not None:
            asset['model'] = model
        assets.append(asset)
        print(f'Prepared: {name}', flush=True)
    manifest = {'repository': settings['repository'], 'tag': settings['tag'], 'assets': assets}
    content = json.dumps(manifest, indent=2) + '\n'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(content)
    (output / manifest_path.name).write_text(content)
    print(f'Manifest: {manifest_path}; release assets: {output}', flush=True)
    return manifest


def main() -> None:
    """Prepare the release described by the project configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/model_release.yaml')
    args = parser.parse_args()
    prepare(yaml.safe_load(local_path(args.config).read_text()))


if __name__ == '__main__':
    main()
