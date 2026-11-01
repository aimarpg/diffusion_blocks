#!/usr/bin/env python3
"""Download release checkpoints and tokenizer files with SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen


def local_path(value: str) -> Path:
    """Accept project-relative paths only, including when symlinks are present."""
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected a project-relative path: {value}')
    path.resolve().relative_to(Path.cwd().resolve())
    return path


def digest(path: Path) -> str:
    """Hash a file without loading checkpoints into memory."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def matches(path: Path, asset: dict) -> bool:
    """Check both size and checksum against the release manifest."""
    return (path.is_file() and path.stat().st_size == asset['size_bytes']
            and digest(path) == asset['sha256'])


def download_asset(asset: dict, base_url: str, force: bool = False) -> None:
    """Install a verified download atomically, preserving existing files on failure."""
    destination = local_path(asset['path'])
    if matches(destination, asset):
        print(f'Verified: {destination}', flush=True)
        return
    if destination.exists() and not force:
        raise ValueError(f'{destination}: existing file differs; use --force to replace it')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    print(f'Downloading {asset["name"]} ({asset["size_bytes"] / 2**20:.1f} MiB)...', flush=True)
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            with urlopen(f'{base_url}/{quote(asset["name"], safe="")}', timeout=60) as response:
                received = 0
                next_progress = 64 * 1024 * 1024
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > asset['size_bytes']:
                        raise ValueError(f'{asset["name"]}: download exceeds expected size')
                    output.write(chunk)
                    if received >= next_progress:
                        print(f'  {received / 2**20:.0f} MiB received', flush=True)
                        next_progress += 64 * 1024 * 1024
        if not matches(temporary, asset):
            raise ValueError(f'{asset["name"]}: size or SHA-256 mismatch')
        temporary.replace(destination)
        print(f'Installed: {destination}', flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Download selected models and their tokenizer, or list the release assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='configs/model_assets.json')
    parser.add_argument('--models', nargs='+', help='Model aliases; default: all six')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--force', action='store_true', help='Replace files with different checksums')
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(local_path(args.manifest).read_text())
        models = {a['model'] for a in manifest['assets'] if a.get('model')}
        selected = set(args.models) if args.models else models
        if selected - models:
            raise ValueError(f'Unknown models: {sorted(selected - models)}')
        base_url = (f'https://github.com/{manifest["repository"]}/releases/download/'
                    f'{quote(manifest["tag"], safe="")}')
        for asset in manifest['assets']:
            if asset.get('model') and asset['model'] not in selected:
                continue
            if args.list:
                print(f'{asset.get("model", "tokenizer")}: {asset["path"]} '
                      f'({asset["size_bytes"] / 2**20:.1f} MiB)')
            else:
                download_asset(asset, base_url, args.force)
    except (OSError, ValueError, KeyError, URLError) as error:
        print(f'Model download failed: {error}', file=sys.stderr)
        print('Check that the release is published and run from the repository root.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
