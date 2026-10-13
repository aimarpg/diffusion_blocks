#!/usr/bin/env python3
"""Prepara subset de TinyStories y tokenizer local, con revisiones y hashes verificables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def prepare(cfg: dict) -> None:
    """Descarga en streaming cada split y guarda IDs por historia y manifiesto."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    root = Path(cfg['output'])
    if (root / 'manifest.json').exists():
        raise FileExistsError('Ya existe un cache. Reutilizarlo o escoger otra salida en configs/.')
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    dataset_sha = api.dataset_info(cfg['dataset_id'], revision=cfg['dataset_revision']).sha
    tokenizer_sha = api.model_info(cfg['tokenizer_id'], revision=cfg['tokenizer_revision']).sha
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_id'], revision=tokenizer_sha)
    tokenizer.save_pretrained(root / 'tokenizer')
    hashes = {}
    counts = {}
    seen = set()
    for split in ('train', 'validation'):
        stream = load_dataset(cfg['dataset_id'], revision=dataset_sha, split=split, streaming=True)
        stories = []
        for row in stream:
            text = row['text']
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(ids) >= 2:
                stories.append(ids)
            if len(stories) == cfg[f'{split}_stories']:
                break
        if len(stories) != cfg[f'{split}_stories']:
            raise ValueError(f'No hay suficientes historias en {split}')
        payload = json.dumps(stories).encode()
        (root / f'{split}.json').write_bytes(payload)
        hashes[split] = hashlib.sha256(payload).hexdigest()
        counts[split] = len(stories)
    manifest = {'dataset_id': cfg['dataset_id'], 'dataset_revision': dataset_sha, 'tokenizer_id': cfg['tokenizer_id'], 'tokenizer_revision': tokenizer_sha, 'vocab_size': len(tokenizer), 'eos_id': tokenizer.eos_token_id, 'hashes': hashes, 'tokenizer_hashes': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / 'tokenizer').iterdir() if path.is_file()}, 'counts': counts, 'selection': 'primeras historias únicas por split oficial, sin shuffle'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Preparado: {root}. Conservar manifest.json para reproducir las revisiones exactas.')


def main() -> None:
    """Lee configuración de preparación y genera el cache una única vez."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/prepare_pilot.yaml'))
    args = parser.parse_args()
    prepare(yaml.safe_load(args.config.read_text()))


if __name__ == '__main__':
    main()
