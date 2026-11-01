"""Datos de texto locales y artefactos tokenizados, sin mezclar historias entre splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader, TensorDataset


class TextTokenizer(Protocol):
    """Operaciones mínimas comunes al tokenizer de bytes y Hugging Face."""

    def encode(self, text: str) -> list[int]:
        """Codifica texto como IDs de tokens."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Recupera texto desde IDs de tokens."""
        ...


class ByteTokenizer:
    """Tokenizer UTF-8 para pruebas offline; 0 es padding y 1 es EOS."""

    def encode(self, text: str) -> list[int]:
        """Codifica bytes con desplazamiento de dos IDs."""
        return [byte + 2 for byte in text.encode('utf-8')]

    def decode(self, ids: list[int]) -> str:
        """Decodifica omitiendo tokens especiales."""
        return bytes(i - 2 for i in ids if 2 <= i < 258).decode('utf-8', errors='replace')


def windows(stories: list[list[int]], length: int, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fragmenta cada historia por separado, conservando un token adicional para baseline."""
    rows, masks = [], []
    for story in stories:
        for start in range(0, len(story) - 1, length):
            row = story[start:start + length + 1]
            if len(row) >= 2:
                masks.append([True] * len(row) + [False] * (length + 1 - len(row)))
                rows.append(row + [pad] * (length + 1 - len(row)))
    if not rows:
        raise ValueError('No hay historias con al menos dos tokens')
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def load_text(cfg: dict, validation_split: str = 'validation') -> tuple[DataLoader, DataLoader, TextTokenizer, dict]:
    """Carga fixture o cache de TinyStories; verifica integridad y vocabulario."""
    data = cfg['datos']
    if data['source'] == 'fixture':
        path = Path(data['path'])
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        tokenizer = ByteTokenizer()
        groups = {split: [tokenizer.encode(row['text']) + [data['eos_id']] for row in records if row['split'] == split] for split in ('train', 'validation')}
        metadata = {'source': 'authored_fixture', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    elif data['source'] == 'cache':
        from transformers import AutoTokenizer
        root = Path(data['path'])
        metadata = json.loads((root / 'manifest.json').read_text())
        expected = cfg['modelo']
        if metadata['vocab_size'] != expected['vocab_size'] or metadata['tokenizer_id'] != data['tokenizer_id']:
            raise ValueError('Cache y configuración del tokenizer incompatibles')
        for name, digest in metadata['tokenizer_hashes'].items():
            if hashlib.sha256((root / 'tokenizer' / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Hash incorrecto en tokenizer/{name}')
        groups = {}
        for split in dict.fromkeys(('train', validation_split)):
            payload = (root / f'{split}.json').read_bytes()
            if hashlib.sha256(payload).hexdigest() != metadata['hashes'][split]:
                raise ValueError(f'Hash incorrecto en {split}')
            groups[split] = json.loads(payload)
        tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer', local_files_only=True)
    else:
        raise ValueError('source debe ser fixture o cache')
    loaders = []
    for split in ('train', validation_split):
        if not 0 < data[f'{split}_stories'] <= len(groups[split]):
            raise ValueError(f'Número de historias inválido para {split}')
        stories = groups[split][:data[f'{split}_stories']]
        tensor, valid = windows(stories, data['sequence_length'], data['pad_id'])
        if tensor.max() >= cfg['modelo']['vocab_size']:
            raise ValueError('ID fuera del vocabulario')
        loaders.append(DataLoader(TensorDataset(tensor, valid), batch_size=data['batch_size'], shuffle=split == 'train', num_workers=data['num_workers']))
    return loaders[0], loaders[1], tokenizer, metadata
