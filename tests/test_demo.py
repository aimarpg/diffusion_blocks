"""Inference contracts: checkpoint parity, terminal flow and tokenizer integrity."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.app.generate import CheckpointGenerator, _tokenizer, main
from src.models.factory import construir_modelo
from src.utils.seed import seed_temporal


@pytest.fixture
def demo_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    """Create tiny baseline/diffusion checkpoints without training or dataset access."""
    cfg = yaml.safe_load(Path('configs/smoke.yaml').read_text())
    settings = yaml.safe_load(Path('configs/demo.yaml').read_text())
    settings.update(device='cpu', default_model='diffusion', tokenizer_path='missing-tokenizer')
    settings['generation']['steps_per_block'] = cfg['modelo']['noise']['steps_per_block']
    settings['models'] = {}
    monkeypatch.chdir(tmp_path)
    for method in ('baseline', 'diffusion'):
        variant = copy.deepcopy(cfg)
        variant['modelo']['method'] = method
        variant['experimento']['nombre'] = method
        with seed_temporal(cfg['experimento']['seed']):
            model = construir_modelo(variant)
        torch.save({'config': variant, 'model': model.state_dict(),
                    'metadata': {'source': 'authored_fixture'}}, f'{method}.pt')
        settings['models'][method] = {'description': method, 'checkpoint': f'{method}.pt'}
    Path('demo.yaml').write_text(yaml.safe_dump(settings))
    return cfg, settings


@pytest.mark.parametrize('method', ['baseline', 'diffusion'])
def test_demo_matches_direct_sampler(demo_files: tuple[dict, dict], method: str) -> None:
    """Saved-checkpoint inference reproduces the evaluator's sampler with identical settings."""
    cfg, settings = demo_files
    runner = CheckpointGenerator(Path(f'{method}.pt'), settings, torch.device('cpu'))
    seed = cfg['generacion']['seed']
    count = cfg['generacion']['max_new_tokens']
    prompt = cfg['generacion']['prompt']
    inputs = torch.tensor([runner.tokenizer.encode(prompt)])
    with seed_temporal(seed):
        expected = runner.model.generate(inputs, count)[0, inputs.size(1):].tolist()
    first = runner.generate(prompt, seed, count)
    second = runner.generate(prompt, seed, count)
    assert first['token_ids'] == second['token_ids'] == expected
    assert not runner.model.training
    assert first['checkpoint_sha256'] == hashlib.sha256(Path(f'{method}.pt').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='exceeds'):
        runner.generate(prompt, seed, settings['generation']['max_total_tokens'])


def test_demo_comparison_and_interactive(demo_files: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    """Multi-model JSONL results append, and the menu supports multiple prompts and clean exit."""
    cfg, _ = demo_files
    args = ['--config', 'demo.yaml', '--model', 'baseline', 'diffusion',
            '--prompt', cfg['generacion']['prompt'], '--tokens', str(cfg['generacion']['max_new_tokens']),
            '--output', 'outputs/examples.jsonl']
    assert main(args) == main(args) == 0
    rows = [json.loads(line) for line in Path('outputs/examples.jsonl').read_text().splitlines()]
    assert [r['alias'] for r in rows] == ['baseline', 'diffusion', 'baseline', 'diffusion']
    assert rows[0]['token_ids'] == rows[2]['token_ids']
    answers = iter(['', '/help', '/unknown', cfg['generacion']['prompt'], '/models', 'baseline', cfg['generacion']['prompt'], '/quit'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert main(['--config', 'demo.yaml', '--tokens', str(cfg['generacion']['max_new_tokens']),
                 '--plain', '--output', 'outputs/interactive.jsonl']) == 0
    interactive = [json.loads(line) for line in Path('outputs/interactive.jsonl').read_text().splitlines()]
    assert [row['alias'] for row in interactive] == ['diffusion', 'baseline']
    assert 'Continuation:' in capsys.readouterr().out
    assert main(['--config', 'demo.yaml', '--model', 'unknown', '--prompt', 'Once']) == 1
    assert main(['--config', 'demo.yaml', '--model', 'baseline', '--prompt', 'Once', '--output', 'baseline.pt']) == 1


def test_demo_trims_at_eos(demo_files: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch) -> None:
    """EOS hides following tokens but raw IDs remain available for evaluation."""
    cfg, settings = demo_files
    runner = CheckpointGenerator(Path('baseline.pt'), settings, torch.device('cpu'))
    prompt = cfg['generacion']['prompt']
    ids = runner.tokenizer.encode(prompt)
    generated = runner.tokenizer.encode('a') + [cfg['datos']['eos_id']] + runner.tokenizer.encode('b')
    monkeypatch.setattr(runner.model, 'generate', lambda *_: torch.tensor([ids + generated]))
    row = runner.generate(prompt, cfg['generacion']['seed'], len(generated))
    assert row['continuation'] == 'a'
    assert row['token_ids'] == generated
    assert row['first_eos_index'] == len(runner.tokenizer.encode('a'))


def test_demo_rejects_wrong_tokenizer(tmp_path: Path) -> None:
    """Reject a changed tokenizer file before loading or using it."""
    cfg = yaml.safe_load(Path('configs/learn_baseline.yaml').read_text())
    payload = b'original'
    metadata = {'vocab_size': cfg['modelo']['vocab_size'], 'eos_id': cfg['datos']['eos_id'],
                'tokenizer_hashes': {'tokenizer.json': hashlib.sha256(payload).hexdigest()}}
    (tmp_path / 'tokenizer.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='does not match'):
        _tokenizer(cfg, metadata, tmp_path)
