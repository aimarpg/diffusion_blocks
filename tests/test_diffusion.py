"""Contratos del denoiser: causalidad, aislamiento, ruido y checkpoint."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from src.data.text import load_text, windows
from src.models.factory import construir_modelo
from src.models.language import attention_mask
from src.training.probes import evaluate, loss_for
from src.utils.seed import fijar_seed


@pytest.fixture
def config() -> dict:
    """Usa los hiperparámetros de smoke como contrato de pruebas."""
    return yaml.safe_load(Path('configs/smoke.yaml').read_text())


def test_no_clean_target_or_future_leak(config: dict) -> None:
    """Modificar target limpio o futuro no altera el denoising del token actual."""
    fijar_seed(config['experimento']['seed'])
    model = construir_modelo(config).eval()
    # Activar gates para que el test ejercite atención, no solo la identidad inicial.
    for layer in model.blocks[0].layers:
        torch.nn.init.normal_(layer.modulation.weight, std=config['modelo']['initializer_range'])
    tokens = torch.tensor([[10, 20, 30, 40]])
    noisy = torch.randn(1, 4, config['modelo']['hidden_size'])
    sigma = model.boundaries[0].expand(1)
    changed = tokens.clone()
    changed[:, 2:] = torch.tensor([[50, 60]])
    first = model.denoise(tokens, noisy, sigma, 0)
    second = model.denoise(changed, noisy, sigma, 0)
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-6)
    assert not torch.allclose(first[:, 3], second[:, 3])


def test_mask_semantics() -> None:
    """La rama limpia es causal; la ruidosa solo ve pasado limpio y sí misma."""
    mask = attention_mask(3, True, torch.device('cpu'))
    assert mask[4].tolist() == [True, False, False, False, True, False]
    assert not mask[:3, 3:].any()


def test_only_active_block_updates(config: dict) -> None:
    """Una actualización no toca codebook ni otros bloques."""
    fijar_seed(config['experimento']['seed'])
    model = construir_modelo(config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    opt = torch.optim.AdamW(model.blocks[1].parameters(), lr=config['entrenamiento']['learning_rate'])
    tokens = torch.tensor([[10, 20, 30, 40]])
    loss = model(tokens, block=1).square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.blocks[0].parameters())
    assert model.embedding.weight.grad is None
    opt.step()
    changed = [key for key, value in model.state_dict().items() if not torch.equal(before[key], value)]
    assert changed and all(key.startswith('blocks.1.') for key in changed)


def test_sigma_ranges_and_shapes(config: dict) -> None:
    """Cada bloque muestrea exclusivamente su intervalo y entrega logits correctos."""
    model = construir_modelo(config)
    for block in range(3):
        sigma = model.sample_sigma(block, 100, torch.device('cpu'))
        assert sigma.min() >= model.boundaries[block + 1] - 1e-7
        assert sigma.max() <= model.boundaries[block] + 1e-7
        assert model(torch.tensor([[2, 3, 4]]), block=block).shape == (1, 3, 258)


def test_padding_and_story_boundaries() -> None:
    """Fragmentación no junta historias y distingue EOS válido de padding con el mismo ID."""
    tokens, valid = windows([[5, 6, 0], [7, 8]], 4, 0)
    assert tokens.tolist() == [[5, 6, 0, 0, 0], [7, 8, 0, 0, 0]]
    assert valid.tolist() == [[True, True, True, False, False], [True, True, False, False, False]]


def test_fixed_evaluation_preserves_rng(config: dict) -> None:
    """Evaluar no cambia el RNG; el ruido fijo permite comparar antes/después."""
    fijar_seed(config['experimento']['seed'])
    model = construir_modelo(config)
    train, validation, _, _ = load_text(config)
    state = torch.get_rng_state().clone()
    a = evaluate(model, validation, config, torch.device('cpu'))
    assert torch.equal(state, torch.get_rng_state())
    b = evaluate(model, validation, config, torch.device('cpu'))
    assert a == b
    assert 'perplexity' not in a


def test_baseline_is_causal_and_shifted(config: dict) -> None:
    """El baseline predice el siguiente token y no accede al futuro."""
    config = copy.deepcopy(config)
    config['modelo']['method'] = 'baseline'
    model = construir_modelo(config).eval()
    tokens = torch.tensor([[10, 20, 30, 40]])
    other = tokens.clone()
    other[:, 2:] = 50
    assert torch.allclose(model(tokens)[:, :2], model(other)[:, :2], atol=1e-6)
    loss, count = loss_for(model, tokens, torch.ones_like(tokens, dtype=torch.bool), 0, config)
    expected = torch.nn.functional.cross_entropy(model(tokens[:, :-1]).reshape(-1, 258), tokens[:, 1:].reshape(-1))
    assert count == 3 and torch.allclose(loss, expected)


def test_checkpoint_generation(config: dict, tmp_path: Path) -> None:
    """Recarga pesos y reproduce sampler con la misma seed."""
    model = construir_modelo(config)
    path = tmp_path / 'model.pt'
    torch.save({'model': model.state_dict(), 'config': config}, path)
    saved = torch.load(path, weights_only=True)
    restored = construir_modelo(saved['config'])
    restored.load_state_dict(saved['model'])
    prompt = torch.tensor([[2, 3]])
    fijar_seed(config['generacion']['seed'])
    a = model.generate(prompt, 2)
    fijar_seed(config['generacion']['seed'])
    b = restored.generate(prompt, 2)
    assert torch.equal(a, b)


def test_common_frozen_codebook(config: dict) -> None:
    """Ambos métodos parten de exactamente el mismo embedding congelado."""
    fijar_seed(config['experimento']['seed'])
    diffusion = construir_modelo(config)
    baseline_cfg = copy.deepcopy(config)
    baseline_cfg['modelo']['method'] = 'baseline'
    fijar_seed(config['experimento']['seed'])
    baseline = construir_modelo(baseline_cfg)
    assert torch.equal(diffusion.embedding.weight, baseline.embedding.weight)
    assert not diffusion.embedding.weight.requires_grad
    assert not baseline.embedding.weight.requires_grad


def test_complete_model_sizes() -> None:
    """Valida tamaños C sin reservar cientos de MB de memoria real."""
    for path, expected in [('configs/pilot_baseline.yaml', 134515008), ('configs/pilot_diffusion.yaml', 144628992)]:
        cfg = yaml.safe_load(Path(path).read_text())
        with torch.device('meta'):
            model = construir_modelo(cfg)
        assert sum(p.numel() for p in model.parameters()) == expected
        assert len(model.blocks) == (3 if model.diffusion else 1)


def test_evaluation_preserves_python_numpy(config: dict) -> None:
    """El contexto de ruido fijo restaura también los RNG usados por otros loaders."""
    import random
    import numpy as np
    from src.utils.seed import seed_temporal
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with seed_temporal(config['evaluacion']['seed']):
        random.random()
        np.random.rand()
    assert random.getstate() == python_state
    restored = np.random.get_state()
    assert numpy_state[0] == restored[0]
    assert np.array_equal(numpy_state[1], restored[1])
    assert numpy_state[2:] == restored[2:]


def test_c_requires_cuda_and_writes_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """El piloto no sustituye silenciosamente la T4 por CPU."""
    import json
    from src.training.probes import run_probe
    cfg = yaml.safe_load(Path('configs/pilot_diffusion.yaml').read_text())
    cfg['salida']['directory'] = 'run'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='requiere CUDA'):
        run_probe(cfg)
    failure = json.loads(Path('run/failure.json').read_text())
    assert failure['error'] == 'RuntimeError'


def test_compare_rejects_different_conditions() -> None:
    """No se presentan ratios como comparables si se cambió la longitud."""
    from scripts.compare_pilot import compare
    base_cfg = yaml.safe_load(Path('configs/pilot_baseline.yaml').read_text())
    diff_cfg = yaml.safe_load(Path('configs/pilot_diffusion.yaml').read_text())
    baseline = {'method': 'baseline', 'prueba': 'C', 'passed': True, 'config': base_cfg}
    diffusion = {'method': 'diffusion', 'prueba': 'C', 'passed': True, 'config': diff_cfg}
    diff_cfg['datos']['sequence_length'] *= 2
    with pytest.raises(ValueError, match='datos'):
        compare(baseline, diffusion)


def test_clamp_projects_to_codebook(config: dict) -> None:
    """Con clamp, la predicción del sampler es exactamente un embedding del codebook."""
    fijar_seed(config['experimento']['seed'])
    model = construir_modelo(config)
    prediction = torch.randn(1, 3, config['modelo']['hidden_size'])
    clamped = model._clamp(prediction)
    ids = torch.nn.functional.linear(prediction, model.embedding.weight).argmax(-1)
    assert torch.equal(clamped, model.embedding.weight[ids])
