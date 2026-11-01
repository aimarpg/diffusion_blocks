"""Contratos del experimento completo: reanudación, acumulación y datos reservados."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from src.data.text import ByteTokenizer, load_text
from src.training.full import run_full


@pytest.mark.parametrize('method', ['baseline', 'diffusion'])
@pytest.mark.parametrize('precision', ['fp32', 'fp16'])
def test_resume_matches_uninterrupted(method: str, precision: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Interrumpir a mitad de ronda reproduce pesos, optimizadores, RNG y próxima época."""
    cfg = yaml.safe_load(Path('configs/full_smoke.yaml').read_text())
    if precision == 'fp16':
        if not torch.cuda.is_available():
            pytest.skip('FP16 resume requiere CUDA')
        gpu = yaml.safe_load(Path('configs/full_baseline.yaml').read_text())
        cfg['entrenamiento'].update({key: gpu['entrenamiento'][key] for key in ('device', 'precision')})
    fixture = Path(cfg['datos']['path'])
    shutil.copy(fixture, tmp_path / 'stories.jsonl')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('src.training.full.abrir_run', lambda cfg: None)
    cfg['datos']['path'] = 'stories.jsonl'
    cfg['modelo']['method'] = method
    cfg['salida']['directory'] = 'uninterrupted'
    run_full(cfg)
    expected = torch.load('uninterrupted/last.pt', weights_only=True)
    cfg['salida']['directory'] = 'resumed'
    run_full(cfg, stop_after=cfg['entrenamiento']['checkpoint_every'] + 1)
    with Path('resumed/steps.jsonl').open('a') as file:
        file.write('{"step":')  # Simular escritura cortada al interrumpirse el proceso.
    run_full(cfg, resume=Path('resumed/last.pt'))
    actual = torch.load('resumed/last.pt', weights_only=True)
    assert expected['updates'] == actual['updates']
    assert expected['tokens_per_block'] == actual['tokens_per_block']
    assert expected['stream']['epoch'] > 0
    assert actual['stream']['position'] == expected['stream']['position']
    assert torch.equal(actual['stream']['order'], expected['stream']['order'])
    assert torch.equal(actual['rng']['torch'], expected['rng']['torch'])
    for name, tensor in expected['model'].items():
        assert torch.equal(tensor, actual['model'][name]), name
    for opt_a, opt_b in zip(expected['optimizers'], actual['optimizers'], strict=True):
        for index, state in opt_a['state'].items():
            for name, value in state.items():
                assert torch.equal(value, opt_b['state'][index][name])
    with pytest.raises(FileExistsError):
        run_full(cfg)
    altered = copy.deepcopy(cfg)
    altered['modelo']['noise']['max'] *= 2
    with pytest.raises(ValueError, match='incompatible'):
        run_full(altered, resume=Path('resumed/last.pt'))


def test_reserved_cache_and_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """La preparación reserva test de validation y el evaluador exporta resultados ciegos."""
    from scripts.prepare_pilot import prepare
    from scripts.evaluate_full import run_evaluation
    fixture_cfg = yaml.safe_load(Path('configs/full_smoke.yaml').read_text())
    rows = [json.loads(line) for line in Path(fixture_cfg['datos']['path']).read_text().splitlines()]
    tokenizer = ByteTokenizer()
    tokenizer.eos_token_id = fixture_cfg['datos']['eos_id']
    # Adaptador offline para probar el contrato de cache sin descargas.
    class LocalTokenizer(ByteTokenizer):
        """Tokenizer simulado que implementa la interfaz de preparación."""
        eos_token_id = tokenizer.eos_token_id

        def __len__(self) -> int:
            return fixture_cfg['modelo']['vocab_size']

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return super().encode(text)

        def save_pretrained(self, path: Path) -> None:
            path.mkdir()
            (path / 'tokenizer.json').write_text('{}')

    local = LocalTokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: local)))
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, split, **kw: [row for row in rows if row['split'] == split]))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha='dataset-sha'), model_info=lambda *a, **kw: SimpleNamespace(sha='tokenizer-sha'))))
    monkeypatch.chdir(tmp_path)
    prepare({'output':'cache', 'dataset_id':'fixture', 'dataset_revision':'main', 'tokenizer_id':'fixture',
             'tokenizer_revision':'main', 'train_stories':8, 'validation_stories':1, 'test_stories':1})
    manifest = json.loads(Path('cache/manifest.json').read_text())
    assert manifest['counts'] == {'train':8,'validation':1,'test':1}
    validation = json.loads(Path('cache/validation.json').read_text())
    test = json.loads(Path('cache/test.json').read_text())
    assert validation != test
    cfg = fixture_cfg
    cfg['datos'].update(source='cache',path='cache',tokenizer_id='fixture',validation_stories=1,test_stories=1)
    monkeypatch.setattr('src.training.full.abrir_run', lambda cfg: None)
    cfg['salida']['directory']='trained'
    run_full(cfg, stop_after=1)
    settings = {'device':'cpu','precision':'fp32','cpu_threads':1,'output':'evaluation',
                'checkpoints':['trained/last.pt'],'data_path':'cache','test_stories':1,'max_batches':1,
                'batch_size':1,'prompt_count':1,'prompt_tokens':2,'max_new_tokens':2,
                'seeds':[cfg['generacion']['seed']], 'steps_per_block':cfg['modelo']['noise']['steps_per_block'],
                'clamp':True,'evaluation_seed':cfg['evaluacion']['seed']}
    run_evaluation(settings)
    assert len(Path('evaluation/generations.jsonl').read_text().splitlines()) == 1
    assert 'model' not in Path('evaluation/blind_ratings.csv').read_text().splitlines()[0]
    summary = json.loads(Path('evaluation/summary.json').read_text())
    assert 'perplexity' not in summary[0]['metrics']
    monkeypatch.setattr('src.models.language.LanguageModel.generate', lambda *a, **kw: pytest.fail('No repetir generaciones ya guardadas'))
    run_evaluation(settings, resume=True)
    assert len(Path('evaluation/generations.jsonl').read_text().splitlines()) == 1
    # Corromper un artefacto reservado debe fallar antes de evaluarlo.
    Path('cache/test.json').write_text('[]')
    with pytest.raises(ValueError, match='Hash incorrecto'):
        load_text(cfg, validation_split='test')


@pytest.mark.parametrize('method', ['baseline', 'diffusion'])
def test_accumulation_matches_effective_batch(method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Acumular microbatches con padding equivale a una actualización del batch completo."""
    cfg = yaml.safe_load(Path('configs/full_smoke.yaml').read_text())
    shutil.copy(Path(cfg['datos']['path']), tmp_path / 'stories.jsonl')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('src.training.full.abrir_run', lambda cfg: None)
    cfg['datos']['path'] = 'stories.jsonl'
    cfg['modelo']['method'] = method
    if method == 'diffusion':
        from src.models.language import LanguageModel
        original = LanguageModel.forward

        def fixed_noise(self: LanguageModel, tokens: torch.Tensor, block: int | None = None,
                        sigma: torch.Tensor | None = None, noise: torch.Tensor | None = None) -> torch.Tensor:
            """Fija las realizaciones para comparar el mismo objetivo estocástico."""
            sigma = torch.full((tokens.size(0),), self.cfg['noise']['max'], device=tokens.device)
            noise = torch.zeros_like(self.embedding(tokens))
            return original(self, tokens, block=block, sigma=sigma, noise=noise)

        monkeypatch.setattr(LanguageModel, 'forward', fixed_noise)
    cfg['salida']['directory'] = 'accumulated'
    run_full(cfg, stop_after=1)
    expected = torch.load('accumulated/last.pt', weights_only=True)
    cfg['datos']['batch_size'] *= cfg['entrenamiento']['gradient_accumulation_steps']
    cfg['entrenamiento']['gradient_accumulation_steps'] = 1
    cfg['salida']['directory'] = 'combined'
    run_full(cfg, stop_after=1)
    actual = torch.load('combined/last.pt', weights_only=True)
    assert expected['tokens_per_block'] == actual['tokens_per_block']
    for name, value in expected['model'].items():
        assert torch.allclose(value, actual['model'][name], atol=1e-6, rtol=1e-5), name


@pytest.mark.parametrize('method', ['baseline', 'diffusion'])
def test_dataset_epoch_covers_every_window_once(method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Una época termina sin repetir ventanas ni omitir el último batch parcial; resume coincide."""
    from src.training.full import BatchStream
    cfg = yaml.safe_load(Path('configs/full_epoch_smoke.yaml').read_text())
    shutil.copy(Path(cfg['datos']['path']), tmp_path / 'stories.jsonl')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('src.training.full.abrir_run', lambda cfg: None)
    cfg['datos']['path'] = 'stories.jsonl'
    cfg['modelo']['method'] = method
    cfg['entrenamiento']['max_train_seconds'] = None
    cfg['presupuesto']['total_seconds'] = None
    cfg['salida']['directory'] = 'whole_dataset'
    visited = []
    original = BatchStream.next

    def record_batch(self: BatchStream) -> tuple[torch.Tensor, torch.Tensor]:
        """Registra índices reales para detectar omisiones y repeticiones."""
        result = original(self)
        visited.extend(self.order[self.position - len(result[0]):self.position].tolist())
        return result

    monkeypatch.setattr(BatchStream, 'next', record_batch)
    report = run_full(cfg)
    assert sorted(visited) == list(range(report['training_windows']))
    assert report['epochs_completed'] == cfg['entrenamiento']['epochs']
    assert report['completed_steps']
    assert report['budget_met'] is None
    expected = torch.load('whole_dataset/last.pt', weights_only=True)
    cfg['salida']['directory'] = 'resumed_epoch'
    run_full(cfg, stop_after=1)
    run_full(cfg, resume=Path('resumed_epoch/last.pt'))
    actual = torch.load('resumed_epoch/last.pt', weights_only=True)
    for name, value in expected['model'].items():
        assert torch.equal(value, actual['model'][name])
