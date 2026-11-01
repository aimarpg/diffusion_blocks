# DiffusionBlocks sobre una arquitectura SmolLM2-135M

Adaptación experimental de **Shing, Koyama y Akiba, DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation** a una arquitectura de tamaños SmolLM2-135M sobre TinyStories. El objetivo inicial es comprobar entrenamiento por bloques, aprendizaje mínimo y presupuesto de cómputo; todavía no hay una réplica de calidad del paper.

- [Paper](https://arxiv.org/abs/2506.14202)
- [Implementación oficial de referencia, ViT](https://github.com/SakanaAI/DiffusionBlocks)
- [Configuración oficial SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json)
- [W&B](https://wandb.ai/trabajo-integrado/diffusion-blocks)
- Autoría: Aimar Pagonabarraga Gallastegui, Jaime Etxebarria Ugarte.

## Instalación

Python 3.11 recomendado. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Mac Intel, para A/B y tests, usar `pip install -r requirements-macos-intel.txt` en lugar del fichero principal. C requiere Linux/CUDA y está preparado para T4.

## Pruebas implementadas

La explicación del método, máscaras, criterios de aceptación, limitaciones y ejecución en Colab está en [docs/pruebas_viabilidad.md](docs/pruebas_viabilidad.md).

```bash
# A: pipeline completo reducido, CPU, menos de 5 minutos
python run.py --smoke

# B: aprendizaje mínimo sobre ocho historias repetidas
python run.py --config configs/overfit.yaml

# C: preparar TinyStories y tokenizer, una sola vez y fuera del entrenamiento
python scripts/prepare_pilot.py --config configs/prepare_pilot.yaml

# C: dos procesos separados, en CUDA/T4
python run.py --config configs/pilot_baseline.yaml
python run.py --config configs/pilot_diffusion.yaml

# D: aprendizaje y generalización en TinyStories (CUDA, ~10 min por test en RTX 2070)
python scripts/prepare_pilot.py --config configs/prepare_learn.yaml
python run.py --config configs/learn_baseline.yaml
python run.py --config configs/learn_diffusion_noise_c.yaml        # ruido de C
python run.py --config configs/learn_diffusion_noise_shifted.yaml  # ruido tuneado

# Comparar recursos de las ejecuciones C
python scripts/compare_pilot.py \
  --baseline runs/prueba_c_baseline/report.json \
  --diffusion runs/prueba_c_diffusion/report.json

pytest tests/ -q
python tools/tiiac_check.py .
```

A/B usan historias de prueba escritas localmente y tokenizer de bytes, sin descargas. C usa el tokenizer oficial y TinyStories con splits oficiales, revisiones resueltas a SHA y hashes del cache tokenizado. No hay importación de pesos oficiales: los modelos se inicializan desde cero.

`configs/default.yaml` es una copia del piloto C diffusion, no un entrenamiento científico completo ya validado. Los comandos CPU explícitos son A y B.

## Artefactos y seguimiento

Cada prueba escribe checkpoint, muestra, pasos e informe JSON en `runs/prueba_*`. La salida es 0 si se cumplen sus criterios y 1 si no. Los errores se propagan y dejan `failure.json` una vez creada la carpeta de salida.

W&B registra todos los entrenamientos en modo offline por defecto. Para sincronizar después: `wandb login` y `wandb sync wandb/offline-run-*`. Los valores de las semillas están en configs y se fijan a través de `src/utils/seed.py`.

## Alcance actual

DiffusionBlocks usa tres bloques de diez capas en C, ruido en embeddings, condicionamiento propio por bloque y una máscara que permite pasado limpio sin revelar el token objetivo. La generación recorre los intervalos mediante Euler.

**Decisión de viabilidad:** el embedding y la proyección de salida están ligados y congelados en ambos pilotos C. Los adaptadores añaden parámetros a diffusion. Los informes distinguen parámetros totales y entrenables. Este prototipo no demuestra aún preentrenamiento completo, calidad competitiva ni viabilidad temporal en T4: C debe ejecutarse allí.

La aplicación web y el fine-tuning de checkpoints oficiales quedan pendientes. El servicio heredado es una plantilla y no consume aún los checkpoints de estas pruebas.

## Estructura

```text
configs/                  Hiperparámetros y presupuestos de cada prueba
src/data/text.py          Tokenización offline y cache de TinyStories
src/models/language.py    Transformer, máscaras, denoisers y sampler
src/models/factory.py     Selección de arquitectura
src/training/probes.py    Entrenamiento, evaluación fija y perfilado
src/utils/                Semillas y seguimiento W&B
scripts/prepare_pilot.py   Preparación de datos para C
scripts/compare_pilot.py   Comparación de recursos bajo condiciones iguales
tests/                    Contratos de causalidad, aislamiento y reproducibilidad
docs/                     Diseño y guía de ejecución
run.py                    Entrada única
```

Uso de IA documentado en [AI_USAGE.md](AI_USAGE.md).

## Experimento completo: tres variantes, 30 capas

La guía de [entrenamiento completo](docs/full_training.md) documenta preparación,
calibración GPU, entrenamiento reanudable, presupuestos y comparación de calidad.

```bash
python scripts/prepare_pilot.py --config configs/prepare_full.yaml
python scripts/train_full.py --phase calibrate
python scripts/train_full.py --phase train
python scripts/evaluate_full.py --quick
```

Para una ejecución exploratoria de hasta una hora de entrenamiento por variante,
fuera del presupuesto docente de 20 minutos, usar `--budget extended` tanto al
entrenar como al evaluar. `--all-seeds` repite las tres semillas configuradas;
`--resume` continúa los artefactos de la misma ejecución sin sobrescribirlos.

Para recorrer **todo el dataset de 200k historias** por modelo, sin corte temporal:

```bash
python scripts/train_full.py --phase train --budget dataset
python scripts/evaluate_full.py --budget dataset --quick
```

`budgets.dataset.epochs` configura las pasadas completas (por defecto una).
El preset `extended` sigue limitado a una hora por variante y no garantiza una época.

## Demo de generación en terminal

Generar con los checkpoints D/E existentes, sin entrenar ni descargar datos:

```bash
python demo.py                         # elegir modelo y escribir un inicio de historia
python demo.py --list                  # ver modelos disponibles
python demo.py --model d-adjusted e-adjusted --prompt "One day, Lily found a shiny key."
```

La [guía de la demo](docs/demo_cli.md) explica el uso interactivo, la comparación,
las semillas, CPU/GPU y cómo guardar ejemplos. Las rutas y valores por defecto
están en `configs/demo.yaml`.
