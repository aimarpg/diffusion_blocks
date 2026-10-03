# Diffusion blocks on decoder-only LLM

Replicación de **Shing, M., Koyama, M., & Akiba, T. (2026). DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation** sobre **TinyStories**, con aplicativo: comparador web construido encima.

- **Paper:** https://arxiv.org/abs/2506.14202
- **Implementación de referencia:** https://github.com/SakanaAI/DiffusionBlocks
- **Seguimiento de experimentos:** https://wandb.ai/trabajo-integrado/diffusion-blocks
- **Autoría:** Aimar Pagonabarraga Gallastegui, Jaime Etxebarria Ugarte

## Resultado

| Métrica    | Paper  | Nuestra réplica | Escala                      |
| ---------- | ------ | --------------- | --------------------------- |
| <accuracy> | <0.72> | <0.68>          | <CIFAR-10, subset de 5.000> |

<Una frase honesta sobre la diferencia. Una desviación bien explicada puntúa
igual que un acierto; una desviación sin explicar, no.>

## Instalación

```bash
git clone https://github.com/aimarpg/diffusion_blocks
cd diffusion_blocks
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

```bash
python scripts/get_data.py
```

Descarga TinyStories de https://huggingface.co/datasets/roneneldan/TinyStories y verifica los SHA-256. Ocupa ~1 GB.
Licencia: [Community Data License Agreement – Sharing, Version 1.0](https://cdla.dev/sharing-1-0/)

## Ejecución

```bash
# Smoke test: el pipeline entero en miniatura. Menos de 5 minutos en CPU.
python run.py --smoke

python run.py --config configs/fast.yaml

# Entrenamiento completo. Menos de 20 minutos en una T4 de Colab gratuito.
python run.py --config configs/default.yaml

python run.py --config configs/full.yaml

# El aplicativo, sobre el checkpoint entrenado.
python -m src.app.serve --checkpoint checkpoints/mejor.pt
```

## Estructura

```
configs/      Todos los hiperparámetros. Ninguno escrito a fuego en el código
src/data/     Carga, splits y transformaciones
src/models/   La arquitectura replicada
src/training/ Bucle de entrenamiento y evaluación
src/app/      La capa de servicio: carga un checkpoint y responde
scripts/      Descarga de datos y utilidades
tests/        Tests de forma, invariancia y reproducibilidad
run.py        Punto de entrada único
```

## Reproducibilidad

- Semilla fijada en `src/utils/seed.py`, valor en `configs/`.
- Dependencias con versión exacta en `requirements.txt`.
- Presupuesto de cómputo declarado: **<tiempo real> en RTX 4090**.
- Qué se recortó respecto al paper y qué se pierde con ello: <explicación>.

## Decisiones donde el paper no especifica

<La sección que distingue replicar de copiar. Lista lo que tuvisteis que
decidir vosotros y con qué criterio. Sale de la ficha de replicación.>

Para resolver cada ambigüedad miramos el código oficial (que solo cubre ViT con CIFAR-100), y si es necesario, los trabajos citados en el paper. Si no resuelven la ambigüedad, decidimos nosotros.

| Ambigüedad del paper                                                                           | Qué decidimos                                                                | Por qué                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| El paper usa 12 capas repartidas en 4 bloques. SmolLM2-135M tiene 30, que no es múltiplo de 4. | Usar 3 bloques de 10 capas.                                                  | Según las tablas 7 y 11 del paper, usando una distribución equiprobable de rangos de ruido gausiano, repartir las capas a partes iguales da el mejor resultado, y usar 3 o 4 bloques da resultados similares (MAUVE 0.65 y 0.67 respectivamente) |
| No da la longitud del prompt ni el modelo de features para MAUVE.                              | Prompt de 50 tokens, 50 generados, features con GPT-2 Large.                 | Protocolo del paper de SEDD (Lou et al., 2024), que es el que cita el paper y GPT-2 Large es el valor por defecto de `mauve-text`.                                                                                                               |
| Condicionamiento de ruido σ "como en DiT", sin detalles, y SmolLM2 usa RMSNorm.                | adaLN-Zero tras cada RMSNorm, con σ codificado en un vector de dimensión 96. | Como en `vit.py` del código oficial (`hidden_size // 6`). Añade ~10.5 M parámetros (~8 %) que el base no tiene.                                                                                                                                  |


## Tests

```bash
pytest tests/ -q
```

## Uso de IA

Documentado en [`AI_USAGE.md`](AI_USAGE.md), según la guía docente.
