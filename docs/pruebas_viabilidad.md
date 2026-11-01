# Pruebas A, B y C de DiffusionBlocks sobre una arquitectura SmolLM2

Estas pruebas verifican ejecución, aprendizaje inicial y recursos. No reproducen todavía los resultados del paper ni demuestran que se pueda preentrenar un generador útil de 135M en veinte minutos. Los resultados de clasificación de la plantilla anterior no son aplicables.

## Diseño implementado

- Transformer con RMSNorm, atención GQA, RoPE y MLP SwiGLU. C conserva los tamaños publicados de SmolLM2-135M: 30 capas, dimensión 576, MLP 1536, 9 cabezas, 3 KV heads y vocabulario 49152. Se inicializa **desde cero**; no descarga pesos preentrenados. La implementación propia no ofrece aún importación de checkpoints oficiales ni KV cache.
- DiffusionBlocks divide las capas en tres grupos iguales. Cada grupo tiene su normalización final y embedding de nivel de ruido, sin parámetros entrenables compartidos con los demás grupos.
- La tabla de tokens sirve también como proyección de salida. Está **congelada y aleatoria en ambas configuraciones C**, para aislar los bloques y comparar recursos con un codebook común. El baseline C también congela esa tabla. Por tanto, son experimentos con aproximadamente 135M parámetros arquitectónicos y menos parámetros entrenables; el informe cuenta ambos. No llamarlos preentrenamiento completo de todos los pesos.
- Los adaptadores de ruido añaden parámetros al método diffusion. Inicialización tipo adaLN-Zero: escala, desplazamiento y gates para atención y MLP sobre RMSNorm. Sus dimensiones están en YAML. No es un port literal del ViT oficial.
- Ruido gaussiano en embeddings: `z_sigma = embedding(tokens) + sigma * epsilon`. Cada bloque recupera los **tokens actuales** con entropía cruzada; el baseline predice los siguientes tokens mediante labels desplazadas.
- Precondicionamiento tipo EDM: entrada ruidosa escalada por `1/sqrt(sigma²+sigma_data²)`, salida `D = c_skip*z_sigma + c_out*block_output`, con `c_skip=sigma_data²/(sigma²+sigma_data²)` y `c_out=sigma*sigma_data/sqrt(sigma²+sigma_data²)`. Los logits son `D @ embedding.T / logit_temperature`. No se añade una ponderación de pérdida dependiente de sigma: esta CE es una adaptación explícita.
- El ruido se muestrea de una log-normal truncada, repartida por cuantiles de igual probabilidad entre bloques. Las pruebas usan rangos ajustados al codebook aleatorio; no los defaults del experimento de imágenes del paper. No hay solapamiento de intervalos en este primer prototipo.
- Los bloques se visitan por **rondas**, no aleatoriamente, para garantizar cobertura y crear todos los estados AdamW durante el calentamiento. Cada paso actualiza solo un bloque. El resto no participa en su forward.

### Causalidad y ausencia de fuga

Cada bloque procesa `[historia limpia, historia ruidosa]`. Se duplican los índices RoPE de las posiciones correspondientes. La máscara booleana SDPA usa `True` para permitir atención:

1. Una posición limpia solo ve posiciones limpias anteriores y ella misma.
2. Una posición ruidosa `i` solo ve posiciones limpias `< i` y su propia posición ruidosa.
3. Ninguna posición limpia ve la rama ruidosa.

Así el token limpio objetivo y el futuro no se revelan, ni indirectamente a través de otras posiciones ruidosas. Esta máscara conservadora es una decisión de adaptación verificable mediante tests.

### Generación

El baseline genera por argmax. DiffusionBlocks genera un token cada vez: inicializa su representación con ruido en el extremo alto, recorre los tres bloques y sus intervalos, e integra con Euler:

`z_next = z + (sigma_next - sigma) * (z - D(z, sigma)) / sigma`

Con `noise.clamp: true`, cada predicción `D` se sustituye por el embedding de su token más probable antes del paso de Euler (truco de *clamping* de Diffusion-LM, Li et al. 2022). Tras el último bloque se decodifica la predicción del denoiser en `sigma_min`, que equivale al paso final de EDM hasta `sigma=0`, y no el estado ruidoso. Sin ambas cosas la media posterior queda entre embeddings, la trayectoria conserva la dirección del ruido inicial (coseno 0,99) y el argmax final depende solo de la seed: dos modelos distintos generaban los mismos tokens aleatorios (ver D).

Después proyecta al vocabulario y toma argmax. El prompt limpio permanece fijo. El número de pasos por bloque está en config; no se usa directamente `transformers.generate`. La implementación recalcula el contexto y no está optimizada para inferencia. Con los pocos pasos de A/C las muestras pueden ser incoherentes: su propósito es verificar que el sampler termina.

## Instalación

Desde la raíz del repositorio, Python 3.11 recomendado:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Mac Intel, para A/B y tests, utilizar en cambio:

```bash
pip install -r requirements-macos-intel.txt
```

Este fichero utiliza PyTorch 2.2.2 y NumPy 1.26.4, compatibles con ese entorno. C se prepara para PyTorch 2.5.1 en Linux/CUDA. Registrar la versión efectiva: los resultados de CPU no validan kernels FP16 en T4.

Todos los entrenamientos se registran en W&B **offline** por defecto, sin API key. No requieren red durante A/B. Para registrar online, cambiar `wandb.modo` a `online` en YAML y ejecutar `wandb login`. Para sincronizar runs offline posteriormente:

```bash
wandb login
wandb sync wandb/offline-run-*
```

No se continúa silenciosamente si falta W&B. La ausencia de su proceso local también puede impedir entrenar en un sandbox que prohíba sockets localhost. Los tests unitarios no abren W&B.

## A: smoke completo en CPU

```bash
python run.py --smoke
```

Configuración: `configs/smoke.yaml`. Seis capas, tres bloques de dos capas, dimensión 48, tokenizer UTF-8 de 258 IDs, secuencias de 32 tokens y microbatch 2. Doce actualizaciones, con tres de calentamiento.

Usa ocho historias breves de entrenamiento y dos de validación de `tests/fixtures/stories.jsonl`, escritas como fixture de pruebas. **No son TinyStories descargadas** ni una evaluación científica. Todos los fragmentos de una historia permanecen en su split. Se conserva una máscara de validez separada para distinguir EOS de padding incluso si comparten ID.

Ejecuta carga, entrenamiento, evaluación, guardado, recarga y generación. Pasa si todos los bloques reciben alguna actualización, el estado recargado coincide y el pipeline termina en menos de 300 s. La generación recargada también debe coincidir con la original al fijar la misma seed; si no coincide se lanza error. No se exige buena calidad ni descenso significativo de pérdida.

## B: aprendizaje mínimo sobre historias repetidas

```bash
python run.py --config configs/overfit.yaml
```

Usa la misma arquitectura y ocho historias, repitiendo el loader hasta 300 pasos: hasta 100 actualizaciones por bloque. La evaluación de entrenamiento cubre sus fragmentos completos en esta configuración. El entrenamiento tiene un límite de 600 s y el pipeline un criterio de 1200 s.

Se evalúa antes y después sobre el mismo conjunto y realizaciones de ruido. La evaluación restaura el estado RNG, por lo que no altera el ruido ni el orden de entrenamiento. También se informa validación sobre historias distintas.

Pasa si cada pérdida de entrenamiento por bloque desciende al menos un 2 %, además de los contratos de checkpoint y generación. El umbral es diagnóstico y vive en `evaluacion.minimum_relative_drop`; no equivale a convergencia. Un descenso solo en el bloque de menor ruido puede ser trivial: comprobar todos los bloques y la validación. No se calcula perplexity del denoiser.

## D: aprendizaje y generalización en TinyStories

B solo demuestra memorización y C solo mide recursos en 42 pasos. D comprueba si el modelo **aprende a generalizar** con datos reales:

```bash
python scripts/prepare_pilot.py --config configs/prepare_learn.yaml   # ~35 s, 50 000 + 500 historias
python run.py --config configs/learn_baseline.yaml
python run.py --config configs/learn_diffusion_noise_c.yaml
python run.py --config configs/learn_diffusion_noise_shifted.yaml
```

Anchura SmolLM2 (576, 9 cabezas, 3 KV, MLP 1536, vocabulario 49152) para conservar el codebook y la geometría de ruido de C, pero 12 capas (tres bloques de cuatro). Secuencias de 256, batch 16, 3000 pasos (1000 por bloque), FP16, LR constante. Los 50 000 relatos dan ~11M tokens y 60 900 ventanas: los 3000 pasos no completan una época, así que nada se repite. Validación cada 300 pasos sobre 16 batches de historias distintas.

Pasa si la pérdida de **validación** de cada bloque baja al menos un 5 %, además de los contratos de checkpoint y tiempo. Se informa también `loss_sigma_max`: la CE del bloque 0 con `sigma` fijo en el máximo, donde el token actual es casi irrecuperable del ruido. Aproxima (algo optimista) la CE de siguiente token y es la única pérdida diffusion comparable con el baseline.

### Hallazgo: con el ruido de C, los bloques 1 y 2 no aprenden nada

Con dimensión 576 y codebook aleatorio, un modelo **sin entrenar** recupera el token con 98 % de acierto para `sigma <= 0.15`: el camino identidad inicial (gates adaLN-Zero a cero y skip EDM) basta. Los intervalos de C (`[1, 0.083]`, `[0.083, 0.03]`, `[0.03, 0.002]`) se ajustaron con la dimensión 48 de A/B, donde el mismo `sigma` oculta mucho más. Resultado: los bloques 1 y 2 tienen pérdida exactamente 0 desde el paso 0, gradiente nulo, y la prueba falla. El ~89 % de las muestras del bloque 0 cae en `sigma < 0.4`, donde casi tampoco hay tarea.

`learn_diffusion_noise_shifted.yaml` reubica la log-normal sobre la zona de transición (`min 0.1`, `max 5`, `log_mean -0.92`, `log_std 0.8`; fronteras `[5, 0.58, 0.30, 0.1]`). Es una propuesta para el experimento principal, no una decisión tomada.

### Resultados (RTX 2070, PyTorch 2.5.1, 25-10-2026)

| | pasa | val `loss_block_*` antes → después | CE siguiente token (val) | s entreno | pico VRAM |
| --- | --- | --- | --- | --- | --- |
| baseline | sí | 11.35 → 2.85 (ppl 17.2) | 2.85 | 573 | 5.0 GB |
| diffusion, ruido C | no | 2.74→0.84 · 0→0 · 0→0 | 15.45 → 4.85 | 508 | 5.2 GB |
| diffusion, ruido reubicado | sí | 15.0→3.80 · 9.75→2.27 · 1.93→0.47 | 18.43 → 4.31 | 507 | 5.2 GB |

Muestras (prompt `Once upon a time, there was a little`, 48 tokens, argmax):

- baseline: *girl named Lily. She loved to play outside in the sunshine. One day, she found a shiny rock on the ground. She picked it up and showed it to her mom.*
- diffusion, ruido reubicado: *girl named Lily. She loved to play outside with her friends. One day, Lily went to eat the hill. She saw a big Carpenter in the park.*
- diffusion, ruido C: *girl named Johnson. She loved opposites playing and watchful to see it in a big Ground. weddings they were revenge.*

Lectura: diffusion aprende lenguaje con actualizaciones independientes por bloque, pero con el mismo número de pasos queda detrás del baseline (4.31 frente a 2.85 de CE aproximada). Cada paso diffusion actualiza 4 capas y el baseline 12. El ruido reubicado mejora tanto la pérdida como la muestra. La RTX 2070 es de la misma generación (Turing) que la T4 pero más rápida: en T4 cada proceso debería seguir dentro de 20 min, aunque falta medirlo.

## C: piloto de arquitectura completa en CUDA/T4

En Colab seleccionar T4 y ejecutar en la raíz de un checkout del repositorio:

```bash
pip install -r requirements.txt
python scripts/prepare_pilot.py --config configs/prepare_pilot.yaml
python run.py --config configs/pilot_baseline.yaml
python run.py --config configs/pilot_diffusion.yaml
python scripts/compare_pilot.py \
  --baseline runs/prueba_c_baseline/report.json \
  --diffusion runs/prueba_c_diffusion/report.json
```

Los dos entrenamientos se ejecutan en **procesos separados** para liberar GPU y estados de optimizadores entre métodos. C rechaza equipos sin CUDA. El informe identifica la GPU real; una ejecución en otra GPU no demuestra el presupuesto de T4.

### Preparación de TinyStories

`prepare_pilot.py` resuelve las revisiones de dataset y tokenizer a SHAs de Hugging Face, descarga por streaming 256 historias de train y 32 de validation, y elimina textos duplicados entre ambos. Guarda IDs por historia, tokenizer local y `manifest.json` con revisiones y SHA-256 de cada archivo de IDs y de los archivos del tokenizer. No usa el test para elegir configuraciones. No cambia `src/data/splits.py` ni crea nuevos splits del dataset.

La selección es el primer subconjunto único en orden del split: apropiada para perfilado, no necesariamente representativa de calidad. Los fragmentos nunca mezclan historias. El entrenamiento comprueba los hashes del cache y no descarga datos ni tokenizer.

La primera preparación usa `main`; para reconstruir exactamente el mismo conjunto, copiar los SHAs del manifiesto a `dataset_revision` y `tokenizer_revision` de `configs/prepare_pilot.yaml`. Si el manifiesto ya existe, la preparación rechaza sobrescribirlo. Escoger otro `output` en config para otra selección. La preparación inicial y descargas se cronometran por separado del entrenamiento; quedan fuera de la regla 5/20 del pipeline con datos preparados.

### Qué mide

Cada método realiza 42 pasos: 12 de calentamiento y 30 medidos. Longitud útil 128, microbatch 2, AdamW y FP16 con GradScaler. Diffusion procesa 256 posiciones por ejemplo debido a la concatenación, aunque los tokens útiles se cuentan una vez.

Cada medición incluye entrega del batch, transferencia a GPU, forward, backward y actualización. CUDA se sincroniza antes y después. W&B y escritura de logs quedan fuera del tiempo por paso y dentro del tiempo total. No hay evaluación en medio de la ventana medida.

Antes del paso medido se reinicia el pico de VRAM, una vez que todos los optimizadores han creado estados en warmup. Los estados de todos los bloques permanecen residentes: el resultado no supone liberar estados entre actualizaciones. Se registran:

- Mediana y percentil 95 del tiempo por paso, tokens útiles/s.
- Pico de memoria **allocated** y **reserved**, en bytes, durante entrenamiento.
- Actualizaciones realmente aplicadas, tokens objetivo por bloque y pasos omitidos por FP16.
- Pérdidas iniciales/finales, parámetros totales/entrenables, GPU y versiones.
- Tiempo de entrenamiento, generación y pipeline, incluyendo cierre de W&B. El cronómetro del informe empieza al entrar al runner, después de importar Python/PyTorch; la regla 5/20 debe comprobarse también con el tiempo de pared del comando.

El guardado/recarga y la evaluación final suceden después de capturar el pico de entrenamiento. La recarga C se comprueba en CPU para evitar duplicar el modelo en la GPU. C verifica igualdad de estado; la igualdad de generación después de recarga se comprueba en A/B, no entre CPU y CUDA.

C pasa si hay actualizaciones de todos los bloques, 30 pasos medidos, ningún paso omitido, checkpoint correcto y tiempo total menor de 1200 s. El entrenamiento se corta si llega a 900 s; si eso impide completar la ventana, la prueba falla. Este límite se revisa entre pasos y no es un watchdog: evaluación/generación/guardado también pueden hacer que el criterio temporal falle.

Un OOM, pérdida no finita u otro error genera `failure.json` y salida no exitosa. Un criterio no alcanzado conserva `report.json` con `passed: false` y devuelve código 1.

La estimación de pasos en 20 minutos usa el percentil 95, descuenta la sobrecarga observada del pipeline y reserva el 20 % del presupuesto. Es conservadora y no constituye una garantía de convergencia. Un paso diffusion actualiza diez capas; un paso baseline actualiza treinta. Comparar también las actualizaciones y tokens **por bloque**, no solo pasos o tokens/s.

`compare_pilot.py` rechaza informes fallidos o con diferencias en datos, configuración, versiones, GPU o ventana medida. Muestra ratios de tiempo y memoria; no interpreta calidad ni tiempo hasta convergencia.

## Artefactos y reproducibilidad

Cada config escribe en su propio directorio `runs/`, ignorado por Git:

| Archivo | Contenido |
| --- | --- |
| `report.json` | Config completa, métricas, decisiones y criterio de aceptación |
| `checkpoint.pt` | Pesos, config, optimizadores, scaler, conteos y metadatos de datos |
| `steps.jsonl` | Pérdida, bloque, tiempo, norma de gradiente y actualización por paso |
| `sample.txt` | Prompt y continuación generada |
| `failure.json` | Diagnóstico de excepción, cuando la hay |

El checkpoint permite cargar pesos para inferencia y contiene estados de optimización para inspección. **No implementa aún reanudación exacta del entrenamiento**: faltarían restauración de RNG, posición del loader y una opción resume.

Semillas exclusivamente mediante `src/utils/seed.py`; valores en configs. No se ajustan estadísticas usando validación. No se aplica augmentation a validación ni test. Las rutas de configuración son relativas. El aplicativo no importa estos bucles.

```bash
pytest tests/ -q
python tools/tiiac_check.py .
```

Tests nuevos: aislamiento de parámetros, máscara causal con gates activos, ausencia de acceso al target limpio, formas, intervalos de ruido, padding/EOS, desplazamiento del baseline, evaluación fija con RNG restaurado y generación tras recarga. Los tests heredados del placeholder se conservan por compatibilidad y no validan el modelo de lenguaje.

## Referencias y siguientes pasos

- Paper: https://arxiv.org/html/2506.14202 (fijar versión al redactar ficha de replicación).
- Código oficial ViT: https://github.com/SakanaAI/DiffusionBlocks.
- Configuración SmolLM2: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json.

Tras A/B y C, decidir la propiedad y entrenamiento de embeddings/cabezas para el experimento principal, completar evaluación de calidad con prompts reservados y múltiples semillas, y estudiar importación/fine-tuning de pesos oficiales. El sampler inicial y el codebook congelado son limitaciones explícitas, no resultados del paper.

## Verificación local realizada

El 5 de octubre de 2026, en Mac Intel con Python 3.11, PyTorch 2.2.2 y NumPy 1.26.4:

- A: pipeline completo, checkpoint y generación recargada correctos; cuatro actualizaciones por bloque. El checker midió cinco segundos de pared, dentro de los cinco minutos.
- B: 300 pasos, 100 actualizaciones por bloque y criterio de descenso alcanzado en los tres bloques; checkpoint y generación recargada correctos.
- `pytest tests/ -q`: 18 tests aprobados.
- `tiiac_check.py`: 95/100; las observaciones restantes pertenecen a la cantidad de entradas y decisiones humanas en la bitácora de IA.
- C: tamaños de modelos comprobados sin asignar memoria real mediante el dispositivo `meta`; ejecución CUDA, FP16, tiempo y VRAM **pendientes de T4**.

Estos checks documentan funcionamiento del prototipo; la interpretación científica de sus resultados corresponde al equipo.
