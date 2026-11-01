# Contexto del proyecto

<!--
Fichero de contexto para asistentes de IA (Claude Code, Copilot, Cursor…).
Se explica en la sesión A1 del 9 de noviembre.

Dos criterios para escribirlo:
  1. Va lo que le dirías a alguien que se incorpora hoy al proyecto.
  2. Si te encuentras repitiendo la misma corrección tres veces, esa
     corrección va aquí.

Dos páginas se leen enteras. Quince, no.
-->

## Qué es esto

Adaptación de DiffusionBlocks a una arquitectura SmolLM2-135M sobre TinyStories.
Objetivo inicial: validar ejecución, aprendizaje mínimo y recursos con pruebas A/B/C.
Después se construirá un comparador de generación de texto. Diseño en docs/pruebas_viabilidad.md.

## Restricciones del curso

- **Regla 5/20:** `python run.py --smoke` en menos de 5 min en CPU;
  el entrenamiento completo en menos de 20 min en una T4 de Colab gratuito.
- Todo entrenamiento se registra en Weights & Biases.
- Las semillas se fijan **solo** en `src/utils/seed.py`.
- Los hiperparámetros viven **solo** en `configs/`. Ninguno escrito a fuego.

## Cómo se ejecuta

```bash
python run.py --smoke                          # pipeline entero en miniatura
python run.py --config configs/default.yaml    # entrenamiento completo
pytest tests/ -q                               # tests
python tools/tiiac_check.py .                  # comprobador de reproducibilidad
```

## Convenciones

- Sin rutas absolutas, nunca.
- Toda función pública lleva docstring y anotaciones de tipo.
- **La normalización usa estadísticas de entrenamiento, no del conjunto completo.**
  Lo contrario es una fuga de datos: no da error e infla la métrica.
- El `data augmentation` se aplica a train, jamás a validación ni a test.
- `src/app/` no importa nada de `src/training/`: el aplicativo solo carga un
  checkpoint y responde.

## Lo que NO hay que tocar

- `src/data/splits.py`: los splits están congelados desde el hito H3.
- <lo que decidáis vosotros>

## Estado actual

<Actualiza esto cuando cambie. Es lo que evita que el asistente reinvente
piezas que ya existen.>

- Hecho: pruebas A/B/C configurables, Transformer con GQA/RoPE, denoising independiente, sampler, cache de texto y documentación. A/B usan fixture offline; C usa TinyStories y tokenizer oficial.
- Verificado en CPU Mac Intel/PyTorch 2.2.2: A/B pasan, 18 tests pasan y tiiac_check da 95/100. Codebook congelado en ambos pilotos; no importar aún pesos oficiales.
- Prueba D (configs/learn_*.yaml, 50k historias): baseline y diffusion con ruido reubicado pasan en RTX 2070. Con el ruido de C los bloques 1-2 tienen pérdida 0 desde el inicio (d=576): no reutilizar esos intervalos sin revisarlos.
- Sampler diffusion: clamping (`noise.clamp`) y decodificación de la predicción final; sin ello generaba ruido.
- En curso: medición de viabilidad C pendiente de hardware CUDA/T4.
- Pendiente: ejecutar C en T4, decidir embeddings/cabezas del experimento principal, evaluar calidad, fine-tuning y aplicativo.
- Los módulos placeholder y sus tests se conservan por compatibilidad; run.py ejecuta src/training/probes.py.

- Experimento E implementado: configs/full_*.yaml, 30 capas (3×10 en diffusion), codebook congelado, acumulación por tokens, LR warmup/coseno, checkpoints last/best y reanudación de RNG/orden de datos. Launcher: scripts/train_full.py; guía: docs/full_training.md.
- E prepara 200k train + 2k validation + 2k test local reservado del split oficial validation. scripts/evaluate_full.py exporta métricas, continuaciones y plantilla de evaluación ciega; no usar test para ajustar hiperparámetros.
- Verificado E en RTX 5000 Ada/PyTorch 2.5.1: las tres variantes ejecutan 6 actualizaciones FP16 con secuencia 256 y batch efectivo 16, sin pasos omitidos (fixture offline, no evidencia de calidad). Preset course: 900 s de entrenamiento; extended: 3600 s, fuera de la regla 5/20. Entrenamiento real E y C en T4 pendientes.
- Verificación de implementación E (1-noviembre-2026): 26 tests pasan (incluye reanudación exacta CUDA FP16 y CPU), smoke A pasa y tiiac_check da 97/100. No se ha lanzado entrenamiento extenso E.
- Preset dataset añadido: --budget dataset recorre epochs completas (por defecto 1) sin límite temporal ni techo fijo de pasos. Deriva pasos del cache y no repite el batch final. Cobertura por modelo; diffusion reparte batches entre bloques. 28 tests pasan, incluyendo cobertura exacta y reanudación en modo época.
- Benchmark de microbatch: scripts/benchmark_batches.py compara 32/48 con acumulación 1 para las tres variantes; configs/batch_calibration.yaml declara candidatos. Procesos aislados, salidas separadas en runs/batch_calibration y tabla/comparison.json; no modifica configs de entrenamiento.

- Corrección de evaluación: configs/evaluate_full.yaml genera en FP32 como D, con salida runs/full_evaluation_fp32*. La configuración FP16 anterior queda en configs/evaluate_full_fp16.yaml; no se cambian checkpoints ni precisión de entrenamiento.

- D reevaluado desde runs/Archive con configs/evaluate_probe_d.yaml sobre el mismo test/prompts quick FP32 de E. Salidas runs/probe_d_evaluation_fp32_quick y comparación runs/probe_d_vs_full_dataset_quick (JSON/CSV y continuaciones emparejadas). D→E: baseline CE 2.833→1.836; loss_sigma_max original 4.905→3.627 y ajustado 4.359→3.177. Evaluación preliminar; cambian profundidad, datos y entrenamiento. 32 tests pasan. Cache externo opt-in verifica tokenizer, hash del subset train y ausencia de solapamiento con test.

- Demo CLI implementada: python demo.py (menú interactivo), --list, --model d-adjusted e-adjusted --prompt "...", --checkpoint y exportación JSONL con --output. Defaults/rutas en configs/demo.yaml; guía docs/demo_cli.md. src/app/generate.py no importa entrenamiento y solo carga checkpoint/tokenizer local. FP32 y sampler iguales a evaluación; seis checkpoints D/E verificados con coincidencia exacta de los 48 tokens de un prompt evaluado. 37 tests pasan. Evidencia local: runs/demo_validation/verification.json.
- Presentación CLI refinada: grupos D/E, acento cian automático en TTY, indicadores de actividad con tiempo, texto adaptado al ancho y resumen de generación. /models cambia el checkpoint durante la sesión; --plain y NO_COLOR desactivan color/animaciones. Sin nuevas dependencias; 37 tests pasan.
- CLI con interfaz de sesión: cabecera compacta, selector de modelos con flechas/Enter en TTY POSIX (números/nombres como fallback), compositor con contexto activo, paneles de respuestas y /help. Esc en /models conserva el modelo actual; historial solo en memoria. Restauración de termios/cursor verificada con pseudo-terminales; 40 tests pasan. Inferencia y configuración de generación sin cambios.

- Distribución de modelos preparada para GitHub Releases: scripts/prepare_model_release.py crea adjuntos en runs/model_release y configs/model_assets.json con hashes; scripts/download_models.py restaura modelos seleccionados y tokenizer en sus rutas originales con SHA-256 y reemplazo atómico. Configuración en configs/model_release.yaml y guía docs/model_release.md. Informes quick FP32 D/E copiados a reports/model_comparison, con hashes de los seis checkpoints contrastados. 46 tests pasan, incluyendo integridad, selección de modelos y preservación de archivos ante descargas fallidas. Publicación remota pendiente; no se incorporan pesos a Git.
