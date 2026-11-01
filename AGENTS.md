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
