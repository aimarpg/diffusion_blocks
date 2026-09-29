# Ficha de propuestas de proyecto · Hito H1

**Entrega: domingo 20 de septiembre, 23:59, en la Tarea «H1 · Propuestas de proyecto».**
**Decisión: domingo 27 de septiembre (hito H2).** Sin visto bueno no se sigue adelante.

Hasta **tres propuestas, en orden de preferencia**. La primera tiene que estar completa; la segunda y
la tercera son vuestras alternativas si la primera no pasa la puerta. Las tres tienen las mismas cuatro
comprobaciones.

---

## Cómo se entrega

1. Copiad esta página entera a un fichero `docs/propuestas.md` de vuestro repositorio y rellenadla ahí.
   Es Markdown: las tablas se editan en cualquier editor, y así queda en la historia de _commits_.
2. Marcad las casillas cambiando `[ ]` por `[x]` **solo cuando sea verdad**. Una casilla marcada sin
   evidencia en la tabla de debajo cuenta como no marcada.
3. Haced _commit_ y _push_, y en la Tarea de ALUD pegad **la URL del fichero en GitHub**. Entrega uno,
   cuenta por los dos.
4. El 27 de septiembre tendréis la decisión de cada propuesta en la retroalimentación de la Tarea. La
   aprobada se convierte en vuestro `docs/viabilidad.md`, que es el que se corrige con E1.

---

## Las dos vías

**Vía A · lista curada.** Uno de los doce papers ya comprobados, que están en el capítulo _Lista de
papers_ de este Libro: ResNet, U-Net, YOLO (v1), Vision Transformer, CycleGAN, Grad-CAM, SimCLR,
ejemplos adversarios (FGSM), Word2Vec, Transformer, BERT (solo preentrenamiento) y Neural
Collaborative Filtering. Sabemos que caben en la regla 5/20 y que tienen un aplicativo natural encima.
La ficha se rellena igual, pero es rápido. **Máximo dos parejas por paper**, por orden de entrega.

**Vía B · propuesta propia.** Cualquier paper o caso industrial que os interese, si la ficha pasa las
cuatro comprobaciones. Da más trabajo al principio, y ese trabajo de acotar es justo lo que venís a
aprender aquí.

---

## Pareja

|                                     |                                                                       |
| ----------------------------------- | --------------------------------------------------------------------- |
| **Nombres**                         |         Jaime Etxebarria, Aimar Pagonabarraga                                                              |
| **Repositorio**                     | https://github.com/aimarpg/diffusion_blocks                           |
| **Rol profesional al que apuntáis** | AI engineer |

---

## Propuesta 1 (Diffusion Blocks)

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | DIFFUSIONBLOCKS: BLOCK-WISE NEURAL NET- WORK TRAINING VIA DIFFUSION INTERPRETATION                                                   |
| **Autores y año**          |   Makoto Shing, Masanori Koyama, Takuya Akiba1, 2026                                                                  |
| **Enlace**                 | https://arxiv.org/pdf/2506.14202                                        |
| **Vía**                    | A (lista curada) / B (propuesta propia)                             |
| **Por qué este y no otro** | Es una técnica muy actual, puede resultar disruptiva y ofrece también una oportunidad de aportar al estado del arte. |

### Comprobación 1 · Datos

- [x] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [x] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           |                                                                |
| **Enlace de descarga**                |                                                                |
| **Tamaño**                            | _En MB/GB y en número de ejemplos_                             |
| **Licencia**                          |                                                                |
| **¿Hace falta registro o solicitud?** | _Si la respuesta es «hay que pedir acceso y tardan», es un no_ |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | https://github.com/SakanaAI/DiffusionBlocks                                                                                            |
| **Si no lo hay, ¿el paper da la arquitectura completa?** |                                                          |
| **Qué NO especifica el paper**                           | No da toda la información necesaria para una reproducción exacta de los experimentos. Cosas que se deberían saber para la implementación similar sería, entre otras cosas, dimensiones exactas de los embeddings, implementación exacta del patch embedding, positional embeddings, cómo se introduce la etiqueta de clase o la estructura exacta de cada DiT block.

### Comprobación 3 · Cómputo · la regla 5/20

* [x] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Escala del paper original**                  | **Reproducción en texto:** Transformer basado en DiT sobre un corpus de texto, siguiendo la configuración del paper para modelos de difusión de texto: 12 capas y 3 DiffusionBlocks. **Experimento autoregresivo:** modelo de aproximadamente 135M de parámetros basado en la arquitectura SmolLM2-135M, entrenado desde cero mediante DiffusionBlocks. El paper evalúa además diferentes configuraciones de bloques y distribuciones de los niveles de ruido/corrupción.                                                                                                       |
| **Escala que vais a hacer vosotros**           | Se realizarán dos líneas experimentales. **(1) Modelo autoregresivo:** se entrenará desde cero un modelo de aproximadamente **135M de parámetros basado en SmolLM2-135M**, utilizando **TinyStories** en lugar de un corpus de cientos de miles de millones de tokens. Posteriormente, este modelo será utilizado para estudiar el fine-tuning mediante DiffusionBlocks. Como referencia, también se realizará fine-tuning mediante el mismo procedimiento sobre el checkpoint preentrenado oficial de SmolLM2-135M, que no fue preentrenado originalmente con DiffusionBlocks. |
| **Tiempo estimado del entrenamiento completo** | Antes de ejecutar los experimentos completos se realizará una **prueba piloto** en el hardware definitivo, midiendo tiempo por paso/época, consumo máximo de VRAM y número de tokens procesados por segundo. Estas medidas se utilizarán para estimar el tiempo de cada configuración y seleccionar el número máximo de pasos compatible con el presupuesto computacional. Se reservará además un margen aproximado del **15–20 %** para evaluación, generación de muestras y ablaciones.                                                                                                 |
| **Qué se pierde al recortar**                  | El principal efecto del recorte será una menor convergencia y calidad absoluta respecto a modelos entrenados con cantidades de datos y cómputo mucho mayores. Esto afecta especialmente a la comparación con modelos preentrenados a gran escala. Por ello, el objetivo no será reproducir la calidad absoluta de un SmolLM2-135M entrenado sobre cientos de miles de millones de tokens, sino realizar una **comparación controlada entre métodos de entrenamiento**. Se priorizará que todos los modelos experimentales comparados utilicen el mismo corpus, presupuesto aproximado y condiciones de evaluación, de manera que las diferencias observadas puedan atribuirse al uso de DiffusionBlocks con mayor confianza.                                                                                                                                             |


### Comprobación 4 · Aplicativo

* [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qué construís encima**       | Una **aplicación web/API de generación y comparación de texto** que permita interactuar con las diferentes versiones del modelo SmolLM2-135M desarrolladas durante el proyecto. El usuario podrá seleccionar entre el modelo entrenado desde cero con DiffusionBlocks, el modelo preentrenado convencionalmente y posteriormente ajustado mediante DiffusionBlocks y, como referencia, el modelo SmolLM2-135M-Instruct disponible públicamente. La aplicación permitirá introducir un prompt y comparar las respuestas generadas por los distintos modelos, mostrando también los principales parámetros y metadatos de cada ejecución. |
| **Quién lo usaría y para qué** | Un estudiante, investigador o desarrollador interesado en **experimentar con diferentes estrategias de pretraining y fine-tuning de modelos de lenguaje**, que quiera observar de forma interactiva las diferencias entre un modelo entrenado convencionalmente y uno entrenado mediante DiffusionBlocks.                                                                                                                                                                                                                                                                                                                               |
| **Qué necesita del modelo**    | **Entrada:** un prompt o contexto textual y los parámetros de generación relevantes, como temperatura, longitud máxima y número de tokens a generar. **Salida:** texto generado, tiempo de inferencia y metadatos sobre el modelo y configuración utilizados. La aplicación podrá permitir generar con varias versiones del modelo a partir del mismo prompt para facilitar su comparación cualitativa.                                                         |
                            |

### Riesgo principal

|                                                    ||
| -------------------------------------------------- | ---------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** | El principal riesgo es que **el entrenamiento desde cero del modelo de 135M con DiffusionBlocks no converja adecuadamente o requiera más recursos computacionales de los disponibles**, especialmente al intentar mantener una escala de entrenamiento suficiente para obtener resultados representativos. También existe el riesgo de que el fine-tuning con DiffusionBlocks sobre el modelo preentrenado convencionalmente produzca resultados significativamente peores o inestables respecto al modelo preentrenado con DiffusionBlocks.                                                                                                                                                                                                                                                                               |
| **Qué haríais si pasa**                            | Primero se garantizará que el pipeline completo funciona mediante **entrenamientos reducidos**, utilizando una cantidad limitada de datos y pasos para comprobar la convergencia y detectar errores en la implementación. Si el entrenamiento completo resulta demasiado costoso, se reducirá progresivamente **el número de tokens utilizados, los pasos/épocas de entrenamiento, el número de configuraciones de DiffusionBlocks y/o el tamaño efectivo de los experimentos**, manteniendo siempre una configuración base que permita realizar la comparación principal. Si el entrenamiento desde cero de 135M no resulta viable, se priorizará la experimentación de fine-tuning y las ablaciones sobre un modelo de menor escala, dejando explícitamente documentada la reducción respecto al planteamiento original. |


### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |

---

## Propuesta 2 (Knowledge Graphs)

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | _Título completo_                                                   |
| **Autores y año**          |                                                                     |
| **Enlace**                 | _arXiv, DOI o web del paper_                                        |
| **Vía**                    | A (lista curada) / B (propuesta propia)                             |
| **Por qué este y no otro** | _Dos líneas. Idealmente conectado con el rol profesional de arriba_ |

### Comprobación 1 · Datos

- [ ] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [ ] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           |                                                                |
| **Enlace de descarga**                |                                                                |
| **Tamaño**                            | _En MB/GB y en número de ejemplos_                             |
| **Licencia**                          |                                                                |
| **¿Hace falta registro o solicitud?** | _Si la respuesta es «hay que pedir acceso y tardan», es un no_ |

### Comprobación 2 · Especificación

- [ ] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | _Sí (enlace) / No_                                                                                            |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | _Capas, dimensiones, función de pérdida, optimizador_                                                         |
| **Qué NO especifica el paper**                           | _Lo importante. Si la lista está vacía, no habéis leído el paper con suficiente atención: siempre falta algo_ |

### Comprobación 3 · Cómputo · la regla 5/20

- [ ] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | Smoltalk (https://huggingface.co/datasets/HuggingFaceTB/smoltalk) 1.04M de entrenamiento, 54.9k test                                                          |
| **Escala que vais a hacer vosotros**           | _Subset de N, M épocas, modelo reducido a…_                                                                       |
| **Tiempo estimado del entrenamiento completo** | _En minutos, en la máquina que vayáis a usar, y cómo lo habéis medido_                                            |
| **Qué se pierde al recortar**                  | _La respuesta honesta. «La métrica bajará de 0.72 a algo en torno a 0.6» es una buena respuesta; «nada» no lo es_ |
| **Hardware que vais a usar**                   | _Portátil / Colab gratuito / otro_                                                                                |

### Comprobación 4 · Aplicativo

- [ ] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | _Un buscador, un detector en vídeo, una API, un panel…_                           |
| **Quién lo usaría y para qué** | _Una frase. Si no se os ocurre, el proyecto no cumple el objetivo del aplicativo_ |
| **Qué necesita del modelo**    | _Entrada, salida, latencia aceptable_                                             |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** |                                                                                              |
| **Qué haríais si pasa**                            | _Útil: «si no converge, reducimos a MNIST y lo declaramos». Inútil: «nada, está controlado»_ |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |

---

## Propuesta 3 (alternativa)

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | _Título completo_                                                   |
| **Autores y año**          |                                                                     |
| **Enlace**                 | _arXiv, DOI o web del paper_                                        |
| **Vía**                    | A (lista curada) / B (propuesta propia)                             |
| **Por qué este y no otro** | _Dos líneas. Idealmente conectado con el rol profesional de arriba_ |

### Comprobación 1 · Datos

- [ ] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [ ] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           |                                                                |
| **Enlace de descarga**                |                                                                |
| **Tamaño**                            | _En MB/GB y en número de ejemplos_                             |
| **Licencia**                          |                                                                |
| **¿Hace falta registro o solicitud?** | _Si la respuesta es «hay que pedir acceso y tardan», es un no_ |

### Comprobación 2 · Especificación

- [ ] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | _Sí (enlace) / No_                                                                                            |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | _Capas, dimensiones, función de pérdida, optimizador_                                                         |
| **Qué NO especifica el paper**                           | _Lo importante. Si la lista está vacía, no habéis leído el paper con suficiente atención: siempre falta algo_ |

### Comprobación 3 · Cómputo · la regla 5/20

- [ ] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | _Dataset completo, N épocas, qué hardware, cuánto tardó_                                                          |
| **Escala que vais a hacer vosotros**           | _Subset de N, M épocas, modelo reducido a…_                                                                       |
| **Tiempo estimado del entrenamiento completo** | _En minutos, en la máquina que vayáis a usar, y cómo lo habéis medido_                                            |
| **Qué se pierde al recortar**                  | _La respuesta honesta. «La métrica bajará de 0.72 a algo en torno a 0.6» es una buena respuesta; «nada» no lo es_ |
| **Hardware que vais a usar**                   | _Portátil / Colab gratuito / otro_                                                                                |

### Comprobación 4 · Aplicativo

- [ ] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | _Un buscador, un detector en vídeo, una API, un panel…_                           |
| **Quién lo usaría y para qué** | _Una frase. Si no se os ocurre, el proyecto no cumple el objetivo del aplicativo_ |
| **Qué necesita del modelo**    | _Entrada, salida, latencia aceptable_                                             |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** |                                                                                              |
| **Qué haríais si pasa**                            | _Útil: «si no converge, reducimos a MNIST y lo declaramos». Inútil: «nada, está controlado»_ |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |
