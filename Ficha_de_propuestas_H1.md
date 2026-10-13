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

|                                |                                                                                                                                                                                    |
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

**### Identificación**

|                                |                                                                                                                                                                                                                                                                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ****Paper o referencia****     | *Towards Practical GraphRAG: Efficient Knowledge Graph Construction and Hybrid Retrieval at Scale*                                                                                                                                                                                                                    |
| ****Autores y año****          | Congmin Min, Sahil Bansal, Joyce Pan, Abbas Keshavarzi, Rhea Mathew y Amar Viswanathan Kannan, 2025                                                                                                                                                                                                                   |
| ****Enlace****                 | *https://arxiv.org/abs/2507.03226*                                                                                                                                                                                                                                                                                    |
| ****Vía****                    | A (lista curada)                                                                                                                                                                                                                                                                                                      |
| ****Por qué este y no otro**** | El paper aborda directamente la construcción eficiente de Knowledge Graphs a partir de texto, comparando técnicas basadas en LLMs y dependency parsing. Además, permite plantear una aportación propia mediante un sistema adaptativo que combine ambos enfoques según la complejidad o confianza de cada extracción. |

**### Comprobación 1 · Datos**

* [x] Son ****públicos y descargables hoy****. Enlace que funciona, no una promesa.
* [x] La licencia permite el uso académico.

|                                           |                                                                                                 |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
| ****Dataset****                           | **REBEL (Relation Extraction By End-to-end Language generation)**                               |
| ****Enlace de descarga****                | *https://huggingface.co/datasets/Babelscape/rebel-dataset*                                      |
| ****Tamaño****                            | Aproximadamente **220.000 ejemplos** de texto con triples de relaciones extraídos de Wikipedia. |
| ****Licencia****                          | **CC BY-SA 3.0**                                                                                |
| ****¿Hace falta registro o solicitud?**** | **No.** Descarga pública y directa desde Hugging Face.                                          |
|                                           
|                                                   
| ****Dataset complementario****                           | **WebNLG / WebNLG2**                               |
| ****Enlace de descarga****                | *https://gitlab.com/webnlg/webnlg-dataset?utm_source=chatgpt.com*                                      |
| ****Tamaño****                            | Del orden de decenas de miles de triples y pares texto–triple, dependiendo de la versión utilizada. |
| ****Licencia****                          | **CC BY-NC-SA 4.0**                                                                                |
| ****¿Hace falta registro o solicitud?**** | **No.** No. Descarga pública.                                          |


### Comprobación 2 · Especificación

* [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como para implementarla sin adivinar.

|                                                              |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ****¿Hay código de referencia?****                           | **Sí, parcialmente.** El trabajo dispone de implementación asociada al método y describe los principales componentes del pipeline. Los autores proporcionan información sobre el proceso de extracción de relaciones y referencias (NER, dependency parsing, coreference resolution, reglas adicionales para estructuras más complejas).                                                                                                                                                                                                                                                                                                         |
| ****Si no lo hay, ¿el paper da la arquitectura completa?**** | El sistema está compuesto por módulos de procesamiento de texto, dependency parsing, extracción de entidades y relaciones, utilización de LLMs y posterior fusión/normalización. El paper describe estos componentes y su funcionamiento.                                                                                                                                                                                                            |
| ****Qué NO especifica el paper****                           | El paper no proporciona todos los detalles necesarios para reproducir exactamente el sistema original, como algunos parámetros y decisiones de implementación del pipeline. El mayor contratiempo es que os datasets originales proceden de un entorno empresarial de SAP y no están disponibles públicamente, por lo que no es posible reproducir exactamente todos los experimentos. |

### Comprobación 3 · Cómputo · la regla 5/20

* [x] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ****Escala del paper original****                  | El paper trabaja con aproximadamente **550 documentos PDF**, que se dividen en unos **2.000 chunks**, además de conjuntos de evaluación específicos del entorno SAP. La extracción combina dependency parsing y LLMs y está orientada a procesamiento de documentos a gran escala.                                                                                                                                                                                                                                                                                   |
| ****Escala que vais a hacer vosotros****           | Se utilizará un **subset de aproximadamente 10.000–20.000 ejemplos de REBEL** para los experimentos iniciales. Se realizarán varias configuraciones de extracción sobre el mismo subconjunto para poder comparar dependency parsing, LLM-based, hybrid y nuestro método adaptativo. Se utilizarán modelos preentrenados de tamaño reducido o APIs/modelos disponibles para la extracción.                                                                                                                                                       |
| ****Tiempo estimado del entrenamiento completo**** | Para el procesamiento mediante dependency parsing, el tiempo esperado será del orden de **minutos a pocas horas**, dependiendo del tamaño del subset. La parte LLM dependerá principalmente del modelo utilizado y del número de llamadas. Se realizará una prueba piloto con **500–1.000 ejemplos** para medir el tiempo real y extrapolarlo al conjunto completo antes de ejecutar los experimentos. |
| ****Qué se pierde al recortar****                  | Al utilizar únicamente una fracción del dataset y modelos LLM más pequeños, los resultados pueden ser menos representativos que los obtenidos a escala completa y la calidad de las extracciones puede disminuir.                                                                                         |
| ****Hardware que vais a usar****                   | **GPU disponible mediante Google Colab o alquiler de GPUs online para determinadas partes de los experimentos.** El dependency parsing y las fases de procesamiento tradicionales se ejecutarán localmente. Para los experimentos con LLM se barajarán las opciones de GPUs.  |                  
                                                  

### Comprobación 4 · Aplicativo

* [x] Hay una capa de servicio natural encima de la replicación.

|                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ****Qué construís encima****       | Una **aplicación web/API para generación de Knowledge Graphs**. El usuario podrá introducir un texto o documento, seleccionar el método de generación (dependency-based, LLM-based o nuestro método adaptativo) y visualizar las entidades y relaciones extraídas en forma de Knowledge Graph. |
| ****Quién lo usaría y para qué**** | Investigadores, estudiantes o desarrolladores que quieran **transformar  documentos de texto en un Knowledge Graph.                                                                                                                                                                                                                            |
| ****Qué necesita del modelo****    | **Entrada:** texto o documento proporcionado por el usuario y método de extracción seleccionado. **Salida:** entidades, relaciones y Knowledge Graph visualizable.                                                                                                 |

### Riesgo principal

|                                                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ****Qué es lo que más probablemente va a salir mal**** | Que el método adaptativo no consiga reducir suficientemente el uso del LLM manteniendo una calidad comparable a la extracción completamente basada en LLMs, Por otro lado, que la implementación de los diferentes pipelines resulte demasiado compleja para los recursos y tiempo disponibles.                                                                                                                                                         |
| ****Qué haríais si pasa****                            | Reduciremos primero el tamaño del dataset y el número de experimentos, manteniendo únicamente las comparaciones esenciales. Si el sistema adaptativo no proporciona una mejora clara, se mantendrá como resultado la comparación entre dependency parsing, LLM y el método híbrido del paper. |

### Visto bueno del profesor (H2) · no rellenar

|                     |                                            |
| ------------------- | ------------------------------------------ |
| ****Decisión****    | Aprobado / Aprobado con recorte / Devuelto |
| ****Condiciones**** |                                            |
