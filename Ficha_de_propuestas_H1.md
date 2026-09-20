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

## Propuesta 1

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

- [ ] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | DiT-S/2 sobre CIFAR-10, con 100 épocas, batch size 512, AdamW y LR 5e-5. Para el experimento de 3 bloques, el modelo tiene 12 capas y se divide en 3 bloques de 4 capas.                                                          |
| **Escala que vais a hacer vosotros**           | DiT-S/2 + CIFAR-10, manteniendo la arquitectura y configuración del paper en la medida en que estén especificadas. Como primera fase, se realizará una reproducción reducida de 10–20 épocas para validar que el pipeline funciona.                                                                       |
| **Tiempo estimado del entrenamiento completo** | Se realizará primero una prueba piloto de 1 época en el hardware definitivo y se medirá el tiempo real por época. Basado en eso se estimará el tiempo total añadiendo un margen del 15–20 % para evaluación.                                         
| **Qué se pierde al recortar**                  | Puede producir una peor calidad de generación y resultados FID menos estables que los del paper. Por tanto, los resultados obtenidos con este entrenamiento no se interpretarán como una reproducción exacta. |
| **Hardware que vais a usar**                   | GPU en la nube (Google Colab) u otra GPU disponible en los laboratorios de la universidad. También una GPU disponible.
|

### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | No tenemos muy claro qué aplicativo crearemos como tal, pero una posibilidad sería una aplicación web/API de generación de imágenes o texto que permita generar contenido utilizando el modelo Diffusion Transformer o Masked Diffusion, por ejemplo. |
| **Quién lo usaría y para qué** | Un usuario que quiera experimentar y comparar distintos métodos de entrenamiento y fine-tuning. |
| **Qué necesita del modelo**    | 
- Entrada: condición de generación y configuración del modelo. 
- Salida: imagen o texto generado y metadatos de la ejecución.                                          |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** |    No conseguir reproducir una generación de imágenes o texto con calidad suficiente o no disponer de recursos computacionales suficientes para entrenar o fine-tunear propiamente la arquitectura escogida.                                                                           |
| **Qué haríais si pasa**                            | Primero garantizar que el pipeline funciona con un entrenamiento reducido sobre la arquitectura seleccionada; y luego si el entrenamiento completo resulta demasiado costoso, reducir épocas y el número de configuraciones. |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |

---

## Propuesta 2 (alternativa)

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
