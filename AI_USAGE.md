# Bitácora de uso de IA

Obligatoria según la guía docente. Vale **4 puntos de E2**.

**No se puntúa usar poco.** Se puntúa el criterio: una entrada que dice
«se lo pedimos, nos dio esto, lo rechazamos porque…» vale más nota que no
haber usado ninguna herramienta.

**La línea roja:** la IA se puede usar para comprender y generar código,
citándola. **No** para el análisis automático de los resultados ni para la
discusión reflexiva sobre el trabajo. Eso es lo que evalúa la asignatura, y se
comprueba en la defensa.

---

| Fecha      | Tarea                                                | Herramienta | Qué devolvió                                                                                                                      | Qué hicimos y por qué                                                                                                                                     |
| ---------- | ---------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3-octubre  | Encontrar ambigüedades en el paper                   | Claude Code | Una lista de posibles ambigüedades del paper y la implementación de referencia.                                                   | Revisarlas, escoger las que tenían sentido, buscar soluciones y redactar el README.                                                                       |
| 3-octubre  | Añadir el dataset a get_data.py                      | Claude Code | Las URLs y hashes                                                                                                                 | Comprobar que funciona y limpiar el código                                                                                                                |
| 5-octubre  | Implementación de pruebas A/B/C y documentación      | Codex       | Pipeline de texto, Transformer, denoisers independientes, máscaras, sampler, perfilado y tests.                                   | Pendiente revisión humana del método y ejecución C en T4; A/B y tests se verifican localmente. No se ha elaborado una discusión científica de resultados. |
| 21-octubre | Logging a W&B con métricas separadas por bloque      | Claude Code |                                                                                                                                   | Comprobar que funciona.                                                                                                                                   |
| 25-octubre | Prueba D: aprendizaje en TinyStories (50k historias) | Claude Code | Configs learn_*.yaml, validación periódica y loss_sigma_max en probes.py, ejecuciones en RTX 2070 y sección D de docs.            | Revisar el método. La interpretación de resultados de la sección D la redactó la IA: tenemos que rehacerla nosotros.                                      |
| 25-octubre | Generación de diffusion que solo producía ruido      | Claude Code | Diagnóstico (el argmax final dependía solo de la seed) y propuesta de clamping de Diffusion-LM y decodificar la predicción final. | Aceptado tras comparar muestras de ambos checkpoints antes y después del cambio.                                                                          |
| 25-octubre | Intervalos de ruido con dimensión 576                | Claude Code | Aviso de que con el ruido de C los bloques 1-2 tienen pérdida 0 desde el inicio y propuesta de ruido reubicado.                   | No adoptado aún en C: queda como config aparte (learn_diffusion_noise_shifted.yaml) hasta decidir si es un cambio permanente.                             |
