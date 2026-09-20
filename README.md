# Comparación de arquitecturas CNN — CIFAR-10

Actividad 2 del Checkpoint 3, Sistemas de aprendizaje profundo.

El entregable principal es **[Arquitecturas_cnn.ipynb](Arquitecturas_cnn.ipynb)**:
dos CNN propias y MobileNetV2 preentrenada en ImageNet, con aumento de datos,
regularización, entrenamiento, evaluación y conclusión experimental.

## Reproducir

Usar Python 3.12 en un entorno virtual:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python ejecutar_notebook.py
```

La primera ejecución descarga CIFAR-10 y los pesos oficiales de MobileNetV2.
Se necesita conexión a Internet y espacio para los datos. Los archivos descargados
se guardan en `.keras/`, excluido de Git. El entrenamiento puede tardar en CPU.
El notebook también se puede abrir en Jupyter o Google Colab y ejecutar en orden.
En Colab se puede usar su TensorFlow preinstalado; las versiones verificadas para
el experimento local están en `requirements.txt`.

`preparar_notebook.py` reconstruye el notebook inicial sin resultados (sobrescribe
el archivo). No es necesario usarlo para consultar ni ejecutar la entrega.

## Protocolo

- 12 000 imágenes de entrenamiento, 3 000 de validación y 10 000 de prueba.
- Selección estratificada; entrenamiento y validación salen exclusivamente del
  conjunto oficial de entrenamiento. Las 35 000 imágenes restantes no se usan.
- Semilla 42, batch 128, máximo 12 épocas por modelo, Adam 0.001.
- EarlyStopping y reducción de tasa según pérdida de validación; restauración
  de mejores pesos. Selección de arquitectura por accuracy de validación.
- Aumento solo en entrenamiento: reflexión horizontal y traslación.
- Accuracy y F1 macro de prueba, tiempos de fit, parámetros y matrices de confusión.

MobileNetV2 tiene su base congelada, entrada ampliada a 96 × 96 y píxeles en [-1, 1].
Su preentrenamiento externo impide interpretar este experimento como una comparación
con idéntico presupuesto total de datos y cómputo. Una sola semilla y el subconjunto
de datos tampoco permiten afirmar superioridad universal o significancia estadística.

## Resultados guardados

Ejecución local en CPU, TensorFlow 2.21.0, 20 de septiembre de 2026:

| Modelo | Accuracy validación | Accuracy prueba | F1 macro prueba | Tiempo de fit |
|---|---:|---:|---:|---:|
| CNN básica | 58.00 % | 57.84 % | 0.5725 | 24.24 s |
| CNN profunda | 67.30 % | 67.14 % | 0.6680 | 143.04 s |
| MobileNetV2 | 84.33 % | 84.03 % | 0.8385 | 241.90 s |

MobileNetV2 fue seleccionada por su accuracy de validación. La CNN básica fue la
más rápida. Los tres modelos completaron 12 épocas; se restauraron los pesos de
las épocas 10, 10 y 11, respectivamente, según la menor pérdida de validación.

La ejecución genera `resultados/comparacion.csv`, `historiales.json`,
`predicciones.npz`, gráficos PNG y `conclusion.md`. El notebook conserva sus salidas.
Los tiempos dependen del equipo y no incluyen descargas ni evaluación.

Para comprobar el entregable sin volver a entrenar:

```bash
python verificar_resultados.py
```

La verificación recalcula accuracy y F1 a partir de las 10 000 predicciones de
prueba de cada modelo, contrasta la época seleccionada y valida el notebook.

Si el servidor de CIFAR-10 descarga demasiado despacio en Windows,
`python descargar_datos.py` permite obtener el mismo archivo por segmentos
con `curl.exe` y comprobar el SHA-256 oficial antes de ejecutar el notebook.

## Fuentes

- [CIFAR-10](https://keras.io/api/datasets/cifar10/)
- [MobileNetV2](https://keras.io/api/applications/mobilenet/)
- [Transfer learning en Keras](https://keras.io/guides/transfer_learning/)
- [TensorFlow: instalación y soporte por plataforma](https://www.tensorflow.org/install/pip)
