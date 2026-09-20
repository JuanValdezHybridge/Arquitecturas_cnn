"""Construye el notebook de la actividad; no ejecuta entrenamientos."""
from pathlib import Path
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# Evaluación comparativa de arquitecturas convolucionales
**Actividad 2 · Checkpoint 3 · Sistemas de aprendizaje profundo**

Se comparan dos CNN propias y MobileNetV2 preentrenada en ImageNet para clasificar CIFAR-10.
El objetivo es evaluar precisión, F1 macro, sobreajuste, parámetros y costo de entrenamiento.

## Diseño experimental
Se usan 12 000 imágenes de entrenamiento y 3 000 de validación, seleccionadas de forma
estratificada del conjunto oficial de entrenamiento. Las 10 000 imágenes oficiales de prueba
se reservan para la evaluación final. Este presupuesto permite ejecutar el experimento en CPU;
no representa el rendimiento máximo de estas arquitecturas. La semilla es 42.

Los tres modelos usan exactamente las mismas particiones, batch de 128, Adam y hasta 12 épocas,
con selección de pesos por la menor pérdida de validación. El conjunto de prueba no interviene
en la selección. Se realiza una sola ejecución por modelo: las diferencias pequeñas requieren
repetir con más semillas antes de considerarlas concluyentes.
''')
code('''import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(__import__("pathlib").Path.cwd() / ".keras"))
import json
import time
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
from IPython.display import display, Markdown

SEED = 42
BATCH = 128
EPOCHS = 12
keras.utils.set_random_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)
print("Python:", platform.python_version(), "TensorFlow:", tf.__version__)
print("Dispositivos:", tf.config.list_physical_devices())
''')
md('''## Datos y prevención de fugas
CIFAR-10 contiene imágenes RGB de 32 × 32 y diez clases balanceadas.
Se conservan valores de píxeles entre 0 y 255: cada modelo incorpora su propio escalado.
El aumento de datos solo se activa durante el entrenamiento; validación y prueba no se aumentan.
''')
code('''(x_all, y_all), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_all, y_test = y_all.ravel(), y_test.ravel()
idx_train, idx_rest = train_test_split(np.arange(len(y_all)), train_size=12000,
                                      stratify=y_all, random_state=SEED)
idx_val, _ = train_test_split(idx_rest, train_size=3000,
                            stratify=y_all[idx_rest], random_state=SEED)
x_train, y_train = x_all[idx_train], y_all[idx_train]
x_val, y_val = x_all[idx_val], y_all[idx_val]
CLASSES = ["avión", "automóvil", "ave", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión"]
assert not set(idx_train) & set(idx_val)
for name, y in [("Entrenamiento", y_train), ("Validación", y_val), ("Prueba", y_test)]:
    print(name, len(y), "imágenes; por clase:", np.bincount(y))
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for label, ax in enumerate(axes.flat):
    ax.imshow(x_train[np.flatnonzero(y_train == label)[0]])
    ax.set_title(CLASSES[label]); ax.axis("off")
plt.tight_layout(); plt.show()

def dataset(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(y), seed=SEED)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
val_ds = dataset(x_val, y_val)
test_ds = dataset(x_test, y_test)
''')
md('''## Arquitecturas y justificación de hiperparámetros

| Modelo | Diseño | Justificación |
|---|---|---|
| CNN básica | Conv 32 → pool → Conv 64 → pool → Flatten → Dense 64 | Referencia de baja profundidad; Flatten conserva posición, pero aumenta los parámetros de la cabeza. |
| CNN profunda | Dos convoluciones por bloque, 32/64/128 filtros, BatchNorm, pooling y GlobalAveragePooling | Mayor capacidad para combinar patrones; GAP reduce la cabeza densa y BatchNorm estabiliza las activaciones. |
| MobileNetV2 | Base ImageNet congelada, entrada 96 × 96, GAP y cabeza de diez clases | Transfer learning reutiliza características aprendidas; convoluciones separables reducen el costo frente a arquitecturas clásicas más grandes. |

Las CNN propias usan filtros 3 × 3, ReLU y padding same, que capturan patrones locales sin
reducir el tamaño antes del pooling 2 × 2. Ambas usan L2=0.0001 y Dropout; la profunda aplica
Dropout creciente (0.15–0.35). Las tres usan reflexión horizontal y traslación de hasta 10 %,
transformaciones razonables para los objetos de CIFAR-10. No se usan inversiones verticales.
Adam con tasa 0.001 es el punto inicial común. ReduceLROnPlateau reduce la tasa ante estancamiento
y EarlyStopping evita continuar sin mejora, restaurando los mejores pesos.

MobileNetV2 requiere píxeles en [-1, 1]; Rescaling(1/127.5, offset=-1) implementa ese escalado.
Se amplía a 96 × 96 para aprovechar los pesos publicados a esa resolución; esto no añade detalle
real. La base se invoca con training=False para mantener sus estadísticas de BatchNorm.
Su preentrenamiento aporta datos y cómputo externos: la comparación evalúa estrategias prácticas,
no un presupuesto total de entrenamiento equivalente.
''')
code('''def augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomTranslation(0.1, 0.1, seed=SEED + 1),
    ], name="aumento")

def cnn_basica():
    inputs = keras.Input((32, 32, 3))
    x = layers.Rescaling(1 / 255.0)(augmentation()(inputs))
    for filters in (32, 64):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    return keras.Model(inputs, layers.Dense(10, activation="softmax")(x), name="cnn_basica")

def cnn_profunda():
    inputs = keras.Input((32, 32, 3))
    x = layers.Rescaling(1 / 255.0)(augmentation()(inputs))
    for filters, dropout in [(32, 0.15), (64, 0.25), (128, 0.35)]:
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, layers.Dense(10, activation="softmax")(x), name="cnn_profunda")

def transfer_mobilenet():
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
    base.trainable = False
    inputs = keras.Input((32, 32, 3))
    x = augmentation()(inputs)
    x = layers.Resizing(96, 96)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inputs, layers.Dense(10, activation="softmax")(x), name="mobilenet_v2")

BUILDERS = {"CNN básica": cnn_basica, "CNN profunda": cnn_profunda,
            "MobileNetV2": transfer_mobilenet}
''')
md('''## Entrenamiento y evaluación
Se reinicia la semilla y se crea un nuevo optimizador, conjunto de entrenamiento y callbacks
para cada modelo. Se guardan los historiales, métricas y predicciones para auditar el reporte.
Los tiempos incluyen únicamente fit, sin descarga de datos/pesos ni evaluación.
La pérdida incluye la penalización L2 donde existe; accuracy y F1 son las métricas de comparación
principal. El mejor modelo se selecciona por accuracy de validación tras restaurar sus pesos.
''')
code('''histories, rows, predictions = {}, [], {}
for name, builder in BUILDERS.items():
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model = builder()
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.CSVLogger(str(RESULTS / (model.name + "_epocas.csv"))),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    start = time.perf_counter()
    history = model.fit(dataset(x_train, y_train, True), validation_data=val_ds,
                        epochs=EPOCHS, callbacks=callbacks, verbose=2)
    seconds = time.perf_counter() - start
    histories[name] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    pred = model.predict(test_ds, verbose=0).argmax(axis=1)
    predictions[name] = pred
    rows.append(dict(modelo=name, parametros=model.count_params(),
                     entrenables=sum(int(np.prod(w.shape)) for w in model.trainable_weights),
                     epocas=len(history.history["loss"]),
                     mejor_epoca=int(np.argmin(history.history["val_loss"]) + 1),
                     segundos=seconds, val_accuracy=val_acc, test_accuracy=test_acc,
                     test_f1_macro=f1_score(y_test, pred, average="macro"), test_loss=test_loss))
    pd.DataFrame(rows).to_csv(RESULTS / "comparacion.csv", index=False)
    (RESULTS / "historiales.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    np.savez_compressed(RESULTS / "predicciones.npz", y_test=y_test, **predictions)
    print(name, "finalizado:", rows[-1], flush=True)
comparison = pd.DataFrame(rows).set_index("modelo")
display(comparison.round(4))
''')
md('''## Estadística y gráficos
Las curvas permiten detectar estancamiento y separación entre entrenamiento y validación.
El aumento y Dropout están activos en entrenamiento; por ello una accuracy de validación
mayor que la de entrenamiento no implica por sí sola un error. Las matrices de confusión
muestran qué clases presentan dificultades, más allá del promedio global.
''')
code('''fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, (name, h) in enumerate(histories.items()):
    epochs = np.arange(1, len(h["loss"]) + 1)
    for row, metric in enumerate(["accuracy", "loss"]):
        ax = axes[row, col]
        ax.plot(epochs, h[metric], label="Entrenamiento")
        ax.plot(epochs, h["val_" + metric], label="Validación")
        ax.set(title=name, xlabel="Época", ylabel=metric)
        ax.legend(); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(RESULTS / "curvas.png", dpi=150); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
comparison[["val_accuracy", "test_accuracy", "test_f1_macro"]].plot.bar(ax=axes[0], rot=0)
axes[0].set_ylim(0, 1); axes[0].set_ylabel("Puntuación")
comparison["segundos"].div(60).plot.bar(ax=axes[1], rot=0, color="teal")
axes[1].set_ylabel("Minutos de entrenamiento")
fig.tight_layout(); fig.savefig(RESULTS / "comparacion.png", dpi=150); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(19, 5))
for ax, (name, pred) in zip(axes, predictions.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=CLASSES,
        normalize="true", cmap="Blues", xticks_rotation=90, ax=ax, colorbar=False, values_format=".1f")
    ax.set_title(name)
fig.tight_layout(); fig.savefig(RESULTS / "confusiones.png", dpi=150); plt.show()
winner = comparison["val_accuracy"].idxmax()
print("Modelo seleccionado por validación:", winner)
print(classification_report(y_test, predictions[winner], target_names=CLASSES, digits=3))
''')
md('''## Conclusión
La conclusión experimental se genera al ejecutar la siguiente celda, a partir de las métricas
medidas. No se asigna un ganador antes del entrenamiento.
''')
code('''winner = comparison["val_accuracy"].idxmax()
best = comparison.loc[winner]
fastest = comparison["segundos"].idxmin()
paragraphs = []
for name, r in comparison.iterrows():
    h = histories[name]
    i = int(r["mejor_epoca"]) - 1
    gap = h["accuracy"][i] - h["val_accuracy"][i]
    paragraphs.append(f"**{name}** obtuvo {r['val_accuracy']:.2%} en validación, "
        f"{r['test_accuracy']:.2%} en prueba y F1 macro {r['test_f1_macro']:.4f}, "
        f"con {int(r['parametros']):,} parámetros ({int(r['entrenables']):,} entrenables) "
        f"y {r['segundos']/60:.2f} minutos de entrenamiento. En la época seleccionada "
        f"la diferencia accuracy entrenamiento menos validación fue {gap:.2%}.")
conclusion = "\\n\\n".join(paragraphs) + (
    f"\\n\\nEl mejor modelo según el criterio fijado (accuracy de validación) fue **{winner}**, "
    f"con {best['val_accuracy']:.2%}; su accuracy en prueba fue {best['test_accuracy']:.2%}. "
    f"El entrenamiento más rápido fue **{fastest}**. La elección práctica depende también del "
    "tiempo y recursos disponibles. La CNN básica establece una referencia compacta; la profunda "
    "aumenta la capacidad de extracción de patrones, sin garantizar mejores resultados con pocos datos "
    "y épocas. MobileNetV2 reutiliza representaciones de ImageNet, pero su base congelada y el cambio "
    "de dominio limitan la adaptación. Estos factores son explicaciones plausibles, no causas "
    "demostradas: las arquitecturas cambian varios componentes a la vez. "
    "\\n\\nMejoraría el experimento utilizando el resto de las imágenes de entrenamiento, repitiendo "
    "con varias semillas y ajustando tasa de aprendizaje, regularización y épocas solo con validación. "
    "Para MobileNetV2 probaría fine-tuning de los últimos bloques con una tasa menor (1e-5), "
    "manteniendo BatchNorm en inferencia. Compararía además versiones sin aumento y sin regularización "
    "para medir su efecto individual. No se afirma significancia estadística con una sola ejecución; "
    "tampoco se contabiliza el costo histórico de preentrenar ImageNet ni se ha medido latencia de despliegue."
)
(RESULTS / "conclusion.md").write_text(conclusion, encoding="utf-8")
display(Markdown(conclusion))
''')
md('''## Referencias
- [CIFAR-10 en Keras](https://keras.io/api/datasets/cifar10/).
- [MobileNetV2 y preprocesamiento](https://keras.io/api/applications/mobilenet/).
- [Guía de transfer learning](https://keras.io/guides/transfer_learning/).
- [Instalación de TensorFlow](https://www.tensorflow.org/install/pip).
''')
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
nb.metadata.language_info = dict(name="python", version="3.12")
nbf.write(nb, "Arquitecturas_cnn.ipynb")
