"""Comprueba la integridad del entregable y recalcula métricas de las predicciones."""
from pathlib import Path
import ast
import json
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

root = Path(__file__).resolve().parent
nb = nbformat.read(root / "Arquitecturas_cnn.ipynb", as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
        assert cell.execution_count is not None, "Celda sin ejecutar"
        assert all(output.output_type != "error" for output in cell.outputs), "Error en notebook"
assert sum(c.cell_type == "markdown" and c.source.startswith("## Conclusión")
           for c in nb.cells) == 1
results = root / "resultados"
table = pd.read_csv(results / "comparacion.csv")
histories = json.loads((results / "historiales.json").read_text(encoding="utf-8"))
predictions = np.load(results / "predicciones.npz")
assert len(table) == 3 and len(histories) == 3
assert len(predictions["y_test"]) == 10000
for row in table.itertuples():
    pred = predictions[row.modelo]
    assert pred.shape == (10000,) and np.all((pred >= 0) & (pred < 10))
    assert np.isclose(accuracy_score(predictions["y_test"], pred), row.test_accuracy, atol=1e-6)
    assert np.isclose(f1_score(predictions["y_test"], pred, average="macro"), row.test_f1_macro)
    assert len(histories[row.modelo]["loss"]) == row.epocas
    assert np.argmin(histories[row.modelo]["val_loss"]) + 1 == row.mejor_epoca
    assert np.isclose(histories[row.modelo]["val_accuracy"][row.mejor_epoca - 1], row.val_accuracy)
for name in ["curvas.png", "comparacion.png", "confusiones.png", "conclusion.md"]:
    assert (results / name).stat().st_size > 0
print("Verificación correcta: notebook ejecutado sin errores y métricas consistentes con 10 000 predicciones por modelo.")
