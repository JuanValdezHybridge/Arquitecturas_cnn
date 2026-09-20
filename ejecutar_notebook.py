"""Ejecuta todas las celdas y conserva resultados incluso si ocurre un error."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
os.chdir(root)
for key, folder in [("MPLCONFIGDIR", ".mplconfig"), ("IPYTHONDIR", ".ipython"),
                    ("JUPYTER_RUNTIME_DIR", ".jupyter_runtime"), ("KERAS_HOME", ".keras")]:
    path = root / folder
    path.mkdir(exist_ok=True)
    os.environ[key] = str(path)
path = root / "Arquitecturas_cnn.ipynb"
nb = nbformat.read(path, as_version=4)
def started(cell, cell_index, **kwargs):
    print(f"Ejecutando celda {cell_index}: {cell.source[:75]}", flush=True)
def completed(cell, cell_index, **kwargs):
    nbformat.write(nb, path)
    print(f"Celda {cell_index} completada", flush=True)
client = NotebookClient(nb, timeout=14400, kernel_name="python3",
                        on_cell_start=started, on_cell_executed=completed)
try:
    client.execute()
    # Materializar el reporte como una sola celda Markdown, tal como pide la actividad.
    conclusion = (root / "resultados" / "conclusion.md").read_text(encoding="utf-8")
    nb.cells = [c for c in nb.cells
                if not (c.cell_type == "markdown" and c.source.startswith("## Conclusión"))]
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and cell.source.startswith('winner = comparison["val_accuracy"]'):
            cell.outputs = []
            nb.cells.insert(index + 1, nbformat.v4.new_markdown_cell("## Conclusión\n\n" + conclusion))
            break
finally:
    nbformat.write(nb, path)
print("Notebook ejecutado y guardado.", flush=True)
