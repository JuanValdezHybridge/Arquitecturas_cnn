"""Descarga opcional por rangos desde el servidor oficial y verifica SHA-256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parent / ".keras" / "datasets"
ROOT.mkdir(parents=True, exist_ok=True)
SIZE = 170498071
COUNT = 16
URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"

def download(i):
    start, end = SIZE * i // COUNT, SIZE * (i + 1) // COUNT - 1
    path = ROOT / f"cifar.part{i:02d}"
    if path.exists() and path.stat().st_size == end - start + 1:
        return i
    subprocess.run(["curl.exe", "--silent", "--show-error", "--fail", "--location",
                    "--retry", "5", "--retry-all-errors", "--max-time", "240", "--range", f"{start}-{end}",
                    "--output", str(path), URL], check=True)
    if path.stat().st_size != end - start + 1:
        raise ValueError(f"Tamaño incorrecto en segmento {i}")
    return i

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(download, i) for i in range(COUNT)]):
            print("Segmento completado:", future.result(), flush=True)
    archive = ROOT / "cifar-10-batches-py-target_archive"
    with archive.open("wb") as out:
        for i in range(COUNT):
            out.write((ROOT / f"cifar.part{i:02d}").read_bytes())
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError(f"SHA-256 incorrecto: {actual}")
    print("CIFAR-10 completo; SHA-256 oficial verificado.", flush=True)
