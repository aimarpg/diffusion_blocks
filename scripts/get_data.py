#!/usr/bin/env python3
"""Descarga y verificacion de los datos.

Los datos NO van al repositorio. Lo que va es este script: pequeno, versionado
y con el hash del fichero, de forma que cualquiera que clone obtenga
exactamente los mismos bytes que tu.

Es lo que hace posible la auditoria cruzada de noviembre: si tu auditor no
puede descargar tus datos, no puede reproducir tu metrica, y eso resta puntos
a las dos partes.

    python scripts/get_data.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DESTINO = Path("data")

_HF_TINYSTORIES = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
       "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/data/")

FUENTES = [
    (_HF_TINYSTORIES + "train-00000-of-00004-2d5a1467fff1081b.parquet", "train-00000-of-00004.parquet",
     "sha256:77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b"),
    (_HF_TINYSTORIES + "train-00001-of-00004-5852b56a2bd28fd9.parquet", "train-00001-of-00004.parquet",
     "sha256:e2f173a6dcc2b5895183e8ad376e115bd5c06764eb0cfc01ce1add156b2d017f"),
    (_HF_TINYSTORIES + "train-00002-of-00004-a26307300439e943.parquet", "train-00002-of-00004.parquet",
     "sha256:cac73cd1a0973a7fccde022a5a5c78f8ba58ea6849541835c58e7c5fc747f2ef"),
    (_HF_TINYSTORIES + "train-00003-of-00004-d243063613e5a057.parquet", "train-00003-of-00004.parquet",
     "sha256:862514ae75723ddd339337f1ef7617b88f82290dcddd57445e1d38d257ca7891"),
    (_HF_TINYSTORIES + "validation-00000-of-00001-869c898b519ad725.parquet", "validation.parquet",
     "sha256:33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url: str, destino: Path) -> None:
    print(f"descargando {url} -> {destino}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destino)


def main() -> int:
    solo_hash = "--hash" in sys.argv
    if not FUENTES:
        print("No hay fuentes declaradas. Rellena FUENTES en este fichero.")
        print("Sin esto, nadie puede reproducir tu trabajo (hito H3).")
        return 1

    for url, nombre, esperado in FUENTES:
        ruta = DESTINO / nombre
        if not ruta.exists():
            descargar(url, ruta)

        real = sha256(ruta)
        if solo_hash:
            print(f"{nombre}: sha256:{real}")
            continue

        if esperado.removeprefix("sha256:") != real:
            print(f"ERROR: el hash de {nombre} no coincide.")
            print(f"  esperado: {esperado}")
            print(f"  obtenido: sha256:{real}")
            print("  El fichero esta corrupto o el origen ha cambiado.")
            return 1
        print(f"ok  {nombre}  sha256 verificado")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
