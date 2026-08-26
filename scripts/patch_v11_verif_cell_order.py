# -*- coding: utf-8 -*-
"""
patch_v11_verif_cell_order.py

La celda VERIF de diputraxv11.ipynb (celda 8, "VERIFICACION de acciones
correctivas") llama a get_Xy/get_groups/cv_auc/cv_auc_grouped/lr_binary,
pero esas funciones se definen en la celda 9 ("Model factories + CV
idénticos a v10"), que va despues. Al ejecutar el notebook de arriba a
abajo, la celda VERIF corre antes de que existan sus dependencias y
falla con NameError en las 8 filas (capturado silenciosamente por el
try/except de la propia celda, asi que nbconvert no lo reporta como
error de ejecucion).

Fix: intercambiar el orden de las celdas 8 y 9. Idempotente (detecta
si la celda de infraestructura ya precede a la de VERIF y no hace nada).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "diputraxv11.ipynb"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(c):
    return "".join(c["source"])


verif_idx = next(
    i for i, c in enumerate(cells)
    if c["cell_type"] == "code" and "VERIFICACION de acciones correctivas" in src(c)
)
infra_idx = next(
    i for i, c in enumerate(cells)
    if c["cell_type"] == "code" and "Model factories + CV" in src(c)
)

if verif_idx < infra_idx:
    cells[verif_idx], cells[infra_idx] = cells[infra_idx], cells[verif_idx]
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Intercambiadas celdas {verif_idx} (VERIF) y {infra_idx} (infra).")
else:
    print("Ya en orden correcto — nada que hacer.")
