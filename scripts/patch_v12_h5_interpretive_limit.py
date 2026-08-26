# -*- coding: utf-8 -*-
"""
patch_v12_h5_interpretive_limit.py

P4 (accionescorrectivas.md §4): v12 cita H5 ("legislativizacion del perfil
en Morena") en la seccion 9.2 como evidencia de que la periodizacion en 4
eras "sostiene el aparato interpretativo de la tesis", sin la salvedad
asociativa/no-causal que v10 §4.6/§11.1 ya documenta (es_partido_mayoria
casi colineal con p_MORENA en ERA_4, VIF~129). Se agrega un parentesis con
la salvedad en el punto exacto donde se cita H5, en vez de una nota aparte
que el lector podria saltarse.

Idempotente: se salta si el parentesis ya esta presente.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "diputraxv12.ipynb"

OLD = "H5 (legislativización del perfil en Morena)"
NEW = ("H5 (legislativización del perfil en Morena — lectura asociativa, no "
       "causal: `es_partido_mayoria` y `p_MORENA` son casi colineales en "
       "ERA_4, VIF≈129, ver v10 §4.6/§11.1)")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
total = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    s = "".join(c["source"])
    if OLD in s:
        s = s.replace(OLD, NEW)
        c["source"] = s.splitlines(keepends=True)
        total += 1
        print(f"  celda {i}: reemplazado")

if total:
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"Total reemplazos: {total}")
