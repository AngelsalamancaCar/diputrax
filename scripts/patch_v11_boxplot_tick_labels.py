# -*- coding: utf-8 -*-
"""
patch_v11_boxplot_tick_labels.py

matplotlib >=3.9 renombro el parametro `labels` de Axes.boxplot a
`tick_labels` (el viejo `labels` fue removido en 3.11, que es lo que
hay instalado). Corrige la unica ocurrencia en diputraxv11.ipynb
(seccion "Observacion 2 - Paso 2"). Idempotente.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "diputraxv11.ipynb"

PAT = re.compile(r"\.boxplot\(([^)]*?)\blabels=")
REPL = lambda m: m.group(0).replace("labels=", "tick_labels=")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

total = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    new, k = PAT.subn(REPL, s)
    if k:
        c["source"] = new.splitlines(keepends=True)
        total += k
        print(f"  celda {i}: {k} reemplazo(s)")

if total:
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"Total reemplazos: {total}")
