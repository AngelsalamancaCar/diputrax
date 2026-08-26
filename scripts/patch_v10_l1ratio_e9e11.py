# -*- coding: utf-8 -*-
"""
patch_v10_l1ratio_e9e11.py

migrate_sklearn18_l1ratio.py corrio antes de que las celdas E9
(lasso_path_analysis) y E11 (bootstrap_selection) existieran en
diputraxv10.ipynb, asi que quedaron con `penalty="l1"` deprecado.
Aplica el mismo reemplazo (penalty='l1' -> l1_ratio=1) solo a esas
dos celdas. Idempotente.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "diputraxv10.ipynb"

PAT = re.compile(r"penalty\s*=\s*(['\"])l1\1")
REPL = "l1_ratio=1"

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
