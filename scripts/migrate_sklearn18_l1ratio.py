# -*- coding: utf-8 -*-
"""
migrate_sklearn18_l1ratio.py

Migra la API deprecada de LogisticRegression en sklearn 1.8:
    penalty='l1'  ->  l1_ratio=1

sklearn 1.8 deprecó el parámetro `penalty` (se elimina en 1.10) en favor de
`l1_ratio` (l1_ratio=1 == L1, l1_ratio=0 == L2). Verificado empíricamente que
`LogisticRegression(l1_ratio=1, solver='liblinear', ...)` produce coeficientes
BIT-IDÉNTICOS a `LogisticRegression(penalty='l1', solver='liblinear', ...)`
(diff máx = 0.0) y SIN warnings. `liblinear` y `saga` soportan `l1_ratio`.

`_L2_PARAMS` no se toca: no fija `penalty`, así que ya usa el default
`l1_ratio=0` (== L2) sin emitir warning.

Alcance: todos los notebooks aplicables + los scripts fuente/generadores que
embeben `penalty='l1'`. Idempotente (el regex no reincide tras la migración).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# penalty = 'l1'  ó  penalty="l1"  (con espacios opcionales)  ->  l1_ratio=1
PAT = re.compile(r"penalty\s*=\s*(['\"])l1\1")
REPL = "l1_ratio=1"

NOTEBOOKS = [
    "notebooks/diputraxv6LR.ipynb",
    "notebooks/diputraxv7LR.ipynb",
    "notebooks/diputraxv8LR.ipynb",
    "notebooks/diputraxv9.ipynb",
    "notebooks/diputraxv10.ipynb",
    "notebooks/diputraxv11.ipynb",
    "notebooks/diputraxv12.ipynb",
]
SCRIPTS = [
    "scripts/build_v11.py",
    "scripts/build_v12.py",
    "scripts/build_v10_anexoD_stats.py",
    "scripts/patch_v6lr.py",
]

total = 0

print("=== NOTEBOOKS ===")
for rel in NOTEBOOKS:
    p = ROOT / rel
    nb = json.loads(p.read_text(encoding="utf-8"))
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        new, k = PAT.subn(REPL, s)
        if k:
            c["source"] = new.splitlines(keepends=True)
            n += k
    if n:
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {rel:34} : {n} reemplazo(s)")
    total += n

print("\n=== SCRIPTS ===")
for rel in SCRIPTS:
    p = ROOT / rel
    txt = p.read_text(encoding="utf-8")
    new, k = PAT.subn(REPL, txt)
    if k:
        p.write_text(new, encoding="utf-8")
    print(f"  {rel:34} : {k} reemplazo(s)")
    total += k

print(f"\nTotal reemplazos: {total}")
