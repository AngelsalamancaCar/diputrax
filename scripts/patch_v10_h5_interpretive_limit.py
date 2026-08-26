# -*- coding: utf-8 -*-
"""
patch_v10_h5_interpretive_limit.py

P4 (accionescorrectivas.md §4): el limite interpretativo de H5/ERA_4
(es_partido_mayoria casi colineal con p_MORENA, VIF~129 sobre KEY_FEATS /
VIF=inf sobre el subconjunto SFM en Anexo D.3) solo vivia en la seccion
DOC nueva (celda 80, "4.6 Acciones correctivas"). Se propaga a las celdas
de interpretacion narrativa donde H5 realmente se argumenta, para que el
lenguaje sea asociativo/predictivo y no causal en el lugar donde el lector
lo encuentra primero.

Celdas tocadas:
  97  (5.5 Interpretacion -- Comisiones Nodales, subseccion ERA_4)
  118 (6.4 Interpretacion -- Comisiones Lastre) -- nota de alcance, no
       aplica el limite (es_partido_mayoria/p_MORENA no son predictores
       dominantes de lastre en ninguna era)
  164 (11.1 Hallazgos principales, H5)

Idempotente: cada insercion se salta si el marcador ya esta presente.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "diputraxv10.ipynb"

MARKER = "Límite interpretativo"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]
total = 0


def append_if_absent(idx, addition):
    global total
    c = cells[idx]
    s = "".join(c["source"])
    if MARKER in s and "Nota de alcance" not in addition:
        return
    if "Nota de alcance" in addition and "Nota de alcance" in s:
        return
    new_s = s.rstrip("\n") + "\n" + addition
    c["source"] = new_s.splitlines(keepends=True)
    total += 1


# --- Celda 97: §5.5 Nodales, cierre de la subseccion ERA_4 ---
append_if_absent(97, """
> **Límite interpretativo (v0 §6, §8; ver §4.6):** en `ERA_4`, `es_partido_mayoria` es casi colineal con `p_MORENA` (VIF≈129 sobre `KEY_FEATS`; VIF=∞ sobre el subconjunto SFM, Anexo D.3) — el modelo no distingue estadísticamente "ser partido mayoritario" de "ser Morena". El giro hacia `n_cargos_legislativos_prev` y la carrera parlamentaria (H5, §11.1) debe leerse como asociación con el perfil de la era, no como efecto causal atribuible específicamente a la identidad partidista de Morena.""")

# --- Celda 118: §6.4 Lastre, nota de alcance al final ---
append_if_absent(118, """
**Nota de alcance:** a diferencia de nodales (§5.5, ERA_4), en lastre `es_partido_mayoria`/`p_MORENA` no figuran entre los predictores dominantes de ninguna era — el problema de colinealidad exacta documentado en §4.6 y Anexo D.3 no aplica a este target.""")

# --- Celda 164: §11.1, H5 ---
c164 = cells[164]
s164 = "".join(c164["source"])
h5_caveat = """

*Límite interpretativo:* en `ERA_4`, `es_partido_mayoria` es casi colineal con la identidad partidista `p_MORENA` (VIF≈129, §4.6; VIF=∞ en el subconjunto SFM, Anexo D.3) — el modelo no puede separar estadísticamente "ser partido mayoritario" de "ser Morena". H5 debe leerse en modo asociativo/descriptivo del perfil de la era, no como atribución causal a una filosofía de gobierno específica de Morena."""
anchor = "Esto es consistente con el discurso de gobierno de Morena pero refleja también la menor experiencia burocrática de sus cuadros."
if MARKER not in s164 and anchor in s164:
    s164 = s164.replace(anchor, anchor + h5_caveat)
    c164["source"] = s164.splitlines(keepends=True)
    total += 1

if total:
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"Total inserciones: {total}")
