# -*- coding: utf-8 -*-
"""Tercer parche de diputraxv12.ipynb: neutraliza las constantes internas de
era (plan12upd.md — documento abstraído de la tesina). Elimina
ERA_ORDER/ERA_LABELS/ERA_COLORS (huérfanas tras el reencuadre), renombra
ERA_MAP -> COAL4_MAP con etiquetas neutrales y la columna auxiliar
df["era"] -> df["grupo4"] (consumida solo por get_groups, AC2).
Requiere re-ejecutar el notebook después.
"""
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "diputraxv12.ipynb"

OLD_CONSTS = '''ERA_MAP = {
    57: "ERA_1_PRI",  58: "ERA_1_PRI",  59: "ERA_1_PRI",
    60: "ERA_2_PAN",  61: "ERA_2_PAN",  62: "ERA_2_PAN",
    63: "ERA_3_TRANS", 64: "ERA_3_TRANS", 65: "ERA_3_TRANS",
    66: "ERA_4_MORENA",
}
ERA_ORDER = ["ERA_1_PRI", "ERA_2_PAN", "ERA_3_TRANS", "ERA_4_MORENA"]
ERA_LABELS = {
    "ERA_1_PRI":    "ERA 1 — PRI (57-59)",
    "ERA_2_PAN":    "ERA 2 — PAN (60-62)",
    "ERA_3_TRANS":  "ERA 3 — Transicion (63-65)",
    "ERA_4_MORENA": "ERA 4 — Morena (66)",
}
ERA_COLORS = {
    "ERA_1_PRI": "#c0392b", "ERA_2_PAN": "#2980b9",
    "ERA_3_TRANS": "#8e44ad", "ERA_4_MORENA": "#27ae60",
}'''

NEW_CONSTS = '''# Mapeo auxiliar de 4 grupos por coalición dominante (lo consume get_groups, AC2)
COAL4_MAP = {
    57: "G1_PRI_57_59",  58: "G1_PRI_57_59",  59: "G1_PRI_57_59",
    60: "G2_PAN_60_62",  61: "G2_PAN_60_62",  62: "G2_PAN_60_62",
    63: "G3_TRANS_63_65", 64: "G3_TRANS_63_65", 65: "G3_TRANS_63_65",
    66: "G4_MOR_66",
}'''

REPL = [
    (4, OLD_CONSTS, NEW_CONSTS),
    (6, 'df["era"] = df["legislatura_num"].map(ERA_MAP)',
        'df["grupo4"] = df["legislatura_num"].map(COAL4_MAP)'),
    (10, 'def get_groups(era):\n    mask = df_enc["era"] == era',
         'def get_groups(group):\n    mask = df_enc["grupo4"] == group'),
]


def cell_src(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    errors = []
    for idx, old, _ in REPL:
        n = cell_src(cells[idx]).count(old)
        if n != 1:
            errors.append(f"celda {idx}: patrón aparece {n} veces: {old[:60]!r}")
    if errors:
        print("ABORT:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    for idx, old, new in REPL:
        cells[idx]["source"] = cell_src(cells[idx]).replace(old, new).splitlines(keepends=True)
        print(f"celda {idx:2d}: reemplazo aplicado")
    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"\nOK — guardado {NB_PATH}")


if __name__ == "__main__":
    main()
