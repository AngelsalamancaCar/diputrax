# -*- coding: utf-8 -*-
"""
patch_v11_v12_correcciones.py

Propaga a diputraxv11 y diputraxv12 las acciones correctivas ya aplicadas en
diputraxv10 (ver scripts/patch_v10_correcciones.py y accionescorrectivas.md),
derivadas del diagnostico de supuestos de diputraxv0 (§8 y §12).

Parchea DOS capas por version para que el cambio sea persistente:
  - el notebook (artefacto actual)  -> notebooks/diputraxvNN.ipynb
  - el build script (fuente de verdad) -> scripts/build_vNN.py

Acciones:
  AC1  Elimina de FEAT_COLS `univ_extranjera` (duplicado exacto de
       `estudios_en_extranjero`) y `n_cargos_legislativos_prev` (suma exacta de
       fue_diputado_local+fue_diputado_federal+fue_senador).
  AC2  Agrega infraestructura de CV agrupada por `diputado_id`
       (get_groups + cv_auc_grouped + cv_mae_grouped).
  DOC  Celda markdown documentando las acciones correctivas.
  VERIF (solo v11 notebook, que expone get_Xy/ERA_ORDER): celda que mide el
        antes/después al ejecutarse.

No aplica:
  AC1b (reindexado de KEY_FEATS) — v11 usa KEY_FEATS dinamico (df_imp_p.index),
       v12 no define KEY_FEATS. Nada que reindexar.

Idempotente: si el marcador ya esta presente, omite el cambio.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

# --- bloques reutilizables (sin triple-comillas: seguros dentro de code(r\"\"\"...\"\"\")) ---
AC2_BLOCK = (
    "# -- AC2(v0): validacion sin fuga por reeleccion (agrupada por diputado_id) --\n"
    "# get_groups usa diputado_id como clave para que un mismo diputado reelecto\n"
    "# no caiga en train y test del mismo fold (fuga por reeleccion).\n"
    "from sklearn.model_selection import StratifiedGroupKFold, GroupKFold\n\n"
    "def get_groups(era):\n"
    "    mask = df_enc[\"era\"] == era\n"
    "    if \"diputado_id\" in df_enc.columns:\n"
    "        return df_enc.loc[mask, \"diputado_id\"].reset_index(drop=True)\n"
    "    return pd.Series(range(int(mask.sum())))\n\n"
    "def cv_auc_grouped(model, X, y, groups, k=5):\n"
    "    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)\n"
    "    s = cross_val_score(model, X, y, groups=groups, cv=cv, scoring=\"roc_auc\")\n"
    "    return s.mean(), s.std()\n\n"
    "def cv_mae_grouped(model, X, y, groups, k=5):\n"
    "    cv = GroupKFold(n_splits=k)\n"
    "    s = -cross_val_score(model, X, y, groups=groups, cv=cv, scoring=\"neg_mean_absolute_error\")\n"
    "    return s.mean(), s.std()\n\n"
)

PRINT_ANCHOR = 'print("Infraestructura de modelado replicada de v10 — OK")'

FEAT_OLD_1 = '    "estudios_en_extranjero", "univ_publica", "univ_privada", "univ_extranjera",\n'
FEAT_NEW_1 = '    "estudios_en_extranjero", "univ_publica", "univ_privada",  # AC1(v0): univ_extranjera eliminada — duplicado exacto (r=1.0)\n'
FEAT_OLD_2 = '    "n_cargos_legislativos_prev", "fue_diputado_local",\n'
FEAT_NEW_2 = '    # AC1(v0): n_cargos_legislativos_prev eliminada — = fue_diputado_local+fue_diputado_federal+fue_senador (combinacion lineal exacta)\n    "fue_diputado_local",\n'


def doc_text(na_line):
    return (
        "## Acciones correctivas (derivadas de diputraxv0)\n"
        "\n"
        "Se propagan a este cuaderno las acciones correctivas aplicadas en diputraxv10, derivadas del "
        "diagnóstico de supuestos de `diputraxv0` (§8 y §12):\n"
        "\n"
        "- **AC1 — Consolidación de features redundantes.** Se elimina `univ_extranjera` (duplicado "
        "exacto de `estudios_en_extranjero`, r=1.0) y `n_cargos_legislativos_prev` (suma exacta de "
        "`fue_diputado_local + fue_diputado_federal + fue_senador`, ya incluidos). Son redundancias "
        "*dentro del espacio-columna*: el AUC/MAE es invariante a su eliminación; solo se depura la "
        "atribución SHAP y los VIF.\n"
        "- **AC2 — Validación sin fuga por reelección.** Se añaden `get_groups()` (por `diputado_id`) y "
        "`cv_auc_grouped` / `cv_mae_grouped` (`StratifiedGroupKFold` / `GroupKFold`) para estimar el "
        "desempeño sin que un mismo diputado reelecto quede en train y test del mismo fold.\n"
        "\n"
        f"**No aplica aquí:** AC1b (reindexado de `KEY_FEATS`) — {na_line}\n"
        "\n"
        "**Límite interpretativo (v0 §6, §8):** en eras de partido dominante `es_partido_mayoria` es casi "
        "colineal con la identidad partidista (VIF≈129 en ERA_4); su lectura debe ser asociativa, no causal."
    )


VERIF_SRC = (
    "# ============================================================\n"
    "# VERIFICACION de acciones correctivas (AC1 + AC2)\n"
    "# Mide el antes/después sobre el modelo productivo LR-L1.\n"
    "# ============================================================\n"
    "_REDUND = [\"univ_extranjera\", \"n_cargos_legislativos_prev\"]  # eliminadas por AC1\n"
    "_rows = []\n"
    "for _t in [\"nodal_bin\", \"lastre_bin\"]:\n"
    "    for _era in ERA_ORDER:\n"
    "        try:\n"
    "            _X, _y = get_Xy(_era, _t)\n"
    "            _g = get_groups(_era)\n"
    "            _mask = df_enc[\"era\"] == _era\n"
    "            _extra = [c for c in _REDUND if c in df_enc.columns and c not in _X.columns]\n"
    "            _X_old = pd.concat([_X, df_enc.loc[_mask, _extra].astype(float).reset_index(drop=True)], axis=1)\n"
    "            auc_clean, _ = cv_auc(lr_binary(), _X, _y)\n"
    "            auc_redun, _ = cv_auc(lr_binary(), _X_old, _y)\n"
    "            auc_group, _ = cv_auc_grouped(lr_binary(), _X, _y, _g)\n"
    "            _rows.append({\"target\": _t, \"era\": _era,\n"
    "                          \"AUC pre-AC1\": round(auc_redun, 4),\n"
    "                          \"AUC post-AC1\": round(auc_clean, 4),\n"
    "                          \"D AC1\": round(auc_clean - auc_redun, 4),\n"
    "                          \"AUC AC2 (GroupKFold)\": round(auc_group, 4),\n"
    "                          \"D AC2\": round(auc_group - auc_clean, 4)})\n"
    "        except Exception as _e:\n"
    "            _rows.append({\"target\": _t, \"era\": _era, \"error\": str(_e)})\n"
    "print(f\"FEAT_COLS depurado: {len(FEAT_COLS)} features (AC1 elimino 2 redundantes)\")\n"
    "df_ac = pd.DataFrame(_rows); df_ac"
)


# ======================================================================
# Capa 1 — NOTEBOOKS
# ======================================================================
def patch_notebook(path, na_line, add_verif):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    log = []

    def src(c):
        return "".join(c["source"])

    # AC1 — FEAT_COLS
    i_feat = next(i for i, c in enumerate(cells)
                  if c["cell_type"] == "code" and "FEAT_COLS = NUMERIC_FEATS + DUMMY_FEATS" in src(c))
    s = src(cells[i_feat])
    if "AC1(v0):" not in s:
        s = s.replace(FEAT_OLD_1, FEAT_NEW_1).replace(FEAT_OLD_2, FEAT_NEW_2)
        cells[i_feat]["source"] = s.splitlines(keepends=True)
        log.append(f"AC1: FEAT_COLS depurado (celda {i_feat})")
    else:
        log.append(f"AC1: celda {i_feat} ya depurada")

    # AC2 — helpers de CV agrupada
    i_cv = next(i for i, c in enumerate(cells)
                if c["cell_type"] == "code" and PRINT_ANCHOR in src(c))
    s = src(cells[i_cv])
    if "cv_auc_grouped" not in s:
        s = s.replace(PRINT_ANCHOR, AC2_BLOCK + PRINT_ANCHOR, 1)
        cells[i_cv]["source"] = s.splitlines(keepends=True)
        log.append(f"AC2: helpers agrupados agregados (celda {i_cv})")
    else:
        log.append(f"AC2: celda {i_cv} ya tiene helpers")

    # DOC (+ VERIF) — insertar tras la celda FEAT_COLS
    already = any(c["cell_type"] == "markdown" and "Acciones correctivas (derivadas de diputraxv0)" in src(c)
                  for c in cells)
    if not already:
        md_cell = {"cell_type": "markdown", "metadata": {},
                   "source": doc_text(na_line).splitlines(keepends=True)}
        insert_at = i_feat + 1
        cells.insert(insert_at, md_cell)
        log.append(f"DOC: markdown insertada tras celda {i_feat}")
        if add_verif:
            verif_cell = {"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": VERIF_SRC.splitlines(keepends=True)}
            cells.insert(insert_at + 1, verif_cell)
            log.append(f"VERIF: celda de verificacion insertada tras celda {i_feat}")
    else:
        log.append("DOC/VERIF: ya presentes")

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return log


# ======================================================================
# Capa 2 — BUILD SCRIPTS (fuente de verdad)
# ======================================================================
def patch_buildscript(path, na_line, doc_anchor):
    text = path.read_text(encoding="utf-8")
    log = []

    # AC1
    if "AC1(v0):" not in text:
        text = text.replace(FEAT_OLD_1, FEAT_NEW_1).replace(FEAT_OLD_2, FEAT_NEW_2)
        log.append("AC1: FEAT_COLS depurado")
    else:
        log.append("AC1: ya depurado")

    # AC2 — inyectar helpers antes del print del bloque CV
    if "cv_auc_grouped" not in text:
        text = text.replace(PRINT_ANCHOR, AC2_BLOCK + PRINT_ANCHOR, 1)
        log.append("AC2: helpers agrupados inyectados")
    else:
        log.append("AC2: ya inyectado")

    # DOC — insertar cells.append(md(...)) antes del anclaje indicado
    if "Acciones correctivas (derivadas de diputraxv0)" not in text:
        doc_append = 'cells.append(md(r"""' + doc_text(na_line) + '"""))\n\n'
        assert doc_anchor in text, f"anclaje DOC no hallado en {path.name}"
        text = text.replace(doc_anchor, doc_append + doc_anchor, 1)
        log.append("DOC: cells.append(md(...)) insertado")
    else:
        log.append("DOC: ya presente")

    path.write_text(text, encoding="utf-8")
    return log


# ======================================================================
TARGETS = [
    ("v11", ROOT / "notebooks/diputraxv11.ipynb", ROOT / "scripts/build_v11.py",
     "v11 usa `KEY_FEATS` dinámico (`df_imp_p.index[:6]`), no hay lista fija que reindexar.",
     True,  # v11 notebook expone get_Xy/ERA_ORDER -> incluir celda de verificacion
     'cells.append(code(r"""# --- Model factories + CV idénticos a v10 (celda 78) ---'),
    ("v12", ROOT / "notebooks/diputraxv12.ipynb", ROOT / "scripts/build_v12.py",
     "v12 no define `KEY_FEATS`.",
     False,  # v12 no expone get_Xy -> sin celda de verificacion
     'cells.append(code(r"""# --- Model factories idénticas a v10 (celda 78) ---'),
]

for name, nb_path, bs_path, na_line, add_verif, doc_anchor in TARGETS:
    print(f"\n===== {name} =====")
    print(" [notebook]", nb_path.name)
    for line in patch_notebook(nb_path, na_line, add_verif):
        print("   -", line)
    print(" [build]   ", bs_path.name)
    for line in patch_buildscript(bs_path, na_line, doc_anchor):
        print("   -", line)

print("\nOK — patch_v11_v12_correcciones aplicado.")
