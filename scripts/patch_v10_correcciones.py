# -*- coding: utf-8 -*-
"""
patch_v10_correcciones.py

Aplica a notebooks/diputraxv10.ipynb las ACCIONES CORRECTIVAS derivadas del
diagnostico de supuestos de diputraxv0 (secciones 8 y 12 de ese cuaderno):

  AC1  Consolidacion de features redundantes en FEAT_COLS
       - elimina `univ_extranjera` (duplicado exacto de `estudios_en_extranjero`,
         r=1.0 en las 4 eras -> "Pares casi duplicados" en v0 §8)
       - elimina `n_cargos_legislativos_prev` (= fue_diputado_local +
         fue_diputado_federal + fue_senador, dependencia lineal exacta ->
         "Dependencia lineal exacta (rango)" en v0 §8)
       - actualiza KEY_FEATS (celda 5.3) sustituyendo el agregado eliminado por
         su componente principal `fue_diputado_federal`, para no romper la
         indexacion SHAP y reflejar la senal ya redistribuida.

  AC2  Validacion sin fuga por reeleccion
       - agrega infraestructura de CV agrupada por `diputado_id`
         (StratifiedGroupKFold / GroupKFold) para que un mismo diputado reelecto
         no aparezca simultaneamente en train y test del mismo fold.

  DOC  Celda markdown "Acciones correctivas (derivadas de diputraxv0)" + una
       celda de VERIFICACION que, al ejecutarse con datos, mide el antes/después:
         (a) invariancia de AUC ante la consolidacion de features (AC1)
         (b) delta de AUC entre StratifiedKFold y StratifiedGroupKFold (AC2)

El script es idempotente: si un marcador ya esta presente, omite ese cambio.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "diputraxv10.ipynb"

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def src_of(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def find_cell(substr):
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code" and substr in src_of(c):
            return idx
    raise SystemExit(f"No se encontro la celda con: {substr!r}")


changes = []

# ----------------------------------------------------------------------
# AC1 — FEAT_COLS: eliminar univ_extranjera y n_cargos_legislativos_prev
# ----------------------------------------------------------------------
i_feat = find_cell("FEAT_COLS   = NUMERIC_FEATS")
s = src_of(cells[i_feat])
if "AC1(v0):" not in s:
    s = s.replace(
        '    "estudios_en_extranjero", "univ_publica", "univ_privada", "univ_extranjera",\n',
        '    "estudios_en_extranjero", "univ_publica", "univ_privada",  # AC1(v0): univ_extranjera eliminada — duplicado exacto de estudios_en_extranjero (r=1.0)\n',
    )
    s = s.replace(
        '    "n_cargos_legislativos_prev", "fue_diputado_local",\n',
        '    # AC1(v0): n_cargos_legislativos_prev eliminada — = fue_diputado_local+fue_diputado_federal+fue_senador (combinacion lineal exacta)\n    "fue_diputado_local",\n',
    )
    set_src(cells[i_feat], s)
    changes.append(f"AC1: FEAT_COLS depurado en celda {i_feat}")
else:
    changes.append(f"AC1: celda {i_feat} ya depurada (sin cambios)")

# ----------------------------------------------------------------------
# AC1b — KEY_FEATS: sustituir el agregado eliminado por su componente
# ----------------------------------------------------------------------
i_key = find_cell('KEY_FEATS = [')
s = src_of(cells[i_key])
if '"n_cargos_legislativos_prev"' in s:
    s = s.replace(
        '    "es_partido_mayoria", "n_cargos_legislativos_prev",\n',
        '    "es_partido_mayoria", "fue_diputado_federal",  # AC1(v0): reemplaza n_cargos_legislativos_prev (suma exacta) por su componente principal\n',
    )
    set_src(cells[i_key], s)
    changes.append(f"AC1b: KEY_FEATS actualizado en celda {i_key}")
else:
    changes.append(f"AC1b: celda {i_key} ya actualizada (sin cambios)")

# ----------------------------------------------------------------------
# AC2 — Infraestructura de CV agrupada por diputado_id
# ----------------------------------------------------------------------
i_infra = find_cell("def cv_auc(model, X, y, k=5):")
s = src_of(cells[i_infra])
if "cv_auc_grouped" not in s:
    anchor = "# SHAP helper"
    injection = (
        "# ── AC2(v0): validacion sin fuga por reeleccion (agrupada por diputado_id) ──\n"
        "from sklearn.model_selection import StratifiedGroupKFold, GroupKFold\n\n"
        "def get_groups(era):\n"
        "    \"\"\"diputado_id como clave de agrupamiento: evita que un mismo diputado\n"
        "    reelecto caiga en train y test del mismo fold (fuga por reeleccion).\"\"\"\n"
        "    mask = df_enc[\"era\"] == era\n"
        "    return df_enc.loc[mask, \"diputado_id\"].reset_index(drop=True)\n\n"
        "def cv_auc_grouped(model, X, y, groups, k=5):\n"
        "    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)\n"
        "    s = cross_val_score(model, X, y, groups=groups, cv=cv, scoring=\"roc_auc\")\n"
        "    return s.mean(), s.std()\n\n"
        "def cv_mae_grouped(model, X, y, groups, k=5):\n"
        "    cv = GroupKFold(n_splits=k)\n"
        "    s = -cross_val_score(model, X, y, groups=groups, cv=cv, scoring=\"neg_mean_absolute_error\")\n"
        "    return s.mean(), s.std()\n\n"
    )
    s = s.replace(anchor, injection + anchor, 1)
    set_src(cells[i_infra], s)
    changes.append(f"AC2: helpers de CV agrupada agregados en celda {i_infra}")
else:
    changes.append(f"AC2: celda {i_infra} ya tiene helpers agrupados (sin cambios)")

# ----------------------------------------------------------------------
# DOC + VERIFICACION — insertar tras la celda "Setup OK" (post-infra)
# ----------------------------------------------------------------------
i_setup = find_cell('print("Setup OK")')

md_doc = {
    "cell_type": "markdown",
    "metadata": {},
    "source": (
        "## 4.6 Acciones correctivas (derivadas de diputraxv0)\n"
        "\n"
        "El cuaderno `diputraxv0` audita los siete supuestos de la regresión lineal clásica sobre "
        "**este mismo pipeline y `FEAT_COLS`**. Dos de sus hallazgos son *acciones de limpieza de "
        "datos independientes del modelo* (v0 §8 y §12) y se implementan aquí:\n"
        "\n"
        "- **AC1 — Consolidación de features redundantes.** Se elimina `univ_extranjera` "
        "(duplicado exacto de `estudios_en_extranjero`, r=1.0) y `n_cargos_legislativos_prev` "
        "(suma exacta de `fue_diputado_local + fue_diputado_federal + fue_senador`, ya incluidos). "
        "Son redundancias *dentro del espacio columna* del diseño: para un modelo lineal/L1 las "
        "predicciones son invariantes a su eliminación (mismo AUC), pero la **atribución SHAP deja "
        "de repartirse** entre columnas idénticas y los VIF/convergencia MLE del Anexo D mejoran.\n"
        "- **AC2 — Validación sin fuga por reelección.** ~625 registros son reelecciones (mismo "
        "`diputado_id` en legislaturas distintas). El `StratifiedKFold` simple puede colocar al "
        "mismo diputado en train y test del mismo fold. Se añade CV **agrupada por `diputado_id`** "
        "(`StratifiedGroupKFold`) para estimar el AUC sin esa fuga.\n"
        "\n"
        "**Límite interpretativo declarado (v0 §6, §8):** en eras de partido dominante "
        "`es_partido_mayoria` es casi colineal con la identidad partidista (`p_PRI` en ERA_1, "
        "`p_MORENA` en ERA_4, VIF≈129). La atribución SHAP entre ambas es intercambiable, por lo "
        "que **H5 (ERA_4) debe leerse en modo asociativo, no causal**: el modelo no distingue "
        "\"ser partido mayoritario\" de \"ser Morena\".\n"
        "\n"
        "La celda siguiente mide empíricamente el efecto de AC1 y AC2 sobre el AUC."
    ).splitlines(keepends=True),
}

code_verif = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# ============================================================\n"
        "# VERIFICACION de acciones correctivas (AC1 + AC2)\n"
        "# Reporta el antes/después sobre el modelo productivo LR-L1.\n"
        "# ============================================================\n"
        "_REDUND = [\"univ_extranjera\", \"n_cargos_legislativos_prev\"]  # eliminadas por AC1\n"
        "_rows = []\n"
        "for _t in [\"nodal_bin\", \"lastre_bin\"]:\n"
        "    for _era in ERA_ORDER:\n"
        "        _X, _y = get_Xy(_era, _t)\n"
        "        _g = get_groups(_era)\n"
        "        # (AC1) mismo modelo con las 2 features redundantes RE-agregadas\n"
        "        _extra = [c for c in _REDUND if c in df_enc.columns and c not in _X.columns]\n"
        "        _mask = df_enc[\"era\"] == _era\n"
        "        _X_old = pd.concat([_X, df_enc.loc[_mask, _extra].astype(float).reset_index(drop=True)], axis=1)\n"
        "        try:\n"
        "            auc_clean, _ = cv_auc(lr_binary(), _X, _y)            # FEAT_COLS depurado (AC1)\n"
        "            auc_redun, _ = cv_auc(lr_binary(), _X_old, _y)        # con redundantes (pre-AC1)\n"
        "            auc_group, _ = cv_auc_grouped(lr_binary(), _X, _y, _g)  # sin fuga reeleccion (AC2)\n"
        "            _rows.append({\n"
        "                \"target\": _t, \"era\": _era,\n"
        "                \"AUC pre-AC1 (con redundantes)\": round(auc_redun, 4),\n"
        "                \"AUC post-AC1 (depurado)\": round(auc_clean, 4),\n"
        "                \"Δ AC1\": round(auc_clean - auc_redun, 4),\n"
        "                \"AUC AC2 (GroupKFold diputado)\": round(auc_group, 4),\n"
        "                \"Δ AC2 (fuga reeleccion)\": round(auc_group - auc_clean, 4),\n"
        "            })\n"
        "        except Exception as _e:\n"
        "            _rows.append({\"target\": _t, \"era\": _era, \"error\": str(_e)})\n"
        "\n"
        "df_ac = pd.DataFrame(_rows)\n"
        "print(f\"FEAT_COLS depurado: {len(FEAT_COLS)} features (AC1 elimino 2 redundantes)\")\n"
        "print(\"Δ AC1 ~ 0 confirma que la consolidacion NO altera el poder predictivo (redundancia dentro del span).\")\n"
        "print(\"Δ AC2 < 0 cuantifica la fuga por reeleccion que corregia el StratifiedKFold simple.\")\n"
        "display(df_ac)"
    ).splitlines(keepends=True),
}

marker_present = any(
    c["cell_type"] == "markdown" and "Acciones correctivas (derivadas de diputraxv0)" in src_of(c)
    for c in cells
)
if not marker_present:
    cells.insert(i_setup + 1, md_doc)
    cells.insert(i_setup + 2, code_verif)
    changes.append(f"DOC: markdown + celda de verificacion insertadas tras celda {i_setup}")
else:
    changes.append("DOC: celdas de acciones correctivas ya presentes (sin cambios)")

# ----------------------------------------------------------------------
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

print("=== patch_v10_correcciones aplicado ===")
for ch in changes:
    print("  -", ch)
print(f"\nCeldas totales ahora: {len(cells)}")
