# -*- coding: utf-8 -*-
"""
Calcula los diagnosticos de regresion lineal clasica (Gauss-Markov) sobre los
datos y el conjunto de variables usados en diputraxv10.ipynb.

Replica el pipeline de datos de v10 (MICE + feature engineering, FEAT_COLS)
de forma identica y autocontenida, ajusta 12 modelos OLS -3 targets
(nodal_bin, lastre_bin como Modelo de Probabilidad Lineal; n_comisiones_tematicas
como regresion lineal) x 4 eras- sobre el conjunto completo FEAT_COLS
estandarizado, y calcula para cada uno los 7 criterios:

  1. Linealidad            -> RESET test (Ramsey)
  2. Homoscedasticidad      -> Breusch-Pagan + White simplificado (Wooldridge)
  3. Independencia errores  -> Durbin-Watson + ANOVA de residuos por partido
  4. Normalidad residuos    -> Shapiro-Wilk + Jarque-Bera
  5. No multicolinealidad   -> VIF + numero de condicion
  6. Exogeneidad            -> discusion cualitativa (no computable sin
                                variables instrumentales; ver notebook)
  7. Outliers influyentes   -> Cook's D, leverage, DFFITS

Guarda un manifiesto JSON (resultados numericos) y figuras PNG en
reports/eda/, que luego consume scripts/build_diputraxv0.py indirectamente
-el notebook re-ejecuta este mismo codigo via nbconvert, este script es la
version standalone usada para validar la metodologia antes de escribirla
en las celdas del cuaderno-.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scistats

import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "clean" / "diputados_20260421_205712.parquet"
REPORT_DIR = ROOT / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = ROOT / "scripts" / "_v0_manifest.json"

# =====================================================================
# Pipeline identico a diputraxv10.ipynb (celdas 69-76)
# =====================================================================
ERA_MAP = {
    57: "ERA_1_PRI",  58: "ERA_1_PRI",  59: "ERA_1_PRI",
    60: "ERA_2_PAN",  61: "ERA_2_PAN",  62: "ERA_2_PAN",
    63: "ERA_3_TRANS", 64: "ERA_3_TRANS", 65: "ERA_3_TRANS",
    66: "ERA_4_MORENA",
}
ERA_ORDER = ["ERA_1_PRI", "ERA_2_PAN", "ERA_3_TRANS", "ERA_4_MORENA"]
ERA_LABELS = {
    "ERA_1_PRI":    "ERA 1 - PRI (57-59)",
    "ERA_2_PAN":    "ERA 2 - PAN (60-62)",
    "ERA_3_TRANS":  "ERA 3 - Transicion (63-65)",
    "ERA_4_MORENA": "ERA 4 - Morena (66)",
}
TOP_PARTIDOS = ["PRI", "PAN", "MORENA", "PRD", "PVEM", "PT", "MC"]

raw = pd.read_parquet(PARQUET)

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

_MICE_COND_VARS = [
    'legislatura_num', 'n_cargos_legislativos_prev', 'fue_diputado_local',
    'fue_diputado_federal', 'fue_senador', 'n_trayectoria_legislativa',
    'n_trayectoria_admin', 'nivel_cargo_max', 'fue_presidente_mun',
    'fue_secretario_cargo', 'fue_director_general', 'fue_subsecretario',
    'admin_en_gobierno_fed', 'admin_en_gobierno_est', 'admin_en_gobierno_mun',
    'edad_al_tomar_cargo',
]
_mice_cols_model = [c for c in _MICE_COND_VARS if c in raw.columns]
_mice_X_model = raw[_mice_cols_model].copy()
_mice_imputer_model = IterativeImputer(
    max_iter=10, random_state=42, initial_strategy='mean', skip_complete=True,
)
_mice_result_model = _mice_imputer_model.fit_transform(_mice_X_model)
_mice_df_model = pd.DataFrame(_mice_result_model, columns=_mice_cols_model, index=raw.index)

raw['edad_missing'] = raw['edad_al_tomar_cargo'].isna().astype(int)
raw['edad_imp'] = np.where(
    raw['edad_al_tomar_cargo'].isna(),
    _mice_df_model['edad_al_tomar_cargo'].clip(18, 90),
    raw['edad_al_tomar_cargo'],
)

df = raw.copy()
df["era"] = df["legislatura_num"].map(ERA_MAP)
df["nodal_bin"] = (df["n_comisiones_nodales"] >= 1).astype(int)
df["lastre_bin"] = (df["n_comisiones_lastre"] >= 1).astype(int)

REGION_MAP = {
    "CDMX": "CDMX",
    **dict.fromkeys(["MEX","HGO","MOR","PUE","TLAX","QRO","GTO","AGS"], "CENTRO"),
    **dict.fromkeys(["VER","OAX","CHIS","TAB","GRO","CAM","YUC","QROO"], "SUR"),
    **dict.fromkeys(["NL","TAMPS","COAH","CHIH","SON","BC","BCS","SIN",
                     "DGO","ZAC","SLP","NAY"], "NORTE"),
    **dict.fromkeys(["JAL","COL","MICH"], "OCCIDENTE"),
    "DESCONOCIDO": "RP",
}
df["region"] = df["entidad_codigo"].map(REGION_MAP).fillna("CENTRO")
df["partido_cat"] = df["partido"].where(df["partido"].isin(TOP_PARTIDOS), "OTRO")
df["univ_elite"] = df[["acad_unam","acad_itam","acad_ibero","acad_itesm"]].max(axis=1)
df["sexo_bin"] = (df["sexo"] == "M").astype(int)

_p    = pd.get_dummies(df["partido_cat"],     prefix="p")
_reg  = pd.get_dummies(df["region"],          prefix="reg")
_area = pd.get_dummies(df["area_formacion"],  prefix="area")
df_enc = pd.concat([df, _p, _reg, _area], axis=1)

NUMERIC_FEATS = [
    "sexo_bin",
    "edad_imp", "edad_missing",
    "mayoria_relativa", "es_partido_mayoria", "legislatura_num",
    "grado_estudios_ord", "tiene_posgrado", "tiene_doctorado",
    "estudios_en_extranjero", "univ_publica", "univ_privada", "univ_extranjera",
    "univ_elite",
    "n_cargos_legislativos_prev", "fue_diputado_local",
    "fue_diputado_federal", "fue_senador", "n_trayectoria_legislativa",
    "n_trayectoria_admin", "nivel_cargo_max",
    "fue_presidente_mun", "fue_presidente_org", "fue_director_general",
    "fue_secretario_cargo", "fue_subsecretario", "fue_director",
    "fue_coordinador", "fue_delegado", "fue_asesor", "fue_regidor", "fue_sindico",
    "admin_en_partido","admin_en_sindicato","admin_en_universidad",
    "admin_en_gobierno_fed","admin_en_gobierno_est","admin_en_gobierno_mun",
    "n_trayectoria_politica", "tiene_exp_juvenil",
    "lider_juvenil_partido","lider_juvenil_gobierno","miembro_org_juvenil",
    "nivel_liderazgo_juvenil",
    "n_trayectoria_empresarial","n_investigacion_docencia","n_organos_gobierno",
]
_AREA_KEEP = [
    "area_Derecho",
    "area_Ciencias Políticas y Sociales",
    "area_Económico-Financiera",
]
DUMMY_FEATS = list(_p.columns) + list(_reg.columns) + [c for c in _AREA_KEEP if c in df_enc.columns]
FEAT_COLS = NUMERIC_FEATS + DUMMY_FEATS
print(f"Features totales: {len(FEAT_COLS)} (numericas={len(NUMERIC_FEATS)}, dummies={len(DUMMY_FEATS)})")

def get_Xy(era, target):
    mask = df_enc["era"] == era
    X = df_enc.loc[mask, FEAT_COLS].astype(float).reset_index(drop=True)
    y = df_enc.loc[mask, target].astype(float).reset_index(drop=True)
    # Nota: "partido_cat" ya entra al diseño como regresor (dummies p_*), por
    # lo que las medias de residuo por partido son cero por construccion
    # (X'e=0 incluye esas columnas) -- probar homogeneidad de residuos por
    # partido seria tautologico. "entidad_codigo" (estado) NO entra al
    # diseño -solo la region agregada, reg_*- por lo que es un agrupador
    # legitimo para probar heterogeneidad NO capturada por el modelo.
    entidad = df_enc.loc[mask, "entidad_codigo"].reset_index(drop=True)
    legis   = df_enc.loc[mask, "legislatura_num"].reset_index(drop=True)
    return X, y, entidad, legis

TARGETS = [
    ("nodal_bin", "Nodales (MPL)"),
    ("lastre_bin", "Lastre (MPL)"),
    ("n_comisiones_tematicas", "Tematicas (lineal)"),
]

# =====================================================================
# Ajuste OLS + diagnosticos
# =====================================================================
def fit_ols(era, target):
    X, y, entidad, legis = get_Xy(era, target)
    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X), columns=FEAT_COLS)
    X_design = sm.add_constant(X_sc, has_constant='add')
    res = sm.OLS(y, X_design).fit()
    return res, X_design, y, entidad, legis


def diag_linearity(res):
    try:
        rt = linear_reset(res, power=3, test_type='fitted', use_f=True)
        return {"reset_F": float(rt.fvalue), "reset_p": float(rt.pvalue)}
    except Exception as e:
        return {"reset_F": None, "reset_p": None, "error": str(e)}


def diag_homoscedasticity(res, X_design):
    resid = res.resid
    bp = het_breuschpagan(resid, X_design.values)
    # White simplificado (Wooldridge 2015 Sec. 8.3): reg. resid^2 ~ yhat + yhat^2
    yhat = res.fittedvalues
    Z = sm.add_constant(np.column_stack([yhat, yhat**2]))
    aux = sm.OLS(resid**2, Z).fit()
    white_F = float(aux.fvalue)
    white_p = float(aux.f_pvalue)
    return {
        "bp_lm": float(bp[0]), "bp_p": float(bp[1]),
        "white_F": white_F, "white_p": white_p,
    }


def diag_independence(res, entidad, legis):
    resid = res.resid.values
    order = np.argsort(legis.values, kind="stable")
    dw = float(durbin_watson(resid[order]))
    # ANOVA de residuos por ESTADO (entidad_codigo). Solo la region agregada
    # (reg_*, 5-6 categorias) entra al diseño -entidad_codigo (32 estados) no-,
    # por lo que esta prueba SI es informativa: detecta heterogeneidad
    # estatal (maquinaria politica local, no observada) que la region
    # agregada no absorbe.
    groups = [resid[entidad.values == e] for e in entidad.unique() if (entidad.values == e).sum() >= 5]
    if len(groups) >= 2:
        f_stat, f_p = scistats.f_oneway(*groups)
    else:
        f_stat, f_p = float("nan"), float("nan")
    return {"durbin_watson": dw, "entidad_anova_F": float(f_stat), "entidad_anova_p": float(f_p),
            "n_entidades_con_5mas": len(groups), "n_legislaturas_era": int(legis.nunique())}


def diag_normality(res):
    resid = res.resid.values
    sh_stat, sh_p = scistats.shapiro(resid if len(resid) <= 5000 else np.random.choice(resid, 5000, replace=False))
    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    return {"shapiro_W": float(sh_stat), "shapiro_p": float(sh_p),
            "jarque_bera": float(jb_stat), "jb_p": float(jb_p),
            "skew": float(skew), "kurtosis": float(kurt)}


DROP_REF_CATS = ["p_OTRO", "reg_CENTRO"]  # categorias de referencia para la version identificada


def _vif_block(X_design):
    vifs, names = [], []
    for i, col in enumerate(X_design.columns):
        if col == "const":
            continue
        v = variance_inflation_factor(X_design.values, i)
        vifs.append(v)
        names.append(col)
    return np.array(vifs), names


def diag_multicollinearity(res, X_design):
    # (a) Tal como esta especificado en v10: dummies completas de partido y
    #     region (sin categoria de referencia) -> trampa de variable dummy
    #     esperada (suma de dummies = 1 = colineal con el intercepto).
    vifs_raw, names_raw = _vif_block(X_design)
    inf_feats = [n for n, v in zip(names_raw, vifs_raw) if not np.isfinite(v)]

    # (b) Version identificada: se elimina (i) una categoria de referencia
    #     por bloque categorico (p_OTRO, reg_CENTRO), (ii) columnas de
    #     varianza (casi) nula dentro de esta era -especialmente frecuente
    #     en ERA_4, n=500- y (iii) una de cada par de columnas casi
    #     duplicadas (|r|>0.999, p.ej. estudios_en_extranjero/univ_extranjera
    #     resultan ser exactamente la misma variable). Esto aisla la
    #     colinealidad SUSTANTIVA neta de estos tres problemas de
    #     especificacion/identificacion.
    X_id = X_design.drop(columns=[c for c in DROP_REF_CATS if c in X_design.columns])

    zero_var = [c for c in X_id.columns if c != "const" and X_id[c].std() < 1e-8]
    X_id = X_id.drop(columns=zero_var)

    corr = X_id.drop(columns="const").corr().abs()
    dup_pairs, dropped_dups, seen = [], [], set()
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in seen or b in seen:
                continue
            if corr.loc[a, b] > 0.999:
                dup_pairs.append((a, b, float(corr.loc[a, b])))
                dropped_dups.append(b)
                seen.add(b)
    X_id = X_id.drop(columns=dropped_dups)

    # (iv) Cualquier dependencia lineal exacta remanente -combinaciones de 3+
    #      columnas que la deteccion pareada no puede ver- se resuelve con
    #      QR con pivoteo de columnas: si rank(X) < n_columnas, se descartan
    #      las columnas linealmente dependientes segun el orden de pivote.
    from scipy.linalg import qr as _qr
    rank_dropped = []
    Xv = X_id.drop(columns="const").values
    rank = np.linalg.matrix_rank(Xv)
    cols_now = X_id.drop(columns="const").columns.tolist()
    if rank < len(cols_now):
        _, _, piv = _qr(Xv, mode="economic", pivoting=True)
        rank_dropped = [cols_now[i] for i in piv[rank:]]
        X_id = X_id.drop(columns=rank_dropped)

    vifs_id, names_id = _vif_block(X_id)

    return {
        "max_vif_raw": float(np.nanmax(vifs_raw)) if len(vifs_raw) else None,
        "n_vif_inf_raw": len(inf_feats),
        "inf_features_raw": inf_feats,
        "condition_number_raw": float(res.condition_number),
        "zero_var_dropped": zero_var,
        "near_duplicate_pairs_dropped": dup_pairs,
        "rank_deficient_dropped": rank_dropped,
        "max_vif_identificado": float(np.nanmax(vifs_id)) if len(vifs_id) else None,
        "n_vif_gt5_identificado": int(np.nansum(vifs_id > 5)),
        "n_vif_gt10_identificado": int(np.nansum(vifs_id > 10)),
        "condition_number_identificado": float(np.linalg.cond(X_id.values)),
        "top5_vif_identificado": sorted(
            [{"feature": n, "VIF": float(v)} for n, v in zip(names_id, vifs_id)],
            key=lambda r: -r["VIF"],
        )[:5],
    }


def diag_influence(res, X_design):
    infl = OLSInfluence(res)
    cooks_d = infl.cooks_distance[0]
    leverage = infl.hat_matrix_diag
    dffits, dffits_thr = infl.dffits
    n = X_design.shape[0]
    k = X_design.shape[1]
    cooks_thr = 4.0 / n
    lev_thr = 2.0 * k / n
    return {
        "n": int(n), "k": int(k),
        "n_cooks_gt_thr": int(np.sum(cooks_d > cooks_thr)),
        "cooks_thr": float(cooks_thr),
        "n_leverage_gt_thr": int(np.sum(leverage > lev_thr)),
        "leverage_thr": float(lev_thr),
        "n_dffits_gt_thr": int(np.sum(np.abs(dffits) > dffits_thr)),
        "dffits_thr": float(dffits_thr),
        "max_cooks_d": float(np.max(cooks_d)),
    }, cooks_d, leverage


def fit_summary(res, target):
    return {
        "n": int(res.nobs), "k": int(res.df_model),
        "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj),
        "f_stat": float(res.fvalue), "f_p": float(res.f_pvalue),
        "aic": float(res.aic), "bic": float(res.bic),
    }


# =====================================================================
# Figuras diagnosticas: 1 figura por target, grid 2x2 por era
# =====================================================================
def make_diagnostic_figures(target_key, target_label, fits):
    # A. Residuos vs. valores ajustados
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, era in zip(axes.flatten(), ERA_ORDER):
        res = fits[era]["res"]
        ax.scatter(res.fittedvalues, res.resid, s=10, alpha=0.4, color="#2E86AB")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(ERA_LABELS[era], fontsize=10)
        ax.set_xlabel("Valores ajustados")
        ax.set_ylabel("Residuos")
    fig.suptitle(f"Residuos vs. ajustados — {target_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p1 = REPORT_DIR / f"v0_resid_fitted_{target_key}.png"
    fig.savefig(p1, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # B. QQ-plot de residuos
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, era in zip(axes.flatten(), ERA_ORDER):
        res = fits[era]["res"]
        sm.qqplot(res.resid, line="s", ax=ax, markersize=3, alpha=0.4)
        ax.set_title(ERA_LABELS[era], fontsize=10)
    fig.suptitle(f"Q-Q plot de residuos — {target_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p2 = REPORT_DIR / f"v0_qqplot_{target_key}.png"
    fig.savefig(p2, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # C. Leverage vs. residuos estudentizados, tamano = Cook's D
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, era in zip(axes.flatten(), ERA_ORDER):
        res = fits[era]["res"]
        cooks_d = fits[era]["cooks_d"]
        leverage = fits[era]["leverage"]
        infl = OLSInfluence(res)
        stud = infl.resid_studentized_external
        size = 20 + 800 * (cooks_d / (cooks_d.max() + 1e-12))
        ax.scatter(leverage, stud, s=size, alpha=0.4, color="#E07A5F", edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(3, color="red", linestyle=":", linewidth=0.8)
        ax.axhline(-3, color="red", linestyle=":", linewidth=0.8)
        ax.set_title(ERA_LABELS[era], fontsize=10)
        ax.set_xlabel("Leverage (hat)")
        ax.set_ylabel("Residuo estudentizado")
    fig.suptitle(f"Influencia: leverage x residuo estudentizado (tamaño = Cook's D) — {target_label}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p3 = REPORT_DIR / f"v0_influence_{target_key}.png"
    fig.savefig(p3, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return {"resid_fitted": p1.name, "qqplot": p2.name, "influence": p3.name}


# =====================================================================
# Corrida principal
# =====================================================================
manifest = {}
for target_key, target_label in TARGETS:
    manifest[target_key] = {"label": target_label, "eras": {}}
    fits = {}
    for era in ERA_ORDER:
        res, X_design, y, entidad, legis = fit_ols(era, target_key)
        infl_stats, cooks_d, leverage = diag_influence(res, X_design)
        fits[era] = {"res": res, "X_design": X_design, "cooks_d": cooks_d, "leverage": leverage}

        entry = {
            "fit": fit_summary(res, target_key),
            "linearity": diag_linearity(res),
            "homoscedasticity": diag_homoscedasticity(res, X_design),
            "independence": diag_independence(res, entidad, legis),
            "normality": diag_normality(res),
            "multicollinearity": diag_multicollinearity(res, X_design),
            "influence": infl_stats,
        }
        manifest[target_key]["eras"][era] = entry
        mc = entry['multicollinearity']
        maxvif_id = mc['max_vif_identificado']
        maxvif_id_str = f"{maxvif_id:.2f}" if maxvif_id is not None else "NA"
        print(f"[{target_key}] {era}: R2={entry['fit']['r2']:.3f} "
              f"RESET_p={entry['linearity']['reset_p']:.4f} "
              f"BP_p={entry['homoscedasticity']['bp_p']:.4g} "
              f"Shapiro_p={entry['normality']['shapiro_p']:.4g} "
              f"VIFinf_raw={mc['n_vif_inf_raw']} "
              f"maxVIF_id={maxvif_id_str} "
              f"zero_var_dropped={mc['zero_var_dropped']} "
              f"dup_dropped={mc['near_duplicate_pairs_dropped']} "
              f"rank_dropped={mc['rank_deficient_dropped']} "
              f"CooksD>thr={entry['influence']['n_cooks_gt_thr']}")

    figs = make_diagnostic_figures(target_key, target_label, fits)
    manifest[target_key]["figures"] = figs

with open(SCRATCH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)

print(f"\nManifiesto guardado en {SCRATCH}")
print("Figuras guardadas en", REPORT_DIR)
