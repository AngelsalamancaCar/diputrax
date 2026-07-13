import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = Path("C:/Users/zigma/Projects/diputrax")
PARQUET = PROJECT_ROOT / "data" / "clean" / "diputados_20260421_205712.parquet"

# =====================================================================
# Replicates notebook cells 69-76 (imports, config, MICE, feature eng)
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
    return X, y

_L1_PARAMS = dict(l1_ratio=1, solver='liblinear', C=0.1, max_iter=3000,
                  class_weight='balanced', random_state=42)

def sfm_report(X, y):
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    sel = SelectFromModel(LogisticRegression(**_L1_PARAMS), threshold="mean")
    sel.fit(X_sc, y)
    return [FEAT_COLS[i] for i, s in enumerate(sel.get_support()) if s]

sfm_selected_nodal = {era: sfm_report(*get_Xy(era, "nodal_bin")) for era in ERA_ORDER}
sfm_selected_lastre = {era: sfm_report(*get_Xy(era, "lastre_bin")) for era in ERA_ORDER}

print("\n--- SFM selected features (nodal) ---")
for era in ERA_ORDER:
    print(era, len(sfm_selected_nodal[era]), sfm_selected_nodal[era])
print("\n--- SFM selected features (lastre) ---")
for era in ERA_ORDER:
    print(era, len(sfm_selected_lastre[era]), sfm_selected_lastre[era])

KEY_FEATS = [
    "es_partido_mayoria", "n_cargos_legislativos_prev",
    "fue_secretario_cargo", "n_trayectoria_admin",
    "n_trayectoria_politica", "n_trayectoria_legislativa",
    "edad_imp", "mayoria_relativa",
]

# =====================================================================
# D.1 — statsmodels Logit clasico (MLE no ponderado) sobre features SFM
# =====================================================================
def logit_classic_table(era, target, sfm_dict):
    feats = sfm_dict[era]
    X, y = get_Xy(era, target)
    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X[feats]), columns=feats)
    X_design = sm.add_constant(X_sc, has_constant='add')
    try:
        res = sm.Logit(y, X_design).fit(disp=0, maxiter=200)
        converged = res.mle_retvals.get('converged', True)
    except Exception as e:
        return None, None, str(e)

    ci = res.conf_int(alpha=0.05)
    tbl = pd.DataFrame({
        'coef': res.params,
        'std_err': res.bse,
        'z': res.tvalues,
        'p_value': res.pvalues,
        'ci_lo': ci[0],
        'ci_hi': ci[1],
    })
    tbl['odds_ratio'] = np.exp(tbl['coef'])
    tbl['or_ci_lo'] = np.exp(tbl['ci_lo'])
    tbl['or_ci_hi'] = np.exp(tbl['ci_hi'])
    tbl['sig'] = tbl['p_value'].apply(lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else '')))

    # classification metrics @ 0.5
    y_pred_prob = res.predict(X_design)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    model_stats = {
        'era': era, 'n': int(res.nobs), 'n_feat': len(feats),
        'converged': converged,
        'llf': res.llf, 'llnull': res.llnull,
        'prsquared_mcfadden': res.prsquared,
        'llr_chi2': res.llr, 'llr_pvalue': res.llr_pvalue,
        'aic': res.aic, 'bic': res.bic,
        'precision': prec, 'recall': rec, 'f1': f1,
        'confusion_matrix': cm.tolist(),
    }
    return tbl, model_stats, None

results = {'nodal': {}, 'lastre': {}}
for era in ERA_ORDER:
    tbl, mstats, err = logit_classic_table(era, "nodal_bin", sfm_selected_nodal)
    results['nodal'][era] = {'table': tbl, 'stats': mstats, 'error': err}
    tbl2, mstats2, err2 = logit_classic_table(era, "lastre_bin", sfm_selected_lastre)
    results['lastre'][era] = {'table': tbl2, 'stats': mstats2, 'error': err2}

for target in ['nodal', 'lastre']:
    print(f"\n=========== {target.upper()} ===========")
    for era in ERA_ORDER:
        r = results[target][era]
        if r['error']:
            print(era, "ERROR:", r['error'])
            continue
        print(f"\n-- {era} -- n={r['stats']['n']} pseudoR2={r['stats']['prsquared_mcfadden']:.4f} "
              f"LLR_p={r['stats']['llr_pvalue']:.4g} conv={r['stats']['converged']} "
              f"P={r['stats']['precision']:.3f} R={r['stats']['recall']:.3f} F1={r['stats']['f1']:.3f}")
        print(r['table'].round(4).to_string())

# =====================================================================
# D.2 — VIF por era (mismo subconjunto SFM) -- multicolinealidad
# =====================================================================
def vif_table(era, sfm_dict, target):
    feats = sfm_dict[era]
    X, y = get_Xy(era, target)
    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X[feats]), columns=feats)
    X_design = sm.add_constant(X_sc, has_constant='add')
    vifs = []
    for i, col in enumerate(X_design.columns):
        if col == 'const':
            continue
        v = variance_inflation_factor(X_design.values, i)
        vifs.append({'feature': col, 'VIF': v})
    return pd.DataFrame(vifs).sort_values('VIF', ascending=False)

vif_results = {'nodal': {}, 'lastre': {}}
for era in ERA_ORDER:
    vif_results['nodal'][era] = vif_table(era, sfm_selected_nodal, "nodal_bin")
    vif_results['lastre'][era] = vif_table(era, sfm_selected_lastre, "lastre_bin")

print("\n=========== VIF NODAL (max por era) ===========")
for era in ERA_ORDER:
    v = vif_results['nodal'][era]
    print(era, "max VIF:", round(v['VIF'].max(), 3), "feature:", v.iloc[0]['feature'])
print("\n=========== VIF LASTRE (max por era) ===========")
for era in ERA_ORDER:
    v = vif_results['lastre'][era]
    print(era, "max VIF:", round(v['VIF'].max(), 3), "feature:", v.iloc[0]['feature'])

# =====================================================================
# D.3 — GLM Poisson clasico sobre KEY_FEATS -- tematicas
# =====================================================================
def poisson_classic_table(era):
    X, y = get_Xy(era, "n_comisiones_tematicas")
    feats = [f for f in KEY_FEATS if f in X.columns]
    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X[feats]), columns=feats)
    X_design = sm.add_constant(X_sc, has_constant='add')
    model = sm.GLM(y, X_design, family=sm.families.Poisson())
    res = model.fit()

    ci = res.conf_int(alpha=0.05)
    tbl = pd.DataFrame({
        'coef': res.params,
        'std_err': res.bse,
        'z': res.tvalues,
        'p_value': res.pvalues,
        'ci_lo': ci[0],
        'ci_hi': ci[1],
    })
    tbl['irr'] = np.exp(tbl['coef'])
    tbl['irr_ci_lo'] = np.exp(tbl['ci_lo'])
    tbl['irr_ci_hi'] = np.exp(tbl['ci_hi'])
    tbl['sig'] = tbl['p_value'].apply(lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else '')))

    null_model = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Poisson()).fit()
    pseudo_r2 = 1 - (res.deviance / null_model.deviance)

    model_stats = {
        'era': era, 'n': int(res.nobs), 'n_feat': len(feats),
        'deviance': res.deviance, 'null_deviance': null_model.deviance,
        'pseudo_r2_deviance': pseudo_r2,
        'aic': res.aic, 'bic': res.bic_llf if hasattr(res, 'bic_llf') else None,
        'pearson_chi2': res.pearson_chi2,
        'dispersion': res.pearson_chi2 / res.df_resid,
    }
    return tbl, model_stats

poisson_results = {}
for era in ERA_ORDER:
    tbl, mstats = poisson_classic_table(era)
    poisson_results[era] = {'table': tbl, 'stats': mstats}

print("\n=========== POISSON GLM (TEMATICAS) ===========")
for era in ERA_ORDER:
    r = poisson_results[era]
    print(f"\n-- {era} -- n={r['stats']['n']} pseudoR2_dev={r['stats']['pseudo_r2_deviance']:.4f} "
          f"dispersion={r['stats']['dispersion']:.3f}")
    print(r['table'].round(4).to_string())

# =====================================================================
# Persist everything for building notebook cell outputs
# =====================================================================
out_dir = Path("C:/Users/zigma/AppData/Local/Temp/claude/C--Users-zigma-Projects-diputrax/fded5d77-b987-4bc7-ad03-bd1ec8c5a023/scratchpad")

def df_to_records(d):
    return d.reset_index().rename(columns={'index': 'feature'}).to_dict('records')

payload = {
    'sfm_selected_nodal': sfm_selected_nodal,
    'sfm_selected_lastre': sfm_selected_lastre,
    'key_feats': KEY_FEATS,
    'logit_nodal': {
        era: {
            'table': df_to_records(results['nodal'][era]['table']) if results['nodal'][era]['table'] is not None else None,
            'stats': results['nodal'][era]['stats'],
        } for era in ERA_ORDER
    },
    'logit_lastre': {
        era: {
            'table': df_to_records(results['lastre'][era]['table']) if results['lastre'][era]['table'] is not None else None,
            'stats': results['lastre'][era]['stats'],
        } for era in ERA_ORDER
    },
    'vif_nodal': {era: vif_results['nodal'][era].to_dict('records') for era in ERA_ORDER},
    'vif_lastre': {era: vif_results['lastre'][era].to_dict('records') for era in ERA_ORDER},
    'poisson_tem': {
        era: {
            'table': df_to_records(poisson_results[era]['table']),
            'stats': poisson_results[era]['stats'],
        } for era in ERA_ORDER
    },
}
with open(out_dir / "anexoD_results.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=1, default=str)

print("\n\nDONE. Results saved to anexoD_results.json")
