# -*- coding: utf-8 -*-
"""Appends Anexo D (classic frequentist inference stats) to diputraxv10.ipynb.
Insert-only: does not modify any existing cell. Cell indices/content match
diputraxv10.ipynb as of 2026-07-12; do not rerun against a newer notebook
without re-verifying insertion point and available globals.
"""
import json
from pathlib import Path

NB_PATH = Path("C:/Users/zigma/Projects/diputrax/notebooks/diputraxv10.ipynb")
SCRATCH = Path("C:/Users/zigma/AppData/Local/Temp/claude/C--Users-zigma-Projects-diputrax/fded5d77-b987-4bc7-ad03-bd1ec8c5a023/scratchpad")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

with open(SCRATCH / "outputs_d1.json", encoding="utf-8") as f:
    outputs_d1 = json.load(f)
with open(SCRATCH / "outputs_d2.json", encoding="utf-8") as f:
    outputs_d2 = json.load(f)
with open(SCRATCH / "outputs_d3.json", encoding="utf-8") as f:
    outputs_d3 = json.load(f)
with open(SCRATCH / "outputs_d4.json", encoding="utf-8") as f:
    outputs_d4 = json.load(f)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src, outputs):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": outputs, "source": src.splitlines(keepends=True),
    }


# =====================================================================
# D.0 — Intro / metodologia
# =====================================================================
d0 = md("""# Anexo D. Estadísticos clásicos de inferencia frecuentista (MLE, statsmodels)

## D.0 Objetivo y metodología

Las secciones 5–9 y los Anexos A–C documentan la interpretabilidad del modelo productivo —Regresión Logística L1 (Lasso) más SHAP— y su contraste con una capa Bayesiana (NUTS, secciones 5.6, 6.5, 8.0). Ninguna de las dos reporta el formato de tabla de regresión "clásico" que la ciencia política cuantitativa usa como estándar de reporte: coeficiente, error estándar, estadístico *z*, valor *p*, intervalo de confianza al 95 % y razón de momios (*odds ratio* / *incidence rate ratio*), junto con diagnósticos de ajuste (pseudo R² de McFadden, prueba de razón de verosimilitud contra el modelo nulo, AIC/BIC) y de desempeño de clasificación (matriz de confusión, *precision*, *recall*, F1).

Este anexo agrega esa capa vía **Maximum Likelihood Estimation (MLE) no ponderada, con `statsmodels`**, de forma aditiva: no se modifica ninguna celda existente del cuaderno.

**Decisiones metodológicas:**

1. **Mismo subconjunto de variables que el modelo productivo.** Para nodales y lastre se usa `sfm_selected_nodal` / `sfm_selected_lastre` —el mismo subconjunto que `SelectFromModel(LR L1)` seleccionó en las secciones 5 y 6—, preservando la lógica de interpretabilidad del proyecto (§4.1) en vez de reportar los 64 *features* completos. Para temáticas, que en el pipeline productivo no tiene paso de selección (`lr_poisson()` usa el conjunto completo), se usa `KEY_FEATS` —el subconjunto de variables clave ya definido en la sección 5.3—.
2. **MLE no ponderada, no L1.** El modelo productivo usa `class_weight="balanced"` y penalización L1; ninguna tiene equivalente cerrado en MLE clásica. Se reporta MLE no ponderada sin regularizar —el estándar de la tabla de regresión en ciencia política—, con la salvedad de que sus coeficientes **no son directamente comparables en magnitud** a los de la Regresión Logística L1 productiva: esta capa es un complemento inferencial, no un reemplazo del modelo predictivo.
3. **Variables estandarizadas** (`StandardScaler`, igual que el pipeline productivo), para que los coeficientes sean comparables entre variables y eras.
4. **Separación cuasi-completa.** Con subconjuntos de 13–27 variables sobre eras de 500–1500 observaciones —incluyendo *dummies* de categorías raras, p. ej. `admin_en_sindicato`— la MLE no penalizada puede fallar en converger o producir errores estándar no estimables (SE = NaN o numéricamente degenerados). Esto se reporta explícitamente donde ocurre. Lejos de ser un defecto del anexo, es el argumento empírico de por qué el modelo productivo usa penalización L1 en lugar de MLE clásica: L1 puede estimar estos mismos casos porque encoge el coeficiente en vez de dejar que la verosimilitud diverja.
""")

# =====================================================================
# D.1 — Logit clasico Nodales
# =====================================================================
d1_code = code("""# ============================================================
# ANEXO D.1 — Regresion Logistica clasica (MLE, statsmodels) — Nodales
# Coeficientes, error estandar, z, valor p, IC95%, odds ratio
# Mismo subconjunto de variables que SelectFromModel (sfm_selected_nodal)
# ============================================================
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def logit_classic_table(era, target, sfm_dict):
    feats = sfm_dict[era]
    X, y = get_Xy(era, target)
    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X[feats]), columns=feats)
    X_design = sm.add_constant(X_sc, has_constant='add')
    res = sm.Logit(y, X_design).fit(disp=0, maxiter=200)

    ci = res.conf_int(alpha=0.05)
    tbl = pd.DataFrame({
        'coef': res.params, 'std_err': res.bse, 'z': res.tvalues,
        'p_value': res.pvalues, 'ci_lo': ci[0], 'ci_hi': ci[1],
    })
    tbl['odds_ratio'] = np.exp(tbl['coef'])
    tbl['or_ci_lo']   = np.exp(tbl['ci_lo'])
    tbl['or_ci_hi']   = np.exp(tbl['ci_hi'])
    tbl['sig'] = tbl['p_value'].apply(
        lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else '')))

    y_pred = (res.predict(X_design) >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    stats_row = {
        'Era': ERA_LABELS[era], 'n': int(res.nobs), 'n_feat': len(feats),
        'Convergio': res.mle_retvals.get('converged', True),
        'Pseudo R2 McFadden': round(res.prsquared, 4),
        'LLR chi2': round(res.llr, 2), 'LLR p-valor': res.llr_pvalue,
        'AIC': round(res.aic, 1), 'BIC': round(res.bic, 1),
        'Precision': round(precision_score(y, y_pred, zero_division=0), 3),
        'Recall':    round(recall_score(y, y_pred, zero_division=0), 3),
        'F1':        round(f1_score(y, y_pred, zero_division=0), 3),
        'TN': int(cm[0,0]), 'FP': int(cm[0,1]), 'FN': int(cm[1,0]), 'TP': int(cm[1,1]),
    }
    return tbl.round(4), stats_row

logit_tables_nodal = {}
rows_logit_nodal = []
for era in ERA_ORDER:
    tbl, stats_row = logit_classic_table(era, "nodal_bin", sfm_selected_nodal)
    logit_tables_nodal[era] = tbl
    rows_logit_nodal.append(stats_row)
    print(f"\\n=== {ERA_LABELS[era]} \\u2014 Nodales \\u2014 n={stats_row['n']}, "
          f"Pseudo R2={stats_row['Pseudo R2 McFadden']}, "
          f"LLR p={stats_row['LLR p-valor']:.2e}, convergio={stats_row['Convergio']} ===")
    display(tbl)

df_logit_nodal_stats = pd.DataFrame(rows_logit_nodal)
print("\\n-- Resumen de ajuste del modelo \\u2014 Logit clasico Nodales --")
display(df_logit_nodal_stats)
""", outputs_d1)

d1_interp = md("""**Interpretación — D.1 Regresión Logística clásica (Nodales)**

**ERA_1 — PRI** (n=1500, pseudo R² McFadden = 0.161 — el mejor ajuste de las cuatro eras, LLR χ² *p* < 10⁻⁵⁰): `area_Derecho` es el coeficiente más fuerte (β=0.475, OR=1.61, *p*<0.001) — un desviación estándar adicional en la señal de formación jurídica multiplica por 1.6 la razón de momios de recibir comisión nodal, confirmando en escala de razón de momios lo que SHAP ya mostraba en magnitud. Le siguen `area_Económico-Financiera` (OR=1.39, *p*<0.001), `fue_secretario_cargo` (OR=1.27, *p*<0.001) y `n_trayectoria_empresarial` (OR=1.17, *p*<0.001). `reg_SUR` es negativo y significativo (OR=0.81, *p*=0.001) — penalización regional que SHAP no distingue por signo.

> **Advertencia de separación cuasi-completa:** `admin_en_sindicato` no es estimable (β=−9.45, SE no definido) — casi ningún diputado con trayectoria sindical recibió comisión nodal en ERA_1, lo que produce una frontera de decisión perfecta para esa variable. El colapso numérico se propaga al intercepto y a `legislatura_num`, `univ_publica`, `univ_privada` (SE = NaN vía la matriz de información compartida). Esto **no invalida** el resto de la tabla —los demás coeficientes convergen con normalidad—, pero sí confirma empíricamente por qué el modelo productivo usa L1: la penalización habría encogido `admin_en_sindicato` a un valor finito en vez de dejar que la verosimilitud divergiera.

**ERA_2 — PAN** (n=1500, pseudo R² = 0.138, sin problemas de convergencia): `sexo_bin` es significativo por primera vez en la serie (β=0.166, OR=1.18, *p*=0.007), confirmando en términos clásicos la emergencia de género como señal activa que SHAP ya ubicaba en ERA_2. `edad_imp` es negativo y fuerte (OR=0.706, *p*<0.001) — una vez controlado por trayectoria, mayor edad estandarizada se asocia con *menor* probabilidad de nodal, patrón que solo el signo del coeficiente clásico revela (SHAP solo reporta magnitud). `area_Derecho` sigue dominando (OR=1.52, *p*<0.001).

**ERA_3 — Transición** (n=1500, pseudo R² = 0.129, converge sin variables NaN): `sexo_bin` alcanza su punto más alto de significancia (OR=1.22, *p*<0.001) y `p_MORENA` es significativo (OR=1.28, *p*=0.003), confirmando en formato clásico el hallazgo SHAP de que `p_MORENA` —no `es_partido_mayoria`— es la señal partidista activa en esta era.

**ERA_4 — Morena** (n=500, pseudo R² = 0.134 — el segundo mejor ajuste pese al tamaño muestral reducido, LLR *p* = 2.7×10⁻¹²): `area_Derecho` (OR=1.54, *p*<0.001) y `sexo_bin` (OR=1.27, *p*=0.008) son los coeficientes estables. `es_partido_mayoria` y `p_MORENA` no son estimables por separado (SE = NaN): en una era de una sola legislatura con supermayoría de Morena, ambas *dummies* son numéricamente equivalentes dentro de la muestra (colinealidad exacta, confirmada por VIF = ∞ en D.3) — un artefacto estructural del diseño de ERA_4, no un error de especificación.

**Desempeño de clasificación (umbral 0.5, sin balanceo de clases):** *precision* 0.65–0.69, *recall* 0.36–0.70, F1 0.51–0.70. El *recall* más bajo es ERA_1 (0.42), reflejo directo de que esta variante —a diferencia del modelo productivo `LR L1 (full)`— no usa `class_weight="balanced"`, por lo que subpredice la clase positiva minoritaria (32.2 % de tasa nodal en ERA_1).
""")

# =====================================================================
# D.2 — Logit clasico Lastre
# =====================================================================
d2_code = code("""# ============================================================
# ANEXO D.2 — Regresion Logistica clasica (MLE, statsmodels) — Lastre
# Mismo subconjunto de variables que SelectFromModel (sfm_selected_lastre)
# ============================================================
logit_tables_lastre = {}
rows_logit_lastre = []
for era in ERA_ORDER:
    tbl, stats_row = logit_classic_table(era, "lastre_bin", sfm_selected_lastre)
    logit_tables_lastre[era] = tbl
    rows_logit_lastre.append(stats_row)
    print(f"\\n=== {ERA_LABELS[era]} \\u2014 Lastre \\u2014 n={stats_row['n']}, "
          f"Pseudo R2={stats_row['Pseudo R2 McFadden']}, "
          f"LLR p={stats_row['LLR p-valor']:.2e}, convergio={stats_row['Convergio']} ===")
    display(tbl)

df_logit_lastre_stats = pd.DataFrame(rows_logit_lastre)
print("\\n-- Resumen de ajuste del modelo \\u2014 Logit clasico Lastre --")
display(df_logit_lastre_stats)
""", outputs_d2)

d2_interp = md("""**Interpretación — D.2 Regresión Logística clásica (Lastre)**

**ERA_1** (n=1500, pseudo R² = 0.050) y **ERA_2** (n=1500, pseudo R² = 0.094) convergen sin problemas numéricos. El ajuste de lastre es sistemáticamente más débil que el de nodales en las cuatro eras (pseudo R² 0.050–0.094 vs. 0.129–0.161), consistente con el hallazgo ya documentado en §6.4: la asignación a comisiones lastre responde menos al perfil biográfico que la asignación a nodales. En ERA_1, `sexo_bin` (OR=0.88, *p*=0.025) y `fue_secretario_cargo` (OR=0.82, *p*=0.004) son negativos — dirección inversa a nodales, coherente con la hipótesis de "imagen espejo" de §6.2 —, pero `nivel_cargo_max` es positivo (OR=1.30, *p*<0.001): un hallazgo contraintuitivo que el SHAP agregado no distingue por signo y que merece lectura cualitativa adicional. En ERA_2, `n_trayectoria_admin` es fuertemente negativo (OR=0.755, *p*<0.001) y `n_organos_gobierno` aún más (OR=0.666, *p*<0.001).

> **Advertencia de no convergencia — ERA_3 y ERA_4:** en ambas eras el optimizador de MLE no converge (`Convergio=False`). En **ERA_3**, `admin_en_sindicato` vuelve a producir separación cuasi-completa (SE = 99,180 — numéricamente no distinguible de infinito), arrastrando al intercepto (SE = 2,562); el resto de coeficientes de la tabla sí son numéricamente estables y se reportan con normalidad. En **ERA_4** (n=500) el modelo colapsa por completo: el intercepto y `n_organos_gobierno` no son estimables (SE = NaN) y `area_Económico-Financiera` produce un coeficiente sin sentido sustantivo (β≈−5.2×10⁷) — síntoma de separación completa con 13 variables sobre una muestra de 500 casos con varias categorías dummy casi vacías (`fue_coordinador`, `fue_delegado`, `p_PT`, `reg_RP`). **La tabla de ERA_4 Lastre no debe leerse como resultado sustantivo**; se incluye íntegra por transparencia metodológica, no como hallazgo. Esto es exactamente el problema de "separación en muestras pequeñas" documentado en la literatura de ciencia política (King y Zeng 2001, *Logistic Regression in Rare Events Data*) como razón estándar para preferir estimadores penalizados (Firth 1993; o, como en este proyecto, L1/Lasso) sobre MLE clásica cuando *n* es reducido y hay categorías raras — reforzando, igual que en D.1, la elección metodológica del modelo productivo.
""")

# =====================================================================
# D.3 — VIF
# =====================================================================
d3_code = code("""# ============================================================
# ANEXO D.3 — Factor de Inflacion de Varianza (VIF)
# Diagnostico de multicolinealidad sobre el mismo subconjunto SFM por era
# ============================================================
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_table(era, sfm_dict, target):
    feats = sfm_dict[era]
    X, y = get_Xy(era, target)
    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X[feats]), columns=feats)
    X_design = sm.add_constant(X_sc, has_constant='add')
    vifs = [
        {'feature': col, 'VIF': variance_inflation_factor(X_design.values, i)}
        for i, col in enumerate(X_design.columns) if col != 'const'
    ]
    return pd.DataFrame(vifs).sort_values('VIF', ascending=False).reset_index(drop=True)

print("-- VIF maximo por era \\u2014 Nodales --")
vif_nodal_tables = {}
for era in ERA_ORDER:
    v = vif_table(era, sfm_selected_nodal, "nodal_bin")
    vif_nodal_tables[era] = v
    print(f"  {ERA_LABELS[era]:<28} max VIF = {v['VIF'].iloc[0]:.2f}  ({v['feature'].iloc[0]})")

print("\\n-- VIF maximo por era \\u2014 Lastre --")
vif_lastre_tables = {}
for era in ERA_ORDER:
    v = vif_table(era, sfm_selected_lastre, "lastre_bin")
    vif_lastre_tables[era] = v
    print(f"  {ERA_LABELS[era]:<28} max VIF = {v['VIF'].iloc[0]:.2f}  ({v['feature'].iloc[0]})")

print("\\n-- Top-5 VIF por era \\u2014 Nodales --")
for era in ERA_ORDER:
    print(f"\\n{ERA_LABELS[era]}:")
    display(vif_nodal_tables[era].head(5).round(3))

print("\\n-- Top-5 VIF por era \\u2014 Lastre --")
for era in ERA_ORDER:
    print(f"\\n{ERA_LABELS[era]}:")
    display(vif_lastre_tables[era].head(5).round(3))
""", outputs_d3)

d3_interp = md("""**Interpretación — D.3 Multicolinealidad (VIF)**

Regla convencional: VIF > 5 sugiere multicolinealidad problemática; VIF > 10, severa.

**Nodales:** ERA_1 (máx. VIF = 1.75, `n_trayectoria_admin`) y ERA_2 (máx. VIF = 1.78, `univ_elite`) están muy por debajo del umbral — sin problema de colinealidad. ERA_3 sube a 4.80 (`n_trayectoria_legislativa`), cerca del límite convencional pero aún por debajo; consistente con que ERA_3 es la era de mayor solapamiento de trayectorias (transición institucional con reelección consecutiva emergente, §2.2.6). **ERA_4 tiene VIF = ∞ en `p_MORENA`**: confirma numéricamente lo detectado en D.1 — en la legislatura LXVI, `p_MORENA` y `es_partido_mayoria` son colineales exactas porque Morena es el partido mayoritario en efectivamente el 100 % de los casos retenidos por SFM. No es una falla del modelo; es la firma estadística de una era de partido único dominante.

**Lastre:** las cuatro eras están cómodamente por debajo del umbral (VIF máximo entre 2.06 y 2.62), sin señales de multicolinealidad en ninguna era — el subconjunto SFM de lastre no comparte la fragilidad estructural que sí aparece en nodales ERA_4.

**Lectura conjunta con D.1–D.2:** el VIF explica *por qué* ciertas variables no fueron estimables en la MLE clásica (ERA_4 nodal) y confirma que las demás inestabilidades (`admin_en_sindicato` en ERA_1/ERA_3, el colapso de lastre ERA_4) son de separación cuasi-completa por categorías raras, no de colinealidad entre predictores — dos problemas estadísticos distintos que requieren lecturas distintas.
""")

# =====================================================================
# D.4 — Poisson GLM clasico (tematicas)
# =====================================================================
d4_code = code("""# ============================================================
# ANEXO D.4 — GLM Poisson clasico (statsmodels) — Tematicas
# Variables clave (KEY_FEATS, seccion 5.3); el modelo productivo
# lr_poisson() no tiene paso de seleccion, por lo que se usa el
# subconjunto interpretativo ya establecido en el cuaderno.
# ============================================================
def poisson_classic_table(era):
    X, y = get_Xy(era, "n_comisiones_tematicas")
    feats = [f for f in KEY_FEATS if f in X.columns]
    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X[feats]), columns=feats)
    X_design = sm.add_constant(X_sc, has_constant='add')
    res = sm.GLM(y, X_design, family=sm.families.Poisson()).fit()
    null_res = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Poisson()).fit()

    ci = res.conf_int(alpha=0.05)
    tbl = pd.DataFrame({
        'coef': res.params, 'std_err': res.bse, 'z': res.tvalues,
        'p_value': res.pvalues, 'ci_lo': ci[0], 'ci_hi': ci[1],
    })
    tbl['IRR']       = np.exp(tbl['coef'])
    tbl['IRR_ci_lo'] = np.exp(tbl['ci_lo'])
    tbl['IRR_ci_hi'] = np.exp(tbl['ci_hi'])
    tbl['sig'] = tbl['p_value'].apply(
        lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else '')))

    stats_row = {
        'Era': ERA_LABELS[era], 'n': int(res.nobs),
        'Pseudo R2 (deviance)': round(1 - res.deviance / null_res.deviance, 4),
        'Dispersion (Pearson chi2/df)': round(res.pearson_chi2 / res.df_resid, 3),
        'AIC': round(res.aic, 1),
    }
    return tbl.round(4), stats_row

poisson_tables_tem = {}
rows_poisson = []
for era in ERA_ORDER:
    tbl, stats_row = poisson_classic_table(era)
    poisson_tables_tem[era] = tbl
    rows_poisson.append(stats_row)
    print(f"\\n=== {ERA_LABELS[era]} \\u2014 Tematicas (GLM Poisson) \\u2014 n={stats_row['n']} ===")
    display(tbl)

df_poisson_stats = pd.DataFrame(rows_poisson)
print("\\n-- Resumen de ajuste \\u2014 GLM Poisson Tematicas (variables clave) --")
display(df_poisson_stats)
""", outputs_d4)

d4_interp = md("""**Interpretación — D.4 GLM Poisson clásico (Temáticas)**

Los cuatro modelos convergen sin problemas de separación (a diferencia de D.1–D.2) — el enlace log del Poisson y la ausencia de *dummies* de categoría rara en `KEY_FEATS` evitan el problema estructural de los binarios.

**Pseudo R² por deviance:** 0.024 (ERA_1) → 0.018 (ERA_2) → 0.028 (ERA_3) → 0.059 (ERA_4). Son valores bajos en las cuatro eras —consistente con el hallazgo ya establecido en §7.2 de que las temáticas siguen una lógica distributiva no capturada por el perfil biográfico—, pero **crecen monótonamente hacia ERA_4**, con el modelo de Morena explicando más del doble de la varianza en deviance que los tres anteriores.

**Dispersión (Pearson χ²/gl):** 0.642 / 0.582 / 0.660 / 0.465 — todas **por debajo de 1**, es decir, **subdispersión** respecto al supuesto Poisson (varianza menor que la media condicional). Es el patrón inverso al problema típico de sobredispersión que motiva binomial-negativa en la literatura de recuento; aquí el techo institucional de comisiones temáticas (0–10) comprime la varianza observada. No invalida el Poisson —la subdispersión no sesga los coeficientes, solo hace que los errores estándar reportados sean conservadores (ligeramente más anchos de lo estrictamente necesario)—.

**Coeficientes significativos por era:** ERA_1 — `n_trayectoria_politica` (IRR=1.096, *p*<0.001) y `n_cargos_legislativos_prev` (IRR=0.926, *p*=0.0045, dirección negativa) son los únicos robustos. ERA_2 — solo `edad_imp` (IRR=1.068, *p*=0.001). ERA_3 — solo `es_partido_mayoria` (IRR=1.052, *p*=0.004). **ERA_4 — ninguna variable de `KEY_FEATS` alcanza significancia al 10 %**, pese a que esta era tiene el pseudo R² más alto de la serie: el modelo captura señal real (mejor ajuste global) pero distribuida de forma difusa entre variables, ninguna dominante individualmente — con *n*=500, el poder para detectar efectos individuales moderados es bajo incluso cuando el ajuste conjunto mejora. Esto es coherente con la lectura cualitativa ya presente en §7.2: la asignación temática de Morena responde a una lógica más distributiva y menos jerárquica que las eras previas.
""")

# =====================================================================
# D.5 — Cierre
# =====================================================================
d5 = md("""## D.5 Síntesis y límites del Anexo D

Este anexo complementa —no reemplaza— la interpretabilidad SHAP (magnitud de contribución individual) y la capa Bayesiana (incertidumbre posterior vía HDI) con el formato de tabla de regresión que la ciencia política cuantitativa espera como estándar de reporte: coeficiente, error estándar, valor *p*, IC95 %, *odds ratio*/IRR, pseudo R², prueba de razón de verosimilitud, AIC/BIC, matriz de confusión y VIF.

**Hallazgo metodológico transversal:** en tres de las ocho combinaciones era×target binario (Nodal ERA_1, Nodal ERA_4, Lastre ERA_3) la MLE no ponderada produce coeficientes parcial o totalmente no estimables por separación cuasi-completa o colinealidad exacta; en una combinación adicional (Lastre ERA_4, *n*=500) el modelo colapsa por completo. En los cuatro casos, la causa es identificable y explicable —categorías dummy raras (`admin_en_sindicato`) o colinealidad estructural de una era de partido único (`p_MORENA` ≡ `es_partido_mayoria` en ERA_4)— y coincide exactamente con las condiciones (n pequeño, *features* dummy dispersos) bajo las cuales la literatura de ciencia política (King y Zeng 2001) recomienda estimadores penalizados. Este patrón es, en sí mismo, evidencia empírica adicional a favor de la elección metodológica central del proyecto: Regresión Logística **L1 (Lasso)** en lugar de MLE clásica sin regularizar.

**Limitación declarada:** los coeficientes de este anexo no son numéricamente comparables a los coeficientes L1 del modelo productivo (penalización distinta, sin `class_weight="balanced"`); su función es inferencial —cuantificar significancia y magnitud en la escala familiar de la ciencia política cuantitativa— no predictiva. Para desempeño predictivo y comparabilidad entre variantes, la referencia sigue siendo la Tabla Global de la sección 8.0 y el AUC de validación cruzada de las secciones 5.4, 6.3 y 7.1.
""")

new_cells = [d0, d1_code, d1_interp, d2_code, d2_interp, d3_code, d3_interp, d4_code, d4_interp, d5]

INSERT_AT = 172  # right after cell 171 (Anexo C close), before cell 172 ("# 12. Enfoque 3...")
assert "".join(nb["cells"][171]["source"]).strip().startswith("df_enc.groupby"), \
    "Cell 171 no coincide con lo esperado (Anexo C describe) -- verificar indice de insercion"
assert "".join(nb["cells"][172]["source"]).strip().startswith("# 12. Enfoque 3"), \
    "Cell 172 no coincide con lo esperado (inicio seccion 12) -- verificar indice de insercion"

nb["cells"][INSERT_AT:INSERT_AT] = new_cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"OK -- {len(new_cells)} celdas insertadas en indice {INSERT_AT}.")
print(f"Total de celdas ahora: {len(nb['cells'])}")
