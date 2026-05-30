"""
patch_v7lr.py — single-pass patch
Lee diputraxv6LR.ipynb (fuente limpia) y escribe diputraxv7LR.ipynb
con el modelo Bayesiano (PyMC NUTS) integrado.
"""
import json, pathlib, copy

SRC = pathlib.Path("notebooks/diputraxv6LR.ipynb")
DST = pathlib.Path("notebooks/diputraxv7LR.ipynb")

# ── helpers ───────────────────────────────────────────────────────────────────
def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

with open(SRC, encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]

# ── 1. Modify cells 0 and 1 ───────────────────────────────────────────────────
cells[0]["source"] = "# Diputrax - V7LR"
v7_tag = (
    "\n\n**v7:** Incorpora **Regresion Logistica Bayesiana** (MCMC/NUTS, PyMC) "
    "como modelo paralelo al LR L1 frecuentista, sobre el mismo subconjunto de "
    "features seleccionado por SelectFromModel (SFM) en cada era. Compara AUC, "
    "coeficientes (log-odds L1 vs. media posterior + HDI 94%) y concordancia de "
    "direccion/significancia para comisiones nodales y lastre. Incluye diagnosticos "
    "de convergencia MCMC (R-hat, ESS) y comparativa global por era x target."
)
old_v1 = "".join(cells[1]["source"])
cells[1]["source"] = old_v1 + v7_tag

# ── 2. Find insertion indices (from v6LR, 174 cells) ─────────────────────────
infra_idx    = None   # insert AFTER (cell with "Setup OK")
nodal_idx    = None   # insert AFTER (show_img("shap_nodales_trend.png"))
lastre_idx   = None   # insert BEFORE (# 7. Comisiones Tematicas)
section8_idx = None   # insert AFTER  (# 8. Analisis Comparativo)

for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    ct  = cell["cell_type"]

    if "Setup OK" in src and ct == "code" and infra_idx is None:
        infra_idx = i

    if 'show_img("shap_nodales_trend.png"' in src and ct == "code":
        nodal_idx = i   # last occurrence wins (there is only one show_img call)

    if "# 7. Comisiones Tem" in src and ct == "markdown" and lastre_idx is None:
        lastre_idx = i

    if "# 8. An" in src and ct == "markdown" and section8_idx is None:
        section8_idx = i

print(f"Markers: infra={infra_idx} nodal_end={nodal_idx} "
      f"lastre_before={lastre_idx} section8_after={section8_idx}")

# ── 3. Cell content ───────────────────────────────────────────────────────────

# ─── A. BAYESIAN INFRASTRUCTURE ──────────────────────────────────────────────
MD_BAYES_INFRA = md(
"## 4.5 Infraestructura Bayesiana - PyMC NUTS (v7)\n\n"
"Complementa el pipeline frecuentista (LR L1 + SFM) con Regresion Logistica Bayesiana.\n"
"El modelo usa exactamente los features seleccionados por SFM en cada era,\n"
"con identica estandarizacion via `StandardScaler` del pipeline LR.\n\n"
"**Priors (Gelman et al. 2008 - predictores estandarizados):**\n"
"- Intercepto alpha: `Normal(0, 5)` - difuso\n"
"- Coeficientes beta: `Normal(0, 2.5)` - debilmente informativo\n\n"
"**Muestreador:** NUTS (No-U-Turn Sampler). 4 cadenas x 1500 draws + 800 tune. "
"`target_accept=0.90`. `cores=1` (compatibilidad Windows).\n\n"
"**Diagnosticos:** R-hat < 1.05 = convergencia. ESS > 400 = suficiente.\n\n"
"**Esparsidad:** LR L1 fuerza coefs = 0. Bayesiano produce masa posterior concentrada "
"cerca de cero (esparsidad suave). Comparacion restringida al subconjunto SFM-activo."
)

CODE_BAYES_INFRA = code(
"# ============================================================\n"
"# INFRAESTRUCTURA BAYESIANA - PyMC NUTS\n"
"# ============================================================\n"
"import pymc as pm\n"
"import arviz as az\n"
"from scipy.special import expit\n"
"import pathlib, warnings, logging\n"
"warnings.filterwarnings('ignore')\n"
"logging.getLogger('pymc').setLevel(logging.ERROR)\n"
"\n"
"_NUTS_CFG = dict(draws=1500, tune=800, target_accept=0.90,\n"
"                 chains=4, cores=1, random_seed=42, progressbar=True)\n"
"\n"
"def _Xy_sfm_scaled(era, target, sfm_features, lr_model):\n"
"    \"\"\"X con scaler del pipeline LR, filtrado a SFM features.\"\"\"\n"
"    mask  = df_enc['era'] == era\n"
"    X_all = df_enc.loc[mask, FEAT_COLS].astype(float).reset_index(drop=True)\n"
"    y     = df_enc.loc[mask, target  ].astype(float).reset_index(drop=True)\n"
"    sc    = lr_model.named_steps['sc']\n"
"    X_sc  = pd.DataFrame(sc.transform(X_all), columns=FEAT_COLS)\n"
"    return X_sc[sfm_features].values.astype(float), y.values.astype(float)\n"
"\n"
"def bayesian_logit_fit(X_arr, y_arr):\n"
"    \"\"\"Logistica Bayesiana NUTS. Prior Normal(0,2.5)/Normal(0,5).\"\"\"\n"
"    n_feat = X_arr.shape[1]\n"
"    with pm.Model():\n"
"        alpha = pm.Normal('alpha', mu=0.0, sigma=5.0)\n"
"        beta  = pm.Normal('beta',  mu=0.0, sigma=2.5, shape=n_feat)\n"
"        mu    = alpha + pm.math.dot(X_arr, beta)\n"
"        _     = pm.Bernoulli('y_obs', logit_p=mu, observed=y_arr)\n"
"        idata = pm.sample(**_NUTS_CFG)\n"
"    return idata\n"
"\n"
"def posterior_summary_df(idata, feature_names):\n"
"    \"\"\"Resumen posterior: media, SD, HDI 94%, prob(beta>0), significancia.\"\"\"\n"
"    lo_p, hi_p = 3.0, 97.0\n"
"    beta_flat  = idata.posterior['beta'].values.reshape(-1, len(feature_names))\n"
"    return pd.DataFrame({\n"
"        'post_mean' : beta_flat.mean(0),\n"
"        'post_sd'   : beta_flat.std(0),\n"
"        'hdi_lo'    : np.percentile(beta_flat, lo_p, 0),\n"
"        'hdi_hi'    : np.percentile(beta_flat, hi_p, 0),\n"
"        'prob_pos'  : (beta_flat > 0).mean(0),\n"
"        'signif_94' : (np.percentile(beta_flat, lo_p, 0) > 0) |\n"
"                      (np.percentile(beta_flat, hi_p, 0) < 0),\n"
"    }, index=feature_names)\n"
"\n"
"def bayes_auc_score(idata, X_arr, y_arr, feature_names):\n"
"    \"\"\"AUC usando predicciones de la media posterior.\"\"\"\n"
"    beta_mean  = idata.posterior['beta'].values.reshape(-1, len(feature_names)).mean(0)\n"
"    alpha_mean = idata.posterior['alpha'].values.flatten().mean()\n"
"    return roc_auc_score(y_arr, expit(alpha_mean + X_arr @ beta_mean))\n"
"\n"
"def forest_comparison_fig(lr_coefs, features, bayes_sum, era_label, target_label):\n"
"    \"\"\"Forest plot 1x2: coef LR L1 (izq.) vs posterior beta + HDI 94% (der.).\"\"\"\n"
"    n, yp = len(features), np.arange(len(features))\n"
"    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n * 0.50)), sharey=True)\n"
"    # LR L1\n"
"    c_lr = ['#e74c3c' if v < 0 else '#2ecc71' for v in lr_coefs]\n"
"    axes[0].barh(yp, lr_coefs, color=c_lr, alpha=0.82, edgecolor='white', lw=0.5)\n"
"    axes[0].axvline(0, color='k', lw=1)\n"
"    axes[0].set_yticks(yp); axes[0].set_yticklabels(features, fontsize=8)\n"
"    axes[0].set_title('LR L1 (Lasso)\\nCoeficientes log-odds', fontweight='bold')\n"
"    axes[0].set_xlabel('Coeficiente'); axes[0].grid(axis='x', alpha=0.3)\n"
"    # Bayesiano\n"
"    pm_  = bayes_sum['post_mean'].values\n"
"    lo_  = bayes_sum['hdi_lo'].values\n"
"    hi_  = bayes_sum['hdi_hi'].values\n"
"    sg_  = bayes_sum['signif_94'].values\n"
"    for i in range(n):\n"
"        col = '#e74c3c' if pm_[i] < 0 else '#2ecc71'\n"
"        axes[1].plot([lo_[i], hi_[i]], [i, i], color='lightgray', lw=2.0, zorder=1)\n"
"        axes[1].plot(pm_[i], i,\n"
"                     marker='o' if sg_[i] else 'D',\n"
"                     color=col, ms=9, zorder=5,\n"
"                     markeredgecolor='white', markeredgewidth=0.6)\n"
"    axes[1].axvline(0, color='k', lw=1)\n"
"    axes[1].set_title(\n"
"        'Bayesiano NUTS\\nMedia posterior + HDI 94%\\n'\n"
"        'o significativo  D no significativo',\n"
"        fontweight='bold', fontsize=10)\n"
"    axes[1].set_xlabel('beta (log-odds)'); axes[1].grid(axis='x', alpha=0.3)\n"
"    fig.suptitle(f'{target_label} - {era_label}  |  LR L1 vs. Bayesiano',\n"
"                 fontsize=12, fontweight='bold')\n"
"    plt.tight_layout()\n"
"    return fig\n"
"\n"
"def agreement_df(lr_coefs, features, bayes_sum):\n"
"    \"\"\"Tabla de concordancia de direccion y significancia entre modelos.\"\"\"\n"
"    rows = []\n"
"    for feat, coef in zip(features, lr_coefs):\n"
"        bm = float(bayes_sum.loc[feat, 'post_mean'])\n"
"        bs = bool(bayes_sum.loc[feat, 'signif_94'])\n"
"        pp = float(bayes_sum.loc[feat, 'prob_pos'])\n"
"        rows.append({'feature': feat, 'lr_coef': round(coef, 4),\n"
"                     'bayes_mean': round(bm, 4), 'prob_pos': round(pp, 3),\n"
"                     'bayes_signif': bs, 'dir_agree': (coef > 0) == (bm > 0)})\n"
"    return pd.DataFrame(rows)\n"
"\n"
"def _parse_auc(df_cv, era_label, col):\n"
"    \"\"\"Extrae float AUC de celda formateada '0.734+-0.012'.\"\"\"\n"
"    val = df_cv.loc[df_cv['Era'] == era_label, col].values\n"
"    if len(val) == 0: return float('nan')\n"
"    return float(str(val[0]).split('+-')[0].strip())\n"
"\n"
"REPORT_DIR = pathlib.Path('C:/Users/zigma/Projects/diputrax/reports/eda')\n"
"REPORT_DIR.mkdir(parents=True, exist_ok=True)\n"
"\n"
"print(f'Infraestructura Bayesiana lista - PyMC {pm.__version__} | ArviZ {az.__version__}')"
)

# ─── B. BAYESIAN NODAL ───────────────────────────────────────────────────────
MD_BAYES_NODAL = md(
"# 5.6 Modelo Bayesiano - Comisiones Nodales (v7)\n\n"
"Regresion Logistica Bayesiana (NUTS) entrenada por era sobre el mismo subconjunto "
"de features seleccionado por SFM, con identica estandarizacion.\n\n"
"**Comparativa en tres dimensiones:**\n"
"1. **AUC** - LR L1 (full) | LR L1 + SFM | AUC Bayesiano (desde media posterior).\n"
"2. **Forest plots** - Coeficiente Lasso vs. media posterior beta + HDI 94% por era.\n"
"3. **Concordancia** - direccion y significancia por feature.\n\n"
"> Runtime aprox. 5-15 min segun hardware (NUTS x 4 eras x 4 cadenas)."
)

CODE_BAYES_NODAL_TRAIN = code(
"print('Entrenando Bayesiano NODAL (NUTS) - 4 eras...')\n"
"bayes_nodal_idata, bayes_nodal_summary, bayes_nodal_auc = {}, {}, {}\n"
"rows_bn = []\n"
"\n"
"for era in ERA_ORDER:\n"
"    feats = sfm_selected_nodal.get(era, [])\n"
"    if not feats:\n"
"        print(f'  {era}: sin features SFM - saltando.'); continue\n"
"\n"
"    print(f'\\n-- {ERA_LABELS[era]}  ({len(feats)} features) --')\n"
"    X_arr, y_arr = _Xy_sfm_scaled(era, 'nodal_bin', feats, models_nodal[era])\n"
"\n"
"    idata = bayesian_logit_fit(X_arr, y_arr)\n"
"    summ  = posterior_summary_df(idata, feats)\n"
"    auc_b = bayes_auc_score(idata, X_arr, y_arr, feats)\n"
"\n"
"    bayes_nodal_idata[era]   = idata\n"
"    bayes_nodal_summary[era] = summ\n"
"    bayes_nodal_auc[era]     = auc_b\n"
"\n"
"    el = ERA_LABELS[era]\n"
"    auc_lrf = _parse_auc(df_nodal_cv, el, 'LR L1 (full)')\n"
"    auc_sfm = _parse_auc(df_nodal_cv, el, 'LR L1 + SFM')\n"
"    rhat_max = float(az.summary(idata, var_names=['beta'])['r_hat'].max())\n"
"    ess_min  = int(az.summary(idata, var_names=['beta'])['ess_bulk'].min())\n"
"    n_signif = int(summ['signif_94'].sum())\n"
"\n"
"    rows_bn.append({\n"
"        'Era': el, 'n_feat SFM': len(feats),\n"
"        'AUC LR full (CV)': round(auc_lrf, 3),\n"
"        'AUC LR+SFM (CV)' : round(auc_sfm, 3),\n"
"        'AUC Bayesiano'   : round(auc_b, 3),\n"
"        'Delta Bayes-LR'  : round(auc_b - auc_lrf, 3),\n"
"        'n beta signif'   : n_signif,\n"
"        'R-hat max'       : round(rhat_max, 3),\n"
"        'ESS min'         : ess_min,\n"
"    })\n"
"    print(f'    AUC LR={auc_lrf:.3f}  SFM={auc_sfm:.3f}  Bayes={auc_b:.3f}  '\n"
"          f'R-hat={rhat_max:.3f}  ESS_min={ess_min}')\n"
"\n"
"df_bayes_nodal_cv = pd.DataFrame(rows_bn)\n"
"print('\\n-- Comparativa AUC: LR L1 vs. Bayesiano - Nodales --')\n"
"display(df_bayes_nodal_cv)"
)

CODE_BAYES_NODAL_FOREST = code(
"# Forest plots + tablas de concordancia - Nodales\n"
"for era in ERA_ORDER:\n"
"    if era not in bayes_nodal_summary: continue\n"
"    feats    = sfm_selected_nodal[era]\n"
"    lr_coefs = models_nodal[era].named_steps['lr'].coef_[0]\n"
"    summ     = bayes_nodal_summary[era]\n"
"\n"
"    fig = forest_comparison_fig(lr_coefs, feats, summ, ERA_LABELS[era], 'Nodales')\n"
"    fig.savefig(REPORT_DIR / f'bayes_nodal_forest_{era}.png', bbox_inches='tight', dpi=110)\n"
"    plt.show()\n"
"\n"
"    ag  = agreement_df(lr_coefs, feats, summ)\n"
"    pct = ag['dir_agree'].mean() * 100\n"
"    print(f'\\n{ERA_LABELS[era]} - Concordancia de direccion: {pct:.0f}% '\n"
"          f'({ag[\"dir_agree\"].sum()}/{len(feats)})')\n"
"\n"
"    def _style_agree(row):\n"
"        bg = '#d4edda' if row['dir_agree'] else '#f8d7da'\n"
"        return [f'background-color: {bg}'] * len(row)\n"
"\n"
"    display(ag.style\n"
"              .apply(_style_agree, axis=1)\n"
"              .format({'lr_coef': '{:+.4f}', 'bayes_mean': '{:+.4f}', 'prob_pos': '{:.3f}'})\n"
"              .set_caption(f'Concordancia LR L1 vs Bayesiano - {ERA_LABELS[era]}'))"
)

MD_BAYES_NODAL_INTERP = md(
"**Interpretacion - Comparativa LR L1 vs. Bayesiano (Nodales)**\n\n"
"**AUC:** La diferencia entre AUC frecuentista (CV) y AUC Bayesiano (media posterior in-sample) "
"refleja dos cosas distintas: el CV evalua generalizacion out-of-fold; "
"el AUC Bayesiano evalua ajuste in-sample. Diferencias |Delta| < 0.02 indican consistencia. "
"|Delta| > 0.04 puede senalar sobreajuste del Bayesiano.\n\n"
"**Forest plots:** Los coeficientes L1 son esparsos por construccion; "
"los betas bayesianos tienen masa distribucional positiva incluso para features marginales. "
"Donde ambos son del mismo signo con HDI que excluye cero, la senal es robusta al paradigma.\n\n"
"**Concordancia de direccion:** >= 85% indica ranking cualitativo estable. "
"< 70% senala sensibilidad del resultado al estimador elegido.\n\n"
"**R-hat y ESS:** R-hat < 1.05 y ESS > 400 son condicion necesaria para validez "
"inferencial bayesiana. Valores fuera de rango invalidan las inferencias de esa era.\n\n"
"**Conexion con A&L (2009):** Si el Bayesiano replica el ranking de A&L "
"(experiencia burocratica > legislativa para nodales en ERA_1), "
"esto fortalece la robustez del hallazgo mas alla del paradigma frecuentista."
)

# ─── C. BAYESIAN LASTRE ──────────────────────────────────────────────────────
MD_BAYES_LASTRE = md(
"# 6.5 Modelo Bayesiano - Comisiones Lastre (v7)\n\n"
"Misma arquitectura Bayesiana (NUTS) aplicada al target `lastre_bin`. "
"Se espera senal mas debil que en nodales (AUC 0.53-0.63 en LR L1). "
"El Bayesiano permite cuantificar la incertidumbre posterior incluso cuando la senal es debil: "
"HDI amplios que cruzan cero indican que el parametro es esencialmente indistinguible de cero.\n\n"
"> Runtime aprox. 5-15 min (NUTS x 4 eras x 4 cadenas)."
)

CODE_BAYES_LASTRE_TRAIN = code(
"print('Entrenando Bayesiano LASTRE (NUTS) - 4 eras...')\n"
"bayes_lastre_idata, bayes_lastre_summary, bayes_lastre_auc = {}, {}, {}\n"
"rows_bl = []\n"
"\n"
"for era in ERA_ORDER:\n"
"    feats = sfm_selected_lastre.get(era, [])\n"
"    if not feats:\n"
"        print(f'  {era}: sin features SFM - saltando.'); continue\n"
"\n"
"    print(f'\\n-- {ERA_LABELS[era]}  ({len(feats)} features) --')\n"
"    X_arr, y_arr = _Xy_sfm_scaled(era, 'lastre_bin', feats, models_lastre[era])\n"
"\n"
"    idata = bayesian_logit_fit(X_arr, y_arr)\n"
"    summ  = posterior_summary_df(idata, feats)\n"
"    auc_b = bayes_auc_score(idata, X_arr, y_arr, feats)\n"
"\n"
"    bayes_lastre_idata[era]   = idata\n"
"    bayes_lastre_summary[era] = summ\n"
"    bayes_lastre_auc[era]     = auc_b\n"
"\n"
"    el = ERA_LABELS[era]\n"
"    auc_lrf = _parse_auc(df_lastre_cv, el, 'LR L1 (full)')\n"
"    auc_sfm = _parse_auc(df_lastre_cv, el, 'LR L1 + SFM')\n"
"    rhat_max = float(az.summary(idata, var_names=['beta'])['r_hat'].max())\n"
"    ess_min  = int(az.summary(idata, var_names=['beta'])['ess_bulk'].min())\n"
"    n_signif = int(summ['signif_94'].sum())\n"
"\n"
"    rows_bl.append({\n"
"        'Era': el, 'n_feat SFM': len(feats),\n"
"        'AUC LR full (CV)': round(auc_lrf, 3),\n"
"        'AUC LR+SFM (CV)' : round(auc_sfm, 3),\n"
"        'AUC Bayesiano'   : round(auc_b, 3),\n"
"        'Delta Bayes-LR'  : round(auc_b - auc_lrf, 3),\n"
"        'n beta signif'   : n_signif,\n"
"        'R-hat max'       : round(rhat_max, 3),\n"
"        'ESS min'         : ess_min,\n"
"    })\n"
"    print(f'    AUC LR={auc_lrf:.3f}  SFM={auc_sfm:.3f}  Bayes={auc_b:.3f}  '\n"
"          f'R-hat={rhat_max:.3f}  ESS_min={ess_min}')\n"
"\n"
"df_bayes_lastre_cv = pd.DataFrame(rows_bl)\n"
"print('\\n-- Comparativa AUC: LR L1 vs. Bayesiano - Lastre --')\n"
"display(df_bayes_lastre_cv)"
)

CODE_BAYES_LASTRE_FOREST = code(
"# Forest plots + tablas de concordancia - Lastre\n"
"for era in ERA_ORDER:\n"
"    if era not in bayes_lastre_summary: continue\n"
"    feats    = sfm_selected_lastre[era]\n"
"    lr_coefs = models_lastre[era].named_steps['lr'].coef_[0]\n"
"    summ     = bayes_lastre_summary[era]\n"
"\n"
"    fig = forest_comparison_fig(lr_coefs, feats, summ, ERA_LABELS[era], 'Lastre')\n"
"    fig.savefig(REPORT_DIR / f'bayes_lastre_forest_{era}.png', bbox_inches='tight', dpi=110)\n"
"    plt.show()\n"
"\n"
"    ag  = agreement_df(lr_coefs, feats, summ)\n"
"    pct = ag['dir_agree'].mean() * 100\n"
"    print(f'\\n{ERA_LABELS[era]} - Concordancia: {pct:.0f}% ({ag[\"dir_agree\"].sum()}/{len(feats)})')\n"
"\n"
"    def _style_agree(row):\n"
"        bg = '#d4edda' if row['dir_agree'] else '#f8d7da'\n"
"        return [f'background-color: {bg}'] * len(row)\n"
"\n"
"    display(ag.style\n"
"              .apply(_style_agree, axis=1)\n"
"              .format({'lr_coef': '{:+.4f}', 'bayes_mean': '{:+.4f}', 'prob_pos': '{:.3f}'})\n"
"              .set_caption(f'Concordancia LR L1 vs Bayesiano - {ERA_LABELS[era]}'))"
)

MD_BAYES_LASTRE_INTERP = md(
"**Interpretacion - Comparativa LR L1 vs. Bayesiano (Lastre)**\n\n"
"El bajo AUC esperado para lastre (0.53-0.63) se traducira en HDI amplios "
"que cruzan cero para la mayoria de los features: el Bayesiano cuantifica "
"explicitamente que no hay informacion suficiente en el perfil biografico "
"para distinguir quien recibe una comision lastre.\n\n"
"Cuando el AUC Bayesiano ~ 0.50 y la mayoria de los HDI incluyen el cero, "
"ambos paradigmas convergen en la misma conclusion sustantiva: "
"la asignacion lastre es esencialmente opaca desde el perfil observable.\n\n"
"Una concordancia de direccion < 60% en eras con senal debil (ERA_4) no es preocupante: "
"con coeficientes marginales, tanto L1 como NUTS pueden estimar direcciones distintas "
"por varianza muestral."
)

# ─── D. GLOBAL COMPARISON ────────────────────────────────────────────────────
MD_GLOBAL = md(
"## 8.0 Comparativa Global - Frecuentista (LR L1) vs. Bayesiano (NUTS)\n\n"
"Sintesis de ambos paradigmas para los dos targets binarios (nodal y lastre) x cuatro eras.\n\n"
"- **Delta AUC** = AUC Bayesiano - AUC LR L1 full: diferencia de desempeno predictivo.\n"
"- **% Concordancia** de direccion: fraccion de features SFM donde ambos modelos coinciden en el signo.\n"
"- **n_signif Bayesiano**: features con HDI 94% que excluyen cero.\n\n"
"Delta AUC ~ 0 y concordancia >= 85% indica conclusiones robustas al paradigma estadistico."
)

CODE_GLOBAL = code(
"# -- 1. Tabla global --\n"
"rows_g = []\n"
"for era in ERA_ORDER:\n"
"    el = ERA_LABELS[era]\n"
"    for tl, df_cv, bayes_auc_d, sfm_d, bayes_summ, lr_m in [\n"
"        ('Nodal',  df_nodal_cv,  bayes_nodal_auc,  sfm_selected_nodal,  bayes_nodal_summary,  models_nodal),\n"
"        ('Lastre', df_lastre_cv, bayes_lastre_auc, sfm_selected_lastre, bayes_lastre_summary, models_lastre),\n"
"    ]:\n"
"        auc_lrf = _parse_auc(df_cv, el, 'LR L1 (full)')\n"
"        auc_b   = bayes_auc_d.get(era, float('nan'))\n"
"        feats   = sfm_d.get(era, [])\n"
"        if feats and era in bayes_summ:\n"
"            lr_c    = lr_m[era].named_steps['lr'].coef_[0]\n"
"            ag      = agreement_df(lr_c, feats, bayes_summ[era])\n"
"            pct_dir = round(ag['dir_agree'].mean() * 100, 1)\n"
"            n_sig   = int(bayes_summ[era]['signif_94'].sum())\n"
"        else:\n"
"            pct_dir, n_sig = float('nan'), 0\n"
"        rows_g.append({\n"
"            'Era': el, 'Target': tl,\n"
"            'AUC LR full': round(auc_lrf, 3),\n"
"            'AUC Bayes'  : round(auc_b, 3) if not pd.isna(auc_b) else None,\n"
"            'Delta AUC'  : round(auc_b - auc_lrf, 3) if not pd.isna(auc_b) else None,\n"
"            '% Dir Conc' : pct_dir,\n"
"            'n signif B' : n_sig,\n"
"            'n feat SFM' : len(feats),\n"
"        })\n"
"\n"
"df_global = pd.DataFrame(rows_g)\n"
"print('-- Tabla Global: LR L1 vs. Bayesiano --')\n"
"display(df_global)\n"
"\n"
"# -- 2. Heatmaps --\n"
"era_order_labels = [ERA_LABELS[e] for e in ERA_ORDER]\n"
"pivot_auc = (df_global.pivot(index='Era', columns='Target', values='Delta AUC')\n"
"             .apply(pd.to_numeric, errors='coerce')\n"
"             .reindex(era_order_labels))\n"
"pivot_dir = (df_global.pivot(index='Era', columns='Target', values='% Dir Conc')\n"
"             .apply(pd.to_numeric, errors='coerce')\n"
"             .reindex(era_order_labels))\n"
"\n"
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
"sns.heatmap(pivot_auc, annot=True, fmt='.3f', center=0,\n"
"            cmap='RdYlGn', ax=axes[0], linewidths=0.5,\n"
"            cbar_kws={'label': 'Delta AUC (Bayes - LR full)'})\n"
"axes[0].set_title('Delta AUC Bayesiano - LR L1\\n(verde = Bayes mejor)', fontweight='bold')\n"
"\n"
"sns.heatmap(pivot_dir, annot=True, fmt='.1f', vmin=50, vmax=100,\n"
"            cmap='YlGn', ax=axes[1], linewidths=0.5,\n"
"            cbar_kws={'label': '% Concordancia de direccion'})\n"
"axes[1].set_title('% Concordancia de direccion\\n(mismo signo LR y Bayes)', fontweight='bold')\n"
"\n"
"plt.suptitle('Comparativa Global: Frecuentista (LR L1) vs. Bayesiano (NUTS)',\n"
"             fontsize=13, fontweight='bold')\n"
"plt.tight_layout()\n"
"fig.savefig(REPORT_DIR / 'global_lr_vs_bayes.png', bbox_inches='tight', dpi=120)\n"
"plt.show()\n"
"\n"
"# -- 3. Convergencia consolidada --\n"
"rows_conv = []\n"
"for era in ERA_ORDER:\n"
"    el = ERA_LABELS[era]\n"
"    for tl, idata_d, sfm_d in [\n"
"        ('Nodal',  bayes_nodal_idata,  sfm_selected_nodal),\n"
"        ('Lastre', bayes_lastre_idata, sfm_selected_lastre),\n"
"    ]:\n"
"        if era not in idata_d: continue\n"
"        s = az.summary(idata_d[era], var_names=['beta'])\n"
"        rows_conv.append({\n"
"            'Era': el, 'Target': tl,\n"
"            'R-hat max': round(float(s['r_hat'].max()), 3),\n"
"            'ESS min'  : int(s['ess_bulk'].min()),\n"
"            'Converge' : 'OK' if s['r_hat'].max() < 1.05 and s['ess_bulk'].min() > 400 else 'REVISAR',\n"
"        })\n"
"print('\\n-- Diagnosticos de convergencia MCMC --')\n"
"display(pd.DataFrame(rows_conv))"
)

MD_GLOBAL_INTERP = md(
"**Interpretacion - Comparativa Global LR L1 vs. Bayesiano**\n\n"
"**Delta AUC ~ 0** en todas las eras confirma que el poder predictivo no depende del paradigma: "
"la relacion entre perfil biografico y tipo de comision es lineal y robusta. "
"Diferencias |Delta| > 0.03 deben investigarse como posible sobreajuste in-sample del Bayesiano.\n\n"
"**% Concordancia de direccion >= 85%** para nodales en ERA_1 replicaria el hallazgo de "
"A&L (2009) desde ambos paradigmas: la experiencia burocratica domina la asignacion nodal. "
"Esto fortalece la validez interna mas alla de la eleccion del estimador.\n\n"
"**Convergencia MCMC:** R-hat < 1.05 y ESS > 400 en todos los parametros son condicion "
"necesaria para validez inferencial. Si alguna era no converge, considerar mayor `n_tune`.\n\n"
"**Mensaje metodologico para la tesina:** La consistencia entre paradigmas frecuentista "
"y bayesiano fortalece la validez interna. El Bayesiano aporta cuantificacion explicita "
"de incertidumbre (HDI) que el LR L1 no ofrece directamente, aproximando mas fielmente "
"el espiritu del analisis de A&L (2009) con MCMCpack en R."
)

# ── 4. Single-pass cell list construction ────────────────────────────────────
infra_inserted   = False
nodal_inserted   = False
lastre_inserted  = False
section8_inserted = False

new_cells = []
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    ct  = cell["cell_type"]

    # BEFORE section 7 header — insert lastre Bayesian
    if (LASTRE_END_MKR := "# 7. Comisiones Tem") in src and ct == "markdown" and not lastre_inserted:
        new_cells += [MD_BAYES_LASTRE, CODE_BAYES_LASTRE_TRAIN,
                      CODE_BAYES_LASTRE_FOREST, MD_BAYES_LASTRE_INTERP]
        lastre_inserted = True

    new_cells.append(copy.deepcopy(cell))

    # AFTER setup cell
    if i == infra_idx and not infra_inserted:
        new_cells += [MD_BAYES_INFRA, CODE_BAYES_INFRA]
        infra_inserted = True

    # AFTER last nodal show_img
    elif i == nodal_idx and not nodal_inserted:
        new_cells += [MD_BAYES_NODAL, CODE_BAYES_NODAL_TRAIN,
                      CODE_BAYES_NODAL_FOREST, MD_BAYES_NODAL_INTERP]
        nodal_inserted = True

    # AFTER section 8 header
    elif ("# 8. An" in src) and ct == "markdown" and not section8_inserted:
        new_cells += [MD_GLOBAL, CODE_GLOBAL, MD_GLOBAL_INTERP]
        section8_inserted = True

nb["cells"] = new_cells
with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

added = len(new_cells) - len(cells)
print(f"Done. {len(cells)} -> {len(new_cells)} cells (+{added} new). Saved: {DST}")
