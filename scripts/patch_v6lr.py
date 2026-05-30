import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def set_code(cell, src):
    cell['source'] = src
    cell['outputs'] = []
    cell['execution_count'] = None

def set_md(cell, src):
    cell['source'] = src

# Cell 0: title
set_md(cells[0], ["# Diputrax — V6LR\n"])

# Cell 88: imports
set_code(cells[88], [
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "import io\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression, PoissonRegressor\n",
    "from sklearn.feature_selection import SelectFromModel\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.experimental import enable_iterative_imputer  # noqa: F401\n",
    "from sklearn.impute import IterativeImputer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score\n",
    "from sklearn.metrics import roc_auc_score, mean_absolute_error\n",
    "import shap\n",
    "\n",
    "pd.set_option(\"display.float_format\", \"{:.3f}\".format)\n",
    "plt.rcParams.update({\"figure.dpi\": 110, \"font.size\": 11})\n",
    "print(\"All imports OK\")\n",
    "print(f\"  shap {shap.__version__}\")\n",
])

# Cell 97: model factories
set_code(cells[97], [
    "def get_Xy(era, target):\n",
    "    mask = df_enc[\"era\"] == era\n",
    "    X = df_enc.loc[mask, FEAT_COLS].astype(float).reset_index(drop=True)\n",
    "    y = df_enc.loc[mask, target].astype(float).reset_index(drop=True)\n",
    "    return X, y\n",
    "\n",
    "# Model factories\n",
    "_L1_PARAMS = dict(penalty='l1', solver='liblinear', C=0.1, max_iter=3000,\n",
    "                  class_weight='balanced', random_state=42)\n",
    "\n",
    "def lr_binary():\n",
    "    \"\"\"L1-penalised LR with StandardScaler (full feature set).\"\"\"\n",
    "    return Pipeline([\n",
    "        (\"sc\", StandardScaler()),\n",
    "        (\"lr\", LogisticRegression(**_L1_PARAMS)),\n",
    "    ])\n",
    "\n",
    "def lr_l1_sfm():\n",
    "    \"\"\"Pipeline: scale -> SelectFromModel(L1 LR) -> L1 LR.\"\"\"\n",
    "    return Pipeline([\n",
    "        (\"sc\",  StandardScaler()),\n",
    "        (\"sfm\", SelectFromModel(\n",
    "            LogisticRegression(**_L1_PARAMS), threshold=\"mean\"\n",
    "        )),\n",
    "        (\"lr\",  LogisticRegression(**_L1_PARAMS)),\n",
    "    ])\n",
    "\n",
    "def lr_poisson():\n",
    "    return Pipeline([\n",
    "        (\"sc\", StandardScaler()),\n",
    "        (\"pr\", PoissonRegressor(alpha=1.0, max_iter=3000)),\n",
    "    ])\n",
    "\n",
    "def sfm_report(X, y):\n",
    "    \"\"\"Fit SFM on full era data; return list of selected feature names.\"\"\"\n",
    "    sc = StandardScaler()\n",
    "    X_sc = sc.fit_transform(X)\n",
    "    sel = SelectFromModel(LogisticRegression(**_L1_PARAMS), threshold=\"mean\")\n",
    "    sel.fit(X_sc, y)\n",
    "    return [FEAT_COLS[i] for i, s in enumerate(sel.get_support()) if s]\n",
    "\n",
    "# CV helpers\n",
    "def cv_auc(model, X, y, k=5):\n",
    "    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)\n",
    "    s  = cross_val_score(model, X, y, cv=cv, scoring=\"roc_auc\")\n",
    "    return s.mean(), s.std()\n",
    "\n",
    "def cv_mae(model, X, y, k=5):\n",
    "    cv = KFold(n_splits=k, shuffle=True, random_state=42)\n",
    "    s  = -cross_val_score(model, X, y, cv=cv, scoring=\"neg_mean_absolute_error\")\n",
    "    return s.mean(), s.std()\n",
    "\n",
    "# SHAP helper\n",
    "def shap_beeswarm_to_img(sv, X, top_n=15, title=\"\", color=\"#333333\"):\n",
    "    top_idx = np.argsort(np.abs(sv).mean(axis=0))[-top_n:][::-1]\n",
    "    fig, ax = plt.subplots(figsize=(8, 6))\n",
    "    shap.summary_plot(sv[:, top_idx], X.iloc[:, top_idx],\n",
    "                      plot_type=\"dot\", show=False, max_display=top_n,\n",
    "                      color_bar=True)\n",
    "    ax = plt.gca()\n",
    "    ax.set_title(title, fontsize=11, fontweight=\"bold\", color=color)\n",
    "    ax.set_xlabel(\"SHAP value\")\n",
    "    buf = io.BytesIO()\n",
    "    plt.savefig(buf, format=\"png\", bbox_inches=\"tight\", dpi=100)\n",
    "    plt.close()\n",
    "    buf.seek(0)\n",
    "    return mpimg.imread(buf)\n",
    "\n",
    "print(\"Helpers definidos.\")\n",
])

# Cell 100: Nodal training
set_code(cells[100], [
    "print(\"Entrenando modelos para NODALES (LR L1 + SelectFromModel)...\")\n",
    "rows_nodal = []\n",
    "sfm_selected_nodal = {}\n",
    "\n",
    "for era in ERA_ORDER:\n",
    "    X, y = get_Xy(era, \"nodal_bin\")\n",
    "\n",
    "    lr_mu,  lr_sd  = cv_auc(lr_binary(),  X, y)\n",
    "    sfm_mu, sfm_sd = cv_auc(lr_l1_sfm(),  X, y)\n",
    "\n",
    "    selected = sfm_report(X, y)\n",
    "    sfm_selected_nodal[era] = selected\n",
    "\n",
    "    rows_nodal.append({\n",
    "        \"Era\": ERA_LABELS[era], \"n\": len(y), \"pos%\": f\"{y.mean():.1%}\",\n",
    "        \"LR L1 (full)\": f\"{lr_mu:.3f}+-{lr_sd:.3f}\",\n",
    "        \"LR L1 + SFM\":  f\"{sfm_mu:.3f}+-{sfm_sd:.3f}\",\n",
    "        \"n_features\": len(selected),\n",
    "    })\n",
    "    print(f\"  {era:<18}  LR={lr_mu:.3f}  SFM={sfm_mu:.3f}  features={len(selected)}/{len(FEAT_COLS)}\")\n",
    "\n",
    "df_nodal_cv = pd.DataFrame(rows_nodal)\n",
    "df_nodal_cv\n",
])

# Cell 102: SHAP Nodal
set_code(cells[102], [
    "shap_nodal   = {}\n",
    "models_nodal = {}\n",
    "\n",
    "for era in ERA_ORDER:\n",
    "    X, y = get_Xy(era, \"nodal_bin\")\n",
    "\n",
    "    m = lr_l1_sfm()\n",
    "    m.fit(X, y)\n",
    "    models_nodal[era] = m\n",
    "\n",
    "    X_sc  = m.named_steps[\"sc\"].transform(X)\n",
    "    X_sel = m.named_steps[\"sfm\"].transform(X_sc)\n",
    "    mask  = m.named_steps[\"sfm\"].get_support()\n",
    "\n",
    "    explainer = shap.LinearExplainer(m.named_steps[\"lr\"], X_sel)\n",
    "    sv_sel    = explainer.shap_values(X_sel)\n",
    "\n",
    "    # Pad SHAP values back to full feature space for consistent downstream use\n",
    "    sv_full = np.zeros((len(X), len(FEAT_COLS)))\n",
    "    for j, col_idx in enumerate([i for i, s in enumerate(mask) if s]):\n",
    "        sv_full[:, col_idx] = sv_sel[:, j]\n",
    "\n",
    "    shap_nodal[era] = (sv_full, X)\n",
    "    print(f\"  {era}: SHAP OK  (selected={mask.sum()}/{len(mask)})\")\n",
])

# Cell 112: static nodal table -> dynamic
set_code(cells[112], [
    "display(df_nodal_cv)\n",
])

# Cell 118: Lastre training
set_code(cells[118], [
    "print(\"Entrenando modelos para LASTRE (LR L1 + SelectFromModel)...\")\n",
    "rows_lastre = []\n",
    "sfm_selected_lastre = {}\n",
    "\n",
    "for era in ERA_ORDER:\n",
    "    X, y = get_Xy(era, \"lastre_bin\")\n",
    "\n",
    "    lr_mu,  lr_sd  = cv_auc(lr_binary(),  X, y)\n",
    "    sfm_mu, sfm_sd = cv_auc(lr_l1_sfm(),  X, y)\n",
    "\n",
    "    selected = sfm_report(X, y)\n",
    "    sfm_selected_lastre[era] = selected\n",
    "\n",
    "    rows_lastre.append({\n",
    "        \"Era\": ERA_LABELS[era], \"n\": len(y), \"pos%\": f\"{y.mean():.1%}\",\n",
    "        \"LR L1 (full)\": f\"{lr_mu:.3f}+-{lr_sd:.3f}\",\n",
    "        \"LR L1 + SFM\":  f\"{sfm_mu:.3f}+-{sfm_sd:.3f}\",\n",
    "        \"n_features\": len(selected),\n",
    "    })\n",
    "    print(f\"  {era:<18}  LR={lr_mu:.3f}  SFM={sfm_mu:.3f}  features={len(selected)}/{len(FEAT_COLS)}\")\n",
    "\n",
    "df_lastre_cv = pd.DataFrame(rows_lastre)\n",
    "df_lastre_cv\n",
])

# Cell 119: SHAP Lastre
set_code(cells[119], [
    "shap_lastre   = {}\n",
    "models_lastre = {}\n",
    "\n",
    "for era in ERA_ORDER:\n",
    "    X, y = get_Xy(era, \"lastre_bin\")\n",
    "\n",
    "    m = lr_l1_sfm()\n",
    "    m.fit(X, y)\n",
    "    models_lastre[era] = m\n",
    "\n",
    "    X_sc  = m.named_steps[\"sc\"].transform(X)\n",
    "    X_sel = m.named_steps[\"sfm\"].transform(X_sc)\n",
    "    mask  = m.named_steps[\"sfm\"].get_support()\n",
    "\n",
    "    explainer = shap.LinearExplainer(m.named_steps[\"lr\"], X_sel)\n",
    "    sv_sel    = explainer.shap_values(X_sel)\n",
    "\n",
    "    sv_full = np.zeros((len(X), len(FEAT_COLS)))\n",
    "    for j, col_idx in enumerate([i for i, s in enumerate(mask) if s]):\n",
    "        sv_full[:, col_idx] = sv_sel[:, j]\n",
    "\n",
    "    shap_lastre[era] = (sv_full, X)\n",
    "    print(f\"  {era}: SHAP OK  (selected={mask.sum()}/{len(mask)})\")\n",
])

# Cell 127: static lastre table -> dynamic
set_code(cells[127], [
    "display(df_lastre_cv)\n",
])

# Cell 131: Tematicas training (GLM Poisson only)
set_code(cells[131], [
    "print(\"Entrenando modelos para TEMATICAS (GLM Poisson)...\")\n",
    "rows_tem = []\n",
    "\n",
    "for era in ERA_ORDER:\n",
    "    X, y = get_Xy(era, \"n_comisiones_tematicas\")\n",
    "    base_mae = float(np.abs(y - y.mean()).mean())\n",
    "\n",
    "    glm_mu, glm_sd = cv_mae(lr_poisson(), X, y)\n",
    "\n",
    "    rows_tem.append({\n",
    "        \"Era\": ERA_LABELS[era], \"n\": len(y), \"media\": f\"{y.mean():.2f}\",\n",
    "        \"Baseline MAE\": f\"{base_mae:.3f}\",\n",
    "        \"GLM Poisson\":  f\"{glm_mu:.3f}+-{glm_sd:.3f}\",\n",
    "        \"mejora%\": f\"{(1 - glm_mu / base_mae) * 100:.1f}%\",\n",
    "    })\n",
    "    print(f\"  {era:<18}  base={base_mae:.3f}  GLM={glm_mu:.3f}\")\n",
    "\n",
    "pd.DataFrame(rows_tem)\n",
])

# Cell 132: SHAP Tematicas
set_code(cells[132], [
    "shap_tem   = {}\n",
    "models_tem = {}\n",
    "\n",
    "for era in ERA_ORDER:\n",
    "    X, y = get_Xy(era, \"n_comisiones_tematicas\")\n",
    "\n",
    "    m = lr_poisson()\n",
    "    m.fit(X, y)\n",
    "    models_tem[era] = m\n",
    "\n",
    "    X_sc = m.named_steps[\"sc\"].transform(X)\n",
    "    explainer = shap.LinearExplainer(m.named_steps[\"pr\"], X_sc)\n",
    "    sv = explainer.shap_values(X_sc)\n",
    "\n",
    "    shap_tem[era] = (sv, X)\n",
    "    print(f\"  {era}: SHAP OK\")\n",
])

# Cell 136: static tematicas table -> dynamic
set_code(cells[136], [
    "display(pd.DataFrame(rows_tem))\n",
])

# Cell 144: rolling_forward using LR L1 SFM
set_code(cells[144], [
    "def rolling_forward(target, is_binary=True):\n",
    "    results = []\n",
    "    for train_era, test_era in zip(ERA_ORDER[:-1], ERA_ORDER[1:]):\n",
    "        X_tr, y_tr = get_Xy(train_era, target)\n",
    "        X_te, y_te = get_Xy(test_era,  target)\n",
    "\n",
    "        if is_binary:\n",
    "            m = lr_l1_sfm()\n",
    "            m.fit(X_tr, y_tr)\n",
    "            prob  = m.predict_proba(X_te)[:, 1]\n",
    "            score = roc_auc_score(y_te, prob)\n",
    "            metric = \"AUC\"\n",
    "        else:\n",
    "            m = lr_poisson()\n",
    "            m.fit(X_tr, y_tr)\n",
    "            score  = mean_absolute_error(y_te, m.predict(X_te))\n",
    "            metric = \"MAE\"\n",
    "\n",
    "        results.append({\n",
    "            \"Transicion\": f\"{ERA_LABELS[train_era].split('(')[0].strip()} -> {ERA_LABELS[test_era].split('(')[0].strip()}\",\n",
    "            metric: round(score, 3),\n",
    "        })\n",
    "    return pd.DataFrame(results)\n",
    "\n",
    "df_roll_n = rolling_forward(\"nodal_bin\",              is_binary=True)\n",
    "df_roll_l = rolling_forward(\"lastre_bin\",             is_binary=True)\n",
    "df_roll_t = rolling_forward(\"n_comisiones_tematicas\", is_binary=False)\n",
    "\n",
    "print(\"Nodales:\");   display(df_roll_n)\n",
    "print(\"Lastre:\");    display(df_roll_l)\n",
    "print(\"Tematicas:\"); display(df_roll_t)\n",
])

# Cell 160: summary table without RF/XGB
set_code(cells[160], [
    "summary = pd.DataFrame([\n",
    "    # Nodales — LR L1 (full) values from v5 reference; SFM updated after run\n",
    "    {\"Target\": \"Nodales\",   \"Epoca\": \"ERA_1 PRI\",    \"Metrica\": \"AUC\",  \"LR L1 (full)\": 0.734, \"LR L1 + SFM\": None, \"Senal\": \"Moderada\"},\n",
    "    {\"Target\": \"Nodales\",   \"Epoca\": \"ERA_2 PAN\",    \"Metrica\": \"AUC\",  \"LR L1 (full)\": 0.720, \"LR L1 + SFM\": None, \"Senal\": \"Moderada\"},\n",
    "    {\"Target\": \"Nodales\",   \"Epoca\": \"ERA_3 Trans.\", \"Metrica\": \"AUC\",  \"LR L1 (full)\": 0.699, \"LR L1 + SFM\": None, \"Senal\": \"Moderada-debil\"},\n",
    "    {\"Target\": \"Nodales\",   \"Epoca\": \"ERA_4 Morena\", \"Metrica\": \"AUC\",  \"LR L1 (full)\": 0.619, \"LR L1 + SFM\": None, \"Senal\": \"Debil\"},\n",
    "    # Lastre\n",
    "    {\"Target\": \"Lastre\",    \"Epoca\": \"ERA_1 PRI\",    \"Metrica\": \"AUC\",  \"LR L1 (full)\": 0.585, \"LR L1 + SFM\": None, \"Senal\": \"Debil\"},\n",
    "    {\"Target\": \"Lastre\",    \"Epoca\": \"ERA_2 PAN\",    \"Metrica\": \"AUC\",  \"LR L1 (full)\": 0.632, \"LR L1 + SFM\": None, \"Senal\": \"Debil\"},\n",
    "    {\"Target\": \"Lastre\",    \"Epoca\": \"ERA_3 Trans.\", \"Metrica\": \"AUC\",  \"LR L1 (full)\": 0.583, \"LR L1 + SFM\": None, \"Senal\": \"Debil\"},\n",
    "    {\"Target\": \"Lastre\",    \"Epoca\": \"ERA_4 Morena\", \"Metrica\": \"AUC\",  \"LR L1 (full)\": 0.530, \"LR L1 + SFM\": None, \"Senal\": \"Muy debil\"},\n",
    "    # Tematicas\n",
    "    {\"Target\": \"Tematicas\", \"Epoca\": \"ERA_1 PRI\",    \"Metrica\": \"MAE\",  \"LR L1 (full)\": 0.803, \"LR L1 + SFM\": None, \"Senal\": \"Marginal\"},\n",
    "    {\"Target\": \"Tematicas\", \"Epoca\": \"ERA_2 PAN\",    \"Metrica\": \"MAE\",  \"LR L1 (full)\": 0.791, \"LR L1 + SFM\": None, \"Senal\": \"Ninguna\"},\n",
    "    {\"Target\": \"Tematicas\", \"Epoca\": \"ERA_3 Trans.\", \"Metrica\": \"MAE\",  \"LR L1 (full)\": 0.885, \"LR L1 + SFM\": None, \"Senal\": \"Marginal\"},\n",
    "    {\"Target\": \"Tematicas\", \"Epoca\": \"ERA_4 Morena\", \"Metrica\": \"MAE\",  \"LR L1 (full)\": 0.742, \"LR L1 + SFM\": None, \"Senal\": \"Marginal\"},\n",
    "])\n",
    "print(\"Nota: LR L1 + SFM se actualiza tras ejecucion. LR L1 (full) = referencia v5.\")\n",
    "display(summary.set_index([\"Target\", \"Epoca\"]))\n",
])

# Cell 169: robustness using lr_l1_sfm
set_code(cells[169], [
    "from sklearn.model_selection import StratifiedKFold\n",
    "import numpy as np\n",
    "\n",
    "ALT_TARGETS = [\n",
    "    (\"nodal_bin\",        \"Base (>=1 nodal)\"),\n",
    "    (\"nodal_bin_strict\", \"Alt-1 Estricta (>=2 nodales)\"),\n",
    "    (\"nodal_bin_pres\",   \"Alt-2 Ampliada (>=1 nodal|presidente)\"),\n",
    "]\n",
    "\n",
    "rows_rob = []\n",
    "kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "\n",
    "for target, label in ALT_TARGETS:\n",
    "    for era in ERA_ORDER:\n",
    "        X, y = get_Xy(era, target)\n",
    "        if y.sum() < 10:\n",
    "            continue\n",
    "        aucs = []\n",
    "        for train_idx, val_idx in kf.split(X, y):\n",
    "            m = lr_l1_sfm()\n",
    "            m.fit(X.iloc[train_idx], y.iloc[train_idx])\n",
    "            aucs.append(roc_auc_score(\n",
    "                y.iloc[val_idx],\n",
    "                m.predict_proba(X.iloc[val_idx])[:, 1]\n",
    "            ))\n",
    "        rows_rob.append({\n",
    "            \"Clasificacion\": label,\n",
    "            \"Era\": ERA_LABELS[era],\n",
    "            \"n\": len(y),\n",
    "            \"tasa_pos\": f\"{y.mean():.1%}\",\n",
    "            \"AUC\": round(float(np.mean(aucs)), 3),\n",
    "            \"AUC_sd\": round(float(np.std(aucs)), 3),\n",
    "        })\n",
    "\n",
    "df_rob = pd.DataFrame(rows_rob)\n",
    "pivot = df_rob.pivot(\n",
    "    index=\"Era\", columns=\"Clasificacion\", values=\"AUC\"\n",
    ").round(3)\n",
    "display(pivot.style\n",
    "        .background_gradient(cmap=\"YlOrRd\", axis=None)\n",
    "        .set_caption(\n",
    "            \"AUC LR L1 + SFM  5-fold CV — Comparacion entre clasificaciones nodales\"\n",
    "        ))\n",
    "print()\n",
    "display(df_rob)\n",
])

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done. Cells modified: 0, 88, 97, 100, 102, 112, 118, 119, 127, 131, 132, 136, 144, 160, 169")
