# -*- coding: utf-8 -*-
"""Construye notebooks/diputraxv13.ipynb a partir de diputraxv10.ipynb (plan10upd.md).

v13 = diseño de v10 con dos cambios de fondo:
  1. Segmentación S3 de diputraxv12: 3 eras (57-59 PRI / 60-62 PAN / 63-66
     Transición-Morena) en lugar de las 4 eras (S4).
  2. Solo modelos de regresión: se eliminan RF, XGBoost (binario y Poisson) y
     la sección 12 completa (PyTorch MTL + baseline diputraxv3). Se conservan
     LR L2/L1/L1+SFM, GLM Poisson y la capa bayesiana (PyMC NUTS).

diputraxv10.ipynb NO se modifica. Verifica índices y contenido vigente antes
de aplicar; aborta con mensaje claro si el notebook no coincide.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks" / "diputraxv10.ipynb"
OUT = ROOT / "notebooks" / "diputraxv13.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]
if len(cells) != 256:
    sys.exit(f"ABORT: se esperaban 256 celdas en v10, hay {len(cells)}")

errors = []


def get(idx):
    s = cells[idx]["source"]
    return s if isinstance(s, str) else "".join(s)


def put(idx, text):
    cells[idx]["source"] = text.splitlines(keepends=True)


# --------------------------------------------------------------------------
# Reemplazos exactos: (celda, viejo, nuevo, n_esperado)
# --------------------------------------------------------------------------
REPL = []


def R(idx, old, new, count=1):
    REPL.append((idx, old, new, count))


# ===== Celda 2 — contexto: narrativa de regímenes a 3 eras =====
R(2, "La hipótesis adicional es que esa señal ha cambiado con los cambios de "
     "régimen político, partiendo de la salida del PRI hegemónico —1997–2006—, "
     "pasando por la alternancia del PAN —2006–2012—, y culminando con el "
     "período de transición multipartidista —2012–2021— que vio su fin con el "
     "acelerado surgimiento y dominio del nuevo bloque Morena —2021–presente—. "
     "Mismos que no operan bajo los mismos criterios de distribución del poder "
     "legislativo.",
     "La hipótesis adicional es que esa señal ha cambiado con los cambios de "
     "régimen político, a lo largo de tres configuraciones de coalición "
     "dominante: la hegemonía priista en retirada —1997–2006—, el bipartidismo "
     "con presidencias panistas —2006–2015— y el ciclo Transición-Morena "
     "posterior a 2015 —fragmentación multipartidista primero, mayorías de "
     "Morena después—. Configuraciones que no operan bajo los mismos criterios "
     "de distribución del poder legislativo.")

# ===== Celda 4 — 86 features -> 61 =====
R(4, "se construyeron 86 *features* organizados en seis bloques",
     "se construyeron 61 *features* organizados en seis bloques")

# ===== Celda 7 — fuera de alcance: subrégimen 2024+ y modelos no lineales =====
R(7, "- **Subrepresentación Morena:** no se cuenta con mucha información sobre "
     "el desarrollo de los patrones de reclutamiento de Morena en el SIL al "
     "momento del análisis. La ERA_4 cubre únicamente la LXVI (`n = 500`), lo "
     "que limita la representatividad de los resultados para el período Morena.",
     "- **Resolución del subrégimen Morena 2024+:** la segmentación S3 modela "
     "las legislaturas 63–66 como un solo grupo (`ERA_3`); el cuaderno no "
     "estima un modelo separado para la LXVI. La evidencia de `diputraxv12` "
     "muestra que un grupo propio para la LXVI (n=500) maximiza la varianza "
     "sin ganancia predictiva; el análisis fino del subrégimen 2024+ queda "
     "para cuando exista la LXVII.\n"
     "- **Modelos no lineales y multitarea:** Random Forest, XGBoost y el MTL "
     "de PyTorch quedan fuera de este cuaderno; su contraste con la línea de "
     "regresión está documentado en `diputraxv10.ipynb` y "
     "`diputraxpytorch.ipynb`.")

# ===== Celda 9 — estrategia: segmentación en 3 épocas =====
R(9, "#### Estrategia de análisis temporal: cuatro épocas legislativas\n"
     "\n"
     "La base cubre las legislaturas LVII–LXVI —1997–presente—. Se segmenta en "
     "cuatro épocas analíticas para comparar evolución de perfiles "
     "legislativos, profesionalización y composición partidista.\n"
     "\n"
     "| Época | Legislaturas | Filas | Diputados únicos | Periodo |\n"
     "|---|---|---:|---:|---|\n"
     "| Primera | LVII–LIX | 1,500 | 1,468 | 1997–2006 |\n"
     "| Segunda | LX–LXII | 1,500 | 1,476 | 2006–2015 |\n"
     "| Tercera | LXIII–LXV | 1,500 | 1,371 | 2015–2024 |\n"
     "| Cuarta | LXVI | 500 | 500 | 2024– |",
     "#### Estrategia de análisis temporal: tres épocas legislativas\n"
     "\n"
     "La base cubre las legislaturas LVII–LXVI —1997–presente—. Se segmenta en "
     "tres épocas analíticas —la segmentación `S3_FUSION34` evaluada en "
     "`diputraxv12`— para comparar evolución de perfiles legislativos, "
     "profesionalización y composición partidista.\n"
     "\n"
     "| Época | Legislaturas | Filas | Periodo |\n"
     "|---|---|---:|---|\n"
     "| Primera | LVII–LIX | 1,500 | 1997–2006 |\n"
     "| Segunda | LX–LXII | 1,500 | 2006–2015 |\n"
     "| Tercera | LXIII–LXVI | 2,000 | 2015–presente |")

R(9, "##### Equilibrio estadístico\n"
     "\n"
     "Las primeras tres épocas tienen exactamente 1,500 filas cada una —tres "
     "legislaturas de aproximadamente 500 diputados—. Esto permite trabajar "
     "con submuestras homogéneas y comparables. La cuarta época —500 "
     "observaciones— se separa para no romper esa simetría.",
     "##### Equilibrio estadístico\n"
     "\n"
     "Los tres grupos tienen entre 1,500 y 2,000 filas (tres o cuatro "
     "legislaturas de aproximadamente 500 diputados). Ningún grupo hereda el "
     "n=500 de una sola legislatura: la evaluación comparativa de "
     "`diputraxv12` (§9) muestra que separar la LXVI en un grupo propio "
     "maximiza la varianza de los estimadores sin ganancia predictiva, y que "
     "la fusión 63–66 alcanza la mayor cohesión interna de coeficientes de "
     "todo el espacio de candidatos (similitud dentro de grupo 0.537).")

R(9, "ERA_3 tiene 129 registros de reelección, consecuencia de la reforma "
     "constitucional de 2014 —efectiva desde la LXIV, 2018—, que rehabilitó la "
     "reelección consecutiva.",
     "ERA_3 —que ahora abarca LXIII–LXVI— concentra los registros de "
     "reelección de la serie, consecuencia de la reforma constitucional de "
     "2014 —efectiva desde la LXIV, 2018—, que rehabilitó la reelección "
     "consecutiva.")

R(9, "**Tercera época —LXIII–LXV, 2015–2024: ascenso y consolidación de "
     "MORENA.**  \n"
     "La LXIV —2018— representa el primer quiebre histórico de mayoría "
     "absoluta por un partido no priísta ni panista. Llega también la "
     "reelección legislativa. Los indicadores de comisiones nodales suben de "
     "0.63 en ERA_2 a 0.82 en ERA_3, reflejando que MORENA concentra "
     "posiciones de poder dentro de la Cámara.\n"
     "\n"
     "**Cuarta época —LXVI, 2024–: supermayoría de MORENA y aliados.**  \n"
     "Este periodo inicia con el gobierno de Claudia Sheinbaum y una "
     "supermayoría legislativa de MORENA y aliados —más de dos tercios del "
     "pleno—. Tratarla como era propia es analíticamente correcto: sus "
     "patrones emergentes —máxima concentración de poder, edad promedio más "
     "alta de 48.37 años y la mayor proporción de comisiones nodales, con "
     "0.88— podrían distorsionar cualquier era anterior si se fusionan.",
     "**Tercera época —LXIII–LXVI, 2015–presente: ascenso, consolidación y "
     "supermayoría de MORENA.**  \n"
     "La LXIV —2018— representa el primer quiebre histórico de mayoría "
     "absoluta por un partido no priísta ni panista; llega también la "
     "reelección legislativa. El ciclo culmina en la LXVI —2024, gobierno de "
     "Claudia Sheinbaum— con una supermayoría de MORENA y aliados de más de "
     "dos tercios del pleno. Los indicadores de comisiones nodales suben de "
     "0.63 en ERA_2 a ~0.84 en este tramo, reflejando la concentración "
     "progresiva de posiciones de poder. La fusión 63–66 en un solo grupo es "
     "la decisión central de esta versión: `diputraxv12` muestra que el costo "
     "predictivo de esa fusión es estadísticamente indistinguible de cero y "
     "que el grupo resultante es el de mayor cohesión interna de "
     "coeficientes; a cambio, el subrégimen 2024+ no se modela por separado "
     "(límite de resolución declarado en §11).")

R(9, "| Variable | ERA 1 | ERA 2 | ERA 3 | ERA 4 |\n"
     "|---|---:|---:|---:|---:|\n"
     "| Edad promedio | 44.82 | 45.76 | 47.37 | 48.37 |\n"
     "| Comisiones nodales | 0.47 | 0.63 | 0.82 | 0.88 |\n"
     "| Comisiones temáticas | 1.53 | 1.93 | 2.15 | 1.87 |",
     "| Variable | ERA 1 | ERA 2 | ERA 3 |\n"
     "|---|---:|---:|---:|\n"
     "| Edad promedio | 44.82 | 45.76 | 47.62 |\n"
     "| Comisiones nodales | 0.47 | 0.63 | 0.84 |\n"
     "| Comisiones temáticas | 1.53 | 1.93 | 2.08 |")

R(9, "Asimismo, la participación en comisiones nodales crece de forma "
     "continua, pasando de 0.47 en la primera era a 0.88 en la cuarta.",
     "Asimismo, la participación en comisiones nodales crece de forma "
     "continua, pasando de 0.47 en la primera era a 0.84 en la tercera.")

# ===== Celda 11 — constantes EDA (era_map minúsculas) =====
R(11, 'era_map = {\n'
      '    57: "ERA_1", 58: "ERA_1", 59: "ERA_1",\n'
      '    60: "ERA_2", 61: "ERA_2", 62: "ERA_2",\n'
      '    63: "ERA_3", 64: "ERA_3", 65: "ERA_3",\n'
      '    66: "ERA_4",\n'
      '}',
      'era_map = {\n'
      '    57: "ERA_1", 58: "ERA_1", 59: "ERA_1",\n'
      '    60: "ERA_2", 61: "ERA_2", 62: "ERA_2",\n'
      '    63: "ERA_3", 64: "ERA_3", 65: "ERA_3", 66: "ERA_3",\n'
      '}')
R(11, '    "ERA_3": "ERA_3 - LXIII-LXV",\n'
      '    "ERA_4": "ERA_4 - LXVI",\n',
      '    "ERA_3": "ERA_3 - LXIII-LXVI",\n')
R(11, '    "ERA_3 - LXIII-LXV": "Tercera época",\n'
      '    "ERA_4 - LXVI": "Cuarta época",\n',
      '    "ERA_3 - LXIII-LXVI": "Tercera época",\n')
R(11, 'era_order = ["ERA_1", "ERA_2", "ERA_3", "ERA_4"]',
      'era_order = ["ERA_1", "ERA_2", "ERA_3"]')
R(11, 'era_nombre_order = ["Primera época", "Segunda época", "Tercera época", "Cuarta época"]',
      'era_nombre_order = ["Primera época", "Segunda época", "Tercera época"]')
R(11, '    "Tercera época": "#3D405B",\n'
      '    "Cuarta época": "#81B29A",\n',
      '    "Tercera época": "#3D405B",\n')

# ===== Celda 20 — validación MICE: etiquetas hardcodeadas =====
R(20, "_era_labels  = ['Primera época', 'Segunda época', 'Tercera época', 'Cuarta época']",
      "_era_labels  = ['Primera época', 'Segunda época', 'Tercera época']")
R(20, "axes[1].set_xticklabels(['Era 1', 'Era 2', 'Era 3', 'Era 4'])",
      "axes[1].set_xticklabels(['Era 1', 'Era 2', 'Era 3'])")

# ===== Grids EDA 2x2 -> 1x3 =====
R(23, "fig, axes = plt.subplots(2, 2, figsize=(16, 12))",
      "fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))")
R(26, "fig, axes = plt.subplots(2, 2, figsize=(16, 10))",
      "fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))")
R(33, "fig, axes = plt.subplots(2, 2, figsize=(16, 12))",
      "fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))")
R(35, 'ax.set_xticklabels(["E1","E2","E3","E4"])',
      'ax.set_xticklabels(["E1","E2","E3"])')
R(38, "fig, axes = plt.subplots(2, 2, figsize=(16, 10))",
      "fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))")
R(41, "fig, axes = plt.subplots(2, 2, figsize=(16, 12))",
      "fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))")
R(43, "    fig, axes = plt.subplots(2, 2, figsize=(16, 12))",
      "    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))")
R(52, "fig, axes = plt.subplots(2, 2, figsize=(16, 12))",
      "fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))")
R(57, "fig, axes = plt.subplots(2, 2, figsize=(16, 12))",
      "fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))")
R(60, 'ax.set_xticklabels(["E1 PRI", "E2 PAN", "E3 Trans.", "E4 Morena"], rotation=15)',
      'ax.set_xticklabels(["E1 PRI", "E2 PAN", "E3 Trans-Mor"], rotation=15)')
R(63, "fig, axes = plt.subplots(2, 2, figsize=(22, 18))",
      "fig, axes = plt.subplots(1, 3, figsize=(30, 9.5))")

# ===== Celda 69 — imports: quitar RF y XGBoost =====
R(69, "from sklearn.linear_model import LogisticRegression, PoissonRegressor\n"
      "from sklearn.ensemble import RandomForestClassifier\n"
      "import xgboost as xgb\n"
      "import shap",
      "from sklearn.linear_model import LogisticRegression, PoissonRegressor\n"
      "import shap")

# ===== Celda 70 — constantes de modelado (ERA_MAP mayúsculas) =====
R(70, 'ERA_MAP = {\n'
      '    57: "ERA_1_PRI",  58: "ERA_1_PRI",  59: "ERA_1_PRI",\n'
      '    60: "ERA_2_PAN",  61: "ERA_2_PAN",  62: "ERA_2_PAN",\n'
      '    63: "ERA_3_TRANS", 64: "ERA_3_TRANS", 65: "ERA_3_TRANS",\n'
      '    66: "ERA_4_MORENA",\n'
      '}\n'
      'ERA_ORDER = ["ERA_1_PRI", "ERA_2_PAN", "ERA_3_TRANS", "ERA_4_MORENA"]\n'
      'ERA_LABELS = {\n'
      '    "ERA_1_PRI":    "ERA 1 — PRI (57-59)",\n'
      '    "ERA_2_PAN":    "ERA 2 — PAN (60-62)",\n'
      '    "ERA_3_TRANS":  "ERA 3 — Transicion (63-65)",\n'
      '    "ERA_4_MORENA": "ERA 4 — Morena (66)",\n'
      '}\n'
      'ERA_COLORS = {\n'
      '    "ERA_1_PRI":    "#c0392b",\n'
      '    "ERA_2_PAN":    "#2980b9",\n'
      '    "ERA_3_TRANS":  "#8e44ad",\n'
      '    "ERA_4_MORENA": "#27ae60",\n'
      '}',
      'ERA_MAP = {\n'
      '    57: "ERA_1_PRI",  58: "ERA_1_PRI",  59: "ERA_1_PRI",\n'
      '    60: "ERA_2_PAN",  61: "ERA_2_PAN",  62: "ERA_2_PAN",\n'
      '    63: "ERA_3_TRANSMOR", 64: "ERA_3_TRANSMOR",\n'
      '    65: "ERA_3_TRANSMOR", 66: "ERA_3_TRANSMOR",\n'
      '}\n'
      'ERA_ORDER = ["ERA_1_PRI", "ERA_2_PAN", "ERA_3_TRANSMOR"]\n'
      'ERA_LABELS = {\n'
      '    "ERA_1_PRI":      "ERA 1 — PRI (57-59)",\n'
      '    "ERA_2_PAN":      "ERA 2 — PAN (60-62)",\n'
      '    "ERA_3_TRANSMOR": "ERA 3 — Transicion-Morena (63-66)",\n'
      '}\n'
      'ERA_COLORS = {\n'
      '    "ERA_1_PRI":      "#c0392b",\n'
      '    "ERA_2_PAN":      "#2980b9",\n'
      '    "ERA_3_TRANSMOR": "#8e44ad",\n'
      '}')

# ===== Celda 76 — comentario ERA_1–ERA_4 =====
R(76, "señal dominante persistente en ERA_1–ERA_4",
      "señal dominante persistente en ERA_1–ERA_3")

# ===== Celda 78 — factories: quitar spw, RF y XGBoost =====
R(78, 'def spw(y):\n'
      '    """scale_pos_weight para XGBoost (ratio neg/pos)."""\n'
      '    n0 = float((y == 0).sum())\n'
      '    n1 = float((y == 1).sum())\n'
      '    return n0 / max(n1, 1.0)\n'
      '\n'
      '# Model factories', '# Model factories')
R(78, 'def rf_binary():\n'
      '    """Bosques Aleatorios (RF) de v4."""\n'
      '    return RandomForestClassifier(\n'
      '        n_estimators=500, max_depth=6, min_samples_leaf=15,\n'
      '        max_features="sqrt", class_weight="balanced",\n'
      '        n_jobs=-1, random_state=42,\n'
      '    )\n'
      '\n'
      'def make_xgb_binary(scale_pos_weight=1.0):\n'
      '    """XGBoost de v4."""\n'
      '    return xgb.XGBClassifier(\n'
      '        n_estimators=300, learning_rate=0.05, max_depth=4,\n'
      '        subsample=0.8, colsample_bytree=0.8,\n'
      '        scale_pos_weight=scale_pos_weight,\n'
      '        eval_metric="auc", verbosity=0,\n'
      '        n_jobs=-1, random_state=42,\n'
      '    )\n'
      '\n'
      'def lr_poisson():', 'def lr_poisson():')
R(78, '\n'
      'def make_xgb_poisson():\n'
      '    """XGBoost Poisson de v4."""\n'
      '    return xgb.XGBRegressor(\n'
      '        objective="count:poisson",\n'
      '        n_estimators=300, learning_rate=0.05, max_depth=4,\n'
      '        subsample=0.8, colsample_bytree=0.8,\n'
      '        verbosity=0, n_jobs=-1, random_state=42,\n'
      '    )\n', '')

# ===== Celda 85 — entrenamiento nodales sin RF/XGB =====
R(85, 'print("Entrenando modelos para NODALES (Clásicos ML + Lineales Lasso)... ")',
      'print("Entrenando modelos para NODALES (variantes de Regresión Logística)... ")')
R(85, '    X, y = get_Xy(era, "nodal_bin")\n'
      '    w = spw(y)\n',
      '    X, y = get_Xy(era, "nodal_bin")\n')
R(85, '    sfm_mu, sfm_sd = cv_auc(lr_l1_sfm(),  X, y)\n'
      '    rf_mu,  rf_sd  = cv_auc(rf_binary(),  X, y)\n'
      '    xg_mu,  xg_sd  = cv_auc(make_xgb_binary(w), X, y)\n',
      '    sfm_mu, sfm_sd = cv_auc(lr_l1_sfm(),  X, y)\n')
R(85, '        "LR L2 (v4)":   f"{v4_mu:.3f}+-{v4_sd:.3f}",\n'
      '        "RF (v4)":      f"{rf_mu:.3f}+-{rf_sd:.3f}",\n'
      '        "XGB (v4)":     f"{xg_mu:.3f}+-{xg_sd:.3f}",\n',
      '        "LR L2 (v4)":   f"{v4_mu:.3f}+-{v4_sd:.3f}",\n')
R(85, '    print(f"  {era:<18}  LR_L2={v4_mu:.3f} RF={rf_mu:.3f} XGB={xg_mu:.3f} '
      'LR_L1={lr_mu:.3f} SFM={sfm_mu:.3f} features={len(selected)}")',
      '    print(f"  {era:<22}  LR_L2={v4_mu:.3f} LR_L1={lr_mu:.3f} '
      'SFM={sfm_mu:.3f} features={len(selected)}")')

# ===== Celda 87 — SHAP nodal + regeneración del grid beeswarm (1x3) =====
R(87, '    shap_nodal[era] = (sv_full, X)\n'
      '    print(f"  {era}: SHAP OK  (selected={mask.sum()}/{len(mask)})")',
      '    shap_nodal[era] = (sv_full, X)\n'
      '    print(f"  {era}: SHAP OK  (selected={mask.sum()}/{len(mask)})")\n'
      '\n'
      '# Grid beeswarm 1x3 (regenerado con la segmentación S3)\n'
      'imgs_nodal = {era: shap_beeswarm_to_img(\n'
      '                  shap_nodal[era][0], shap_nodal[era][1], top_n=15,\n'
      '                  title=ERA_LABELS[era], color=ERA_COLORS[era])\n'
      '              for era in ERA_ORDER}\n'
      'fig, axes = plt.subplots(1, 3, figsize=(24, 7))\n'
      'for ax, era in zip(axes, ERA_ORDER):\n'
      '    ax.imshow(imgs_nodal[era])\n'
      '    ax.axis("off")\n'
      'plt.suptitle("Comisiones Nodales — SHAP beeswarm por era",\n'
      '             fontsize=15, fontweight="bold")\n'
      'plt.tight_layout()\n'
      'plt.savefig(REPORT_DIR / "shap_nodales_beeswarm.png", bbox_inches="tight", dpi=120)\n'
      'plt.show()')

# ===== Celda 93 — range(4) -> range(len(ERA_ORDER)) =====
R(93, "    ax.plot(range(4), df_trend_n[feat], marker=markers[i],",
      "    ax.plot(range(len(ERA_ORDER)), df_trend_n[feat], marker=markers[i],")
R(93, "ax.set_xticks(range(4))",
      "ax.set_xticks(range(len(ERA_ORDER)))")

# ===== Celdas 102/103, 119/120 — Bayes: 4 eras -> 3 eras =====
R(102, "> Runtime aprox. 5-15 min segun hardware (NUTS x 4 eras x 4 cadenas).",
       "> Runtime aprox. 5-15 min segun hardware (NUTS x 3 eras x 4 cadenas).")
R(103, "print('Entrenando Bayesiano NODAL (NUTS) - 4 eras...')",
       "print('Entrenando Bayesiano NODAL (NUTS) - 3 eras...')")
R(119, "> Runtime aprox. 5-15 min (NUTS x 4 eras x 4 cadenas).",
       "> Runtime aprox. 5-15 min (NUTS x 3 eras x 4 cadenas).")
R(120, "print('Entrenando Bayesiano LASTRE (NUTS) - 4 eras...')",
       "print('Entrenando Bayesiano LASTRE (NUTS) - 3 eras...')")

# ===== Celda 107 — entrenamiento lastre sin RF/XGB =====
R(107, 'print("Entrenando modelos para LASTRE (Clásicos ML + Lineales Lasso)... ")',
       'print("Entrenando modelos para LASTRE (variantes de Regresión Logística)... ")')
R(107, '    X, y = get_Xy(era, "lastre_bin")\n'
       '    w = spw(y)\n',
       '    X, y = get_Xy(era, "lastre_bin")\n')
R(107, '    sfm_mu, sfm_sd = cv_auc(lr_l1_sfm(),  X, y)\n'
       '    rf_mu,  rf_sd  = cv_auc(rf_binary(),  X, y)\n'
       '    xg_mu,  xg_sd  = cv_auc(make_xgb_binary(w), X, y)\n',
       '    sfm_mu, sfm_sd = cv_auc(lr_l1_sfm(),  X, y)\n')
R(107, '        "LR L2 (v4)":   f"{v4_mu:.3f}+-{v4_sd:.3f}",\n'
       '        "RF (v4)":      f"{rf_mu:.3f}+-{rf_sd:.3f}",\n'
       '        "XGB (v4)":     f"{xg_mu:.3f}+-{xg_sd:.3f}",\n',
       '        "LR L2 (v4)":   f"{v4_mu:.3f}+-{v4_sd:.3f}",\n')
R(107, '    print(f"  {era:<18}  LR_L2={v4_mu:.3f} RF={rf_mu:.3f} XGB={xg_mu:.3f} '
       'LR_L1={lr_mu:.3f} SFM={sfm_mu:.3f} features={len(selected)}")',
       '    print(f"  {era:<22}  LR_L2={v4_mu:.3f} LR_L1={lr_mu:.3f} '
       'SFM={sfm_mu:.3f} features={len(selected)}")')

# ===== Celda 114 — dict espejo: 3 claves (textos definitivos en fase 3) =====
R(114, '_mirror_read = {\n'
       '    "ERA_1_PRI":    ("Relación inversa débil",  "asignación no es por exclusión simple"),\n'
       '    "ERA_2_PAN":    ("Inversa moderada",         "la más cercana a un patrón bimodal mayoría/oposición"),\n'
       '    "ERA_3_TRANS":  ("Relación inversa débil",  "fragmentación diluye cualquier lógica de espejo"),\n'
       '    "ERA_4_MORENA": ("Inversa moderada",         "patrón similar a ERA_2 bajo nuevo alineamiento"),\n'
       '}',
       '_mirror_read = {\n'
       '    "ERA_1_PRI":      ("Relación inversa débil", "asignación no es por exclusión simple"),\n'
       '    "ERA_2_PAN":      ("Inversa moderada",        "la más cercana a un patrón bimodal mayoría/oposición"),\n'
       '    "ERA_3_TRANSMOR": ("Ver corrida",             "grupo fusionado 63-66 (actualizar tras ejecución)"),\n'
       '}')

# ===== Celda 124 — temáticas: solo GLM Poisson =====
R(124, 'print("Entrenando modelos para TEMATICAS (GLM Poisson + XGBoost Poisson)... ")',
       'print("Entrenando modelos para TEMATICAS (GLM Poisson)... ")')
R(124, '    poi_mu, poi_sd = cv_mae(lr_poisson(), X, y)\n'
       '    xgb_poi_mu, xgb_poi_sd = cv_mae(make_xgb_poisson(), X, y)\n',
       '    poi_mu, poi_sd = cv_mae(lr_poisson(), X, y)\n')
R(124, '        "Poisson GLM": f"{poi_mu:.3f}+-{poi_sd:.3f}",\n'
       '        "XGB Poisson": f"{xgb_poi_mu:.3f}+-{xgb_poi_sd:.3f}",\n',
       '        "Poisson GLM": f"{poi_mu:.3f}+-{poi_sd:.3f}",\n')
R(124, '    print(f"  {era:<18}  GLM={poi_mu:.3f}  XGB_Poi={xgb_poi_mu:.3f}")',
       '    print(f"  {era:<22}  GLM={poi_mu:.3f}")')

# ===== Celda 126 — grid beeswarm temáticas 1x3 =====
R(126, "fig, axes = plt.subplots(2, 2, figsize=(18, 14))",
       "fig, axes = plt.subplots(1, 3, figsize=(24, 7))")

# ===== Celda 132 — cuatro eras -> tres eras =====
R(132, "Sintesis de ambos paradigmas para los dos targets binarios (nodal y lastre) x cuatro eras.",
       "Sintesis de ambos paradigmas para los dos targets binarios (nodal y lastre) x tres eras.")

# ===== Celda 150 — waterfall 1x3 =====
R(150, "fig, axes = plt.subplots(2, 2, figsize=(20, 16))",
       "fig, axes = plt.subplots(1, 3, figsize=(27, 8))")

# ===== Celda 156 — resumen: 4 eras -> 3, 86 -> 61 =====
R(156, "Vista unificada de los resultados: 3 targets × 4 eras × 2 variantes",
       "Vista unificada de los resultados: 3 targets × 3 eras × 2 variantes")
R(156, "cuántas de las 86 variables del perfil biográfico",
       "cuántas de las 61 variables del perfil biográfico")

# ===== Celda 162 — suptitle sin referencia a ERA_4 n=500 =====
R(162, '    "Potencia estadistica: AUC observado e intervalos de confianza 95% (Hanley-McNeil)\\n"\n'
       '    "Las barras de error amplias de ERA_4 reflejan n=500 (una legislatura)",',
       '    "Potencia estadistica: AUC observado e intervalos de confianza 95% (Hanley-McNeil)\\n"\n'
       '    "Los tres grupos tienen n=1,500-2,000; ninguno hereda el n=500 de una sola legislatura",')

# ===== Celda 170 — 86+ -> 61 =====
R(170, "Cubre las 86+ variables del modelo",
       "Cubre las 61 variables del modelo")

# ===== Celda 174 — Anexo D.0: n de eras y features =====
R(174, "en vez de reportar los 64 *features* completos",
       "en vez de reportar los 61 *features* completos")
R(174, "sobre eras de 500–1500 observaciones",
       "sobre eras de 1,500–2,000 observaciones")

# ===== Celda 184 — Anexo E: 8 -> 6 modelos =====
R(184, "(targets `nodal_bin` y `lastre_bin`, cuatro eras cada uno → 8 modelos).",
       "(targets `nodal_bin` y `lastre_bin`, tres eras cada uno → 6 modelos).")

# ===== Celda 185 — print de 8 modelos =====
R(185, 'print("Predicciones OOF (out-of-fold) calculadas para los 8 modelos.")',
       'print(f"Predicciones OOF (out-of-fold) calculadas para los {len(LASSO_MODELS)} modelos.")')

# ===== Celda 188 — matrices de confusión 2x3 =====
R(188, "fig, axes = plt.subplots(2, 4, figsize=(20, 9))",
       "fig, axes = plt.subplots(2, 3, figsize=(16, 9))")

# ===== Celda 195 — panel SHAP representativo: grupo 3 =====
R(195, '# Un panel SHAP representativo por familia (era 4) para no saturar; el resto\n'
       '# queda disponible vía la función shap_panels(m).\n'
       'for m in LASSO_MODELS:\n'
       '    if m["era"] == "ERA_4_MORENA":',
       '# Un panel SHAP representativo por familia (era 3) para no saturar; el resto\n'
       '# queda disponible vía la función shap_panels(m).\n'
       'for m in LASSO_MODELS:\n'
       '    if m["era"] == "ERA_3_TRANSMOR":')

# ===== Celdas 197/199 — subplots por familia: 4 -> 3 =====
R(197, 'fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), sharey=True)',
       'fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)', count=2)
R(199, 'fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True)',
       'fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True)')

# ===== Celda 80 — límite interpretativo generalizado =====
R(80, "(`p_PRI` en ERA_1, `p_MORENA` en ERA_4, VIF≈129)",
      "(`p_PRI` en ERA_1; `p_MORENA` en el tramo final de la serie — VIF alto, ver Anexo D.3)")
R(80, "**H5 (ERA_4) debe leerse en modo asociativo, no causal**: el modelo no "
      "distingue \"ser partido mayoritario\" de \"ser Morena\".",
      "**la lectura partidista de ERA_3 debe ser asociativa, no causal**: en "
      "los tramos de partido dominante el modelo no distingue \"ser partido "
      "mayoritario\" de la identidad partidista concreta.")

# --------------------------------------------------------------------------
# Reescrituras completas: {celda: (ancla_inicio, nuevo_source)}
# --------------------------------------------------------------------------
FULL = {}

FULL[0] = ("# Diputrax V10 — Cuaderno Unificado", """\
# Diputrax V13 — Cuaderno Unificado (segmentación S3, modelos de regresión)

Derivado del diseño de `diputraxv10.ipynb` con dos cambios de fondo:

1. **Segmentación temporal S3** (evaluada en `diputraxv12`): tres eras — PRI (57–59), PAN (60–62) y **Transición-Morena (63–66)** — en lugar de las cuatro eras de v10. La fusión del tramo 2015–presente elimina el grupo de n=500 y se sitúa en la frontera eficiente del intercambio sesgo-varianza.
2. **Solo modelos de regresión**: Regresión Logística L2/L1/L1+`SelectFromModel`, GLM Poisson y Regresión Logística bayesiana (PyMC NUTS). Los modelos de árboles (Random Forest, XGBoost) y el enfoque multitarea de PyTorch quedan documentados en `diputraxv10.ipynb` y `diputraxpytorch.ipynb`.
""")

FULL[1] = ("# 1. Resumen Ejecutivo Integrado", """\
# 1. Resumen Ejecutivo Integrado

Este cuaderno es la **versión v13** del proyecto Diputrax: el cuaderno unificado de la línea de **modelos de regresión**, con la segmentación temporal **S3** derivada de la evaluación comparativa de `diputraxv12`. Analiza la asignación de diputados federales mexicanos a comisiones legislativas durante el periodo 1997–presente mediante dos vertientes complementarias:

1. **Regresión Logística Frecuentista L1 (Lasso):** modelado lineal interpretable por era, en dos configuraciones (L1 completa y L1 + `SelectFromModel`), con la especificación L2 histórica como línea de robustez y un GLM Poisson para el conteo de comisiones temáticas. Espacio de 61 *features* con imputación MICE e interpretación SHAP (`LinearExplainer`).
2. **Regresión Logística Bayesiana (PyMC, NUTS):** capa de robustez probabilística sobre el mismo subconjunto de variables seleccionado por SFM en cada era, que cuantifica la incertidumbre posterior (HDI 94%) y valida la dirección de los coeficientes frecuentistas.

**Cambio central respecto a v10 — segmentación temporal.** Las diez legislaturas (LVII–LXVI) se agrupan en **tres eras**: `ERA_1_PRI` (57–59), `ERA_2_PAN` (60–62) y `ERA_3_TRANSMOR` (63–66). La decisión proviene de `diputraxv12`, que evaluó seis segmentaciones candidatas bajo un protocolo neutral (predicciones *out-of-fold* sobre estratos fijos por legislatura): la fusión Transición+Morena está en la frontera eficiente del intercambio sesgo-varianza, alcanza la mayor cohesión interna de coeficientes de todo el espacio de candidatos (0.537) y su costo predictivo frente al mejor esquema es estadísticamente indistinguible de cero, a la vez que elimina el grupo de n=500 que limitaba la potencia estadística de v10.

**Cambio de alcance — solo regresión.** Los modelos de árboles (Random Forest, XGBoost) y el aprendizaje profundo multitarea (PyTorch MTL) de v10 quedan fuera de este cuaderno; su evidencia comparativa —que la estructura de la asignación es predominantemente lineal— se hereda de v10 como premisa de diseño y no se re-estima aquí.
""")

FULL[6] = ("# 1.5 Alcance del proyecto", """\
# 1.5 Alcance del proyecto

El proyecto cubre:

- **Sujeto:** diputadas y diputados federales de la Cámara de Diputados del Congreso de la Unión de México, legislaturas LVII–LXVI.
- **Variables explicativas:** perfil biográfico observable en el SIL —trayectorias administrativa, política y legislativa; formación académica; filiación partidaria; tipo de elección, sea mayoría relativa o representación proporcional.
- **Variables objetivo:** asignación a comisión nodal —binaria—, asignación a comisión lastre —binaria— y número de comisiones temáticas recibidas —conteo—.
- **Período analítico:** tres eras definidas a partir de la coalición dominante de la Cámara: ERA_1 PRI —57–59—, ERA_2 PAN —60–62—, ERA_3 Transición-Morena —63–66—. La segmentación corresponde al esquema `S3_FUSION34` evaluado en `diputraxv12`, que la sitúa en la frontera eficiente del intercambio sesgo-varianza (máxima cohesión interna de coeficientes; costo predictivo dentro del umbral de empate; ningún grupo con n<1,500).
- **Enfoque metodológico:** Regresión Logística L1 en dos configuraciones sobre cada combinación de era y variable objetivo binaria, con la especificación L2 (v4) como línea de robustez, complementada por una Regresión Logística Bayesiana —NUTS—. Se ejecuta un análisis SHAP por era, una validación temporal *rolling forward* y una caracterización de perfiles prototípicos por era.
- **Modelo:** Regresión Logística L1 con `SelectFromModel` para selección automática de *features* por era. SHAP con `LinearExplainer` —valores en espacio *log-odds* para clasificación y log-conteo para Poisson—.

---
""")

FULL[68] = ("# 4.1 Diseño del estudio y lógica temporal por eras", """\
# 4.1 Diseño del estudio y lógica temporal por eras

## Pregunta de investigación

¿El perfil biográfico, educativo y de trayectoria de un diputado predice a qué tipo de comisión es asignado, y ese perfil ha cambiado entre épocas políticas?

---

## Tipología de comisiones

| Tipo | Definición operacional | Implicación política |
|---|---|---|
| Nodal | ≥1 comisión nodal (presupuesto, hacienda, seguridad) | Alta influencia · cargo de confianza del grupo mayoritario |
| Lastre | ≥1 comisión lastre (sin recursos ni dictámenes) | Marginalización · oposición o primíparos sin red |
| Temática | Conteo de comisiones temáticas (0–10) | Especialización · volumen de trabajo legislativo |

---

## Segmentación temporal: tres eras (S3, diputraxv12)

| Época | Legislaturas | Régimen | n | Tasa nodal | Tasa lastre | Media temáticas |
|---|---|---|---:|---:|---:|---:|
| ERA_1 | 57–59 | PRI hegemónico | 1500 | 32.2 % | 42.1 % | 1.53 |
| ERA_2 | 60–62 | Alternancia PAN | 1500 | 40.7 % | 45.9 % | 1.93 |
| ERA_3 | 63–66 | Transición-Morena | 2000 | ~51.0 % | ~47.2 % | ~2.08 |

**Por qué tres eras y no cuatro.** `diputraxv12` sometió la periodización a una búsqueda de la segmentación temporal óptima: seis esquemas candidatos (del modelo único al modelo por legislatura), mismo espacio de features, mismo modelo (LR L1+SFM), misma semilla, evaluados con predicciones *out-of-fold* sobre estratos fijos por legislatura. El esquema S3 —tres cortes de coalición, sin separar la LXVI— quedó en la **frontera eficiente** del intercambio sesgo-varianza:

- máxima cohesión interna de coeficientes de todo el espacio de candidatos (similitud coseno dentro de grupo 0.537; Δ dentro−entre = 0.083, familia de cortes políticos);
- costo predictivo frente al mejor esquema dentro del umbral de empate (±0.02 AUC estratificado);
- elimina el grupo de n=500 (la LXVI aislada), principal fuente de varianza de la periodización en 4 eras de v10.

**Trade-off aceptado (límite de resolución).** Al fusionar 63–66, el cuaderno ya no estima un vector de coeficientes propio para el subrégimen de supermayoría de Morena (2024–). Los hallazgos de v10 que dependían de esa resolución (H5, la convergencia de género de H7) se reformulan en §11 como límites declarados, no como resultados de este cuaderno.

---

## *Feature engineering* — 61 *features*

| Bloque | Variables representativas |
|---|---|
| Político-electoral | `sexo_bin`, `mayoria_relativa`, `es_partido_mayoria`, `legislatura_num` |
| Trayectoria legislativa | `fue_diputado_local`, `fue_diputado_federal`, `fue_senador`, `n_trayectoria_legislativa` |
| Trayectoria administrativa | `n_trayectoria_admin`, `nivel_cargo_max`, `fue_secretario_cargo`, `fue_presidente_mun` |
| Trayectoria política | `n_trayectoria_politica`, `n_trayectoria_empresarial`, `lider_juvenil_partido` |
| Educación | `grado_estudios_ord`, `tiene_posgrado`, `univ_elite`, `estudios_en_extranjero` |
| Dummies | Partido —8 categorías—, Región —6—, Área de formación —3: `area_Derecho`, `C. Políticas y Sociales`, `Económico-Financiera`— |

**Criterio de selección:** se retienen todas las variables de los cinco bloques de trayectoria y las de contexto institucional. Se eliminaron las 10 *dummies* individuales de institución universitaria (reemplazadas por `univ_elite`), las *dummies* de área disciplinaria excedentes, las variables compuestas redundantes (`carrera_depth`, `edu_calidad`, `exp_alta_jerarquia`) y —por AC1 (§4.6)— `univ_extranjera` y `n_cargos_legislativos_prev`.

---

## Lógica del diseño de modelado

Cada uno de los tres *targets* se modela por separado dentro de cada era. Esto permite:

1. Detectar si el perfil que predice la asignación cambia entre épocas —AUC/MAE por era—.
2. Identificar qué *features* ganan o pierden importancia a lo largo del tiempo —SHAP por era—.
3. Medir la transferencia del modelo entre períodos —validación *rolling forward*—.

El modelo empleado es **Regresión Logística L1 —Lasso—**, evaluada en dos variantes:

| Variante | Descripción |
|---|---|
| `LR L1 (full)` | L1 sobre los 61 *features* sin preselección. |
| `LR L1 + SFM` | *Pipeline* `StandardScaler` → `SelectFromModel(LR L1)` → `LR L1`, donde la selección automática identifica el subconjunto mínimo de *features* con peso no nulo por era. |

La penalización L1 produce modelos *sparse*: los coeficientes de *features* irrelevantes convergen a cero durante el entrenamiento, haciendo que la selección forme parte del proceso de ajuste. `SelectFromModel`, con `threshold='mean'`, retiene únicamente las variables cuya magnitud de coeficiente supera la media. La comparación entre `LR L1 (full)` y `LR L1 + SFM` permite cuantificar el costo/beneficio de la selección automática en AUC; la especificación **L2 (v4)** se conserva como línea base de robustez (§8.0b).

---

# 4.2 Guía de interpretación de métricas

## AUC — Nodales y Lastre: clasificación binaria

| Rango | Lectura |
|---|---|
| 0.50 | Aleatorio: el perfil no predice la asignación |
| 0.55–0.65 | Señal débil: factores no observados dominan |
| 0.65–0.75 | Señal moderada: el perfil importa, pero no determina |
| 0.75–0.85 | Señal fuerte: el perfil es factor relevante |
| > 0.85 | Señal muy fuerte: asignación casi determinista |

---

## MAE — Temáticas: regresión Poisson

La métrica se compara contra el *baseline* de predecir siempre la media. Una mejora **≤5 %** es prácticamente nula dada la varianza del *target*.

---

## Preguntas de investigación y *tests* asociados

| Pregunta | Test |
|---|---|
| ¿Cambió el perfil de reclutamiento a nodales? | Tendencia SHAP: si la trayectoria administrativa cae y la carrera legislativa/partidista sube hacia ERA_3 → sí |
| ¿Lastre es imagen espejo de nodales? | Correlación `SHAP(nodal)` vs. `-SHAP(lastre)`: si `r ≈ −1.0` → sí |
| ¿`es_partido_mayoria` domina? | Posición en *heatmap* SHAP |
| ¿Hay ruptura temporal entre eras? | AUC *rolling* ERA_2→ERA_3 vs. ERA_1→ERA_2 |

---

## Notas de calidad

- `grado_estudios_ord` en LIX tiene promedio 1.49 —vs. aproximadamente 4 en otras legislaturas—: probable error de captura que afecta ERA_1.
- 10.2 % de nulos en edad fueron imputados con MICE (§2.2.1c).
- 625 registros son reelecciones válidas: no son *leakage* entre eras; la CV agrupada por `diputado_id` (AC2, §4.6) controla la fuga por reelección.

---

# 4.3 Justificación del parámetro `k = 5` en la validación cruzada

La validación cruzada estratificada de 5 pliegues —*5-fold stratified CV*— se adopta como método único en todas las eras. La elección responde a tres criterios.

---

## 1. Tamaño de *fold* y estabilidad del AUC

Con la segmentación S3 el grupo más pequeño tiene n=1,500 (300 observaciones por *fold* de prueba, ~97 positivos nodales en ERA_1, la era de menor tasa). Todos los *folds* quedan holgadamente por encima del umbral en el que el estimador de AUC se vuelve inestable (Hanley & McNeil, 1982). A diferencia de v10 —donde la ERA_4 de n=500 forzaba la elección—, aquí `k = 5` no está restringido por ningún grupo corto.

---

## 2. Consistencia metodológica entre eras y con las versiones previas

Usar `k` distinto por era introduciría sesgo sistemático en la comparación de AUC entre épocas. Además, mantener `k = 5` con `random_state=42` preserva la comparabilidad directa con v10 (S4) y con la evaluación de segmentaciones de `diputraxv12`, que usan exactamente el mismo protocolo.

---

## 3. Respaldo en la literatura

Kohavi (1995) muestra empíricamente que `k = 5` y `k = 10` producen estimaciones de error prácticamente idénticas en la mayoría de los conjuntos de datos evaluados. James et al. (2021, §5.1.3) presentan ambos como opciones equivalentes. Arlot y Celisse (2010) demuestran que LOO tiene varianza asintótica más alta que *k-fold* para estimadores no lineales. LOO sería además computacionalmente prohibitivo (>27,000 entrenamientos para 3 eras × 3 targets × 2 variantes).
""")

FULL[144] = ("df_roll = pd.DataFrame([", """\
df_roll = (df_roll_n.rename(columns={"AUC": "Nodales AUC"})
           .merge(df_roll_l.rename(columns={"AUC": "Lastre AUC"}), on="Transicion")
           .merge(df_roll_t.rename(columns={"MAE": "Temáticas MAE"}), on="Transicion")
           .rename(columns={"Transicion": "Transición"}))
display(df_roll.set_index("Transición"))
""")

FULL[153] = ("df_prof = pd.DataFrame([", """\
# Tabla comparativa construida desde df_profiles (celda de perfiles prototipicos)
_grade_lbl = {0: 'Sin dato (0)', 1: 'Primaria (1)', 2: 'Secundaria (2)',
              4: 'Preparatoria (4)', 5: 'Lic. incompleta (5)',
              6: 'Licenciatura (6)', 7: 'Especialidad (7)', 9: 'Doctorado (9)'}

def _sn(v):
    return 'Sí' if float(v) >= 1 else 'No'

rows_prof = []
for era in ERA_ORDER:
    p = df_profiles.loc[ERA_LABELS[era]]
    rows_prof.append({
        "Época":                ERA_LABELS[era],
        "Partido mayoría":      _sn(p["es_partido_mayoria"]),
        "Fue diputado federal": _sn(p["fue_diputado_federal"]),
        "Secretario cargo":     _sn(p["fue_secretario_cargo"]),
        "Tray. administrativa": int(p["n_trayectoria_admin"]),
        "Tray. política":       int(p["n_trayectoria_politica"]),
        "Grado estudios":       _grade_lbl.get(int(p["grado_estudios_ord"]),
                                              str(int(p["grado_estudios_ord"]))),
        "Posgrado":             _sn(p["tiene_posgrado"]),
        "Univ. elite":          _sn(p["univ_elite"]),
        "Sexo":                 'Hombre' if float(p["sexo_bin"]) >= 1 else 'Mujer',
        "Edad":                 round(float(p["edad_imp"])),
        "Mayoria relativa":     _sn(p["mayoria_relativa"]),
        "Fue senador":          _sn(p["fue_senador"]),
        "Nivel cargo máx":      int(p["nivel_cargo_max"]),
    })
df_prof = pd.DataFrame(rows_prof).set_index("Época")
display(df_prof.T)
""")

FULL[158] = ("summary = pd.DataFrame([", """\
# Resumen construido desde las tablas de CV ejecutadas (df_nodal_cv, df_lastre_cv, df_tem_cv)
def _mu(cell):
    return float(str(cell).split('+-')[0])

def _senal_auc(a):
    if a >= 0.70:
        return "Moderada"
    if a >= 0.65:
        return "Moderada-debil"
    if a >= 0.60:
        return "Debil"
    return "Muy debil"

rows_sum = []
for _, r in df_nodal_cv.iterrows():
    rows_sum.append({"Target": "Nodales", "Epoca": r["Era"], "Metrica": "AUC",
                     "LR L1 (full)": _mu(r["LR L1 (full)"]),
                     "LR L1 + SFM": _mu(r["LR L1 + SFM"]),
                     "Senal": _senal_auc(_mu(r["LR L1 + SFM"]))})
for _, r in df_lastre_cv.iterrows():
    rows_sum.append({"Target": "Lastre", "Epoca": r["Era"], "Metrica": "AUC",
                     "LR L1 (full)": _mu(r["LR L1 (full)"]),
                     "LR L1 + SFM": _mu(r["LR L1 + SFM"]),
                     "Senal": _senal_auc(_mu(r["LR L1 + SFM"]))})
for _, r in df_tem_cv.iterrows():
    rows_sum.append({"Target": "Tematicas", "Epoca": r["Era"], "Metrica": "MAE",
                     "LR L1 (full)": _mu(r["Poisson GLM"]),
                     "LR L1 + SFM": None, "Senal": "Marginal"})
summary = pd.DataFrame(rows_sum)
print("Nota: LR L1 + SFM tomado de la ejecucion de 5.4/6.3 (Nodales/Lastre). "
      "Para Tematicas la columna 'LR L1 (full)' reporta el MAE del GLM Poisson "
      "(regresion de conteo, sin variante SFM).")
display(summary.set_index(["Target", "Epoca"]))
""")

FULL[160] = ("import numpy as np", """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Hanley-McNeil SE for AUC ---
def hm_se(auc, n1, n0):
    Q1 = auc / (2 - auc)
    Q2 = 2 * auc**2 / (1 + auc)
    var = (auc*(1-auc) + (n1-1)*(Q1 - auc**2) + (n0-1)*(Q2 - auc**2)) / (n1*n0)
    return float(np.sqrt(var))

# --- Parametros por era: n y tasas desde df_enc; AUC del LR L1+SFM ejecutado ---
ERA_PARAMS = {}
for _era in ERA_ORDER:
    _m = df_enc["era"] == _era
    ERA_PARAMS[_era] = {
        "n":         int(_m.sum()),
        "p_nodal":   float(df_enc.loc[_m, "nodal_bin"].mean()),
        "p_lastre":  float(df_enc.loc[_m, "lastre_bin"].mean()),
        "auc_nodal": _parse_auc(df_nodal_cv,  ERA_LABELS[_era], "LR L1 + SFM"),
        "auc_lastre": _parse_auc(df_lastre_cv, ERA_LABELS[_era], "LR L1 + SFM"),
    }

Z_ALPHA2 = 1.96    # alpha=0.05 two-tailed
Z_BETA80 = 0.842   # power=0.80
Z_BETA90 = 1.282   # power=0.90

# --- Build power table ---
rows = []
for era, p in ERA_PARAMS.items():
    n = p["n"]
    for target, key_p, key_auc in [
        ("Nodal",  "p_nodal",  "auc_nodal"),
        ("Lastre", "p_lastre", "auc_lastre"),
    ]:
        n1 = int(n * p[key_p])
        n0 = n - n1
        auc_obs = p[key_auc]
        se_obs  = hm_se(auc_obs, n1, n0)
        se_null = hm_se(0.5, n1, n0)
        mde80   = (Z_ALPHA2 + Z_BETA80) * se_null  # one-sample vs AUC=0.5
        mde90   = (Z_ALPHA2 + Z_BETA90) * se_null
        z_obs   = (auc_obs - 0.5) / se_null
        sig     = "SI" if abs(z_obs) > Z_ALPHA2 else "NO"
        rows.append({
            "Era":       ERA_LABELS[era],
            "Target":    target,
            "n":         n,
            "n_pos":     n1,
            "AUC_obs":   round(auc_obs, 3),
            "SE":        round(se_obs, 4),
            "IC95 +-":   round(Z_ALPHA2 * se_obs, 4),
            "MDE_80%":   round(mde80, 4),
            "MDE_90%":   round(mde90, 4),
            "Umbral_80": round(0.5 + mde80, 3),
            "z_obs":     round(z_obs, 2),
            ">MDE?":     sig,
        })

df_power = pd.DataFrame(rows)

display(
    df_power.style
    .format({"AUC_obs": "{:.3f}", "SE": "{:.4f}", "IC95 +-": "{:.4f}",
             "MDE_80%": "{:.4f}", "MDE_90%": "{:.4f}",
             "Umbral_80": "{:.3f}", "z_obs": "{:.2f}"})
    .set_caption(
        "Analisis de potencia estadistica — AUC por era y target "
        "(Hanley-McNeil 1982; alpha=0.05)"
    )
)
""")

FULL[161] = ("# --- Minimum detectable DELTA-AUC", """\
# --- Minimum detectable DELTA-AUC entre pares de eras ---
# Two-sample test: SE_diff = sqrt(SE_a^2 + SE_b^2)
from itertools import combinations

rows_diff = []
for era_a, era_b in combinations(ERA_ORDER, 2):
    for target, key_p, key_auc in [
        ("Nodal",  "p_nodal",  "auc_nodal"),
        ("Lastre", "p_lastre", "auc_lastre"),
    ]:
        p_a, p_b = ERA_PARAMS[era_a], ERA_PARAMS[era_b]
        n1_a = int(p_a["n"] * p_a[key_p])
        n1_b = int(p_b["n"] * p_b[key_p])
        se_a = hm_se(p_a[key_auc], n1_a, p_a["n"] - n1_a)
        se_b = hm_se(p_b[key_auc], n1_b, p_b["n"] - n1_b)
        se_diff  = float(np.sqrt(se_a**2 + se_b**2))
        mde80    = round((Z_ALPHA2 + Z_BETA80) * se_diff, 4)
        obs_diff = round(abs(p_a[key_auc] - p_b[key_auc]), 4)
        detectable = "SI" if obs_diff >= mde80 else "NO"
        _la = ERA_LABELS[era_a].split(chr(40))[0].strip()
        _lb = ERA_LABELS[era_b].split(chr(40))[0].strip()
        rows_diff.append({
            "Comparacion":        f"{_la} vs {_lb}",
            "Target":             target,
            "delta_AUC_obs":      obs_diff,
            "SE_diff":            round(se_diff, 4),
            "MDE_80pct":          mde80,
            "Detectable_80pct?":  detectable,
        })

df_diff = pd.DataFrame(rows_diff)

def style_detectable(val):
    return "color: green; font-weight: bold" if val == "SI" else "color: red"

display(
    df_diff.style
    .map(style_detectable, subset=["Detectable_80pct?"])
    .format({"delta_AUC_obs": "{:.4f}", "SE_diff": "{:.4f}", "MDE_80pct": "{:.4f}"})
    .set_caption(
        "Diferencias de AUC entre pares de eras — "
        "detectable si |delta| >= MDE al 80% de potencia"
    )
)
""")

# --------------------------------------------------------------------------
# Verificación previa
# --------------------------------------------------------------------------
for idx, old, _new, cnt in REPL:
    n = get(idx).count(old)
    if n != cnt:
        errors.append(f"celda {idx}: patron aparece {n} veces (esperadas {cnt}): {old[:80]!r}")

for idx, (anchor, _text) in FULL.items():
    src = get(idx)
    if not src.lstrip().startswith(anchor):
        errors.append(f"celda {idx}: no empieza con {anchor!r} "
                      f"(empieza con {src.lstrip()[:70]!r})")

if errors:
    print("ABORT — diputraxv10.ipynb no coincide con lo esperado:")
    for e in errors:
        print("  -", e)
    sys.exit(1)

# --------------------------------------------------------------------------
# Aplicación
# --------------------------------------------------------------------------
for idx, old, new, _cnt in REPL:
    put(idx, get(idx).replace(old, new))
for idx in sorted({i for i, *_ in REPL}):
    print(f"celda {idx:3d}: reemplazos aplicados")

for idx, (_anchor, text) in sorted(FULL.items()):
    put(idx, text.rstrip("\n"))
    print(f"celda {idx:3d}: reescrita")

# Eliminar sección 12 (PyTorch MTL, celdas 203-255)
first12 = "".join(cells[203]["source"] if isinstance(cells[203]["source"], list)
                  else [cells[203]["source"]])
assert first12.lstrip().startswith("# 12. Enfoque 3"), \
    f"celda 203 no es el inicio de la sección 12: {first12[:60]!r}"
del cells[203:256]
print(f"sección 12 eliminada: quedan {len(cells)} celdas")

# Limpiar outputs y execution_count
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

nb["cells"] = cells
# Kernel del proyecto (.venv) — el "python3" de v10 resuelve al Python de
# anaconda, cuyo statsmodels es incompatible con su scipy.
nb["metadata"]["kernelspec"] = {
    "display_name": "diputrax", "language": "python", "name": "diputrax",
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nOK -> {OUT}  ({len(cells)} celdas)")
