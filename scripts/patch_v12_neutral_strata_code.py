# -*- coding: utf-8 -*-
"""Patch diputraxv12: evaluación neutral por estrato-legislatura (plan12.md, fase 1).

Reemplaza las tablas/figuras que usaban las 4 eras como estratos fijos por la
versión estratificada por legislatura (refinamiento común más fino de los 6
esquemas), y actualiza el markdown de diseño/justificación (§4, §5) y las guías
de lectura de figuras. Las interpretaciones con cifras (§6.1, §7.1, §9) se
parchan en fase 2, después de re-ejecutar el cuaderno.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "diputraxv12.ipynb"
nb = json.load(open(NB, encoding="utf-8"))


def set_src(idx, anchor, text):
    cell = nb["cells"][idx]
    src = "".join(cell["source"])
    assert src.startswith(anchor), f"celda {idx}: ancla no coincide -> {src[:80]!r}"
    cell["source"] = text.splitlines(keepends=True)


# ---------------------------------------------------------------- celda 7 ----
set_src(7, "**Lectura — tasas base de los targets por era.**", """\
**Lectura — tasas base de los targets por era (vista descriptiva).** La tabla muestra la prevalencia de cada target dentro de cada era. La proporción de diputados con **comisión nodal** sube de forma monótona (0.322 → 0.407 → 0.495 → 0.554): acceder a una comisión estratégica se ha vuelto progresivamente más común. El **lastre** se mantiene en torno a ~0.42–0.50 y cae a 0.392 en ERA 4 (Morena reparte menos comisiones de bajo perfil). Las **temáticas** (conteo medio) alcanzan su máximo en ERA 3 (2.146).

*Relevancia para el análisis.* Estas tasas base tan distintas son la primera evidencia de que los periodos son **poblaciones heterogéneas**, no cortes arbitrarios del tiempo. También son la razón técnica por la que la evaluación posterior debe estratificarse: un modelo agrupado podría inflar su AUC solo por explotar estas diferencias de prevalencia entre periodos (§4, §5.2), sin aprender nada del diputado individual. Una precisión importante: aquí las 4 eras se usan solo como **lente descriptiva** — son la partición del candidato S4, uno de los seis esquemas bajo prueba, y por eso la evaluación comparativa de §4 en adelante se estratifica por **legislatura** (la unidad neutral común a todos los esquemas), no por era.
""")

# --------------------------------------------------------------- celda 14 ----
set_src(14, "# 4. Diseño de evaluación comparable entre esquemas", """\
# 4. Diseño de evaluación comparable entre esquemas

**El problema (i): comparabilidad.** El AUC de validación cruzada *dentro de cada grupo* no es comparable entre esquemas: cambia la n, cambia la mezcla de poblaciones y, sobre todo, un modelo agrupado puede ganar AUC "gratis" al distinguir entre periodos con tasas base distintas (predecir que un diputado de la LXVI tiene comisión nodal es fácil si la tasa de la LXVI es 0.55 y la global 0.42, sin saber nada del diputado). La solución es evaluar sobre **estratos fijos, idénticos para todos los esquemas**.

**El problema (ii): neutralidad de los estratos.** Los estratos fijos no pueden ser las 4 eras canónicas, porque las eras son exactamente la partición de uno de los candidatos (S4). Usarlas como vara sesga la comparación en un sentido técnico preciso: estratificar por era elimina el crédito por tasas base *entre* eras —justo en las fronteras de S4— pero deja intacto el crédito por diferencias de prevalencia *entre legislaturas dentro de una misma era*, que un esquema más fino que las eras (S5, S6) sí puede explotar. La vara neutral es el **estrato-legislatura**: la legislatura es la unidad mínima de asignación de los 6 esquemas (cada grupo de cada esquema es una unión de legislaturas), es decir, el *refinamiento común más fino* de todos los candidatos. Dentro de un estrato-legislatura no queda periodo alguno que distinguir —ningún esquema puede ganar crédito por tasas base— y las fronteras de los estratos no coinciden con las de ningún candidato en particular.

**El protocolo.** Para cada esquema:

1. Dentro de cada grupo se generan **predicciones *out-of-fold*** con `cross_val_predict` (StratifiedKFold *k*=5, `shuffle=True`, `random_state=42`) — cada diputado recibe una probabilidad predicha por un modelo que **nunca lo vio en entrenamiento**.
2. Al terminar, **los 5,000 diputados tienen una predicción OOF** bajo cada esquema.
3. Las métricas se calculan sobre **los 10 estratos-legislatura fijos, idénticos para todos los esquemas**:
   - **AUC por estrato-legislatura** y su **promedio ponderado por n** (`Pond. legis`) — métrica principal;
   - AUC global (se reporta pero se interpreta con cautela por el efecto de tasas base descrito arriba);
   - la misma ponderación con el modelo LR L1 completo (sin `SelectFromModel`) como columna de robustez.

Evaluar dentro de estratos-legislatura elimina el crédito por tasas base entre periodos **para todos los esquemas por igual y sin privilegiar las fronteras de ningún candidato**: lo único que puede subir `Pond. legis` es ordenar mejor a los diputados *dentro* de una misma legislatura. Es la comparación honesta y neutral entre agrupaciones.
""")

# --------------------------------------------------------------- celda 15 ----
set_src(15, "# 5. Justificación metodológica", """\
# 5. Justificación metodológica — ¿por qué estos métodos son útiles y relevantes?

Antes de leer los resultados conviene explicitar por qué el instrumental de este cuaderno —validación cruzada *out-of-fold*, evaluación sobre estratos fijos y neutrales, AUC y MAE, un *baseline* sin features, y la similitud coseno de coeficientes— es el adecuado para la pregunta planteada en §1.4. La pregunta no es "¿qué modelo predice mejor?" sino "**¿qué agrupación de legislaturas sirve mejor al análisis?**", y esa diferencia dicta cada elección metodológica.

## 5.1 Validación cruzada *out-of-fold* (OOF): honestidad predictiva

Ajustar un modelo y evaluarlo sobre las mismas filas premia el **sobreajuste**: con 61 features y grupos de n≈500, un modelo puede memorizar ruido y exhibir un AUC engañosamente alto. La validación cruzada *k*=5 rompe ese círculo —cada diputado recibe una predicción de un modelo que **nunca lo vio en entrenamiento**— de modo que el desempeño reportado estima la capacidad de **generalización**, no de memorización. Es relevante aquí porque los esquemas finos (S6, n≈500) son precisamente los más expuestos al sobreajuste: sin OOF, S6 parecería competitivo por una razón espuria. Fijar `shuffle=True, random_state=42` en todas las particiones garantiza que las diferencias entre esquemas provengan del **esquema**, no de un reparto de folds distinto.

## 5.2 Estratos fijos y neutrales: la única comparación honesta entre esquemas

El AUC calculado *dentro de cada grupo* no es comparable entre esquemas porque cada esquema define grupos distintos, con distinta n y —crítico— distinta **tasa base**. Un modelo agrupado puede subir su AUC "gratis" solo por distinguir periodos con prevalencias diferentes, sin aprender nada sobre el diputado individual (§4). Recomponer las predicciones OOF y evaluarlas siempre sobre **los mismos 10 estratos-legislatura** neutraliza ese crédito por tasas base **para todos los esquemas por igual**: lo único que puede mover el AUC estratificado es ordenar mejor a los diputados *dentro* de una misma legislatura. La elección de la legislatura como estrato no es estética sino de **neutralidad**: es el refinamiento común más fino de las 6 particiones candidatas, de modo que la vara no coincide con las fronteras de ningún candidato — estratificar por era, en cambio, removería el crédito por tasas base exactamente en las fronteras de S4 y dejaría a los esquemas más finos el crédito intra-era, sesgando la comparación (§4).

## 5.3 AUC para lo binario, MAE para el conteo: cada métrica a su escala

- **AUC (targets binarios `nodal`/`lastre`).** El AUC mide la probabilidad de que el modelo asigne mayor score a un positivo que a un negativo tomados al azar: 0.5 = azar, 1.0 = perfecto. Su virtud decisiva es que es **invariante a la tasa base**, por lo que estratos con prevalencias distintas (≈0.32 en las primeras legislaturas, 0.554 en la LXVI) son directamente comparables y no premia simplemente predecir la clase mayoritaria —esencial con clases desbalanceadas y con `class_weight='balanced'`.
- **MAE (conteo `n_comisiones_tematicas`).** Para un conteo, el error absoluto medio está **en las unidades del fenómeno** (comisiones): un MAE de 0.82 significa "erramos por ~0.82 comisiones en promedio". Es interpretable, comparable entre esquemas (misma escala, mismos estratos) y no sufre la distorsión de tasa base del AUC.

## 5.4 *Baseline* sin features: la vara del valor añadido

Una métrica en abstracto no dice si el modelo **aporta**. Por eso las temáticas se contrastan contra un *baseline* que predice la media del estrato-legislatura **sin usar ninguna feature**. Si los modelos no baten esa media, el perfil biográfico es irrelevante para el target, con independencia del esquema. El *baseline* convierte una cifra de MAE en un juicio: cuánta señal hay realmente. (Nótese que el baseline por legislatura es más exigente que uno por era: al ser el estrato más fino, su media local captura más variación temporal.)

## 5.5 Similitud coseno de coeficientes: desempeño ≠ estructura

El AUC responde *cuánto* predice cada esquema, pero no *si corta donde de verdad cambia el mecanismo* de asignación. Un esquema puede predecir bien y aun así trazar fronteras en el lugar equivocado. Para esa pregunta estructural se ajusta una LR L1 **por legislatura** y se compara la dirección de sus vectores de coeficientes con la **similitud coseno** (1 = misma lógica de asignación, 0 = ortogonal). Si una periodización es sustantiva, la similitud debe ser mayor *dentro* de sus grupos que *entre* ellos (estructura de bloques) — un contraste que se calcula por igual para cada esquema, de modo que también esta prueba es neutral entre candidatos. Esta métrica es relevante porque valida una periodización por su **contenido interpretativo** —el que sostiene los hallazgos H4/H5/H7 de la tesis—, no solo por su exactitud.

## 5.6 Regularización L1 + `SelectFromModel` y reproducibilidad

La penalización **L1 (Lasso)** lleva a cero los coeficientes irrelevantes: produce modelos parsimoniosos, estables con n pequeña y con coeficientes **comparables posición a posición** entre legislaturas (requisito de §5.5). `SelectFromModel` replica el pipeline principal de v10 (Tabla 7), de modo que la comparación de esquemas se hace sobre **exactamente el mismo modelo** que usa la tesis —no sobre un sustituto—, lo que hace las conclusiones trasladables. Mantener features, semilla y modelo constantes, variando **solo** qué filas entran juntas al entrenamiento, aísla el efecto de la agrupación: es un experimento controlado, no una comparación de modelos distintos.

**En conjunto**, estos métodos convierten la pregunta "¿son las 4 eras la mejor partición?" en una prueba falsable, comparable y reproducible: OOF elimina el optimismo, la estratificación por legislatura elimina el crédito por tasas base sin privilegiar a ningún candidato, AUC/MAE miden en la escala correcta, el *baseline* fija la vara del valor añadido, y la similitud coseno separa desempeño de validez estructural.
""")

# --------------------------------------------------------------- celda 18 ----
set_src(18, "# --- Tablas de resultados: AUC por estrato-era y ponderados ---", """\
# --- Tablas de resultados: AUC por estrato-legislatura y ponderados ---
def build_results(target):
    rows, rows_leg = [], []
    for sch in SCHEME_ORDER:
        per_leg, w_leg = strat_auc(OOF[(sch, target, "sfm")], target, "legis_str")
        _, w_leg_l1 = strat_auc(OOF[(sch, target, "l1")], target, "legis_str")
        glob = roc_auc_score(df_enc[target], OOF[(sch, target, "sfm")])
        rows.append({
            "Esquema": SCHEME_SHORT[sch],
            "Pond. legis": w_leg, "Global": glob,
            "Pond. legis (L1 full)": w_leg_l1,
        })
        rows_leg.append({
            "Esquema": SCHEME_SHORT[sch],
            **{f"LEG {l}": per_leg.get(str(l), np.nan) for l in LEGIS},
        })
    return (pd.DataFrame(rows).set_index("Esquema"),
            pd.DataFrame(rows_leg).set_index("Esquema"))

res_nodal, res_nodal_leg = build_results("nodal_bin")
res_lastre, res_lastre_leg = build_results("lastre_bin")

print("NODAL — AUC out-of-fold estratificado por legislatura (modelo LR L1+SFM)")
display(res_nodal.round(3))
print("NODAL — desagregado: AUC por estrato-legislatura (SFM)")
display(res_nodal_leg.round(3))
print("\\nLASTRE — AUC out-of-fold estratificado por legislatura (modelo LR L1+SFM)")
display(res_lastre.round(3))
print("LASTRE — desagregado: AUC por estrato-legislatura (SFM)")
display(res_lastre_leg.round(3))
""")

# --------------------------------------------------------------- celda 19 ----
set_src(19, "**Lectura — AUC *out-of-fold* estratificado (métrica principal).**", """\
**Lectura — AUC *out-of-fold* estratificado por legislatura (métrica principal).** Cada fila es un esquema. **`Pond. legis`** es el promedio, ponderado por n, del AUC calculado dentro de cada uno de los 10 estratos-legislatura fijos: la cifra-titular de este cuaderno, y la única **neutral** frente a los 6 candidatos, porque el estrato-legislatura es el refinamiento común más fino de todas las particiones (§4). La tabla desagregada (LEG 57–66) muestra en qué legislaturas se localizan las diferencias entre esquemas. Recordatorio de escala: AUC 0.5 = azar, 1.0 = discriminación perfecta; al ser **invariante a la tasa base**, es comparable entre estratos con prevalencias muy distintas (§5.3).

La columna **`Global`** aparece inflada respecto a `Pond. legis`: esa brecha es exactamente el **crédito por tasas base** entre periodos que la estratificación elimina (§4), y por eso `Pond. legis` —no `Global`— es la métrica válida para comparar esquemas. La última columna (LR L1 completo, sin SFM) sirve de **robustez**: si el ordenamiento se conserva, no depende del selector de features. La síntesis está en §6.1.
""")

# --------------------------------------------------------------- celda 23 ----
set_src(23, "# --- Figura A: AUC ponderado por eras, por esquema (nodal | lastre) ---", """\
# --- Figura A: AUC ponderado por legislaturas, por esquema (nodal | lastre) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, res, ttl in [(axes[0], res_nodal, "Nodal"), (axes[1], res_lastre, "Lastre")]:
    vals = res["Pond. legis"].values
    ypos = np.arange(len(SCHEME_ORDER))[::-1]
    bars = ax.barh(ypos, vals, height=0.62, color=SCHEME_COLORS, alpha=0.92)
    for yp, v in zip(ypos, vals):
        ax.text(v + 0.004, yp, f"{v:.3f}", va="center", fontsize=10, color=TXT)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SCHEME_SHORT[s] for s in SCHEME_ORDER], fontsize=10)
    ax.set_xlim(0.5, max(vals) + 0.05)
    ax.axvline(0.5, color="#999999", lw=1, ls="--")
    ax.set_xlabel("AUC out-of-fold, promedio ponderado por estrato-legislatura")
    ax.set_title(f"{ttl} — AUC estratificado por esquema de agrupación",
                 fontsize=12, fontweight="bold", color=TXT)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
plt.tight_layout()
plt.show()
""")

# --------------------------------------------------------------- celda 24 ----
set_src(24, "**Cómo leer la Figura A.**", """\
**Cómo leer la Figura A.** Barras horizontales = AUC ponderado por estrato-legislatura (`Pond. legis`) de cada esquema; la línea discontinua en 0.5 marca el azar. Para **nodal** las barras se agolpan en una banda estrecha: visualmente, la elección de agrupación apenas mueve la discriminación agregada. Para **lastre** las barras están más abajo y aún más planas, confirmando que ese target es poco separable. El mensaje de la figura es la **ausencia de una brecha material** entre los esquemas gruesos y la periodización de la tesis (S4); el único claramente rezagado es S6.
""")

# --------------------------------------------------------------- celda 25 ----
set_src(25, "# --- Figura B: AUC por estrato-era, una línea por esquema ---", """\
# --- Figura B: AUC por estrato-legislatura, una línea por esquema ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
xt = np.arange(len(LEGIS))
for ax, res_leg, ttl in [(axes[0], res_nodal_leg, "Nodal"),
                         (axes[1], res_lastre_leg, "Lastre")]:
    for i, sch in enumerate(SCHEME_ORDER):
        y = [res_leg.loc[SCHEME_SHORT[sch], f"LEG {l}"] for l in LEGIS]
        ax.plot(xt, y, marker="o", ms=5, lw=1.8, alpha=0.85,
                color=SCHEME_COLORS[i], label=SCHEME_SHORT[sch])
    ax.set_xticks(xt)
    ax.set_xticklabels([str(l) for l in LEGIS], fontsize=9)
    ax.axhline(0.5, color="#999999", lw=1, ls="--")
    ax.set_xlabel("Legislatura")
    ax.set_ylabel("AUC out-of-fold en el estrato")
    ax.set_title(f"{ttl} — AUC por estrato-legislatura y esquema",
                 fontsize=12, fontweight="bold", color=TXT)
axes[0].legend(fontsize=8.5, loc="lower left", frameon=True)
plt.tight_layout()
plt.show()
""")

# --------------------------------------------------------------- celda 26 ----
set_src(26, "**Cómo leer la Figura B.**", """\
**Cómo leer la Figura B.** Cada línea es un esquema recorriendo los 10 estratos-legislatura (57–66); ningún esquema va resaltado — la figura compara a los seis candidatos en pie de igualdad. Donde las líneas se **superponen**, la agrupación es irrelevante: todos ordenan igual de bien a los diputados de esa legislatura. Donde se **abren en abanico**, los esquemas difieren; en particular, las legislaturas donde los esquemas de grupo pequeño (S6, y S4 en la LXVI) caen respecto a los gruesos delatan la penalización por n reducida. La lectura completa del patrón está en §6.1.
""")

# --------------------------------------------------------------- celda 29 ----
set_src(29, "# --- OOF Poisson por esquema + tabla ---", """\
# --- OOF Poisson por esquema + tabla ---
rows = []
for sch in SCHEME_ORDER:
    oof = oof_predict(SCHEMES[sch], "n_comisiones_tematicas", lr_poisson,
                      is_binary=False)
    per_leg, w_leg = strat_mae(oof, "n_comisiones_tematicas", "legis_str")
    rows.append({
        "Esquema": SCHEME_SHORT[sch],
        **{f"LEG {l}": per_leg.get(str(l), np.nan) for l in LEGIS},
        "Pond. legis": w_leg,
    })
res_tem = pd.DataFrame(rows).set_index("Esquema")

# Baseline: predecir la media del estrato-legislatura (sin features)
base_rows, base_ns = {}, []
for l in LEGIS:
    m = df_enc["legislatura_num"] == l
    y = df_enc.loc[m, "n_comisiones_tematicas"]
    base_rows[f"LEG {l}"] = float(np.abs(y - y.mean()).mean())
    base_ns.append(int(m.sum()))
base_w = float(np.average(list(base_rows.values()), weights=base_ns))
res_tem.loc["Baseline (media por legislatura)"] = {**base_rows,
                                                   "Pond. legis": base_w}

print("TEMÁTICAS — MAE out-of-fold por estrato-legislatura (GLM Poisson; menor = mejor)")
display(res_tem.round(3))
""")

# --------------------------------------------------------------- celda 31 ----
set_src(31, "# --- Figura C: MAE ponderado por esquema ---", """\
# --- Figura C: MAE ponderado por esquema ---
fig, ax = plt.subplots(figsize=(8.5, 4.6))
vals = res_tem.loc[[SCHEME_SHORT[s] for s in SCHEME_ORDER], "Pond. legis"].values
ypos = np.arange(len(SCHEME_ORDER))[::-1]
ax.barh(ypos, vals, height=0.62, color=SCHEME_COLORS, alpha=0.92)
for yp, v in zip(ypos, vals):
    ax.text(v + 0.003, yp, f"{v:.3f}", va="center", fontsize=10, color=TXT)
ax.axvline(res_tem.loc["Baseline (media por legislatura)", "Pond. legis"],
           color="#999999", lw=1.4, ls="--",
           label="Baseline (media por legislatura)")
ax.set_yticks(ypos)
ax.set_yticklabels([SCHEME_SHORT[s] for s in SCHEME_ORDER], fontsize=10)
ax.set_xlabel("MAE out-of-fold ponderado por estrato-legislatura (menor = mejor)")
ax.set_title("Temáticas — MAE por esquema de agrupación (GLM Poisson)",
             fontsize=12, fontweight="bold", color=TXT)
ax.set_xlim(min(vals) - 0.05, max(max(vals), base_w) + 0.05)
ax.legend(fontsize=9, loc="lower right", frameon=True)
ax.grid(axis="y", visible=False)
sns.despine(ax=ax, left=True)
plt.tight_layout()
plt.show()
""")

# --------------------------------------------------------------- celda 32 ----
set_src(32, "**Cómo leer la Figura C.**", """\
**Cómo leer la Figura C.** Barras = MAE ponderado por estrato-legislatura de cada esquema; la línea discontinua es el *baseline* de "predecir la media de la legislatura" sin usar features. Todas las barras se agolpan apenas a la izquierda del baseline y entre sí: mejora **marginal y plana**. La figura hace visible de un vistazo la conclusión de §7.1 — para las temáticas, ningún esquema de agrupación mueve la aguja.
""")

# ------------------------------------------- celda 35: solo título heatmap ----
c35 = "".join(nb["cells"][35]["source"])
old_t = '"líneas blancas = fronteras de las 4 eras de la tesis"'
new_t = '"líneas blancas = fronteras del esquema S4 (solo referencia)"'
assert old_t in c35, "celda 35: título del heatmap no encontrado"
nb["cells"][35]["source"] = c35.replace(old_t, new_t).splitlines(keepends=True)

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("OK — fase 1 aplicada:", NB)
