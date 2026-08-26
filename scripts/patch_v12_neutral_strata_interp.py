# -*- coding: utf-8 -*-
"""Patch diputraxv12: interpretaciones con cifras de la corrida neutral (plan12.md, fase 2).

Reescribe las celdas de lectura/interpretación (21, 27, 30, 33, 36, 38, 39, 40)
con los números reales de la ejecución estratificada por legislatura. Incluye la
corrección honesta de §8: en esta corrida S2 maximiza Δ (0.086) y S4 queda en
0.073 — el orden fino S2–S4 no es estable entre corridas y la narrativa deja de
apoyarse en él.
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


# --------------------------------------------------------------- celda 21 ----
set_src(21, "**Lectura — AUC dentro de grupo (vista *naive*, no comparable).**", """\
**Lectura — AUC dentro de grupo (vista *naive*, no comparable).** Esta tabla reporta el AUC de validación cruzada 5-fold calculado **por separado dentro de cada grupo** de cada esquema. Es la vista intuitiva ("qué tan bien predice cada modelo en su propio grupo"), pero **no es comparable entre esquemas** porque cambian la n, la mezcla de poblaciones y el crédito por tasas base (§4, §5.2) — de ahí la tabla estratificada anterior.

*Para qué sirve entonces.* Como diagnóstico, revela dos cosas que la métrica ponderada esconde: (1) la **inestabilidad de la granularidad fina** — los grupos de S6 oscilan de 0.611 (LEG 65, ±0.018) a 0.739 (LEG 57, ±0.056), con LEG 58 en 0.641±0.094, pura varianza muestral; y ERA_4 con n=500 llega solo a 0.626±0.058 —; y (2) la **heterogeneidad interna de los grupos gruesos** — el `POST_2018` de S2 rinde 0.687 dentro de grupo frente a 0.732 del `PRE_2018`, señal de que el corte de 2018 mete regímenes distintos en una misma bolsa. Ambos hallazgos alimentan la interpretación de §6.1.
""")

# --------------------------------------------------------------- celda 27 ----
set_src(27, "## 6.1 Interpretación — targets binarios", """\
## 6.1 Interpretación — targets binarios

**Panorama.** El AUC ponderado por estrato-legislatura (`Pond. legis`, LR L1+SFM) ordena así los esquemas:

| Esquema | Nodal | Lastre |
|---|---:|---:|
| S2 — Corte 2018 | **0.714** | **0.610** |
| S1 — Pooled | 0.713 | 0.604 |
| S3 — Fusión T+M | 0.704 | 0.601 |
| S4 — 4 eras (tesis) | 0.700 | 0.591 |
| S5 — Pares uniformes | 0.696 | 0.592 |
| S6 — Por legislatura | 0.671 | 0.586 |

Cuatro lecturas:

**1. La brecha entre esquemas gruesos y la periodización de la tesis es marginal y se concentra al final de la serie.** La diferencia S2−S4 en nodal es de 0.014 puntos de AUC ponderado — un orden de magnitud por debajo del MDE inter-grupo (≈0.08) del análisis de potencia de v10 §10.2. El desagregado por legislatura muestra que en las legislaturas 57–62 los modelos por era rinden a la par del pooled (diferencias dentro de ±0.02; en la LEG 59, S4 incluso gana: 0.730 vs. 0.706). **La ventaja de los esquemas gruesos se concentra en las legislaturas 63–66**, y sobre todo en la LXVI: nodal 0.683 (S1) vs. 0.624 (S4); lastre 0.587 vs. 0.519. Con n=500, el modelo entrenado solo en la LXVI es el más ruidoso de la serie, y los esquemas gruesos lo compensan tomando fuerza prestada de las otras 4,500 observaciones. Este resultado es coherente con el *rolling forward* de v10 (§8.2), donde el modelo entrenado en ERA 3 predice ERA 4 mejor que el modelo propio de ERA 4.

**2. P2 se confirma: más granularidad no ayuda.** El esquema por legislatura (S6) es el peor en ambos targets (nodal 0.671, lastre 0.586) y en casi todos los estratos. La ganancia por homogeneidad no compensa el costo muestral de bajar a n≈500 por modelo. La vista *naive* (tabla anterior) lo corrobora: los AUC por legislatura oscilan de 0.611 a 0.739 con desviaciones de hasta ±0.094 — inestabilidad pura.

**3. A granularidad comparable, la ventaja predictiva de las fronteras sustantivas es pequeña.** S4 (4 eras, n=500–1,500) supera a S5 (5 pares uniformes, n=1,000 fijo) en nodal por solo 0.004 (0.700 vs. 0.696) y empata en lastre (0.591 vs. 0.592). Bajo la vara neutral, el AUC **no** distingue con claridad los cortes políticos de las ventanas arbitrarias a este nivel de granularidad; esa distinción la aporta el análisis estructural de coeficientes (§8), no el desempeño predictivo.

**4. El corte único en 2018 es el mejor esquema compacto en desempeño — pero su grupo posterior es internamente heterogéneo.** S2 encabeza ambos targets (0.714/0.610), apenas por encima del pooled. Su vista *naive* muestra, sin embargo, que el grupo POST_2018 (LXIV–LXVI, n=1,500) alcanza solo 0.687 dentro de grupo frente a 0.732 del PRE_2018: el corte en 2018 junta el final de la Transición con la LXVI de mayoría calificada de Morena, dos regímenes cuya distinción se examina en la sección 8.

**Nota sobre el AUC global.** Para S1 el AUC global (0.729) supera al estratificado (0.713): esa diferencia es exactamente el crédito por tasas base entre periodos descrito en §4, y es la razón por la que la comparación honesta entre esquemas debe hacerse dentro de estratos fijos por legislatura. La columna de robustez (LR L1 completo, sin SFM) replica el ordenamiento en ambos targets (S2/S1 arriba con 0.715, S6 abajo con 0.673).
""")

# --------------------------------------------------------------- celda 30 ----
set_src(30, "**Lectura — MAE de comisiones temáticas (GLM Poisson).**", """\
**Lectura — MAE de comisiones temáticas (GLM Poisson).** El MAE está en la **escala del conteo**: un valor de ~0.82 significa que la predicción se aleja, en promedio, ~0.82 comisiones del valor real (menor = mejor). A diferencia del AUC, es directamente comparable entre esquemas porque comparte escala y los mismos 10 estratos-legislatura, sin distorsión por tasa base. La fila **`Baseline (media por legislatura)`** predice la media del estrato **sin usar features** y fija la vara del valor añadido (§5.4); al ser la media del estrato más fino, es una vara más exigente que una media por era.

*Qué dicen los números.* Todos los esquemas caen en 0.815–0.819, apenas por debajo del baseline (0.838): los features aportan solo ~2–3% de mejora, y la **agrupación es indiferente** para este target. Es la evidencia cuantitativa de que el volumen de comisiones temáticas responde a una lógica distributiva que el perfil biográfico no captura (síntesis en §7.1).
""")

# --------------------------------------------------------------- celda 33 ----
set_src(33, "## 7.1 Interpretación — temáticas", """\
## 7.1 Interpretación — temáticas

El MAE ponderado es **plano entre esquemas**: de 0.815 (S2) a 0.819 (S1 y S4), una banda de 0.004 comisiones — ruido. Ningún esquema mejora el baseline de "predecir la media de la legislatura" (0.838) en más de ~2.7%, y en la LEG 64 el baseline incluso supera a los seis esquemas (0.742 vs. 0.767–0.795). La elección de agrupación es **irrelevante** para este target, lo que refuerza el hallazgo H3 de la tesis: el volumen de comisiones temáticas es una asignación distributiva/administrativa que el perfil biográfico no captura, se agrupe como se agrupe. Este target no discrimina entre periodizaciones y no aporta evidencia ni a favor ni en contra de las 4 eras.
""")

# --------------------------------------------------------------- celda 36 ----
set_src(36, "**Lectura — heatmap de similitud coseno de coeficientes.**", """\
**Lectura — heatmap de similitud coseno de coeficientes.** Cada celda es la similitud coseno (0–1) entre los vectores de 61 coeficientes de la LR L1 ajustada en dos legislaturas: **1 = misma lógica de asignación**, 0 = ortogonal. Las líneas blancas marcan las fronteras del esquema S4 y son **solo una referencia visual** — la estructura de bloques de cada esquema se cuantifica de forma neutral en la tabla siguiente, que calcula la similitud dentro vs. entre grupos para los seis candidatos por igual. *Advertencia de estabilidad:* los coeficientes L1 ajustados con n≈500 por legislatura tienen varianza alta, y los valores finos de la matriz varían entre corridas/entornos; las comparaciones celda a celda no deben sobreinterpretarse.

*Relevancia para el análisis.* Este diagnóstico responde una pregunta que el AUC **no puede** contestar: no *cuánto* predice cada esquema, sino **si sus fronteras caen donde de verdad cambia el mecanismo** de asignación. Opera sobre los **coeficientes** (la lógica del modelo), no sobre las predicciones (su desempeño); por eso es el instrumento que valida una periodización por su contenido interpretativo (§5.5).
""")

# --------------------------------------------------------------- celda 38 ----
set_src(38, "**Lectura — cohesión de coeficientes (Δ = dentro − entre).**", """\
**Lectura — cohesión de coeficientes (Δ = dentro − entre).** Para cada esquema se promedia la similitud coseno **dentro** de sus grupos y **entre** grupos; **Δ** es su diferencia. Un Δ grande significa que la partición traza sus fronteras justo donde la lógica de coeficientes cambia (bloques internamente parecidos, contraste alto hacia afuera). S1 y S6 muestran `NaN` porque, por construcción, no tienen pares entre-grupos (un solo grupo) o dentro-de-grupo (grupos de una legislatura), respectivamente.

*Cómo leerla con cautela.* En esta corrida, S2 alcanza el Δ máximo (0.086), seguido de cerca por S3 (0.083) y S4 (0.073); S5 queda claramente abajo (0.060). La banda S2–S4 es estrecha y su orden fino **no es estable entre corridas**: una ejecución previa de este mismo cuaderno, con un entorno ligeramente distinto, colocaba a S4 primero (0.089). Lo robusto no es el ranking dentro de la banda, sino el **contraste entre los esquemas con fronteras políticas (Δ ≥ 0.073) y las ventanas uniformes ciegas a la política (0.060)**. La interpretación está en §8.1.
""")

# --------------------------------------------------------------- celda 39 ----
set_src(39, "## 8.1 Interpretación — heterogeneidad de coeficientes", """\
## 8.1 Interpretación — heterogeneidad de coeficientes

Este análisis pregunta algo distinto al AUC: no *cuánto* predice cada esquema, sino **si sus fronteras cortan donde de verdad cambia la lógica de asignación**. El indicador Δ (similitud media dentro de grupo − entre grupos) mide la calidad estructural de cada partición:

| Esquema | Sim. dentro | Sim. entre | Δ |
|---|---:|---:|---:|
| S2 — Corte 2018 | 0.516 | 0.430 | **0.086** |
| S3 — Fusión T+M | **0.537** | 0.454 | 0.083 |
| S4 — 4 eras (tesis) | 0.535 | 0.462 | 0.073 |
| S5 — Pares uniformes | 0.530 | 0.470 | 0.060 |

Tres lecturas:

- **Los cortes políticos superan a las ventanas arbitrarias.** Los tres esquemas cuyas fronteras siguen cambios de coalición dominante (S2, S3, S4) se agrupan en Δ 0.073–0.086, por encima de las ventanas uniformes de S5 (0.060), que parten los bloques reales por la mitad. Este contraste — no el orden fino dentro de la banda política — es el hallazgo estructural robusto.
- **2018 es la frontera individual más fuerte.** El corte único de S2 obtiene en esta corrida el mayor Δ (0.086): separar el periodo pre- y post-Morena es donde más cambia la lógica de coeficientes. S4 conserva una cohesión interna alta (0.535, al nivel de S3) y un Δ de 0.073; su ventaja sobre S2 no es estructural-estadística sino interpretativa (§9).
- **El orden fino S2–S4 es frágil.** Los coeficientes L1 por legislatura (n≈500) varían entre corridas; una ejecución previa de este cuaderno arrojaba S4 primero (Δ=0.089, con S3 en 0.057). Ninguna conclusión debe descansar en qué esquema de la banda política queda primero; sí es estable que S5 queda último y que la frontera de 2018 aparece siempre entre las más nítidas.

La similitud media global entre los 10 vectores de coeficientes es 0.476 — lejos de 1.0: la lógica de asignación nodal **no** es estable a lo largo de las diez legislaturas, y por eso el modelo único (S1), aunque predice bien en promedio, estima un mecanismo que no corresponde a ningún periodo en particular.
""")

# --------------------------------------------------------------- celda 40 ----
set_src(40, "# 9. Conclusiones", """\
# 9. Conclusiones

## 9.1 Veredicto sobre las predicciones de §1.4

**P1 (contra agrupaciones más gruesas) — se cumple en las legislaturas 57–62, con un matiz en 63–66.** En los estratos-legislatura del periodo 1997–2015 los modelos por era rinden a la par del pooled y del corte 2018 (diferencias dentro de ±0.02 de AUC, muy por debajo del MDE≈0.08; en la LEG 59, S4 incluso gana). La ventaja de los esquemas gruesos se concentra al final de la serie y sobre todo en la LXVI (nodal 0.683 vs. 0.624; lastre 0.587 vs. 0.519), donde el modelo propio se entrena con solo n=500: fuerza prestada, no mejor teoría. Esto **matiza pero no refuta** la periodización: la limitación muestral de la última era ya estaba reconocida en v10 §10.2, y el remedio natural es más datos (la LXVII en 2027) o *partial pooling* jerárquico — no abandonar la frontera.

**P2 (contra agrupaciones más finas) — se cumple sin matices.** El esquema por legislatura es el peor en ambos targets (0.671/0.586) y en casi todos los estratos. Las eras no están mezclando legislaturas heterogéneas en un grado que justifique partirlas.

## 9.2 ¿Por qué mantener la división en 4 eras? — síntesis con la vara neutral

Evaluados todos los esquemas sobre los mismos 10 estratos-legislatura (la vara neutral de §4), la respuesta es más matizada que "porque predice mejor" — no predice mejor, y la defensa se reparte en tres piezas:

1. **Su costo predictivo es despreciable**: −0.014 (nodal) y −0.019 (lastre) de AUC ponderado frente al mejor esquema (S2), concentrado en la LXVI; en las legislaturas 57–62 empata o gana.
2. **La evidencia estructural sostiene sus fronteras como familia, no como ranking.** Lo robusto del análisis de coeficientes (§8) es que los esquemas con fronteras políticas (S2, S3, S4; Δ 0.073–0.086) superan a las ventanas uniformes (S5, 0.060). El orden fino dentro de esa banda es inestable entre corridas — en esta ejecución S2 maximiza Δ (0.086) y S4 queda en 0.073; una corrida previa daba el orden inverso — así que este cuaderno **no** afirma que S4 sea la partición estructuralmente óptima, solo que sus cortes pertenecen a la familia de fronteras reales.
3. **Es la única partición que sostiene el aparato interpretativo de la tesis**: los hallazgos H4 (ruptura en la transición), H5 (legislativización del perfil en Morena — lectura asociativa, no causal: `es_partido_mayoria` y `p_MORENA` son casi colineales en ERA_4, VIF≈129, ver v10 §4.6/§11.1) y H7 (cierre de la brecha de género en ERA 4) requieren un vector de coeficientes por era. Con un corte único en 2018, H4 y H5 serían inobservables por construcción; con el pooled, los coeficientes describirían un promedio de regímenes que no existió en ningún periodo (§8.1).

En suma: **la división en 4 eras no se justifica por una ventaja de AUC ni por el ranking de cohesión estructural — se justifica porque compra la validez comparativa e interpretativa de la tesis a un costo predictivo estadísticamente indistinguible de cero, con fronteras que pertenecen a la familia de cortes políticos que la evidencia estructural sí distingue de los cortes arbitrarios.** El corte único en 2018 (S2) emerge de la evaluación neutral como el mejor esquema compacto — encabeza `Pond. legis` en ambos targets y el Δ de esta corrida — y es la robustez obligada; pero es demasiado grueso para las preguntas de la tesis: funde PRI con PAN y el final de la Transición con Morena, exactamente las distinciones que los hallazgos comparativos explotan.

## 9.3 Recomendaciones

- **Mantener las 4 eras como especificación principal** por razones interpretativas; citar este cuaderno como prueba de robustez con vara neutral (estratos-legislatura).
- **Reportar el pooled (S1) y el corte 2018 (S2) como análisis de sensibilidad**; S2 es la alternativa compacta más fuerte (mejor o empatada en desempeño y con la frontera individual más nítida).
- **No apoyar la defensa de S4 en el orden fino de Δ** (§8.1): es inestable entre corridas. Apoyarla en el contraste político-vs-uniforme, en el costo predictivo nulo y en el requisito interpretativo de H4/H5/H7.
- **Para la LXVI, considerar *partial pooling*** (modelo jerárquico con eras como niveles — extensión natural de la capa bayesiana de v7/v10) mientras no exista la LXVII: la evidencia de este cuaderno cuantifica cuánto AUC deja sobre la mesa el no-pooling en la legislatura corta (≈0.06 en LEG 66).
- **No invertir en re-agrupar para temáticas**: ningún esquema mueve el MAE (banda 0.815–0.819 frente a baseline 0.838).
""")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("OK — fase 2 aplicada:", NB)
