# -*- coding: utf-8 -*-
"""Reencuadre de diputraxv12.ipynb según plan12upd.md:
de "robustez de la periodización de la tesis" a "búsqueda de la segmentación
temporal óptima (LR Lasso)". Documento independiente, sin referencias a la
tesina, v10/v11, diputraxv0 ni hallazgos H1-H7. S4 pasa a ser un candidato
más, sin estatus de línea base.

Verifica índices y contenido vigente antes de aplicar; aborta con mensaje
claro si el notebook no coincide con lo esperado.
"""
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "diputraxv12.ipynb"

# --------------------------------------------------------------------------
# 1. Reescrituras completas de celdas markdown: {índice: (primera_línea_esperada, nuevo_source)}
# --------------------------------------------------------------------------

FULL = {}

FULL[0] = ("# Diputrax V12 — Robustez de la periodización", """\
# Diputrax V12 — Búsqueda de la segmentación temporal óptima (LR Lasso)

**Pregunta central de este cuaderno:**

> **¿Cuál es la división de segmentos de tiempo (agrupación de legislaturas) que minimiza el sesgo y la varianza mientras maximiza la capacidad predictiva y la validez estructural, con un modelo de regresión logística con regularización Lasso?**

**Objeto.** Los datos cubren diez legislaturas de la Cámara de Diputados de México (LVII–LXVI, 1997–presente; ~5,000 registros, una fila por diputado-legislatura). Tres targets se modelan de forma independiente: `nodal_bin` (recibió ≥1 comisión nodal), `lastre_bin` (recibió ≥1 comisión lastre) y `n_comisiones_tematicas` (conteo de comisiones temáticas). El mecanismo de asignación de comisiones cambia a lo largo del tiempo, así que cualquier modelo debe decidir **qué tramos del tiempo se modelan juntos**. Este cuaderno trata esa decisión como un problema de optimización: define un espacio de seis segmentaciones candidatas —del modelo único al modelo por legislatura— y las evalúa bajo criterios explícitos, sin privilegiar a ninguna a priori.

**Estructura.** La sección 1 plantea el problema (intercambio sesgo-varianza) y operacionaliza los cuatro criterios de decisión. Las secciones 2–3 describen datos, pipeline y el espacio de candidatos. La sección 4 define el protocolo de evaluación comparable entre segmentaciones (predicciones *out-of-fold* sobre estratos fijos por legislatura) y la sección 5 justifica los métodos y fija la regla de decisión. Las secciones 6–7 reportan la capacidad predictiva (AUC para los targets binarios; MAE para el conteo). La sección 8 mide la validez estructural (similitud de coeficientes entre legislaturas). La sección 9 aplica la regla de decisión y emite el veredicto.

**Ejecución.** El cuaderno es autocontenido y corre de principio a fin: imputación MICE de `edad_al_tomar_cargo` → *feature engineering* (61 features) → `LogisticRegression` L1 (+`SelectFromModel` en el modelo principal), validación cruzada estratificada *k*=5 con `random_state=42` en todas las particiones.

**Nota de entorno.** Solo requiere `scikit-learn`, `pandas`, `numpy`, `matplotlib` y `seaborn`: la pregunta es de **comparación de segmentaciones temporales**, no de interpretación de features individuales ni de inferencia bayesiana.
""")

FULL[1] = ("# 1. Argumentación", """\
# 1. El problema — segmentar el tiempo es un intercambio sesgo-varianza

Modelar los diez ciclos legislativos exige decidir qué filas entran juntas al entrenamiento. Los dos extremos del espectro fallan por razones opuestas:

- **Un solo modelo para todo el periodo** (máxima muestra, n=5,000) asume que el mecanismo de asignación es estable durante casi tres décadas. Si no lo es, el modelo estima un **promedio de regímenes distintos** que no corresponde a ningún periodo real: **sesgo** de especificación, invisible en el AUC agregado pero fatal para los coeficientes.
- **Un modelo por legislatura** (máxima homogeneidad, n≈500) no mezcla regímenes, pero estima 61 coeficientes con muestras diez veces menores: **varianza** — coeficientes y métricas inestables, sensibles al reparto de folds y a la semilla.

Toda segmentación intermedia es un punto en ese intercambio. La pregunta del cuaderno es qué punto lo resuelve mejor.

## 1.1 Criterios de decisión y su operacionalización

| Criterio | Qué significa aquí | Métrica que lo operacionaliza | Dónde |
|---|---|---|---|
| **Minimizar sesgo** | No mezclar en un grupo periodos con lógicas de asignación distintas | Heterogeneidad interna de los grupos (AUC *naive* por subgrupo) y similitud de coeficientes *dentro* de grupo | §6, §8 |
| **Minimizar varianza** | No estimar con muestras tan pequeñas que el resultado sea ruido | Desviación estándar de la CV por grupo; estabilidad de los coeficientes L1 con n pequeña | §6, §8 |
| **Maximizar capacidad predictiva** | Ordenar mejor a los diputados dentro de cada legislatura, sin crédito por tasas base | **AUC *out-of-fold* ponderado por estrato-legislatura** (`Pond. legis`); MAE ponderado para el conteo | §6, §7 |
| **Maximizar validez estructural** | Que las fronteras caigan donde de verdad cambia la lógica de asignación | **Δ = similitud coseno media dentro − entre grupos** de los coeficientes por legislatura | §8 |

Sesgo y varianza no son criterios independientes de los otros dos: son los **mecanismos** que explican por qué una segmentación gana o pierde capacidad predictiva y validez estructural. Las secciones 6–8 los diagnostican por separado para que el veredicto de §9 pueda atribuir cada resultado a su causa.

## 1.2 Construcción del espacio de candidatos

El espacio de búsqueda (sección 3) recorre la granularidad completa —1, 2, 3, 4, 5 y 10 grupos— y contiene dos familias:

- **Cortes políticos**: fronteras en años electorales que cambiaron la coalición dominante de la Cámara (2006, 2015, 2018, 2024). La asignación de comisiones la negocia la Junta de Coordinación Política y la controla, en la práctica, la coalición mayoritaria de cada ciclo; si el mecanismo cambia, es plausible que cambie en esas fronteras.
- **Cortes mecánicos**: ventanas uniformes de dos legislaturas, ciegas a la política. Sirven de control: si rinden igual que los cortes políticos a granularidad comparable, la ubicación de las fronteras sería irrelevante y solo importaría el tamaño de grupo.

Ningún candidato se privilegia a priori: los seis se evalúan con el mismo espacio de features, el mismo modelo, la misma semilla y la misma vara (§4).

## 1.3 Regla de decisión (resumen; formalizada en §5.7)

1. La **capacidad predictiva estratificada** (`Pond. legis`) es la métrica principal; diferencias por debajo del error de estimación (≈±0.02 por estrato con estas n) se tratan como **empate**.
2. Entre candidatos empatados en predicción decide la **validez estructural** (Δ), leída como contraste entre familias —no como ranking fino, cuya inestabilidad el propio cuaderno documenta (§8).
3. Los diagnósticos de **sesgo** (heterogeneidad interna) y **varianza** (n por grupo, dispersión de CV) desempatan y explican: un candidato que gana por fuerza muestral prestada, o que esconde regímenes distintos dentro de un grupo, recibe el descuento correspondiente en la lectura final.
""")

FULL[2] = ("# 2. Infraestructura replicada", """\
# 2. Datos, imputación y pipeline de modelado

Pipeline autocontenido: carga del parquet → imputación MICE de `edad_al_tomar_cargo` → *feature engineering* (61 features) → *model factories* (LR L1, LR L1+`SelectFromModel`, GLM Poisson). Todas las particiones usan `random_state=42`.
""")

FULL[7] = ("**Lectura — tasas base de los targets por era", """\
**Lectura — tasas base de los targets por legislatura (vista descriptiva).** La tabla muestra la prevalencia de cada target en cada una de las diez legislaturas. La proporción de diputados con **comisión nodal** sube a lo largo de la serie (~0.32 al inicio → ~0.55 en la LXVI): acceder a una comisión estratégica se ha vuelto progresivamente más común. El **lastre** se mantiene en torno a ~0.42–0.50 y cae al final de la serie. Las **temáticas** (conteo medio) alcanzan su máximo en la segunda mitad de la serie.

*Relevancia para el análisis.* Dos consecuencias. (1) La deriva de las tasas base es la primera evidencia de **heterogeneidad temporal**: las legislaturas no son muestras intercambiables de una misma población, así que el tiempo debe segmentarse de *algún* modo — la pregunta del cuaderno es cómo. (2) Es también la razón técnica por la que la evaluación posterior debe **estratificarse**: un modelo agrupado podría inflar su AUC solo por explotar estas diferencias de prevalencia entre periodos (§4, §5.2), sin aprender nada del diputado individual.
""")

FULL[9] = ("## Acciones correctivas (derivadas de diputraxv0)", """\
## Acciones correctivas del pipeline

El espacio de features y la infraestructura de validación incorporan dos correcciones derivadas del diagnóstico de supuestos del pipeline:

- **AC1 — Consolidación de features redundantes.** Se elimina `univ_extranjera` (duplicado exacto de `estudios_en_extranjero`, r=1.0) y `n_cargos_legislativos_prev` (suma exacta de `fue_diputado_local + fue_diputado_federal + fue_senador`, ya incluidos). Son redundancias *dentro del espacio-columna*: el AUC/MAE es invariante a su eliminación; solo se depura la atribución de importancias y los VIF.
- **AC2 — Validación sin fuga por reelección.** Se definen `get_groups()` (por `diputado_id`) y `cv_auc_grouped` / `cv_mae_grouped` (`StratifiedGroupKFold` / `GroupKFold`) para poder estimar el desempeño sin que un mismo diputado reelecto quede en train y test del mismo fold.

**Límite interpretativo:** en periodos de partido dominante `es_partido_mayoria` es casi colineal con la identidad partidista (VIF≈129 en el tramo final de la serie); su lectura debe ser asociativa, no causal.
""")

FULL[11] = ("# 3. Esquemas alternativos de agrupación", """\
# 3. Espacio de candidatos: seis segmentaciones temporales

Seis segmentaciones que recorren el espectro completo de granularidad, del modelo único al modelo por legislatura, con dos familias: cortes **políticos** (fronteras en cambios de coalición dominante: 2006, 2015, 2018, 2024) y cortes **mecánicos** (ventanas uniformes). `S2_2018` divide el periodo en dos con un único corte en la elección de 2018 (LVII–LXIII antes; LXIV–LXVI después).

| Esquema | Grupos | Racionalidad |
|---|---|---|
| `S1_POOLED` | 1 (57–66) | Hipótesis nula: mecanismo único y estable; máxima muestra (n=5,000), mínima varianza, máximo sesgo potencial. |
| `S2_2018` | 2 (57–63 / 64–66) | Un solo corte político, en la elección de 2018. |
| `S3_FUSION34` | 3 (57–59 / 60–62 / 63–66) | Tres cortes de coalición, sin separar el último tramo (63–66 juntos). |
| `S4_ERAS` | 4 (57–59 / 60–62 / 63–65 / 66) | Cuatro cortes de coalición (2006, 2015, 2024); el último grupo es una sola legislatura. |
| `S5_PARES` | 5 (pares consecutivos) | Ventanas uniformes de 2 legislaturas, ciegas a la política: control de la familia mecánica. |
| `S6_LEGIS` | 10 (una por legislatura) | Máxima homogeneidad, mínima muestra (n≈500): el extremo de varianza. |

Todos los candidatos usan **el mismo espacio de 61 features, el mismo modelo (LR L1+SFM) y la misma semilla**; lo único que cambia es qué filas entran juntas al entrenamiento. Nótese que `legislatura_num` está en `FEAT_COLS`: los esquemas gruesos pueden usarla para adaptarse parcialmente al tiempo, lo que hace la comparación *conservadora* a favor de los esquemas gruesos.
""")

FULL[13] = ("**Lectura — composición de los esquemas.**", """\
**Lectura — composición de los candidatos.** La tabla despliega, para cada segmentación, el tamaño muestral (`n`) y las tasas base de cada grupo. Recorre el espectro completo de granularidad: de **S1** (un solo grupo, n=5,000) a **S6** (un grupo por legislatura, n≈500). El grupo más corto del espacio de candidatos es `G4_MOR_66` en **S4**, con **n=500** (una sola legislatura), frente a n=1,500 de sus otros tres grupos.

*Relevancia para el análisis.* La `n` por grupo gobierna directamente la **varianza de los coeficientes y del AUC**: grupos más finos son más homogéneos pero más ruidosos. Comparar segmentaciones es, en el fondo, recorrer el intercambio **sesgo** (mezclar regímenes distintos) contra **varianza** (estimar con poca muestra) planteado en §1. Las tasas base por grupo también anticipan qué particiones funden poblaciones dispares —p. ej., `POST_2018` de S2 (n=1,500) promedia legislaturas con tasas nodales distintas—, algo que la evaluación estratificada y la §8 pondrán a prueba.
""")

FULL[27] = ("## 6.1 Interpretación — targets binarios", """\
## 6.1 Interpretación — targets binarios

**Panorama.** El AUC ponderado por estrato-legislatura (`Pond. legis`, LR L1+SFM) ordena así las segmentaciones:

| Esquema | Nodal | Lastre |
|---|---:|---:|
| S2 — Corte 2018 | **0.714** | **0.610** |
| S1 — Pooled | 0.713 | 0.604 |
| S3 — Fusión T+M | 0.704 | 0.601 |
| S4 — 4 cortes coalición | 0.700 | 0.591 |
| S5 — Pares uniformes | 0.696 | 0.592 |
| S6 — Por legislatura | 0.671 | 0.586 |

Cuatro lecturas:

**1. Entre 1 y 5 grupos, las diferencias predictivas son sub-resolución — empate según la regla de §5.7.** La banda S1–S5 abarca 0.018 puntos de AUC en nodal y 0.019 en lastre, dentro del umbral de empate (≈±0.02). El desagregado por legislatura muestra que en las legislaturas 57–62 todos los candidatos de esa banda rinden a la par (diferencias dentro de ±0.02; en la LEG 59, S4 incluso encabeza: 0.730 vs. 0.706 del pooled). **Las diferencias se concentran en las legislaturas 63–66**, sobre todo en la LXVI: nodal 0.683 (S1) vs. 0.624 (S4); lastre 0.587 vs. 0.519.

**2. La penalización del extremo fino es varianza pura.** S6 es el peor candidato en ambos targets (0.671/0.586) y en casi todos los estratos, y S4 reproduce el mismo síntoma exactamente donde su grupo se reduce a una legislatura (LXVI, n=500). La vista *naive* lo corrobora: los AUC por legislatura oscilan de 0.611 a 0.739 con desviaciones de hasta ±0.094. Más homogeneidad no compensa el costo muestral de bajar a n≈500: el intercambio sesgo-varianza tiene un límite duro por el lado fino.

**3. A granularidad comparable, la ubicación de las fronteras no mueve la predicción.** S4 (fronteras políticas, n=500–1,500) supera a S5 (ventanas mecánicas, n=1,000) por solo 0.004 en nodal y empata en lastre. La capacidad predictiva estratificada distingue **granularidades**, no familias: la evidencia sobre si las fronteras políticas son mejores que las mecánicas debe venir del análisis estructural (§8), no del AUC.

**4. El mejor candidato compacto es S2 — con una reserva de sesgo.** S2 encabeza ambos targets (0.714/0.610), apenas por encima del pooled. Su vista *naive* muestra, sin embargo, que el grupo `POST_2018` (LXIV–LXVI, n=1,500) alcanza solo 0.687 dentro de grupo frente a 0.732 del `PRE_2018`: el corte único junta legislaturas con dinámicas distintas en su tramo final — heterogeneidad interna que la sección 8 examina con los coeficientes.

**Nota sobre el AUC global.** Para S1 el AUC global (0.729) supera al estratificado (0.713): esa brecha es exactamente el crédito por tasas base entre periodos descrito en §4, y es la razón por la que la comparación honesta debe hacerse dentro de estratos fijos por legislatura. La columna de robustez (LR L1 completo, sin SFM) replica el ordenamiento en ambos targets (S2/S1 arriba con 0.715, S6 abajo con 0.673).
""")

FULL[33] = ("## 7.1 Interpretación — temáticas", """\
## 7.1 Interpretación — temáticas

El MAE ponderado es **plano entre segmentaciones**: de 0.815 (S2) a 0.819 (S1 y S4), una banda de 0.004 comisiones — ruido. Ninguna segmentación mejora el baseline de "predecir la media de la legislatura" (0.838) en más de ~2.7%, y en la LEG 64 el baseline incluso supera a los seis esquemas (0.742 vs. 0.767–0.795). El conteo de comisiones temáticas es **insensible a la segmentación temporal**: el perfil biográfico apenas aporta señal para este target, se agrupe como se agrupe. Conforme a la regla de §5.7, este target no discrimina entre candidatos y queda fuera del veredicto de §9.
""")

FULL[34] = ("# 8. Heterogeneidad de coeficientes entre legislaturas", """\
# 8. Validez estructural — heterogeneidad de coeficientes entre legislaturas

El AUC mide cuánto predice cada segmentación; esta sección mide si sus **fronteras caen donde de verdad cambia la lógica de asignación**. Para cada candidato la prueba es la misma: si su partición es estructuralmente válida, los vectores de coeficientes de una LR L1 ajustada **por legislatura** deben parecerse más *dentro* de sus grupos que *entre* ellos (estructura de bloques). Se ajusta `lr_binary` (L1 completo, sin SFM, para que los 61 coeficientes sean comparables posición a posición) sobre `nodal_bin` en cada legislatura, excluyendo `legislatura_num` del espacio de features (constante dentro de cada grupo), y se calcula la **similitud coseno** entre los 10 vectores.
""")

FULL[36] = ("**Lectura — heatmap de similitud coseno", """\
**Lectura — heatmap de similitud coseno de coeficientes.** Cada celda es la similitud coseno (0–1) entre los vectores de 61 coeficientes de la LR L1 ajustada en dos legislaturas: **1 = misma lógica de asignación**, 0 = ortogonal. La matriz no resalta las fronteras de ningún candidato: la estructura de bloques de cada segmentación se cuantifica de forma neutral en la tabla siguiente, que calcula la similitud dentro vs. entre grupos para los seis candidatos por igual. *Advertencia de estabilidad:* los coeficientes L1 ajustados con n≈500 por legislatura tienen varianza alta, y los valores finos de la matriz varían entre corridas/entornos; las comparaciones celda a celda no deben sobreinterpretarse — es la manifestación directa del criterio de **varianza** de §1.

*Relevancia para el análisis.* Este diagnóstico responde una pregunta que el AUC **no puede** contestar: no *cuánto* predice cada segmentación, sino **si sus fronteras caen donde de verdad cambia el mecanismo** de asignación. Opera sobre los **coeficientes** (la lógica del modelo), no sobre las predicciones (su desempeño); por eso es el instrumento que mide la validez estructural (§5.5).
""")

FULL[38] = ("**Lectura — cohesión de coeficientes", """\
**Lectura — cohesión de coeficientes (Δ = dentro − entre).** Para cada segmentación se promedia la similitud coseno **dentro** de sus grupos y **entre** grupos; **Δ** es su diferencia. Un Δ grande significa que la partición traza sus fronteras justo donde la lógica de coeficientes cambia (bloques internamente parecidos, contraste alto hacia afuera). S1 y S6 muestran `NaN` porque, por construcción, no tienen pares entre-grupos (un solo grupo) o dentro-de-grupo (grupos de una legislatura), respectivamente.

*Cómo leerla con cautela.* En esta corrida, S2 alcanza el Δ máximo (0.086), seguido de cerca por S3 (0.083) y S4 (0.073); S5 queda claramente abajo (0.060). La banda S2–S4 es estrecha y su orden fino **no es estable entre corridas**: una ejecución previa de este mismo cuaderno, con un entorno ligeramente distinto, colocaba a S4 primero (0.089). Lo robusto no es el ranking dentro de la banda, sino el **contraste entre la familia de cortes políticos (Δ ≥ 0.073) y las ventanas uniformes ciegas a la política (0.060)**. La interpretación está en §8.1.
""")

FULL[39] = ("## 8.1 Interpretación — heterogeneidad de coeficientes", """\
## 8.1 Interpretación — validez estructural

Este análisis pregunta algo distinto al AUC: no *cuánto* predice cada segmentación, sino **si sus fronteras cortan donde de verdad cambia la lógica de asignación**. El indicador Δ (similitud media dentro de grupo − entre grupos) mide la calidad estructural de cada partición:

| Esquema | Sim. dentro | Sim. entre | Δ |
|---|---:|---:|---:|
| S2 — Corte 2018 | 0.516 | 0.430 | **0.086** |
| S3 — Fusión T+M | **0.537** | 0.454 | 0.083 |
| S4 — 4 cortes coalición | 0.535 | 0.462 | 0.073 |
| S5 — Pares uniformes | 0.530 | 0.470 | 0.060 |

Tres lecturas:

- **La familia política maximiza la validez estructural.** Los tres candidatos cuyas fronteras siguen cambios de coalición dominante (S2, S3, S4) se agrupan en Δ 0.073–0.086, por encima de las ventanas uniformes de S5 (0.060), que parten los bloques reales por la mitad. Las fronteras electorales (2006, 2015, 2018, 2024) capturan cambios reales del mecanismo; las mecánicas, no. Este contraste — no el orden fino dentro de la banda política — es el hallazgo estructural robusto.
- **2018 es la frontera individual más fuerte de esta corrida.** El corte único de S2 obtiene el mayor Δ (0.086): separar el periodo pre y post 2018 es donde más cambia la lógica de coeficientes. S3 y S4 conservan una cohesión interna mayor (0.537 y 0.535 vs. 0.516 de S2): sus grupos son más homogéneos por dentro — menos sesgo interno — a costa de más grupos y menos muestra por grupo.
- **El orden fino S2–S3–S4 es frágil — varianza en acción.** Los coeficientes L1 por legislatura (n≈500) varían entre corridas; una ejecución previa de este cuaderno arrojaba S4 primero (Δ=0.089, con S3 en 0.057). Ninguna conclusión debe descansar en qué candidato de la banda política queda primero; sí es estable que S5 queda último y que la frontera de 2018 aparece siempre entre las más nítidas.

La similitud media global entre los 10 vectores de coeficientes es 0.476 — lejos de 1.0: la lógica de asignación nodal **no** es estable a lo largo de las diez legislaturas. Es la evidencia directa de **sesgo** contra el candidato sin segmentación: S1, aunque predice bien en promedio, estima un mecanismo que no corresponde a ningún periodo en particular.
""")

FULL[40] = ("# 9. Conclusiones", """\
# 9. Conclusiones — la segmentación temporal óptima

## 9.1 Aplicación de la regla de decisión (§5.7)

| Candidato | Predicción (`Pond. legis` nodal / lastre) | Validez estructural (Δ) | Sesgo (heterogeneidad interna) | Varianza (n por grupo) | Estado |
|---|---|---|---|---|---|
| S1 — Pooled | 0.713 / 0.604 (empate, banda alta) | n/d (sin fronteras) | Máximo: similitud global 0.476 → estima un promedio de regímenes que no existió | Mínima (n=5,000) | Descartado por sesgo |
| S2 — Corte 2018 | **0.714 / 0.610** (encabeza) | **0.086** (máx. de esta corrida) | Moderado: `POST_2018` naive 0.687 vs. 0.732 de `PRE_2018` | Baja (n=3,500/1,500) | **Óptimo** |
| S3 — Fusión T+M | 0.704 / 0.601 (empate) | 0.083; cohesión interna máx. (0.537) | Bajo | Media (n=1,500–2,000) | Eficiente (más resolución, menos sesgo) |
| S4 — 4 cortes coalición | 0.700 / 0.591 (empate) | 0.073; cohesión interna 0.535 | Bajo | Alta en su grupo final (n=500) | Eficiente (máxima resolución política) |
| S5 — Pares uniformes | 0.696 / 0.592 (empate) | 0.060 (mín. de los comparables) | — | Media (n=1,000) | Descartado por validez estructural |
| S6 — Por legislatura | 0.671 / 0.586 (peor) | n/d | Mínimo | Máxima (n≈500, σ hasta ±0.094) | Descartado por varianza |

Los extremos caen cada uno por su criterio: S6 por varianza (peor predicción de todo el espacio), S1 por sesgo (predice en banda alta, pero sus coeficientes promedian regímenes con similitud media 0.476), y S5 por validez estructural (misma granularidad aproximada que S4, fronteras en el lugar equivocado). La frontera eficiente del intercambio queda en la familia de cortes políticos: **S2, S3 y S4**.

## 9.2 Veredicto

**Dentro de la frontera eficiente, la segmentación óptima de esta evaluación es S2: un único corte en 2018.** Encabeza la capacidad predictiva estratificada en ambos targets (0.714 nodal / 0.610 lastre), obtiene el mayor contraste estructural de esta corrida (Δ=0.086) y, con solo dos grupos (n=3,500/1,500), es el candidato de menor varianza dentro de la familia política.

Dos reservas, ambas internas al cuaderno:

1. **Sesgo residual de `POST_2018`.** Su grupo posterior es internamente heterogéneo (AUC naive 0.687 vs. 0.732 del anterior): el corte único agrupa legislaturas cuyo mecanismo aún difiere. S3 y S4 reducen ese sesgo (cohesión interna 0.537/0.535 vs. 0.516) a cambio de más varianza; el costo predictivo de esa resolución adicional es ≤0.014 de AUC — dentro del umbral de empate.
2. **Fragilidad del orden fino.** El ranking Δ dentro de la banda política se invierte entre corridas (§8.1). El veredicto **fuerte** es sobre la familia: la segmentación óptima usa cortes políticos con granularidad baja (2–4 grupos). El veredicto **débil** es que, dentro de ella, S2 es el mejor representante de esta ejecución.

## 9.3 Respuesta directa a la pregunta del cuaderno

La división de segmentos de tiempo que minimiza sesgo y varianza mientras maximiza capacidad predictiva y validez estructural, con LR Lasso, es **una partición de granularidad baja cuyas fronteras siguen los cambios de coalición dominante; en esta evaluación, el corte único en la elección de 2018**. Los extremos fallan por construcción — sin segmentar, el modelo estima un promedio de mecanismos que no existió (sesgo); por legislatura, el ruido muestral domina (varianza) — y las ventanas ciegas a la política fallan por validez estructural: a igual granularidad, trazar las fronteras en los años de cambio de coalición produce bloques de coeficientes más cohesivos que trazarlas mecánicamente. Si el análisis exige más resolución temporal que dos segmentos, S3 (tres grupos) y S4 (cuatro grupos) pagan un costo predictivo estadísticamente indistinguible de cero a cambio de grupos internamente más homogéneos.

## 9.4 Limitaciones y extensiones

- **Una sola corrida para la evidencia estructural.** El Δ de §8 proviene de una ejecución con una semilla; su orden fino es inestable. Extensión natural: *bootstrap* o multi-semilla sobre los coeficientes por legislatura para poner intervalos alrededor de Δ.
- **Espacio de candidatos discreto y pequeño.** Se evaluaron 6 segmentaciones de las 512 particiones contiguas posibles de 10 legislaturas. Una búsqueda exhaustiva con el mismo protocolo (OOF + estratos-legislatura) es computacionalmente factible y convertiría el veredicto en un óptimo global, no solo en el mejor de seis.
- **El grupo final corto.** Cualquier segmentación que aísle la última legislatura hereda n=500; un modelo jerárquico con *partial pooling* entre segmentos permitiría resolución fina sin pagar toda la varianza. Con datos de la próxima legislatura (2027), el tramo posterior a 2018/2024 podrá reevaluarse con más muestra.
- **Las temáticas no discriminan.** El conteo de comisiones temáticas es insensible a la segmentación (banda MAE 0.815–0.819 vs. baseline 0.838) y no participa del veredicto; la conclusión aplica a los targets binarios.
""")

# --------------------------------------------------------------------------
# 2. Reemplazos exactos: [(índice, viejo, nuevo), ...]  (cada viejo debe
#    aparecer exactamente 1 vez en la celda)
# --------------------------------------------------------------------------

REPL = [
    # --- celda 4: comentario de constantes ---
    (4, "# --- Constantes idénticas a v10 (celda 70) ---",
        "# --- Constantes del pipeline ---"),
    # --- celda 5: comentario MICE ---
    (5, "# --- MICE idéntico a v10 (celda 74) ---",
        "# --- Imputación MICE de edad_al_tomar_cargo ---"),
    # --- celda 6: comentarios + tabla descriptiva por legislatura ---
    (6, "# --- Feature engineering idéntico a v10 (celda 75) ---",
        "# --- Feature engineering ---"),
    (6, "# Targets ORIGINALES de la tesis (idénticos a v10, intactos)",
        "# Targets: nodal_bin, lastre_bin, n_comisiones_tematicas"),
    (6, 'print("\\nTasas de los targets por era:")\n'
        'display(df.groupby("era")[["nodal_bin","lastre_bin","n_comisiones_tematicas"]]\n'
        '          .mean().reindex(ERA_ORDER).round(3))',
        'print("\\nTasas de los targets por legislatura:")\n'
        'display(df.groupby("legislatura_num")[["nodal_bin","lastre_bin","n_comisiones_tematicas"]]\n'
        '          .mean().round(3))'),
    # --- celda 8: comentarios AC1(v0) y encabezado ---
    (8, "# --- Espacio de features idéntico a v10 (celda 76) ---",
        "# --- Espacio de features (61 columnas) ---"),
    (8, "# AC1(v0): univ_extranjera eliminada",
        "# AC1: univ_extranjera eliminada"),
    (8, "# AC1(v0): n_cargos_legislativos_prev eliminada",
        "# AC1: n_cargos_legislativos_prev eliminada"),
    # --- celda 10: docstrings, comentarios, print final ---
    (10, "# --- Model factories idénticas a v10 (celda 78) ---",
         "# --- Model factories ---"),
    (10, '"LR L1 (Lasso) completo — idéntico a v10."',
         '"LR L1 (Lasso) completo."'),
    (10, '"scale -> SelectFromModel(L1) -> L1 — pipeline principal de v10 (Tabla 7)."',
         '"scale -> SelectFromModel(L1) -> L1 — modelo principal del cuaderno."'),
    (10, '"GLM Poisson — idéntico a v10."',
         '"GLM Poisson."'),
    (10, "# Paleta categórica para los ESQUEMAS (orden fijo, validada CVD;\n"
         "# las eras conservan ERA_COLORS de v10)",
         "# Paleta categórica para los ESQUEMAS (orden fijo, validada CVD)"),
    (10, "# -- AC2(v0): validacion sin fuga por reeleccion",
         "# -- AC2: validacion sin fuga por reeleccion"),
    (10, 'print("Infraestructura de modelado replicada de v10 — OK")',
         'print("Pipeline de modelado listo — OK")'),
    # --- celda 12: mapeo neutral de S4 + etiqueta corta ---
    (12, '    "S4_ERAS": dict(ERA_MAP),',
         '    "S4_ERAS": {\n'
         '        **{l: "G1_PRI_57_59" for l in (57, 58, 59)},\n'
         '        **{l: "G2_PAN_60_62" for l in (60, 61, 62)},\n'
         '        **{l: "G3_TRANS_63_65" for l in (63, 64, 65)},\n'
         '        66: "G4_MOR_66",\n'
         '    },'),
    (12, '"S4_ERAS":   "S4 — 4 eras (tesis)",',
         '"S4_ERAS":   "S4 — 4 cortes coalición",'),
    # --- celda 14: neutralidad de estratos sin privilegiar S4 ---
    (14, "**El problema (ii): neutralidad de los estratos.** Los estratos fijos no "
         "pueden ser las 4 eras canónicas, porque las eras son exactamente la "
         "partición de uno de los candidatos (S4). Usarlas como vara sesga la "
         "comparación en un sentido técnico preciso: estratificar por era elimina "
         "el crédito por tasas base *entre* eras —justo en las fronteras de S4— "
         "pero deja intacto el crédito por diferencias de prevalencia *entre "
         "legislaturas dentro de una misma era*, que un esquema más fino que las "
         "eras (S5, S6) sí puede explotar.",
         "**El problema (ii): neutralidad de los estratos.** Los estratos fijos no "
         "pueden ser los grupos de ninguno de los candidatos, porque eso usaría la "
         "vara de uno para medir a todos. El sesgo es técnico y preciso: "
         "estratificar por los grupos de un candidato dado elimina el crédito por "
         "tasas base exactamente en *sus* fronteras, pero deja intacto el crédito "
         "por diferencias de prevalencia *dentro* de sus grupos, que los "
         "candidatos más finos sí pueden explotar."),
    # --- celda 15: justificación metodológica ---
    (15, 'La pregunta no es "¿qué modelo predice mejor?" sino "**¿qué agrupación '
         'de legislaturas sirve mejor al análisis?**", y esa diferencia dicta cada '
         'elección metodológica.',
         'La pregunta es de **optimización de la segmentación temporal** —qué '
         'partición minimiza sesgo y varianza mientras maximiza capacidad '
         'predictiva y validez estructural— y esa formulación dicta cada elección '
         'metodológica.'),
    (15, "estratificar por era, en cambio, removería el crédito por tasas base "
         "exactamente en las fronteras de S4 y dejaría a los esquemas más finos "
         "el crédito intra-era, sesgando la comparación (§4).",
         "estratificar por los grupos de un candidato, en cambio, removería el "
         "crédito por tasas base exactamente en sus fronteras y dejaría a los "
         "candidatos más finos el crédito intra-grupo, sesgando la comparación (§4)."),
    (15, "valida una periodización por su **contenido interpretativo** —el que "
         "sostiene los hallazgos H4/H5/H7 de la tesis—, no solo por su exactitud.",
         "valida una segmentación por su **contenido estructural** —que sus "
         "coeficientes describan mecanismos reales y no promedios de regímenes "
         "mezclados—, no solo por su exactitud."),
    (15, "Si una periodización es sustantiva, la similitud debe ser mayor "
         "*dentro* de sus grupos que *entre* ellos",
         "Si una segmentación es sustantiva, la similitud debe ser mayor "
         "*dentro* de sus grupos que *entre* ellos"),
    (15, "`SelectFromModel` replica el pipeline principal de v10 (Tabla 7), de "
         "modo que la comparación de esquemas se hace sobre **exactamente el "
         "mismo modelo** que usa la tesis —no sobre un sustituto—, lo que hace "
         "las conclusiones trasladables.",
         "`SelectFromModel` añade una capa de selección de features que "
         "estabiliza el modelo con n pequeña; el LR L1 completo se reporta como "
         "columna de robustez para verificar que el ordenamiento no depende del "
         "selector."),
    (15, 'estos métodos convierten la pregunta "¿son las 4 eras la mejor '
         'partición?" en una prueba falsable, comparable y reproducible:',
         'estos métodos convierten la pregunta "¿cuál es la segmentación '
         'temporal óptima?" en una prueba falsable, comparable y reproducible:'),
    (15, "el *baseline* fija la vara del valor añadido, y la similitud coseno "
         "separa desempeño de validez estructural.",
         "el *baseline* fija la vara del valor añadido, y la similitud coseno "
         "separa desempeño de validez estructural.\n"
         "\n"
         "## 5.7 Regla de decisión\n"
         "\n"
         "El veredicto de §9 aplica mecánicamente esta regla:\n"
         "\n"
         "1. **Métrica principal:** AUC OOF ponderado por estrato-legislatura "
         "(`Pond. legis`). Diferencias menores a ≈0.02 (el orden del error de "
         "estimación por estrato con estas n) se tratan como **empate**.\n"
         "2. **Desempate estructural:** entre candidatos empatados en "
         "predicción, gana el de mayor validez estructural (Δ), leída al nivel "
         "de **familias** (cortes políticos vs. mecánicos; granularidad), no de "
         "ranking fino — la inestabilidad del orden fino entre corridas se "
         "documenta en §8.\n"
         "3. **Descuentos por sesgo y varianza:** un candidato cuya ventaja "
         "provenga de fuerza muestral prestada (grupos grandes pero "
         "heterogéneos) o que esconda regímenes distintos dentro de un grupo "
         "(diagnóstico en la vista *naive* de §6 y en §8) recibe la reserva "
         "correspondiente en el veredicto; el conteo de temáticas solo entra si "
         "discrimina entre candidatos."),
    # --- celda 17: comentario del modelo principal ---
    (17, "# Modelo principal: LR L1+SFM (Tabla 7 de v10). Robustez: LR L1 completo.",
         "# Modelo principal: LR L1+SFM. Robustez: LR L1 completo."),
    # --- celda 21: vista naive con vocabulario sesgo/varianza ---
    (21, "(1) la **inestabilidad de la granularidad fina**",
         "(1) la **varianza de la granularidad fina**"),
    (21, "y ERA_4 con n=500 llega solo a 0.626±0.058",
         "y el grupo de una sola legislatura de S4 (`G4_MOR_66`, n=500) llega solo a 0.626±0.058"),
    (21, "(2) la **heterogeneidad interna de los grupos gruesos**",
         "(2) el **sesgo por heterogeneidad interna de los grupos gruesos**"),
    # --- celda 24: lectura Figura A ---
    (24, "El mensaje de la figura es la **ausencia de una brecha material** entre "
         "los esquemas gruesos y la periodización de la tesis (S4); el único "
         "claramente rezagado es S6.",
         "El mensaje de la figura es que la elección de segmentación apenas mueve "
         "la predicción agregada entre 1 y 5 grupos — la **brecha material "
         "aparece solo en el extremo fino**: S6 es el único claramente rezagado."),
    # --- celda 26: lectura Figura B ---
    (26, "en particular, las legislaturas donde los esquemas de grupo pequeño "
         "(S6, y S4 en la LXVI) caen respecto a los gruesos delatan la "
         "penalización por n reducida.",
         "en particular, las legislaturas donde los esquemas con grupos de una "
         "sola legislatura (S6 en toda la serie, S4 en la LXVI) caen respecto a "
         "los gruesos delatan la **penalización por varianza** de entrenar con "
         "n≈500."),
    # --- celda 28: encabezado temáticas ---
    (28, "Mismo protocolo OOF con el GLM Poisson de v10 (`StandardScaler`",
         "Mismo protocolo OOF con el GLM Poisson (`StandardScaler`"),
    # --- celda 35: quitar líneas de fronteras S4 del heatmap ---
    (35, '# Fronteras de era (después de la 59, la 62 y la 65)\n'
         'for b in [3, 6, 9]:\n'
         '    ax.axhline(b, color="white", lw=3)\n'
         '    ax.axvline(b, color="white", lw=3)\n'
         'ax.set_title("Similitud coseno de coeficientes LR L1 (nodal) entre legislaturas\\n"\n'
         '             "líneas blancas = fronteras del esquema S4 (solo referencia)",\n'
         '             fontsize=11.5, fontweight="bold", color=TXT)',
         'ax.set_title("Similitud coseno de coeficientes LR L1 (nodal) entre legislaturas",\n'
         '             fontsize=11.5, fontweight="bold", color=TXT)'),
]

# --------------------------------------------------------------------------
# Aplicación
# --------------------------------------------------------------------------

def cell_src(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    if len(cells) != 41:
        sys.exit(f"ABORT: se esperaban 41 celdas, hay {len(cells)}")

    errors = []

    # Reescrituras completas
    for idx, (expect_start, new_text) in FULL.items():
        src = cell_src(cells[idx])
        if not src.lstrip().startswith(expect_start):
            errors.append(f"celda {idx}: no empieza con {expect_start!r} "
                          f"(empieza con {src.lstrip()[:60]!r})")

    # Reemplazos exactos
    for idx, old, _ in REPL:
        src = cell_src(cells[idx])
        n = src.count(old)
        if n != 1:
            errors.append(f"celda {idx}: patrón aparece {n} veces "
                          f"(esperada 1): {old[:70]!r}")

    if errors:
        print("ABORT — el notebook no coincide con lo esperado:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    for idx, (_, new_text) in FULL.items():
        set_src(cells[idx], new_text.rstrip("\n"))
        print(f"celda {idx:2d}: reescrita")

    for idx, old, new in REPL:
        src = cell_src(cells[idx]).replace(old, new)
        set_src(cells[idx], src)
    for idx in sorted({i for i, _, _ in REPL}):
        print(f"celda {idx:2d}: reemplazos aplicados")

    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"\nOK — guardado {NB_PATH}")


if __name__ == "__main__":
    main()
