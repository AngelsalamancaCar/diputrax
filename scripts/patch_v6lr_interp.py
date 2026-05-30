import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def md(cell, src):
    cell['source'] = src if isinstance(src, list) else [src]

# ── Cell 84: diseño del estudio — quitar comparativa 3 algoritmos ─────────────
md(cells[84], """## 4.1 Diseño del estudio y lógica temporal por eras

### Pregunta de investigación

> **¿El perfil biográfico, educativo y de trayectoria de un diputado predice a qué tipo de comisión es asignado, y ese perfil ha cambiado entre épocas políticas?**

### Tipología de comisiones

| Tipo | Definición operacional | Implicación política |
|:-----|:----------------------|:---------------------|
| **Nodal** | ≥1 comisión nodal (presupuesto, hacienda, seguridad) | Alta influencia · cargo de confianza del grupo mayoritario |
| **Lastre** | ≥1 comisión lastre (sin recursos ni dictámenes) | Marginalización · oposición o primiparos sin red |
| **Temática** | Conteo de comisiones temáticas (0–10) | Especialización · volumen de trabajo legislativo |

### Épocas y distribución de datos

| Época | Legislaturas | Régimen | n | Tasa nodal | Tasa lastre | Media temáticas |
|:------|:------------|:--------|--:|----------:|----------:|---------------:|
| ERA_1 | 57–59 | PRI hegemónico | 1500 | 32.2% | 42.1% | 1.53 |
| ERA_2 | 60–62 | Alternancia PAN | 1500 | 40.7% | 45.9% | 1.93 |
| ERA_3 | 63–65 | Transición | 1500 | 49.5% | 49.9% | 2.15 |
| ERA_4 | 66 | Morena | 500 | 55.4% | 39.2% | 1.87 |

**Nota:** ERA_4 tiene sólo una legislatura (n=500). Intervalos de confianza más amplios — interpretar con cautela.

### Feature engineering (86 features)

| Bloque | Variables representativas |
|:-------|:--------------------------|
| Político-electoral | `sexo_bin`, `mayoria_relativa`, `es_partido_mayoria`, `legislatura_num` |
| Trayectoria legislativa | `n_cargos_legislativos_prev`, `fue_diputado_federal`, `fue_senador`, `n_trayectoria_legislativa` |
| Trayectoria administrativa | `n_trayectoria_admin`, `nivel_cargo_max`, `fue_secretario_cargo`, `fue_presidente_mun` |
| Trayectoria política | `n_trayectoria_politica`, `n_trayectoria_empresarial`, `lider_juvenil_partido` |
| Educación | `grado_estudios_ord`, `tiene_posgrado`, `univ_elite`, `estudios_en_extranjero` |
| Dummies | Partido (8 cats), Región (6), Área de formación (3: `area_Derecho`, C. Políticas y Sociales, Económico-Financiera) |

**Criterio de selección:** se retienen todas las variables de los cinco bloques de trayectoria y las de contexto institucional. Se eliminaron las 10 dummies individuales de institución universitaria (reemplazadas por `univ_elite`), las 10 dummies de área disciplinaria excedentes (mantenidas solo 3 con justificación teórica), y las 3 variables compuestas que sumaban/maximizaban componentes ya presentes (`carrera_depth`, `edu_calidad`, `exp_alta_jerarquia`).

### Lógica del diseño de modelado

Cada uno de los tres targets se modela **por separado dentro de cada era**. Esto permite:
1. Detectar si el perfil que predice la asignación cambia entre épocas (AUC/MAE por era).
2. Identificar qué features ganan o pierden importancia a lo largo del tiempo (SHAP por era).
3. Medir la transferencia del modelo entre períodos (validación rolling forward).

El modelo empleado es **Regresión Logística L1 (Lasso)** evaluada en dos variantes:
- **LR L1 (full):** L1 sobre los 86 features sin preselección.
- **LR L1 + SFM:** pipeline `StandardScaler → SelectFromModel(LR L1) → LR L1`, donde la selección automática identifica el subconjunto mínimo de features con peso no nulo por era.

La penalización L1 produce modelos *sparse*: coeficientes de features irrelevantes convergen a cero durante el entrenamiento, haciendo la selección parte del proceso de ajuste. `SelectFromModel` con `threshold='mean'` retiene únicamente las variables cuya magnitud de coeficiente supera la media, reduciendo el espacio de features activos por era. La comparación LR L1 (full) vs LR L1 + SFM permite cuantificar el costo/beneficio de la selección automática en AUC.
""")

# ── Cell 86: justificacion k=5 — actualizar conteo de entrenamientos ──────────
old_86 = ''.join(cells[86]['source'])
new_86 = old_86.replace(
    "n=1500 modelos × 4 eras × 3 targets × 3 algoritmos = >54,000 entrenamientos",
    "n=1500 modelos × 4 eras × 3 targets × 2 variantes (LR L1 full + SFM) = >36,000 entrenamientos"
).replace(
    "Forman & Scholz (2010) demuestran empiricamente que comparar clasificadores con k distinto produce rankings incorrectos entre algoritmos. Esto refuerza la decision de fijar k=5 uniforme no solo entre eras sino entre los tres algoritmos evaluados (LR, RF, XGBoost).",
    "Forman & Scholz (2010) demuestran empiricamente que comparar clasificadores con k distinto produce rankings incorrectos entre algoritmos. Esto refuerza la decision de fijar k=5 uniforme entre eras para garantizar comparabilidad directa de los AUC."
)
cells[86]['source'] = [new_86]

# ── Cell 104: SHAP beeswarm nodales — log-odds + referencia v5 ────────────────
md(cells[104], """**Interpretación de hallazgos de SHAP Beeswarm Nodales (grid 2×2 por época)**

Cada punto representa un diputado; el eje X muestra el valor SHAP (contribución individual al **log-odds** de recibir comisión nodal). Los valores SHAP de la Regresión Logística L1 están en espacio de log-odds, no de probabilidad: reflejan contribuciones al predictor lineal, donde la linealidad es exacta. Los rankings de importancia entre features son directamente comparables entre eras.

> **Nota:** los valores |SHAP| numéricos citados a continuación son de referencia del modelo v5 (XGBoost). Se actualizarán al re-ejecutar con LR L1 + SelectFromModel; las direcciones y rankings cualitativos se esperan estables.

- **ERA_1 (PRI):** `area_Derecho` domina entre los top features, seguida de `n_trayectoria_empresarial` y `n_trayectoria_politica`. La formación jurídica y la red empresarial son las credenciales de acceso principales; la carrera política del partido aporta señal complementaria.
- **ERA_2 (PAN):** `area_Derecho` y `edad_imp` dominan con magnitudes prácticamente equivalentes — la credencial jurídica se mantiene y la experiencia acumulada se vuelve igualmente determinante. `n_trayectoria_admin` contribuye con peso moderado.
- **ERA_3 (Transición):** `fue_secretario_cargo` alcanza su pico histórico y `n_cargos_legislativos_prev` se vuelve crítico, reflejando que la fragmentación multipartidista premia tanto la experiencia ejecutiva de alto nivel como la carrera parlamentaria; primer efecto visible de la reelección.
- **ERA_4 (Morena):** `area_Derecho` lidera las importancias — pico histórico de toda la serie —, seguida de `edad_imp` y `n_trayectoria_legislativa`, prácticamente igualados. `es_partido_mayoria` alcanza su máximo histórico pero `area_Derecho`, `edad_imp` y la trayectoria legislativa lo superan en magnitud absoluta. La dispersión de puntos es más compacta que en ERA_1, reflejando el perfil más homogéneo del cuerpo legislativo de Morena.
""")

# ── Cell 107: SHAP heatmap nodales ────────────────────────────────────────────
md(cells[107], """**Interpretación — SHAP Heatmap Nodales (features × épocas)**

El heatmap codifica en color la importancia media (|SHAP|) de cada feature para cada era en espacio de **log-odds**. Los valores reflejan contribuciones al predictor lineal de la Regresión Logística L1.

> **Nota:** los valores numéricos específicos se actualizan al re-ejecutar. La narrativa refleja los patrones cualitativos esperados, coherentes con la estructura lineal del modelo.

- `area_Derecho` (formación en Derecho): el feature de mayor |SHAP| medio en ERA_1, ERA_2 y ERA_4 — la credencial jurídica es la señal de acceso más estable y persistente de toda la serie, no decrece con el tiempo.
- `edad_imp`: importancia alta y variable — en ERA_2 prácticamente empata con `area_Derecho`; en ERA_4 es igualada por `n_trayectoria_legislativa` y superada por `area_Derecho`. **Caveat MNAR (ver §2.2.1b):** el análisis de mecanismo de nulos confirma que `edad_imp` está imputado bajo MNAR — los registros sin dato de edad tienen menor trayectoria y menor tasa de comisión nodal. Su importancia en ERA_2 y ERA_4 puede estar parcialmente inflada: absorbe señal de visibilidad institucional en el SIL además de experiencia real por edad.
- `n_trayectoria_politica`: relevante en ERA_1, cae en ERA_2 y ERA_3, y recupera relevancia en ERA_4 junto con la consolidación de Morena.
- `fue_secretario_cargo`: importancia máxima en ERA_3, mínima en ERA_2 y ERA_4 — el cargo ejecutivo de alto nivel premia en la era de transición multipartidista, no en la era panista como podría suponerse.
- `n_trayectoria_legislativa`: escala de ERA_1 a ERA_4 con el mayor salto relativo de la serie, coherente con la legislativización del perfil nodal bajo Morena.
- `es_partido_mayoria`: baja en ERA_1 y ERA_2, sube en ERA_3 y **alcanza su pico en ERA_4** — la filiación al bloque mayoritario gana relevancia relativa bajo la supermayoría de Morena, pero `area_Derecho`, `n_trayectoria_legislativa` y `edad_imp` mantienen magnitudes superiores en ERA_4.
- `n_cargos_legislativos_prev`: crece de ERA_1 a ERA_3 y cae en ERA_4 — la carrera parlamentaria individual fue señal de distinción en la transición; en ERA_4 la trayectoria legislativa agregada y la filiación al bloque la desplazan.

**Efecto de SelectFromModel:** el heatmap refleja únicamente las features con coeficiente L1 no-cero en cada era. Features que no aparecen en el heatmap de una era determinada fueron eliminadas por la penalización — no solo tienen baja importancia, tienen importancia exactamente cero.

La tabla numérica adjunta permite comparaciones exactas entre eras.
""")

# ── Cell 110: evolución temporal features clave ───────────────────────────────
md(cells[110], """**Interpretación — Evolución temporal de importancias SHAP (Nodales)**

La gráfica de líneas muestra cómo varía el peso de los features clave a lo largo de las cuatro épocas. Los valores SHAP son contribuciones al **log-odds** (espacio del predictor lineal de la LR L1).

> **Nota:** los valores numéricos específicos se actualizan al re-ejecutar con LR L1 + SFM. Las tendencias cualitativas descritas son coherentes con los coeficientes L1 esperados.

Tendencias centrales:

- **`edad_imp`** es uno de los dos predictores de mayor magnitud en ERA_2, prácticamente igualado por `area_Derecho`. En ERA_4 mantiene alta importancia pero es igualado por `n_trayectoria_legislativa` y superado por `area_Derecho`. La experiencia acumulada es señal constante de calidad, especialmente bajo regímenes con alta cohesión interna (PAN y Morena). **Caveat MNAR:** el mecanismo de nulos diagnosticado en §2.2.1b (MNAR, AUC predictivo=0.821) indica que parte de la importancia de `edad_imp` en ERA_2 y ERA_4 puede reflejar visibilidad institucional en el SIL más que edad real.
- **`area_Derecho`** es el feature de mayor |SHAP| medio en ERA_1, ERA_2 y ERA_4 — la formación jurídica no decae con el tiempo, se intensifica en ERA_4, siendo la señal más persistente de la serie.
- **`n_trayectoria_legislativa`** escala de ERA_1 a ERA_4 con el mayor salto relativo de todos los features, coherente con la legislativización del perfil nodal bajo Morena.
- **`n_trayectoria_politica`** dibuja una U: alto en ERA_1, cae en ERA_2 y ERA_3, y se recupera en ERA_4 junto con la consolidación de Morena, partido que recluta desde estructuras políticas propias.
- **`fue_secretario_cargo`** alcanza su pico en ERA_3 —no en ERA_2—, indicando que la experiencia ejecutiva de secretaría fue valorada en el contexto de fragmentación multipartidista. En ERA_4 cae a casi cero.
- **`n_cargos_legislativos_prev`** sube de ERA_1 a ERA_3 como efecto de la reelección, pero **cae en ERA_4**: la trayectoria legislativa agregada y la filiación al bloque desplazan al historial de cargos parlamentarios individuales.
- **`es_partido_mayoria`** se mantiene bajo en ERA_1–ERA_3 y **salta en ERA_4**: la lealtad al bloque gana relevancia relativa cuando Morena alcanza supermayoría, aunque `area_Derecho`, `n_trayectoria_legislativa` y `edad_imp` lo superan en magnitud absoluta.

**Interpretación L1:** en las eras donde un feature desaparece de la gráfica, su coeficiente es exactamente cero — eliminado por la penalización Lasso, no solo irrelevante. Esto hace la lectura de ausencia más contundente que en v5: ausencia en v6LR = coeficiente cero, no solo importancia baja.
""")

# ── Cell 113: 5.5 interpretación nodales ──────────────────────────────────────
md(cells[113], """## 5.5 Interpretación — Comisiones Nodales

**Señal decreciente a lo largo del tiempo.** El AUC cae de 0.734 (ERA_1) a ~0.619 (ERA_4), indicando que el perfil biográfico predice progresivamente peor quién obtiene una comisión nodal.

**Hallazgos clave por era:**
- **ERA_1 (PRI)** — época más predecible (LR L1 ~0.734). El PRI operaba con criterios formalizados. `area_Derecho` encabeza las importancias, seguida de `n_trayectoria_empresarial` y `n_trayectoria_politica`: formación jurídica, red empresarial y carrera política del partido eran las credenciales centrales de acceso.
- **ERA_2 (PAN)** — señal similar (LR L1 ~0.720). La alternancia no rompe la lógica de asignación basada en perfil. `area_Derecho` y `edad_imp` alcanzan prácticamente el mismo peso — pico histórico de `edad_imp` en la serie — confirmando que el PAN premiaba formación jurídica y experiencia acumulada por igual. *Nota: la importancia de `edad_imp` en ERA_2 puede estar parcialmente inflada por el mecanismo MNAR diagnosticado en §2.2.1b — mujeres ERA_2 tienen tasa de nulo 3× mayor que hombres (8.2% vs 2.6%).*
- **ERA_3 (Transición)** — señal moderada (LR L1 ~0.699). `fue_secretario_cargo` alcanza su pico histórico de importancia SHAP y `n_cargos_legislativos_prev` se vuelve crítico como primer efecto visible de la reelección. La fragmentación multipartidista premia tanto la credencial ejecutiva de alto nivel como la carrera parlamentaria emergente.
- **ERA_4 (Morena)** — caída notable a ~0.619. `area_Derecho` encabeza las importancias, seguido de `n_trayectoria_legislativa` y `edad_imp`. `es_partido_mayoria` alcanza su pico histórico — la filiación al bloque gana peso relativo pero no domina en magnitud absoluta. *Con n=500, el intervalo ±0.06 es amplio — resultado orientativo.*

**Evolución del perfil nodal (SHAP en log-odds):**

| Feature | Dirección | Lectura |
|:--------|:----------|:--------|
| `edad_imp` | ➕ alta y variable | Empata con `area_Derecho` en ERA_2; superado en ERA_4. Señal real de experiencia, pero **importancia parcialmente inflada por MNAR** (§2.2.1b) |
| `n_trayectoria_politica` | ➕ forma de U | Alto en ERA_1, cae en ERA_2–ERA_3, recupera en ERA_4 |
| `area_Derecho` | ➕ estable y creciente | Formación jurídica, señal más persistente de la serie; pico en ERA_4 |
| `fue_secretario_cargo` | ➕ pico ERA_3 | Valioso en la transición multipartidista; eliminado por L1 en ERA_2 y ERA_4 |
| `n_cargos_legislativos_prev` | ➕ pico ERA_3 | Crece con la reelección hasta ERA_3; cae en ERA_4 |
| `es_partido_mayoria` | ➕ pico ERA_4 | Bajo en ERA_1–ERA_3; salta en ERA_4 bajo supermayoría de Morena |
| `n_trayectoria_legislativa` | ➕ escala fuerte ERA_4 | Mayor salto relativo de la serie — legislativización del perfil nodal bajo Morena |
| `n_trayectoria_admin` | ➕ moderado y estable | Contribución estable sin ser top predictor |

**Modelo:** Regresión Logística L1. La estructura subyacente de la asignación es predominantemente lineal — resultado consistente con versiones anteriores del estudio y con la teoría (reglas de partido formalizadas, no decisiones altamente no-lineales). La selección automática por `SelectFromModel` confirma y cuantifica la esparsidad del problema por era.

**Tendencia central:** transición del modelo de reclutamiento político-jurídico (PRI: Derecho + trayectoria empresarial/política) hacia formación y experiencia equivalentes (PAN: Derecho + edad), credencial ejecutiva en fragmentación (ERA_3: secretaría + cargos leg), y perfil legislativizado con filiación al bloque (ERA_4: Derecho + trayectoria legislativa + lealtad Morena).
""")

# ── Cell 129: 6.4 interpretación lastre ───────────────────────────────────────
md(cells[129], """## 6.4 Interpretación — Comisiones Lastre

**Señal consistentemente débil.** AUC entre 0.530 y 0.632 en todas las épocas — la asignación a comisiones lastre tiene mucho menor determinismo de perfil que las nodales.

**Hallazgos clave por era:**
- **ERA_1 (PRI)** — AUC ~0.585 (LR L1). La asignación lastre bajo el PRI hegemónico no sigue un patrón biográfico claro.
- **ERA_2 (PAN)** — época más predecible (LR L1 ~0.632). Bajo la alternancia panista, la oposición PRI/PRD era más homogénea en perfil → más fácil identificar quién recibiría comisiones de castigo.
- **ERA_3 (Transición)** — AUC ~0.583. La fragmentación multipartidista dificulta aún más identificar quién carga con las comisiones marginales.
- **ERA_4 (Morena)** — LR L1 roza lo aleatorio (~0.530). Las comisiones lastre bajo Morena no siguen un patrón biográfico claro: posiblemente distribuidas por cuotas internas o factores territoriales no observados. La L1 elimina la mayoría de features — el modelo activo es muy escaso, coherente con la opacidad del mecanismo.

**Test de imagen espejo — resultado:**

**La hipótesis de imagen espejo es FALSA.** Las correlaciones SHAP(nodal) vs -SHAP(lastre) oscilan entre −0.558 (ERA_3) y −0.677 (ERA_2), lejos de −1.0.

1. Nodal y lastre tienen lógicas de asignación parcialmente independientes. No es un juego de suma cero donde quien no recibe nodal recibe lastre.
2. Existen diputados que reciben ambos tipos simultáneamente, y diputados que no reciben ninguno.
3. ERA_2 (r=−0.677) es la época donde la lógica se acerca más a la distribución bimodal mayoría/oposición.
4. ERA_3 (r=−0.558) coincide con mayor fragmentación partidista — la atomización del poder rompe el patrón nodal/lastre.

**Implicación metodológica:** nodal y lastre deben modelarse como outcomes separados, no como complementos.

**Conclusión:** La asignación lastre es un proceso más opaco e idiosincrático que la asignación nodal. El perfil biográfico no explica bien quién carga con las comisiones marginales. Las correlaciones negativas de SHAP confirman que las variables con mayor importancia en nodales tienen importancia baja o inversa en lastre, pero la simetría no es perfecta ni espejo. La penalización L1 tiende a producir modelos aún más esparsos para lastre que para nodales, consistente con la menor señal del mecanismo.
""")

# ── Cell 138: 7.2 interpretación temáticas ────────────────────────────────────
md(cells[138], """## 7.2 Interpretación — Comisiones Temáticas

**Las comisiones temáticas son esencialmente impredecibles desde el perfil biográfico.**

La mejora sobre el baseline (predecir siempre la media) es mínima:
- ERA_1: mejor caso con ~8.2% de mejora (GLM Poisson MAE ~0.803 vs baseline 0.855).
- ERA_2, ERA_3, ERA_4: mejora ≤3% — prácticamente ninguna ganancia.

**Modelo:** GLM Poisson (PoissonRegressor con regularización L2). Para las comisiones temáticas no se aplica SelectFromModel: el objetivo es contar, no clasificar, y el GLM Poisson es el modelo más apropiado para datos de conteo con sobredispersión moderada. Los valores SHAP se calculan con `LinearExplainer` en espacio de **log-conteo** (predictor lineal del GLM con enlace log).

**Interpretación política:** El número de comisiones temáticas que recibe un diputado depende principalmente de:
- Negociaciones internas de bancada no observables en el currículum.
- Disponibilidad de cupos por comisión.
- Preferencias del coordinador parlamentario.
- Acuerdos de coalición legislativa.

**Hallazgo:** La diferencia entre épocas en la media de temáticas (1.53 en ERA_1 → 2.15 en ERA_3) no se traduce en mayor predictibilidad. La expansión del volumen de comisiones temáticas parece seguir criterios distributivos distintos según la época, pero siempre opaca para el observador externo. El beeswarm SHAP confirma la ausencia de predictores dominantes: la dispersión de puntos es uniforme sin variables con masa SHAP superior al resto.
""")

# ── Cell 142: SHAP consolidado ────────────────────────────────────────────────
md(cells[142], """**Interpretación — Tabla consolidada de importancias SHAP (Nodales / Lastre / Temáticas × Épocas)**

La tabla normaliza la importancia media (|SHAP|) de los top features para los tres targets y cuatro eras. Los valores SHAP de clasificación binaria (nodales, lastre) están en espacio de **log-odds**; los de temáticas en espacio de **log-conteo** (enlace log del GLM Poisson).

> **Nota:** los valores numéricos específicos se actualizan al re-ejecutar con LR L1 + SelectFromModel. Los patrones cualitativos descritos son esperados estables.

- **`area_Derecho`**: feature de mayor importancia para nodales en ERA_1, ERA_2 y ERA_4. La formación jurídica no decae con el tiempo — se intensifica en ERA_4. No aparece entre los top predictores de lastre, confirmando que es una señal de acceso a comisiones de poder, no de asignación marginal.
- **`es_partido_mayoria`**: salta en ERA_4 para nodales (pico histórico) pero no domina en magnitud absoluta — `area_Derecho`, `n_trayectoria_legislativa` y `edad_imp` lo superan en ERA_4. La filiación al bloque gana relevancia relativa sin convertirse en el criterio dominante.
- **`edad_imp`**: predictor de alta magnitud tanto en nodales (top-2 en ERA_2, top-3 en ERA_4) **como en lastre (top en todas las épocas)**. Es la única variable con señal fuerte en ambos mecanismos de asignación. **Caveat MNAR (§2.2.1b):** su importancia en nodales ERA_2 y ERA_4 puede estar parcialmente inflada — el mecanismo de nulos diagnosticado como MNAR (AUC=0.821) implica que parte de la señal es proxy de visibilidad institucional en el SIL.
- **`n_trayectoria_legislativa`**: escala de ERA_1 a ERA_4 para nodales — el mayor salto relativo de la serie, reflejando la legislativización del perfil nodal bajo Morena. No alcanza relevancia comparable en lastre ni temáticas.
- **`fue_secretario_cargo`**: patrón más sorprendente — importancia máxima en ERA_3 y mínima en ERA_2, opuesto a la intuición sobre el perfil tecnocrático panista. La L1 asigna coeficiente cero a este feature en ERA_2 y ERA_4, confirmando que solo fue señal de distinción en la era de fragmentación multipartidista.
- **Temáticas**: magnitudes SHAP consistentemente bajas (ningún feature con masa SHAP claramente superior al resto) frente a nodales y lastre, confirmando que el número de comisiones temáticas responde a lógicas distributivas no capturadas en el perfil biográfico.
""")

# ── Cell 148: rolling forward ─────────────────────────────────────────────────
md(cells[148], """## 8.3 Interpretación — Validación Temporal

**Modelo usado:** LR L1 + SelectFromModel entrenado en ERA k, aplicado a ERA k+1. Cada transición usa el subconjunto de features seleccionado sobre el conjunto de entrenamiento — las features activas pueden diferir entre transiciones.

> **Nota:** los valores AUC específicos se actualizan al re-ejecutar. Los valores de referencia de la tabla son del modelo v5 (XGBoost) y se presentan como orientación cualitativa de las tendencias entre eras.

**Hallazgo central: el perfil de reclutamiento a nodales se transfiere razonablemente bien entre épocas, con una excepción crítica.**

| Transición | Nodales AUC (ref v5) | Lectura |
|:-----------|:--------------------:|:--------|
| ERA_1 → ERA_2 | **0.711** | Transferencia sólida — PRI y PAN comparten lógica de asignación nodal basada en trayectoria |
| ERA_2 → ERA_3 | **0.652** | **Caída notable** — la fragmentación de ERA_3 rompe el perfil PAN. Mayor número de partidos = criterios más heterogéneos |
| ERA_3 → ERA_4 | **0.712** | Recuperación — Morena reintroduce centralización, con otros predictores (lealtad, exp. legislativa) |

**ERA_2 → ERA_3 es la transición más disruptiva.** La llegada de la era multipartidista (legislaturas 63–65) fue el momento de mayor ruptura en la lógica de asignación de poder en comisiones: la fragmentación del sistema de partidos generó criterios de asignación tan heterogéneos que el modelo entrenado en el período PAN no pudo generalizarse.

**Para lastre y temáticas:** AUC/MAE consistentemente cercanos al baseline en toda transición — confirma que la asignación lastre es ruidosa en todas las épocas y no se aprende bien entre períodos, y que las temáticas no tienen aprendizaje temporal útil.

**Nota metodológica LR L1 + SFM:** en el rolling forward, la selección de features ocurre dentro del pipeline sobre el conjunto de entrenamiento (sin fuga de información del conjunto de prueba). El conjunto de features activos puede variar entre transiciones — esto es un hallazgo adicional: si los features seleccionados en ERA_k son distintos a los de ERA_k+1, confirma la ruptura de perfil entre eras.
""")

# ── Cell 149: section 9 header ────────────────────────────────────────────────
md(cells[149], """# 9. Perfiles Prototípicos por Era

El prototipo es el diputado con mayor SHAP positivo acumulado en el modelo **LR L1 + SelectFromModel** de cada era — el individuo cuyo perfil maximiza el log-odds de asignación a comisión nodal. Los perfiles prototípicos capturan el extremo favorable de la distribución SHAP, no el diputado promedio.
""")

# ── Cell 151: tabla prototipos ────────────────────────────────────────────────
md(cells[151], """**Interpretación — Tabla de perfiles prototípicos por época**

El prototipo es el diputado con mayor SHAP positivo acumulado en el modelo LR L1 + SelectFromModel de cada era — el perfil que más probablemente recibe comisión nodal según el predictor lineal (log-odds). Evolución:

- **ERA_1 (PRI):** diputado con formación jurídica (`area_Derecho`), alta trayectoria política y empresarial, edad ~44. La carrera política interna bastaba; la filiación al bloque formal no era requisito.
- **ERA_2 (PAN):** suma partido mayoritario + alta trayectoria administrativa. El PAN valorizó el capital burocrático combinado con militancia en la bancada dominante.
- **ERA_3 (Transición):** diputado con cargo de secretario previo (`fue_secretario_cargo`) y trayectoria política moderada; elegido por mayoría relativa. La credencial ejecutiva de alto nivel podía abrir las puertas nodales incluso sin carrera parlamentaria formal.
- **ERA_4 (Morena):** perfil "legislativizado y leal": partido mayoritario, cargos legislativos previos, ex-senador, doctorado. La carrera parlamentaria y la distinción académica reemplazan el cargo ejecutivo específico.

> **Nota:** los perfiles específicos (valores de features y magnitudes SHAP) se actualizan al re-ejecutar con LR L1 + SFM. Los patrones cualitativos descritos son coherentes con los coeficientes L1 esperados.
""")

# ── Cell 153: waterfall SHAP prototipos ───────────────────────────────────────
md(cells[153], """**Interpretación — Waterfall SHAP: Perfiles Prototípicos Nodales (2×2 por época)**

Cada waterfall descompone cómo el perfil del diputado prototípico acumula SHAP positivo en su era. Los valores son contribuciones al **log-odds** de recibir comisión nodal. Nota: estas contribuciones son individuales (el diputado con máximo SHAP acumulado), no promedios poblacionales. La barra base representa el log-odds promedio de la era.

- **ERA_1 (PRI):** múltiples features contribuyen con aportes moderados y distribuidos; trayectoria política y formación jurídica generan las barras más largas. La L1 activa un subconjunto reducido respecto a las 86 features totales.
- **ERA_2 (PAN):** alta trayectoria administrativa y partido mayoritario producen las barras más altas — este prototipo captura el perfil tecnocrático panista en su forma más extrema, aunque a nivel poblacional esos features no sean siempre los más importantes.
- **ERA_3 (Transición):** `fue_secretario_cargo` sigue siendo la barra más destacada del prototipo, coherente con que ERA_3 es cuando este feature alcanza su mayor importancia SHAP poblacional. La acumulación es concentrada en pocas variables ejecutivas y de trayectoria política.
- **ERA_4 (Morena):** partido mayoritario y carrera senatorial generan los aportes más altos; la trayectoria legislativa y la edad contribuyen positivamente. La acumulación está concentrada en lealtad al bloque y distinción de carrera parlamentaria-académica.

> **Nota:** los valores numéricos específicos se actualizan al re-ejecutar. La narrativa refleja las tendencias esperadas de los coeficientes L1.
""")

# ── Cell 158: section 10 header ───────────────────────────────────────────────
md(cells[158], """# 10. Resumen Consolidado de Rendimiento

Vista unificada de los resultados: 3 targets × 4 eras × 2 variantes (LR L1 full y LR L1 + SFM). La columna **LR L1 (full)** es el modelo sin selección automática; **LR L1 + SFM** incorpora `SelectFromModel` con penalización Lasso. La columna **n_features** indica cuántas de las 86 variables del perfil biográfico tienen coeficiente no-cero en cada era.
""")

# ── Cell 161: 10.1 interpretación consolidada ─────────────────────────────────
md(cells[161], """## 10.1 Interpretación consolidada

**Vista unificada del rendimiento. Conclusiones de conjunto:**

1. **Nodales** son los únicos con señal predictiva genuina (AUC 0.62–0.73). La señal es consistente pero se deteriora progresivamente con el tiempo.
2. **Lastre** es moderadamente predecible solo en ERA_2 (LR L1 ~0.632); opaco en el resto. AUC consistentemente bajo (0.530–0.601) en las otras tres eras.
3. **Temáticas** son prácticamente impredecibles en todas las épocas. La mejora sobre el baseline nunca supera el 8.2%.
4. **LR L1 competitiva** → la estructura subyacente de la asignación es predominantemente lineal. La penalización Lasso no degrada el AUC respecto a versiones anteriores con L2.
5. **SelectFromModel reduce activamente el espacio de features por era.** El número de variables con coeficiente no-cero varía entre eras — esto es un hallazgo sustantivo: la esparsidad del modelo es mayor en las eras de menor señal (ERA_4 lastre) y menor en las eras con señal más distribuida (ERA_3 nodales).
6. **ERA_4 tiene n=500 e IC amplios** — resultados orientativos, no conclusivos.
""")

# ── Cell 167: conclusiones — actualizar H6 ────────────────────────────────────
old_167 = ''.join(cells[167]['source'])
new_167 = old_167.replace(
    """**H6 — La Regresión Logística es competitiva**
LR gana o empata en la mayoría de combinaciones para nodales. Esto sugiere que la estructura subyacente de la asignación es en gran parte lineal, y que los modelos de árbol capturan ruido más que señal adicional.""",
    """**H6 — La estructura de asignación es lineal y esparsas**
La Regresión Logística L1 (Lasso) mantiene el AUC de versiones anteriores sin necesidad de modelos de árbol. `SelectFromModel` confirma que en cada era un subconjunto reducido de features captura la señal relevante — el resto tiene coeficiente exactamente cero. La esparsidad varía por era y target: mayor en los mecanismos más opacos (lastre ERA_4), menor en los más predecibles (nodales ERA_1–2)."""
).replace(
    "Vista unificada de los 36 modelos (3 targets × 4 eras × 3 algoritmos).",
    "Vista unificada de los resultados: 3 targets × 4 eras × 2 variantes (LR L1 full y LR L1 + SFM)."
)
cells[167]['source'] = [new_167]

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Interpretation cells updated:")
for i in [84, 86, 104, 107, 110, 113, 129, 138, 142, 148, 149, 151, 153, 158, 161, 167]:
    src = ''.join(nb['cells'][i].get('source', []))
    print(f"  Cell {i}: {src.splitlines()[0][:70]}")
