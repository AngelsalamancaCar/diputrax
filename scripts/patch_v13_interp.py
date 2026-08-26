# -*- coding: utf-8 -*-
"""Fase 3 de plan10upd.md: reescribe las celdas markdown interpretativas de
diputraxv13.ipynb con los números de la corrida ejecutada (S3, solo regresión).

También corrige 61->62 features en los textos estructurales de fase 1 y
actualiza el dict del test espejo (celda 114, fuente + output mostrado, sin
alterar ningún valor calculado).

Verifica anclas contra el archivo vigente; aborta si no coinciden.
"""
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "diputraxv13.ipynb"

FULL = {}

# ============================ EDA ============================

FULL[16] = ("### Interpretación — Variables con datos nulos por época", """\
### Interpretación — Variables con datos nulos por época

Solo tres variables presentan valores faltantes:

- `edad_al_tomar_cargo` — 10.2 %
- `y_nacimiento` — 10.1 %
- `distrito_circ` — 4.2 %

El último caso ocurre por diseño, ya que las candidaturas de diputaciones plurinominales no tienen distrito asignado.

La gráfica por épocas revela que el patrón es heterogéneo: la **Tercera época** (LXIII–LXVI) concentra el mayor porcentaje agregado de nulos en `edad_al_tomar_cargo` (14.4 %), pero con fuerte heterogeneidad interna — la LXIII alcanza ~28 % (inconsistencias del SIL durante la migración de plataforma) mientras la LXVI tiene la mejor cobertura de toda la serie (0.8 %). La Segunda época es la de menor tasa (4.5 %).

La desagregación por era y por legislatura permite tratar los nulos de forma localizada en el modelado, en lugar de imputar con un promedio global.

---

### 2.2.1b Mecanismo de valores faltantes en `edad_al_tomar_cargo` — MCAR / MAR / MNAR

La imputación por media de legislatura es válida bajo mecanismos **MCAR** o **MAR**, pero introduce sesgo sistemático bajo **MNAR**.

> **v4:** Esta limitación se aborda con MICE condicionado a experiencia legislativa y administrativa —ver sección **2.2.1c** a continuación.

Si los diputados con menor perfil institucional tienen menos datos en el SIL, el valor imputado sobreestima sistemáticamente su edad real. En ese caso, el *flag* `edad_missing` no corrige la distorsión de dirección.

Este bloque diagnostica el mecanismo mediante tres pruebas empíricas:

1. **Tasa de nulos por era y partido**
   Evalúa si la ausencia de datos es uniforme entre grupos.

2. **Comparación de perfil entre registros con y sin edad**
   Utiliza prueba de Mann-Whitney y *boxplots* para identificar diferencias sistemáticas.

3. **Regresión logística para predecir la ausencia**
   Evalúa si las variables observables pueden predecir la falta de dato, funcionando como prueba del mecanismo MAR.
""")

FULL[21] = ("### Interpretación — Mecanismo de valores faltantes", """\
### Interpretación — Mecanismo de valores faltantes en `edad_al_tomar_cargo`: diagnóstico MNAR

**Veredicto:** MNAR confirmado, severidad alta.

---

#### Prueba 1 — Distribución de nulos por era y legislatura: MCAR descartado

La tasa de valores nulos varía de **0.8 %** en la LXVI a **28.0 %** en la LXIII entre legislaturas.

Por era, la distribución es la siguiente:

| Era | Tasa de nulos |
|---|---:|
| ERA_1 (57–59) | 10.4 % |
| ERA_2 (60–62) | 4.5 % |
| ERA_3 (63–66) | 14.4 % |

La heterogeneidad es demasiado extrema para ser aleatoria. La **ERA_3** concentra el problema por la migración de plataforma del SIL en las legislaturas LXIII–LXV; su propio tramo final (LXVI, 0.8 %) tiene en cambio la mejor cobertura de la serie. El supuesto **MCAR** queda descartado.

---

#### Prueba 2 — Comparación de perfil: todos los deltas negativos y significativos

Los **510 registros sin edad** presentan trayectorias sistemáticamente menores que los **4,490 registros con dato**. Las diferencias son estadísticamente significativas en todas las variables de trayectoria (`p = 0.0000`).

| Variable | Media con dato | Media sin dato | Delta | p MW |
|---|---:|---:|---:|---:|
| Tray. política | 6.460 | 3.139 | −3.321 | < 0.001 |
| Tray. administrativa | 3.281 | 2.151 | −1.130 | < 0.001 |
| Tray. empresarial | 1.721 | 0.867 | −0.854 | < 0.001 |
| Grado de estudios | 3.718 | 2.982 | −0.735 | < 0.001 |
| Tray. legislativa | 0.864 | 0.271 | −0.594 | < 0.001 |
| Cargos leg. previos | 0.574 | 0.167 | −0.407 | < 0.001 |

Los diputados sin edad registrada tienen, en promedio, **la mitad de trayectoria política** que los diputados con edad registrada. Esto sugiere que el SIL documenta menos a quienes tienen menor visibilidad pública e institucional, lo que es consistente con un mecanismo **MNAR**.

---

#### Prueba 3 — Regresión logística: AUC = 0.787

Un modelo logístico con variables de trayectoria y era predice la ausencia del dato con:

- **AUC:** 0.787
- **Pseudo R² de McFadden:** 0.156
- **N:** 5,000

Los coeficientes más grandes son negativos:

| Variable | Coeficiente |
|---|---:|
| `n_cargos_legislativos_prev` | −0.719 |
| `n_trayectoria_empresarial` | −0.203 |
| `n_comisiones_lastre` | −0.166 |
| `ERA_3` | +0.314 |

El coeficiente positivo de `ERA_3` confirma que la migración de plataforma agravó la cobertura en las legislaturas 63–65.

---

#### Prueba 4 — Correlación con el *outcome*: MNAR respecto al *target*

Los diputados sin dato de edad tienen menos comisiones nodales (media 0.514 vs. 0.680, `p < 0.001`); entre quienes tienen ≥1 nodal, la tasa de `edad_missing` es 8.1 % frente a 11.7 % de quienes no la tienen (`p < 0.001`). También presentan menor participación en comisiones temáticas (tasa de nulo 9.1 % con temáticas vs. 19.0 % sin ellas, `p < 0.001`). La comisión lastre, en cambio, no difiere entre grupos (`p = 0.71`).

Esto confirma la forma más severa de **MNAR** para el modelo de comisiones nodales: el dato faltante está correlacionado con el *outcome* de interés.

---

#### Prueba 5 — Brecha de género: mujeres con mayor tasa de nulo en las eras tempranas

Las mujeres presentan mayor tasa de `edad_missing` en ERA_1 (13.0 % vs. 9.7 %) y ERA_2 (8.2 % vs. 2.6 %, χ² p < 0.001). En el tramo final de la serie (LXVI) la brecha desaparece (0.8 % en ambos sexos). Esto introduce un sesgo adicional en la comparación de edades por género en ERA_1 y ERA_2.

---

#### Implicación metodológica

| Componente | Impacto del MNAR |
|---|---|
| `edad_imp` en modelo nodales ERA_2 — \\|SHAP\\| = 0.224 — y ERA_3 — \\|SHAP\\| = 0.108 | Importancia parcialmente inflada: absorbe señal de visibilidad institucional en el SIL, además de edad real. |
| `edad_missing` como *feature* | Mitiga el sesgo, ya que el modelo aprende que "dato ausente = menor perfil", pero no corrige la distorsión cuantitativa del valor imputado. |
| Lastre | Sin impacto significativo (`p = 0.71` entre grupos). |
| Temáticas | Impacto menor: los registros sin dato tienen menos temáticas, pero la señal del modelo ya es marginal. |
| Análisis de género | Caveat en ERA_1 y ERA_2: las mujeres están más subrepresentadas en el dato de edad. |

---

#### Acción adoptada

`edad_imp` se mantiene en el modelo porque sigue siendo señal real de edad.

El *flag* `edad_missing` se incluye como corrección parcial.

La importancia SHAP de `edad_imp` en ERA_2 y ERA_3 debe interpretarse con cautela: parte de la señal refleja visibilidad institucional en el SIL, no solo experiencia acumulada por edad.
""")

FULL[24] = ("**Interpretación — Composición partidista por época**", """\
**Interpretación — Composición partidista por época**

Las tres gráficas muestran el peso relativo de los principales partidos dentro de cada era:

- Primera época (LVII–LIX): el PRI domina pero sin mayoría absoluta; el PAN y PRD consolidan la primera alternancia real desde 1929.
- Segunda época (LX–LXII): bipartidismo funcional PAN–PRI, con el PRD como tercera fuerza y surgimiento de partidos satélite.
- Tercera época (LXIII–LXVI): el ciclo completo de Morena — desde posición minoritaria en la LXIII hasta la mayoría calificada (LXIV–LXV) y la supermayoría con aliados de la LXVI; el bloque opositor (PAN, PRI, PRD) queda progresivamente reducido y la oposición se fragmenta.

La segmentación por era hace visible la ruptura estructural que sería invisible en un gráfico de legislatura individual; dentro de la Tercera época, la serie por legislatura muestra la aceleración interna del ascenso de Morena.
""")

FULL[27] = ("**Interpretación — Distribución de edades por época**", """\
**Interpretación — Distribución de edades por época**

La distribución de edades es aproximadamente normal en todas las épocas, con un desplazamiento sostenido hacia la derecha conforme avanzan las eras:

- **Primera época:** media ≈ 44.8 años; distribución más joven y dispersa.
- **Segunda época:** media ≈ 45.8 años; perfil más compacto, coherente con bipartidismo estabilizado.
- **Tercera época:** media ≈ 47.7 años; la mayor del periodo. El efecto de la reelección consecutiva (vigente desde la LXIV) eleva la edad promedio al mantener en funciones a legisladores de mayor edad; por legislatura, la serie interna sube de 46.0 (LXIII) a 48.3–48.4 (LXIV, LXVI).

El envejecimiento progresivo del cuerpo legislativo es estadísticamente visible cuando se desagrega por era, mientras que la serie global esconde la discontinuidad post-2018.
""")

FULL[29] = ("**Interpretación — Edad promedio por época**", """\
**Interpretación — Edad promedio por época**

La gráfica de barras confirma el envejecimiento gradual del cuerpo legislativo: de 44.8 años en la Primera época a 47.7 en la Tercera. La línea por legislatura (coloreada por era) permite además identificar la variabilidad intra-era: el mayor salto ocurre entre la LXIII (46.0) y la LXIV (48.3), coincidiendo con la entrada en vigor de la reelección consecutiva (reforma 2014, efectiva desde 2018) que retiene en cámara a legisladores con carreras más largas. La tendencia ascendente es monotónica entre épocas, lo que sugiere un factor estructural y no aleatorio.
""")

FULL[32] = ("**Interpretación — Composición por sexo por época**", """\
**Interpretación — Composición por sexo por época**

La representación femenina muestra crecimiento sostenido con aceleración entre épocas:

- **Primera época (PRI, LVII–LIX):** proporción de mujeres entre 19.2 % (LVIII) y 24.8 % (LIX), promedio ≈21.5 %. La LIX sube 5.6 pp respecto a la LVIII, efecto de la cuota del 30 % aprobada en 2002.
- **Segunda época (PAN, LX–LXII):** avance sostenido de 25.8 % (LX) a 41.4 % (LXII). La LXII registra el mayor salto de legislatura a legislatura de la serie (+8.6 pp respecto a LXI), producto de la cuota del 40 % implementada en 2008.
- **Tercera época (Transición-Morena, LXIII–LXVI):** la cámara cruza el umbral de paridad en la LXV (50.2 %) por primera vez en su historia; la LXIV (48.8 %) ya era prácticamente paritaria y la LXVI (49.8 %) consolida la paridad como norma operativa. La reforma constitucional de paridad de 2019 opera como piso vinculante.

La tendencia es monotónica entre épocas. La discontinuidad normativa (cuotas → paridad) es visible en la gráfica: ERA_1 y ERA_2 son épocas de avance incremental sujeto a cuotas; la ERA_3 consolida la paridad como estándar constitucional en su tramo final.
""")

FULL[34] = ("**Interpretación — Proporción de mujeres por partido y época**", """\
**Interpretación — Proporción de mujeres por partido y época**

La variación inter-partidaria refleja la cultura organizacional de cada fuerza política:

- **Primera época:** el PRD registra la mayor proporción femenina entre los partidos grandes, coherente con su herencia de izquierda que priorizó la participación de mujeres antes de que existieran cuotas obligatorias. El PRI y PAN muestran valores menores, similares entre sí.
- **Segunda época:** el PRD mantiene el liderazgo en paridad. El PAN mejora su proporción femenina respecto a ERA_1 pero no supera al PRD. El PRI converge hacia la media de la era impulsado por la cuota del 40 %.
- **Tercera época:** Morena debuta con proporción femenina alta (cercana o superior al 48 %), consistente con su plataforma de paridad progresiva, y hacia el final del tramo la varianza inter-partidaria se reduce drásticamente: la paridad constitucional de 2019 funcionó como igualador entre organizaciones con culturas de género distintas — el bloque opositor (PAN, PRI, PRD) y el bloque de Morena convergen en proporciones cercanas al 50 %.

La reducción de la dispersión entre partidos entre ERA_1 y el final de ERA_3 es la señal más relevante del gráfico: la heterogeneidad cultural en la representación femenina fue erosionada por la presión normativa, no por convergencia espontánea.
""")

FULL[39] = ("**Interpretación — Distribución de grado de estudios por época**", """\
**Interpretación — Distribución de grado de estudios por época**

La comparación entre épocas revela una mejora progresiva en el nivel educativo declarado:

- **Primera era:** el porcentaje sin dato supera el 46 %; entre quienes tienen registro predomina la licenciatura (40.3 %), con presencia visible de preparatoria y licenciatura incompleta.
- **Segunda era:** crece la licenciatura completa (49.9 %) y se reduce la categoría sin dato (31.5 %).
- **Tercera era:** la licenciatura se consolida como nivel modal (49.1 %) y el doctorado alcanza su máximo de la serie (5.4 %, frente a 2.5 % en ERA_1).

La anomalía conocida de la LIX (promedio ordinal 1.49) se concentra en la Primera época y no contamina las eras posteriores. La comparación por era permite detectar que la mejora educativa es gradual y real, no un artefacto de la calidad de datos.
""")

FULL[42] = ("**Interpretación — Grado de estudios promedio por partido y época**", """\
**Interpretación — Grado de estudios promedio por partido y época**

El ranking educativo por partido cambia entre épocas:

- **Primera era (PRI dominante):** el PRI encabeza el promedio entre los partidos grandes (3.40), con PAN (3.04) y PRD (2.83) por debajo; los partidos pequeños muestran valores bajos y volátiles.
- **Segunda era (PAN):** los partidos chicos con pocas curules (MC, PANAL, Convergencia) exhiben promedios altos por n reducido; entre los grandes, el PAN (3.96) mantiene el perfil universitario coherente con su imagen tecnocrática.
- **Tercera era (Transición-Morena):** el ranking se comprime — PRD, PT y PVEM aparecen arriba con muestras medianas, y Morena (3.96, n=756) se sitúa en la media alta. La entrada masiva de cuadros nuevos no degradó el promedio educativo de la cámara.

La comparación por era evita el sesgo que produce agregar partidos de épocas distintas con composiciones de muestra incomparables.
""")

FULL[44] = ("**Interpretación — Área de formación por época**", """\
**Interpretación — Área de formación por época**

El Derecho domina en todas las épocas, pero su peso relativo disminuye levemente en la Tercera época, donde crecen las áreas de Ciencias Políticas/Administración Pública y Economía/Administración. Esto es consistente con la entrada de cuadros de Morena formados en administración pública y movimientos sociales más que en el litigio jurídico tradicional. La Ingeniería mantiene presencia baja pero estable. La Medicina aparece con mayor frecuencia en la Segunda época (PAN), coincidiendo con el perfil tecnocrático de ese partido. La comparación por era revela que el perfil formativo de la clase legislativa no es estático sino que refleja la cultura organizacional del partido dominante en cada periodo.
""")

FULL[46] = ("### Interpretación — Tipo de universidad por época", """\
### Interpretación — Tipo de universidad por época

| Época | Pública | Privada | Extranjera | Sin info |
|---|---:|---:|---:|---:|
| Primera — PRI | 53.7 % | 23.3 % | 10.3 % | 34.2 % |
| Segunda — PAN | 40.3 % | 21.7 % | 8.1 % | 45.6 % |
| Tercera — Transición-Morena | 39.1 % | 22.1 % | 6.5 % | 44.7 % |

#### Primera época

Máxima presencia de egresados de universidad pública (53.7 %) y de formación en el extranjero (10.3 %). El PRI hegemónico reclutó simultáneamente desde universidades públicas y desde una élite con trayectoria internacional.

#### Segunda época

Se observa una caída de **13.3 puntos porcentuales** en universidad pública (de 53.7 % a 40.3 %), el mayor descenso de la serie. La universidad privada no aumenta (23.3 % a 21.7 %); el cambio más pronunciado es el salto en la categoría **"sin dato"**, que pasa de 34.2 % a 45.6 %. La narrativa del "PAN tecnocrático-privado" no está respaldada por un aumento de egresados de universidades privadas — lo que cambia es la menor presencia de la UNAM entre quienes declaran universidad (ver la tabla siguiente).

#### Tercera época

La distribución pública/privada es prácticamente idéntica a ERA_2 (39.1 % / 22.1 %) y la formación en el extranjero continúa su descenso hasta el mínimo de la serie (6.5 %). La cobertura de datos universitarios no mejora (44.7 % sin dato).

#### Cautela interpretativa

Con casi la mitad de los registros sin información universitaria en ERA_2 y ERA_3, el descenso de la universidad pública puede reflejar parcialmente mayor subregistro en el SIL para esas eras, no solo un cambio real en el perfil educativo.
""")

FULL[48] = ("### Interpretación — Indicadores de universidad principales por época", """\
### Interpretación — Indicadores de universidad principales por época

| Institución | ERA_1 PRI | ERA_2 PAN | ERA_3 Trans.-Morena |
|---|---:|---:|---:|
| UNAM | 22.4 % | 12.3 % | 7.4 % |
| IBERO | 6.7 % | 5.3 % | 4.3 % |
| ITESM | 6.3 % | 6.8 % | 4.4 % |
| IPN | 4.7 % | 2.8 % | 1.3 % |
| ITAM | 4.4 % | 3.2 % | 2.0 % |
| UDG | 3.5 % | 3.4 % | 2.9 % |

#### Primera época

La UNAM domina con **22.4 %** de la Cámara, el pico histórico de la serie. El ITAM también alcanza su máximo en **ERA_1** (4.4 %), no en ERA_2 como podría suponerse del perfil panista. El PRI hegemónico reclutaba simultáneamente desde la UNAM y desde el ITAM, reflejo de la coexistencia de un ala populista y un ala tecnocrática dentro del mismo partido.

#### Segunda época

El ITESM (no el ITAM) alcanza su pico histórico, con **6.8 %**. El ITAM cae de 4.4 % a 3.2 %. La narrativa del "PAN ITAM" está parcialmente respaldada en composición de bancada, pero el ITESM es el vector privado-tecnocrático más representado en la Cámara de ERA_2. La UNAM se desploma de **22.4 %** a **12.3 %**, el mayor descenso de la serie.

#### Tercera época

Descenso monotónico en todas las instituciones identificadas: la UNAM queda en **7.4 %** (un tercio de su peso en ERA_1); ITESM e ITAM caen a 4.4 % y 2.0 %.

#### Interpretación general

Este patrón confirma que la procedencia institucional específica es señal relevante en las dos primeras eras y decrece en la Tercera, donde la filiación política y la carrera parlamentaria desplazan a la credencial universitaria como predictor de acceso a comisiones nodales (ver §5).
""")

FULL[51] = ("**Interpretación — Experiencia legislativa previa por época**", """\
**Interpretación — Experiencia legislativa previa por época**

El patrón varía estructuralmente entre épocas:

- **Primera época:** la fracción de ex diputados locales (20.8 %) y ex diputados federales (11.3 %) es moderada; la prohibición de reelección consecutiva vigente desde 1933 limita la acumulación de mandatos federales consecutivos.
- **Segunda época:** sube la experiencia local (36.2 %) y federal (16.5 %); el bipartidismo PAN–PRI favorece la circulación de cuadros entre niveles de gobierno.
- **Tercera época:** la proporción con mandato federal previo salta a 25.6 % — el efecto de la reelección consecutiva activa desde la LXIV es visible en la gráfica de legislaturas, que muestra la discontinuidad en 2018 dentro del propio tramo.

La comparación por era, complementada con la serie por legislatura, hace evidente que la reforma de 2014 produjo una discontinuidad real, no solo incremental.
""")

FULL[53] = ("**Interpretación — Experiencia administrativa previa por época**", """\
**Interpretación — Experiencia administrativa previa por época**

La presidencia municipal es el cargo ejecutivo previo más frecuente en todas las épocas, pero su peso relativo y el de otros cargos cambia:

- **Primera época:** la presidencia municipal y los cargos de dirección en organizaciones (sindicatos, cámaras) son dominantes, reflejando la estructura corporativa del PRI.
- **Segunda época:** crece la fracción de ex secretarios y subsecretarios; el PAN reclutó más intensamente de la burocracia federal tecnocrática.
- **Tercera época:** la presidencia municipal se mantiene como vía principal — Morena construye desde gobiernos locales y movimientos sociales — mientras la fracción con cargos de dirección general y secretarías se diluye hacia el final del tramo, diversificándose el origen administrativo.

Este patrón es consistente con los hallazgos SHAP (§5): la señal administrativa (`fue_secretario_cargo`) es máxima en ERA_1 y pierde peso relativo frente a la carrera legislativa en ERA_3.
""")

FULL[55] = ("**Interpretación — Trayectorias por tipo y época**", """\
**Interpretación — Trayectorias por tipo y época**

La gráfica de barras agrupadas muestra promedios por tipo de trayectoria en cada era:

- **Trayectoria política** es la más alta en todas las épocas, pero desciende de forma sostenida: 7.38 (ERA_1) → 6.74 (ERA_2) → 4.71 (ERA_3).
- **Trayectoria administrativa** alcanza su pico en la Segunda época (3.52), coherente con el reclutamiento tecnocrático panista, y cae en la Tercera (3.06).
- **Trayectoria legislativa** crece de forma monotónica (0.68 → 0.80 → 0.90), con la discontinuidad interna en la LXIV (primera legislatura con reelección activa) visible en la serie por legislatura.
- **Trayectoria empresarial** es la más baja y estable del periodo (1.5–1.9).

El cruce de las dos tendencias — militancia/gestión a la baja, carrera parlamentaria al alza — es el trasfondo descriptivo del cambio de perfil nodal que el modelado documenta en §5.
""")

FULL[58] = ("**Interpretación — Experiencia en comisiones por partido y época**", """\
**Interpretación — Experiencia en comisiones por partido y época**

Las tres gráficas muestran cómo se distribuyen las comisiones entre los principales partidos de cada era:

- Primera época: el PRI concentra las comisiones nodales y presidencias; el PAN y PRD acumulan relativamente más comisiones lastre como oposición emergente.
- Segunda época: el PAN lidera en comisiones nodales y presidencias; el PRI en oposición y el PRD muestran mayor proporción de comisiones temáticas y lastre.
- Tercera época: Morena asciende hasta concentrar el promedio nodal más alto entre los partidos grandes (0.89 por diputado), con sus aliados (PVEM 0.97, PT 0.90) también arriba; la fragmentación primero, y la supermayoría después, reparten presidencias entre la coalición dominante y dejan al bloque opositor con promedios menores de comisiones de poder.

Este patrón es la evidencia descriptiva central del estudio: la distribución de comisiones no es aleatoria sino sistemáticamente diferencial por partido y época, lo que justifica el modelado predictivo.
""")

FULL[61] = ("### Interpretación — Brechas de género en comisiones", """\
### Interpretación — Brechas de género en comisiones

| Tipo | ERA_1 M | ERA_1 H | Δ | ERA_2 M | ERA_2 H | Δ | ERA_3 M | ERA_3 H | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nodal — ≥1 | 27.3 % | 33.5 % | −6.2 pp | 34.8 % | 43.7 % | −8.9 pp | 44.9 % | 56.6 % | −11.7 pp |
| Lastre — ≥1 | 48.1 % | 40.4 % | +7.7 pp | 51.8 % | 43.0 % | +8.8 pp | 50.1 % | 44.6 % | +5.5 pp |
| Temáticas — media | 1.53 | 1.53 | ≈0 | 2.03 | 1.88 | −0.15 | 2.18 | 1.98 | −0.19 |
| Presidencias — media | 0.140 | 0.162 | −0.022 | 0.186 | 0.247 | −0.061 | 0.157 | 0.182 | −0.025 |

#### Cuatro hallazgos centrales

##### 1. La brecha nodal se amplía, no se cierra

A pesar del avance sostenido hacia la paridad electoral, la brecha de acceso a comisiones nodales crece monotónicamente entre eras y alcanza su máximo en **ERA_3** (−11.7 pp). La paridad electoral no se traduce automáticamente en paridad de poder comisionario.

> **Límite de resolución (S3):** la evidencia de v10 sugería una reducción de la brecha en el subtramo 2024+ (LXVI). Con la segmentación S3 ese subtramo queda agregado dentro de ERA_3 y la posible convergencia reciente no es observable en esta tabla; documentarla requiere el desglose por legislatura.

##### 2. Las mujeres reciben consistentemente más comisiones lastre en todas las épocas

La brecha favorable a las mujeres en lastre se reduce de +7.7/+8.8 pp en ERA_1–ERA_2 a +5.5 pp en ERA_3. La convergencia es real, pero la dirección no se invierte.

##### 3. Las comisiones temáticas son el espacio más igualitario

La diferencia es prácticamente nula en ERA_1 (1.53 vs. 1.53) y crece levemente a favor de las mujeres en ERA_2–ERA_3. No hay evidencia de discriminación activa en la asignación temática.

##### 4. Las presidencias muestran una brecha modesta pero persistente

Los hombres promedian consistentemente más presidencias; la brecha es máxima en **ERA_2** (0.061) y se modera en ERA_3 (0.025).

---

#### Interpretación general

El patrón conjunto (más comisiones lastre y más comisiones temáticas para mujeres, pero menos comisiones nodales y menos presidencias) confirma una **asignación diferencial selectiva**: la inequidad opera en los espacios de poder, no en la distribución general del trabajo legislativo.
""")

FULL[64] = ("**Interpretación — Matriz de correlación por época**", """\
**Interpretación — Matriz de correlación por época**

Desagregar la correlación por era revela que las relaciones entre variables no son estables en el tiempo:

- **Primera época (PRI):** la correlación entre n_trayectoria_admin y n_presidencias es más alta que en épocas posteriores, reflejando que el aparato burocrático priísta era la vía natural hacia comisiones de poder.
- **Segunda época (PAN):** la correlación entre grado_estudios_ord y n_comisiones_nodales es la más alta del periodo, consistente con el perfil tecnocrático del PAN.
- **Tercera época (Transición-Morena):** las correlaciones con las variables de trayectoria administrativa se debilitan en general — la fragmentación multipartidista y luego la recomposición de Morena introducen mayor heterogeneidad de perfiles — mientras la trayectoria legislativa gana asociación con las comisiones, coherente con la reelección consecutiva.

Las correlaciones entre tipos de trayectoria (administrativa, política, legislativa) son positivas en todas las épocas: el capital político sigue siendo acumulativo, aunque su composición cambia.
""")

FULL[66] = ("**Interpretación — Pairplots de trayectorias (uno por época)**", """\
**Interpretación — Pairplots de trayectorias (uno por época)**

Cada pairplot muestra la estructura de relaciones entre los cuatro tipos de trayectoria dentro de una sola era, lo que permite comparar patrones sin que el solapamiento entre eras distorsione la lectura.

- **Primera era (PRI):** la nube de n_trayectoria_politica vs n_trayectoria_admin concentra la mayor densidad en valores moderados-altos, reflejando la acumulación conjunta de capital político-burocrático como norma del reclutamiento priísta. n_trayectoria_empresarial muestra dispersión baja y desvinculada del resto.
- **Segunda era (PAN):** n_trayectoria_admin exhibe mayor varianza hacia valores altos que en ERA_1, consistente con el reclutamiento tecnocrático panista; n_trayectoria_legislativa aún tiene valores bajos dado que la reelección no era vigente.
- **Tercera era (Transición-Morena):** la nube de n_trayectoria_legislativa se desplaza hacia la derecha — efecto de la reelección consecutiva (vigente desde la LXIV) — y su correlación con la trayectoria política se intensifica, mientras la densidad en valores altos de trayectoria administrativa se reduce: los perfiles del tramo son más parlamentarios y menos burocráticos.

La trayectoria empresarial permanece la más desvinculada del resto en todas las épocas: quienes tienen experiencia empresarial no necesariamente acumulan cargos políticos o administrativos, y viceversa.
""")

# ============================ MODELADO ============================

FULL[88] = ("### Interpretación de hallazgos de SHAP Beeswarm Nodales", """\
### Interpretación de hallazgos de SHAP Beeswarm Nodales — grid 1×3 por época

Cada punto representa un diputado. El eje X muestra el valor SHAP, es decir, la contribución individual al *log-odds* de recibir comisión nodal. Los valores están expresados en espacio de *log-odds*: contribuciones al predictor lineal de la Regresión Logística L1.

---

#### ERA_1 — PRI

`area_Derecho` domina con `|SHAP| = 0.336`. Le siguen:

| Variable | \\|SHAP\\| |
|---|---:|
| `area_Derecho` | 0.336 |
| `fue_secretario_cargo` | 0.175 |
| `univ_privada` | 0.160 |
| `legislatura_num` | 0.128 |
| `reg_SUR` | 0.125 |
| `area_Económico-Financiera` | 0.119 |
| `n_organos_gobierno` | 0.112 |
| `mayoria_relativa` | 0.103 |

La formación jurídica y la credencial ejecutiva priísta (secretaría) son las señales centrales; la universidad privada como señal de distinción sugiere que la credencial de élite operaba como *proxy* de reclutamiento dentro del aparato priísta.

---

#### ERA_2 — PAN

`area_Derecho` lidera nuevamente, con `|SHAP| = 0.340`. Le siguen:

| Variable | \\|SHAP\\| |
|---|---:|
| `area_Derecho` | 0.340 |
| `edad_imp` | 0.224 |
| `tiene_posgrado` | 0.166 |
| `fue_director` | 0.161 |
| `mayoria_relativa` | 0.136 |
| `sexo_bin` | 0.123 |

`edad_imp` alcanza aquí su máximo histórico de la serie (caveat MNAR, §2.2.1b). `sexo_bin` emerge por primera vez como *feature* activo: bajo el PAN, el género adquiere poder predictivo sobre la asignación nodal.

---

#### ERA_3 — Transición-Morena

El liderazgo de `area_Derecho` (0.268) se estrecha frente a la carrera parlamentaria:

| Variable | \\|SHAP\\| |
|---|---:|
| `area_Derecho` | 0.268 |
| `n_trayectoria_legislativa` | 0.224 |
| `p_MORENA` | 0.199 |
| `sexo_bin` | 0.181 |
| `area_Ciencias Políticas y Sociales` | 0.127 |
| `univ_publica` | 0.121 |
| `tiene_posgrado` | 0.119 |
| `p_PAN` | 0.108 |
| `edad_imp` | 0.108 |
| `fue_secretario_cargo` | 0.105 |

`n_trayectoria_legislativa` alcanza su pico histórico (0.224): la experiencia parlamentaria acumulada es la segunda señal del grupo fusionado, consistente con la reelección consecutiva vigente desde 2018. `p_MORENA` (0.199) es la variable partidista activa — `es_partido_mayoria` es eliminada por L1 en las tres eras — y `sexo_bin` toca su máximo (0.181).

#### Hallazgo transversal

El grupo fusionado 63–66 promedia la fragmentación de la Transición y el dominio de Morena: el perfil resultante combina formación (Derecho, posgrado), carrera legislativa y filiación partidista. La resolución fina del subrégimen 2024+ (documentada en v10) queda fuera del alcance de esta segmentación.
""")

FULL[91] = ("### Interpretación — SHAP Heatmap Nodales", """\
### Interpretación — SHAP Heatmap Nodales — *features* × épocas

El *heatmap* codifica en color la importancia media —`|SHAP|`— de cada *feature* para cada era en espacio de *log-odds*. **Cero exacto** significa que el coeficiente fue eliminado por la penalización L1.

---

| Feature | ERA_1 | ERA_2 | ERA_3 | Interpretación |
|---|---:|---:|---:|---|
| `area_Derecho` | 0.336 | 0.340 | 0.268 | *Feature* de mayor `|SHAP|` en las tres eras. La señal más persistente de la serie: la formación jurídica no decae, aunque su ventaja se estrecha en ERA_3. |
| `n_trayectoria_legislativa` | 0.083 | 0.039 | 0.224 | Pico histórico en ERA_3 (segunda señal del grupo): la carrera parlamentaria escala con la reelección consecutiva. |
| `edad_imp` | 0.000 | 0.224 | 0.108 | Eliminado en ERA_1; pico en ERA_2; se modera en ERA_3. **Caveat MNAR** (§2.2.1b): parte de la señal en ERA_2 refleja visibilidad institucional en el SIL. |
| `fue_secretario_cargo` | 0.175 | 0.070 | 0.105 | Pico histórico en ERA_1: la credencial de secretaría fue una señal priísta. Cae bajo el PAN y repunta moderadamente en ERA_3. |
| `sexo_bin` | 0.000 | 0.123 | 0.181 | Ausente en ERA_1; emerge en ERA_2 y alcanza su máximo en ERA_3. Diferenciación sistemática de género en el acceso nodal. |
| `p_MORENA` | 0.000 | 0.000 | 0.199 | Variable partidista activa en ERA_3 — no `es_partido_mayoria`, que L1 elimina en las tres eras. |
| `tiene_posgrado` | 0.000 | 0.166 | 0.119 | Activo en ERA_2–ERA_3: el posgrado opera como señal en las eras tecnocrática y reciente. |
| `mayoria_relativa` | 0.103 | 0.136 | 0.000 | Relevante en el bipartidismo ERA_1–ERA_2; eliminada en ERA_3. |
| `univ_privada` | 0.160 | 0.080 | 0.000 | Señal de distinción decreciente; desaparece en ERA_3, donde `univ_publica` (0.121) toma su lugar. |
| `fue_director` | 0.000 | 0.161 | 0.068 | Credencial administrativa panista; se diluye en ERA_3. |

---

#### Nota sobre `SelectFromModel`

Los *features* con valor cero en una era tienen coeficiente exactamente igual a cero. Esto significa que fueron eliminados por la penalización L1, no simplemente que tienen baja importancia.
""")

FULL[94] = ("**Interpretación — Evolución temporal de importancias SHAP (Nodales)**", """\
**Interpretación — Evolución temporal de importancias SHAP (Nodales)**

Gráfica de líneas: variación del peso de features clave a lo largo de las tres épocas. Valores en log-odds (predictor lineal LR L1). Cero = eliminado por L1.

- `fue_secretario_cargo` (0.175 / 0.070 / 0.105): pico histórico en ERA_1 — la experiencia de secretaría fue señal del reclutamiento priísta; cae bajo el PAN y repunta parcialmente en el tramo reciente.
- `n_trayectoria_legislativa` (0.083 / 0.039 / 0.224): el mayor salto relativo de la serie — pico en ERA_3, efecto directo de la reelección consecutiva. Es la señal que "parlamentariza" el perfil nodal reciente.
- `edad_imp` (0.000 / 0.224 / 0.108): eliminado en ERA_1; pico en ERA_2; se modera después. Caveat MNAR (§2.2.1b).
- `mayoria_relativa` (0.103 / 0.136 / 0.000): activa solo durante el bipartidismo; eliminada en ERA_3.
- `fue_diputado_federal` (0.000 / 0.042 / 0.070): crece de forma monotónica pero moderada — la señal parlamentaria concreta la captura mejor `n_trayectoria_legislativa`.
- `n_trayectoria_admin` (0.076 / 0.000 / 0.000): activa solo en ERA_1 dentro de este conjunto — coherente con el declive de la vía burocrática.
- `es_partido_mayoria` y `n_trayectoria_politica` (0.000 en las tres eras): **eliminadas por L1 en toda la serie**. La primera es desplazada por las dummies partidistas concretas (`p_MORENA`, `p_PAN`); la segunda, por los indicadores específicos de cargo. La militancia genérica no es señal una vez condicionada al resto del perfil.

**Interpretación L1:** ausencia en una era = coeficiente exactamente cero. Más contundente que baja importancia en modelos no-lineales.
""")

FULL[97] = ("## 5.5 Interpretación — Comisiones Nodales", """\
## 5.5 Interpretación — Comisiones Nodales

Se observa una **señal moderada con descenso suave a lo largo del tiempo**. El AUC del modelo **LR L1 + SFM** pasa de **0.727** en `ERA_1` a **0.685** en `ERA_3`: el perfil biográfico sigue prediciendo el acceso nodal, pero pierde nitidez conforme el mecanismo de asignación se pluraliza.

### Hallazgos clave por era

#### ERA_1 — PRI

- **AUC:** 0.727 ± 0.030
- **Features activos:** 22

`area_Derecho` encabeza la importancia media (`0.336`), seguido de `fue_secretario_cargo` (`0.175`) y `univ_privada` (`0.160`). En esta era, la formación jurídica y la credencial ejecutiva priísta son las señales centrales, no la trayectoria de carrera política pura (`n_trayectoria_politica` es eliminada por L1).

#### ERA_2 — PAN

- **AUC:** 0.719 ± 0.031
- **Features activos:** 21

`area_Derecho` (`0.340`) y `edad_imp` (`0.224`) lideran, con pico histórico de `edad_imp`. También son relevantes `tiene_posgrado` (`0.166`) y `fue_director` (`0.161`), mientras que `sexo_bin` (`0.123`) emerge como señal activa.

> **Nota:** La importancia de `edad_imp` en `ERA_2` puede estar parcialmente inflada por MNAR (§2.2.1b).

#### ERA_3 — Transición-Morena

- **AUC:** 0.685 ± 0.016
- **Features activos:** 24

`area_Derecho` (`0.268`) mantiene el liderazgo pero con la ventaja más estrecha de la serie: `n_trayectoria_legislativa` (`0.224`, pico histórico) y `p_MORENA` (`0.199`) lo siguen de cerca, con `sexo_bin` (`0.181`) en su máximo. El perfil del grupo fusionado combina formación, carrera parlamentaria y filiación con la coalición dominante.

> Con n=2,000, el IC95 del AUC es ±0.023 — el más estrecho de la serie (en v10, la ERA_4 aislada tenía ±0.048 con n=500).

---

### Evolución del perfil nodal
**SHAP en log-odds**

| Feature | ERA_1 | ERA_2 | ERA_3 | Patrón |
|---|---:|---:|---:|---|
| `area_Derecho` | 0.336 | 0.340 | 0.268 | Señal persistente; top-1 en las tres eras |
| `n_trayectoria_legislativa` | 0.083 | 0.039 | 0.224 | Pico en ERA_3; parlamentarización del perfil |
| `edad_imp` | 0.000 | 0.224 | 0.108 | Pico en ERA_2; caveat MNAR |
| `fue_secretario_cargo` | 0.175 | 0.070 | 0.105 | Pico en ERA_1; credencial priísta |
| `sexo_bin` | 0.000 | 0.123 | 0.181 | Emerge en ERA_2; máximo en ERA_3 |
| `p_MORENA` | 0.000 | 0.000 | 0.199 | Variable partidista activa en ERA_3 |
| `tiene_posgrado` | 0.000 | 0.166 | 0.119 | Activa en ERA_2–ERA_3 |
| `mayoria_relativa` | 0.103 | 0.136 | 0.000 | Activa solo en el bipartidismo |

---

### Modelo

El modelo utilizado es **LR L1 + SelectFromModel**. Su estructura es predominantemente lineal, lo cual es coherente con la teoría, en tanto refleja reglas de partido relativamente formalizadas. La esparsidad varía por era, con entre **21 y 24 features activos** de un total de **62**.

---

### Tendencia central

La evolución del perfil nodal puede sintetizarse como una transición desde un perfil político-jurídico priísta hacia una lógica más parlamentarizada:

1. **ERA_1 — PRI:** Derecho + secretaría + credencial de élite.
2. **ERA_2 — PAN:** Derecho + edad + posgrado + credencial administrativa.
3. **ERA_3 — Transición-Morena:** Derecho + carrera legislativa + filiación Morena.

`sexo_bin` opera como barrera implícita desde `ERA_2`, con su máximo en `ERA_3`.

> **Límite interpretativo (§4.6):** la lectura partidista de `ERA_3` es asociativa, no causal — el modelo no separa la identidad partidista concreta del resto de atributos del bloque dominante. Y el detalle del subrégimen 2024+ (la legislativización extrema que v10 documentaba en la LXVI aislada) queda promediado dentro del grupo: es un límite de resolución declarado de la segmentación S3, no un hallazgo negativo.
""")

FULL[105] = ("**Interpretacion - Comparativa LR L1 vs. Bayesiano (Nodales)**", """\
**Interpretacion - Comparativa LR L1 vs. Bayesiano (Nodales)**

**AUC:** La diferencia entre AUC frecuentista (CV) y AUC Bayesiano (media posterior in-sample) refleja dos cosas distintas: el CV evalua generalizacion out-of-fold; el AUC Bayesiano evalua ajuste in-sample. Diferencias |Delta| < 0.02 indican consistencia. |Delta| > 0.04 puede senalar sobreajuste del Bayesiano.

**Forest plots:** Los coeficientes L1 son esparsos por construccion; los betas bayesianos tienen masa distribucional positiva incluso para features marginales. Donde ambos son del mismo signo con HDI que excluye cero, la senal es robusta al paradigma.

**Concordancia de direccion:** >= 85% indica ranking cualitativo estable. < 70% senala sensibilidad del resultado al estimador elegido.

**Resultados observados (v13):** Concordancia de direccion = 100% en las 3 eras (ERA_1: 22/22, ERA_2: 21/21, ERA_3: 24/24). AUC Bayesiano supera al LR L1 en todas las eras: ERA_1=0.765 (+0.036), ERA_2=0.746 (+0.025), ERA_3=0.725 (+0.035) — deltas homogéneos (0.025–0.036), en su mayor parte atribuibles a la comparación in-sample vs. out-of-fold. A diferencia de v10, ya no existe el delta atípico del grupo corto (la ERA_4 de n=500 mostraba +0.102). Features con HDI que excluyen cero: ERA_1=15, ERA_2=11, ERA_3=18 de los seleccionados por SFM.

**R-hat y ESS:** Convergencia MCMC impecable en los tres grupos — R-hat=1.000 y ESS_min=2667/4744/6227. Condicion necesaria para validez inferencial bayesiana, cumplida.

**Conexion con A&L (2009):** El Bayesiano confirma el ranking en ERA_1: `area_Derecho`, `fue_secretario_cargo` y las credenciales ejecutivas tienen betas positivos con HDI que excluye cero, replicando la prioridad de la experiencia burocratica sobre la legislativa en la era priista. En ERA_3, el HDI de `n_trayectoria_legislativa` y `p_MORENA` excluye cero: la parlamentarizacion del perfil es robusta al paradigma.
""")

FULL[111] = ("**Variables con mayor impacto en análisis SHAP: Nodales vs Lastre**", """\
**Variables con mayor impacto en análisis SHAP: Nodales vs Lastre**

Las barras enfrentadas muestran los features más importantes para cada tipo de comisión. Diferencias estructurales:

- `area_Derecho` (SHAP nodal: 0.336 / 0.340 / 0.268) no aparece entre los top predictores de lastre (solo 0.094 en ERA_2) — es señal de acceso a comisiones de poder, no de asignación marginal.
- `edad_imp` es la excepción notable: top-2 en nodales ERA_2 (0.224) y el predictor de mayor magnitud media en lastre (0.130), activo en las tres eras (0.133 / 0.151 / 0.105).
- `n_trayectoria_admin` aparece entre los top predictores de lastre en ERA_2 (0.173) pero no figura en el top de nodales de ninguna era — la trayectoria burocrática tiene señal en el mecanismo de asignación marginal panista, no en el nodal.
- `sexo_bin` aparece con importancia moderada en lastre (ERA_1=0.083, ERA_2=0.108) y con más fuerza en nodales (ERA_2=0.123, ERA_3=0.181) — señal bidireccional que opera en ambos tipos de asignación.
- En ERA_3 el lastre desarrolla marcadores propios: `n_trayectoria_legislativa` (0.126) y `univ_privada` (0.116) — perfil de oposición veterana — que no espejan a los predictores nodales del mismo grupo.
- Los coeficientes de correlación entre SHAP(nodal) y −SHAP(lastre) van de −0.095 (ERA_1) a −0.393 (ERA_2), lejos de −1.0. Nodal y lastre son mecanismos cualitativamente distintos, no simplemente opuestos.
""")

FULL[118] = ("## 6.4 Interpretacion -- Comisiones Lastre", """\
## 6.4 Interpretacion -- Comisiones Lastre

Senal consistentemente debil. AUC (LR L1 + SFM) entre 0.584 y 0.635.

Hallazgos por era:

- ERA_1 (PRI) -- AUC=0.584±0.015, 20 features activos. `nivel_cargo_max` (0.147) lidera, seguido de `edad_imp` (0.133) y `legislatura_num` (0.127). Ningun feature de trayectoria politica o juridica destaca -- el mecanismo de asignacion lastre en el PRI parece regido por variables de contexto institucional (nivel de cargo, momento de la legislatura) mas que por perfil de carrera.
- ERA_2 (PAN) -- AUC=0.635±0.032, 25 features activos. La epoca mas predecible. `n_trayectoria_admin` (0.173) y `n_organos_gobierno` (0.144) lideran -- la oposicion PRI/PRD era mas homogenea en perfil administrativo. `edad_imp` (0.151) tambien relevante.
- ERA_3 (Transicion-Morena) -- AUC=0.610±0.020, 25 features activos. `n_trayectoria_legislativa` (0.126), `univ_privada` (0.116) y `edad_imp` (0.105) encabezan: los diputados con mas trayectoria parlamentaria y formacion privada -- perfil de oposicion veterana -- concentran las comisiones lastre del tramo reciente.

**Test de imagen espejo -- resultado:**

**La hipotesis de imagen espejo es FALSA.** Correlaciones SHAP(nodal) vs -SHAP(lastre): ERA_1=-0.095, ERA_2=-0.393, ERA_3=-0.336 -- todas lejos de -1.0 (ver tabla en §6.2).

1. Nodal y lastre tienen logicas parcialmente independientes.
2. ERA_2 (r=-0.393) es la epoca mas cercana a una logica bimodal mayoria/oposicion; ERA_3 (r=-0.336) hereda un patron similar bajo el alineamiento post-2015.
3. Los predictores dominantes de lastre (`edad_imp`, `n_trayectoria_admin`, `univ_privada`, `nivel_cargo_max`) son distintos de los de nodales (`area_Derecho`, `n_trayectoria_legislativa`, `sexo_bin`, `p_MORENA`).

**Implicacion metodologica:** nodal y lastre deben modelarse como outcomes separados.

**Nota de alcance:** a diferencia de nodales (§5.5), en lastre `es_partido_mayoria`/`p_MORENA` no figuran entre los predictores dominantes de ninguna era — la advertencia de colinealidad partidista de §4.6 no aplica a este target.
""")

FULL[122] = ("**Interpretacion - Comparativa LR L1 vs. Bayesiano (Lastre)**", """\
**Interpretacion - Comparativa LR L1 vs. Bayesiano (Lastre)**

**ERA_1 y ERA_2 — convergencia OK, lectura sustantiva válida.** AUC Bayesiano: ERA_1=0.645 (+0.059 sobre LR), ERA_2=0.693 (+0.054), con R-hat=1.000 y ESS_min=5105/5553, y concordancia de direccion del 100% (20/20 y 25/25). El AUC LR observado (0.586/0.639) se traduce en HDI amplios que cruzan cero para buena parte de los features (12 y 18 significativos de 20 y 25): el Bayesiano cuantifica explicitamente que la informacion del perfil biografico sobre el lastre es limitada, aunque la direccion cualitativa de los coeficientes es estable entre paradigmas.

**ERA_3 — el muestreador NO converge; la inferencia bayesiana de este grupo no es válida.** La corrida registra 1,500 divergencias tras el tuning, R-hat max=1.530 y ESS_min=7, muy fuera de los umbrales (R-hat<1.05, ESS>400). Consecuencia visible: AUC Bayesiano 0.579 (−0.031 respecto al LR), solo 4 betas "significativos" y concordancia de direccion de 64% — cifras que **no deben leerse sustantivamente**: son sintoma del fallo de muestreo, no evidencia sobre el mecanismo de asignacion. La tabla lo marca como `REVISAR` en los diagnosticos (§8.0). La senal frecuentista de ERA_3 (AUC CV=0.610) no esta afectada.

**Extension pendiente:** re-muestrear el grupo ERA_3-lastre con `target_accept` mas alto (p.ej. 0.99), mas `tune`, o reparametrizacion (priors mas informativos / QR), y verificar diagnosticos antes de usar sus HDI.

Cuando el AUC es bajo y la mayoria de los HDI incluyen el cero — como en ERA_1–ERA_2 — ambos paradigmas convergen en la misma conclusion sustantiva: la asignacion lastre es esencialmente opaca desde el perfil observable.
""")

FULL[127] = ("**Interpretación — SHAP Beeswarm Temáticas", """\
**Interpretación — SHAP Beeswarm Temáticas (grid 1×3 por época)**

A diferencia del beeswarm de nodales, aquí la dispersión de puntos es más uniforme y sin predictores dominantes claros en ninguna época: ninguna variable supera `|SHAP| = 0.05` (los mayores son `legislatura_num`, 0.036 medio, y `area_Derecho`, 0.033). Esta ausencia de estructura SHAP es coherente con el bajo poder predictivo del modelo (mejora ≤5.3% sobre el baseline de predecir siempre la media, y 0% en ERA_3). El número de comisiones temáticas responde a lógicas distributivas internas — cuotas de bancada, disponibilidad de cupos, acuerdos de coalición — que no quedan registradas en el perfil biográfico.
""")

FULL[131] = ("## 7.2 Interpretación — Comisiones Temáticas", """\
## 7.2 Interpretación — Comisiones Temáticas

Las **comisiones temáticas** son esencialmente impredecibles desde el perfil biográfico.

La mejora sobre el baseline —predecir siempre la media de la era— es mínima:

| Era | MAE GLM Poisson | Baseline (media) | Mejora |
|---|---:|---:|---:|
| ERA_1 — PRI | 0.810 | 0.855 | 5.3 % |
| ERA_2 — PAN | 0.788 | 0.791 | 0.4 % |
| ERA_3 — Transición-Morena | 0.842 | 0.842 | 0.0 % |

En ERA_3 el modelo empata exactamente con el baseline: el perfil biográfico no aporta **nada** a la predicción del conteo en el tramo reciente.

### Modelo

El modelo utilizado es un **GLM Poisson**, implementado mediante `PoissonRegressor` con regularización L2. Para las comisiones temáticas **no se aplica SelectFromModel**, ya que el objetivo es contar, no clasificar. Los valores SHAP se calculan con `LinearExplainer` en espacio de **log-conteo** (predictor lineal del GLM con enlace log).

---

### Interpretación política

El número de comisiones temáticas que recibe un diputado depende principalmente de factores no observables directamente en el currículum: negociaciones internas de bancada, disponibilidad de cupos por comisión, preferencias del coordinador parlamentario y acuerdos de coalición legislativa.

---

### Hallazgo principal

La diferencia entre épocas en la media de comisiones temáticas —de **1.53 en ERA_1** a **2.08 en ERA_3**— no se traduce en mayor predictibilidad. La expansión del volumen de temáticas sigue criterios distributivos distintos según la época, pero estos permanecen opacos para el observador externo. El beeswarm SHAP confirma la ausencia de predictores dominantes. Este resultado replica, con la segmentación S3, la insensibilidad de este target ya documentada en `diputraxv12` (MAE plano entre los seis esquemas de agrupación evaluados).
""")

FULL[134] = ("**Interpretacion - Comparativa Global LR L1 vs. Bayesiano**", """\
**Interpretacion - Comparativa Global LR L1 vs. Bayesiano**

**Delta AUC observado:** nodal ERA_1=+0.036, ERA_2=+0.025, ERA_3=+0.035; lastre ERA_1=+0.059, ERA_2=+0.054, ERA_3=−0.031. Los cinco primeros son positivos y homogéneos — consistencia entre paradigmas, con la parte esperable de ajuste in-sample. El delta negativo de lastre ERA_3 no es un resultado sustantivo sino el sintoma del **fallo de convergencia** de ese muestreo (ver abajo).

**Concordancia de direccion = 100%** en cinco de las seis combinaciones era×target (nodal: 22/22, 21/21, 24/24; lastre: 20/20, 25/25). Para nodales ERA_1, el Bayesiano confirma el ranking de A&L (2009): la experiencia burocratico-juridica domina con betas positivos y HDI que excluyen cero. La excepcion es lastre ERA_3 (64%), invalida por convergencia.

**Convergencia MCMC:** OK en 5 de 6 combinaciones (R-hat=1.000, ESS_min entre 2667 y 6227). **Lastre ERA_3: REVISAR** — 1,500 divergencias, R-hat=1.530, ESS_min=7; sus cifras bayesianas (AUC 0.579, 4 significativos, 64% concordancia) no son evidencia valida y quedan documentadas solo por transparencia (§6.5).

**Mensaje metodologico para la tesina:** La consistencia entre paradigmas frecuentista y bayesiano — alli donde el muestreo converge — fortalece la validez interna. El Bayesiano aporta cuantificacion explicita de incertidumbre (HDI) que el LR L1 no ofrece directamente. El caso de lastre ERA_3 ilustra la disciplina inversa: cuando los diagnosticos fallan, el resultado bayesiano se descarta en vez de leerse selectivamente.
""")

FULL[137] = ("**Interpretación — Comparativa de variantes de Regresión Logística**", """\
**Interpretación — Comparativa de variantes de Regresión Logística**

Las tres especificaciones parten del mismo pipeline (`StandardScaler` + `LogisticRegression`, `class_weight="balanced"`, `C=0.1`, semilla 42) y difieren únicamente en el tipo de penalización (L2 vs. L1) y en si se aplica selección de variables (`SelectFromModel`).

- **L2 (v4) vs. L1 (full):** los deltas por era van de −0.005 a +0.010 — la elección de penalización es predictivamente neutra. En promedio: nodal 0.710 (L2) vs. 0.713 (L1); lastre 0.611 vs. 0.612. La ventaja de L1 es de interpretabilidad (esparsidad de coeficientes), no de poder predictivo.
- **L1+SFM vs. L2 (v4):** los deltas también son ≤|0.010|; la reducción del espacio de variables (a 20–25 features activos de 62) no perjudica el desempeño — el modelo se sostiene sobre un subconjunto compacto de predictores robustos, consistente con los hallazgos SHAP de las secciones 5 y 6.
- **Lectura metodológica:** la "mejor variante" alterna por era y target sin patrón material (L1 full domina levemente en nodales, L2 en lastre), confirmando que la migración hacia L1 —adoptada por su valor interpretativo— no implicó costo en AUC. La comparativa documenta además la trayectoria del proyecto (L2 fue la especificación de la v4).
""")

FULL[140] = ("**Interpretación — Tabla consolidada de importancias SHAP**", """\
**Interpretación — Tabla consolidada de importancias SHAP**

**Nodales / Lastre / Temáticas × Épocas**

La tabla normaliza la importancia media (`|SHAP|`) de los principales features para los tres targets y las tres eras. Clasificación binaria en espacio de **log-odds**; temáticas en **log-conteo**.

---

**Nodales — Hallazgos estructurales**

- **`area_Derecho`** (0.336 / 0.340 / 0.268; media 0.315): el feature con mayor importancia media de toda la tabla. Lidera en las tres eras; su señal no decae, aunque la ventaja se estrecha en ERA_3.
- **`fue_secretario_cargo`** (0.175 / 0.070 / 0.105; media 0.117): pico en ERA_1 — la credencial ejecutiva fue más valorada bajo el PRI que bajo el PAN.
- **`n_trayectoria_legislativa`** (0.083 / 0.039 / 0.224; media 0.115): pico histórico en ERA_3; escala con la reelección consecutiva y parlamentariza el perfil reciente.
- **`edad_imp`** (0.000 / 0.224 / 0.108; media 0.111): pico en ERA_2; caveat MNAR (§2.2.1b).
- **`sexo_bin`** (0.000 / 0.123 / 0.181; media 0.102): señal robusta desde ERA_2, máximo en ERA_3 — diferenciación de género en la asignación de comisiones de poder.
- **`p_MORENA`** (0.000 / 0.000 / 0.199; media 0.066): variable partidista activa solo en ERA_3; no confundir con `es_partido_mayoria`, que el L1 elimina en toda la serie.

---

**Lastre — Hallazgos estructurales**

- **`edad_imp`** (0.133 / 0.151 / 0.105; media 0.130): el predictor de mayor magnitud media en lastre, activo en las tres eras.
- **`legislatura_num`** (0.127 / 0.124 / 0.057; media 0.103): el momento del ciclo importa para la asignación marginal.
- **`n_trayectoria_admin`** (0.066 / 0.173 / 0.000; media 0.080): relevante en ERA_2 para lastre pero ausente del top nodal — señal específica del mecanismo marginal panista.
- **`univ_privada`** (0.000 / 0.000 / 0.116) y **`n_trayectoria_legislativa`** (0.000 / 0.000 / 0.126): emergen en ERA_3 — el lastre reciente marca a la oposición veterana con formación privada.

---

**Temáticas**

Todas las magnitudes SHAP son menores a `0.05`. Los mayores promedios son `legislatura_num` (0.036) y `area_Derecho` (0.033). La ausencia de predictores dominantes confirma que las comisiones temáticas responden a lógicas distributivas no capturadas por el perfil biográfico.
""")

FULL[146] = ("## 8.3 Interpretación — Validación Temporal", """\
## 8.3 Interpretación — Validación Temporal

**Modelo:** LR L1 + SFM entrenado en `ERA_k` y aplicado a `ERA_k+1`. Con tres eras, la serie tiene dos transiciones.

| Transición | Nodales AUC | Lastre AUC | Temáticas MAE | Lectura nodales |
|---|---:|---:|---:|---|
| `ERA_1 → ERA_2` | 0.706 | 0.633 | 0.808 | Transferencia sólida: PRI y PAN comparten lógica nodal |
| `ERA_2 → ERA_3` | 0.665 | 0.575 | 1.008 | Caída notable: la frontera de 2015 rompe el perfil PAN |

`ERA_2 → ERA_3` es la transición más disruptiva en los tres targets. El modelo panista pierde 0.041 puntos de AUC nodal al cruzar 2015 (0.706 → 0.665), y queda 0.020 por debajo del AUC dentro-de-era del propio grupo receptor (0.685): el perfil aprendido en el bipartidismo transfiere solo parcialmente al ciclo Transición-Morena.

**Lastre**

El AUC rolling cae de 0.633 a 0.575 en la segunda transición — el mecanismo de lastre, ya ruidoso dentro de era, es aún menos estable entre eras.

**Temáticas**

El MAE de `ERA_2 → ERA_3` es 1.008, muy por encima del baseline de ERA_3 (0.842, predecir la media): el modelo entrenado en la era PAN **degrada** activamente la predicción del conteo en el tramo reciente. Es la señal de ruptura más nítida de los tres targets, aunque sobre un target sin señal dentro de era.

---

**Nota metodológica**

La divergencia en los features seleccionados entre `ERA_k` y `ERA_k+1` constituye evidencia adicional de ruptura de perfil entre eras. La transición E3→E4 de v10 (AUC 0.712) ya no existe como tal en esta segmentación: la frontera de 2024 queda dentro del grupo fusionado — decisión respaldada por la evaluación de segmentaciones de `diputraxv12`.
""")

FULL[149] = ("**Interpretacion -- Tabla de perfiles prototipicos por epoca**", """\
**Interpretacion -- Tabla de perfiles prototipicos por epoca**

El prototipo es el diputado con mayor SHAP positivo acumulado en el modelo LR L1 + SFM de cada era. Lectura de la tabla (valores reales en la salida anterior):

- **ERA_1 (PRI):** hombre del partido mayoritario, con experiencia previa de secretaria (`fue_secretario_cargo`=Sí) y la trayectoria administrativa mas alta de la serie (14) junto con trayectoria politica alta (13). Sin mandato federal previo -- el ascenso se construyo dentro del aparato ejecutivo-partidista, no en el Congreso. Licenciatura, sin posgrado, universidad de elite. Edad 52.
- **ERA_2 (PAN):** hombre del partido mayoritario, tambien con secretaria previa, pero con perfil mas comedido en trayectoria administrativa (8) y politica (7); compensa con posgrado y un mandato federal previo. Electo por mayoria relativa. Edad 52.
- **ERA_3 (Transicion-Morena):** ruptura clara respecto a las dos eras previas -- **es mujer**, la primera de la serie, y **no** tiene experiencia de secretaria. Exhibe mandato federal previo, posgrado, la trayectoria administrativa (4) y politica (6) mas bajas de la serie, sin universidad de elite, electa por mayoria relativa, edad 58 (la mayor de la serie). La carrera parlamentaria y la formacion de posgrado reemplazan al cargo ejecutivo y a la militancia acumulada como ejes del perfil.

**Constante y ruptura transversal:** los tres prototipos pertenecen al partido mayoritario (`es_partido_mayoria`=Sí) -- esa constante sobrevive a los cambios de regimen. La constante de genero de v10 (cuatro prototipos hombres), en cambio, **se rompe**: el prototipo del tramo 2015-presente es una mujer, coherente con la paridad constitucional. El matiz importa: a nivel poblacional `sexo_bin` sigue siendo una señal *pro-hombre* activa en ERA_3 (|SHAP|=0.181, §5.2) -- el extremo individual favorable y la tendencia poblacional no coinciden, y no deben confundirse (ver §9.2).
""")

FULL[151] = ("**Interpretacion -- Waterfall SHAP: Perfiles Prototipicos Nodales", """\
**Interpretacion -- Waterfall SHAP: Perfiles Prototipicos Nodales (1x3 por epoca)**

Cada waterfall descompone el SHAP acumulado del diputado prototipico. Valores en log-odds; la barra base es el log-odds promedio de la era. Lectura estructural (las magnitudes exactas por barra estan en la figura):

**Era 1 (PRI):** el prototipo acumula multiples credenciales ejecutivas — secretaria, cargos de direccion/delegacion, organos de gobierno — junto con la señal juridica (`area_Derecho`). El extremo favorable se construye combinando varias credenciales del aparato, no por una sola senal dominante: es la firma del reclutamiento corporativo priista.

**Era 2 (PAN):** la señal de formacion encabeza (`area_Derecho`, posgrado) con aportes distribuidos de credenciales administrativas (direccion, asesoria, gobierno estatal) — un perfil tecnocratico repartido entre varias barras moderadas, coherente con las importancias poblacionales de la era (edad, posgrado, direccion).

**Era 3 (Transicion-Morena):** las barras dominantes provienen de la carrera parlamentaria (`n_trayectoria_legislativa`, mandato federal previo) y de la filiacion con la coalicion dominante (`p_MORENA`), con la formacion (posgrado, universidad publica) como refuerzo — y sin la barra de secretaria que definia a los prototipos anteriores. El waterfall individual amplifica exactamente el cambio de mezcla que el heatmap poblacional (§5.2) muestra en promedios: del capital burocratico al capital parlamentario.
""")

FULL[155] = ("## 9.2 Lectura comparativa — Evolución del perfil nodal", """\
## 9.2 Lectura comparativa — Evolución del perfil nodal

| Dimensión | ERA_1 PRI | ERA_2 PAN | ERA_3 Trans.-Morena |
|---|---|---|---|
| Partido mayoría | Sí | Sí | Sí |
| Sexo | Hombre | Hombre | **Mujer** |
| Mandato federal previo | No | Sí | Sí |
| Cargo ejecutivo previo (secretaría) | Sí | Sí | **No** |
| Capital administrativo (`n_trayectoria_admin`) | Pico (14) | Moderado (8) | Mínimo (4) |
| Capital político (`n_trayectoria_politica`) | Pico (13) | Moderado (7) | Mínimo (6) |
| Educación | Licenciatura + elite, sin posgrado | Licenciatura + posgrado + elite | Licenciatura + posgrado, sin elite |
| Edad | 52 | 52 | Máxima (58) |
| Feature SHAP dominante (poblacional) | `area_Derecho` (0.336) | `area_Derecho` (0.340) / `edad_imp` (0.224) | `area_Derecho` (0.268) / `n_tray_leg` (0.224) / `p_MORENA` (0.199) |

**Patrón evolutivo**

**PRI → PAN:** el perfil se modera en capital político (13→7) y administrativo (14→8), pero gana posgrado, elección por mayoría relativa y el primer mandato federal previo. El prototipo se "profesionaliza" sin abandonar la militancia mayoritaria ni la credencial de secretaría.

**PAN → Transición-Morena:** la ruptura de la serie. El capital administrativo cae a su mínimo (4) y la secretaría desaparece del prototipo; permanecen el mandato federal previo y el posgrado. El perfil deja de construirse sobre la gestión ejecutiva y la militancia acumulada, y pasa a sostenerse sobre la carrera parlamentaria — coherente con el pico poblacional de `n_trayectoria_legislativa` (0.224) y la señal de `p_MORENA` (0.199) en esta era.

---

**Hallazgo transversal — revisado**

La constante que sobrevive a los tres regímenes es la **pertenencia al partido mayoritario**. La constante de género de v10 se rompe: el prototipo de ERA_3 es mujer. Ese dato individual debe leerse junto con la señal poblacional, que va en dirección contraria — `sexo_bin` (hombre=1) es positivo y alcanza su máximo poblacional justamente en ERA_3 (0.181): en promedio los hombres siguen teniendo ventaja de acceso nodal en el tramo reciente, aunque el caso extremo de la distribución sea una diputada. El prototipo captura el extremo favorable, no la tendencia central; la coexistencia de ambos hechos es exactamente lo que la paridad electoral sin paridad de poder predice (§2.2.8).
""")

FULL[159] = ("## 10.1 Interpretación consolidada", """\
## 10.1 Interpretación consolidada

**Vista unificada del rendimiento. Conclusiones de conjunto:**

1. **Nodales** son los únicos con señal predictiva genuina (AUC LR L1+SFM: ERA_1=0.727, ERA_2=0.719, ERA_3=0.685). La señal es consistente, con un descenso suave hacia el tramo reciente.
2. **Lastre** es moderadamente predecible solo en ERA_2 (0.635); débil en ERA_3 (0.610) y muy débil en ERA_1 (0.584). Los predictores de lastre son estructuralmente distintos a los de nodales.
3. **Temáticas** son prácticamente impredecibles. Mejora sobre baseline: ERA_1=5.3%, ERA_2=0.4%, ERA_3=0.0%.
4. LR L1 competitiva — la estructura de la asignación es predominantemente lineal (premisa heredada de v10, donde LR ≥ RF/XGBoost). Las tres variantes de LR difieren en ≤0.010 de AUC (§8.0b); SFM no degrada el desempeño.
5. Features activos por era: nodales 21–24 de 62 totales; lastre 20–25. La penalización L1 elimina el 60–68% del espacio en todos los modelos.
6. `sexo_bin` es señal activa en nodales desde ERA_2 (0.123) con máximo en ERA_3 (0.181): la brecha de género en asignaciones nodales tiene expresión predictiva directa en el modelo, independiente de trayectoria y formación.
7. Con la segmentación S3 ningún grupo baja de n=1,500: los IC95 del AUC son ±0.023–0.029 en todos los casos (en v10, la ERA_4 aislada tenía ±0.048–0.052). La precisión es homogénea entre eras.

---

## 10.2 Análisis de potencia estadística — comparación entre eras

Con tres grupos de n=1,500–2,000, este apartado formaliza la precisión de los AUC reportados: (a) error estándar e IC95 por era y target; (b) tamaño de efecto mínimo detectable (MDE) a potencia 80% y 90% frente al azar; (c) detectabilidad de las diferencias de AUC entre pares de eras.

El error estándar se estima con la aproximación de Hanley y McNeil (1982), estándar para AUC en clasificación binaria:

$$
\\widehat{\\text{SE}}(\\widehat{\\text{AUC}}) = \\sqrt{\\frac{\\widehat{\\text{AUC}}(1 - \\widehat{\\text{AUC}}) + (n_1 - 1)(Q_1 - \\widehat{\\text{AUC}}^2) + (n_0 - 1)(Q_2 - \\widehat{\\text{AUC}}^2)}{n_1 \\cdot n_0}}
$$

donde $Q_1 = \\widehat{\\text{AUC}} / (2 - \\widehat{\\text{AUC}})$ y $Q_2 = 2\\widehat{\\text{AUC}}^2 / (1 + \\widehat{\\text{AUC}})$.
""")

FULL[163] = ("### 10.2.1 Interpretación — Resultados del análisis de potencia", """\
### 10.2.1 Interpretación — Resultados del análisis de potencia

**Tabla 1 — Potencia por era y target**

- **IC 95% ±:** nodal ±0.029 / ±0.027 / ±0.023; lastre ±0.029 / ±0.028 / ±0.025. La precisión es homogénea entre las tres eras — desaparece el grupo impreciso de v10 (ERA_4, ±0.048–0.052 con n=500).
- **AUC observados (LR L1+SFM):** nodal 0.727 / 0.719 / 0.685; lastre 0.584 / 0.635 / 0.610.
- **MDE (80%) frente a AUC=0.5:** umbrales de 0.536–0.545 según era y target.

**Veredicto frente al azar:** los seis AUC superan holgadamente su umbral (z entre 5.6 —lastre ERA_1, el caso más justo— y ~14 en nodales). **Toda la señal reportada es estadísticamente genuina**, incluido el lastre en las tres eras — a diferencia de v10, donde el lastre de la ERA_4 corta (0.530, n=500) era indistinguible del azar.

---

**Tabla 2 — Diferencias entre pares de eras**

| Comparación | Δ AUC nodal | Δ AUC lastre | MDE inter-era (≈) | Detectable |
|---|---:|---:|---:|---|
| ERA_1 vs ERA_2 | 0.008 | 0.051 | 0.056–0.058 | NO / NO |
| ERA_1 vs ERA_3 | 0.042 | 0.026 | 0.053–0.055 | NO / NO |
| ERA_2 vs ERA_3 | 0.034 | 0.025 | 0.051–0.054 | NO / NO |

**Ninguna diferencia de AUC entre eras es detectable al 80% de potencia.** El descenso nodal ERA_1→ERA_3 (Δ=0.042) y la ventaja de ERA_2 en lastre (Δ=0.051) quedan por debajo del MDE (≈0.05–0.06): la deriva temporal del desempeño es una tendencia **indicativa, no confirmable estadísticamente** con estas muestras. Las conclusiones comparativas del cuaderno (H1, H4) se apoyan por eso en el *rolling forward* y en el cambio de composición SHAP — evidencia de mecanismo — y no en la diferencia puntual de AUC entre eras.

---

**Implicación metodológica**

La fusión S3 compra precisión homogénea (IC ±0.023–0.029) y elimina el caso "no distinguible de azar" de v10, al costo de no poder aislar el subrégimen 2024+. Para detectar diferencias inter-era de ~0.03–0.04 de AUC se necesitarían grupos sustancialmente mayores; la incorporación de la LXVII (2027) al tramo reciente será el primer paso natural.
""")

FULL[164] = ("# 11. Conclusiones y Hallazgos Clave", """\
# 11. Conclusiones y Hallazgos Clave

---

## 11.1 Hallazgos principales

**H1 — Las comisiones nodales son moderadamente predecibles (AUC 0.69–0.73)**

El perfil biográfico explica una parte real pero no dominante de la asignación nodal. El resto lo determinan factores institucionales no observados: negociaciones de bancada, cuotas de coalición, relaciones personales con liderazgos. La señal desciende suavemente hacia el tramo reciente (0.727 → 0.685), aunque esa diferencia queda por debajo del mínimo detectable (§10.2.1): el deterioro es tendencia indicativa, no hecho estadísticamente confirmado.

**H2 — Las comisiones lastre son esencialmente opacas (AUC 0.58–0.64)**

La hipótesis de que el lastre es el perfil inverso del nodal queda rechazada. Las correlaciones SHAP(nodal) vs −SHAP(lastre) oscilan entre −0.095 (ERA_1) y −0.393 (ERA_2), lejos de −1.0. Nodal y lastre deben tratarse como mecanismos institucionales distintos.

**H3 — Las comisiones temáticas son prácticamente impredecibles desde el perfil**

La mejora sobre el baseline es 5.3% en ERA_1, 0.4% en ERA_2 y **0.0%** en ERA_3. El volumen de comisiones temáticas es una asignación de naturaleza distributiva/administrativa, no meritocrática — resultado además insensible a la segmentación temporal (diputraxv12).

**H4 — La frontera de 2015 es la ruptura más profunda en la lógica de asignación**

El rolling forward muestra que el modelo entrenado en ERA_2 (PAN) predice ERA_3 con AUC=0.665 (vs. 0.706 de la transición PRI→PAN) y que su MAE de temáticas (1.008) degrada incluso al baseline del grupo receptor (0.842). El cambio de régimen post-2015 —fragmentación multipartidista y recomposición bajo Morena— generó la mayor discontinuidad de criterios de la serie.

**H5 — El perfil nodal se parlamentariza en el ciclo 2015–presente** *(reformulado respecto a v10)*

En ERA_3, `n_trayectoria_legislativa` alcanza su pico histórico (|SHAP|=0.224, segunda señal del grupo tras `area_Derecho`) y `p_MORENA` (0.199) es la variable partidista activa, mientras la vía burocrática (`fue_secretario_cargo`, trayectoria administrativa) pierde el peso que tenía en ERA_1–ERA_2. El prototipo del grupo ya no tiene credencial de secretaría.

*Límite de resolución declarado:* v10 atribuía la legislativización específicamente al subrégimen de Morena (LXVI aislada: cargos legislativos previos, senaduría, doctorado). Con la segmentación S3 ese subrégimen queda promediado dentro del grupo 63–66 y **este cuaderno no puede separar cuánto del giro corresponde a la Transición y cuánto a la supermayoría de Morena**. La atribución fina queda documentada en v10 (S4) y podrá reevaluarse con la LXVII.

**H6 — La estructura de asignación es lineal y esparsa** *(premisa heredada + evidencia interna)*

La evidencia comparativa contra modelos de árboles (RF/XGBoost) pertenece a v10, donde la LR L1 igualó o superó su AUC; este cuaderno la hereda como premisa de diseño y no la re-estima. La evidencia interna de v13 es consistente con ella: las tres variantes de LR difieren en ≤0.010 de AUC (§8.0b), la penalización L1 elimina el 60–68% de las 62 variables sin costo predictivo, y la capa bayesiana concuerda en dirección al 100% donde converge (§8.0).

**H7 — Brecha de género en comisiones nodales** *(reformulado respecto a v10)*

La brecha descriptiva de acceso nodal crece monotónicamente: −6.2 pp (ERA_1) → −8.9 pp (ERA_2) → −11.7 pp (ERA_3), pese al avance electoral hacia la paridad. En el modelo, `sexo_bin` es señal activa desde ERA_2 y máxima en ERA_3 (0.123 → 0.181): el género tiene expresión predictiva directa, independiente de trayectoria y formación.

*Límite de resolución declarado:* la convergencia de la brecha en el subrégimen 2024+ que v10 documentaba (cierre en la LXVI) no es observable con S3 — queda promediada dentro de ERA_3. El matiz nuevo de esta corrida: el prototipo nodal de ERA_3 es una mujer (§9), aun cuando la señal poblacional favorece a los hombres — paridad en el extremo, brecha en el promedio.

---

## 11.2 Limitaciones

| Limitación | Impacto |
|---|---|
| Resolución del subrégimen 2024+ | La LXVI no se modela por separado (fusión S3); H5 y H7 pierden su componente "Morena aislada". Contrapartida: ningún grupo con n<1,500 y IC homogéneos (±0.023–0.029). |
| Bayesiano lastre ERA_3 sin converger | R-hat=1.53, ESS=7, 1,500 divergencias: sus HDI no son válidos (§6.5). Pendiente re-muestreo con `target_accept` mayor. |
| Diferencias inter-era no detectables | Todos los Δ AUC entre eras < MDE (≈0.05); la deriva temporal es indicativa (§10.2.1). |
| Anomalía en `grado_estudios_ord` en LIX | Grado promedio 1.49 vs ~4 en otras legislaturas — posible error de captura que afecta ERA_1. |
| `edad_al_tomar_cargo`: 10.2% nulos, MNAR confirmado (§2.2.1b) | Importancia SHAP de `edad_imp` en ERA_2 (0.224) y ERA_3 (0.108) parcialmente inflada; `edad_missing` mitiga pero no cancela. |
| 625 reelecciones válidas no separadas | Sin leakage entre eras (CV agrupada AC2), pero el perfil de reelectos puede sesgar importancias SHAP. |
| Sin contraste no lineal interno | RF/XGBoost/MTL quedan en v10 y diputraxpytorch; H6 es aquí premisa, no resultado. |
| Factores no observados (redes, negociaciones) | Techo de AUC real desconocido — el ~30% no explicado puede ser ruido o información estructuralmente ausente. |

---

## 11.3 Próximos pasos sugeridos

1. Incorporar la LXVII (2027) y reevaluar si el tramo 2024+ amerita grupo propio — el diseño S3 vs. S4 puede re-someterse al protocolo de `diputraxv12` con la muestra ampliada.
2. Re-muestrear el Bayesiano de lastre ERA_3 (target_accept≥0.99, más tune o reparametrización) para recuperar HDI válidos.
3. Separar reelectos de primerizos para ver si la señal SHAP varía — los reelectos pueden tener lógica de asignación distinta.
4. Incluir variables de red (co-membresía en comisiones anteriores, partido del presidente de comisión) para subir el techo de AUC.
5. Calibración de probabilidades (Platt) si el modelo se usara para señalar diputados "en riesgo" de lastre.

---

## 11.4 Notas de calidad de datos

- `grado_estudios_ord` en LIX tiene promedio 1.49 (vs ~4 en otras) — probable error de captura.
- 10.2% nulos en `edad_al_tomar_cargo` — mecanismo MNAR confirmado (AUC predictivo=0.787, ver §2.2.1b); la tasa por era es 10.4% / 4.5% / 14.4%, con máximo intra-ERA_3 en la LXIII (28%).
- 625 registros son reelecciones válidas — no son leakage entre eras; la CV agrupada por `diputado_id` (AC2) controla la fuga por reelección.
""")

FULL[167] = ("## B.1 Interpretación — Robustez de la clasificación nodal", """\
## B.1 Interpretación — Robustez de la clasificación nodal

| Era | Base (≥1) | Alt-1 Estricta (≥2) | Alt-2 Ampliada (pres) | Max Δ | ¿Robusto? |
|---|---:|---:|---:|---:|---|
| ERA_1 | 0.727 | 0.723 | 0.726 | 0.004 | **SÍ** |
| ERA_2 | 0.719 | 0.728 | 0.677 | 0.051 | **NO** |
| ERA_3 | 0.685 | 0.716 | 0.695 | 0.031 | **NO** (marginal) |

- **ERA_1:** diferencia < 0.03 — conclusiones robustas al umbral operacional adoptado.
- **ERA_2:** diferencia 0.051, concentrada en Alt-2 Ampliada (0.677) — la clasificación nodal en ERA_2 es sensible a si se incluyen presidencias como nodales: las presidencias de esa era capturan un perfil distinto al nodal estructural.
- **ERA_3:** diferencia 0.031, apenas sobre el umbral, y en dirección informativa: la clasificación estricta (≥2 nodales) **sube** el AUC a 0.716 — los diputados con presencia sostenida en comisiones de alto perfil son más distinguibles que los de la frontera ≥1, que incluye casos ambiguos.

> **Recomendación:** revisar el diccionario `COMISION_TIPO` para ERA_2 (inclusión de presidencias) antes de concluir sobre esa era; para ERA_3, considerar reportar la variante estricta como análisis complementario.

> **Limitación metodológica:** las alternativas operan sobre conteos ya agregados; no acceden a los nombres individuales de comisiones. Una revisión completa de robustez requiere re-etiquetar comisiones en el ETL.

---

## B.2 Robustez de la operacionalización de comisiones temáticas

Las comisiones temáticas tienen como target un conteo (`n_comisiones_tematicas`, 0–10). A diferencia de los targets binarios, la robustez no se evalúa variando una frontera ETL ya construida, sino binarizando el conteo a distintos umbrales y evaluando si el poder predictivo del perfil biográfico cambia según la definición adoptada.

| Especificación | Definición | Métrica | Justificación |
|---|---|---|---|
| **Alt-1 Binaria (≥1)** | `n_comisiones_tematicas >= 1` | AUC | Frontera de inclusión mínima: ¿participó en alguna? |
| **Alt-2 Binaria (≥2)** | `n_comisiones_tematicas >= 2` | AUC | Umbral exigente: participación sostenida |
| **Alt-3 Binaria (≥mediana)** | `>= mediana` del conteo en la era | AUC | Umbral relativo: arriba/abajo de la mediana por época |

**Criterio:** si todos los AUC se mantienen por debajo de 0.65 en todas las eras y umbrales, las conclusiones de la sección 7.2 son robustas. En esta corrida los AUC van de 0.613 a 0.674 — ninguno alcanza 0.70 — y el máximo Δ intra-era es 0.056 (ERA_1, marcado REVISAR por poco), con ERA_2 y ERA_3 robustas (Δ=0.021 y 0.018). El poder predictivo sobre las temáticas es débil bajo cualquier operacionalización.
""")

FULL[176] = ("**Interpretación — D.1 Regresión Logística clásica (Nodales)**", """\
**Interpretación — D.1 Regresión Logística clásica (Nodales)**

**ERA_1 — PRI** (n=1500, **la MLE no penalizada no converge**): el ajuste global no es utilizable —`Pseudo R² McFadden = −inf`, `LLR p = 1.00`, `Convergió = False`— porque `admin_en_sindicato` produce **separación cuasi-completa**: casi ningún diputado con trayectoria sindical recibió comisión nodal en ERA_1, y la verosimilitud diverge llevando ese coeficiente a +∞ (β≈9.7×10¹⁰, OR=∞) y arrastrando al intercepto. **Los coeficientes no afectados por la separación sí se estiman con normalidad:** `area_Derecho` es el más fuerte (OR=1.60, *p*<0.001) —una desviación estándar adicional en la señal de formación jurídica multiplica por 1.6 la razón de momios de recibir comisión nodal—, seguido de `area_Económico-Financiera` (OR=1.38), `fue_subsecretario` (OR=1.26), `univ_privada` (OR=1.26) y `fue_secretario_cargo` (OR=1.25), todos con *p*<0.001. `reg_SUR` es negativo y significativo (OR=0.81, *p*=0.002) —penalización regional que SHAP no distingue por signo—. Los estadísticos de ajuste global de esta era (pseudo R², LLR, AIC/BIC) carecen de sentido al no converger.

> **Advertencia de separación cuasi-completa:** el colapso numérico se concentra en `admin_en_sindicato` y el intercepto; el resto de la tabla converge y se reporta con normalidad. Esto **no invalida** los coeficientes estables, pero confirma empíricamente por qué el modelo productivo usa L1: la penalización habría encogido `admin_en_sindicato` a un valor finito en vez de dejar que la verosimilitud divergiera.

**ERA_2 — PAN** (n=1500, pseudo R² = 0.138, converge sin problemas): `sexo_bin` es significativo por primera vez en la serie (OR=1.18, *p*=0.007), confirmando en términos clásicos la emergencia del género como señal activa. `edad_imp` es negativo y fuerte (OR=0.710, *p*<0.001) —una vez controlado por trayectoria, mayor edad estandarizada se asocia con *menor* probabilidad de nodal, patrón que solo el signo del coeficiente clásico revela—. `area_Derecho` sigue dominando (OR=1.52, *p*<0.001) y `tiene_posgrado` es positivo (OR=1.23, *p*=0.001).

**ERA_3 — Transición-Morena** (n=2000, pseudo R² = 0.120, converge sin variables NaN): `n_trayectoria_legislativa` es el coeficiente positivo más fuerte tras el área jurídica (OR=1.37, *p*=0.002); `sexo_bin` alcanza su máximo clásico (OR=1.23, *p*<0.001); `p_MORENA` es positivo y significativo (OR=1.27, *p*=0.001) mientras `p_PAN` (OR=0.84) y `p_PRI` (OR=0.87) son negativos — la firma partidista del tramo. `edad_imp` repite el signo negativo (OR=0.84, *p*=0.001). El formato clásico confirma el hallazgo SHAP central: carrera parlamentaria + filiación con la coalición dominante + formación.

**Desempeño de clasificación (umbral 0.5, sin balanceo de clases):** *precision* 0.65–0.67, *recall* 0.40–0.66, F1 0.50–0.66. El *recall* más bajo es ERA_1 (0.40), reflejo directo de que esta variante —a diferencia del modelo productivo— no usa `class_weight="balanced"`, por lo que subpredice la clase positiva minoritaria (32.2 % de tasa nodal en ERA_1).
""")

FULL[178] = ("**Interpretación — D.2 Regresión Logística clásica (Lastre)**", """\
**Interpretación — D.2 Regresión Logística clásica (Lastre)**

**Las tres eras convergen** (`Convergió=True`), a diferencia de v10 —donde el grupo corto de n=500 colapsaba por completo—. El ajuste de lastre es sistemáticamente más débil que el de nodales (pseudo R² 0.050–0.094 vs. 0.120–0.138), consistente con §6.4: la asignación a comisiones lastre responde menos al perfil biográfico.

**ERA_1** (pseudo R² = 0.050): `sexo_bin` (OR=0.88, *p*=0.025) y `fue_secretario_cargo` (OR=0.82, *p*=0.004) son negativos — dirección inversa a nodales, coherente con la hipótesis de "imagen espejo" parcial de §6.2 —, pero `nivel_cargo_max` es positivo (OR=1.30, *p*<0.001): un hallazgo contraintuitivo que el SHAP agregado no distingue por signo y que merece lectura cualitativa adicional.

**ERA_2** (pseudo R² = 0.094, la más ajustada): `n_trayectoria_admin` es fuertemente negativo (OR=0.756, *p*<0.001) y `n_organos_gobierno` aún más (OR=0.665, *p*<0.001) — el capital burocrático protege del lastre en la era panista —, mientras `fue_director` (OR=1.25) y `admin_en_partido` (OR=1.15) son positivos.

**ERA_3** (pseudo R² = 0.057): converge, con una salvedad numérica local — `admin_en_sindicato` y el intercepto muestran errores estándar degenerados (SE≈10⁶–10⁷, *p*=1.00) por cuasi-separación de esa categoría rara; el resto de la tabla es estable. Los coeficientes significativos son mayoritariamente **negativos**: `univ_privada` (OR=0.84, *p*=0.001), `n_trayectoria_legislativa` (OR=0.84, *p*=0.018), `edad_imp` (OR=0.86, *p*=0.003), `n_organos_gobierno` (OR=0.85, *p*=0.007) y `admin_en_partido` (OR=0.88, *p*=0.018); solo `fue_regidor` es positivo (OR=1.11, *p*=0.034). Nota de lectura: el signo clásico de `univ_privada` y `n_trayectoria_legislativa` (protectores) matiza la lectura SHAP de §6.4 — su |SHAP| alto refleja magnitud de contribución, y la dirección dominante en la MLE es hacia *menos* lastre; la asignación marginal del tramo reciente golpea a los perfiles con *menos* credenciales, no a la oposición veterana per se.

> **Lectura conjunta con D.1:** la única separación cuasi-completa restante de la serie (`admin_en_sindicato` en Nodal ERA_1 y, localmente, en Lastre ERA_3) es de categoría rara, no de tamaño muestral — el problema estructural de "separación en muestras pequeñas" (King y Zeng 2001) que v10 exhibía en su ERA_4 de n=500 desaparece con la segmentación S3, reforzando igualmente la elección del estimador penalizado (L1) para el modelo productivo.
""")

FULL[180] = ("**Interpretación — D.3 Multicolinealidad (VIF)**", """\
**Interpretación — D.3 Multicolinealidad (VIF)**

Regla convencional: VIF > 5 sugiere multicolinealidad problemática; VIF > 10, severa.

**Nodales:** ERA_1 (máx. VIF = 1.78, `n_trayectoria_admin`), ERA_2 (3.01, `n_trayectoria_legislativa`) y ERA_3 (3.71, `n_trayectoria_legislativa`) están todas por debajo del umbral convencional. El VIF crece hacia el tramo reciente por el solapamiento natural entre los indicadores de carrera legislativa (`n_trayectoria_legislativa`, `fue_diputado_federal`, `fue_diputado_local`) bajo la reelección consecutiva, pero sin llegar a niveles problemáticos.

**Lastre:** máximos de 2.62 (ERA_1), 1.83 (ERA_2) y 2.95 (ERA_3, `reg_CENTRO` por el bloque regional) — sin señales de multicolinealidad en ninguna era.

**Cambio estructural respecto a v10:** desaparece el `VIF = ∞` de `p_MORENA` que la ERA_4 aislada producía (con una sola legislatura de supermayoría, `p_MORENA` y `es_partido_mayoria` eran colineales exactas). Al fusionar 63–66, el grupo contiene legislaturas con mayorías distintas y ambas variables vuelven a ser estimables por separado (VIF de `p_MORENA` = 2.01 en ERA_3). Es un beneficio directo de la segmentación S3: el límite interpretativo de §4.6 se **suaviza** — la advertencia asociativa se mantiene como prudencia general en tramos de partido dominante, pero ya no existe la colinealidad exacta que en v10 obligaba a leer H5 con reservas duras.

**Lectura conjunta con D.1–D.2:** las inestabilidades restantes (`admin_en_sindicato`) son de separación cuasi-completa por categoría rara, no de colinealidad entre predictores — dos problemas estadísticos distintos que requieren lecturas distintas.
""")

FULL[182] = ("**Interpretación — D.4 GLM Poisson clásico (Temáticas)**", """\
**Interpretación — D.4 GLM Poisson clásico (Temáticas)**

Los tres modelos convergen sin problemas de separación (a diferencia de D.1) —el enlace log del Poisson y la ausencia de *dummies* de categoría rara en `KEY_FEATS` evitan el problema estructural de los binarios—.

**Pseudo R² por deviance:** 0.023 (ERA_1) → 0.017 (ERA_2) → 0.032 (ERA_3). Valores bajos en las tres eras —consistente con §7.2: las temáticas siguen una lógica distributiva no capturada por el perfil biográfico—, con el máximo (modesto) en el tramo reciente.

**Dispersión (Pearson χ²/gl):** 0.642 / 0.582 / 0.623 — todas **por debajo de 1**, es decir, **subdispersión** respecto al supuesto Poisson. Es el patrón inverso a la sobredispersión que motiva binomial-negativa; aquí el techo institucional de comisiones temáticas comprime la varianza observada. No invalida el Poisson — la subdispersión no sesga los coeficientes, solo hace conservadores los errores estándar.

**Coeficientes significativos por era (α=0.05):**

- **ERA_1** — `n_trayectoria_politica` (IRR=1.096, *p*<0.001, positiva) y `fue_diputado_federal` (IRR=0.934, *p*=0.009, negativa).
- **ERA_2** — solo `edad_imp` (IRR=1.069, *p*=0.001).
- **ERA_3** — `es_partido_mayoria` (IRR=1.034, *p*=0.032, positiva) y `n_trayectoria_legislativa` (IRR=0.909, *p*<0.001, negativa): más experiencia legislativa previa se asocia con **menos** comisiones temáticas en el tramo reciente — los perfiles parlamentarios veteranos concentran comisiones de poder (§5) y reciben menos carga temática, coherente con una lógica distributiva que reparte el trabajo sectorial entre los perfiles sin capital parlamentario.

Pese al pseudo R² máximo de la serie, la señal de ERA_3 se concentra en esas dos variables: la asignación temática sigue siendo esencialmente distributiva, con la experiencia legislativa previa como único filtro individual robusto.
""")

FULL[183] = ("## D.5 Síntesis y límites del Anexo D", """\
## D.5 Síntesis y límites del Anexo D

Este anexo complementa —no reemplaza— la interpretabilidad SHAP (magnitud de contribución individual) y la capa Bayesiana (incertidumbre posterior vía HDI) con el formato de tabla de regresión que la ciencia política cuantitativa espera como estándar de reporte: coeficiente, error estándar, valor *p*, IC95 %, *odds ratio*/IRR, pseudo R², prueba de razón de verosimilitud, AIC/BIC, matriz de confusión y VIF.

**Hallazgo metodológico transversal:** de las seis combinaciones era×target binario, solo una (Nodal ERA_1) produce una MLE no convergente, por separación cuasi-completa de una categoría rara (`admin_en_sindicato`); Lastre ERA_3 converge con degeneración local en esa misma dummy. La causa es identificable —categorías dummy casi vacías— y coincide con las condiciones bajo las cuales la literatura (King y Zeng 2001; Firth 1993) recomienda estimadores penalizados: evidencia empírica adicional a favor de la elección metodológica central del proyecto (Regresión Logística **L1** en lugar de MLE clásica sin regularizar). Respecto a v10, la patología se **reduce** (allí fallaban tres combinaciones y una colapsaba por completo): los grupos de n≥1,500 de la segmentación S3 eliminan la separación por muestra corta y la colinealidad exacta `p_MORENA`≡`es_partido_mayoria` de la antigua ERA_4 (D.3).

**Limitación declarada:** los coeficientes de este anexo no son numéricamente comparables a los coeficientes L1 del modelo productivo (penalización distinta, sin `class_weight="balanced"`); su función es inferencial —cuantificar significancia y magnitud en la escala familiar de la ciencia política cuantitativa— no predictiva. Para desempeño predictivo y comparabilidad entre variantes, la referencia sigue siendo la Tabla Global de la sección 8.0 y el AUC de validación cruzada de las secciones 5.4, 6.3 y 7.1.
""")

# --------------------------------------------------------------------------
# Reemplazos exactos menores
# --------------------------------------------------------------------------
REPL = [
    # 61 -> 62 features en textos de fase 1
    (1, "Espacio de 61 *features* con imputación MICE",
        "Espacio de 62 *features* con imputación MICE"),
    (4, "se construyeron 61 *features* organizados en seis bloques",
        "se construyeron 62 *features* organizados en seis bloques"),
    (68, "## *Feature engineering* — 61 *features*",
         "## *Feature engineering* — 62 *features*"),
    (68, "| `LR L1 (full)` | L1 sobre los 61 *features* sin preselección. |",
         "| `LR L1 (full)` | L1 sobre los 62 *features* sin preselección. |"),
    (156, "cuántas de las 61 variables del perfil biográfico",
          "cuántas de las 62 variables del perfil biográfico"),
    (170, "Cubre las 61 variables del modelo",
          "Cubre las 62 variables del modelo"),
    (174, "en vez de reportar los 61 *features* completos",
          "en vez de reportar los 62 *features* completos"),
]

# Celda 114: dict del test espejo — texto definitivo para ERA_3 (se actualiza
# fuente Y output mostrado; los valores calculados r no cambian).
MIRROR_SRC_OLD_1 = '"Ver corrida",             "grupo fusionado 63-66 (actualizar tras ejecución)"'
MIRROR_SRC_NEW_1 = '"Inversa moderada",        "patrón bimodal mayoría/oposición post-2015"'
MIRROR_OUT = [
    ("Ver corrida", "Inversa moderada"),
    ("grupo fusionado 63-66 (actualizar tras ejecución)",
     "patrón bimodal mayoría/oposición post-2015"),
]


def cell_src(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    if len(cells) != 203:
        sys.exit(f"ABORT: se esperaban 203 celdas, hay {len(cells)}")

    errors = []
    for idx, (anchor, _text) in FULL.items():
        src = cell_src(cells[idx])
        if not src.lstrip().startswith(anchor):
            errors.append(f"celda {idx}: no empieza con {anchor!r} "
                          f"(empieza con {src.lstrip()[:70]!r})")
    for idx, old, _new in REPL:
        n = cell_src(cells[idx]).count(old)
        if n != 1:
            errors.append(f"celda {idx}: patron aparece {n} veces: {old[:70]!r}")
    c114 = cell_src(cells[114])
    if MIRROR_SRC_OLD_1 not in c114:
        errors.append("celda 114: patron del dict espejo no encontrado")

    if errors:
        print("ABORT — el notebook no coincide con lo esperado:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    for idx, (_a, text) in sorted(FULL.items()):
        set_src(cells[idx], text.rstrip("\n"))
        print(f"celda {idx:3d}: reescrita")
    for idx, old, new in REPL:
        set_src(cells[idx], cell_src(cells[idx]).replace(old, new))
        print(f"celda {idx:3d}: reemplazo aplicado")

    # celda 114: fuente + outputs
    set_src(cells[114], c114.replace(MIRROR_SRC_OLD_1, MIRROR_SRC_NEW_1))
    for out in cells[114].get("outputs", []):
        data = out.get("data", {})
        for mime in list(data.keys()):
            content = data[mime]
            text = content if isinstance(content, str) else "".join(content)
            for old, new in MIRROR_OUT:
                text = text.replace(old, new)
            data[mime] = text.splitlines(keepends=True)
        if out.get("output_type") == "stream":
            text = "".join(out.get("text", []))
            for old, new in MIRROR_OUT:
                text = text.replace(old, new)
            out["text"] = text.splitlines(keepends=True)
    print("celda 114: dict espejo y output actualizados")

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"\nOK — fase 3 aplicada: {NB_PATH}")


if __name__ == "__main__":
    main()
