# -*- coding: utf-8 -*-
"""Segundo parche de diputraxv12.ipynb (tras patch_v12_reframe_question.py y
re-ejecución): reconcilia los números citados en las celdas interpretativas
con los de la corrida final (plan12upd.md §4.3.2).

Corrida final: nodal Pond. legis S1/S2 0.712, S3 0.700, S4 0.696, S5 0.691,
S6 0.666; lastre S2 0.605 ... S4 0.585; Δ = S3 0.080 > S2 0.076 > S4 0.064 >
S5 0.052; similitud global 0.375. Solo celdas markdown — no requiere
re-ejecutar el notebook.
"""
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "diputraxv12.ipynb"

FULL = {}

FULL[27] = ("## 6.1 Interpretación — targets binarios", """\
## 6.1 Interpretación — targets binarios

**Panorama.** El AUC ponderado por estrato-legislatura (`Pond. legis`, LR L1+SFM) ordena así las segmentaciones:

| Esquema | Nodal | Lastre |
|---|---:|---:|
| S2 — Corte 2018 | **0.712** | **0.605** |
| S1 — Pooled | 0.712 | 0.596 |
| S3 — Fusión T+M | 0.700 | 0.597 |
| S4 — 4 cortes coalición | 0.696 | 0.585 |
| S5 — Pares uniformes | 0.691 | 0.591 |
| S6 — Por legislatura | 0.666 | 0.586 |

Cuatro lecturas:

**1. Entre 1 y 5 grupos, las diferencias predictivas son sub-resolución — empate según la regla de §5.7.** La banda S1–S5 abarca 0.021 puntos de AUC en nodal (0.691–0.712) y 0.020 en lastre (0.585–0.605), en el orden del umbral de empate (≈±0.02). El desagregado por legislatura muestra que en las legislaturas 57–62 los candidatos de esa banda rinden a la par (en la LEG 59, S4 incluso encabeza: 0.724 vs. 0.702 del pooled). **Las diferencias se concentran en las legislaturas 63–66**, sobre todo en la LXVI: nodal 0.681 (S1) vs. 0.619 (S4); lastre 0.580 vs. 0.503.

**2. La penalización del extremo fino es varianza pura.** S6 es el peor candidato en ambos targets (0.666/0.586) y en casi todos los estratos, y S4 reproduce el mismo síntoma exactamente donde su grupo se reduce a una legislatura (LXVI, n=500). La vista *naive* lo corrobora: los AUC nodal por legislatura oscilan de 0.619 a 0.724 con desviaciones de hasta ±0.071. Más homogeneidad no compensa el costo muestral de bajar a n≈500: el intercambio sesgo-varianza tiene un límite duro por el lado fino.

**3. A granularidad comparable, la ubicación de las fronteras no mueve la predicción.** S4 (fronteras políticas, n=500–1,500) supera a S5 (ventanas mecánicas, n=1,000) por solo 0.005 en nodal (0.696 vs. 0.691) y queda 0.006 por debajo en lastre (0.585 vs. 0.591). La capacidad predictiva estratificada distingue **granularidades**, no familias: la evidencia sobre si las fronteras políticas son mejores que las mecánicas debe venir del análisis estructural (§8), no del AUC.

**4. El mejor candidato compacto es S2 — con una reserva de sesgo.** S2 encabeza lastre (0.605) y empata con el pooled en nodal (0.712). Su vista *naive* muestra, sin embargo, que el grupo `POST_2018` (LXIV–LXVI, n=1,500) alcanza solo 0.687 dentro de grupo frente a 0.730 del `PRE_2018`: el corte único junta legislaturas con dinámicas distintas en su tramo final — heterogeneidad interna que la sección 8 examina con los coeficientes.

**Nota sobre el AUC global.** Para S1 el AUC global (0.728) supera al estratificado (0.712): esa brecha es exactamente el crédito por tasas base entre periodos descrito en §4, y es la razón por la que la comparación honesta debe hacerse dentro de estratos fijos por legislatura. La columna de robustez (LR L1 completo, sin SFM) replica el ordenamiento en ambos targets (S1/S2 arriba con 0.712/0.711 en nodal, S6 abajo con 0.676).
""")

FULL[38] = ("**Lectura — cohesión de coeficientes", """\
**Lectura — cohesión de coeficientes (Δ = dentro − entre).** Para cada segmentación se promedia la similitud coseno **dentro** de sus grupos y **entre** grupos; **Δ** es su diferencia. Un Δ grande significa que la partición traza sus fronteras justo donde la lógica de coeficientes cambia (bloques internamente parecidos, contraste alto hacia afuera). S1 y S6 muestran `NaN` porque, por construcción, no tienen pares entre-grupos (un solo grupo) o dentro-de-grupo (grupos de una legislatura), respectivamente.

*Cómo leerla con cautela.* En esta corrida, S3 alcanza el Δ máximo (0.080), seguido de cerca por S2 (0.076); S4 queda en 0.064 y S5 claramente abajo (0.052). La banda de cortes políticos es estrecha y su orden fino **no es estable entre corridas**: ejecuciones previas de este mismo cuaderno, con entornos ligeramente distintos, colocaban primero a S2 (0.086) o a S4 (0.089). Lo robusto no es el ranking dentro de la banda, sino el **contraste entre la familia de cortes políticos (Δ 0.064–0.080) y las ventanas uniformes ciegas a la política (0.052)**. La interpretación está en §8.1.
""")

FULL[39] = ("## 8.1 Interpretación — validez estructural", """\
## 8.1 Interpretación — validez estructural

Este análisis pregunta algo distinto al AUC: no *cuánto* predice cada segmentación, sino **si sus fronteras cortan donde de verdad cambia la lógica de asignación**. El indicador Δ (similitud media dentro de grupo − entre grupos) mide la calidad estructural de cada partición:

| Esquema | Sim. dentro | Sim. entre | Δ |
|---|---:|---:|---:|
| S3 — Fusión T+M | **0.434** | 0.354 | **0.080** |
| S2 — Corte 2018 | 0.411 | 0.334 | 0.076 |
| S4 — 4 cortes coalición | 0.426 | 0.362 | 0.064 |
| S5 — Pares uniformes | 0.421 | 0.369 | 0.052 |

Tres lecturas:

- **La familia política maximiza la validez estructural.** Los tres candidatos cuyas fronteras siguen cambios de coalición dominante (S2, S3, S4) se agrupan en Δ 0.064–0.080, por encima de las ventanas uniformes de S5 (0.052), que parten los bloques reales por la mitad. Las fronteras electorales (2006, 2015, 2018, 2024) capturan cambios reales del mecanismo; las mecánicas, no. Este contraste — no el orden fino dentro de la banda política — es el hallazgo estructural robusto.
- **En esta corrida, S3 encabeza la validez estructural.** Sus tres grupos (cortes en 2006 y 2015) logran el mayor Δ (0.080) y la mayor cohesión interna (0.434); S2 queda a solo 0.004 (Δ=0.076). En ejecuciones previas del mismo cuaderno el primer lugar lo ocupaban S2 (0.086) o S4 (0.089): el orden fino **rota entre corridas** y ninguna conclusión debe descansar en él. Lo que sí se repite en todas las corridas es que las fronteras de coalición (2006, 2015, 2018) aparecen en la banda alta y S5 queda último.
- **La fragilidad del orden fino es varianza en acción.** Los coeficientes L1 por legislatura se estiman con n≈500 — el mismo cuello muestral que castiga a S6 en predicción hace inestables los valores finos de Δ. Es la razón por la que la regla de decisión (§5.7) lee este criterio al nivel de familias, no de ranking.

La similitud media global entre los 10 vectores de coeficientes es 0.375 — lejos de 1.0: la lógica de asignación nodal **no** es estable a lo largo de las diez legislaturas. Es la evidencia directa de **sesgo** contra el candidato sin segmentación: S1, aunque predice bien en promedio, estima un mecanismo que no corresponde a ningún periodo en particular.
""")

FULL[40] = ("# 9. Conclusiones", """\
# 9. Conclusiones — la segmentación temporal óptima

## 9.1 Aplicación de la regla de decisión (§5.7)

| Candidato | Predicción (`Pond. legis` nodal / lastre) | Validez estructural (Δ) | Sesgo (heterogeneidad interna) | Varianza (n por grupo) | Estado |
|---|---|---|---|---|---|
| S1 — Pooled | 0.712 / 0.596 (empate, banda alta) | n/d (sin fronteras) | Máximo: similitud global 0.375 → estima un promedio de regímenes que no existió | Mínima (n=5,000) | Descartado por sesgo |
| S2 — Corte 2018 | **0.712 / 0.605** (encabeza o empata) | 0.076 | Moderado: `POST_2018` naive 0.687 vs. 0.730 de `PRE_2018` | Baja (n=3,500/1,500) | **Frontera óptima (predicción)** |
| S3 — Fusión T+M | 0.700 / 0.597 (empate) | **0.080** (máx.); cohesión interna máx. (0.434) | Bajo | Media (n=1,500–2,000) | **Frontera óptima (estructura)** |
| S4 — 4 cortes coalición | 0.696 / 0.585 (empate) | 0.064 | Bajo | Alta en su grupo final (n=500, naive 0.619±0.059) | Eficiente, con castigo de varianza al final |
| S5 — Pares uniformes | 0.691 / 0.591 (empate) | 0.052 (mín. de los comparables) | — | Media (n=1,000) | Descartado por validez estructural |
| S6 — Por legislatura | 0.666 / 0.586 (peor) | n/d | Mínimo | Máxima (n≈500, σ hasta ±0.071) | Descartado por varianza |

Los extremos caen cada uno por su criterio: S6 por varianza (peor predicción de todo el espacio), S1 por sesgo (predice en banda alta, pero sus coeficientes promedian regímenes con similitud media de solo 0.375), y S5 por validez estructural (granularidad comparable a S4, fronteras en el lugar equivocado). La frontera eficiente del intercambio queda en la familia de cortes políticos de granularidad baja: **S2 y S3**, con S4 justo detrás (paga la varianza de aislar la última legislatura con n=500).

## 9.2 Veredicto

**La segmentación óptima es un corte político de granularidad baja; en esta evaluación, la frontera eficiente se reduce a dos candidatos: S2 (corte único en 2018) y S3 (tres tramos con cortes en 2006 y 2015).**

- **S2** gana si la capacidad predictiva manda: encabeza lastre (0.605), empata el mejor nodal (0.712) y, con solo dos grupos (n=3,500/1,500), es el candidato de menor varianza de la familia política. Su costo: heterogeneidad interna del grupo `POST_2018` (AUC naive 0.687 vs. 0.730 del anterior) — sesgo residual.
- **S3** gana si la validez estructural manda: máximo Δ de esta corrida (0.080), máxima cohesión interna (0.434), predicción dentro del umbral de empate con S2 (−0.012 nodal, −0.008 lastre). Ofrece más resolución temporal con grupos internamente más homogéneos.

Reserva transversal: el orden fino de Δ dentro de la familia política **rota entre corridas** (S4 0.089 y S2 0.086 en ejecuciones previas; S3 0.080 en esta). El veredicto **fuerte** — estable en todas las corridas — es que la segmentación óptima usa cortes políticos con 2–4 grupos, y que los extremos y las ventanas mecánicas quedan descartados. El veredicto **débil** — específico de esta ejecución — es la preferencia puntual por S2 (predicción) o S3 (estructura).

## 9.3 Respuesta directa a la pregunta del cuaderno

La división de segmentos de tiempo que minimiza sesgo y varianza mientras maximiza capacidad predictiva y validez estructural, con LR Lasso, es **una partición de granularidad baja (2–3 grupos) cuyas fronteras siguen los cambios de coalición dominante: el corte único en la elección de 2018 si se prioriza la capacidad predictiva, o los tres tramos con cortes en 2006 y 2015 si se prioriza la validez estructural** — ambos estadísticamente empatados en predicción. Los extremos fallan por construcción — sin segmentar, el modelo estima un promedio de mecanismos que no existió (sesgo, similitud global 0.375); por legislatura, el ruido muestral domina (varianza) — y las ventanas ciegas a la política fallan por validez estructural: a igual granularidad, trazar las fronteras en los años de cambio de coalición produce bloques de coeficientes más cohesivos que trazarlas mecánicamente.

## 9.4 Limitaciones y extensiones

- **Una sola corrida para la evidencia estructural.** El Δ de §8 proviene de una ejecución con una semilla; su orden fino es inestable entre corridas. Extensión natural: *bootstrap* o multi-semilla sobre los coeficientes por legislatura para poner intervalos alrededor de Δ y convertir la preferencia S2/S3 en una afirmación con incertidumbre cuantificada.
- **Espacio de candidatos discreto y pequeño.** Se evaluaron 6 segmentaciones de las 512 particiones contiguas posibles de 10 legislaturas. Una búsqueda exhaustiva con el mismo protocolo (OOF + estratos-legislatura) es computacionalmente factible y convertiría el veredicto en un óptimo global, no solo en el mejor de seis.
- **El grupo final corto.** Cualquier segmentación que aísle la última legislatura hereda n=500; un modelo jerárquico con *partial pooling* entre segmentos permitiría resolución fina sin pagar toda la varianza. Con datos de la próxima legislatura (2027), el tramo posterior a 2018/2024 podrá reevaluarse con más muestra.
- **Las temáticas no discriminan.** El conteo de comisiones temáticas es insensible a la segmentación (banda MAE 0.815–0.819 vs. baseline 0.838) y no participa del veredicto; la conclusión aplica a los targets binarios.
""")

REPL = [
    # --- celda 7: cifras de la tabla por legislatura ---
    (7, "sube a lo largo de la serie (~0.32 al inicio → ~0.55 en la LXVI)",
        "sube a lo largo de la serie (0.280 en la LVII → 0.554 en la LXVI, con máximo de 0.578 en la LXV)"),
    (7, "El **lastre** se mantiene en torno a ~0.42–0.50 y cae al final de la serie.",
        "El **lastre** se mueve entre ~0.36 y ~0.53 y cae a 0.392 en la LXVI."),
    # --- celda 21: cifras de la vista naive ---
    (21, "los grupos de S6 oscilan de 0.611 (LEG 65, ±0.018) a 0.739 (LEG 57, ±0.056), "
         "con LEG 58 en 0.641±0.094, pura varianza muestral; y el grupo de una sola "
         "legislatura de S4 (`G4_MOR_66`, n=500) llega solo a 0.626±0.058",
         "los grupos de S6 oscilan de 0.619 (LEG 66, ±0.059) a 0.724 (LEG 57, ±0.067), "
         "con LEG 58 en 0.649±0.071, pura varianza muestral; y el grupo de una sola "
         "legislatura de S4 (`G4_MOR_66`, n=500) llega solo a 0.619±0.059"),
    (21, "el `POST_2018` de S2 rinde 0.687 dentro de grupo frente a 0.732 del `PRE_2018`",
         "el `POST_2018` de S2 rinde 0.687 dentro de grupo frente a 0.730 del `PRE_2018`"),
]


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
    for idx, (expect_start, _) in FULL.items():
        src = cell_src(cells[idx])
        if not src.lstrip().startswith(expect_start):
            errors.append(f"celda {idx}: no empieza con {expect_start!r}")
    for idx, old, _ in REPL:
        n = cell_src(cells[idx]).count(old)
        if n != 1:
            errors.append(f"celda {idx}: patrón aparece {n} veces: {old[:70]!r}")
    if errors:
        print("ABORT — el notebook no coincide con lo esperado:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    for idx, (_, new_text) in FULL.items():
        set_src(cells[idx], new_text.rstrip("\n"))
        print(f"celda {idx:2d}: reescrita")
    for idx, old, new in REPL:
        set_src(cells[idx], cell_src(cells[idx]).replace(old, new))
        print(f"celda {idx:2d}: reemplazo aplicado")

    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"\nOK — guardado {NB_PATH}")


if __name__ == "__main__":
    main()
