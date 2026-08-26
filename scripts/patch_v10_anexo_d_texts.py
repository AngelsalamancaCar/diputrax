"""Corrige los textos interpretativos del Anexo D (celdas 176 y 182) de
diputraxv10.ipynb para que coincidan con el output real/comiteado.

Discrepancias corregidas (verificadas re-ejecutando el Anexo D):
- Celda 176 (D.1 Nodales, ERA_1): la narrativa afirmaba pseudo R²=0.161,
  LLR p<1e-50 y admin_en_sindicato β=-9.45. El output real muestra que la MLE
  NO converge en ERA_1 (pseudo R²=-inf, LLR p=1.00, admin_en_sindicato
  β≈+9.7e10, intercepto≈5.6e9). Los coeficientes no afectados por la separación
  sí se estiman con normalidad. También se corrigen recall/F1 de ERA_1 y el
  pseudo R² de ERA_4 (0.125, no 0.134).
- Celda 182 (D.4 Poisson): la narrativa afirmaba que en ERA_4 "ninguna variable
  alcanza significancia al 10%" (falso: n_trayectoria_legislativa IRR=0.877,
  p=0.011) y citaba n_cargos_legislativos_prev en ERA_1 (variable no presente en
  KEY_FEATS; la real es fue_diputado_federal, IRR=0.934, p=0.009). Se añade la
  significancia de n_trayectoria_legislativa en ERA_3 y se ajustan pseudo R².
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "diputraxv10.ipynb"

CELL_176 = """**Interpretación — D.1 Regresión Logística clásica (Nodales)**

**ERA_1 — PRI** (n=1500, **la MLE no penalizada no converge**): el ajuste global no es utilizable —`Pseudo R² McFadden = −inf`, `LLR p = 1.00`, `Convergió = False`— porque `admin_en_sindicato` produce **separación cuasi-completa**: casi ningún diputado con trayectoria sindical recibió comisión nodal en ERA_1, y la verosimilitud diverge llevando ese coeficiente a +∞ (β≈9.7×10¹⁰, OR=∞) y arrastrando al intercepto (β≈5.6×10⁹). **Los coeficientes no afectados por la separación sí se estiman con normalidad y son legibles:** `area_Derecho` es el más fuerte (β=0.467, OR=1.60, *p*<0.001) —una desviación estándar adicional en la señal de formación jurídica multiplica por 1.6 la razón de momios de recibir comisión nodal, confirmando en escala de razón de momios lo que SHAP ya mostraba en magnitud—, seguido de `area_Económico-Financiera` (OR=1.38, *p*<0.001), `fue_secretario_cargo` (OR=1.25, *p*<0.001) y `n_trayectoria_empresarial` (OR=1.17, *p*=0.010). `reg_SUR` es negativo y significativo (OR=0.81, *p*=0.002) —penalización regional que SHAP no distingue por signo—. Lo que **no** puede leerse en esta era es ningún estadístico de ajuste global (pseudo R², LLR, AIC/BIC): al no converger, carecen de sentido.

> **Advertencia de separación cuasi-completa:** el colapso numérico se concentra en `admin_en_sindicato` y el intercepto; el resto de la tabla converge con normalidad y se reporta con normalidad. Esto **no invalida** los coeficientes estables, pero sí confirma empíricamente por qué el modelo productivo usa L1: la penalización habría encogido `admin_en_sindicato` a un valor finito en vez de dejar que la verosimilitud divergiera y contaminara los diagnósticos globales de la era.

**ERA_2 — PAN** (n=1500, pseudo R² = 0.138, converge sin problemas): `sexo_bin` es significativo por primera vez en la serie (β=0.167, OR=1.18, *p*=0.007), confirmando en términos clásicos la emergencia de género como señal activa que SHAP ya ubicaba en ERA_2. `edad_imp` es negativo y fuerte (OR=0.710, *p*<0.001) —una vez controlado por trayectoria, mayor edad estandarizada se asocia con *menor* probabilidad de nodal, patrón que solo el signo del coeficiente clásico revela (SHAP solo reporta magnitud)—. `area_Derecho` sigue dominando (OR=1.52, *p*<0.001).

**ERA_3 — Transición** (n=1500, pseudo R² = 0.129, converge sin variables NaN): `n_trayectoria_legislativa` es el coeficiente positivo más fuerte (OR=1.49, *p*<0.001); `sexo_bin` se mantiene alto (OR=1.22, *p*<0.001) y `p_MORENA` es significativo (OR=1.28, *p*=0.004), confirmando en formato clásico el hallazgo SHAP de que `p_MORENA` —no `es_partido_mayoria`— es la señal partidista activa en esta era.

**ERA_4 — Morena** (n=500, pseudo R² = 0.125, LLR *p* = 1.4×10⁻¹¹): `area_Derecho` (OR=1.56, *p*<0.001), `tiene_exp_juvenil` (OR=1.30, *p*=0.014) y `sexo_bin` (OR=1.26, *p*=0.025) son los coeficientes estables. `es_partido_mayoria` y `p_MORENA` **no son estimables por separado** (SE≈8.5×10⁶, *z*=0, *p*=1.00): en una era de una sola legislatura con supermayoría de Morena, ambas *dummies* son numéricamente equivalentes dentro de la muestra (colinealidad exacta, confirmada por VIF = ∞ en D.3) —un artefacto estructural del diseño de ERA_4, no un error de especificación—.

**Desempeño de clasificación (umbral 0.5, sin balanceo de clases):** *precision* 0.65–0.69, *recall* 0.40–0.70, F1 0.50–0.69. El *recall* más bajo es ERA_1 (0.40), reflejo directo de que esta variante —a diferencia del modelo productivo `LR L1 (full)`— no usa `class_weight="balanced"`, por lo que subpredice la clase positiva minoritaria (32.2 % de tasa nodal en ERA_1)."""

CELL_182 = """**Interpretación — D.4 GLM Poisson clásico (Temáticas)**

Los cuatro modelos convergen sin problemas de separación (a diferencia de D.1–D.2) —el enlace log del Poisson y la ausencia de *dummies* de categoría rara en `KEY_FEATS` evitan el problema estructural de los binarios—.

**Pseudo R² por deviance:** 0.023 (ERA_1) → 0.017 (ERA_2) → 0.029 (ERA_3) → 0.059 (ERA_4). Son valores bajos en las cuatro eras —consistente con el hallazgo ya establecido en §7.2 de que las temáticas siguen una lógica distributiva no capturada por el perfil biográfico—, pero **crecen hacia ERA_4**, con el modelo de Morena explicando más del doble de la varianza en deviance que los tres anteriores.

**Dispersión (Pearson χ²/gl):** 0.642 / 0.582 / 0.659 / 0.464 — todas **por debajo de 1**, es decir, **subdispersión** respecto al supuesto Poisson (varianza menor que la media condicional). Es el patrón inverso al problema típico de sobredispersión que motiva binomial-negativa en la literatura de recuento; aquí el techo institucional de comisiones temáticas (0–10) comprime la varianza observada. No invalida el Poisson —la subdispersión no sesga los coeficientes, solo hace que los errores estándar reportados sean conservadores (ligeramente más anchos de lo estrictamente necesario)—.

**Coeficientes significativos por era (α=0.05):** ERA_1 — `n_trayectoria_politica` (IRR=1.096, *p*<0.001, dirección positiva) y `fue_diputado_federal` (IRR=0.934, *p*=0.009, negativa). ERA_2 — solo `edad_imp` (IRR=1.069, *p*=0.001). ERA_3 — `es_partido_mayoria` (IRR=1.051, *p*=0.005, positiva) y `n_trayectoria_legislativa` (IRR=0.933, *p*=0.011, negativa). ERA_4 — la única señal individual significativa es `n_trayectoria_legislativa` (IRR=0.877, *p*=0.011, negativa): más experiencia legislativa previa se asocia con **menos** comisiones temáticas. Es notable que `n_trayectoria_legislativa` sea significativa y negativa tanto en ERA_3 como en ERA_4 —un patrón consistente de que los perfiles legislativos veteranos reciben *menos* carga temática en las dos eras más recientes—. Pese a que ERA_4 tiene el pseudo R² más alto de la serie, el efecto se concentra en esa sola variable y el resto de `KEY_FEATS` no alcanza significancia individual, coherente con la lectura cualitativa de §7.2: la asignación temática de Morena responde a una lógica más distributiva y menos jerárquica, con la experiencia legislativa previa como único filtro individual detectable a *n*=500."""


def as_source(text):
    """nbformat guarda source como lista de líneas con '\\n' salvo la última."""
    return text.splitlines(keepends=True)


def main():
    nb = json.load(open(NB, encoding="utf-8"))
    cells = nb["cells"]

    targets = {176: ("D.1", CELL_176), 182: ("D.4", CELL_182)}
    for idx, (tag, newtext) in targets.items():
        cur = "".join(cells[idx]["source"])
        assert cells[idx]["cell_type"] == "markdown", f"celda {idx} no es markdown"
        assert f"Interpretación — {tag}" in cur, (
            f"celda {idx} no contiene 'Interpretación — {tag}'; "
            f"encabezado actual: {cur[:80]!r}")
        cells[idx]["source"] = as_source(newtext)
        print(f"celda {idx} ({tag}) actualizada: "
              f"{len(cur)} -> {len(newtext)} chars")

    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook guardado:", NB)


if __name__ == "__main__":
    main()
