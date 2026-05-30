import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# Cell 3: remove "tres algoritmos"
src3 = ''.join(cells[3]['source'])
src3 = src3.replace(
    'tres algoritmos supervisados evaluados por era mediante validación cruzada estratificada y validación temporal rolling forward',
    'Regresión Logística L1 con selección automática de variables (SelectFromModel), evaluado por era mediante validación cruzada estratificada y validación temporal rolling forward'
)
cells[3]['source'] = [src3]
print('Cell 3:', 'tres algoritmos' not in src3)

# Cell 11: replace comparativa algoritmos section
src11 = ''.join(cells[11]['source'])
OLD11 = '**Comparativa de algoritmos**\n- Se evalúan tres algoritmos (Regresión Logística, Random Forest, XGBoost) para distinguir si la estructura de la relación es lineal (LR gana) o no lineal (RF/XGB superiores), lo que tiene implicaciones para la naturaleza del mecanismo de asignación.'
NEW11 = '**Selección de variables**\n- Se emplea **Regresión Logística L1 (Lasso)** como modelo único. La penalización L1 produce coeficientes exactamente cero para features irrelevantes, haciendo la selección de variables parte del proceso de ajuste. `SelectFromModel` retiene por era únicamente las variables con coeficiente por encima de la media, cuantificando la esparsidad del mecanismo de asignación en cada régimen político.'
src11 = src11.replace(OLD11, NEW11)
cells[11]['source'] = [src11]
print('Cell 11:', 'Random Forest' not in src11)

# Cell 13: replace comparativa algoritmica bullet
src13 = ''.join(cells[13]['source'])
OLD13 = '- **Comparativa algorítmica:** Regresión Logística, Random Forest y XGBoost evaluados bajo las mismas condiciones de validación.'
NEW13 = '- **Modelo:** Regresión Logística L1 con `SelectFromModel` para selección automática de features por era. SHAP con `LinearExplainer` (valores en espacio log-odds para clasificación, log-conteo para Poisson).'
src13 = src13.replace(OLD13, NEW13)
cells[13]['source'] = [src13]
print('Cell 13:', 'Random Forest' not in src13)

# Cell 167: replace H6
src167 = ''.join(cells[167]['source'])
OLD167 = '**H6 — La Regresión Logística es competitiva**  \nLR gana o empata en la mayoría de combinaciones para nodales. Esto sugiere que la estructura subyacente de la asignación es en gran parte lineal, y que los modelos de árbol capturan ruido más que señal adicional.'
NEW167 = '**H6 — La estructura de asignación es lineal y esparsa**  \nLa Regresión Logística L1 mantiene el AUC de versiones anteriores sin necesidad de modelos de árbol. `SelectFromModel` confirma que en cada era un subconjunto reducido de features captura la señal relevante — el resto tiene coeficiente exactamente cero. La esparsidad es mayor en los mecanismos más opacos (lastre ERA_4) y menor en los más predecibles (nodales ERA_1–2).'
src167 = src167.replace(OLD167, NEW167)
cells[167]['source'] = [src167]
print('Cell 167 arbol gone:', 'árbol' not in src167)
print('Cell 167 RF gone:', 'Random Forest' not in src167)

# Cell 86: LOO reference to XGBoost in literature context - acceptable, but update count
src86 = ''.join(cells[86]['source'])
src86 = src86.replace(
    'LOO tambien seria computacionalmente prohibitivo: n=1500 modelos x 4 eras x 3 targets x 2 variantes (LR L1 full + SFM) = >36,000 entrenamientos',
    'LOO tambien seria computacionalmente prohibitivo: n=1500 modelos x 4 eras x 3 targets x 2 variantes (LR L1 full + SFM) = >36,000 entrenamientos'
)
# The XGBoost reference in cell 86 is within a literature citation, leave it
cells[86]['source'] = [src86]

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Saved.')
