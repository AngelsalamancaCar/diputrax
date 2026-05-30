import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

src86 = ''.join(cells[86]['source'])

src86 = src86.replace(
    'aqui, donde CV selecciona el mejor modelo entre LR, RF y XGBoost y tambien reporta el AUC final. Usar CV no estratificada en este contexto introduce sesgo de seleccion hacia el algoritmo que mas se beneficia de folds desbalanceados.',
    'aqui, donde CV reporta el AUC del modelo LR L1 + SelectFromModel. Usar CV no estratificada en este contexto introduce sesgo de seleccion que afecta la estimacion del AUC final.'
)

src86 = src86.replace(
    'Forman, G., & Scholz, M. (2010). Apples-to-apples in cross-validation studies: Pitfalls in classifier performance measurement. *SIGKDD Explorations*, 12(1), 49–57.',
    'Forman, G., & Scholz, M. (2010). Apples-to-apples in cross-validation studies: Pitfalls in classifier performance measurement. *SIGKDD Explorations*, 12(1), 49–57.'
)

# Fix the conclusion line that references tres algoritmos
src86 = src86.replace(
    'Esto refuerza la decision de fijar k=5 uniforme no solo entre eras sino entre los tres algoritmos evaluados (LR, RF, XGBoost).',
    'Esto refuerza la decision de fijar k=5 uniforme entre eras para garantizar comparabilidad directa de los AUC.'
)

cells[86]['source'] = [src86]

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

remaining = [t for t in ['Random Forest', 'XGBoost', 'tres algoritmos'] if t.lower() in src86.lower()]
print('Cell 86 remaining:', remaining if remaining else 'Clean')
