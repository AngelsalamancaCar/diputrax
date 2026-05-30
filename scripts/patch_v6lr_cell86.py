import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

src86 = ''.join(cells[86]['source'])
src86 = src86.replace(
    'LOO (k=n) tiene varianza asimptotica mas alta que k-fold para estimadores no lineales (RF, XGBoost), lo que lo hace suboptimo incluso en muestras pequenas.',
    'LOO (k=n) tiene varianza asimptotica mas alta que k-fold para estimadores no lineales, lo que lo hace suboptimo incluso en muestras pequenas.'
)
src86 = src86.replace(
    'Forman & Scholz (2010) demuestran empiricamente que comparar clasificadores con k distinto produce rankings incorrectos entre algoritmos. Esto refuerza la decision de fijar k=5 uniforme entre eras para garantizar comparabilidad directa de los AUC.',
    'Forman & Scholz (2010) demuestran empiricamente que comparar modelos con k distinto produce rankings incorrectos. Esto refuerza la decision de fijar k=5 uniforme entre eras para garantizar comparabilidad directa de los AUC.'
)
# Also fix the summary table if it still mentions RF/XGBoost
src86 = src86.replace(
    'k=n (LOO) | AUC indefinida en fold de 1 obs.; >54k entrenamientos | Indefinicion estadistica + costo computacional',
    'k=n (LOO) | AUC indefinida en fold de 1 obs.; >18k entrenamientos | Indefinicion estadistica + costo computacional'
)
cells[86]['source'] = [src86]

with open(r'C:\Users\zigma\Projects\diputrax\notebooks\diputraxv6LR.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Verify
remaining = [t for t in ['Random Forest', 'XGBoost', 'tres algoritmos', '3 algoritmos'] if t.lower() in src86.lower()]
print('Cell 86 remaining stale terms:', remaining if remaining else 'None')
print('Done.')
