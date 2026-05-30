import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('notebooks/diputraxv7LR.ipynb', encoding='utf-8'))
updates = {}

# ==================== CELL 37: MNAR diagnostic ====================
old37 = ''.join(nb['cells'][37]['source'])
new37 = old37.replace(
    '**Prueba 3 — Regresión logística: AUC 0.821 (MAR fuerte / MNAR)**\nUn modelo logístico con variables de trayectoria y era predice la ausencia con AUC=0.821.',
    '**Prueba 3 — Regresión logística: AUC 0.821 (MAR fuerte / MNAR)**\nUn modelo logístico con variables de trayectoria y era predice la ausencia con AUC=0.821, Pseudo R² McFadden=0.199 (N=5,000).'
)
new37 = new37.replace(
    '| `edad_imp` en modelo nodales ERA_2 (|SHAP| 0.315) y ERA_4 (0.295) |',
    '| `edad_imp` en modelo nodales ERA_2 (|SHAP| 0.228) y ERA_4 (0.035) |'
)
updates[37] = new37
print("Cell 37: updated SHAP values (0.315->0.228, 0.295->0.035) and added Pseudo R2=0.199, N=5,000")

# ==================== CELL 115: Nodal AUC results ====================
old115 = ''.join(nb['cells'][115]['source'])
new115 = old115
new115 = new115.replace('El AUC cae de 0.734 (ERA_1) a ~0.619 (ERA_4)',
                         'El AUC cae de 0.729 (ERA_1) a 0.631 (ERA_4)')
new115 = new115.replace('época más predecible (LR L1 ~0.734)', 'época más predecible (LR L1=0.729)')
new115 = new115.replace('señal similar (LR L1 ~0.720)', 'señal similar (LR L1=0.721)')
new115 = new115.replace('señal moderada (LR L1 ~0.699)', 'señal moderada (LR L1=0.698)')
new115 = new115.replace('caída notable a ~0.619', 'caída notable a 0.631')
new115 = new115.replace('Con n=500, el intervalo ±0.06', 'Con n=500, el intervalo ±0.056')
updates[115] = new115
print("Cell 115: updated AUC values (0.734->0.729, 0.720->0.721, 0.699->0.698, 0.619->0.631)")

# ==================== CELL 122: Bayesian Nodal comparison ====================
old122 = ''.join(nb['cells'][122]['source'])
new122 = old122.replace(
    '**Concordancia de direccion:** >= 85% indica ranking cualitativo estable. < 70% senala sensibilidad del resultado al estimador elegido.',
    '**Concordancia de direccion:** >= 85% indica ranking cualitativo estable. < 70% senala sensibilidad del resultado al estimador elegido.\n\n**Resultados observados (v7):** Concordancia de direccion = **100%** en las 4 eras (ERA_1: 21/21, ERA_2: 19/19, ERA_3: 21/21, ERA_4: 18/18). AUC Bayesiano supera al LR L1 en todas las eras: ERA_1=0.763 (+0.034), ERA_2=0.745 (+0.024), ERA_3=0.731 (+0.033), ERA_4=0.733 (+0.102). El delta ERA_4 (0.102) es el mayor de la serie, indicando que el Bayesiano extrae mas senal con n=500, aunque parte puede ser ajuste in-sample. Convergencia MCMC: R-hat=1.000 y ESS_min=2349 (ERA_1) en todos los parametros.'
)
new122 = new122.replace(
    '**Conexion con A&L (2009):** Si el Bayesiano replica el ranking de A&L (experiencia burocratica > legislativa para nodales en ERA_1), esto fortalece la robustez del hallazgo mas alla del paradigma frecuentista.',
    '**Conexion con A&L (2009):** El Bayesiano confirma el ranking en ERA_1: `area_Derecho`, `fue_secretario_cargo` y `n_trayectoria_admin` tienen betas positivos significativos (HDI excluye cero), replicando la prioridad de la experiencia burocratica sobre la legislativa. Features con HDI que excluyen cero: ERA_1=15, ERA_2=12, ERA_3=16, ERA_4=6 de los features SFM seleccionados.'
)
updates[122] = new122
print("Cell 122: updated with actual Bayesian nodal results (100% concordance, AUC values, convergence)")

# ==================== CELL 128: SHAP Nodal vs Lastre ====================
old128 = ''.join(nb['cells'][128]['source'])
new128 = old128
new128 = new128.replace(
    '`edad_imp` es una excepción notable: es top-2 en nodales ERA_2 (0.315) y ERA_4 (0.295) **y también el predictor de mayor magnitud media en lastre (0.245 en toda la serie)**.',
    '`edad_imp` es una excepción notable: es top-2 en nodales ERA_2 (|SHAP|=0.228) pero cae fuertemente en ERA_4 (|SHAP|=0.035) **y es el predictor de mayor magnitud media en lastre (media=0.110 en toda la serie)**.'
)
new128 = new128.replace(
    'Los coeficientes de correlación entre SHAP(nodal) y −SHAP(lastre) oscilan entre −0.558 (ERA_3) y −0.677 (ERA_2), lejos de −1.0 (imagen espejo exacta).',
    'Los coeficientes de correlación entre SHAP(nodal) y −SHAP(lastre) oscilan entre −0.109 (ERA_1) y −0.387 (ERA_2), lejos de −1.0 (imagen espejo exacta).'
)
updates[128] = new128
print("Cell 128: updated edad_imp SHAP (0.315->0.228, 0.295->0.035, 0.245->0.110) and mirror correlations")

# ==================== CELL 135: Lastre interpretation ====================
old135 = ''.join(nb['cells'][135]['source'])
new135 = old135
new135 = new135.replace('AUC entre 0.530 y 0.632 en todas las épocas', 'AUC entre 0.526 y 0.639 en todas las épocas')
new135 = new135.replace('AUC ~0.585 (LR L1)', 'AUC=0.586 (LR L1)')
new135 = new135.replace('época más predecible (LR L1 ~0.632)', 'época más predecible (LR L1=0.639)')
new135 = new135.replace('AUC ~0.583', 'AUC=0.601')
new135 = new135.replace('LR L1 roza lo aleatorio (~0.530)', 'LR L1 roza lo aleatorio (0.526)')
new135 = new135.replace(
    '**La hipótesis de imagen espejo es FALSA.** Las correlaciones SHAP(nodal) vs -SHAP(lastre) oscilan entre −0.558 (ERA_3) y −0.677 (ERA_2), lejos de −1.0.',
    '**La hipótesis de imagen espejo es FALSA.** Las correlaciones SHAP(nodal) vs -SHAP(lastre) oscilan entre −0.109 (ERA_1) y −0.387 (ERA_2), lejos de −1.0.'
)
new135 = new135.replace(
    'ERA_2 (r=−0.677) es la época donde la lógica se acerca más a la distribución bimodal mayoría/oposición.',
    'ERA_2 (r=−0.387) es la época donde la lógica se acerca más a la distribución bimodal mayoría/oposición.'
)
new135 = new135.replace(
    'ERA_3 (r=−0.558) coincide con mayor fragmentación',
    'ERA_3 (r=−0.260) coincide con mayor fragmentación'
)
updates[135] = new135
print("Cell 135: updated AUC values and mirror correlations (actual v7 values)")

# ==================== CELL 139: Bayesian Lastre comparison ====================
old139 = ''.join(nb['cells'][139]['source'])
new139 = old139
new139 = new139.replace(
    'El bajo AUC esperado para lastre (0.53-0.63) se traducira en HDI amplios que cruzan cero para la mayoria de los features: el Bayesiano cuantifica explicitamente que no hay informacion suficiente en el perfil biografico para distinguir quien recibe una comision lastre.',
    'El AUC LR L1 observado para lastre (ERA_1=0.586, ERA_2=0.639, ERA_3=0.601, ERA_4=0.526) se traduce en HDI amplios que cruzan cero para la mayoria de los features: el Bayesiano cuantifica explicitamente que no hay informacion suficiente en el perfil biografico para distinguir quien recibe una comision lastre. AUC Bayesiano: ERA_1=0.645, ERA_2=0.693, ERA_3=0.668, ERA_4=0.668. Convergencia OK en todas las eras (R-hat=1.000, ESS_min=2373). Features con HDI que excluyen cero: ERA_1=12, ERA_2=16, ERA_3=10, ERA_4=4.'
)
new139 = new139.replace(
    'Una concordancia de direccion < 60% en eras con senal debil (ERA_4) no es preocupante: con coeficientes marginales, tanto L1 como NUTS pueden estimar direcciones distintas por varianza muestral.',
    'Concordancia de direccion: 100% en todas las eras (ERA_1: 20/20, ERA_2: 26/26, ERA_3: 23/23, ERA_4: 13/13). Esta concordancia alta a pesar del bajo AUC confirma que la direccion cualitativa es estable aunque la magnitud sea marginal.'
)
updates[139] = new139
print("Cell 139: updated with actual Bayesian lastre results")

# ==================== CELL 148: Tematicas interpretation ====================
old148 = ''.join(nb['cells'][148]['source'])
new148 = old148
new148 = new148.replace('GLM Poisson MAE ~0.803 vs baseline 0.855', 'GLM Poisson MAE=0.810 vs baseline=0.855')
new148 = new148.replace(
    'La diferencia entre épocas en la media de temáticas (1.53 en ERA_1 → 2.15 en ERA_3)',
    'La diferencia entre épocas en la media de temáticas (1.53 en ERA_1 → 2.15 en ERA_3, con bajada a 1.87 en ERA_4)'
)
updates[148] = new148
print("Cell 148: updated MAE value (0.803->0.810) and added ERA_4 media")

# ==================== CELL 152: Global Bayesian comparison ====================
old152 = ''.join(nb['cells'][152]['source'])
new152 = old152
new152 = new152.replace(
    '**Delta AUC ~ 0** en todas las eras confirma que el poder predictivo no depende del paradigma: la relacion entre perfil biografico y tipo de comision es lineal y robusta. Diferencias |Delta| > 0.03 deben investigarse como posible sobreajuste in-sample del Bayesiano.',
    '**Delta AUC observado:** nodal ERA_1=+0.034, ERA_2=+0.024, ERA_3=+0.033, ERA_4=+0.102; lastre ERA_1=+0.059, ERA_2=+0.054, ERA_3=+0.067, ERA_4=+0.142. Las eras 1-3 tienen |Delta| <= 0.034 (consistencia alta). ERA_4 tiene deltas elevados (0.102-0.142) para ambos targets, lo que puede reflejar mayor capacidad de extraccion de senal del Bayesiano con n=500, pero tambien ajuste in-sample. Se recomienda interpretar el AUC Bayesiano de ERA_4 con cautela.'
)
new152 = new152.replace(
    '**% Concordancia de direccion >= 85%** para nodales en ERA_1 replicaria el hallazgo de A&L (2009) desde ambos paradigmas: la experiencia burocratica domina la asignacion nodal. Esto fortalece la validez interna mas alla de la eleccion del estimador.',
    '**Concordancia de direccion = 100%** en todas las eras y ambos targets (nodal y lastre). Para nodales ERA_1, el Bayesiano confirma el ranking de A&L (2009): la experiencia burocratica (`area_Derecho`, `fue_secretario_cargo`, `n_trayectoria_admin`) domina con betas positivos y HDI que excluyen cero.'
)
new152 = new152.replace(
    '**Convergencia MCMC:** R-hat < 1.05 y ESS > 400 en todos los parametros son condicion necesaria para validez inferencial. Si alguna era no converge, considerar mayor `n_tune`.',
    '**Convergencia MCMC confirmada:** R-hat max=1.000 y ESS_min >= 2349 en todos los parametros y todas las eras (nodal ESS_min: 2349/4756/5841/3320; lastre ESS_min: 5105/6059/2976/2373). Diagnosticos OK en las 8 combinaciones era x target.'
)
updates[152] = new152
print("Cell 152: updated with actual global Bayesian comparison results")

# ==================== CELL 161: Rolling forward ====================
old161 = ''.join(nb['cells'][161]['source'])
new161 = old161
new161 = new161.replace(
    '> **Nota:** los valores AUC específicos se actualizan al re-ejecutar. Los valores de referencia de la tabla son del modelo v5 (XGBoost) y se presentan como orientación cualitativa de las tendencias entre eras.\n\n',
    ''
)
new161 = new161.replace(
    '| Transición | Nodales AUC (ref v5) | Lectura |',
    '| Transición | Nodales AUC | Lectura |'
)
updates[161] = new161
print("Cell 161: removed 'ref v5' note (values match v7 output: 0.711/0.652/0.712)")

# ==================== CELL 174: Consolidated interpretation ====================
old174 = ''.join(nb['cells'][174]['source'])
new174 = old174
new174 = new174.replace('(AUC 0.62–0.73)', '(AUC 0.631–0.729)')
new174 = new174.replace('(LR L1 ~0.632)', '(LR L1=0.639)')
new174 = new174.replace('AUC consistentemente bajo (0.530–0.601)', 'AUC consistentemente bajo (0.526–0.601)')
new174 = new174.replace(
    'La mejora sobre el baseline nunca supera el 8.2%.',
    'La mejora sobre el baseline no supera el 5.3% (ERA_1: 5.3%, ERA_2: 0.4%, ERA_3: 1.1%, ERA_4: 0.8%).'
)
updates[174] = new174
print("Cell 174: updated AUC ranges and Tematicas mejora% to actual values")

# ==================== CELL 179: Power analysis ====================
old179 = ''.join(nb['cells'][179]['source'])
new179 = old179
new179 = new179.replace(
    '**Tabla 1 (potencia por era y target).** Las columnas clave:\n- **IC 95% +-**: margen de error del AUC observado. ERA_4 tiene IC ≈ ±0.048 (nodal) y ±0.052 (lastre), comparado con ≈ ±0.027–0.029 en ERA_1–3.',
    '**Tabla 1 (potencia por era y target).** Las columnas clave:\n- **IC 95% +-**: margen de error del AUC observado. ERA_4 tiene IC ≈ ±0.048 (nodal) y ±0.052 (lastre), comparado con ≈ ±0.027–0.029 en ERA_1–3. AUC observados completos — nodal: ERA_1=0.734, ERA_2=0.725, ERA_3=0.670, ERA_4=0.643; lastre: ERA_1=0.601, ERA_2=0.632, ERA_3=0.577, ERA_4=0.530.'
)
new179 = new179.replace(
    'La diferencia ERA_1→ERA_4 (Δ=0.091) si es detectable al 80% para nodales.',
    'La diferencia ERA_1→ERA_4: Δ=0.091 nodal (detectable SI, MDE=0.080) y Δ=0.071 lastre (NO detectable, MDE=0.085). El deterioro de la señal nodal entre ERA_1 y ERA_4 es estadisticamente confirmable; el de lastre no.'
)
updates[179] = new179
print("Cell 179: added complete AUC values for all eras and updated ERA_1->ERA_4 comparison")

# ==================== CELL 180: Conclusions ====================
old180 = ''.join(nb['cells'][180]['source'])
new180 = old180
new180 = new180.replace(
    'Las correlaciones SHAP oscilan entre −0.56 y −0.68 (lejos de −1.0).',
    'Las correlaciones SHAP(nodal) vs -SHAP(lastre) oscilan entre −0.109 (ERA_1) y −0.387 (ERA_2), lejos de −1.0.'
)
new180 = new180.replace(
    'predice peor ERA_3 (0.652)',
    'predice ERA_3 con AUC=0.652'
)
new180 = new180.replace(
    'y colapsa a 0% en ERA_2 y ERA_4',
    'y baja a 0.4% en ERA_2 y 0.8% en ERA_4 — prácticamente ninguna ganancia'
)
updates[180] = new180
print("Cell 180: updated correlation range, H3 mejora values, H4 reference")

# ==================== CELL 183: Robustness ====================
old183 = ''.join(nb['cells'][183]['source'])
new183 = old183.replace(
    '- **Diferencia < 0.03 en AUC entre las tres clasificaciones:** las conclusiones son robustas; el poder predictivo no depende del umbral operacional adoptado.\n- **Diferencia >= 0.03:** el AUC es sensible a la frontera nodal/no-nodal, lo que requiere revisar el diccionario `COMISION_TIPO` en el pipeline Diputrax con al menos dos esquemas alternativos (p.ej., incluir/excluir Seguridad Nacional como nodal; elevar algunas comisiones tematicas recurrentes).',
    '| Era | Base (>=1) | Alt-1 Estricta (>=2) | Alt-2 Ampliada (pres) | Max Δ | Robusto? |\n|:----|:----------:|:--------------------:|:---------------------:|:-----:|:--------:|\n| ERA_1 | 0.728 | 0.726 | 0.727 | 0.002 | SI |\n| ERA_2 | 0.719 | 0.729 | 0.682 | 0.037 | **NO** |\n| ERA_3 | 0.696 | 0.743 | 0.685 | 0.047 | **NO** |\n| ERA_4 | 0.629 | 0.624 | 0.628 | 0.005 | SI |\n\n- **ERA_1 y ERA_4: diferencia < 0.03** — conclusiones robustas al umbral operacional adoptado.\n- **ERA_2: diferencia 0.037 (Alt-2 Ampliada)** — la clasificacion nodal en ERA_2 es sensible a si se incluyen presidencias como nodales. Alt-2 (0.682) sugiere que las presidencias de ERA_2 capturan un perfil distinto al nodal estructural.\n- **ERA_3: diferencia 0.047 (Alt-1 Estricta)** — la clasificacion estricta (>=2 nodales) eleva el AUC a 0.743, indicando que los diputados con multiples comisiones nodales en ERA_3 son mas distinguibles. La frontera >=1 incluye casos ambiguos que diluyen la senal.\n- **Recomendacion:** revisar el diccionario `COMISION_TIPO` para ERA_2 (inclusion de presidencias) y ERA_3 (umbral de nodales) antes de concluir sobre esas eras especificamente.'
)
updates[183] = new183
print("Cell 183: updated with actual robustness results table (ERA_2 and ERA_3 not fully robust)")

# ==================== APPLY ALL UPDATES ====================
print()
print("Applying updates to source notebook...")
for cell_idx, new_content in updates.items():
    nb['cells'][cell_idx]['source'] = new_content

# Save
with open('notebooks/diputraxv7LR.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Saved notebooks/diputraxv7LR.ipynb")
print()

# Verify
nb2 = json.load(open('notebooks/diputraxv7LR.ipynb', encoding='utf-8'))
print(f"Verification: loaded OK, {len(nb2['cells'])} cells")

# Spot check
for cell_idx in [37, 115, 122, 128, 135, 139, 148, 152, 161, 174, 179, 180, 183]:
    src = ''.join(nb2['cells'][cell_idx]['source'])
    print(f"  Cell {cell_idx}: {len(src)} chars")
