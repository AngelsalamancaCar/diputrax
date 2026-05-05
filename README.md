# diputrax

Análisis de patrones de reclutamiento para comisiones legislativas de la Cámara de Diputados del Congreso de la Unión de México, legislaturas LVII–LXVI (1997–presente).

**Pregunta de investigación:** ¿El perfil biográfico, educativo y de trayectoria de un diputado federal predice el tipo de comisión al que es asignado — y ese perfil ha cambiado entre épocas políticas?

## Inicio rápido

```bash
make setup    # crea .venv, instala dependencias, registra kernel de Jupyter
make notebook # abre JupyterLab
```

Para ejecutar sin interfaz (re-ejecuta todas las celdas en lugar):

```bash
make run
```

## Dependencias

| Paquete | Uso |
|---------|-----|
| pandas | carga y manipulación de datos |
| numpy | operaciones numéricas |
| matplotlib | gráficas base |
| seaborn | gráficas estadísticas |
| scikit-learn | preprocesamiento, validación cruzada |
| xgboost | clasificador/regresor gradient boosting |
| shap | importancia de features (valores SHAP) |
| statsmodels | regresión Poisson |
| scipy | pruebas estadísticas |
| jupyterlab | interfaz de notebook |
| ipykernel | registro del kernel |

## Fuente de datos

El notebook lee un archivo parquet limpio producido por el ETL [legisdatamxsil](../legisdatamxsil), que extrae perfiles públicos del Sistema de Información Legislativa (SIL) de la Secretaría de Gobernación:

```
data/database/clean/diputados_YYYYMMDD_HHMMSS.parquet
```

Una fila por diputado-legislatura (~5,000 registros totales). Ejecutar el ETL antes de correr este notebook.

**Calidad de datos relevante:**
- `edad_al_tomar_cargo` y `y_nacimiento`: 10.2% nulos (imputados por media de legislatura)
- `distrito_circ`: 4.2% nulos
- `grado_estudios_ord` en legislatura LIX: promedio anómalo (1.49 vs ~4 en otras) — posible error de captura
- 625 registros son reelecciones válidas entre legislaturas distintas

## Épocas políticas

| Época | Partido dominante | Legislaturas | Periodo | n |
|-------|-------------------|--------------|---------|---|
| ERA_1 | PRI | LVII–LIX | 1997–2006 | ~1,500 |
| ERA_2 | PAN | LX–LXII | 2006–2015 | ~1,500 |
| ERA_3 | Transición | LXIII–LXV | 2015–2021 | ~1,500 |
| ERA_4 | Morena | LXVI | 2021–presente | ~500 |

## Tipología de comisiones

| Tipo | Definición operacional | Implicación política |
|------|----------------------|----------------------|
| **Nodal** | ≥1 comisión nodal (presupuesto, hacienda, seguridad) | Alta influencia — cargo de confianza del partido mayoritario |
| **Lastre** | ≥1 comisión lastre (sin recursos ni dictámenes) | Marginación — oposición o sanción intra-partido |
| **Temáticas** | Conteo de comisiones temáticas (0–10) | Distribución negociada — no estructuralmente predecible |

## Estructura del notebook

| Sección | Descripción |
|---------|-------------|
| 1 | Resumen ejecutivo — contexto, objetivos, datos, alcance |
| 2 | Estrategia metodológica |
| 2.1 | Esquema, análisis y calidad de datos |
| 2.2 | Análisis exploratorio de datos (EDA) |
| 3 | Análisis de relaciones multivariadas |
| 4 | Estrategia de modelado: modelo Diputrax |
| 4.1 | Diseño del estudio y lógica temporal por eras |
| 4.2 | Guía de interpretación de métricas |
| 4.3 | Carga de datos y feature engineering |
| 4.4 | Infraestructura de modelado |
| 5 | Comisiones Nodales — clasificación binaria |
| 5.1 | SHAP beeswarm por era |
| 5.2 | Heatmap de importancias SHAP por era |
| 5.3 | Evolución temporal de features clave |
| 5.4 | Resultados por era — métricas AUC |
| 5.5 | Interpretación — comisiones nodales |
| 6 | Comisiones Lastre — clasificación binaria |
| 6.1 | Comparativa SHAP nodales vs lastre |
| 6.2 | Test de imagen espejo |
| 6.3 | Resultados por era — métricas AUC |
| 6.4 | Interpretación — comisiones lastre |
| 7 | Comisiones Temáticas — regresión Poisson |
| 7.1 | Resultados por era — MAE |
| 7.2 | Interpretación — comisiones temáticas |
| 8 | Análisis comparativo entre eras |
| 8.1 | Importancias SHAP consolidadas |
| 8.2 | Validación temporal — rolling forward |
| 8.3 | Interpretación — validación temporal |
| 9 | Perfiles prototípicos por era |
| 9.1 | Tabla comparativa de perfiles |
| 9.2 | Lectura comparativa — evolución del perfil nodal |
| 10 | Resumen consolidado de rendimiento (36 modelos) |
| 10.1 | Interpretación consolidada |
| 11 | Conclusiones y hallazgos clave |

## Guía de métricas

| Métrica | Target | Interpretación |
|---------|--------|----------------|
| AUC | Nodales, Lastre | 0.50 = aleatorio · 0.65–0.75 = señal moderada · >0.75 = señal fuerte |
| MAE | Temáticas | Comparado contra baseline (predecir siempre la media) |

Validación cruzada estratificada de 5 pliegues dentro de cada era. Validación temporal: entrenar en ERA k → predecir ERA k+1.

## Hallazgos clave

**H1 — Comisiones nodales moderadamente predecibles (AUC 0.62–0.73).** La señal decae a lo largo del tiempo: más fuerte bajo el PRI (ERA_1, AUC 0.734), más débil bajo Morena (ERA_4, AUC 0.619–0.643).

**H2 — Comisiones lastre esencialmente opacas (AUC 0.53–0.63).** La hipótesis de imagen espejo queda rechazada: las correlaciones SHAP entre nodal y lastre oscilan entre −0.56 y −0.68, lejos de −1.0. Son mecanismos institucionales distintos.

**H3 — Comisiones temáticas prácticamente impredecibles.** Mejora sobre el baseline ≤8.2% en ERA_1; colapsa a ≈0% en ERA_2 y ERA_4. La asignación es de naturaleza distributiva/administrativa, no meritocrática.

**H4 — ERA_2 → ERA_3 fue la ruptura más profunda.** El rolling forward muestra AUC 0.652 en esa transición, la caída más pronunciada. La fragmentación multipartidista de las legislaturas LXIII–LXV generó la mayor heterogeneidad de criterios.

**H5 — Morena legislativizó el perfil nodal.** ERA_4 premia cargos legislativos previos y formación de posgrado sobre la trayectoria administrativa que dominaba en PRI y PAN. `es_partido_mayoria` sube a |SHAP| 0.170 — filiación al bloque como requisito de acceso.

**H6 — Regresión Logística es competitiva.** LR gana o empata en la mayoría de combinaciones para nodales, sugiriendo que la estructura subyacente de asignación es en gran parte lineal.

**Predictores clave a lo largo de eras:** `es_partido_mayoria`, `n_cargos_legislativos_prev`, `n_trayectoria_admin`, `n_trayectoria_politica`, `edad_imp`, `area_Derecho`.

## Perfiles prototípicos por era

| Dimensión | ERA_1 PRI | ERA_2 PAN | ERA_3 Trans. | ERA_4 Morena |
|-----------|-----------|-----------|--------------|--------------|
| Partido mayoría | No (trayectoria sobre filiación) | Sí (militancia PAN) | No (mayoría relativa) | Sí (bloque Morena) |
| Capital político | Alto (10 cargos) | Alto (11 cargos) | Moderado (6) | Bajo pol., alta exp. leg. |
| Capital administrativo | Moderado (4) | Muy alto (13) | Moderado (4) | Alto (8) |
| Educación | Posgrado + elite | Tecnocrático privado | Licenciatura pública | Posgrado público |

## Limitaciones

| Limitación | Impacto |
|------------|---------|
| ERA_4 n≈500 (una sola legislatura) | Intervalos AUC ±0.06 — resultados orientativos |
| Anomalía `grado_estudios_ord` en LIX | Sesgo potencial en ERA_1 |
| 10.2% nulos en edad imputados | Sesgo leve hacia la media |
| Reelecciones no separadas | Perfil de reelectos puede sesgar importancias SHAP |
| Factores no observados (redes, negociaciones) | Techo real de AUC desconocido |

## Próximos pasos

1. Separar reelectos de primiparos para analizar si la lógica de asignación difiere.
2. Incluir variables de red (co-membresía previa, partido del presidente de comisión).
3. Modelar ERA_4 con más datos cuando se incorporen legislaturas LXVII y LXVIII.
4. Analizar interacciones entre `es_partido_mayoria` y `n_cargos_legislativos_prev` en ERA_4.
5. Calibrar probabilidades con Platt scaling para aplicación operativa del modelo lastre.

## License

This repository is made available for non-commercial, educational, research, and personal use.

### Code

The source code, scripts, notebooks, and software components in this repository are licensed under the **PolyForm Noncommercial License 1.0.0**.

Commercial use, including use by for-profit entities, paid consulting work, commercial products, proprietary services, or revenue-generating applications, is not permitted without prior written authorization from the copyright holder.

### Documentation and analysis

Documentation, reports, charts, and written materials are licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** unless otherwise stated.

### Data

This project may process information obtained from public institutional sources. Original public-source data remains subject to the terms, rights, and conditions of its original source. This repository only licenses the author’s code, data-processing logic, derived structures, annotations, documentation, and analytical outputs where legally applicable.

### Commercial licensing

For commercial use, please contact the repository owner to request a separate commercial license.
