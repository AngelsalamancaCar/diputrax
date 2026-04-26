# diputrax

Analysis of legislative committee recruitment patterns in Mexico's Chamber of Deputies (Cámara de Diputados, H. Congreso de la Unión), legislatures LVII–LXVI (1997–present).

**Research question:** Does a deputy's biographical, educational, and career-trajectory profile predict what type of committee they're assigned to — and has that profile changed across political eras?

## Quickstart

```bash
make setup    # create .venv, install deps, register Jupyter kernel
make notebook # open JupyterLab
```

To run headless (re-execute all cells in place):

```bash
make run
```

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | data loading and wrangling |
| numpy | numeric ops |
| matplotlib | base plotting |
| seaborn | statistical charts |
| scikit-learn | preprocessing, cross-validation |
| xgboost | gradient boosting classifier / regressor |
| shap | SHAP feature importance |
| statsmodels | Poisson regression |
| scipy | statistical tests |
| jupyterlab | notebook interface |
| ipykernel | kernel registration |

## Data source

Notebook reads a cleaned parquet file produced by the [legisdatamxsil](../legisdatamxsil) ETL pipeline:

```
data/database/clean/diputados_YYYYMMDD_HHMMSS.parquet
```

One row per deputy-legislature. Run the ETL pipeline in `legisdatamxsil` before executing this notebook.

## Political eras

| Era | Dominant party | Legislatures |
|-----|---------------|--------------|
| ERA_1 | PRI | LVII–LIX |
| ERA_2 | PAN | LX–LXII |
| ERA_3 | Transition | LXIII–LXIV |
| ERA_4 | Morena | LXV–LXVI |

## Committee typology

| Type | Operational definition | Political implication |
|------|----------------------|----------------------|
| **Nodal** | ≥1 nodal commission (presupuesto, hacienda, seguridad) | High influence — majority-party trust assignment |
| **Lastre** | ≥1 lastre commission (no resources, no dictámenes) | Marginalization — opposition or intra-party sanction |
| **Temáticas** | Count of thematic commissions (0–10) | Distributed by negotiation — not structurally predicted |

## Notebook structure

| Section | Description |
|---------|-------------|
| 1 | Executive summary — context, objectives, data, scope |
| 2 | Methodological strategy |
| 3 | Regulatory compliance and interpretability requirements |
| 4–5 | Analysis scope and out-of-scope |
| 6 | Exploratory data analysis (EDA) |
| 6.1 | Data quality audit |
| 6.2 | Legislative distribution |
| 6.3 | Demographic profile |
| 6.4 | Educational profile |
| 6.5 | Prior experience |
| 6.6 | Career trajectories |
| 6.7 | Commission participation |
| 6.8 | Multivariate relations |
| 7 | Diputrax model |
| 7.1 | Data loading and feature engineering |
| 7.2 | Modeling infrastructure |
| 7.3 | Nodal commissions — binary classification |
| 7.4 | Lastre commissions — binary classification |
| 7.5 | Thematic commissions — Poisson regression |
| 7.6 | Cross-era importance comparison |
| 7.7 | Temporal validation — rolling forward |
| 7.8 | Prototypical profiles (highest SHAP) |
| 7.9 | Summary table — all models |
| 7.10 | Interpretation guide |
| 8 | Conclusions and key findings |

## Key findings

- **Nodal** committees are moderately predictable (AUC 0.62–0.73). Signal decays across eras: strongest under PRI (ERA_1), weakest under Morena (ERA_4).
- **Lastre** committees are essentially opaque (AUC 0.53–0.63). Mirror-image hypothesis rejected — nodal and lastre operate under partially independent assignment logic (SHAP correlation −0.56 to −0.68, not −1.0).
- **Thematic** commissions are unpredictable from biographical profile alone (≤8% improvement over mean baseline). Assignment is driven by unobservable intra-party negotiations.
- **Temporal transfer** (train ERA k → predict ERA k+1) is solid for ERA_1→ERA_2 and ERA_3→ERA_4 (AUC ≈ 0.71), with a notable drop at ERA_2→ERA_3 reflecting political fragmentation.
- Top predictors across eras: `es_partido_mayoria`, `n_cargos_legislativos_prev`, `n_trayectoria_admin`, `n_trayectoria_politica`.
