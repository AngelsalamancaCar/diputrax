# diputrax

Exploratory data analysis of Mexico's federal deputies across all legislatures.

Reads ETL snapshots produced by [legisdatamxsil](../legisdatamxsil) and produces charts on party composition, gender, age, education, prior experience, commissions, and administrative careers.

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
| jupyterlab | notebook interface |
| ipykernel | kernel registration |

## Data source

Notebook reads from the most recent timestamped subdirectory of:

```
/home/miso/Projects/legisdatamxsil/data/etl/YYYYMMDD_HHMMSS/
```

Each directory contains one CSV per legislature. Run the ETL pipeline in `legisdatamxsil` before executing this notebook.

## Analyses

| Section | Description |
|---------|-------------|
| 1 | Load latest ETL snapshot |
| 2 | Schema & missing-value audit |
| 3 | Deputies per legislature |
| 4 | Party distribution & share over time |
| 5 | Gender & election type (MR vs PR) |
| 6 | Age at time of office |
| 7 | Education level |
| 8 | Field of study |
| 9 | Prior legislative experience |
| 10 | Commission participation |
| 11 | Administrative career trajectory |
| 12 | Academic profile |
| 13 | Correlation heatmap (numeric features) |
| 14 | Pairplot — career trajectories by party |
| 15 | Key stats summary |
