.DEFAULT_GOAL := help
VENV         := .venv
PYTHON       := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
JUPYTER      := $(VENV)/bin/jupyter

.PHONY: help setup install kernel notebook run clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

setup: $(VENV)/bin/activate install kernel ## Create venv, install deps, register kernel

install: $(VENV)/bin/activate ## Install Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install pandas numpy matplotlib seaborn jupyterlab ipykernel \
	               scikit-learn xgboost shap statsmodels scipy

kernel: ## Register venv as Jupyter kernel named 'diputrax'
	$(PYTHON) -m ipykernel install --user --name diputrax --display-name "diputrax"

notebook: ## Launch JupyterLab
	$(JUPYTER) lab eda_diputados.ipynb

run: ## Execute EDA notebook non-interactively, in-place
	$(JUPYTER) nbconvert --to notebook --execute --inplace eda_diputados.ipynb

modelo: ## Execute model notebook non-interactively, in-place
	$(JUPYTER) nbconvert --to notebook --execute --inplace modelo_diputados.ipynb

clean: ## Remove venv and Jupyter checkpoints
	rm -rf $(VENV) .ipynb_checkpoints
