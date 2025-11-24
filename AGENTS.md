# Repository Guidelines

## Project Structure & Module Organization
- Analysis lives in numbered notebooks at the repo root (`1.PAS_DA_EDA.ipynb`, `2.Data Preparation and Cleaning.ipynb`, `3. Model Selection and Initialization.ipynb`). Use these in order to reproduce the workflow.
- Raw datasets are stored in `data/raw/`; derived datasets live in `data/clean/`. Avoid overwriting raw inputs—write new outputs to a dated file under `data/clean/`.
- Training artifacts from CatBoost runs are in `catboost_info/` (logs, metrics). Treat as outputs; regenerate rather than editing.

## Environment, Build, and Run Commands
- Create an isolated environment (Windows example): `python -m venv .venv && .\.venv\Scripts\activate`.
- Install dependencies (if no `requirements.txt`, install what notebooks import: pandas, numpy, scikit-learn, catboost, matplotlib, seaborn, jupyter).
- Launch notebooks: `jupyter notebook` (or `jupyter lab`) from the repo root. Execute notebooks top-to-bottom to rebuild cleaned data and models.
- For non-interactive runs, you can execute a notebook to file: `jupyter nbconvert --to notebook --execute "3. Model Selection and Initialization.ipynb" --output executed.ipynb`.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, 100-character lines where practical, snake_case for functions/variables, UPPER_SNAKE for constants.
- Name new notebooks sequentially (`4. Feature Engineering.ipynb`) to preserve execution order; keep spaces consistent with existing files.
- When adding Python modules, prefer a `src/` directory and mirror notebook section names in module filenames (e.g., `feature_engineering.py`).
- Keep notebook outputs light: clear large cell outputs or images before committing; persist artifacts to `data/clean/` instead.

## Testing & Validation
- If you extract reusable code into modules, add unit tests with `pytest` under `tests/`. Aim for coverage on data transforms and model helpers.
- For data-heavy steps, use small fixtures or sampled CSVs (e.g., first 1,000 rows) to keep tests fast and reproducible.
- Document any randomness controls (e.g., `random_state`) in notebooks and tests so results can be replicated.

## Commit & Pull Request Guidelines
- Write concise, present-tense commit messages that describe the change scope (e.g., `Add catboost baseline`, `Clean NY accidents data`).
- Keep commits focused: one logical change per commit (data cleaning, feature addition, or modeling tweak).
- Pull requests should summarize goals, key notebook sections touched, data dependencies (input/output files), and how to reproduce results. Include metrics or validation plots when model performance changes.
- Do not commit raw data beyond what already exists; prefer linking to source and documenting retrieval steps.
