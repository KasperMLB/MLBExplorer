# MLB Matchup Dashboard

Two-tier MLB matchup tooling built around local Statcast CSV files:

- `local_app.py`: a full local Streamlit explorer backed by DuckDB and raw Statcast history.
- `hosted_app.py`: a hosted Streamlit companion that reads precomputed artifacts only.
- `mlb_dashboard/build.py`: builds the DuckDB analytical store and hosted artifacts.
- `mlb_dashboard/publish.py`: uploads artifacts to Hugging Face Hub.

## Install

```powershell
pip install -r requirements.txt
```

## Build the local database and hosted artifacts

```powershell
python -m mlb_dashboard.build `
  --csv-dir "C:\Users\Sheldon\Documents\MLB_Data" `
  --db-path "C:\Users\Sheldon\Documents\MLB_Data\artifacts\statcast.duckdb" `
  --artifacts-dir "C:\Users\Sheldon\Documents\MLB_Data\artifacts" `
  --target-date 2026-03-27
```

## Run the local explorer

```powershell
streamlit run local_app.py
```

## Run the hosted companion locally

```powershell
streamlit run hosted_app.py
```

## Publish hosted artifacts to Hugging Face Hub

Set `HF_TOKEN` and `HF_REPO_ID`, then run:

```powershell
python -m mlb_dashboard.publish `
  --artifacts-dir "C:\Users\Sheldon\Documents\MLB_Data\artifacts" `
  --repo-id your-username/your-dataset
```
