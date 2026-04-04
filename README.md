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

To stop a build when Cockroach event sources are stale, add:

```powershell
python -m mlb_dashboard.build `
  --csv-dir "C:\Users\Sheldon\Documents\MLB_Data" `
  --db-path "C:\Users\Sheldon\Documents\MLB_Data\artifacts\statcast.duckdb" `
  --artifacts-dir "C:\Users\Sheldon\Documents\MLB_Data\artifacts" `
  --target-date 2026-03-27 `
  --require-fresh-sources
```

## Check Cockroach source freshness

```powershell
python -m mlb_dashboard.health_check `
  --target-date 2026-04-02 `
  --lookback-days 7
```

Use `--strict` to return a non-zero exit code when either tracked Cockroach source is stale.

## Ingest 2026 live events into Cockroach

```powershell
python -m mlb_dashboard.ingest --date 2026-04-02
```

Backfill a date range:

```powershell
python -m mlb_dashboard.ingest --start-date 2026-03-27 --end-date 2026-04-02
```

Sync recent days and then build:

```powershell
python -m mlb_dashboard.ingest --sync-recent 3 --build-after
```

## Automatically re-grade recent tracked dates

Use the repo-owned rolling sync command to re-ingest and rebuild recent dates so
previously incomplete tracked rows can become graded once live event data is
available:

```powershell
python -m mlb_dashboard.tracking_sync --days 3
```

Required environment:

- `DATABASE_URL`
- `ODDS_API_KEY` if you want rebuilt dates to refresh captured odds rows too

Expected CLI output:

- per-date ingest counts
- per-date hitter/pitcher snapshot, outcome, and graded-row counts
- per-date odds row counts
- final source freshness summary

Recommended workflow:

1. Run the normal build during the day for today's slate.
2. Run `python -m mlb_dashboard.tracking_sync --days 3` on a rolling basis.
3. Verify Backtesting diagnostics show `Graded Rows` increasing after the sync.

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
