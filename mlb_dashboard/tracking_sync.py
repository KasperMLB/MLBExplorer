from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .build import BuildContext, prepare_historical_base, run_build
from .config import AppConfig
from .ingest import ingest_date
from .local_store import read_source_freshness_report


@dataclass(frozen=True)
class TrackingSyncStatus:
    target_date: date
    hitter_snapshot_rows: int
    hitter_outcome_rows: int
    hitter_graded_rows: int
    pitcher_snapshot_rows: int
    pitcher_outcome_rows: int
    pitcher_graded_rows: int
    odds_rows: int


def _resolve_target_dates(days: int) -> list[date]:
    window = max(int(days), 1)
    return [date.today() - timedelta(days=offset) for offset in reversed(range(window))]


def _status_for_date(config: AppConfig, target_date: date) -> TrackingSyncStatus:
    from .local_store import read_hitter_backtest_data, read_pitcher_backtest_data, read_prop_odds_history

    hitter_snapshots, hitter_outcomes, _ = read_hitter_backtest_data(config, target_date, target_date)
    pitcher_snapshots, pitcher_outcomes, _ = read_pitcher_backtest_data(config, target_date, target_date)
    odds = read_prop_odds_history(config, target_date, target_date, tuple())
    return TrackingSyncStatus(
        target_date=target_date,
        hitter_snapshot_rows=len(hitter_snapshots),
        hitter_outcome_rows=len(hitter_outcomes),
        hitter_graded_rows=int(hitter_outcomes.get("outcome_complete", pd.Series(dtype=bool)).fillna(False).sum()) if not hitter_outcomes.empty else 0,
        pitcher_snapshot_rows=len(pitcher_snapshots),
        pitcher_outcome_rows=len(pitcher_outcomes),
        pitcher_graded_rows=int(pitcher_outcomes.get("outcome_complete", pd.Series(dtype=bool)).fillna(False).sum()) if not pitcher_outcomes.empty else 0,
        odds_rows=len(odds),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-ingest and rebuild a rolling recent window so tracked rows become graded automatically.")
    parser.add_argument("--days", type=int, default=3, help="Number of recent dates to reprocess. Defaults to 3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    if not config.odds_api_key:
        print("Warning: ODDS_API_KEY is not set. Rebuilds will not refresh captured odds rows.")

    target_dates = _resolve_target_dates(args.days)
    for target_date in target_dates:
        ingest_summary = ingest_date(config, target_date)
        print(
            f"[ingest] {target_date.isoformat()} "
            f"games={ingest_summary.processed_games}/{ingest_summary.scheduled_games} "
            f"live_rows={ingest_summary.live_rows} baseline_rows={ingest_summary.baseline_rows}"
        )
    latest_target_date = max(target_dates)
    historical_base = prepare_historical_base(config, config.csv_dir)
    for target_date in target_dates:
        run_build(
            BuildContext(config=config, target_date=target_date, csv_dir=config.csv_dir),
            historical_statcast_override=historical_base,
            refresh_reusable_artifacts=target_date == latest_target_date,
            capture_odds=target_date == latest_target_date,
        )
        status = _status_for_date(config, target_date)
        print(
            f"[grade] {target_date.isoformat()} "
            f"hitter_snapshots={status.hitter_snapshot_rows} hitter_outcomes={status.hitter_outcome_rows} hitter_graded={status.hitter_graded_rows} "
            f"pitcher_snapshots={status.pitcher_snapshot_rows} pitcher_outcomes={status.pitcher_outcome_rows} pitcher_graded={status.pitcher_graded_rows} "
            f"odds_rows={status.odds_rows}"
        )

    freshness = read_source_freshness_report(config, target_date=max(target_dates), lookback_days=max(len(target_dates), 3))
    for summary in freshness:
        print(
            f"[source] {summary['source_name']} max_event_date={summary.get('max_event_date') or 'none'} "
            f"lag_days={summary.get('lag_days') if summary.get('lag_days') is not None else 'unknown'} "
            f"missing_dates={', '.join(summary.get('missing_dates') or []) or 'none'}"
        )


if __name__ == "__main__":
    main()
