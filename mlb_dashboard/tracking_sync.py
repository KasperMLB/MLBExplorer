from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .build import BuildContext, prepare_historical_base, run_build
from .cockroach_loader import _ensure_driver, _normalize_database_url, read_source_freshness_report
from .config import AppConfig
from .ingest import ingest_date

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None


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


def _is_retryable_cockroach_error(exc: Exception) -> bool:
    if psycopg is not None and isinstance(exc, psycopg.errors.SerializationFailure):
        return True
    message = str(exc).lower()
    return "restart transaction" in message or "readwithinuncertaintyinterval" in message


def _run_with_retry(task_name: str, target_date: date, fn, attempts: int = 6):
    last_error: Exception | None = None
    for attempt in range(1, max(int(attempts), 1) + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if not _is_retryable_cockroach_error(exc) or attempt >= attempts:
                raise
            delay_seconds = float(attempt * 2)
            print(
                f"[retry] {task_name} {target_date.isoformat()} "
                f"attempt={attempt}/{attempts} waiting={delay_seconds:.1f}s reason={exc.__class__.__name__}"
            )
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


def _resolve_target_dates(days: int) -> list[date]:
    window = max(int(days), 1)
    return [date.today() - timedelta(days=offset) for offset in reversed(range(window))]


def _status_for_date(config: AppConfig, target_date: date) -> TrackingSyncStatus:
    _ensure_driver()
    if not config.database_url:
        raise RuntimeError("DATABASE_URL must be set to inspect tracking status.")
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    COALESCE((SELECT COUNT(*)::INT8 FROM {config.cockroach_hitter_snapshot_table} WHERE slate_date = %(target_date)s), 0) AS hitter_snapshot_rows,
                    COALESCE((SELECT COUNT(*)::INT8 FROM {config.cockroach_hitter_outcome_table} WHERE slate_date = %(target_date)s), 0) AS hitter_outcome_rows,
                    COALESCE((SELECT COUNT(*)::INT8 FROM {config.cockroach_hitter_outcome_table} WHERE slate_date = %(target_date)s AND outcome_complete = true), 0) AS hitter_graded_rows,
                    COALESCE((SELECT COUNT(*)::INT8 FROM {config.cockroach_pitcher_snapshot_table} WHERE slate_date = %(target_date)s), 0) AS pitcher_snapshot_rows,
                    COALESCE((SELECT COUNT(*)::INT8 FROM {config.cockroach_pitcher_outcome_table} WHERE slate_date = %(target_date)s), 0) AS pitcher_outcome_rows,
                    COALESCE((SELECT COUNT(*)::INT8 FROM {config.cockroach_pitcher_outcome_table} WHERE slate_date = %(target_date)s AND outcome_complete = true), 0) AS pitcher_graded_rows,
                    COALESCE((SELECT COUNT(*)::INT8 FROM {config.cockroach_props_odds_table} WHERE CAST(substr(commence_time, 1, 10) AS DATE) = %(target_date)s), 0) AS odds_rows
                """,
                {"target_date": target_date},
            )
            row = cur.fetchone()
    if row is None:
        return TrackingSyncStatus(target_date, 0, 0, 0, 0, 0, 0, 0)
    return TrackingSyncStatus(
        target_date=target_date,
        hitter_snapshot_rows=int(row[0] or 0),
        hitter_outcome_rows=int(row[1] or 0),
        hitter_graded_rows=int(row[2] or 0),
        pitcher_snapshot_rows=int(row[3] or 0),
        pitcher_outcome_rows=int(row[4] or 0),
        pitcher_graded_rows=int(row[5] or 0),
        odds_rows=int(row[6] or 0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-ingest and rebuild a rolling recent window so tracked rows become graded automatically.")
    parser.add_argument("--days", type=int, default=3, help="Number of recent dates to reprocess. Defaults to 3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    if not config.database_url:
        raise RuntimeError("DATABASE_URL must be set before running tracking_sync.")
    if not config.odds_api_key:
        print("Warning: ODDS_API_KEY is not set. Rebuilds will not refresh captured odds rows.")

    target_dates = _resolve_target_dates(args.days)
    for target_date in target_dates:
        ingest_summary = _run_with_retry("ingest", target_date, lambda: ingest_date(config, target_date))
        print(
            f"[ingest] {target_date.isoformat()} "
            f"games={ingest_summary.processed_games}/{ingest_summary.scheduled_games} "
            f"live_rows={ingest_summary.live_rows} baseline_rows={ingest_summary.baseline_rows}"
        )
    latest_target_date = max(target_dates)
    historical_base = _run_with_retry(
        "prepare",
        latest_target_date,
        lambda: prepare_historical_base(config, config.csv_dir),
    )
    for target_date in target_dates:
        _run_with_retry(
            "build",
            target_date,
            lambda target_date=target_date: run_build(
                BuildContext(config=config, target_date=target_date, csv_dir=config.csv_dir),
                historical_statcast_override=historical_base,
                refresh_reusable_artifacts=target_date == latest_target_date,
                capture_odds=target_date == latest_target_date,
            ),
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
