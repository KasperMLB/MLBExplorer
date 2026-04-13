from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import AppConfig
from .ingest import ingest_date


def _within_window(now: datetime, start_hour: int, end_hour: int) -> bool:
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= now.hour < end_hour
    return now.hour >= start_hour or now.hour < end_hour


def _sleep_until_next(interval_minutes: int) -> None:
    time.sleep(max(interval_minutes, 1) * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live ingest on a loop during game windows.")
    parser.add_argument("--date", type=lambda value: datetime.fromisoformat(value).date(), default=None)
    parser.add_argument("--interval-minutes", type=int, default=10)
    parser.add_argument("--start-hour", type=int, default=11, help="Local hour to begin looping (America/Chicago).")
    parser.add_argument("--end-hour", type=int, default=2, help="Local hour to stop looping (America/Chicago).")
    parser.add_argument("--timezone", type=str, default="America/Chicago")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tz = ZoneInfo(args.timezone)
    config = AppConfig()
    while True:
        now = datetime.now(tz)
        if _within_window(now, args.start_hour, args.end_hour):
            target_date = args.date or now.date()
            summary = ingest_date(config, target_date)
            print(
                f"[live-ingest] {target_date.isoformat()} "
                f"games={summary.processed_games}/{summary.scheduled_games} "
                f"live_rows={summary.live_rows} baseline_rows={summary.baseline_rows}"
            )
        else:
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            print(
                f"[live-ingest] Outside window ({args.start_hour}:00-{args.end_hour}:00). "
                f"Sleeping until {next_run.astimezone(tz).strftime('%Y-%m-%d %H:%M %Z')}."
            )
            time.sleep(max(int((next_run - now).total_seconds()), 60))
            continue
        _sleep_until_next(args.interval_minutes)


if __name__ == "__main__":
    main()
