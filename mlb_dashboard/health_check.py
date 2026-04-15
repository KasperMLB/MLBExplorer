from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .config import AppConfig
from .local_store import read_source_freshness_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report local Statcast event-source freshness for recent MLB builds.")
    parser.add_argument("--target-date", type=lambda value: date.fromisoformat(value), default=date.today())
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any tracked source is stale.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    return parser.parse_args()


def _format_summary(summary: dict[str, object]) -> str:
    recent_counts = summary.get("recent_daily_counts") or []
    counts_text = ", ".join(
        f"{row['event_date']}: {row['row_count']}"
        for row in recent_counts
        if isinstance(row, dict) and "event_date" in row and "row_count" in row
    )
    if not counts_text:
        counts_text = "none"
    missing_dates = summary.get("missing_dates") or []
    missing_text = ", ".join(str(value) for value in missing_dates) if missing_dates else "none"
    return "\n".join(
        [
            f"[{summary['source_name']}] {summary['table_name']}",
            f"  target date: {summary['target_date']}",
            f"  max event date: {summary['max_event_date'] or 'none'}",
            f"  lag days: {summary['lag_days'] if summary['lag_days'] is not None else 'unknown'}",
            f"  total rows: {summary['row_count']}",
            f"  recent daily counts: {counts_text}",
            f"  missing dates: {missing_text}",
            f"  status: {'fresh' if summary.get('is_fresh') else 'stale'}",
        ]
    )


def main() -> None:
    args = parse_args()
    config = AppConfig()
    report = read_source_freshness_report(
        config,
        target_date=args.target_date,
        lookback_days=max(args.lookback_days, 1),
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        if not report:
            print("No local Statcast freshness report was produced.", file=sys.stderr)
        for index, summary in enumerate(report):
            if index:
                print()
            print(_format_summary(summary))
    if args.strict and any(not bool(summary.get("is_fresh")) for summary in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
