from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from .build import BuildContext, run_build
from .cockroach_loader import read_source_freshness_report, replace_live_event_payload
from .config import AppConfig
from .metrics import add_metric_flags
from .mlb_api import fetch_game_feed, fetch_schedule


LIVE_COLUMNS = [
    "event_key",
    "pa_key",
    "game_pk",
    "game_date",
    "source_season",
    "batter",
    "pitcher",
    "player_name",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "pitch_name",
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "plate_x",
    "plate_z",
    "zone",
    "balls",
    "strikes",
    "outs_when_up",
    "bat_score",
    "fld_score",
    "post_bat_score",
    "post_fld_score",
    "type",
    "description",
    "events",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
    "spray_angle",
    "bb_type",
    "launch_speed_angle",
    "barrel",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "delta_home_win_exp",
    "delta_run_exp",
    "created_at",
]

BASELINE_COLUMNS = [
    "player_name",
    "event_key",
    "batter",
    "pitcher",
    "game_date",
    "game_pk",
    "source_season",
    "pitch_type",
    "pitch_name",
    "events",
    "description",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "plate_x",
    "plate_z",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "spray_angle",
    "hc_x",
    "hc_y",
    "bb_type",
    "balls",
    "strikes",
    "outs_when_up",
    "bat_score",
    "fld_score",
    "post_bat_score",
    "post_fld_score",
    "pitcher_hand",
    "batter_stand",
    "movement_magnitude",
    "spin_efficiency_proxy",
    "release_height_proxy",
    "release_side_proxy",
    "count_string",
    "baseline_mode",
    "prior_sample_size",
    "season_2026_sample_size",
    "prior_weight",
    "season_2026_weight",
    "baseline_driver",
    "rolling_overlay_active",
    "baseline_role",
    "baseline_source",
    "snapshot_built_at",
    "snapshot_version",
    "source_status",
]


@dataclass(frozen=True)
class IngestSummary:
    target_date: date
    scheduled_games: int
    processed_games: int
    live_rows: int
    baseline_rows: int


def _as_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _status_slug(value: object) -> str:
    text = str(value or "").strip().lower()
    return text.replace(" ", "_") if text else "unknown"


def _score_pair(is_top_inning: bool, away_score: int | None, home_score: int | None) -> tuple[int | None, int | None]:
    if is_top_inning:
        return away_score, home_score
    return home_score, away_score


def _safe_game_date(game_date: object) -> date:
    parsed = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse game date: {game_date!r}")
    return parsed.date()


def _prepare_live_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=LIVE_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["event_key"], keep="last").copy()
    frame = add_metric_flags(frame)
    if "is_barrel" in frame.columns:
        frame["barrel"] = frame["is_barrel"].fillna(False).astype(int)
    return frame.reindex(columns=LIVE_COLUMNS)


def _prepare_baseline_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=BASELINE_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["event_key"], keep="last").copy()
    return frame.reindex(columns=BASELINE_COLUMNS)


def _normalize_game_feed_to_rows(game: dict, feed: dict, target_date: date, captured_at: datetime) -> tuple[list[dict], list[dict]]:
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})
    plays = live_data.get("plays", {}).get("allPlays", []) or []
    status_slug = _status_slug(game_data.get("status", {}).get("detailedState"))
    teams = game_data.get("teams", {})
    home_team = str(teams.get("home", {}).get("abbreviation") or game.get("home_team") or "").strip()
    away_team = str(teams.get("away", {}).get("abbreviation") or game.get("away_team") or "").strip()
    live_rows: list[dict] = []
    baseline_rows: list[dict] = []
    away_score_before = 0
    home_score_before = 0
    snapshot_built_at = captured_at.isoformat()
    snapshot_version = "statsapi_v1"

    for play in plays:
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        result = play.get("result", {})
        is_top_inning = bool(about.get("isTopInning"))
        inning_topbot = "Top" if is_top_inning else "Bottom"
        at_bat_number = _as_int(about.get("atBatIndex"))
        at_bat_number = 1 if at_bat_number is None else at_bat_number + 1
        game_pk = _as_int(game.get("game_pk"))
        game_date = target_date
        source_season = target_date.year
        batter_id = _as_int(matchup.get("batter", {}).get("id"))
        pitcher_id = _as_int(matchup.get("pitcher", {}).get("id"))
        pitcher_name = str(matchup.get("pitcher", {}).get("fullName") or "").strip()
        stand = str(matchup.get("batSide", {}).get("code") or "").strip()
        p_throws = str(matchup.get("pitchHand", {}).get("code") or "").strip()
        post_away_score = _as_int(result.get("awayScore"))
        post_home_score = _as_int(result.get("homeScore"))
        bat_score, fld_score = _score_pair(is_top_inning, away_score_before, home_score_before)
        post_bat_score, post_fld_score = _score_pair(is_top_inning, post_away_score, post_home_score)

        pitch_events = [event for event in play.get("playEvents", []) if event.get("isPitch") and event.get("pitchData")]
        for pitch_index, event in enumerate(pitch_events):
            details = event.get("details", {})
            pitch_data = event.get("pitchData", {})
            coordinates = pitch_data.get("coordinates", {})
            breaks = pitch_data.get("breaks", {})
            hit_data = event.get("hitData", {}) or {}
            hit_coordinates = hit_data.get("coordinates", {}) or {}
            pitch_number = _as_int(event.get("pitchNumber"))
            if pitch_number is None:
                continue
            is_terminal_pitch = pitch_index == len(pitch_events) - 1
            event_key = f"{game_pk}_{at_bat_number}_{pitch_number}"
            pa_key = f"{game_pk}_{at_bat_number}"
            description = str(details.get("description") or "").strip() or None
            live_row = {
                "event_key": event_key,
                "pa_key": pa_key,
                "game_pk": game_pk,
                "game_date": game_date.isoformat(),
                "source_season": source_season,
                "batter": batter_id,
                "pitcher": pitcher_id,
                "player_name": pitcher_name or None,
                "stand": stand or None,
                "p_throws": p_throws or None,
                "home_team": home_team or None,
                "away_team": away_team or None,
                "inning": _as_int(about.get("inning")),
                "inning_topbot": inning_topbot,
                "at_bat_number": at_bat_number,
                "pitch_number": pitch_number,
                "pitch_type": str(details.get("type", {}).get("code") or "").strip() or None,
                "pitch_name": str(details.get("type", {}).get("description") or "").strip() or None,
                "release_speed": _as_float(pitch_data.get("startSpeed")),
                "effective_speed": _as_float(pitch_data.get("endSpeed")),
                "release_spin_rate": _as_float(breaks.get("spinRate")),
                "spin_axis": _as_float(breaks.get("spinDirection")),
                "pfx_x": _as_float(coordinates.get("pfxX")),
                "pfx_z": _as_float(coordinates.get("pfxZ")),
                "release_pos_x": _as_float(coordinates.get("x0")),
                "release_pos_y": _as_float(coordinates.get("y0")),
                "release_pos_z": _as_float(coordinates.get("z0")),
                "release_extension": _as_float(pitch_data.get("extension")),
                "plate_x": _as_float(coordinates.get("pX")),
                "plate_z": _as_float(coordinates.get("pZ")),
                "zone": _as_int(pitch_data.get("zone")),
                "balls": _as_int(event.get("count", {}).get("balls")),
                "strikes": _as_int(event.get("count", {}).get("strikes")),
                "outs_when_up": _as_int(event.get("count", {}).get("outs")),
                "bat_score": bat_score,
                "fld_score": fld_score,
                "post_bat_score": post_bat_score,
                "post_fld_score": post_fld_score,
                "type": str(details.get("call", {}).get("code") or details.get("code") or "").strip() or None,
                "description": description,
                "events": str(result.get("event") or "").strip() or None if is_terminal_pitch else None,
                "launch_speed": _as_float(hit_data.get("launchSpeed")),
                "launch_angle": _as_float(hit_data.get("launchAngle")),
                "hit_distance_sc": _as_float(hit_data.get("totalDistance")),
                "hc_x": _as_float(hit_coordinates.get("coordX")),
                "hc_y": _as_float(hit_coordinates.get("coordY")),
                "spray_angle": None,
                "bb_type": str(hit_data.get("trajectory") or "").strip().lower() or None,
                "launch_speed_angle": None,
                "barrel": None,
                "estimated_ba_using_speedangle": None,
                "estimated_woba_using_speedangle": None,
                "woba_value": None,
                "woba_denom": None,
                "delta_home_win_exp": None,
                "delta_run_exp": None,
                "created_at": captured_at.replace(tzinfo=None),
            }
            live_rows.append(live_row)

            movement_magnitude = None
            if live_row["pfx_x"] is not None and live_row["pfx_z"] is not None:
                movement_magnitude = math.sqrt((live_row["pfx_x"] ** 2) + (live_row["pfx_z"] ** 2))
            baseline_rows.append(
                {
                    "player_name": pitcher_name or None,
                    "event_key": event_key,
                    "batter": batter_id,
                    "pitcher": pitcher_id,
                    "game_date": game_date.isoformat(),
                    "game_pk": game_pk,
                    "source_season": source_season,
                    "pitch_type": live_row["pitch_type"],
                    "pitch_name": live_row["pitch_name"],
                    "events": live_row["events"],
                    "description": description,
                    "stand": stand or None,
                    "p_throws": p_throws or None,
                    "home_team": home_team or None,
                    "away_team": away_team or None,
                    "inning": live_row["inning"],
                    "inning_topbot": inning_topbot,
                    "at_bat_number": at_bat_number,
                    "pitch_number": pitch_number,
                    "plate_x": live_row["plate_x"],
                    "plate_z": live_row["plate_z"],
                    "release_speed": live_row["release_speed"],
                    "release_spin_rate": live_row["release_spin_rate"],
                    "release_extension": live_row["release_extension"],
                    "release_pos_x": live_row["release_pos_x"],
                    "release_pos_z": live_row["release_pos_z"],
                    "pfx_x": live_row["pfx_x"],
                    "pfx_z": live_row["pfx_z"],
                    "launch_speed": live_row["launch_speed"],
                    "launch_angle": live_row["launch_angle"],
                    "estimated_woba_using_speedangle": None,
                    "spray_angle": None,
                    "hc_x": live_row["hc_x"],
                    "hc_y": live_row["hc_y"],
                    "bb_type": live_row["bb_type"],
                    "balls": live_row["balls"],
                    "strikes": live_row["strikes"],
                    "outs_when_up": live_row["outs_when_up"],
                    "bat_score": bat_score,
                    "fld_score": fld_score,
                    "post_bat_score": post_bat_score,
                    "post_fld_score": post_fld_score,
                    "pitcher_hand": p_throws or None,
                    "batter_stand": stand or None,
                    "movement_magnitude": movement_magnitude,
                    "spin_efficiency_proxy": None,
                    "release_height_proxy": live_row["release_pos_z"],
                    "release_side_proxy": live_row["release_pos_x"],
                    "count_string": f"{live_row['balls']}-{live_row['strikes']}" if live_row["balls"] is not None and live_row["strikes"] is not None else None,
                    "baseline_mode": "game_feed",
                    "prior_sample_size": 0,
                    "season_2026_sample_size": 1,
                    "prior_weight": 0.0,
                    "season_2026_weight": 1.0,
                    "baseline_driver": "statsapi",
                    "rolling_overlay_active": False,
                    "baseline_role": "pitcher",
                    "baseline_source": "season_2026",
                    "snapshot_built_at": snapshot_built_at,
                    "snapshot_version": snapshot_version,
                    "source_status": status_slug,
                }
            )

        if post_away_score is not None:
            away_score_before = post_away_score
        if post_home_score is not None:
            home_score_before = post_home_score

    return live_rows, baseline_rows


def ingest_date(config: AppConfig, target_date: date) -> IngestSummary:
    schedule = fetch_schedule(target_date)
    captured_at = datetime.now(UTC)
    live_rows: list[dict] = []
    baseline_rows: list[dict] = []
    processed_games = 0
    errors: list[str] = []
    for game in schedule:
        game_pk = game.get("game_pk")
        try:
            feed = fetch_game_feed(int(game_pk))
            game_live_rows, game_baseline_rows = _normalize_game_feed_to_rows(game, feed, target_date, captured_at)
            live_rows.extend(game_live_rows)
            baseline_rows.extend(game_baseline_rows)
            processed_games += 1
        except Exception as exc:  # pragma: no cover
            errors.append(f"{game_pk}: {exc}")
    if errors:
        raise RuntimeError(
            "Failed to ingest one or more games for "
            f"{target_date.isoformat()}:\n" + "\n".join(errors)
        )
    live_frame = _prepare_live_frame(live_rows)
    baseline_frame = _prepare_baseline_frame(baseline_rows)
    replace_live_event_payload(
        config,
        live_frame,
        baseline_frame,
        start_date=target_date,
        end_date=target_date,
    )
    return IngestSummary(
        target_date=target_date,
        scheduled_games=len(schedule),
        processed_games=processed_games,
        live_rows=len(live_frame),
        baseline_rows=len(baseline_frame),
    )


def _resolve_target_dates(args: argparse.Namespace) -> list[date]:
    if args.date:
        return [args.date]
    if args.start_date and args.end_date:
        days = (args.end_date - args.start_date).days
        if days < 0:
            raise ValueError("--end-date must be on or after --start-date.")
        return [args.start_date + timedelta(days=offset) for offset in range(days + 1)]
    if args.sync_recent:
        return [date.today() - timedelta(days=offset) for offset in reversed(range(max(args.sync_recent, 1)))]
    raise ValueError("Specify --date, --start-date/--end-date, or --sync-recent.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest MLB StatsAPI game-feed events into the 2026 Cockroach live tables.")
    parser.add_argument("--date", type=lambda value: date.fromisoformat(value))
    parser.add_argument("--start-date", type=lambda value: date.fromisoformat(value))
    parser.add_argument("--end-date", type=lambda value: date.fromisoformat(value))
    parser.add_argument("--sync-recent", type=int)
    parser.add_argument("--build-after", action="store_true", help="Run downstream build(s) after ingest completes.")
    parser.add_argument(
        "--build-target-date",
        action="append",
        type=lambda value: date.fromisoformat(value),
        default=[],
        help="Specific target date(s) to build after ingest. Defaults to the ingested dates when --build-after is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dates = _resolve_target_dates(args)
    config = AppConfig()
    summaries: list[IngestSummary] = []
    for target_date in target_dates:
        summary = ingest_date(config, target_date)
        summaries.append(summary)
        print(
            f"Ingested {summary.target_date.isoformat()}: "
            f"{summary.processed_games}/{summary.scheduled_games} games, "
            f"{summary.live_rows} live rows, {summary.baseline_rows} baseline rows."
        )

    report = read_source_freshness_report(config, target_date=max(target_dates), lookback_days=max(len(target_dates), 7))
    if report:
        for summary in report:
            print(
                f"[{summary['source_name']}] max_event_date={summary.get('max_event_date') or 'none'} "
                f"lag_days={summary.get('lag_days') if summary.get('lag_days') is not None else 'unknown'} "
                f"missing_dates={', '.join(summary.get('missing_dates') or []) or 'none'}"
            )

    if args.build_after:
        build_dates = args.build_target_date or target_dates
        for build_date in sorted(set(build_dates)):
            print(f"Running downstream build for {build_date.isoformat()}...")
            run_build(BuildContext(config=config, target_date=build_date, csv_dir=config.csv_dir))


if __name__ == "__main__":
    main()
