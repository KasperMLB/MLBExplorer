from __future__ import annotations

import argparse
import json
import uuid
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from .cockroach_loader import CockroachPayload, load_cockroach_payload, write_props_odds_snapshot, write_tracking_payload
from .config import AppConfig, DEFAULT_PITCH_GROUPS, DEFAULT_RECENT_WINDOWS, DEFAULT_SPLITS, ensure_directories
from .dashboard_views import add_hitter_matchup_score, add_pitcher_rank_score, apply_projected_lineup, build_pitcher_matchup_key
from .mlb_api import fetch_schedule, fetch_team_rosters_for_schedule
from .metrics import add_metric_flags, apply_year_weights, likely_starter_scores
from .odds_service import PropsBoardPayload, load_live_props_board
from .rotowire_lineups import fetch_rotowire_lineups, resolve_rotowire_lineups

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None


REQUIRED_COLUMNS = [
    "game_date",
    "game_year",
    "game_pk",
    "home_team",
    "away_team",
    "inning_topbot",
    "pitcher",
    "player_name",
    "pitch_name",
    "pitch_type",
    "p_throws",
    "stand",
    "batter",
    "estimated_woba_using_speedangle",
    "bb_type",
    "description",
    "events",
    "launch_angle",
    "launch_speed",
    "release_speed",
    "release_spin_rate",
    "at_bat_number",
    "balls",
    "strikes",
    "hc_x",
    "zone",
    "bat_score",
    "post_bat_score",
]


@dataclass
class BuildContext:
    config: AppConfig
    target_date: date
    csv_dir: Path


OUTCOME_STATUS_COMPLETE = "complete"
OUTCOME_STATUS_SOURCE_LAG = "source_lag"
OUTCOME_STATUS_SOURCE_EMPTY = "source_empty"
OUTCOME_STATUS_SOURCE_MISSING_ROWS = "source_missing_rows"


def _csv_glob(csv_dir: Path) -> list[str]:
    return [str(path) for path in sorted(csv_dir.glob("statcast_*.csv")) if path.is_file()]


def _load_raw_statcast(csv_paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        frame = pd.read_csv(path, usecols=lambda column: column in REQUIRED_COLUMNS, low_memory=False)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No statcast_*.csv files were found.")
    combined = pd.concat(frames, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined["team"] = combined.apply(lambda row: row["away_team"] if row["inning_topbot"] == "Top" else row["home_team"], axis=1)
    combined["fielding_team"] = combined.apply(lambda row: row["home_team"] if row["inning_topbot"] == "Top" else row["away_team"], axis=1)
    combined["pitcher_name"] = combined["player_name"]
    return add_metric_flags(combined)


def _merge_historical_and_live(history: pd.DataFrame, live_payload: CockroachPayload) -> pd.DataFrame:
    if live_payload.live_pitch_mix.empty:
        return history
    historical = history.loc[pd.to_numeric(history["game_year"], errors="coerce").fillna(0) != 2026].copy()
    combined = pd.concat([historical, live_payload.live_pitch_mix], ignore_index=True, sort=False)
    return combined


def _live_source_max_event_date(live_payload: CockroachPayload) -> date | None:
    if live_payload.live_pitch_mix.empty or "game_date" not in live_payload.live_pitch_mix.columns:
        return None
    max_date = pd.to_datetime(live_payload.live_pitch_mix["game_date"], errors="coerce").max()
    return None if pd.isna(max_date) else max_date.date()


def _build_live_tracking_health(
    target_date: date,
    schedule: list[dict],
    live_payload: CockroachPayload,
    source_table: str,
) -> pd.DataFrame:
    columns = [
        "audit_date",
        "live_rows",
        "live_games",
        "expected_games",
        "has_live_data",
        "is_target_date",
        "source_table",
        "source_max_event_date",
        "lag_days",
    ]
    grouped = pd.DataFrame(columns=["audit_date", "live_rows", "live_games"])
    if not live_payload.live_pitch_mix.empty and "game_date" in live_payload.live_pitch_mix.columns:
        grouped = (
            live_payload.live_pitch_mix.assign(
                audit_date=pd.to_datetime(live_payload.live_pitch_mix["game_date"], errors="coerce").dt.date
            )
            .dropna(subset=["audit_date"])
            .groupby("audit_date", as_index=False)
            .agg(
                live_rows=("game_pk", "size"),
                live_games=("game_pk", "nunique"),
            )
        )
    source_max_event_date = _live_source_max_event_date(live_payload)
    lag_days = None if source_max_event_date is None else (target_date - source_max_event_date).days
    start_date = target_date - pd.Timedelta(days=6)
    rows: list[dict] = []
    grouped_map = grouped.set_index("audit_date").to_dict("index") if not grouped.empty else {}
    for day in pd.date_range(start=start_date, end=target_date, freq="D"):
        audit_date = day.date()
        observed = grouped_map.get(audit_date, {})
        rows.append(
            {
                "audit_date": audit_date,
                "live_rows": int(observed.get("live_rows", 0)),
                "live_games": int(observed.get("live_games", 0)),
                "expected_games": int(len(schedule)) if audit_date == target_date else pd.NA,
                "has_live_data": bool(observed.get("live_rows", 0) > 0),
                "is_target_date": audit_date == target_date,
                "source_table": source_table,
                "source_max_event_date": source_max_event_date,
                "lag_days": lag_days,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _empty_hitter_outcome_frame(snapshot_players: pd.DataFrame, target_date: date, status: str, source_max_event_date: date | None) -> pd.DataFrame:
    empty = snapshot_players.copy()
    empty["had_plate_appearance"] = pd.NA
    empty["started"] = pd.NA
    empty["plate_appearances"] = pd.NA
    empty["hits"] = pd.NA
    empty["home_runs"] = pd.NA
    empty["total_bases"] = pd.NA
    empty["runs"] = pd.NA
    empty["rbi"] = pd.NA
    empty["walks"] = pd.NA
    empty["strikeouts"] = pd.NA
    empty["outcome_complete"] = False
    empty["outcome_status"] = status
    empty["source_max_event_date"] = source_max_event_date
    empty["last_updated_at"] = datetime.now(UTC)
    return empty


def _empty_pitcher_outcome_frame(snapshot_pitchers: pd.DataFrame, target_date: date, status: str, source_max_event_date: date | None) -> pd.DataFrame:
    empty = snapshot_pitchers.copy()
    empty["had_pitch"] = pd.NA
    empty["started"] = pd.NA
    empty["outs_recorded"] = pd.NA
    empty["batters_faced"] = pd.NA
    empty["hits_allowed"] = pd.NA
    empty["home_runs_allowed"] = pd.NA
    empty["runs_allowed"] = pd.NA
    empty["earned_runs"] = pd.NA
    empty["walks"] = pd.NA
    empty["strikeouts"] = pd.NA
    empty["outcome_complete"] = False
    empty["outcome_status"] = status
    empty["source_max_event_date"] = source_max_event_date
    empty["last_updated_at"] = datetime.now(UTC)
    return empty


def _window_cutoff(max_date: pd.Timestamp, window: str) -> pd.Timestamp:
    if window == "last_14_days":
        return max_date - pd.Timedelta(days=14)
    if window == "last_45_days":
        return max_date - pd.Timedelta(days=45)
    return pd.Timestamp("1900-01-01")


def _split_mask(frame: pd.DataFrame, split_key: str) -> pd.Series:
    if split_key == "vs_rhp":
        return frame["p_throws"].fillna("") == "R"
    if split_key == "vs_lhp":
        return frame["p_throws"].fillna("") == "L"
    if split_key == "home":
        return frame["team"] == frame["home_team"]
    if split_key == "away":
        return frame["team"] == frame["away_team"]
    return pd.Series(True, index=frame.index)


def _weighted_sum(work: pd.DataFrame, source_column: str, value_index: pd.Index) -> float:
    series = pd.to_numeric(work.loc[value_index, source_column], errors="coerce")
    weights = work.loc[value_index, "metric_weight"]
    return float((series.fillna(0) * weights).sum())


def _weighted_denominator(work: pd.DataFrame, source_column: str, value_index: pd.Index) -> float:
    series = pd.to_numeric(work.loc[value_index, source_column], errors="coerce")
    return float(work.loc[value_index, "metric_weight"][series.notna()].sum())


def _aggregate_hitter_metrics(frame: pd.DataFrame, weighted_mode: str, year_weights: dict[int, float]) -> pd.DataFrame:
    work = frame.copy()
    work["batter"] = pd.to_numeric(work["batter"], errors="coerce")
    work = work.loc[work["batter"].notna()].copy()
    work["batter"] = work["batter"].astype(int)
    work["metric_weight"] = apply_year_weights(work, year_weights) if weighted_mode == "weighted" else 1.0
    rows: list[dict] = []
    for batter, group in work.groupby("batter", sort=False):
        latest_team = None
        if "game_date" in group.columns and "team" in group.columns:
            latest_rows = group.sort_values(["game_date"], ascending=[False], na_position="last")
            latest_team = latest_rows["team"].dropna().astype(str).iloc[0] if latest_rows["team"].notna().any() else None
        elif "team" in group.columns and group["team"].notna().any():
            latest_team = group["team"].dropna().astype(str).value_counts().idxmax()
        hitter_side = group["stand"].dropna().astype(str).value_counts().idxmax() if group["stand"].notna().any() else None
        pitch_count = len(group)
        bip = int(group["is_batted_ball"].sum())
        pitch_weight_sum = float(group["metric_weight"].sum())
        bip_weight_sum = float(group.loc[group["is_batted_ball"], "metric_weight"].sum())
        tracked_bbe_weight_sum = float(group.loc[group["is_tracked_bbe"], "metric_weight"].sum())
        barrel_weight_sum = float((group["is_barrel"].astype(int) * group["metric_weight"]).sum())
        pulled_barrel_weight_sum = float((group["is_pulled_barrel"].astype(int) * group["metric_weight"]).sum())
        sweet_spot_weight_sum = float((((group["is_sweet_spot"] & group["is_tracked_bbe"]).astype(int)) * group["metric_weight"]).sum())
        rows.append(
            {
                "team": latest_team,
                "batter": batter,
                "hitter_name": str(batter),
                "stand": hitter_side,
                "pitch_count": pitch_count,
                "bip": bip,
                "xwoba": _weighted_sum(group, "xwoba_value", group.index) / max(_weighted_denominator(group, "xwoba_value", group.index), 1e-9),
                "xwoba_con": _weighted_sum(group.loc[group["is_batted_ball"]], "xwoba_value", group.loc[group["is_batted_ball"]].index) / max(float(group.loc[group["is_batted_ball"], "metric_weight"].sum()), 1e-9),
                "swstr_pct": float((group["is_swinging_strike"].astype(int) * group["metric_weight"]).sum()) / max(pitch_weight_sum, 1e-9),
                "barrel_bbe_pct": barrel_weight_sum / max(tracked_bbe_weight_sum, 1e-9),
                "barrel_bip_pct": barrel_weight_sum / max(bip_weight_sum, 1e-9),
                "pulled_barrel_pct": pulled_barrel_weight_sum / max(tracked_bbe_weight_sum, 1e-9),
                "sweet_spot_pct": sweet_spot_weight_sum / max(tracked_bbe_weight_sum, 1e-9),
                "fb_pct": float((group["is_fly_ball"].astype(int) * group["metric_weight"]).sum()) / max(bip_weight_sum, 1e-9),
                "hard_hit_pct": float((group["is_hard_hit"].astype(int) * group["metric_weight"]).sum()) / max(bip_weight_sum, 1e-9),
                "avg_launch_angle": _weighted_sum(group, "launch_angle_value", group.index) / max(_weighted_denominator(group, "launch_angle_value", group.index), 1e-9),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_zone_profiles(frame: pd.DataFrame, zone_year_weights: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["zone"] = pd.to_numeric(work["zone"], errors="coerce")
    work = work.loc[work["zone"].notna() & work["pitch_type"].notna()].copy()
    if work.empty:
        batter_columns = ["batter_id", "pitcher_hand_key", "pitch_type", "zone", "sample_size", "hit_rate", "hr_rate"]
        pitcher_columns = ["pitcher_id", "batter_side_key", "pitch_type", "zone", "sample_size", "usage_rate"]
        return pd.DataFrame(columns=batter_columns), pd.DataFrame(columns=pitcher_columns)

    work["zone_metric_weight"] = apply_year_weights(work, zone_year_weights)
    work["pitcher_hand_key"] = work["p_throws"].fillna("").map({"R": "vs_rhp", "L": "vs_lhp"}).fillna("overall")
    work["batter_side_key"] = work["stand"].fillna("").map({"L": "vs_lhh", "R": "vs_rhh"}).fillna("overall")
    work["zone"] = work["zone"].astype(int)

    batter_rows: list[pd.DataFrame] = []
    for pitcher_hand_key, hand_frame in {
        "overall": work,
        "vs_rhp": work.loc[work["p_throws"].fillna("") == "R"],
        "vs_lhp": work.loc[work["p_throws"].fillna("") == "L"],
    }.items():
        if hand_frame.empty:
            continue
        grouped = (
            hand_frame.assign(
                hit_weight=hand_frame["is_hit_event"].astype(int) * hand_frame["zone_metric_weight"],
                hr_weight=hand_frame["is_home_run_event"].astype(int) * hand_frame["zone_metric_weight"],
            )
            .groupby(["batter", "pitch_type", "zone"], as_index=False)
            .agg(
                sample_size=("zone_metric_weight", "sum"),
                hit_weight=("hit_weight", "sum"),
                hr_weight=("hr_weight", "sum"),
            )
        )
        grouped["batter_id"] = grouped["batter"]
        grouped["pitcher_hand_key"] = pitcher_hand_key
        grouped["hit_rate"] = grouped["hit_weight"] / grouped["sample_size"].where(grouped["sample_size"] != 0)
        grouped["hr_rate"] = grouped["hr_weight"] / grouped["sample_size"].where(grouped["sample_size"] != 0)
        batter_rows.append(grouped[["batter_id", "pitcher_hand_key", "pitch_type", "zone", "sample_size", "hit_rate", "hr_rate"]])

    pitcher_rows: list[pd.DataFrame] = []
    for batter_side_key, side_frame in {
        "overall": work,
        "vs_lhh": work.loc[work["stand"].fillna("") == "L"],
        "vs_rhh": work.loc[work["stand"].fillna("") == "R"],
    }.items():
        if side_frame.empty:
            continue
        grouped = side_frame.groupby(["pitcher", "pitch_type", "zone"], as_index=False).agg(sample_size=("zone_metric_weight", "sum"))
        grouped["pitcher_id"] = grouped["pitcher"]
        grouped["batter_side_key"] = batter_side_key
        totals = grouped.groupby("pitcher_id")["sample_size"].transform("sum")
        grouped["usage_rate"] = grouped["sample_size"] / totals.where(totals != 0)
        pitcher_rows.append(grouped[["pitcher_id", "batter_side_key", "pitch_type", "zone", "sample_size", "usage_rate"]])

    batter_profiles = pd.concat(batter_rows, ignore_index=True) if batter_rows else pd.DataFrame(columns=["batter_id", "pitcher_hand_key", "pitch_type", "zone", "sample_size", "hit_rate", "hr_rate"])
    pitcher_profiles = pd.concat(pitcher_rows, ignore_index=True) if pitcher_rows else pd.DataFrame(columns=["pitcher_id", "batter_side_key", "pitch_type", "zone", "sample_size", "usage_rate"])
    return batter_profiles, pitcher_profiles


def _aggregate_pitcher_metrics(frame: pd.DataFrame, weighted_mode: str, year_weights: dict[int, float]) -> pd.DataFrame:
    work = frame.copy()
    work["metric_weight"] = apply_year_weights(work, year_weights) if weighted_mode == "weighted" else 1.0
    rows: list[dict] = []
    for (pitcher_id, throws), group in work.groupby(["pitcher", "p_throws"], sort=False):
        pitcher_name = group["pitcher_name"].dropna().astype(str).value_counts().idxmax() if group["pitcher_name"].notna().any() else str(pitcher_id)
        pitch_count = len(group)
        bip = int(group["is_batted_ball"].sum())
        pitch_weight_sum = float(group["metric_weight"].sum())
        bip_weight_sum = float(group.loc[group["is_batted_ball"], "metric_weight"].sum())
        tracked_bbe_weight_sum = float(group.loc[group["is_tracked_bbe"], "metric_weight"].sum())
        barrel_weight_sum = float((group["is_barrel"].astype(int) * group["metric_weight"]).sum())
        pulled_barrel_weight_sum = float((group["is_pulled_barrel"].astype(int) * group["metric_weight"]).sum())
        ground_ball_weight_sum = float((group["is_ground_ball"].astype(int) * group["metric_weight"]).sum())
        fly_ball_weight_sum = float((group["is_fly_ball"].astype(int) * group["metric_weight"]).sum())
        ball_weight_sum = float((group["is_ball"].astype(int) * group["metric_weight"]).sum())
        called_strike_weight_sum = float((group["is_called_strike"].astype(int) * group["metric_weight"]).sum())
        csw_weight_sum = called_strike_weight_sum + float((group["is_swinging_strike"].astype(int) * group["metric_weight"]).sum())

        pa_events = (
            group.groupby(["game_pk", "at_bat_number"], as_index=False, sort=False)
            .tail(1)
            .copy()
        )
        two_strike_finish_chance = (
            group.assign(two_strike_flag=pd.to_numeric(group["strikes"], errors="coerce").fillna(0).ge(2).astype(int))
            .groupby(["game_pk", "at_bat_number"], as_index=False, sort=False)["two_strike_flag"]
            .max()
            .rename(columns={"two_strike_flag": "had_two_strike_finish_chance"})
        )
        pa_events = pa_events.merge(two_strike_finish_chance, on=["game_pk", "at_bat_number"], how="left")
        pa_events["had_two_strike_finish_chance"] = pd.to_numeric(pa_events["had_two_strike_finish_chance"], errors="coerce").fillna(0).astype(int)
        pa_weight_sum = float(pa_events["metric_weight"].sum())
        event_text = pa_events["events"].fillna("").astype(str).str.lower()
        bb_type_text = pa_events["bb_type"].fillna("").astype(str).str.lower()
        strikeout_weight_sum = float((event_text.isin({"strikeout", "strikeout_double_play"}).astype(int) * pa_events["metric_weight"]).sum())
        putaway_chance_weight_sum = float((pa_events["had_two_strike_finish_chance"].astype(int) * pa_events["metric_weight"]).sum())
        putaway_strikeout_weight_sum = float(
            (
                event_text.isin({"strikeout", "strikeout_double_play"}).astype(int)
                * pa_events["had_two_strike_finish_chance"].astype(int)
                * pa_events["metric_weight"]
            ).sum()
        )
        walk_weight_sum = float((event_text.isin({"walk", "intent_walk", "intentional_walk"}).astype(int) * pa_events["metric_weight"]).sum())
        gb_minus_air_weight_sum = float(
            (
                (
                    bb_type_text.eq("ground_ball").astype(int)
                    - bb_type_text.isin({"fly_ball", "popup"}).astype(int)
                )
                * pa_events["metric_weight"]
            ).sum()
        )
        strikeout_rate = strikeout_weight_sum / max(pa_weight_sum, 1e-9)
        walk_rate = walk_weight_sum / max(pa_weight_sum, 1e-9)
        gb_minus_air_rate = gb_minus_air_weight_sum / max(pa_weight_sum, 1e-9)
        siera = (
            6.145
            - 16.986 * strikeout_rate
            + 11.434 * walk_rate
            - 1.858 * gb_minus_air_rate
            + 7.653 * (strikeout_rate ** 2)
            - 6.664 * gb_minus_air_rate * abs(gb_minus_air_rate)
            + 10.130 * strikeout_rate * gb_minus_air_rate
            - 5.195 * walk_rate * gb_minus_air_rate
            - 8.000 * strikeout_rate * walk_rate
        )
        rows.append(
            {
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher_name,
                "p_throws": throws,
                "pitch_count": pitch_count,
                "bip": bip,
                "xwoba": _weighted_sum(group, "xwoba_value", group.index) / max(_weighted_denominator(group, "xwoba_value", group.index), 1e-9),
                "swstr_pct": float((group["is_swinging_strike"].astype(int) * group["metric_weight"]).sum()) / max(pitch_weight_sum, 1e-9),
                "called_strike_pct": called_strike_weight_sum / max(pitch_weight_sum, 1e-9),
                "csw_pct": csw_weight_sum / max(pitch_weight_sum, 1e-9),
                "ball_pct": ball_weight_sum / max(pitch_weight_sum, 1e-9),
                "putaway_pct": putaway_strikeout_weight_sum / max(putaway_chance_weight_sum, 1e-9),
                "siera": siera,
                "barrel_bbe_pct": barrel_weight_sum / max(tracked_bbe_weight_sum, 1e-9),
                "barrel_bip_pct": barrel_weight_sum / max(bip_weight_sum, 1e-9),
                "pulled_barrel_pct": pulled_barrel_weight_sum / max(tracked_bbe_weight_sum, 1e-9),
                "fb_pct": float((group["is_fly_ball"].astype(int) * group["metric_weight"]).sum()) / max(bip_weight_sum, 1e-9),
                "gb_pct": ground_ball_weight_sum / max(bip_weight_sum, 1e-9),
                "gb_fb_ratio": ground_ball_weight_sum / max(fly_ball_weight_sum, 1e-9),
                "hard_hit_pct": float((group["is_hard_hit"].astype(int) * group["metric_weight"]).sum()) / max(bip_weight_sum, 1e-9),
                "avg_launch_angle": _weighted_sum(group, "launch_angle_value", group.index) / max(_weighted_denominator(group, "launch_angle_value", group.index), 1e-9),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_pitcher_summary_by_hand(frame: pd.DataFrame, weighted_mode: str, year_weights: dict[int, float]) -> pd.DataFrame:
    split_frames = {
        "all": frame,
        "vs_lhh": frame.loc[frame["stand"].fillna("") == "L"],
        "vs_rhh": frame.loc[frame["stand"].fillna("") == "R"],
    }
    outputs: list[pd.DataFrame] = []
    for side_key, side_frame in split_frames.items():
        if side_frame.empty:
            continue
        aggregated = _aggregate_pitcher_metrics(side_frame, weighted_mode, year_weights)
        aggregated["batter_side_key"] = side_key
        outputs.append(aggregated)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _aggregate_pitcher_arsenal(frame: pd.DataFrame, weighted_mode: str, year_weights: dict[int, float]) -> pd.DataFrame:
    work = frame.copy()
    work["metric_weight"] = apply_year_weights(work, year_weights) if weighted_mode == "weighted" else 1.0
    rows: list[dict] = []
    for (pitcher_id, pitch_name), group in work.groupby(["pitcher", "pitch_name"], sort=False):
        pitcher_name = group["pitcher_name"].dropna().astype(str).value_counts().idxmax() if group["pitcher_name"].notna().any() else str(pitcher_id)
        pitch_weight_sum = float(group["metric_weight"].sum())
        bip_weight_sum = float(group.loc[group["is_batted_ball"], "metric_weight"].sum())
        rows.append(
            {
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher_name,
                "pitch_name": pitch_name,
                "pitch_count": len(group),
                "usage_weight": float(group["metric_weight"].sum()),
                "swstr_pct": float((group["is_swinging_strike"].astype(int) * group["metric_weight"]).sum()) / max(pitch_weight_sum, 1e-9),
                "hard_hit_pct": float((group["is_hard_hit"].astype(int) * group["metric_weight"]).sum()) / max(bip_weight_sum, 1e-9),
                "avg_release_speed": _weighted_sum(group, "release_speed_value", group.index) / max(_weighted_denominator(group, "release_speed_value", group.index), 1e-9),
                "avg_spin_rate": _weighted_sum(group, "spin_rate_value", group.index) / max(_weighted_denominator(group, "spin_rate_value", group.index), 1e-9),
                "xwoba_con": _weighted_sum(group.loc[group["is_batted_ball"]], "xwoba_value", group.loc[group["is_batted_ball"]].index) / max(float(group.loc[group["is_batted_ball"], "metric_weight"].sum()), 1e-9),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    totals = result.groupby("pitcher_id")["usage_weight"].transform("sum")
    result["usage_pct"] = result["usage_weight"] / totals.where(totals != 0)
    return result.drop(columns=["usage_weight"])


def _aggregate_pitcher_arsenal_by_hand(frame: pd.DataFrame, weighted_mode: str, year_weights: dict[int, float]) -> pd.DataFrame:
    split_frames = {
        "all": frame,
        "vs_lhh": frame.loc[frame["stand"].fillna("") == "L"],
        "vs_rhh": frame.loc[frame["stand"].fillna("") == "R"],
    }
    outputs: list[pd.DataFrame] = []
    for side_key, side_frame in split_frames.items():
        if side_frame.empty:
            continue
        aggregated = _aggregate_pitcher_arsenal(side_frame, weighted_mode, year_weights)
        aggregated["batter_side_key"] = side_key
        outputs.append(aggregated)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _aggregate_pitcher_usage_by_count(frame: pd.DataFrame, weighted_mode: str, year_weights: dict[int, float]) -> pd.DataFrame:
    work = frame.copy()
    work["metric_weight"] = apply_year_weights(work, year_weights) if weighted_mode == "weighted" else 1.0
    balls = pd.to_numeric(work["balls"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(work["strikes"], errors="coerce").fillna(0).astype(int)
    split_frames = {
        "all": work,
        "vs_lhh": work.loc[work["stand"].fillna("") == "L"],
        "vs_rhh": work.loc[work["stand"].fillna("") == "R"],
    }
    bucket_masks = {
        "All counts": pd.Series(True, index=work.index),
        "Early count": (balls + strikes) <= 1,
        "Even count": balls == strikes,
        "Pitcher ahead": strikes > balls,
        "Pitcher behind": balls > strikes,
        "Two-strike": strikes >= 2,
        "Pre two-strike": strikes < 2,
        "Full count": (balls == 3) & (strikes == 2),
    }
    rows: list[dict] = []
    for side_key, side_frame in split_frames.items():
        if side_frame.empty:
            continue
        for count_bucket, mask in bucket_masks.items():
            bucket_frame = side_frame.loc[mask.reindex(side_frame.index, fill_value=False)]
            if bucket_frame.empty:
                continue
            totals = bucket_frame.groupby("pitcher")["metric_weight"].sum()
            for (pitcher_id, pitch_name), group in bucket_frame.groupby(["pitcher", "pitch_name"], sort=False):
                pitcher_name = group["pitcher_name"].dropna().astype(str).value_counts().idxmax() if group["pitcher_name"].notna().any() else str(pitcher_id)
                denom = float(totals.get(pitcher_id, 0.0))
                rows.append(
                    {
                        "pitcher_id": pitcher_id,
                        "pitcher_name": pitcher_name,
                        "pitch_name": pitch_name,
                        "count_bucket": count_bucket,
                        "batter_side_key": side_key,
                        "usage_pct": float(group["metric_weight"].sum()) / max(denom, 1e-9),
                    }
                )
    return pd.DataFrame(rows)


def _build_hitter_pitcher_exclusions(raw_statcast: pd.DataFrame, inning_threshold: float = 5.0) -> pd.DataFrame:
    columns = ["player_id", "pitcher_name", "outs_recorded", "innings_pitched", "exclude_from_hitter_tables", "override_reason"]
    if raw_statcast.empty:
        return pd.DataFrame(columns=columns)

    required = {"game_pk", "pitcher", "at_bat_number", "events"}
    if not required.issubset(raw_statcast.columns):
        return pd.DataFrame(columns=columns)

    pa_events = (
        raw_statcast.sort_values(["game_pk", "pitcher", "at_bat_number"])
        .groupby(["game_pk", "pitcher", "at_bat_number"], as_index=False)
        .tail(1)
        .copy()
    )
    event_text = pa_events["events"].fillna("").astype(str).str.lower()
    out_map = {
        "field_out": 1,
        "force_out": 1,
        "sac_fly": 1,
        "sac_bunt": 1,
        "strikeout": 1,
        "fielders_choice_out": 1,
        "other_out": 1,
        "double_play": 2,
        "grounded_into_double_play": 2,
        "strikeout_double_play": 2,
        "sac_fly_double_play": 2,
        "sac_bunt_double_play": 2,
        "triple_play": 3,
    }
    pa_events["outs_recorded"] = event_text.map(out_map).fillna(0).astype(int)
    pa_events["pitcher_name"] = pa_events["pitcher_name"].fillna(pa_events["player_name"]).astype(str)

    exclusions = (
        pa_events.groupby("pitcher", as_index=False)
        .agg(
            pitcher_name=("pitcher_name", lambda values: values.dropna().astype(str).value_counts().idxmax() if values.notna().any() else ""),
            outs_recorded=("outs_recorded", "sum"),
        )
        .rename(columns={"pitcher": "player_id"})
    )
    exclusions["innings_pitched"] = exclusions["outs_recorded"] / 3.0
    exclusions["override_reason"] = pd.NA

    ohtani_mask = exclusions["pitcher_name"].fillna("").str.strip().str.lower().eq("shohei ohtani")
    threshold_mask = exclusions["innings_pitched"].fillna(0) >= float(inning_threshold)
    exclusions["exclude_from_hitter_tables"] = threshold_mask & ~ohtani_mask
    exclusions.loc[ohtani_mask, "override_reason"] = "two_way_exception"

    return exclusions[columns]


def _append_split_metadata(frame: pd.DataFrame, split_key: str, recent_window: str, weighted_mode: str) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["split_key"] = split_key
    enriched["recent_window"] = recent_window
    enriched["weighted_mode"] = weighted_mode
    return enriched


def _pitch_family_from_series(pitch_name: pd.Series, pitch_type: pd.Series) -> pd.Series:
    name_series = pitch_name.fillna("").astype(str).str.strip().str.lower()
    type_series = pitch_type.fillna("").astype(str).str.strip().str.upper()
    family = pd.Series(pd.NA, index=name_series.index, dtype="object")

    fastball_names = {"4-seam fastball", "four-seam fastball", "sinker", "cutter"}
    breaking_names = {"slider", "sweeper", "curveball", "knuckle curve", "slurve"}
    offspeed_names = {"changeup", "split-finger", "splitter", "forkball", "screwball"}

    family.loc[name_series.isin(fastball_names)] = "fastball"
    family.loc[name_series.isin(breaking_names)] = "breaking"
    family.loc[name_series.isin(offspeed_names)] = "offspeed"

    family.loc[family.isna() & type_series.isin({"FF", "FA", "FT", "SI", "FC"})] = "fastball"
    family.loc[family.isna() & type_series.isin({"SL", "ST", "CU", "KC", "SV", "CS"})] = "breaking"
    family.loc[family.isna() & type_series.isin({"CH", "FS", "FO", "SC"})] = "offspeed"
    return family


def _zone_bucket_from_location(frame: pd.DataFrame) -> pd.Series:
    plate_x = pd.to_numeric(frame.get("plate_x"), errors="coerce")
    plate_z = pd.to_numeric(frame.get("plate_z"), errors="coerce")
    heart = plate_x.abs().le(0.55) & plate_z.between(1.90, 3.10, inclusive="both")
    shadow = plate_x.abs().le(1.10) & plate_z.between(1.45, 3.55, inclusive="both") & ~heart
    bucket = pd.Series(pd.NA, index=frame.index, dtype="object")
    bucket.loc[heart] = "heart"
    bucket.loc[shadow] = "shadow"
    return bucket


def _weighted_mean_from_columns(frame: pd.DataFrame, value_column: str, weight_column: str = "context_weight") -> pd.Series:
    values = pd.to_numeric(frame.get(value_column), errors="coerce")
    weights = pd.to_numeric(frame.get(weight_column), errors="coerce").fillna(0.0)
    valid = values.notna() & weights.gt(0)
    value_weight = pd.Series(0.0, index=frame.index, dtype="float64")
    value_denom = pd.Series(0.0, index=frame.index, dtype="float64")
    value_weight.loc[valid] = values.loc[valid] * weights.loc[valid]
    value_denom.loc[valid] = weights.loc[valid]
    return value_weight, value_denom


def _prepare_weighted_pitch_context_source(live_payload: CockroachPayload, config: AppConfig) -> pd.DataFrame:
    live = live_payload.live_pitch_mix.copy()
    baseline = live_payload.pitcher_baseline_event_rows.copy()

    if not baseline.empty:
        baseline = baseline.loc[pd.to_numeric(baseline.get("game_year"), errors="coerce").fillna(0).astype(int) < 2026].copy()
    frames = [frame for frame in [live, baseline] if not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["game_year"] = pd.to_numeric(combined.get("game_year", combined.get("source_season")), errors="coerce")
    combined = combined.loc[combined["game_year"].notna()].copy()
    combined["game_year"] = combined["game_year"].astype(int)
    combined = combined.loc[combined["game_year"].isin(config.movement_year_weights.keys())].copy()
    if combined.empty:
        return combined

    combined["pitch_family"] = _pitch_family_from_series(combined.get("pitch_name", pd.Series(index=combined.index, dtype="object")), combined.get("pitch_type", pd.Series(index=combined.index, dtype="object")))
    combined["zone_bucket"] = _zone_bucket_from_location(combined)
    combined["context_weight"] = apply_year_weights(combined, config.movement_year_weights)
    combined["sample_2026"] = combined["game_year"].eq(2026).astype(int)
    combined["sample_prior"] = combined["game_year"].lt(2026).astype(int)
    combined["weight_2026"] = combined["context_weight"].where(combined["game_year"].eq(2026), 0.0)
    combined["weight_prior"] = combined["context_weight"].where(combined["game_year"].lt(2026), 0.0)
    combined["pitcher_id"] = pd.to_numeric(combined.get("pitcher"), errors="coerce")
    combined["pitcher_name"] = combined.get("pitcher_name", combined.get("player_name", pd.Series(index=combined.index, dtype="object")))
    combined["p_throws"] = combined.get("p_throws", combined.get("pitcher_hand", pd.Series(index=combined.index, dtype="object")))
    return combined


def _aggregate_weighted_pitcher_family_zone_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame["pitcher_id"].notna() & frame["pitch_family"].notna() & frame["zone_bucket"].notna()].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "pitcher_id",
                "pitcher_name",
                "p_throws",
                "pitch_family",
                "zone_bucket",
                "sample_size",
                "weighted_sample_size",
                "sample_2026",
                "sample_prior",
                "weight_2026",
                "weight_prior",
                "prior_weight_share",
                "usage_rate_within_family",
                "usage_rate_overall",
                "whiff_rate",
                "called_strike_rate",
                "ball_in_play_rate",
                "hit_allowed_rate",
                "hr_allowed_rate",
                "damage_allowed_rate",
                "avg_launch_speed_allowed",
                "avg_launch_angle_allowed",
                "xwoba_allowed",
            ]
        )

    weights = pd.to_numeric(work["context_weight"], errors="coerce").fillna(0.0)
    work["weighted_sample_size"] = weights
    work["whiff_weight"] = work["is_swinging_strike"].astype(int) * weights
    work["called_strike_weight"] = work["is_called_strike"].astype(int) * weights
    work["ball_in_play_weight"] = work["is_batted_ball"].astype(int) * weights
    work["hit_allowed_weight"] = work["is_hit_event"].astype(int) * weights
    work["hr_allowed_weight"] = work["is_home_run_event"].astype(int) * weights
    work["damage_allowed_weight"] = ((work["is_hit_event"].astype(int) * 0.6) + (work["is_home_run_event"].astype(int) * 0.4)) * weights
    work["launch_speed_weight"], work["launch_speed_denom"] = _weighted_mean_from_columns(work, "launch_speed")
    work["launch_angle_weight"], work["launch_angle_denom"] = _weighted_mean_from_columns(work, "launch_angle")
    work["xwoba_weight"], work["xwoba_denom"] = _weighted_mean_from_columns(work, "estimated_woba_using_speedangle")

    grouped = (
        work.groupby(["pitcher_id", "pitcher_name", "p_throws", "pitch_family", "zone_bucket"], as_index=False)
        .agg(
            sample_size=("pitcher_id", "size"),
            weighted_sample_size=("weighted_sample_size", "sum"),
            sample_2026=("sample_2026", "sum"),
            sample_prior=("sample_prior", "sum"),
            weight_2026=("weight_2026", "sum"),
            weight_prior=("weight_prior", "sum"),
            whiff_weight=("whiff_weight", "sum"),
            called_strike_weight=("called_strike_weight", "sum"),
            ball_in_play_weight=("ball_in_play_weight", "sum"),
            hit_allowed_weight=("hit_allowed_weight", "sum"),
            hr_allowed_weight=("hr_allowed_weight", "sum"),
            damage_allowed_weight=("damage_allowed_weight", "sum"),
            launch_speed_weight=("launch_speed_weight", "sum"),
            launch_speed_denom=("launch_speed_denom", "sum"),
            launch_angle_weight=("launch_angle_weight", "sum"),
            launch_angle_denom=("launch_angle_denom", "sum"),
            xwoba_weight=("xwoba_weight", "sum"),
            xwoba_denom=("xwoba_denom", "sum"),
        )
    )

    overall_totals = grouped.groupby("pitcher_id")["weighted_sample_size"].transform("sum")
    family_totals = grouped.groupby(["pitcher_id", "pitch_family"])["weighted_sample_size"].transform("sum")
    grouped["prior_weight_share"] = grouped["weight_prior"] / grouped["weighted_sample_size"].where(grouped["weighted_sample_size"] != 0)
    grouped["usage_rate_overall"] = grouped["weighted_sample_size"] / overall_totals.where(overall_totals != 0)
    grouped["usage_rate_within_family"] = grouped["weighted_sample_size"] / family_totals.where(family_totals != 0)
    grouped["whiff_rate"] = grouped["whiff_weight"] / grouped["weighted_sample_size"].where(grouped["weighted_sample_size"] != 0)
    grouped["called_strike_rate"] = grouped["called_strike_weight"] / grouped["weighted_sample_size"].where(grouped["weighted_sample_size"] != 0)
    grouped["ball_in_play_rate"] = grouped["ball_in_play_weight"] / grouped["weighted_sample_size"].where(grouped["weighted_sample_size"] != 0)
    grouped["hit_allowed_rate"] = grouped["hit_allowed_weight"] / grouped["weighted_sample_size"].where(grouped["weighted_sample_size"] != 0)
    grouped["hr_allowed_rate"] = grouped["hr_allowed_weight"] / grouped["weighted_sample_size"].where(grouped["weighted_sample_size"] != 0)
    grouped["damage_allowed_rate"] = grouped["damage_allowed_weight"] / grouped["weighted_sample_size"].where(grouped["weighted_sample_size"] != 0)
    grouped["avg_launch_speed_allowed"] = grouped["launch_speed_weight"] / grouped["launch_speed_denom"].where(grouped["launch_speed_denom"] != 0)
    grouped["avg_launch_angle_allowed"] = grouped["launch_angle_weight"] / grouped["launch_angle_denom"].where(grouped["launch_angle_denom"] != 0)
    grouped["xwoba_allowed"] = grouped["xwoba_weight"] / grouped["xwoba_denom"].where(grouped["xwoba_denom"] != 0)
    return grouped[
        [
            "pitcher_id",
            "pitcher_name",
            "p_throws",
            "pitch_family",
            "zone_bucket",
            "sample_size",
            "weighted_sample_size",
            "sample_2026",
            "sample_prior",
            "weight_2026",
            "weight_prior",
            "prior_weight_share",
            "usage_rate_within_family",
            "usage_rate_overall",
            "whiff_rate",
            "called_strike_rate",
            "ball_in_play_rate",
            "hit_allowed_rate",
            "hr_allowed_rate",
            "damage_allowed_rate",
            "avg_launch_speed_allowed",
            "avg_launch_angle_allowed",
            "xwoba_allowed",
        ]
    ]


def _aggregate_weighted_pitcher_movement_arsenal(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame["pitcher_id"].notna() & frame["pitch_name"].notna() & frame["pitch_family"].notna()].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "pitcher_id",
                "pitcher_name",
                "p_throws",
                "pitch_name",
                "pitch_type",
                "pitch_family",
                "sample_size",
                "weighted_sample_size",
                "sample_2026",
                "sample_prior",
                "weight_2026",
                "weight_prior",
                "prior_weight_share",
                "usage_rate",
                "avg_velocity",
                "avg_spin_rate",
                "avg_extension",
                "avg_pfx_x",
                "avg_pfx_z",
                "avg_release_pos_x",
                "avg_release_pos_z",
                "avg_spin_axis",
            ]
        )

    weights = pd.to_numeric(work["context_weight"], errors="coerce").fillna(0.0)
    work["weighted_sample_size"] = weights
    work["pitch_type_group"] = work.get("pitch_type", pd.Series(index=work.index, dtype="object")).fillna(work["pitch_name"])
    for column in [
        "release_speed",
        "release_spin_rate",
        "release_extension",
        "pfx_x",
        "pfx_z",
        "release_pos_x",
        "release_pos_z",
        "spin_axis",
    ]:
        value_weight, value_denom = _weighted_mean_from_columns(work, column)
        work[f"{column}_weight"] = value_weight
        work[f"{column}_denom"] = value_denom

    grouped = (
        work.groupby(["pitcher_id", "pitcher_name", "p_throws", "pitch_name", "pitch_type_group", "pitch_family"], as_index=False)
        .agg(
            sample_size=("pitcher_id", "size"),
            weighted_sample_size=("weighted_sample_size", "sum"),
            sample_2026=("sample_2026", "sum"),
            sample_prior=("sample_prior", "sum"),
            weight_2026=("weight_2026", "sum"),
            weight_prior=("weight_prior", "sum"),
            release_speed_weight=("release_speed_weight", "sum"),
            release_speed_denom=("release_speed_denom", "sum"),
            release_spin_rate_weight=("release_spin_rate_weight", "sum"),
            release_spin_rate_denom=("release_spin_rate_denom", "sum"),
            release_extension_weight=("release_extension_weight", "sum"),
            release_extension_denom=("release_extension_denom", "sum"),
            pfx_x_weight=("pfx_x_weight", "sum"),
            pfx_x_denom=("pfx_x_denom", "sum"),
            pfx_z_weight=("pfx_z_weight", "sum"),
            pfx_z_denom=("pfx_z_denom", "sum"),
            release_pos_x_weight=("release_pos_x_weight", "sum"),
            release_pos_x_denom=("release_pos_x_denom", "sum"),
            release_pos_z_weight=("release_pos_z_weight", "sum"),
            release_pos_z_denom=("release_pos_z_denom", "sum"),
            spin_axis_weight=("spin_axis_weight", "sum"),
            spin_axis_denom=("spin_axis_denom", "sum"),
        )
    )
    grouped = grouped.rename(columns={"pitch_type_group": "pitch_type"})
    totals = grouped.groupby("pitcher_id")["weighted_sample_size"].transform("sum")
    grouped["prior_weight_share"] = grouped["weight_prior"] / grouped["weighted_sample_size"].where(grouped["weighted_sample_size"] != 0)
    grouped["usage_rate"] = grouped["weighted_sample_size"] / totals.where(totals != 0)
    grouped["avg_velocity"] = grouped["release_speed_weight"] / grouped["release_speed_denom"].where(grouped["release_speed_denom"] != 0)
    grouped["avg_spin_rate"] = grouped["release_spin_rate_weight"] / grouped["release_spin_rate_denom"].where(grouped["release_spin_rate_denom"] != 0)
    grouped["avg_extension"] = grouped["release_extension_weight"] / grouped["release_extension_denom"].where(grouped["release_extension_denom"] != 0)
    grouped["avg_pfx_x"] = grouped["pfx_x_weight"] / grouped["pfx_x_denom"].where(grouped["pfx_x_denom"] != 0)
    grouped["avg_pfx_z"] = grouped["pfx_z_weight"] / grouped["pfx_z_denom"].where(grouped["pfx_z_denom"] != 0)
    grouped["avg_release_pos_x"] = grouped["release_pos_x_weight"] / grouped["release_pos_x_denom"].where(grouped["release_pos_x_denom"] != 0)
    grouped["avg_release_pos_z"] = grouped["release_pos_z_weight"] / grouped["release_pos_z_denom"].where(grouped["release_pos_z_denom"] != 0)
    grouped["avg_spin_axis"] = grouped["spin_axis_weight"] / grouped["spin_axis_denom"].where(grouped["spin_axis_denom"] != 0)
    return grouped[
        [
            "pitcher_id",
            "pitcher_name",
            "p_throws",
            "pitch_name",
            "pitch_type",
            "pitch_family",
            "sample_size",
            "weighted_sample_size",
            "sample_2026",
            "sample_prior",
            "weight_2026",
            "weight_prior",
            "prior_weight_share",
            "usage_rate",
            "avg_velocity",
            "avg_spin_rate",
            "avg_extension",
            "avg_pfx_x",
            "avg_pfx_z",
            "avg_release_pos_x",
            "avg_release_pos_z",
            "avg_spin_axis",
        ]
    ]


def build_movement_context_tables(
    live_payload: CockroachPayload,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = _prepare_weighted_pitch_context_source(live_payload, config)
    pitcher_family_zone_context = _aggregate_weighted_pitcher_family_zone_profiles(source)
    pitcher_movement_arsenal = _aggregate_weighted_pitcher_movement_arsenal(source)
    batter_family_zone_profiles = live_payload.batter_family_zone_profiles.copy()
    return pitcher_family_zone_context, pitcher_movement_arsenal, batter_family_zone_profiles, source


def _pitch_shape_diagnostics(
    schedule: list[dict],
    live_payload: CockroachPayload,
    source: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    pitcher_movement_arsenal: pd.DataFrame,
) -> pd.DataFrame:
    probable_rows: list[dict] = []
    for game in schedule:
        for side in ("away", "home"):
            probable_rows.append(
                {
                    "team": game.get(f"{side}_team"),
                    "pitcher_id": game.get(f"{side}_probable_pitcher_id"),
                    "pitcher_name": game.get(f"{side}_probable_pitcher_name"),
                    "opponent": game.get("home_team") if side == "away" else game.get("away_team"),
                    "game_pk": game.get("game_pk"),
                }
            )
    probable = pd.DataFrame(probable_rows).dropna(subset=["pitcher_id"]).drop_duplicates(subset=["pitcher_id"]).copy()
    if probable.empty:
        return probable
    probable["pitcher_id"] = pd.to_numeric(probable["pitcher_id"], errors="coerce").astype("Int64")

    live_counts = (
        live_payload.live_pitch_mix.assign(pitcher_id=pd.to_numeric(live_payload.live_pitch_mix.get("pitcher"), errors="coerce"))
        .groupby("pitcher_id", as_index=False)
        .agg(live_2026_rows=("pitcher_id", "size"))
        if not live_payload.live_pitch_mix.empty
        else pd.DataFrame(columns=["pitcher_id", "live_2026_rows"])
    )
    baseline_counts = (
        live_payload.pitcher_baseline_event_rows.assign(pitcher_id=pd.to_numeric(live_payload.pitcher_baseline_event_rows.get("pitcher"), errors="coerce"))
        .groupby("pitcher_id", as_index=False)
        .agg(
            baseline_rows=("pitcher_id", "size"),
            baseline_pitch_type_present=("pitch_type", lambda s: int(pd.Series(s).notna().sum())),
            baseline_location_present=("plate_x", lambda s: int(pd.Series(s).notna().sum())),
            baseline_location_z_present=("plate_z", lambda s: int(pd.Series(s).notna().sum())),
            baseline_movement_present=("release_speed", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
        if not live_payload.pitcher_baseline_event_rows.empty
        else pd.DataFrame(
            columns=[
                "pitcher_id",
                "baseline_rows",
                "baseline_pitch_type_present",
                "baseline_location_present",
                "baseline_location_z_present",
                "baseline_movement_present",
            ]
        )
    )
    source_counts = (
        source.groupby("pitcher_id", as_index=False).agg(
            prepared_rows=("pitcher_id", "size"),
            prepared_pitch_type_present=("pitch_type", lambda s: int(pd.Series(s).notna().sum())),
            prepared_zone_bucket_present=("zone_bucket", lambda s: int(pd.Series(s).notna().sum())),
        )
        if not source.empty
        else pd.DataFrame(columns=["pitcher_id", "prepared_rows", "prepared_pitch_type_present", "prepared_zone_bucket_present"])
    )
    movement_counts = (
        pitcher_movement_arsenal.groupby("pitcher_id", as_index=False).agg(movement_rows=("pitcher_id", "size"))
        if not pitcher_movement_arsenal.empty
        else pd.DataFrame(columns=["pitcher_id", "movement_rows"])
    )
    family_counts = (
        pitcher_family_zone_context.groupby("pitcher_id", as_index=False).agg(family_rows=("pitcher_id", "size"))
        if not pitcher_family_zone_context.empty
        else pd.DataFrame(columns=["pitcher_id", "family_rows"])
    )

    diagnostics = probable.merge(live_counts, on="pitcher_id", how="left")
    diagnostics = diagnostics.merge(baseline_counts, on="pitcher_id", how="left")
    diagnostics = diagnostics.merge(source_counts, on="pitcher_id", how="left")
    diagnostics = diagnostics.merge(movement_counts, on="pitcher_id", how="left")
    diagnostics = diagnostics.merge(family_counts, on="pitcher_id", how="left")
    fill_zero_columns = [
        "live_2026_rows",
        "baseline_rows",
        "baseline_pitch_type_present",
        "baseline_location_present",
        "baseline_location_z_present",
        "baseline_movement_present",
        "prepared_rows",
        "prepared_pitch_type_present",
        "prepared_zone_bucket_present",
        "movement_rows",
        "family_rows",
    ]
    for column in fill_zero_columns:
        diagnostics[column] = pd.to_numeric(diagnostics.get(column), errors="coerce").fillna(0).astype(int)

    def root_causes(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["live_2026_rows"] == 0:
            reasons.append("no_2026_live_rows")
        if row["baseline_rows"] == 0:
            reasons.append("no_baseline_rows")
        if row["baseline_rows"] > 0 and row["prepared_pitch_type_present"] == 0:
            reasons.append("missing_pitch_type")
        if row["baseline_rows"] > 0 and row["prepared_zone_bucket_present"] == 0:
            reasons.append("missing_usable_location")
        if row["baseline_rows"] > 0 and row["baseline_movement_present"] == 0:
            reasons.append("missing_movement_fields")
        if row["movement_rows"] == 0:
            reasons.append("no_movement_output")
        if row["family_rows"] == 0:
            reasons.append("no_family_output")
        return ",".join(reasons)

    diagnostics["root_causes"] = diagnostics.apply(root_causes, axis=1)
    return diagnostics


def build_metric_tables(raw_statcast: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hitters: list[pd.DataFrame] = []
    pitchers: list[pd.DataFrame] = []
    pitcher_summaries_by_hand: list[pd.DataFrame] = []
    arsenals: list[pd.DataFrame] = []
    arsenals_by_hand: list[pd.DataFrame] = []
    usages_by_count: list[pd.DataFrame] = []
    max_date = raw_statcast["game_date"].max()
    starter_scores = likely_starter_scores(raw_statcast[["team", "batter", "game_pk", "at_bat_number", "game_date"]])
    if "batter" in starter_scores.columns:
        starter_scores["batter"] = pd.to_numeric(starter_scores["batter"], errors="coerce")
        starter_scores = starter_scores.loc[starter_scores["batter"].notna()].copy()
        starter_scores["batter"] = starter_scores["batter"].astype(int)
    for recent_window in DEFAULT_RECENT_WINDOWS:
        recent_cutoff = _window_cutoff(max_date, recent_window)
        window_frame = raw_statcast.loc[raw_statcast["game_date"] >= recent_cutoff]
        for split_key in DEFAULT_SPLITS:
            split_frame = window_frame.loc[_split_mask(window_frame, split_key)]
            for weighted_mode in ("weighted", "unweighted"):
                hitters_frame = _aggregate_hitter_metrics(split_frame, weighted_mode, config.year_weights)
                if not hitters_frame.empty:
                    hitters_frame["batter"] = pd.to_numeric(hitters_frame["batter"], errors="coerce")
                    hitters_frame = hitters_frame.loc[hitters_frame["batter"].notna()].copy()
                    hitters_frame["batter"] = hitters_frame["batter"].astype(int)
                    hitters_frame = hitters_frame.merge(starter_scores, on=["batter"], how="left")
                hitters.append(_append_split_metadata(hitters_frame, split_key, recent_window, weighted_mode))
                pitchers.append(_append_split_metadata(_aggregate_pitcher_metrics(split_frame, weighted_mode, config.year_weights), split_key, recent_window, weighted_mode))
                pitcher_summaries_by_hand.append(
                    _append_split_metadata(
                        _aggregate_pitcher_summary_by_hand(split_frame, weighted_mode, config.year_weights),
                        split_key,
                        recent_window,
                        weighted_mode,
                    )
                )
                arsenals.append(_append_split_metadata(_aggregate_pitcher_arsenal(split_frame, weighted_mode, config.year_weights), split_key, recent_window, weighted_mode))
                arsenals_by_hand.append(
                    _append_split_metadata(
                        _aggregate_pitcher_arsenal_by_hand(split_frame, weighted_mode, config.year_weights),
                        split_key,
                        recent_window,
                        weighted_mode,
                    )
                )
                usages_by_count.append(
                    _append_split_metadata(
                        _aggregate_pitcher_usage_by_count(split_frame, weighted_mode, config.year_weights),
                        split_key,
                        recent_window,
                        weighted_mode,
                    )
                )
    hitter_metrics = pd.concat(hitters, ignore_index=True)
    if raw_statcast["batter"].notna().any() and hitter_metrics.empty:
        raise RuntimeError("Hitter metrics build produced zero rows despite non-empty batter data. Aborting build to avoid publishing empty hitter artifacts.")
    return (
        hitter_metrics,
        pd.concat(pitchers, ignore_index=True),
        pd.concat(pitcher_summaries_by_hand, ignore_index=True),
        pd.concat(arsenals, ignore_index=True),
        pd.concat(arsenals_by_hand, ignore_index=True),
        pd.concat(usages_by_count, ignore_index=True),
    )


def _save_daily_files(
    context: BuildContext,
    schedule: list[dict],
    rosters: list[dict],
    hitter_metrics: pd.DataFrame,
    pitcher_metrics: pd.DataFrame,
    pitcher_summary_by_hand: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    pitcher_arsenal_by_hand: pd.DataFrame,
    pitcher_usage_by_count: pd.DataFrame,
    hitter_rolling: pd.DataFrame,
    pitcher_rolling: pd.DataFrame,
    batter_zone_profiles: pd.DataFrame,
    pitcher_zone_profiles: pd.DataFrame,
    batter_family_zone_profiles: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    pitcher_movement_arsenal: pd.DataFrame,
    hitter_pitcher_exclusions: pd.DataFrame,
    pitch_shape_diagnostics: pd.DataFrame,
    tracking_health: pd.DataFrame,
) -> None:
    target_dir = context.config.daily_dir / context.target_date.isoformat()
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "slate.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    (target_dir / "rosters.json").write_text(json.dumps(rosters, indent=2), encoding="utf-8")
    pd.DataFrame(schedule).to_parquet(target_dir / "slate.parquet", index=False)
    if rosters:
        pd.DataFrame(rosters).to_parquet(target_dir / "rosters.parquet", index=False)
    probable_pitchers = {game.get("home_probable_pitcher_id") for game in schedule} | {game.get("away_probable_pitcher_id") for game in schedule}
    probable_pitchers.discard(None)
    pitcher_metrics.loc[pitcher_metrics["pitcher_id"].isin(probable_pitchers)].to_parquet(target_dir / "daily_pitcher_metrics.parquet", index=False)
    pitcher_summary_by_hand.loc[pitcher_summary_by_hand["pitcher_id"].isin(probable_pitchers)].to_parquet(
        target_dir / "daily_pitcher_summary_by_hand.parquet", index=False
    )
    pitcher_arsenal.loc[pitcher_arsenal["pitcher_id"].isin(probable_pitchers)].to_parquet(target_dir / "daily_pitcher_arsenal.parquet", index=False)
    pitcher_arsenal_by_hand.loc[pitcher_arsenal_by_hand["pitcher_id"].isin(probable_pitchers)].to_parquet(
        target_dir / "daily_pitcher_arsenal_by_hand.parquet", index=False
    )
    pitcher_usage_by_count.loc[pitcher_usage_by_count["pitcher_id"].isin(probable_pitchers)].to_parquet(
        target_dir / "daily_pitcher_usage_by_count.parquet", index=False
    )
    hitter_rolling.to_parquet(target_dir / "daily_hitter_rolling.parquet", index=False)
    pitcher_rolling.to_parquet(target_dir / "daily_pitcher_rolling.parquet", index=False)
    batter_zone_profiles.to_parquet(target_dir / "daily_batter_zone_profiles.parquet", index=False)
    pitcher_zone_profiles.to_parquet(target_dir / "daily_pitcher_zone_profiles.parquet", index=False)
    batter_family_zone_profiles.to_parquet(target_dir / "daily_batter_family_zone_profiles.parquet", index=False)
    pitcher_family_zone_context.loc[pitcher_family_zone_context["pitcher_id"].isin(probable_pitchers)].to_parquet(
        target_dir / "daily_pitcher_family_zone_context.parquet",
        index=False,
    )
    pitcher_movement_arsenal.loc[pitcher_movement_arsenal["pitcher_id"].isin(probable_pitchers)].to_parquet(
        target_dir / "daily_pitcher_movement_arsenal.parquet",
        index=False,
    )
    pitch_shape_diagnostics.to_parquet(target_dir / "pitch_shape_diagnostics.parquet", index=False)
    tracking_health.to_parquet(target_dir / "tracking_health.parquet", index=False)
    hitter_metrics.to_parquet(target_dir / "daily_hitter_metrics.parquet", index=False)
    hitter_pitcher_exclusions.to_parquet(target_dir / "hitter_pitcher_exclusions.parquet", index=False)
    metadata = {
        "build_timestamp_utc": datetime.now(UTC).isoformat(),
        "target_date": context.target_date.isoformat(),
        "metrics_version": context.config.metrics_version,
        "split_keys": list(DEFAULT_SPLITS),
        "recent_windows": list(DEFAULT_RECENT_WINDOWS),
        "pitch_groups": DEFAULT_PITCH_GROUPS,
        "movement_year_weights": context.config.movement_year_weights,
        "live_source_table": context.config.cockroach_live_pitch_mix_table,
        "live_source_max_event_date": None if tracking_health.empty else pd.to_datetime(tracking_health["source_max_event_date"], errors="coerce").dropna().max().date().isoformat() if pd.to_datetime(tracking_health["source_max_event_date"], errors="coerce").notna().any() else None,
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_duckdb(
    config: AppConfig,
    hitter_metrics: pd.DataFrame,
    pitcher_metrics: pd.DataFrame,
    pitcher_summary_by_hand: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    pitcher_arsenal_by_hand: pd.DataFrame,
    pitcher_usage_by_count: pd.DataFrame,
    hitter_rolling: pd.DataFrame,
    pitcher_rolling: pd.DataFrame,
    batter_zone_profiles: pd.DataFrame,
    pitcher_zone_profiles: pd.DataFrame,
    batter_family_zone_profiles: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    pitcher_movement_arsenal: pd.DataFrame,
    hitter_pitcher_exclusions: pd.DataFrame,
) -> None:
    if duckdb is None:
        raise RuntimeError("duckdb is required to build the local explorer database.")
    conn = duckdb.connect(str(config.db_path))
    conn.register("hitter_metrics_df", hitter_metrics)
    conn.register("pitcher_metrics_df", pitcher_metrics)
    conn.register("pitcher_summary_by_hand_df", pitcher_summary_by_hand)
    conn.register("pitcher_arsenal_df", pitcher_arsenal)
    conn.register("pitcher_arsenal_by_hand_df", pitcher_arsenal_by_hand)
    conn.register("pitcher_usage_by_count_df", pitcher_usage_by_count)
    conn.register("hitter_rolling_df", hitter_rolling)
    conn.register("pitcher_rolling_df", pitcher_rolling)
    conn.register("batter_zone_profiles_df", batter_zone_profiles)
    conn.register("pitcher_zone_profiles_df", pitcher_zone_profiles)
    conn.register("batter_family_zone_profiles_df", batter_family_zone_profiles)
    conn.register("pitcher_family_zone_context_df", pitcher_family_zone_context)
    conn.register("pitcher_movement_arsenal_df", pitcher_movement_arsenal)
    conn.register("hitter_pitcher_exclusions_df", hitter_pitcher_exclusions)
    conn.execute("CREATE OR REPLACE TABLE hitter_metrics AS SELECT * FROM hitter_metrics_df")
    conn.execute("CREATE OR REPLACE TABLE pitcher_metrics AS SELECT * FROM pitcher_metrics_df")
    conn.execute("CREATE OR REPLACE TABLE pitcher_summary_by_hand AS SELECT * FROM pitcher_summary_by_hand_df")
    conn.execute("CREATE OR REPLACE TABLE pitcher_arsenal AS SELECT * FROM pitcher_arsenal_df")
    conn.execute("CREATE OR REPLACE TABLE pitcher_arsenal_by_hand AS SELECT * FROM pitcher_arsenal_by_hand_df")
    conn.execute("CREATE OR REPLACE TABLE pitcher_usage_by_count AS SELECT * FROM pitcher_usage_by_count_df")
    conn.execute("CREATE OR REPLACE TABLE hitter_rolling AS SELECT * FROM hitter_rolling_df")
    conn.execute("CREATE OR REPLACE TABLE pitcher_rolling AS SELECT * FROM pitcher_rolling_df")
    conn.execute("CREATE OR REPLACE TABLE batter_zone_profiles AS SELECT * FROM batter_zone_profiles_df")
    conn.execute("CREATE OR REPLACE TABLE pitcher_zone_profiles AS SELECT * FROM pitcher_zone_profiles_df")
    conn.execute("CREATE OR REPLACE TABLE batter_family_zone_profiles AS SELECT * FROM batter_family_zone_profiles_df")
    conn.execute("CREATE OR REPLACE TABLE pitcher_family_zone_context AS SELECT * FROM pitcher_family_zone_context_df")
    conn.execute("CREATE OR REPLACE TABLE pitcher_movement_arsenal AS SELECT * FROM pitcher_movement_arsenal_df")
    conn.execute("CREATE OR REPLACE TABLE hitter_pitcher_exclusions AS SELECT * FROM hitter_pitcher_exclusions_df")
    conn.close()


def _write_reusable_artifacts(
    config: AppConfig,
    hitter_metrics: pd.DataFrame,
    pitcher_metrics: pd.DataFrame,
    pitcher_summary_by_hand: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    pitcher_arsenal_by_hand: pd.DataFrame,
    pitcher_usage_by_count: pd.DataFrame,
    hitter_rolling: pd.DataFrame,
    pitcher_rolling: pd.DataFrame,
    batter_zone_profiles: pd.DataFrame,
    pitcher_zone_profiles: pd.DataFrame,
    batter_family_zone_profiles: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    pitcher_movement_arsenal: pd.DataFrame,
    hitter_pitcher_exclusions: pd.DataFrame,
) -> None:
    hitter_metrics.to_parquet(config.reusable_dir / "hitter_metrics.parquet", index=False)
    pitcher_metrics.to_parquet(config.reusable_dir / "pitcher_metrics.parquet", index=False)
    pitcher_summary_by_hand.to_parquet(config.reusable_dir / "pitcher_summary_by_hand.parquet", index=False)
    pitcher_arsenal.to_parquet(config.reusable_dir / "pitcher_arsenal.parquet", index=False)
    pitcher_arsenal_by_hand.to_parquet(config.reusable_dir / "pitcher_arsenal_by_hand.parquet", index=False)
    pitcher_usage_by_count.to_parquet(config.reusable_dir / "pitcher_usage_by_count.parquet", index=False)
    hitter_rolling.to_parquet(config.reusable_dir / "hitter_rolling.parquet", index=False)
    pitcher_rolling.to_parquet(config.reusable_dir / "pitcher_rolling.parquet", index=False)
    batter_zone_profiles.to_parquet(config.reusable_dir / "batter_zone_profiles.parquet", index=False)
    pitcher_zone_profiles.to_parquet(config.reusable_dir / "pitcher_zone_profiles.parquet", index=False)
    batter_family_zone_profiles.to_parquet(config.reusable_dir / "batter_family_zone_profiles.parquet", index=False)
    pitcher_family_zone_context.to_parquet(config.reusable_dir / "pitcher_family_zone_context.parquet", index=False)
    pitcher_movement_arsenal.to_parquet(config.reusable_dir / "pitcher_movement_arsenal.parquet", index=False)
    hitter_pitcher_exclusions.to_parquet(config.reusable_dir / "hitter_pitcher_exclusions.parquet", index=False)


def _pitcher_lookup(pitcher_metrics: pd.DataFrame) -> dict[int, dict[str, object]]:
    if pitcher_metrics.empty:
        return {}
    work = pitcher_metrics.sort_values(["recent_window", "weighted_mode", "split_key"], na_position="last").drop_duplicates("pitcher_id")
    lookup: dict[int, dict[str, object]] = {}
    for _, row in work.iterrows():
        pitcher_id = row.get("pitcher_id")
        if pd.isna(pitcher_id):
            continue
        lookup[int(pitcher_id)] = {
            "pitcher_name": row.get("pitcher_name"),
            "p_throws": row.get("p_throws"),
        }
    return lookup


def _base_hitter_pool(
    hitter_metrics: pd.DataFrame,
    rosters_frame: pd.DataFrame,
    team: str,
    split_key: str,
    recent_window: str,
    weighted_mode: str,
) -> pd.DataFrame:
    lookup = rosters_frame.loc[rosters_frame["team"] == team, ["player_id", "player_name"]].drop_duplicates("player_id") if not rosters_frame.empty else pd.DataFrame()
    if lookup.empty:
        return pd.DataFrame()
    frame = hitter_metrics.loc[
        hitter_metrics["batter"].isin(lookup["player_id"])
        & (hitter_metrics["split_key"] == split_key)
        & (hitter_metrics["recent_window"] == recent_window)
        & (hitter_metrics["weighted_mode"] == weighted_mode)
    ].copy()
    if frame.empty:
        return frame
    enriched = frame.merge(lookup, left_on="batter", right_on="player_id", how="inner")
    enriched["team"] = team
    enriched["hitter_name"] = enriched["player_name"].fillna(enriched["hitter_name"])
    return enriched.drop(columns=["player_id", "player_name"], errors="ignore")


def _build_hitter_tracking_snapshots(
    target_date: date,
    schedule: list[dict],
    rosters_frame: pd.DataFrame,
    hitter_metrics: pd.DataFrame,
    pitcher_metrics: pd.DataFrame,
    batter_zone_profiles: pd.DataFrame,
    pitcher_zone_profiles: pd.DataFrame,
) -> pd.DataFrame:
    if not schedule or hitter_metrics.empty:
        return pd.DataFrame()
    pitcher_lookup = _pitcher_lookup(pitcher_metrics)
    rows: list[pd.DataFrame] = []
    build_date_value = pd.Timestamp.utcnow().date()
    for game in schedule:
        game_pk = game["game_pk"]
        game_label = f"{game['away_team']} @ {game['home_team']}"
        home_pitcher_id = game.get("home_probable_pitcher_id")
        away_pitcher_id = game.get("away_probable_pitcher_id")
        home_pitcher = pitcher_lookup.get(int(home_pitcher_id), {}) if home_pitcher_id else {}
        away_pitcher = pitcher_lookup.get(int(away_pitcher_id), {}) if away_pitcher_id else {}
        away_pitcher_hand = home_pitcher.get("p_throws")
        home_pitcher_hand = away_pitcher.get("p_throws")
        for split_key in DEFAULT_SPLITS:
            for recent_window in DEFAULT_RECENT_WINDOWS:
                for weighted_mode in ("weighted", "unweighted"):
                    away_frame = _base_hitter_pool(hitter_metrics, rosters_frame, game["away_team"], split_key, recent_window, weighted_mode)
                    home_frame = _base_hitter_pool(hitter_metrics, rosters_frame, game["home_team"], split_key, recent_window, weighted_mode)
                    away_frame = add_hitter_matchup_score(
                        away_frame,
                        batter_zone_profiles=batter_zone_profiles,
                        pitcher_zone_profiles=pitcher_zone_profiles,
                        opposing_pitcher_id=home_pitcher_id,
                        opposing_pitcher_hand=away_pitcher_hand,
                    )
                    home_frame = add_hitter_matchup_score(
                        home_frame,
                        batter_zone_profiles=batter_zone_profiles,
                        pitcher_zone_profiles=pitcher_zone_profiles,
                        opposing_pitcher_id=away_pitcher_id,
                        opposing_pitcher_hand=home_pitcher_hand,
                    )
                    for frame, opponent, opposing_id, opposing_name, opposing_hand in (
                        (away_frame, game["home_team"], home_pitcher_id, home_pitcher.get("pitcher_name"), away_pitcher_hand),
                        (home_frame, game["away_team"], away_pitcher_id, away_pitcher.get("pitcher_name"), home_pitcher_hand),
                    ):
                        if frame.empty:
                            continue
                        work = frame.copy()
                        work["build_date"] = build_date_value
                        work["slate_date"] = target_date
                        work["game_pk"] = game_pk
                        work["game_label"] = game_label
                        work["opponent"] = opponent
                        work["opposing_pitcher_id"] = opposing_id
                        work["opposing_pitcher_name"] = opposing_name
                        work["opposing_pitcher_hand"] = opposing_hand
                        work["snapshot_id"] = work.apply(
                            lambda row: str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_URL,
                                    "|".join(
                                        [
                                            str(target_date),
                                            str(game_pk),
                                            str(int(row["batter"])) if pd.notna(row["batter"]) else "",
                                            str(row["split_key"]),
                                            str(row["recent_window"]),
                                            str(row["weighted_mode"]),
                                        ]
                                    ),
                                )
                            ),
                            axis=1,
                        )
                        rows.append(
                            work[
                                [
                                    "snapshot_id",
                                    "build_date",
                                    "slate_date",
                                    "game_pk",
                                    "game_label",
                                    "team",
                                    "opponent",
                                    "batter",
                                    "hitter_name",
                                    "opposing_pitcher_id",
                                    "opposing_pitcher_name",
                                    "opposing_pitcher_hand",
                                    "split_key",
                                    "recent_window",
                                    "weighted_mode",
                                    "matchup_score",
                                    "ceiling_score",
                                    "zone_fit_score",
                                    "likely_starter_score",
                                    "xwoba",
                                    "xwoba_con",
                                    "swstr_pct",
                                    "pulled_barrel_pct",
                                    "barrel_bbe_pct",
                                    "barrel_bip_pct",
                                    "sweet_spot_pct",
                                    "fb_pct",
                                    "hard_hit_pct",
                                    "avg_launch_angle",
                                    "pitch_count",
                                    "bip",
                                ]
                            ].rename(columns={"batter": "batter_id"})
                        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_hitter_board_winners(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["slate_date", "game_pk", "batter_id", "hitter_name", "team", "board_name", "board_rank", "board_score", "source_metric"])
    canonical = snapshots.loc[
        (snapshots["split_key"] == "overall")
        & (snapshots["recent_window"] == "season")
        & (snapshots["weighted_mode"] == "weighted")
    ].copy()
    if canonical.empty:
        return pd.DataFrame(columns=["slate_date", "game_pk", "batter_id", "hitter_name", "team", "board_name", "board_rank", "board_score", "source_metric"])

    def _rank_board(frame: pd.DataFrame, board_name: str, score_column: str, depth: int = 10) -> pd.DataFrame:
        ranked = frame.sort_values([score_column, "xwoba"], ascending=[False, False], na_position="last").head(depth).copy()
        ranked["board_name"] = board_name
        ranked["board_rank"] = range(1, len(ranked) + 1)
        ranked["board_score"] = ranked[score_column]
        ranked["source_metric"] = score_column
        return ranked[["slate_date", "game_pk", "batter_id", "hitter_name", "team", "board_name", "board_rank", "board_score", "source_metric"]]

    best_candidates: list[pd.DataFrame] = []
    for _, game_frame in canonical.groupby("game_pk", sort=False):
        best = game_frame.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last").head(3).copy()
        if not best.empty:
            best_candidates.append(best)
    best_source = pd.concat(best_candidates, ignore_index=True) if best_candidates else pd.DataFrame(columns=canonical.columns)
    boards = [
        _rank_board(canonical, "top_slate_matchup", "matchup_score"),
        _rank_board(canonical, "ceiling", "ceiling_score"),
    ]
    if not best_source.empty:
        boards.append(_rank_board(best_source, "best_matchups", "matchup_score"))
    return pd.concat(boards, ignore_index=True) if boards else pd.DataFrame(columns=["slate_date", "game_pk", "batter_id", "hitter_name", "team", "board_name", "board_rank", "board_score", "source_metric"])


def _build_hitter_game_outcomes(
    target_date: date,
    raw_statcast: pd.DataFrame,
    snapshots: pd.DataFrame,
    source_max_event_date: date | None,
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["slate_date", "game_pk", "team", "batter_id", "hitter_name", "had_plate_appearance", "started", "plate_appearances", "hits", "home_runs", "total_bases", "runs", "rbi", "walks", "strikeouts", "outcome_complete", "outcome_status", "source_max_event_date", "last_updated_at"])
    snapshot_players = snapshots[["slate_date", "game_pk", "team", "batter_id", "hitter_name"]].drop_duplicates(["slate_date", "game_pk", "batter_id"]).copy()
    day_frame = raw_statcast.loc[pd.to_datetime(raw_statcast["game_date"]).dt.date == target_date].copy()
    if day_frame.empty:
        status = OUTCOME_STATUS_SOURCE_LAG if source_max_event_date is not None and source_max_event_date < target_date else OUTCOME_STATUS_SOURCE_EMPTY
        empty = _empty_hitter_outcome_frame(snapshot_players, target_date, status, source_max_event_date)
        return empty
    day_frame = day_frame.loc[day_frame["game_pk"].isin(snapshots["game_pk"].unique()) & day_frame["batter"].isin(snapshots["batter_id"].unique())].copy()
    if day_frame.empty:
        empty = _empty_hitter_outcome_frame(snapshot_players, target_date, OUTCOME_STATUS_SOURCE_MISSING_ROWS, source_max_event_date)
        return empty

    pa_events = (
        day_frame.sort_values(["game_pk", "batter", "at_bat_number", "pitch_number"])
        .groupby(["game_pk", "team", "batter", "at_bat_number"], as_index=False)
        .tail(1)
        .copy()
    )
    event_text = pa_events["events"].fillna("").astype(str).str.lower()
    pa_events["is_hit"] = event_text.isin({"single", "double", "triple", "home_run"}).astype(int)
    pa_events["is_home_run"] = event_text.eq("home_run").astype(int)
    pa_events["total_bases_value"] = event_text.map({"single": 1, "double": 2, "triple": 3, "home_run": 4}).fillna(0).astype(int)
    pa_events["is_walk"] = event_text.isin({"walk", "intent_walk", "intentional_walk"}).astype(int)
    pa_events["is_strikeout"] = event_text.isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa_events["rbi_value"] = (
        pd.to_numeric(pa_events.get("post_bat_score"), errors="coerce").fillna(0)
        - pd.to_numeric(pa_events.get("bat_score"), errors="coerce").fillna(0)
    ).clip(lower=0).astype(int)
    pa_events["run_value"] = pa_events["is_home_run"]

    outcomes = (
        pa_events.groupby(["game_pk", "team", "batter"], as_index=False)
        .agg(
            plate_appearances=("at_bat_number", "nunique"),
            hits=("is_hit", "sum"),
            home_runs=("is_home_run", "sum"),
            total_bases=("total_bases_value", "sum"),
            runs=("run_value", "sum"),
            rbi=("rbi_value", "sum"),
            walks=("is_walk", "sum"),
            strikeouts=("is_strikeout", "sum"),
        )
    )
    names = snapshots[["game_pk", "batter_id", "hitter_name"]].drop_duplicates(["game_pk", "batter_id"]).rename(columns={"batter_id": "batter"})
    outcomes = outcomes.merge(names, on=["game_pk", "batter"], how="left")
    outcomes["slate_date"] = target_date
    outcomes["had_plate_appearance"] = outcomes["plate_appearances"].gt(0)
    outcomes["started"] = outcomes["had_plate_appearance"]
    outcomes["outcome_complete"] = True
    outcomes["outcome_status"] = OUTCOME_STATUS_COMPLETE
    outcomes["source_max_event_date"] = source_max_event_date
    outcomes["last_updated_at"] = datetime.now(UTC)
    outcomes = outcomes.rename(columns={"batter": "batter_id"})
    outcomes = outcomes[
        [
            "slate_date",
            "game_pk",
            "team",
            "batter_id",
            "hitter_name",
            "had_plate_appearance",
            "started",
            "plate_appearances",
            "hits",
            "home_runs",
            "total_bases",
            "runs",
            "rbi",
            "walks",
            "strikeouts",
            "last_updated_at",
        ]
    ]
    full_outcomes = snapshot_players.merge(
        outcomes,
        on=["slate_date", "game_pk", "team", "batter_id", "hitter_name"],
        how="left",
    )
    for column in ["plate_appearances", "hits", "home_runs", "total_bases", "runs", "rbi", "walks", "strikeouts"]:
        full_outcomes[column] = pd.to_numeric(full_outcomes[column], errors="coerce").fillna(0).astype(int)
    full_outcomes["had_plate_appearance"] = full_outcomes["had_plate_appearance"].fillna(False)
    full_outcomes["started"] = full_outcomes["started"].fillna(False)
    full_outcomes["outcome_complete"] = full_outcomes["outcome_complete"].fillna(True)
    full_outcomes["outcome_status"] = full_outcomes["outcome_status"].fillna(OUTCOME_STATUS_COMPLETE)
    full_outcomes["source_max_event_date"] = full_outcomes["source_max_event_date"].fillna(source_max_event_date)
    full_outcomes["last_updated_at"] = full_outcomes["last_updated_at"].fillna(datetime.now(UTC))
    return full_outcomes[
        [
            "slate_date",
            "game_pk",
            "team",
            "batter_id",
            "hitter_name",
            "had_plate_appearance",
            "started",
            "plate_appearances",
            "hits",
            "home_runs",
            "total_bases",
            "runs",
            "rbi",
            "walks",
            "strikeouts",
            "outcome_complete",
            "outcome_status",
            "source_max_event_date",
            "last_updated_at",
        ]
    ]


def _build_probable_pitcher_lookup(schedule: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for game in schedule:
        game_pk = game["game_pk"]
        game_label = f"{game['away_team']} @ {game['home_team']}"
        away_id = game.get("away_probable_pitcher_id")
        home_id = game.get("home_probable_pitcher_id")
        if away_id:
            rows.append(
                {
                    "slate_date": game.get("game_date"),
                    "game_pk": game_pk,
                    "game_label": game_label,
                    "pitcher_id": int(away_id),
                    "team": game["away_team"],
                    "opponent": game["home_team"],
                }
            )
        if home_id:
            rows.append(
                {
                    "slate_date": game.get("game_date"),
                    "game_pk": game_pk,
                    "game_label": game_label,
                    "pitcher_id": int(home_id),
                    "team": game["home_team"],
                    "opponent": game["away_team"],
                }
            )
    return pd.DataFrame(rows)


def _build_pitcher_opponent_hitter_map(
    schedule: list[dict],
    rosters_frame: pd.DataFrame,
    hitter_metrics: pd.DataFrame,
    pitcher_metrics: pd.DataFrame,
    rotowire_lineups: dict[str, dict[str, object]] | None,
) -> dict[tuple[object, object, object, object, object], pd.DataFrame]:
    if not schedule or hitter_metrics.empty or pitcher_metrics.empty:
        return {}
    opponent_hitters_by_key: dict[tuple[object, object, object, object, object], pd.DataFrame] = {}
    for game in schedule:
        home_pitcher_id = game.get("home_probable_pitcher_id")
        away_pitcher_id = game.get("away_probable_pitcher_id")
        for split_key in DEFAULT_SPLITS:
            for recent_window in DEFAULT_RECENT_WINDOWS:
                for weighted_mode in ("weighted", "unweighted"):
                    away_hitters = _base_hitter_pool(hitter_metrics, rosters_frame, game["away_team"], split_key, recent_window, weighted_mode)
                    home_hitters = _base_hitter_pool(hitter_metrics, rosters_frame, game["home_team"], split_key, recent_window, weighted_mode)
                    away_hitters = apply_projected_lineup(away_hitters, game["away_team"], rotowire_lineups)
                    home_hitters = apply_projected_lineup(home_hitters, game["home_team"], rotowire_lineups)
                    if away_pitcher_id:
                        opponent_hitters_by_key[
                            build_pitcher_matchup_key(game["game_pk"], away_pitcher_id, split_key, recent_window, weighted_mode)
                        ] = home_hitters.copy()
                    if home_pitcher_id:
                        opponent_hitters_by_key[
                            build_pitcher_matchup_key(game["game_pk"], home_pitcher_id, split_key, recent_window, weighted_mode)
                        ] = away_hitters.copy()
    return opponent_hitters_by_key


def _build_pitcher_tracking_snapshots(
    target_date: date,
    probable_pitchers: pd.DataFrame,
    pitcher_metrics: pd.DataFrame,
    opponent_hitters_by_key: dict[tuple[object, object, object, object, object], pd.DataFrame] | None = None,
    batter_family_zone_profiles: pd.DataFrame | None = None,
    pitcher_family_zone_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if probable_pitchers.empty or pitcher_metrics.empty:
        return pd.DataFrame()
    probable = probable_pitchers.drop_duplicates(["game_pk", "pitcher_id"]).copy()
    probable_ids = probable["pitcher_id"].dropna().astype(int).unique()
    candidates = pitcher_metrics.loc[pitcher_metrics["pitcher_id"].isin(probable_ids)].copy()
    if candidates.empty or probable.empty:
        return pd.DataFrame()
    starter_rows = probable.merge(candidates, on="pitcher_id", how="inner")
    if starter_rows.empty:
        return pd.DataFrame()
    starter_rows = add_pitcher_rank_score(
        starter_rows,
        opponent_hitters_by_key=opponent_hitters_by_key,
        batter_family_zone_profiles=batter_family_zone_profiles,
        pitcher_family_zone_context=pitcher_family_zone_context,
    )
    build_date_value = pd.Timestamp.utcnow().date()
    starter_rows["build_date"] = build_date_value
    starter_rows["slate_date"] = target_date
    starter_rows["snapshot_id"] = starter_rows.apply(
        lambda row: str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "|".join(
                    [
                        str(target_date),
                        str(int(row["game_pk"])),
                        str(int(row["pitcher_id"])),
                        str(row["split_key"]),
                        str(row["recent_window"]),
                        str(row["weighted_mode"]),
                    ]
                ),
            )
        ),
        axis=1,
    )
    return starter_rows[
        [
            "snapshot_id",
            "build_date",
            "slate_date",
            "game_pk",
            "game_label",
            "team",
            "opponent",
            "pitcher_id",
            "pitcher_name",
            "p_throws",
            "split_key",
            "recent_window",
            "weighted_mode",
            "pitcher_score",
            "strikeout_score",
            "raw_pitcher_score",
            "raw_strikeout_score",
            "pitcher_matchup_adjustment",
            "strikeout_matchup_adjustment",
            "opponent_lineup_quality",
            "opponent_contact_threat",
            "opponent_whiff_tendency",
            "opponent_family_fit_allowed",
            "lineup_source",
            "lineup_hitter_count",
            "xwoba",
            "called_strike_pct",
            "csw_pct",
            "swstr_pct",
            "putaway_pct",
            "ball_pct",
            "siera",
            "barrel_bbe_pct",
            "barrel_bip_pct",
            "pulled_barrel_pct",
            "fb_pct",
            "gb_pct",
            "gb_fb_ratio",
            "hard_hit_pct",
            "avg_launch_angle",
            "pitch_count",
            "bip",
        ]
    ].copy()


def _build_pitcher_arsenal_snapshots(
    target_date: date,
    probable_pitchers: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    pitcher_arsenal_by_hand: pd.DataFrame,
) -> pd.DataFrame:
    if probable_pitchers.empty:
        return pd.DataFrame()
    probable = probable_pitchers.drop_duplicates(["game_pk", "pitcher_id"]).copy()
    probable_ids = probable["pitcher_id"].dropna().astype(int).unique()
    rows: list[pd.DataFrame] = []
    all_frame = pitcher_arsenal.loc[pitcher_arsenal["pitcher_id"].isin(probable_ids)].copy()
    if not all_frame.empty:
        all_frame["batter_side_key"] = "all"
        rows.append(all_frame)
    by_hand = pitcher_arsenal_by_hand.loc[pitcher_arsenal_by_hand["pitcher_id"].isin(probable_ids)].copy()
    if not by_hand.empty:
        rows.append(by_hand)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    enriched = combined.merge(probable[["game_pk", "pitcher_id"]].drop_duplicates(), on="pitcher_id", how="inner")
    enriched["slate_date"] = target_date
    return enriched[
        [
            "slate_date",
            "game_pk",
            "pitcher_id",
            "pitcher_name",
            "split_key",
            "recent_window",
            "weighted_mode",
            "batter_side_key",
            "pitch_name",
            "usage_pct",
            "swstr_pct",
            "hard_hit_pct",
            "avg_release_speed",
            "avg_spin_rate",
            "xwoba_con",
        ]
    ].drop_duplicates(
        ["slate_date", "game_pk", "pitcher_id", "split_key", "recent_window", "weighted_mode", "batter_side_key", "pitch_name"]
    )


def _build_pitcher_count_snapshots(
    target_date: date,
    probable_pitchers: pd.DataFrame,
    pitcher_usage_by_count: pd.DataFrame,
) -> pd.DataFrame:
    if probable_pitchers.empty or pitcher_usage_by_count.empty:
        return pd.DataFrame()
    probable = probable_pitchers.drop_duplicates(["game_pk", "pitcher_id"]).copy()
    probable_ids = probable["pitcher_id"].dropna().astype(int).unique()
    combined = pitcher_usage_by_count.loc[pitcher_usage_by_count["pitcher_id"].isin(probable_ids)].copy()
    if combined.empty:
        return pd.DataFrame()
    enriched = combined.merge(probable[["game_pk", "pitcher_id"]].drop_duplicates(), on="pitcher_id", how="inner")
    enriched["slate_date"] = target_date
    return enriched[
        [
            "slate_date",
            "game_pk",
            "pitcher_id",
            "pitcher_name",
            "split_key",
            "recent_window",
            "weighted_mode",
            "batter_side_key",
            "pitch_name",
            "count_bucket",
            "usage_pct",
        ]
    ].drop_duplicates(
        ["slate_date", "game_pk", "pitcher_id", "split_key", "recent_window", "weighted_mode", "batter_side_key", "pitch_name", "count_bucket"]
    )


def _build_pitcher_board_winners(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["slate_date", "game_pk", "pitcher_id", "pitcher_name", "team", "board_name", "board_rank", "board_score", "source_metric"])
    canonical = snapshots.loc[
        (snapshots["split_key"] == "overall")
        & (snapshots["recent_window"] == "season")
        & (snapshots["weighted_mode"] == "weighted")
    ].copy()
    if canonical.empty:
        return pd.DataFrame(columns=["slate_date", "game_pk", "pitcher_id", "pitcher_name", "team", "board_name", "board_rank", "board_score", "source_metric"])
    ranked = canonical.sort_values(["pitcher_score", "xwoba"], ascending=[False, True], na_position="last").head(10).copy()
    ranked["board_name"] = "top_slate_pitchers"
    ranked["board_rank"] = range(1, len(ranked) + 1)
    ranked["board_score"] = ranked["pitcher_score"]
    ranked["source_metric"] = "pitcher_score"
    return ranked[["slate_date", "game_pk", "pitcher_id", "pitcher_name", "team", "board_name", "board_rank", "board_score", "source_metric"]]


def _build_pitcher_game_outcomes(
    target_date: date,
    raw_statcast: pd.DataFrame,
    snapshots: pd.DataFrame,
    source_max_event_date: date | None,
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["slate_date", "game_pk", "team", "pitcher_id", "pitcher_name", "had_pitch", "started", "outs_recorded", "batters_faced", "hits_allowed", "home_runs_allowed", "runs_allowed", "earned_runs", "walks", "strikeouts", "outcome_complete", "outcome_status", "source_max_event_date", "last_updated_at"])
    snapshot_pitchers = snapshots[["slate_date", "game_pk", "team", "pitcher_id", "pitcher_name"]].drop_duplicates(["slate_date", "game_pk", "pitcher_id"]).copy()
    day_frame = raw_statcast.loc[pd.to_datetime(raw_statcast["game_date"]).dt.date == target_date].copy()
    if day_frame.empty:
        status = OUTCOME_STATUS_SOURCE_LAG if source_max_event_date is not None and source_max_event_date < target_date else OUTCOME_STATUS_SOURCE_EMPTY
        empty = _empty_pitcher_outcome_frame(snapshot_pitchers, target_date, status, source_max_event_date)
        return empty
    day_frame = day_frame.loc[day_frame["game_pk"].isin(snapshots["game_pk"].unique()) & day_frame["pitcher"].isin(snapshots["pitcher_id"].unique())].copy()
    if day_frame.empty:
        empty = _empty_pitcher_outcome_frame(snapshot_pitchers, target_date, OUTCOME_STATUS_SOURCE_MISSING_ROWS, source_max_event_date)
        return empty
    pa_events = (
        day_frame.sort_values(["game_pk", "pitcher", "at_bat_number", "pitch_number"])
        .groupby(["game_pk", "fielding_team", "pitcher", "at_bat_number"], as_index=False)
        .tail(1)
        .copy()
    )
    event_text = pa_events["events"].fillna("").astype(str).str.lower()
    pa_events["is_hit_allowed"] = event_text.isin({"single", "double", "triple", "home_run"}).astype(int)
    pa_events["is_home_run_allowed"] = event_text.eq("home_run").astype(int)
    pa_events["is_walk"] = event_text.isin({"walk", "intent_walk", "intentional_walk"}).astype(int)
    pa_events["is_strikeout"] = event_text.isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa_events["runs_allowed_delta"] = (
        pd.to_numeric(pa_events.get("post_bat_score"), errors="coerce")
        - pd.to_numeric(pa_events.get("bat_score"), errors="coerce")
    )
    outcomes = (
        pa_events.groupby(["game_pk", "fielding_team", "pitcher"], as_index=False)
        .agg(
            batters_faced=("at_bat_number", "nunique"),
            hits_allowed=("is_hit_allowed", "sum"),
            home_runs_allowed=("is_home_run_allowed", "sum"),
            walks=("is_walk", "sum"),
            strikeouts=("is_strikeout", "sum"),
            runs_allowed=("runs_allowed_delta", lambda s: pd.to_numeric(s, errors="coerce").clip(lower=0).sum() if s.notna().any() else None),
        )
    )
    names = snapshots[["game_pk", "pitcher_id", "pitcher_name"]].drop_duplicates(["game_pk", "pitcher_id"]).rename(columns={"pitcher_id": "pitcher"})
    outcomes = outcomes.merge(names, on=["game_pk", "pitcher"], how="left")
    outcomes["slate_date"] = target_date
    outcomes["had_pitch"] = outcomes["batters_faced"].gt(0)
    outcomes["started"] = outcomes["had_pitch"]
    outcomes["outs_recorded"] = None
    outcomes["earned_runs"] = None
    outcomes["outcome_complete"] = True
    outcomes["outcome_status"] = OUTCOME_STATUS_COMPLETE
    outcomes["source_max_event_date"] = source_max_event_date
    outcomes["last_updated_at"] = datetime.now(UTC)
    outcomes = outcomes.rename(columns={"pitcher": "pitcher_id", "fielding_team": "team"})
    outcomes = outcomes[
        [
            "slate_date",
            "game_pk",
            "team",
            "pitcher_id",
            "pitcher_name",
            "had_pitch",
            "started",
            "outs_recorded",
            "batters_faced",
            "hits_allowed",
            "home_runs_allowed",
            "runs_allowed",
            "earned_runs",
            "walks",
            "strikeouts",
            "outcome_complete",
            "outcome_status",
            "source_max_event_date",
            "last_updated_at",
        ]
    ]
    full_outcomes = snapshot_pitchers.merge(
        outcomes,
        on=["slate_date", "game_pk", "team", "pitcher_id", "pitcher_name"],
        how="left",
    )
    for column in ["batters_faced", "hits_allowed", "home_runs_allowed", "walks", "strikeouts"]:
        full_outcomes[column] = pd.to_numeric(full_outcomes[column], errors="coerce").fillna(0).astype(int)
    full_outcomes["had_pitch"] = full_outcomes["had_pitch"].fillna(False)
    full_outcomes["started"] = full_outcomes["started"].fillna(False)
    full_outcomes["outcome_complete"] = full_outcomes["outcome_complete"].fillna(True)
    full_outcomes["outcome_status"] = full_outcomes["outcome_status"].fillna(OUTCOME_STATUS_COMPLETE)
    full_outcomes["source_max_event_date"] = full_outcomes["source_max_event_date"].fillna(source_max_event_date)
    full_outcomes["last_updated_at"] = full_outcomes["last_updated_at"].fillna(datetime.now(UTC))
    return full_outcomes[
        [
            "slate_date",
            "game_pk",
            "team",
            "pitcher_id",
            "pitcher_name",
            "had_pitch",
            "started",
            "outs_recorded",
            "batters_faced",
            "hits_allowed",
            "home_runs_allowed",
            "runs_allowed",
            "earned_runs",
            "walks",
            "strikeouts",
            "outcome_complete",
            "outcome_status",
            "source_max_event_date",
            "last_updated_at",
        ]
    ]


def run_build(context: BuildContext) -> None:
    ensure_directories(context.config)
    csv_paths = _csv_glob(context.csv_dir)
    historical_statcast = _load_raw_statcast(csv_paths)
    live_payload = load_cockroach_payload(context.config)
    raw_statcast = _merge_historical_and_live(historical_statcast, live_payload)
    hitter_pitcher_exclusions = _build_hitter_pitcher_exclusions(raw_statcast)
    batter_zone_profiles, pitcher_zone_profiles = _aggregate_zone_profiles(raw_statcast, context.config.zone_year_weights)
    hitter_metrics, pitcher_metrics, pitcher_summary_by_hand, pitcher_arsenal, pitcher_arsenal_by_hand, pitcher_usage_by_count = build_metric_tables(raw_statcast, context.config)
    pitcher_family_zone_context, pitcher_movement_arsenal, batter_family_zone_profiles, pitch_context_source = build_movement_context_tables(live_payload, context.config)
    _write_duckdb(
        context.config,
        hitter_metrics,
        pitcher_metrics,
        pitcher_summary_by_hand,
        pitcher_arsenal,
        pitcher_arsenal_by_hand,
        pitcher_usage_by_count,
        live_payload.hitter_rolling,
        live_payload.pitcher_rolling,
        batter_zone_profiles,
        pitcher_zone_profiles,
        batter_family_zone_profiles,
        pitcher_family_zone_context,
        pitcher_movement_arsenal,
        hitter_pitcher_exclusions,
    )
    _write_reusable_artifacts(
        context.config,
        hitter_metrics,
        pitcher_metrics,
        pitcher_summary_by_hand,
        pitcher_arsenal,
        pitcher_arsenal_by_hand,
        pitcher_usage_by_count,
        live_payload.hitter_rolling,
        live_payload.pitcher_rolling,
        batter_zone_profiles,
        pitcher_zone_profiles,
        batter_family_zone_profiles,
        pitcher_family_zone_context,
        pitcher_movement_arsenal,
        hitter_pitcher_exclusions,
    )
    schedule = fetch_schedule(context.target_date)
    rosters = fetch_team_rosters_for_schedule(schedule, context.target_date)
    valid_teams = tuple(sorted({game["away_team"] for game in schedule} | {game["home_team"] for game in schedule}))
    rosters_frame = pd.DataFrame(rosters) if rosters else pd.DataFrame(columns=["team", "player_id", "player_name"])
    try:
        rotowire_lineups = resolve_rotowire_lineups(fetch_rotowire_lineups(context.target_date, valid_teams), rosters_frame)
    except Exception:
        rotowire_lineups = {}
    source_max_event_date = _live_source_max_event_date(live_payload)
    tracking_health = _build_live_tracking_health(
        context.target_date,
        schedule,
        live_payload,
        context.config.cockroach_live_pitch_mix_table,
    )
    if source_max_event_date is None or source_max_event_date < context.target_date:
        warnings.warn(
            f"Live event source {context.config.cockroach_live_pitch_mix_table} is stale for {context.target_date.isoformat()}. "
            f"Latest available event date is {source_max_event_date.isoformat() if source_max_event_date else 'none'}; "
            "tracked outcomes will be marked incomplete and excluded from backtesting."
        )
    pitch_shape_diagnostics = _pitch_shape_diagnostics(
        schedule,
        live_payload,
        pitch_context_source,
        pitcher_family_zone_context,
        pitcher_movement_arsenal,
    )
    missing_pitch_shape = pitch_shape_diagnostics.loc[
        pitch_shape_diagnostics["movement_rows"].eq(0) | pitch_shape_diagnostics["family_rows"].eq(0)
    ].copy()
    if not missing_pitch_shape.empty:
        warning_lines = [
            f"{row['pitcher_name']} ({int(row['pitcher_id'])}) -> {row['root_causes']}"
            for _, row in missing_pitch_shape.iterrows()
        ]
        warnings.warn(
            "Pitch shape context gaps detected for probable starters:\n" + "\n".join(warning_lines)
        )
    if context.config.odds_api_key:
        try:
            props_payload = load_live_props_board(context.config, context.target_date, rosters_frame)
            if isinstance(props_payload, PropsBoardPayload) and not props_payload.raw_rows.empty:
                write_props_odds_snapshot(context.config, props_payload.raw_rows)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Unable to capture live odds during build for {context.target_date.isoformat()}: {exc}")
    snapshots = _build_hitter_tracking_snapshots(
        context.target_date,
        schedule,
        rosters_frame,
        hitter_metrics,
        pitcher_metrics,
        batter_zone_profiles,
        pitcher_zone_profiles,
    )
    board_winners = _build_hitter_board_winners(snapshots)
    outcomes = _build_hitter_game_outcomes(context.target_date, raw_statcast, snapshots, source_max_event_date)
    probable_pitchers = _build_probable_pitcher_lookup(schedule)
    pitcher_opponent_hitters = _build_pitcher_opponent_hitter_map(
        schedule,
        rosters_frame,
        hitter_metrics,
        pitcher_metrics,
        rotowire_lineups,
    )
    pitcher_snapshots = _build_pitcher_tracking_snapshots(
        context.target_date,
        probable_pitchers,
        pitcher_metrics,
        opponent_hitters_by_key=pitcher_opponent_hitters,
        batter_family_zone_profiles=batter_family_zone_profiles,
        pitcher_family_zone_context=pitcher_family_zone_context,
    )
    pitcher_arsenal_snapshots = _build_pitcher_arsenal_snapshots(context.target_date, probable_pitchers, pitcher_arsenal, pitcher_arsenal_by_hand)
    pitcher_count_snapshots = _build_pitcher_count_snapshots(context.target_date, probable_pitchers, pitcher_usage_by_count)
    pitcher_board_winners = _build_pitcher_board_winners(pitcher_snapshots)
    pitcher_outcomes = _build_pitcher_game_outcomes(context.target_date, raw_statcast, pitcher_snapshots, source_max_event_date)
    write_tracking_payload(
        context.config,
        snapshots,
        outcomes,
        board_winners,
        pitcher_snapshots=pitcher_snapshots,
        pitcher_outcomes=pitcher_outcomes,
        pitcher_board_winners=pitcher_board_winners,
        pitcher_arsenal_snapshots=pitcher_arsenal_snapshots,
        pitcher_count_snapshots=pitcher_count_snapshots,
    )
    _save_daily_files(
        context,
        schedule,
        rosters,
        hitter_metrics,
        pitcher_metrics,
        pitcher_summary_by_hand,
        pitcher_arsenal,
        pitcher_arsenal_by_hand,
        pitcher_usage_by_count,
        live_payload.hitter_rolling,
        live_payload.pitcher_rolling,
        batter_zone_profiles,
        pitcher_zone_profiles,
        batter_family_zone_profiles,
        pitcher_family_zone_context,
        pitcher_movement_arsenal,
        hitter_pitcher_exclusions,
        pitch_shape_diagnostics,
        tracking_health,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local DuckDB and hosted artifacts from Statcast CSVs.")
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--target-date", type=lambda value: date.fromisoformat(value), default=date.today())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig(csv_dir=args.csv_dir, db_path=args.db_path, artifacts_dir=args.artifacts_dir)
    context = BuildContext(config=config, target_date=args.target_date, csv_dir=args.csv_dir)
    run_build(context)


if __name__ == "__main__":
    main()
