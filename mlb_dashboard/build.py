from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .cockroach_loader import CockroachPayload, load_cockroach_payload
from .config import AppConfig, DEFAULT_PITCH_GROUPS, DEFAULT_RECENT_WINDOWS, DEFAULT_SPLITS, ensure_directories
from .metrics import add_metric_flags, apply_year_weights, likely_starter_scores
from .mlb_api import fetch_schedule, fetch_team_rosters_for_schedule

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
]


@dataclass
class BuildContext:
    config: AppConfig
    target_date: date
    csv_dir: Path


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
    work["metric_weight"] = apply_year_weights(work, year_weights) if weighted_mode == "weighted" else 1.0
    rows: list[dict] = []
    for (team, batter), group in work.groupby(["team", "batter"], sort=False):
        hitter_side = group["stand"].dropna().astype(str).value_counts().idxmax() if group["stand"].notna().any() else None
        pitch_count = len(group)
        bip = int(group["is_batted_ball"].sum())
        pitch_weight_sum = float(group["metric_weight"].sum())
        bip_weight_sum = float(group.loc[group["is_batted_ball"], "metric_weight"].sum())
        tracked_bbe_weight_sum = float(group.loc[group["is_tracked_bbe"], "metric_weight"].sum())
        barrel_weight_sum = float((group["is_barrel"].astype(int) * group["metric_weight"]).sum())
        pulled_barrel_weight_sum = float((group["is_pulled_barrel"].astype(int) * group["metric_weight"]).sum())
        rows.append(
            {
                "team": team,
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
        rows.append(
            {
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher_name,
                "p_throws": throws,
                "pitch_count": pitch_count,
                "bip": bip,
                "xwoba": _weighted_sum(group, "xwoba_value", group.index) / max(_weighted_denominator(group, "xwoba_value", group.index), 1e-9),
                "swstr_pct": float((group["is_swinging_strike"].astype(int) * group["metric_weight"]).sum()) / max(pitch_weight_sum, 1e-9),
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


def _append_split_metadata(frame: pd.DataFrame, split_key: str, recent_window: str, weighted_mode: str) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["split_key"] = split_key
    enriched["recent_window"] = recent_window
    enriched["weighted_mode"] = weighted_mode
    return enriched


def build_metric_tables(raw_statcast: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hitters: list[pd.DataFrame] = []
    pitchers: list[pd.DataFrame] = []
    pitcher_summaries_by_hand: list[pd.DataFrame] = []
    arsenals: list[pd.DataFrame] = []
    arsenals_by_hand: list[pd.DataFrame] = []
    usages_by_count: list[pd.DataFrame] = []
    max_date = raw_statcast["game_date"].max()
    starter_scores = likely_starter_scores(raw_statcast[["team", "batter", "game_pk", "at_bat_number", "game_date"]])
    for recent_window in DEFAULT_RECENT_WINDOWS:
        recent_cutoff = _window_cutoff(max_date, recent_window)
        window_frame = raw_statcast.loc[raw_statcast["game_date"] >= recent_cutoff]
        for split_key in DEFAULT_SPLITS:
            split_frame = window_frame.loc[_split_mask(window_frame, split_key)]
            for weighted_mode in ("weighted", "unweighted"):
                hitters_frame = _aggregate_hitter_metrics(split_frame, weighted_mode, config.year_weights)
                if not hitters_frame.empty:
                    hitters_frame = hitters_frame.merge(starter_scores, on=["team", "batter"], how="left")
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
    return (
        pd.concat(hitters, ignore_index=True),
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
    hitter_metrics.to_parquet(target_dir / "daily_hitter_metrics.parquet", index=False)
    metadata = {
        "build_timestamp_utc": datetime.utcnow().isoformat(),
        "target_date": context.target_date.isoformat(),
        "metrics_version": context.config.metrics_version,
        "split_keys": list(DEFAULT_SPLITS),
        "recent_windows": list(DEFAULT_RECENT_WINDOWS),
        "pitch_groups": DEFAULT_PITCH_GROUPS,
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


def run_build(context: BuildContext) -> None:
    ensure_directories(context.config)
    csv_paths = _csv_glob(context.csv_dir)
    historical_statcast = _load_raw_statcast(csv_paths)
    live_payload = load_cockroach_payload(context.config)
    raw_statcast = _merge_historical_and_live(historical_statcast, live_payload)
    batter_zone_profiles, pitcher_zone_profiles = _aggregate_zone_profiles(raw_statcast, context.config.zone_year_weights)
    hitter_metrics, pitcher_metrics, pitcher_summary_by_hand, pitcher_arsenal, pitcher_arsenal_by_hand, pitcher_usage_by_count = build_metric_tables(raw_statcast, context.config)
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
    )
    schedule = fetch_schedule(context.target_date)
    rosters = fetch_team_rosters_for_schedule(schedule, context.target_date)
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
