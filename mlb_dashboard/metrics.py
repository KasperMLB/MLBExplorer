from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

SWINGING_STRIKE_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike", "automatic_strike"}
BALL_DESCRIPTIONS = {"ball", "blocked_ball", "pitchout", "intent_ball"}
IN_PLAY_TYPES = {"line_drive", "fly_ball", "ground_ball", "popup"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}


def apply_year_weights(frame: pd.DataFrame, year_weights: dict[int, float]) -> pd.Series:
    return frame["game_year"].map(year_weights).fillna(1.0)


def is_barrel(frame: pd.DataFrame) -> pd.Series:
    launch_speed = pd.to_numeric(frame["launch_speed"], errors="coerce")
    launch_angle = pd.to_numeric(frame["launch_angle"], errors="coerce")
    capped_speed = launch_speed.clip(upper=116)
    speed_bucket = np.floor(capped_speed)

    lower_band = pd.Series(np.nan, index=frame.index, dtype="float64")
    upper_band = pd.Series(np.nan, index=frame.index, dtype="float64")

    lower_band = lower_band.mask(speed_bucket.eq(98), 26)
    upper_band = upper_band.mask(speed_bucket.eq(98), 30)

    lower_band = lower_band.mask(speed_bucket.eq(99), 25)
    upper_band = upper_band.mask(speed_bucket.eq(99), 31)

    lower_band = lower_band.mask(speed_bucket.eq(100), 24)
    upper_band = upper_band.mask(speed_bucket.eq(100), 33)

    lower_band = lower_band.mask(speed_bucket.eq(101), 23)
    upper_band = upper_band.mask(speed_bucket.eq(101), 34)

    lower_band = lower_band.mask(speed_bucket.eq(102), 22)
    upper_band = upper_band.mask(speed_bucket.eq(102), 36)

    lower_band = lower_band.mask(speed_bucket.eq(103), 21)
    upper_band = upper_band.mask(speed_bucket.eq(103), 37)

    lower_band = lower_band.mask(speed_bucket.eq(104), 20)
    upper_band = upper_band.mask(speed_bucket.eq(104), 39)

    lower_band = lower_band.mask(speed_bucket.eq(105), 19)
    upper_band = upper_band.mask(speed_bucket.eq(105), 40)

    lower_band = lower_band.mask(speed_bucket.eq(106), 18)
    upper_band = upper_band.mask(speed_bucket.eq(106), 42)

    lower_band = lower_band.mask(speed_bucket.eq(107), 17)
    upper_band = upper_band.mask(speed_bucket.eq(107), 43)

    lower_band = lower_band.mask(speed_bucket.eq(108), 16)
    upper_band = upper_band.mask(speed_bucket.eq(108), 45)

    lower_band = lower_band.mask(speed_bucket.eq(109), 15)
    upper_band = upper_band.mask(speed_bucket.eq(109), 46)

    lower_band = lower_band.mask(speed_bucket.eq(110), 14)
    upper_band = upper_band.mask(speed_bucket.eq(110), 48)

    lower_band = lower_band.mask(speed_bucket.eq(111), 13)
    upper_band = upper_band.mask(speed_bucket.eq(111), 49)

    lower_band = lower_band.mask(speed_bucket.eq(112), 12)
    upper_band = upper_band.mask(speed_bucket.eq(112), 50)

    lower_band = lower_band.mask(speed_bucket.eq(113), 11)
    upper_band = upper_band.mask(speed_bucket.eq(113), 50)

    lower_band = lower_band.mask(speed_bucket.eq(114), 10)
    upper_band = upper_band.mask(speed_bucket.eq(114), 50)

    lower_band = lower_band.mask(speed_bucket.eq(115), 9)
    upper_band = upper_band.mask(speed_bucket.eq(115), 50)

    lower_band = lower_band.mask(speed_bucket.ge(116), 8)
    upper_band = upper_band.mask(speed_bucket.ge(116), 50)

    return launch_speed.ge(98) & launch_angle.ge(lower_band) & launch_angle.le(upper_band)


def is_hard_hit(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["launch_speed"], errors="coerce").ge(95)


def is_fly_ball(frame: pd.DataFrame) -> pd.Series:
    return frame["bb_type"].fillna("").astype(str).str.lower().eq("fly_ball")


def is_ground_ball(frame: pd.DataFrame) -> pd.Series:
    return frame["bb_type"].fillna("").astype(str).str.lower().eq("ground_ball")


def is_sweet_spot(frame: pd.DataFrame) -> pd.Series:
    launch_angle = pd.to_numeric(frame["launch_angle"], errors="coerce")
    return launch_angle.ge(8) & launch_angle.le(32)


def is_batted_ball(frame: pd.DataFrame) -> pd.Series:
    return frame["bb_type"].fillna("").astype(str).str.lower().isin(IN_PLAY_TYPES)


def is_pulled_batted_ball(frame: pd.DataFrame) -> pd.Series:
    hc_x = pd.to_numeric(frame["hc_x"], errors="coerce")
    stand = frame["stand"].fillna("")
    return frame["is_batted_ball"] & (
        ((stand == "R") & hc_x.lt(125))
        | ((stand == "L") & hc_x.gt(125))
    )


def is_swinging_strike(frame: pd.DataFrame) -> pd.Series:
    return frame["description"].fillna("").astype(str).str.lower().isin(SWINGING_STRIKE_DESCRIPTIONS)


def is_ball(frame: pd.DataFrame) -> pd.Series:
    return frame["description"].fillna("").astype(str).str.lower().isin(BALL_DESCRIPTIONS)


def is_called_strike(frame: pd.DataFrame) -> pd.Series:
    return frame["description"].fillna("").astype(str).str.lower().isin(CALLED_STRIKE_DESCRIPTIONS)


def is_hit_event(frame: pd.DataFrame) -> pd.Series:
    return frame.get("events", pd.Series(index=frame.index, dtype="object")).fillna("").astype(str).str.lower().isin(HIT_EVENTS)


def is_home_run_event(frame: pd.DataFrame) -> pd.Series:
    return frame.get("events", pd.Series(index=frame.index, dtype="object")).fillna("").astype(str).str.lower().eq("home_run")


def add_metric_flags(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["is_batted_ball"] = is_batted_ball(enriched)
    enriched["is_tracked_bbe"] = (
        enriched["is_batted_ball"]
        & pd.to_numeric(enriched["launch_speed"], errors="coerce").notna()
        & pd.to_numeric(enriched["launch_angle"], errors="coerce").notna()
    )
    enriched["is_barrel"] = is_barrel(enriched)
    enriched["is_hard_hit"] = is_hard_hit(enriched)
    enriched["is_fly_ball"] = is_fly_ball(enriched)
    enriched["is_ground_ball"] = is_ground_ball(enriched)
    enriched["is_sweet_spot"] = is_sweet_spot(enriched)
    enriched["is_pulled_batted_ball"] = is_pulled_batted_ball(enriched)
    enriched["is_pulled_barrel"] = enriched["is_barrel"] & enriched["is_pulled_batted_ball"]
    launch_speed_angle = pd.to_numeric(
        enriched.get("launch_speed_angle", pd.Series(pd.NA, index=enriched.index)),
        errors="coerce",
    )
    bb_type = enriched["bb_type"].fillna("").astype(str).str.lower()
    launch_angle = pd.to_numeric(enriched["launch_angle"], errors="coerce")
    launch_speed = pd.to_numeric(enriched["launch_speed"], errors="coerce")
    productive_air_fallback = (
        enriched["is_tracked_bbe"]
        & bb_type.isin({"fly_ball", "line_drive"})
        & launch_speed.ge(95)
        & launch_angle.between(18, 35, inclusive="both")
    )
    enriched["launch_speed_angle_value"] = launch_speed_angle
    enriched["is_hr_window"] = enriched["is_tracked_bbe"] & launch_angle.between(18, 35, inclusive="both")
    enriched["is_productive_air"] = enriched["is_tracked_bbe"] & (
        (launch_speed_angle.notna() & launch_speed_angle.isin([5, 6]))
        | (launch_speed_angle.isna() & productive_air_fallback)
    )
    enriched["is_swinging_strike"] = is_swinging_strike(enriched)
    enriched["is_ball"] = is_ball(enriched)
    enriched["is_called_strike"] = is_called_strike(enriched)
    enriched["is_hit_event"] = is_hit_event(enriched)
    enriched["is_home_run_event"] = is_home_run_event(enriched)
    flag_columns = [
        "is_batted_ball",
        "is_tracked_bbe",
        "is_barrel",
        "is_hard_hit",
        "is_fly_ball",
        "is_ground_ball",
        "is_sweet_spot",
        "is_pulled_batted_ball",
        "is_pulled_barrel",
        "is_hr_window",
        "is_productive_air",
        "is_swinging_strike",
        "is_ball",
        "is_called_strike",
        "is_hit_event",
        "is_home_run_event",
    ]
    for column in flag_columns:
        enriched[column] = enriched[column].fillna(False).astype(bool)
    enriched["xwoba_value"] = pd.to_numeric(enriched["estimated_woba_using_speedangle"], errors="coerce")
    enriched["launch_angle_value"] = pd.to_numeric(enriched["launch_angle"], errors="coerce")
    enriched["release_speed_value"] = pd.to_numeric(enriched["release_speed"], errors="coerce")
    enriched["spin_rate_value"] = pd.to_numeric(enriched["release_spin_rate"], errors="coerce")
    enriched["balls_value"] = pd.to_numeric(enriched["balls"], errors="coerce").fillna(0).astype(int) if "balls" in enriched else 0
    enriched["strikes_value"] = pd.to_numeric(enriched["strikes"], errors="coerce").fillna(0).astype(int) if "strikes" in enriched else 0
    enriched["batter_side_key"] = enriched["stand"].fillna("").map({"L": "vs_lhh", "R": "vs_rhh"}).fillna("unknown")
    enriched["count_bucket"] = classify_count_buckets(enriched)
    return enriched


def classify_count_buckets(frame: pd.DataFrame) -> pd.Series:
    balls = pd.to_numeric(frame["balls"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(frame["strikes"], errors="coerce").fillna(0).astype(int)
    conditions = [
        (balls == 3) & (strikes == 2),
        (balls + strikes) <= 1,
        strikes >= 2,
        strikes < 2,
        balls > strikes,
        strikes > balls,
        (balls == strikes) & ~((balls == 0) & (strikes == 0)),
    ]
    choices = [
        "Full count",
        "Early count",
        "Two-strike",
        "Pre two-strike",
        "Pitcher behind",
        "Pitcher ahead",
        "Even count",
    ]
    return pd.Series(np.select(conditions, choices, default="Even count"), index=frame.index, dtype="object")


def likely_starter_scores(plate_appearance_history: pd.DataFrame, recent_days: int = 14) -> pd.DataFrame:
    if plate_appearance_history.empty:
        return pd.DataFrame(columns=["batter", "likely_starter_score"])
    history = plate_appearance_history.copy()
    history["batter"] = pd.to_numeric(history["batter"], errors="coerce")
    history = history.loc[history["batter"].notna()].copy()
    history["batter"] = history["batter"].astype(int)
    history["game_date"] = pd.to_datetime(history["game_date"])
    cutoff = history["game_date"].max() - pd.Timedelta(days=recent_days)
    recent = history.loc[history["game_date"] >= cutoff]
    grouped = (
        recent.groupby(["batter"], as_index=False)
        .agg(
            games=("game_pk", "nunique"),
            plate_appearances=("at_bat_number", "nunique"),
            last_seen=("game_date", "max"),
        )
    )
    grouped["days_since"] = (grouped["last_seen"].max() - grouped["last_seen"]).dt.days.fillna(recent_days)
    grouped["likely_starter_score"] = (
        grouped["games"] * 2.0
        + grouped["plate_appearances"] * 0.5
        - grouped["days_since"] * 0.2
    )
    return grouped[["batter", "likely_starter_score"]]


def safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def series_dict(frame: pd.DataFrame, columns: Iterable[str]) -> list[dict]:
    output: list[dict] = []
    for row in frame.loc[:, list(columns)].to_dict(orient="records"):
        output.append({key: safe_float(value) if isinstance(value, (int, float)) else value for key, value in row.items()})
    return output
