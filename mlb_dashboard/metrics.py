from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

SWINGING_STRIKE_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
IN_PLAY_TYPES = {"line_drive", "fly_ball", "ground_ball", "popup"}


def apply_year_weights(frame: pd.DataFrame, year_weights: dict[int, float]) -> pd.Series:
    return frame["game_year"].map(year_weights).fillna(1.0)


def is_barrel(frame: pd.DataFrame) -> pd.Series:
    launch_speed = pd.to_numeric(frame["launch_speed"], errors="coerce")
    launch_angle = pd.to_numeric(frame["launch_angle"], errors="coerce")
    capped_speed = launch_speed.clip(upper=116)

    lower_band = pd.Series(np.nan, index=frame.index, dtype="float64")
    upper_band = pd.Series(np.nan, index=frame.index, dtype="float64")

    lower_band = lower_band.mask(capped_speed.eq(98), 26)
    upper_band = upper_band.mask(capped_speed.eq(98), 30)

    lower_band = lower_band.mask(capped_speed.eq(99), 25)
    upper_band = upper_band.mask(capped_speed.eq(99), 31)

    lower_band = lower_band.mask(capped_speed.eq(100), 24)
    upper_band = upper_band.mask(capped_speed.eq(100), 33)

    lower_band = lower_band.mask(capped_speed.eq(101), 23)
    upper_band = upper_band.mask(capped_speed.eq(101), 34)

    lower_band = lower_band.mask(capped_speed.eq(102), 22)
    upper_band = upper_band.mask(capped_speed.eq(102), 36)

    lower_band = lower_band.mask(capped_speed.eq(103), 21)
    upper_band = upper_band.mask(capped_speed.eq(103), 37)

    lower_band = lower_band.mask(capped_speed.eq(104), 20)
    upper_band = upper_band.mask(capped_speed.eq(104), 39)

    lower_band = lower_band.mask(capped_speed.eq(105), 19)
    upper_band = upper_band.mask(capped_speed.eq(105), 40)

    lower_band = lower_band.mask(capped_speed.eq(106), 18)
    upper_band = upper_band.mask(capped_speed.eq(106), 42)

    lower_band = lower_band.mask(capped_speed.eq(107), 17)
    upper_band = upper_band.mask(capped_speed.eq(107), 43)

    lower_band = lower_band.mask(capped_speed.eq(108), 16)
    upper_band = upper_band.mask(capped_speed.eq(108), 45)

    lower_band = lower_band.mask(capped_speed.eq(109), 15)
    upper_band = upper_band.mask(capped_speed.eq(109), 46)

    lower_band = lower_band.mask(capped_speed.eq(110), 14)
    upper_band = upper_band.mask(capped_speed.eq(110), 48)

    lower_band = lower_band.mask(capped_speed.eq(111), 13)
    upper_band = upper_band.mask(capped_speed.eq(111), 49)

    lower_band = lower_band.mask(capped_speed.eq(112), 12)
    upper_band = upper_band.mask(capped_speed.eq(112), 50)

    lower_band = lower_band.mask(capped_speed.eq(113), 11)
    upper_band = upper_band.mask(capped_speed.eq(113), 50)

    lower_band = lower_band.mask(capped_speed.eq(114), 10)
    upper_band = upper_band.mask(capped_speed.eq(114), 50)

    lower_band = lower_band.mask(capped_speed.eq(115), 9)
    upper_band = upper_band.mask(capped_speed.eq(115), 50)

    lower_band = lower_band.mask(capped_speed.ge(116), 8)
    upper_band = upper_band.mask(capped_speed.ge(116), 50)

    return launch_speed.ge(98) & launch_angle.ge(lower_band) & launch_angle.le(upper_band)


def is_hard_hit(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["launch_speed"], errors="coerce").ge(95)


def is_fly_ball(frame: pd.DataFrame) -> pd.Series:
    return frame["bb_type"].fillna("").eq("fly_ball")


def is_ground_ball(frame: pd.DataFrame) -> pd.Series:
    return frame["bb_type"].fillna("").eq("ground_ball")


def is_batted_ball(frame: pd.DataFrame) -> pd.Series:
    return frame["bb_type"].fillna("").isin(IN_PLAY_TYPES)


def is_swinging_strike(frame: pd.DataFrame) -> pd.Series:
    return frame["description"].fillna("").isin(SWINGING_STRIKE_DESCRIPTIONS)


def add_metric_flags(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["is_batted_ball"] = is_batted_ball(enriched)
    enriched["is_barrel"] = is_barrel(enriched)
    enriched["is_hard_hit"] = is_hard_hit(enriched)
    enriched["is_fly_ball"] = is_fly_ball(enriched)
    enriched["is_ground_ball"] = is_ground_ball(enriched)
    enriched["is_swinging_strike"] = is_swinging_strike(enriched)
    enriched["xwoba_value"] = pd.to_numeric(enriched["estimated_woba_using_speedangle"], errors="coerce")
    enriched["launch_angle_value"] = pd.to_numeric(enriched["launch_angle"], errors="coerce")
    enriched["release_speed_value"] = pd.to_numeric(enriched["release_speed"], errors="coerce")
    enriched["spin_rate_value"] = pd.to_numeric(enriched["release_spin_rate"], errors="coerce")
    return enriched


def likely_starter_scores(plate_appearance_history: pd.DataFrame, recent_days: int = 14) -> pd.DataFrame:
    if plate_appearance_history.empty:
        return pd.DataFrame(columns=["team", "batter", "likely_starter_score"])
    history = plate_appearance_history.copy()
    history["game_date"] = pd.to_datetime(history["game_date"])
    cutoff = history["game_date"].max() - pd.Timedelta(days=recent_days)
    recent = history.loc[history["game_date"] >= cutoff]
    grouped = (
        recent.groupby(["team", "batter"], as_index=False)
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
    return grouped[["team", "batter", "likely_starter_score"]]


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
