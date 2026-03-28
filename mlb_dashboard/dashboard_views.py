from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

COUNT_BUCKET_ORDER = [
    "All counts",
    "Early count",
    "Even count",
    "Pitcher ahead",
    "Pitcher behind",
    "Two-strike",
    "Pre two-strike",
    "Full count",
]

BATTER_SIDE_LABELS = {
    "all": "All",
    "vs_lhh": "vs LHH",
    "vs_rhh": "vs RHH",
}

HITTER_PRESETS = {
    "Power": [
        "hitter_name",
        "team",
        "matchup_score",
        "xwoba",
        "xwoba_con",
        "pulled_barrel_pct",
        "barrel_bip_pct",
        "hard_hit_pct",
        "avg_launch_angle",
    ],
    "Contact": [
        "hitter_name",
        "team",
        "matchup_score",
        "swstr_pct",
        "xwoba",
        "xwoba_con",
        "pitch_count",
        "bip",
        "likely_starter_score",
    ],
    "Pitcher attack": [
        "hitter_name",
        "team",
        "matchup_score",
        "pulled_barrel_pct",
        "barrel_bip_pct",
        "fb_pct",
        "hard_hit_pct",
        "avg_launch_angle",
        "xwoba",
    ],
    "All stats": [
        "hitter_name",
        "team",
        "matchup_score",
        "zone_fit_score",
        "pitch_count",
        "bip",
        "xwoba",
        "xwoba_con",
        "swstr_pct",
        "pulled_barrel_pct",
        "barrel_bip_pct",
        "fb_pct",
        "hard_hit_pct",
        "avg_launch_angle",
        "likely_starter_score",
    ],
}

BEST_MATCHUP_COLUMNS = [
    "hitter_name",
    "team",
    "matchup_score",
    "xwoba",
    "swstr_pct",
    "pulled_barrel_pct",
    "hard_hit_pct",
    "fb_pct",
    "avg_launch_angle",
]

ZONE_FIT_SAMPLE_FLOOR = 15.0

TOP_PITCHER_COLUMNS = [
    "game",
    "pitcher_name",
    "p_throws",
    "pitcher_score",
    "xwoba",
    "swstr_pct",
    "pulled_barrel_pct",
    "barrel_bip_pct",
    "fb_pct",
    "hard_hit_pct",
]

PITCHER_SUMMARY_COLUMNS = [
    "pitcher_name",
    "p_throws",
    "pitch_count",
    "bip",
    "xwoba",
    "swstr_pct",
    "pulled_barrel_pct",
    "barrel_bip_pct",
    "fb_pct",
    "hard_hit_pct",
    "avg_launch_angle",
]

ARSENAL_COLUMNS = [
    "pitch_name",
    "usage_pct",
    "swstr_pct",
    "hard_hit_pct",
    "avg_release_speed",
    "avg_spin_rate",
    "xwoba_con",
]

COUNT_USAGE_COLUMNS = [
    "pitch_name",
    "All counts",
    "Early count",
    "Even count",
    "Pitcher ahead",
    "Pitcher behind",
    "Two-strike",
    "Pre two-strike",
    "Full count",
]

HITTER_ROLLING_COLUMNS = [
    "player_name",
    "rolling_window",
    "games_in_window",
    "pulled_barrel_pct",
    "hard_hit_pct",
    "fb_pct",
    "avg_launch_angle",
    "xwoba",
]

PITCHER_ROLLING_COLUMNS = [
    "player_name",
    "rolling_window",
    "games_in_window",
    "avg_release_speed",
    "barrel_bip_pct",
    "hard_hit_pct",
    "fb_pct",
    "avg_launch_angle",
]

BATTER_ZONE_COLUMNS = [
    "pitch_type",
    "zone",
    "sample_size",
    "whiff_rate",
    "hit_rate",
    "hr_rate",
    "damage_rate",
]

PITCHER_ZONE_COLUMNS = [
    "pitch_type",
    "zone",
    "sample_size",
    "usage_rate",
]

PITCHER_LOWER_IS_BETTER = {
    "xwoba",
    "pulled_barrel_pct",
    "barrel_bbe_pct",
    "barrel_bip_pct",
    "fb_pct",
    "hard_hit_pct",
    "avg_launch_angle",
}

PITCHER_HIGHER_IS_BETTER = {
    "swstr_pct",
    "gb_pct",
    "gb_fb_ratio",
    "avg_release_speed",
    "avg_spin_rate",
    "usage_pct",
    "pitcher_score",
}

ZONE_DISPLAY_ORDER = [11, 1, 2, 3, 13, 4, 5, 6, 14, 7, 8, 9, 12]


def normalize_series(series: pd.Series, inverse: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(0.5, index=series.index)
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value) or abs(max_value - min_value) < 1e-9:
        normalized = pd.Series(0.5, index=series.index)
    else:
        normalized = (numeric - min_value) / (max_value - min_value)
    if inverse:
        normalized = 1.0 - normalized
    return normalized.fillna(0.5)


def launch_angle_score(series: pd.Series, low: float = 20.0, ideal: float = 27.5, high: float = 35.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    def score(value: float | int | None) -> float:
        if pd.isna(value):
            return 0.5
        value = float(value)
        if low <= value <= high:
            if value <= ideal:
                return 0.8 + 0.2 * ((value - low) / max(ideal - low, 1e-9))
            return 0.8 + 0.2 * ((high - value) / max(high - ideal, 1e-9))
        if value < low:
            return max(0.0, 0.8 - ((low - value) / max(low, 1e-9)) * 0.8)
        return max(0.0, 0.8 - ((value - high) / max(high, 1e-9)) * 0.8)

    return numeric.apply(score)


def _weighted_average(series: pd.Series, weights: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    valid = numeric.notna() & numeric_weights.gt(0)
    if not valid.any():
        return None
    total_weight = float(numeric_weights.loc[valid].sum())
    if total_weight <= 0:
        return None
    return float((numeric.loc[valid] * numeric_weights.loc[valid]).sum() / total_weight)


def _select_batter_zone_frame(frame: pd.DataFrame, batter_id: object, opposing_pitcher_hand: str | None) -> pd.DataFrame:
    if frame.empty or "batter_id" not in frame.columns:
        return pd.DataFrame()
    work = frame.loc[frame["batter_id"] == batter_id].copy()
    if work.empty:
        return work
    if "pitcher_hand_key" in work.columns:
        hand_key = {"R": "vs_rhp", "L": "vs_lhp"}.get(opposing_pitcher_hand or "", "overall")
        specific = work.loc[work["pitcher_hand_key"] == hand_key]
        if not specific.empty:
            return specific
        overall = work.loc[work["pitcher_hand_key"] == "overall"]
        if not overall.empty:
            return overall
    return work


def _select_pitcher_zone_frame(frame: pd.DataFrame, pitcher_id: object, hitter_side: str | None) -> pd.DataFrame:
    if frame.empty or "pitcher_id" not in frame.columns:
        return pd.DataFrame()
    work = frame.loc[frame["pitcher_id"] == pitcher_id].copy()
    if work.empty:
        return work
    if "batter_side_key" in work.columns:
        side_key = {"L": "vs_lhh", "R": "vs_rhh"}.get(hitter_side or "", "overall")
        specific = work.loc[work["batter_side_key"] == side_key]
        if not specific.empty:
            return specific
        overall = work.loc[work["batter_side_key"] == "overall"]
        if not overall.empty:
            return overall
    return work


def _zone_fit_for_hitter(
    hitter_row: pd.Series,
    batter_zone_profiles: pd.DataFrame | None,
    pitcher_zone_profiles: pd.DataFrame | None,
    opposing_pitcher_id: int | None,
    opposing_pitcher_hand: str | None,
) -> float:
    if batter_zone_profiles is None or pitcher_zone_profiles is None or opposing_pitcher_id is None:
        return 0.5
    batter_frame = _select_batter_zone_frame(batter_zone_profiles, hitter_row.get("batter"), opposing_pitcher_hand)
    pitcher_frame = _select_pitcher_zone_frame(pitcher_zone_profiles, opposing_pitcher_id, hitter_row.get("stand"))
    if batter_frame.empty or pitcher_frame.empty:
        return 0.5
    batter_map = aggregate_batter_zone_map(batter_frame, "All pitches")
    pitcher_map = aggregate_pitcher_zone_map(pitcher_frame, "All pitches")
    overlay = build_zone_overlay_map(batter_map, pitcher_map)
    if overlay.empty:
        return 0.5
    sample = pd.to_numeric(overlay["sample_size"], errors="coerce").fillna(0)
    if float(sample.sum()) < ZONE_FIT_SAMPLE_FLOOR:
        return 0.5
    score = _weighted_average(overlay["zone_value"], sample)
    if score is None:
        return 0.5
    return min(max(float(score), 0.0), 1.0)


def add_hitter_matchup_score(
    frame: pd.DataFrame,
    batter_zone_profiles: pd.DataFrame | None = None,
    pitcher_zone_profiles: pd.DataFrame | None = None,
    opposing_pitcher_id: int | None = None,
    opposing_pitcher_hand: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy()
    if batter_zone_profiles is not None and pitcher_zone_profiles is not None and opposing_pitcher_id is not None:
        enriched["zone_fit_score"] = enriched.apply(
            lambda row: _zone_fit_for_hitter(row, batter_zone_profiles, pitcher_zone_profiles, opposing_pitcher_id, opposing_pitcher_hand),
            axis=1,
        )
    else:
        enriched["zone_fit_score"] = 0.5
    swstr_score = normalize_series(enriched["swstr_pct"], inverse=True)
    barrel_score = normalize_series(enriched["barrel_bbe_pct"])
    la_score = launch_angle_score(enriched["avg_launch_angle"])
    base_score = ((swstr_score * 0.35) + (barrel_score * 0.30) + (la_score * 0.20) + (enriched["zone_fit_score"] * 0.15)) * 100.0
    pulled_barrel_scale = normalize_series(enriched["pulled_barrel_pct"])
    pulled_barrel_bonus = ((pulled_barrel_scale - 0.5).clip(lower=0.0) / 0.5) * 0.08
    enriched["matchup_score"] = (base_score * (1.0 + pulled_barrel_bonus)).clip(lower=0.0, upper=100.0)
    return enriched.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last")


def add_pitcher_rank_score(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy()
    xwoba_score = normalize_series(enriched["xwoba"], inverse=True)
    barrel_bbe_score = normalize_series(enriched["barrel_bbe_pct"], inverse=True)
    barrel_bip_score = normalize_series(enriched["barrel_bip_pct"], inverse=True)
    hh_score = normalize_series(enriched["hard_hit_pct"], inverse=True)
    swstr_score = normalize_series(enriched["swstr_pct"])
    gbfb_score = normalize_series(enriched["gb_fb_ratio"])
    enriched["pitcher_score"] = (
        (xwoba_score * 0.25)
        + (barrel_bbe_score * 0.18)
        + (barrel_bip_score * 0.12)
        + (hh_score * 0.15)
        + (swstr_score * 0.18)
        + (gbfb_score * 0.12)
    ) * 100.0
    return enriched.sort_values(["pitcher_score", "xwoba"], ascending=[False, True], na_position="last")


def hitter_columns_for_preset(preset_name: str) -> list[str]:
    return HITTER_PRESETS.get(preset_name, HITTER_PRESETS["All stats"])


def with_game_label(frame: pd.DataFrame, game_label: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy()
    enriched.insert(0, "game", game_label)
    return enriched


def build_best_matchups(away_hitters: pd.DataFrame, home_hitters: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([away_hitters, home_hitters], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    return combined.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last").head(3)


def latest_built_date(daily_dir: Path) -> date | None:
    if not daily_dir.exists():
        return None
    valid_dates: list[date] = []
    for path in daily_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            valid_dates.append(date.fromisoformat(path.name))
        except ValueError:
            continue
    if not valid_dates:
        return None
    today = date.today()
    if today in valid_dates:
        return today
    return max(valid_dates)


def apply_roster_names(frame: pd.DataFrame, rosters: pd.DataFrame, team: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    lookup = rosters.loc[rosters["team"] == team, ["player_id", "player_name"]].drop_duplicates("player_id")
    if lookup.empty:
        return frame
    enriched = frame.merge(lookup, left_on="batter", right_on="player_id", how="inner")
    enriched["hitter_name"] = enriched["player_name"]
    return enriched.drop(columns=["player_id", "player_name"], errors="ignore")


def pivot_count_usage(count_usage: pd.DataFrame, arsenal_frame: pd.DataFrame) -> pd.DataFrame:
    if count_usage.empty and arsenal_frame.empty:
        return pd.DataFrame(columns=COUNT_USAGE_COLUMNS)
    pivot = count_usage.pivot_table(index="pitch_name", columns="count_bucket", values="usage_pct", aggfunc="first")
    if not pivot.empty:
        pivot = pivot.groupby(level=0).first()
    pivot = pivot.reindex(columns=[bucket for bucket in COUNT_BUCKET_ORDER if bucket != "All counts"])
    if arsenal_frame.empty:
        all_count_map: dict[str, float] = {}
    else:
        all_counts = (
            arsenal_frame.groupby("pitch_name", as_index=True)["usage_pct"]
            .max()
            .sort_values(ascending=False)
        )
        all_count_map = all_counts.to_dict()
    combined = pivot.copy()
    combined.insert(0, "All counts", [all_count_map.get(pitch_name) for pitch_name in combined.index])
    combined = combined.reset_index()
    if "All counts" in combined:
        combined = combined.sort_values("All counts", ascending=False, na_position="last")
    return combined.reindex(columns=COUNT_USAGE_COLUMNS)


def build_game_export_options(
    game_title: str,
    away_team: str,
    home_team: str,
    best_matchups: pd.DataFrame,
    away_sections: list[dict],
    home_sections: list[dict],
    away_hitters: pd.DataFrame,
    home_hitters: pd.DataFrame,
) -> dict[str, list[dict]]:
    away_arsenal_sections = [section for section in away_sections if "Summary" in section["title"] or "Arsenal" in section["title"]]
    home_arsenal_sections = [section for section in home_sections if "Summary" in section["title"] or "Arsenal" in section["title"]]
    away_count_sections = [section for section in away_sections if "Summary" in section["title"] or "Count Usage" in section["title"]]
    home_count_sections = [section for section in home_sections if "Summary" in section["title"] or "Count Usage" in section["title"]]
    full_game_sections = [{"title": "Best Matchups", "frame": best_matchups[BEST_MATCHUP_COLUMNS]}]
    full_game_sections.extend(away_sections)
    full_game_sections.extend(home_sections)
    full_game_sections.extend(
        [
            {"title": f"{away_team} Hitters", "frame": away_hitters},
            {"title": f"{home_team} Hitters", "frame": home_hitters},
        ]
    )
    return {
        "Full game": full_game_sections,
        "Hitters - Both teams": [
            {"title": f"{away_team} Hitters", "frame": away_hitters},
            {"title": f"{home_team} Hitters", "frame": home_hitters},
        ],
        f"Hitters - {away_team} only": [{"title": f"{away_team} Hitters", "frame": away_hitters}],
        f"Hitters - {home_team} only": [{"title": f"{home_team} Hitters", "frame": home_hitters}],
        "Pitchers - Both starters": [*away_sections, *home_sections],
        f"Pitchers - {away_team} only": away_sections,
        f"Pitchers - {home_team} only": home_sections,
        "Arsenal - Both starters": [*away_arsenal_sections, *home_arsenal_sections],
        f"Arsenal - {away_team} only": away_arsenal_sections,
        f"Arsenal - {home_team} only": home_arsenal_sections,
        "Count Usage - Both starters": [*away_count_sections, *home_count_sections],
        f"Count Usage - {away_team} only": away_count_sections,
        f"Count Usage - {home_team} only": home_count_sections,
    }


def aggregate_batter_zone_map(frame: pd.DataFrame, pitch_type: str = "All pitches") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["zone", "sample_size", "zone_value", "display_value"])
    work = frame.copy()
    if "pitcher_hand_key" in work.columns and (work["pitcher_hand_key"] == "overall").any():
        work = work.loc[work["pitcher_hand_key"] == "overall"]
    if pitch_type != "All pitches" and "pitch_type" in work.columns:
        work = work.loc[work["pitch_type"] == pitch_type]
    if work.empty:
        return pd.DataFrame(columns=["zone", "sample_size", "zone_value", "display_value"])

    rows: list[dict] = []
    for zone, group in work.groupby("zone", sort=False):
        sample_size = pd.to_numeric(group["sample_size"], errors="coerce").fillna(0)
        hit_rate = _weighted_average(group["hit_rate"], sample_size)
        hr_rate = _weighted_average(group["hr_rate"], sample_size)
        if hit_rate is None and hr_rate is None:
            zone_value = None
        else:
            zone_value = (float(hit_rate or 0.0) * 0.6) + (float(hr_rate or 0.0) * 0.4)
        rows.append(
            {
                "zone": int(float(zone)),
                "sample_size": int(sample_size.sum()),
                "zone_value": zone_value,
                "display_value": zone_value,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("zone", key=lambda s: s.map({zone: idx for idx, zone in enumerate(ZONE_DISPLAY_ORDER)}), na_position="last")
        .reset_index(drop=True)
    )


def aggregate_pitcher_zone_map(frame: pd.DataFrame, pitch_type: str = "All pitches") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["zone", "sample_size", "zone_value", "display_value"])
    work = frame.copy()
    if "batter_side_key" in work.columns and (work["batter_side_key"] == "overall").any():
        work = work.loc[work["batter_side_key"] == "overall"]
    if pitch_type != "All pitches" and "pitch_type" in work.columns:
        work = work.loc[work["pitch_type"] == pitch_type]
    if work.empty:
        return pd.DataFrame(columns=["zone", "sample_size", "zone_value", "display_value"])

    rows: list[dict] = []
    for zone, group in work.groupby("zone", sort=False):
        sample_size = pd.to_numeric(group["sample_size"], errors="coerce").fillna(0)
        usage = pd.to_numeric(group["usage_rate"], errors="coerce").fillna(0)
        rows.append(
            {
                "zone": int(float(zone)),
                "sample_size": int(sample_size.sum()),
                "zone_value": float(usage.sum()),
                "display_value": float(usage.sum()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("zone", key=lambda s: s.map({zone: idx for idx, zone in enumerate(ZONE_DISPLAY_ORDER)}), na_position="last")
        .reset_index(drop=True)
    )


def build_zone_overlay_map(batter_map: pd.DataFrame, pitcher_map: pd.DataFrame) -> pd.DataFrame:
    if batter_map.empty or pitcher_map.empty:
        return pd.DataFrame(columns=["zone", "sample_size", "zone_value", "display_value"])
    merged = batter_map.merge(
        pitcher_map,
        on="zone",
        how="inner",
        suffixes=("_batter", "_pitcher"),
    )
    if merged.empty:
        return pd.DataFrame(columns=["zone", "sample_size", "zone_value", "display_value"])

    batter_scale = normalize_series(merged["zone_value_batter"])
    pitcher_scale = normalize_series(merged["zone_value_pitcher"])
    overlay_score = batter_scale * pitcher_scale
    merged["zone_value"] = overlay_score
    merged["display_value"] = overlay_score
    merged["sample_size"] = merged[["sample_size_batter", "sample_size_pitcher"]].min(axis=1)
    return (
        merged[["zone", "sample_size", "zone_value", "display_value"]]
        .sort_values("zone", key=lambda s: s.map({zone: idx for idx, zone in enumerate(ZONE_DISPLAY_ORDER)}), na_position="last")
        .reset_index(drop=True)
    )
