from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from .config import AppConfig
from .metrics import add_metric_flags

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None


@dataclass
class CockroachPayload:
    live_pitch_mix: pd.DataFrame
    hitter_rolling: pd.DataFrame
    pitcher_rolling: pd.DataFrame
    batter_zone_profiles: pd.DataFrame
    pitcher_zone_profiles: pd.DataFrame


def _ensure_driver() -> None:
    if psycopg is None:
        raise RuntimeError("psycopg is required for Cockroach integration.")


def _normalize_database_url(database_url: str) -> str:
    if not database_url:
        return database_url
    parsed = urlsplit(database_url)
    if parsed.scheme == "cockroachdb":
        return urlunsplit(("postgresql", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    return database_url


def _normalize_live_pitch_mix(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    normalized["game_date"] = pd.to_datetime(normalized["game_date"])
    normalized["game_year"] = pd.to_numeric(normalized.get("game_year", normalized.get("source_season", 2026)), errors="coerce").fillna(2026).astype(int)
    normalized["team"] = normalized.apply(lambda row: row["away_team"] if row["inning_topbot"] == "Top" else row["home_team"], axis=1)
    normalized["fielding_team"] = normalized.apply(lambda row: row["home_team"] if row["inning_topbot"] == "Top" else row["away_team"], axis=1)
    normalized["release_spin_rate"] = pd.to_numeric(normalized.get("release_spin_rate"), errors="coerce")
    normalized["pitcher_name"] = normalized["player_name"]
    return add_metric_flags(normalized)


def _compute_hitter_rolling_15(live_pitch_mix: pd.DataFrame) -> pd.DataFrame:
    if live_pitch_mix.empty:
        return pd.DataFrame(columns=["player_name", "rolling_window", "pulled_barrel_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle", "xwoba", "games_in_window"])
    game_log = (
        live_pitch_mix.sort_values(["batter", "game_date", "game_pk"])
        .groupby(["batter", "player_name", "game_pk", "game_date"], as_index=False)
        .agg(
            tracked_bbe=("is_tracked_bbe", "sum"),
            pulled_barrels=("is_pulled_barrel", "sum"),
            hard_hits=("is_hard_hit", "sum"),
            fly_balls=("is_fly_ball", "sum"),
            bip=("is_batted_ball", "sum"),
            launch_angle_sum=("launch_angle_value", "sum"),
            launch_angle_count=("launch_angle_value", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
            xwoba_sum=("xwoba_value", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
            xwoba_count=("xwoba_value", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
        )
    )
    game_log["games_rank"] = game_log.groupby("batter").cumcount(ascending=False)
    recent = game_log.groupby("batter", group_keys=False).apply(lambda group: group.tail(15))
    rows: list[dict] = []
    for (batter, player_name), group in recent.groupby(["batter", "player_name"], sort=False):
        tracked_bbe = float(group["tracked_bbe"].sum())
        bip = float(group["bip"].sum())
        launch_count = float(group["launch_angle_count"].sum())
        xwoba_count = float(group["xwoba_count"].sum())
        rows.append(
            {
                "player_id": batter,
                "player_name": player_name,
                "rolling_window": "Rolling 15",
                "pulled_barrel_pct": float(group["pulled_barrels"].sum()) / max(tracked_bbe, 1e-9),
                "hard_hit_pct": float(group["hard_hits"].sum()) / max(bip, 1e-9),
                "fb_pct": float(group["fly_balls"].sum()) / max(bip, 1e-9),
                "avg_launch_angle": float(group["launch_angle_sum"].sum()) / max(launch_count, 1e-9),
                "xwoba": float(group["xwoba_sum"].sum()) / max(xwoba_count, 1e-9),
                "games_in_window": int(group["game_pk"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _map_hitter_rolling(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["player_name", "rolling_window", "pulled_barrel_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle", "xwoba", "games_in_window"])
    rows: list[dict] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "player_name": row["player_name"],
                "rolling_window": "Rolling 5",
                "pulled_barrel_pct": row.get("batter_pulled_barrel_rate_5g"),
                "hard_hit_pct": row.get("batter_hard_hit_rate_5g"),
                "fb_pct": row.get("batter_fb_rate_5g"),
                "avg_launch_angle": row.get("batter_avg_launch_angle_5g"),
                "xwoba": None,
                "games_in_window": row.get("batter_games_in_window_5g"),
            }
        )
        rows.append(
            {
                "player_name": row["player_name"],
                "rolling_window": "Rolling 10",
                "pulled_barrel_pct": None,
                "hard_hit_pct": row.get("batter_hard_hit_rate_10g"),
                "fb_pct": row.get("batter_fb_rate_10g"),
                "avg_launch_angle": row.get("batter_avg_launch_angle_10g"),
                "xwoba": None,
                "games_in_window": row.get("batter_games_in_window_10g"),
            }
        )
    return pd.DataFrame(rows)


def _map_pitcher_rolling(frame: pd.DataFrame, live_pitch_mix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if not frame.empty:
        for _, row in frame.iterrows():
            rows.append(
                {
                    "player_name": row["player_name"],
                    "rolling_window": "Rolling 5",
                    "hard_hit_pct": row.get("pitcher_hard_hit_rate_allowed_5g"),
                    "fb_pct": row.get("pitcher_fb_rate_allowed_5g"),
                    "avg_launch_angle": row.get("pitcher_avg_launch_angle_allowed_5g"),
                    "barrel_bip_pct": row.get("pitcher_barrel_rate_allowed_5g"),
                    "avg_release_speed": row.get("pitcher_avg_release_speed_5g"),
                    "games_in_window": row.get("pitcher_games_in_window_5g"),
                }
            )
            rows.append(
                {
                    "player_name": row["player_name"],
                    "rolling_window": "Rolling 10",
                    "hard_hit_pct": row.get("pitcher_hard_hit_rate_allowed_10g"),
                    "fb_pct": None,
                    "avg_launch_angle": None,
                    "barrel_bip_pct": row.get("pitcher_barrel_rate_allowed_10g"),
                    "avg_release_speed": row.get("pitcher_avg_release_speed_10g"),
                    "games_in_window": row.get("pitcher_games_in_window_10g"),
                }
            )
    if not live_pitch_mix.empty:
        game_log = (
            live_pitch_mix.sort_values(["pitcher", "game_date", "game_pk"])
            .groupby(["pitcher", "player_name", "game_pk", "game_date"], as_index=False)
            .agg(
                tracked_bbe=("is_tracked_bbe", "sum"),
                barrels=("is_barrel", "sum"),
                hard_hits=("is_hard_hit", "sum"),
                fly_balls=("is_fly_ball", "sum"),
                bip=("is_batted_ball", "sum"),
                release_speed_sum=("release_speed_value", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
                release_speed_count=("release_speed_value", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
                launch_angle_sum=("launch_angle_value", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
                launch_angle_count=("launch_angle_value", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
            )
        )
        recent = game_log.groupby("pitcher", group_keys=False).apply(lambda group: group.tail(15))
        for (pitcher_id, player_name), group in recent.groupby(["pitcher", "player_name"], sort=False):
            bip = float(group["bip"].sum())
            tracked_bbe = float(group["tracked_bbe"].sum())
            velo_count = float(group["release_speed_count"].sum())
            la_count = float(group["launch_angle_count"].sum())
            rows.append(
                {
                    "player_name": player_name,
                    "player_id": pitcher_id,
                    "rolling_window": "Rolling 15",
                    "hard_hit_pct": float(group["hard_hits"].sum()) / max(bip, 1e-9),
                    "fb_pct": float(group["fly_balls"].sum()) / max(bip, 1e-9),
                    "avg_launch_angle": float(group["launch_angle_sum"].sum()) / max(la_count, 1e-9),
                    "barrel_bip_pct": float(group["barrels"].sum()) / max(tracked_bbe, 1e-9),
                    "avg_release_speed": float(group["release_speed_sum"].sum()) / max(velo_count, 1e-9),
                    "games_in_window": int(group["game_pk"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _normalize_batter_zone_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    if "batter" in normalized.columns and "batter_id" not in normalized.columns:
        normalized = normalized.rename(columns={"batter": "batter_id"})
    return normalized


def _normalize_pitcher_zone_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    if "pitcher" in normalized.columns and "pitcher_id" not in normalized.columns:
        normalized = normalized.rename(columns={"pitcher": "pitcher_id"})
    return normalized


def _read_query(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def load_cockroach_payload(config: AppConfig) -> CockroachPayload:
    _ensure_driver()
    if not config.database_url:
        raise RuntimeError("DATABASE_URL must be set to pull 2026 Cockroach data.")
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url, autocommit=True) as conn:
        live_pitch_mix = _read_query(conn, f"SELECT * FROM {config.cockroach_live_pitch_mix_table}")
        hitter_rolling = _read_query(conn, f"SELECT * FROM {config.cockroach_hitter_rolling_table}")
        pitcher_rolling = _read_query(conn, f"SELECT * FROM {config.cockroach_pitcher_rolling_table}")
        batter_zone_profiles = _read_query(conn, f"SELECT * FROM {config.cockroach_batter_zone_table}")
        pitcher_zone_profiles = _read_query(conn, f"SELECT * FROM {config.cockroach_pitcher_zone_table}")
    live_pitch_mix = _normalize_live_pitch_mix(live_pitch_mix)
    hitter_rolling_mapped = pd.concat([_map_hitter_rolling(hitter_rolling), _compute_hitter_rolling_15(live_pitch_mix)], ignore_index=True, sort=False)
    pitcher_rolling_mapped = _map_pitcher_rolling(pitcher_rolling, live_pitch_mix)
    batter_zone_profiles = _normalize_batter_zone_profiles(batter_zone_profiles)
    pitcher_zone_profiles = _normalize_pitcher_zone_profiles(pitcher_zone_profiles)
    return CockroachPayload(
        live_pitch_mix=live_pitch_mix,
        hitter_rolling=hitter_rolling_mapped,
        pitcher_rolling=pitcher_rolling_mapped,
        batter_zone_profiles=batter_zone_profiles,
        pitcher_zone_profiles=pitcher_zone_profiles,
    )
