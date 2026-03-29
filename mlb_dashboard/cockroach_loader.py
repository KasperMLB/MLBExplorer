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


def _create_tracking_tables(conn, config: AppConfig) -> None:
    snapshot_columns = {
        "snapshot_id": "STRING PRIMARY KEY",
        "build_date": "DATE NOT NULL",
        "slate_date": "DATE NOT NULL",
        "game_pk": "INT8 NOT NULL",
        "game_label": "STRING NOT NULL",
        "team": "STRING NOT NULL",
        "opponent": "STRING NOT NULL",
        "batter_id": "INT8 NOT NULL",
        "hitter_name": "STRING NOT NULL",
        "opposing_pitcher_id": "INT8",
        "opposing_pitcher_name": "STRING",
        "opposing_pitcher_hand": "STRING",
        "split_key": "STRING NOT NULL",
        "recent_window": "STRING NOT NULL",
        "weighted_mode": "STRING NOT NULL",
        "matchup_score": "FLOAT8",
        "ceiling_score": "FLOAT8",
        "zone_fit_score": "FLOAT8",
        "likely_starter_score": "FLOAT8",
        "xwoba": "FLOAT8",
        "xwoba_con": "FLOAT8",
        "swstr_pct": "FLOAT8",
        "pulled_barrel_pct": "FLOAT8",
        "barrel_bbe_pct": "FLOAT8",
        "barrel_bip_pct": "FLOAT8",
        "fb_pct": "FLOAT8",
        "hard_hit_pct": "FLOAT8",
        "avg_launch_angle": "FLOAT8",
        "pitch_count": "INT8",
        "bip": "INT8",
    }
    outcome_columns = {
        "slate_date": "DATE NOT NULL",
        "game_pk": "INT8 NOT NULL",
        "team": "STRING NOT NULL",
        "batter_id": "INT8 NOT NULL",
        "hitter_name": "STRING NOT NULL",
        "had_plate_appearance": "BOOL",
        "started": "BOOL",
        "plate_appearances": "INT8",
        "hits": "INT8",
        "home_runs": "INT8",
        "total_bases": "INT8",
        "runs": "INT8",
        "rbi": "INT8",
        "walks": "INT8",
        "strikeouts": "INT8",
        "last_updated_at": "TIMESTAMPTZ",
    }
    board_columns = {
        "slate_date": "DATE NOT NULL",
        "game_pk": "INT8 NOT NULL",
        "batter_id": "INT8 NOT NULL",
        "hitter_name": "STRING NOT NULL",
        "team": "STRING NOT NULL",
        "board_name": "STRING NOT NULL",
        "board_rank": "INT8 NOT NULL",
        "board_score": "FLOAT8",
        "source_metric": "STRING NOT NULL",
    }
    pitcher_snapshot_columns = {
        "snapshot_id": "STRING PRIMARY KEY",
        "build_date": "DATE NOT NULL",
        "slate_date": "DATE NOT NULL",
        "game_pk": "INT8 NOT NULL",
        "game_label": "STRING NOT NULL",
        "team": "STRING NOT NULL",
        "opponent": "STRING NOT NULL",
        "pitcher_id": "INT8 NOT NULL",
        "pitcher_name": "STRING NOT NULL",
        "p_throws": "STRING",
        "split_key": "STRING NOT NULL",
        "recent_window": "STRING NOT NULL",
        "weighted_mode": "STRING NOT NULL",
        "pitcher_score": "FLOAT8",
        "xwoba": "FLOAT8",
        "swstr_pct": "FLOAT8",
        "barrel_bbe_pct": "FLOAT8",
        "barrel_bip_pct": "FLOAT8",
        "pulled_barrel_pct": "FLOAT8",
        "fb_pct": "FLOAT8",
        "gb_pct": "FLOAT8",
        "gb_fb_ratio": "FLOAT8",
        "hard_hit_pct": "FLOAT8",
        "avg_launch_angle": "FLOAT8",
        "pitch_count": "INT8",
        "bip": "INT8",
    }
    pitcher_outcome_columns = {
        "slate_date": "DATE NOT NULL",
        "game_pk": "INT8 NOT NULL",
        "team": "STRING NOT NULL",
        "pitcher_id": "INT8 NOT NULL",
        "pitcher_name": "STRING NOT NULL",
        "had_pitch": "BOOL",
        "started": "BOOL",
        "outs_recorded": "INT8",
        "batters_faced": "INT8",
        "hits_allowed": "INT8",
        "home_runs_allowed": "INT8",
        "runs_allowed": "INT8",
        "earned_runs": "INT8",
        "walks": "INT8",
        "strikeouts": "INT8",
        "last_updated_at": "TIMESTAMPTZ",
    }
    pitcher_board_columns = {
        "slate_date": "DATE NOT NULL",
        "game_pk": "INT8 NOT NULL",
        "pitcher_id": "INT8 NOT NULL",
        "pitcher_name": "STRING NOT NULL",
        "team": "STRING NOT NULL",
        "board_name": "STRING NOT NULL",
        "board_rank": "INT8 NOT NULL",
        "board_score": "FLOAT8",
        "source_metric": "STRING NOT NULL",
    }
    pitcher_arsenal_columns = {
        "slate_date": "DATE NOT NULL",
        "game_pk": "INT8 NOT NULL",
        "pitcher_id": "INT8 NOT NULL",
        "pitcher_name": "STRING NOT NULL",
        "split_key": "STRING NOT NULL",
        "recent_window": "STRING NOT NULL",
        "weighted_mode": "STRING NOT NULL",
        "batter_side_key": "STRING NOT NULL",
        "pitch_name": "STRING NOT NULL",
        "usage_pct": "FLOAT8",
        "swstr_pct": "FLOAT8",
        "hard_hit_pct": "FLOAT8",
        "avg_release_speed": "FLOAT8",
        "avg_spin_rate": "FLOAT8",
        "xwoba_con": "FLOAT8",
    }
    pitcher_count_columns = {
        "slate_date": "DATE NOT NULL",
        "game_pk": "INT8 NOT NULL",
        "pitcher_id": "INT8 NOT NULL",
        "pitcher_name": "STRING NOT NULL",
        "split_key": "STRING NOT NULL",
        "recent_window": "STRING NOT NULL",
        "weighted_mode": "STRING NOT NULL",
        "batter_side_key": "STRING NOT NULL",
        "pitch_name": "STRING NOT NULL",
        "count_bucket": "STRING NOT NULL",
        "usage_pct": "FLOAT8",
    }
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_hitter_snapshot_table} ({', '.join(f'{name} {definition}' for name, definition in snapshot_columns.items())})")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS hitter_model_snapshots_unique_idx ON {config.cockroach_hitter_snapshot_table} (slate_date, game_pk, batter_id, split_key, recent_window, weighted_mode)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_hitter_outcome_table} ({', '.join(f'{name} {definition}' for name, definition in outcome_columns.items())})")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS hitter_game_outcomes_unique_idx ON {config.cockroach_hitter_outcome_table} (slate_date, game_pk, batter_id)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_hitter_board_table} ({', '.join(f'{name} {definition}' for name, definition in board_columns.items())})")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS hitter_board_winners_unique_idx ON {config.cockroach_hitter_board_table} (slate_date, board_name, board_rank)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_snapshot_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_snapshot_columns.items())})")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_model_snapshots_unique_idx ON {config.cockroach_pitcher_snapshot_table} (slate_date, game_pk, pitcher_id, split_key, recent_window, weighted_mode)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_outcome_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_outcome_columns.items())})")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_game_outcomes_unique_idx ON {config.cockroach_pitcher_outcome_table} (slate_date, game_pk, pitcher_id)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_board_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_board_columns.items())})")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_board_winners_unique_idx ON {config.cockroach_pitcher_board_table} (slate_date, board_name, board_rank)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_arsenal_snapshot_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_arsenal_columns.items())})")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_arsenal_snapshots_unique_idx ON {config.cockroach_pitcher_arsenal_snapshot_table} (slate_date, game_pk, pitcher_id, split_key, recent_window, weighted_mode, batter_side_key, pitch_name)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_count_snapshot_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_count_columns.items())})")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_count_usage_snapshots_unique_idx ON {config.cockroach_pitcher_count_snapshot_table} (slate_date, game_pk, pitcher_id, split_key, recent_window, weighted_mode, batter_side_key, pitch_name, count_bucket)")


def _prepare_records(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    work = frame.loc[:, columns].copy()
    for column in work.columns:
        if column == "last_updated_at":
            work[column] = pd.to_datetime(work[column], errors="coerce").apply(lambda value: value.to_pydatetime() if pd.notna(value) else None)
        elif column in {"slate_date", "build_date"}:
            work[column] = pd.to_datetime(work[column], errors="coerce").dt.date
        else:
            work[column] = work[column].where(pd.notna(work[column]), None)
    return [tuple(row[column] for column in columns) for _, row in work.iterrows()]


def _upsert_frame(conn, table_name: str, frame: pd.DataFrame, conflict_columns: list[str]) -> None:
    if frame.empty:
        return
    columns = list(frame.columns)
    placeholders = ", ".join(["%s"] * len(columns))
    assignments = ", ".join(f"{column}=EXCLUDED.{column}" for column in columns if column not in conflict_columns)
    sql = (
        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({', '.join(conflict_columns)}) DO UPDATE SET {assignments}"
    )
    records = _prepare_records(frame, columns)
    with conn.cursor() as cur:
        cur.executemany(sql, records)


def write_tracking_payload(
    config: AppConfig,
    snapshots: pd.DataFrame,
    outcomes: pd.DataFrame,
    board_winners: pd.DataFrame,
    pitcher_snapshots: pd.DataFrame | None = None,
    pitcher_outcomes: pd.DataFrame | None = None,
    pitcher_board_winners: pd.DataFrame | None = None,
    pitcher_arsenal_snapshots: pd.DataFrame | None = None,
    pitcher_count_snapshots: pd.DataFrame | None = None,
) -> None:
    _ensure_driver()
    if not config.database_url:
        return
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url, autocommit=True) as conn:
        _create_tracking_tables(conn, config)
        _upsert_frame(
            conn,
            config.cockroach_hitter_snapshot_table,
            snapshots,
            ["slate_date", "game_pk", "batter_id", "split_key", "recent_window", "weighted_mode"],
        )
        _upsert_frame(
            conn,
            config.cockroach_hitter_outcome_table,
            outcomes,
            ["slate_date", "game_pk", "batter_id"],
        )
        _upsert_frame(
            conn,
            config.cockroach_hitter_board_table,
            board_winners,
            ["slate_date", "board_name", "board_rank"],
        )
        _upsert_frame(
            conn,
            config.cockroach_pitcher_snapshot_table,
            pitcher_snapshots if pitcher_snapshots is not None else pd.DataFrame(),
            ["slate_date", "game_pk", "pitcher_id", "split_key", "recent_window", "weighted_mode"],
        )
        _upsert_frame(
            conn,
            config.cockroach_pitcher_outcome_table,
            pitcher_outcomes if pitcher_outcomes is not None else pd.DataFrame(),
            ["slate_date", "game_pk", "pitcher_id"],
        )
        _upsert_frame(
            conn,
            config.cockroach_pitcher_board_table,
            pitcher_board_winners if pitcher_board_winners is not None else pd.DataFrame(),
            ["slate_date", "board_name", "board_rank"],
        )
        _upsert_frame(
            conn,
            config.cockroach_pitcher_arsenal_snapshot_table,
            pitcher_arsenal_snapshots if pitcher_arsenal_snapshots is not None else pd.DataFrame(),
            ["slate_date", "game_pk", "pitcher_id", "split_key", "recent_window", "weighted_mode", "batter_side_key", "pitch_name"],
        )
        _upsert_frame(
            conn,
            config.cockroach_pitcher_count_snapshot_table,
            pitcher_count_snapshots if pitcher_count_snapshots is not None else pd.DataFrame(),
            ["slate_date", "game_pk", "pitcher_id", "split_key", "recent_window", "weighted_mode", "batter_side_key", "pitch_name", "count_bucket"],
        )


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
