from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

from .config import AppConfig
from .metrics import add_metric_flags

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None


@dataclass
class CockroachPayload:
    live_pitch_mix: pd.DataFrame
    pitcher_baseline_event_rows: pd.DataFrame
    hitter_rolling: pd.DataFrame
    pitcher_rolling: pd.DataFrame
    batter_zone_profiles: pd.DataFrame
    pitcher_zone_profiles: pd.DataFrame
    batter_family_zone_profiles: pd.DataFrame


@dataclass(frozen=True)
class SourceFreshnessSummary:
    source_name: str
    table_name: str
    target_date: date
    max_event_date: date | None
    lag_days: int | None
    row_count: int
    recent_daily_counts: list[dict[str, object]]
    missing_dates: list[str]

    @property
    def is_fresh(self) -> bool:
        return self.max_event_date is not None and self.max_event_date >= self.target_date


def _ensure_driver() -> None:
    if psycopg is None:
        raise RuntimeError("psycopg is required for Cockroach integration.")


def _normalize_database_url(database_url: str) -> str:
    if not database_url:
        return database_url
    parsed = urlsplit(database_url)
    scheme = "postgresql" if parsed.scheme == "cockroachdb" else parsed.scheme
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = dict(query_pairs)
    sslmode = str(query.get("sslmode", "")).strip().lower()
    if sslmode in {"verify-ca", "verify-full", "require", "prefer"} and "sslrootcert" not in query:
        query["sslrootcert"] = certifi.where() if certifi is not None else "system"
    return urlunsplit((scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
def _normalize_live_pitch_mix(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    normalized["game_date"] = pd.to_datetime(normalized["game_date"])
    normalized["game_year"] = pd.to_numeric(normalized.get("game_year", normalized.get("source_season", 2026)), errors="coerce").fillna(2026).astype(int)
    normalized["team"] = normalized.apply(lambda row: row["away_team"] if row["inning_topbot"] == "Top" else row["home_team"], axis=1)
    normalized["fielding_team"] = normalized.apply(lambda row: row["home_team"] if row["inning_topbot"] == "Top" else row["away_team"], axis=1)
    normalized["release_spin_rate"] = pd.to_numeric(normalized.get("release_spin_rate"), errors="coerce")
    normalized["batter_name"] = normalized.get("batter_name", normalized.get("player_name", pd.Series(index=normalized.index, dtype="object")))
    normalized["pitcher_name"] = normalized.get("pitcher_name", pd.Series(index=normalized.index, dtype="object")).fillna(normalized.get("player_name"))
    normalized["player_name"] = normalized["batter_name"].fillna(normalized["player_name"])
    return add_metric_flags(normalized)


def _normalize_pitcher_baseline_event_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce")
    normalized["game_year"] = pd.to_numeric(normalized.get("source_season"), errors="coerce").fillna(2025).astype(int)
    normalized["pitch_name"] = normalized.get("pitch_name", pd.Series(index=normalized.index, dtype="object")).astype("string")
    normalized["pitch_type"] = normalized.get("pitch_type", pd.Series(index=normalized.index, dtype="object")).astype("string")
    normalized["pitch_type"] = normalized["pitch_type"].fillna(normalized["pitch_name"])
    normalized["p_throws"] = normalized.get("p_throws", normalized.get("pitcher_hand", pd.Series(index=normalized.index, dtype="object")))
    normalized["stand"] = normalized.get("stand", normalized.get("batter_stand", pd.Series(index=normalized.index, dtype="object")))
    normalized["plate_x"] = pd.to_numeric(normalized.get("plate_x"), errors="coerce")
    normalized["plate_z"] = pd.to_numeric(normalized.get("plate_z"), errors="coerce")
    normalized["release_speed"] = pd.to_numeric(normalized.get("release_speed"), errors="coerce")
    normalized["release_spin_rate"] = pd.to_numeric(normalized.get("release_spin_rate"), errors="coerce")
    normalized["release_extension"] = pd.to_numeric(normalized.get("release_extension"), errors="coerce")
    normalized["release_pos_x"] = pd.to_numeric(normalized.get("release_pos_x"), errors="coerce")
    normalized["release_pos_z"] = pd.to_numeric(normalized.get("release_pos_z"), errors="coerce")
    normalized["pfx_x"] = pd.to_numeric(normalized.get("pfx_x"), errors="coerce")
    normalized["pfx_z"] = pd.to_numeric(normalized.get("pfx_z"), errors="coerce")
    normalized["spin_axis"] = pd.to_numeric(normalized.get("spin_axis"), errors="coerce")
    normalized["team"] = normalized.apply(
        lambda row: row["away_team"] if row.get("inning_topbot") == "Top" else row["home_team"],
        axis=1,
    )
    normalized["fielding_team"] = normalized.apply(
        lambda row: row["home_team"] if row.get("inning_topbot") == "Top" else row["away_team"],
        axis=1,
    )
    normalized["batter_name"] = normalized.get("batter_name", normalized.get("player_name", pd.Series(index=normalized.index, dtype="object")))
    normalized["pitcher_name"] = normalized.get("pitcher_name", pd.Series(index=normalized.index, dtype="object")).fillna(normalized.get("player_name"))
    normalized["player_name"] = normalized["batter_name"].fillna(normalized["player_name"])
    normalized["release_pos_y"] = pd.to_numeric(normalized.get("release_pos_y"), errors="coerce")
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


def _normalize_batter_family_zone_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    normalized["pitch_family"] = normalized.get("pitch_family", pd.Series(dtype="object")).fillna("").astype(str).str.strip().str.lower()
    normalized["zone_bucket"] = normalized.get("zone_bucket", pd.Series(dtype="object")).fillna("").astype(str).str.strip().str.lower()
    return normalized


def _read_query(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def _read_frame(conn, sql: str, params: dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params or {})
        rows = cur.fetchall()
        columns = [desc.name for desc in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=columns)


def _read_source_freshness_summary(
    conn,
    *,
    source_name: str,
    table_name: str,
    target_date: date,
    lookback_days: int,
) -> SourceFreshnessSummary:
    max_date_frame = _read_frame(
        conn,
        f"""
        SELECT
            MAX(game_date::date) AS max_event_date,
            COUNT(*)::INT8 AS row_count
        FROM {table_name}
        """,
    )
    max_event_date = None
    row_count = 0
    if not max_date_frame.empty:
        max_value = pd.to_datetime(max_date_frame.loc[0, "max_event_date"], errors="coerce")
        max_event_date = None if pd.isna(max_value) else max_value.date()
        row_count = int(pd.to_numeric(max_date_frame.loc[0, "row_count"], errors="coerce") or 0)

    start_date = target_date - timedelta(days=max(lookback_days - 1, 0))
    daily_counts = _read_frame(
        conn,
        f"""
        SELECT
            game_date::date AS event_date,
            COUNT(*)::INT8 AS row_count
        FROM {table_name}
        WHERE game_date::date BETWEEN %(start_date)s AND %(target_date)s
        GROUP BY game_date::date
        ORDER BY game_date::date
        """,
        {"start_date": start_date, "target_date": target_date},
    )
    counts_map: dict[date, int] = {}
    recent_daily_counts: list[dict[str, object]] = []
    if not daily_counts.empty:
        for _, row in daily_counts.iterrows():
            parsed_date = pd.to_datetime(row["event_date"], errors="coerce")
            if pd.isna(parsed_date):
                continue
            event_date = parsed_date.date()
            count = int(pd.to_numeric(row["row_count"], errors="coerce") or 0)
            counts_map[event_date] = count
            recent_daily_counts.append({"event_date": event_date.isoformat(), "row_count": count})

    missing_dates = [
        day.date().isoformat()
        for day in pd.date_range(start=start_date, end=target_date, freq="D")
        if counts_map.get(day.date(), 0) <= 0
    ]
    lag_days = None if max_event_date is None else (target_date - max_event_date).days
    return SourceFreshnessSummary(
        source_name=source_name,
        table_name=table_name,
        target_date=target_date,
        max_event_date=max_event_date,
        lag_days=lag_days,
        row_count=row_count,
        recent_daily_counts=recent_daily_counts,
        missing_dates=missing_dates,
    )


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
        "sweet_spot_pct": "FLOAT8",
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
        "outcome_complete": "BOOL",
        "outcome_status": "STRING",
        "source_max_event_date": "DATE",
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
        "strikeout_score": "FLOAT8",
        "raw_pitcher_score": "FLOAT8",
        "raw_strikeout_score": "FLOAT8",
        "pitcher_matchup_adjustment": "FLOAT8",
        "strikeout_matchup_adjustment": "FLOAT8",
        "opponent_lineup_quality": "FLOAT8",
        "opponent_contact_threat": "FLOAT8",
        "opponent_whiff_tendency": "FLOAT8",
        "opponent_family_fit_allowed": "FLOAT8",
        "lineup_source": "STRING",
        "lineup_hitter_count": "INT8",
        "xwoba": "FLOAT8",
        "called_strike_pct": "FLOAT8",
        "csw_pct": "FLOAT8",
        "swstr_pct": "FLOAT8",
        "putaway_pct": "FLOAT8",
        "ball_pct": "FLOAT8",
        "siera": "FLOAT8",
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
        "outcome_complete": "BOOL",
        "outcome_status": "STRING",
        "source_max_event_date": "DATE",
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
    _ensure_table_columns(conn, config.cockroach_hitter_snapshot_table, snapshot_columns)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS hitter_model_snapshots_unique_idx ON {config.cockroach_hitter_snapshot_table} (slate_date, game_pk, batter_id, split_key, recent_window, weighted_mode)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_hitter_outcome_table} ({', '.join(f'{name} {definition}' for name, definition in outcome_columns.items())})")
    _ensure_table_columns(conn, config.cockroach_hitter_outcome_table, outcome_columns)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS hitter_game_outcomes_unique_idx ON {config.cockroach_hitter_outcome_table} (slate_date, game_pk, batter_id)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_hitter_board_table} ({', '.join(f'{name} {definition}' for name, definition in board_columns.items())})")
    _ensure_table_columns(conn, config.cockroach_hitter_board_table, board_columns)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS hitter_board_winners_unique_idx ON {config.cockroach_hitter_board_table} (slate_date, board_name, board_rank)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_snapshot_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_snapshot_columns.items())})")
    _ensure_table_columns(conn, config.cockroach_pitcher_snapshot_table, pitcher_snapshot_columns)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_model_snapshots_unique_idx ON {config.cockroach_pitcher_snapshot_table} (slate_date, game_pk, pitcher_id, split_key, recent_window, weighted_mode)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_outcome_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_outcome_columns.items())})")
    _ensure_table_columns(conn, config.cockroach_pitcher_outcome_table, pitcher_outcome_columns)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_game_outcomes_unique_idx ON {config.cockroach_pitcher_outcome_table} (slate_date, game_pk, pitcher_id)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_board_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_board_columns.items())})")
    _ensure_table_columns(conn, config.cockroach_pitcher_board_table, pitcher_board_columns)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_board_winners_unique_idx ON {config.cockroach_pitcher_board_table} (slate_date, board_name, board_rank)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_arsenal_snapshot_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_arsenal_columns.items())})")
    _ensure_table_columns(conn, config.cockroach_pitcher_arsenal_snapshot_table, pitcher_arsenal_columns)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_arsenal_snapshots_unique_idx ON {config.cockroach_pitcher_arsenal_snapshot_table} (slate_date, game_pk, pitcher_id, split_key, recent_window, weighted_mode, batter_side_key, pitch_name)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_count_snapshot_table} ({', '.join(f'{name} {definition}' for name, definition in pitcher_count_columns.items())})")
    _ensure_table_columns(conn, config.cockroach_pitcher_count_snapshot_table, pitcher_count_columns)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS pitcher_count_usage_snapshots_unique_idx ON {config.cockroach_pitcher_count_snapshot_table} (slate_date, game_pk, pitcher_id, split_key, recent_window, weighted_mode, batter_side_key, pitch_name, count_bucket)")


def _table_parts(table_name: str) -> tuple[str, str]:
    if "." in table_name:
        schema_name, bare_name = table_name.split(".", 1)
        return schema_name, bare_name
    return "public", table_name


def _ensure_table_columns(conn, table_name: str, column_definitions: dict[str, str]) -> None:
    schema_name, bare_name = _table_parts(table_name)
    existing = {
        row[0]
        for row in conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema_name, bare_name),
        ).fetchall()
    }
    for column_name, definition in column_definitions.items():
        if column_name in existing:
            continue
        clean_definition = definition.replace("PRIMARY KEY", "").replace("NOT NULL", "").strip()
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column_name} {clean_definition}")


def _create_live_event_tables(conn, config: AppConfig) -> None:
    live_pitch_mix_columns = {
        "event_key": "STRING PRIMARY KEY",
        "pa_key": "STRING",
        "game_pk": "INT8",
        "game_date": "STRING",
        "source_season": "INT8",
        "batter": "INT8",
        "pitcher": "INT8",
        "batter_name": "STRING",
        "pitcher_name": "STRING",
        "player_name": "STRING",
        "stand": "STRING",
        "p_throws": "STRING",
        "home_team": "STRING",
        "away_team": "STRING",
        "inning": "INT8",
        "inning_topbot": "STRING",
        "at_bat_number": "INT8",
        "pitch_number": "INT8",
        "pitch_type": "STRING",
        "pitch_name": "STRING",
        "release_speed": "FLOAT8",
        "effective_speed": "FLOAT8",
        "release_spin_rate": "FLOAT8",
        "spin_axis": "FLOAT8",
        "pfx_x": "FLOAT8",
        "pfx_z": "FLOAT8",
        "release_pos_x": "FLOAT8",
        "release_pos_y": "FLOAT8",
        "release_pos_z": "FLOAT8",
        "release_extension": "FLOAT8",
        "plate_x": "FLOAT8",
        "plate_z": "FLOAT8",
        "zone": "INT8",
        "balls": "INT8",
        "strikes": "INT8",
        "outs_when_up": "INT8",
        "bat_score": "INT8",
        "fld_score": "INT8",
        "post_bat_score": "INT8",
        "post_fld_score": "INT8",
        "type": "STRING",
        "description": "STRING",
        "events": "STRING",
        "launch_speed": "FLOAT8",
        "launch_angle": "FLOAT8",
        "hit_distance_sc": "FLOAT8",
        "hc_x": "FLOAT8",
        "hc_y": "FLOAT8",
        "spray_angle": "FLOAT8",
        "bb_type": "STRING",
        "launch_speed_angle": "INT8",
        "barrel": "INT8",
        "estimated_ba_using_speedangle": "FLOAT8",
        "estimated_woba_using_speedangle": "FLOAT8",
        "woba_value": "FLOAT8",
        "woba_denom": "INT8",
        "delta_home_win_exp": "FLOAT8",
        "delta_run_exp": "FLOAT8",
        "created_at": "TIMESTAMP",
    }
    baseline_columns = {
        "rowid": "INT8 NOT NULL DEFAULT unique_rowid()",
        "batter_name": "STRING",
        "pitcher_name": "STRING",
        "player_name": "STRING",
        "event_key": "STRING",
        "batter": "INT8",
        "pitcher": "INT8",
        "game_date": "STRING",
        "game_pk": "INT8",
        "source_season": "INT8",
        "pitch_type": "STRING",
        "pitch_name": "STRING",
        "events": "STRING",
        "description": "STRING",
        "stand": "STRING",
        "p_throws": "STRING",
        "home_team": "STRING",
        "away_team": "STRING",
        "inning": "INT8",
        "inning_topbot": "STRING",
        "at_bat_number": "INT8",
        "pitch_number": "INT8",
        "plate_x": "FLOAT8",
        "plate_z": "FLOAT8",
        "release_speed": "FLOAT8",
        "release_spin_rate": "FLOAT8",
        "release_extension": "FLOAT8",
        "release_pos_x": "FLOAT8",
        "release_pos_z": "FLOAT8",
        "pfx_x": "FLOAT8",
        "pfx_z": "FLOAT8",
        "launch_speed": "FLOAT8",
        "launch_angle": "FLOAT8",
        "estimated_woba_using_speedangle": "FLOAT8",
        "spray_angle": "FLOAT8",
        "hc_x": "FLOAT8",
        "hc_y": "FLOAT8",
        "bb_type": "STRING",
        "balls": "FLOAT8",
        "strikes": "FLOAT8",
        "outs_when_up": "FLOAT8",
        "bat_score": "FLOAT8",
        "fld_score": "FLOAT8",
        "post_bat_score": "FLOAT8",
        "post_fld_score": "FLOAT8",
        "pitcher_hand": "STRING",
        "batter_stand": "STRING",
        "movement_magnitude": "FLOAT8",
        "spin_efficiency_proxy": "FLOAT8",
        "release_height_proxy": "FLOAT8",
        "release_side_proxy": "FLOAT8",
        "count_string": "STRING",
        "baseline_mode": "STRING",
        "prior_sample_size": "INT8",
        "season_2026_sample_size": "INT8",
        "prior_weight": "FLOAT8",
        "season_2026_weight": "FLOAT8",
        "baseline_driver": "STRING",
        "rolling_overlay_active": "BOOL",
        "baseline_role": "STRING",
        "baseline_source": "STRING",
        "snapshot_built_at": "STRING",
        "snapshot_version": "STRING",
        "source_status": "STRING",
    }
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {config.cockroach_live_pitch_mix_table} ({', '.join(f'{name} {definition}' for name, definition in live_pitch_mix_columns.items())})"
    )
    _ensure_table_columns(conn, config.cockroach_live_pitch_mix_table, live_pitch_mix_columns)
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_lpm_game_pk ON {config.cockroach_live_pitch_mix_table} (game_pk, event_key)"
    )
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_lpm_pa_key ON {config.cockroach_live_pitch_mix_table} (pa_key, event_key)"
    )
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_lpm_batter_date ON {config.cockroach_live_pitch_mix_table} (batter, game_date, event_key)"
    )

    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {config.cockroach_pitcher_baseline_event_table} ({', '.join(f'{name} {definition}' for name, definition in baseline_columns.items())}, PRIMARY KEY (rowid))"
    )
    _ensure_table_columns(conn, config.cockroach_pitcher_baseline_event_table, baseline_columns)
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_shared_pitcher_baseline_event_rows_player_name ON {config.cockroach_pitcher_baseline_event_table} (player_name)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_shared_pitcher_baseline_event_rows_player_date ON {config.cockroach_pitcher_baseline_event_table} (player_name, game_date)"
    )


def _create_props_odds_table(conn, config: AppConfig) -> None:
    columns = {
        "rowid": "INT8 NOT NULL DEFAULT unique_rowid()",
        "fetched_at": "STRING",
        "cache_key": "STRING",
        "provider": "STRING",
        "event_id": "STRING",
        "commence_time": "STRING",
        "away_team": "STRING",
        "home_team": "STRING",
        "sportsbook": "STRING",
        "sportsbook_key": "STRING",
        "market_key": "STRING",
        "market": "STRING",
        "player_name_raw": "STRING",
        "player_name": "STRING",
        "odds_american": "INT8",
        "line": "FLOAT8",
        "selection_label": "STRING",
        "selection_scope": "STRING",
        "selection_side": "STRING",
        "market_family": "STRING",
        "market_variant": "STRING",
        "threshold": "INT8",
        "display_label": "STRING",
        "is_primary_line": "BOOL",
        "is_modeled": "BOOL",
        "player_event_market_key": "STRING",
        "row_source_type": "STRING",
        "coverage_completion_status": "STRING",
        "hr_books_requested": "STRING",
        "hr_books_present": "STRING",
        "hr_books_missing": "STRING",
    }
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {config.cockroach_props_odds_table} ({', '.join(f'{name} {definition}' for name, definition in columns.items())}, PRIMARY KEY (rowid))"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS props_odds_lookup_idx ON {config.cockroach_props_odds_table} (market_key, player_name, fetched_at, event_id, rowid)"
    )


def _prepare_records(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    work = frame.loc[:, columns].copy().astype(object)
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


def _insert_frame(conn, table_name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    columns = list(frame.columns)
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
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


def read_hitter_backtest_data(
    config: AppConfig,
    start_date,
    end_date,
    split_key: str | None = None,
    recent_window: str | None = None,
    weighted_mode: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_driver()
    if not config.database_url:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    where = ["slate_date BETWEEN %(start_date)s AND %(end_date)s"]
    params: dict[str, object] = {"start_date": start_date, "end_date": end_date}
    if split_key:
        where.append("split_key = %(split_key)s")
        params["split_key"] = split_key
    if recent_window:
        where.append("recent_window = %(recent_window)s")
        params["recent_window"] = recent_window
    if weighted_mode:
        where.append("weighted_mode = %(weighted_mode)s")
        params["weighted_mode"] = weighted_mode
    where_sql = " AND ".join(where)
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url, autocommit=True) as conn:
        snapshots = _read_frame(
            conn,
            f"""
            SELECT *
            FROM {config.cockroach_hitter_snapshot_table}
            WHERE {where_sql}
            ORDER BY slate_date DESC, game_pk, matchup_score DESC NULLS LAST
            """,
            params,
        )
        outcomes = _read_frame(
            conn,
            f"""
            SELECT *
            FROM {config.cockroach_hitter_outcome_table}
            WHERE slate_date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY slate_date DESC, game_pk, hitter_name
            """,
            {"start_date": start_date, "end_date": end_date},
        )
        boards = _read_frame(
            conn,
            f"""
            SELECT *
            FROM {config.cockroach_hitter_board_table}
            WHERE slate_date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY slate_date DESC, board_name, board_rank
            """,
            {"start_date": start_date, "end_date": end_date},
        )
    return snapshots, outcomes, boards


def read_pitcher_backtest_data(
    config: AppConfig,
    start_date,
    end_date,
    split_key: str | None = None,
    recent_window: str | None = None,
    weighted_mode: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_driver()
    if not config.database_url:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    where = ["slate_date BETWEEN %(start_date)s AND %(end_date)s"]
    params: dict[str, object] = {"start_date": start_date, "end_date": end_date}
    if split_key:
        where.append("split_key = %(split_key)s")
        params["split_key"] = split_key
    if recent_window:
        where.append("recent_window = %(recent_window)s")
        params["recent_window"] = recent_window
    if weighted_mode:
        where.append("weighted_mode = %(weighted_mode)s")
        params["weighted_mode"] = weighted_mode
    where_sql = " AND ".join(where)
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url, autocommit=True) as conn:
        snapshots = _read_frame(
            conn,
            f"""
            SELECT *
            FROM {config.cockroach_pitcher_snapshot_table}
            WHERE {where_sql}
            ORDER BY slate_date DESC, game_pk, pitcher_score DESC NULLS LAST
            """,
            params,
        )
        outcomes = _read_frame(
            conn,
            f"""
            SELECT *
            FROM {config.cockroach_pitcher_outcome_table}
            WHERE slate_date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY slate_date DESC, game_pk, pitcher_name
            """,
            {"start_date": start_date, "end_date": end_date},
        )
        boards = _read_frame(
            conn,
            f"""
            SELECT *
            FROM {config.cockroach_pitcher_board_table}
            WHERE slate_date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY slate_date DESC, board_name, board_rank
            """,
            {"start_date": start_date, "end_date": end_date},
        )
    return snapshots, outcomes, boards


def read_prop_odds_history(
    config: AppConfig,
    start_date,
    end_date,
    markets: tuple[str, ...] = ("batter_home_runs", "pitcher_strikeouts"),
) -> pd.DataFrame:
    _ensure_driver()
    if not config.database_url:
        return pd.DataFrame()
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url, autocommit=True) as conn:
        return _read_frame(
            conn,
            f"""
            SELECT fetched_at, cache_key, provider, event_id, commence_time, away_team, home_team, sportsbook,
                   sportsbook_key, market_key, market, player_name_raw, player_name, odds_american, line,
                   selection_label, selection_scope, selection_side, market_family, market_variant, threshold,
                   display_label, is_primary_line, is_modeled, player_event_market_key, row_source_type,
                   coverage_completion_status, hr_books_requested, hr_books_present, hr_books_missing
            FROM {config.cockroach_props_odds_table}
            WHERE market_key = ANY(%(markets)s)
              AND CAST(substr(commence_time, 1, 10) AS DATE) BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY fetched_at DESC, player_name, market_key, sportsbook
            """,
            {"markets": list(markets), "start_date": start_date, "end_date": end_date},
        )


def read_latest_prop_odds_snapshot(
    config: AppConfig,
    target_date,
    markets: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    _ensure_driver()
    if not config.database_url:
        return pd.DataFrame()
    database_url = _normalize_database_url(config.database_url)
    market_filter = ""
    params: dict[str, object] = {"target_date": target_date}
    if markets:
        market_filter = "AND market_key = ANY(%(markets)s)"
        params["markets"] = list(markets)
    with psycopg.connect(database_url, autocommit=True) as conn:
        latest = _read_frame(
            conn,
            f"""
            SELECT fetched_at, cache_key
            FROM {config.cockroach_props_odds_table}
            WHERE CAST(substr(commence_time, 1, 10) AS DATE) = %(target_date)s
              {market_filter}
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            params,
        )
        if latest.empty:
            return pd.DataFrame()
        fetched_at = latest["fetched_at"].iloc[0]
        cache_key = latest["cache_key"].iloc[0]
        return _read_frame(
            conn,
            f"""
            SELECT fetched_at, cache_key, provider, event_id, commence_time, away_team, home_team, sportsbook,
                   sportsbook_key, market_key, market, player_name_raw, player_name, odds_american, line,
                   selection_label, selection_scope, selection_side, market_family, market_variant, threshold,
                   display_label, is_primary_line, is_modeled, player_event_market_key, row_source_type,
                   coverage_completion_status, hr_books_requested, hr_books_present, hr_books_missing
            FROM {config.cockroach_props_odds_table}
            WHERE fetched_at = %(fetched_at)s
              AND cache_key = %(cache_key)s
            ORDER BY player_name, market_key, sportsbook
            """,
            {"fetched_at": fetched_at, "cache_key": cache_key},
        )


def read_hitter_exit_velo_events(
    config: AppConfig,
    end_date=None,
) -> pd.DataFrame:
    _ensure_driver()
    columns = [
        "game_date",
        "game_pk",
        "away_team",
        "home_team",
        "inning_topbot",
        "batter",
        "batter_name",
        "pitcher_name",
        "player_name",
        "at_bat_number",
        "pitch_number",
        "pitch_type",
        "pitch_name",
        "zone",
        "stand",
        "hc_x",
        "bb_type",
        "events",
        "launch_speed",
        "launch_angle",
        "release_speed",
    ]
    if not config.database_url:
        raise RuntimeError("DATABASE_URL must be set to load exit velocity results from Cockroach.")
    database_url = _normalize_database_url(config.database_url)
    where = [
        "batter IS NOT NULL",
        "launch_speed IS NOT NULL",
        "launch_angle IS NOT NULL",
        "CAST(game_date AS DATE) >= DATE '2026-01-01'",
    ]
    params: dict[str, object] = {}
    if end_date is not None:
        where.append("CAST(game_date AS DATE) <= %(end_date)s")
        params["end_date"] = end_date
    where_sql = " AND ".join(where)
    with psycopg.connect(database_url, autocommit=True) as conn:
        frame = _read_frame(
            conn,
            f"""
            WITH base AS (
                SELECT game_date, game_pk, away_team, home_team, inning_topbot, batter, batter_name, pitcher_name, player_name,
                       at_bat_number, pitch_number, pitch_type, pitch_name, zone, stand, hc_x, bb_type, events, launch_speed, launch_angle, release_speed
                FROM {config.cockroach_pitcher_baseline_event_table}
                WHERE {where_sql}
            ),
            ranked_games AS (
                SELECT batter, game_date, game_pk,
                       DENSE_RANK() OVER (
                           PARTITION BY batter
                           ORDER BY CAST(game_date AS DATE) DESC, game_pk DESC
                       ) AS recent_game_rank
                FROM (
                    SELECT DISTINCT batter, game_date, game_pk
                    FROM base
                )
            )
            SELECT b.game_date, b.game_pk, b.away_team, b.home_team, b.inning_topbot, b.batter, b.batter_name, b.pitcher_name, b.player_name,
                   b.at_bat_number, b.pitch_number, b.pitch_type, b.pitch_name, b.zone, b.stand, b.hc_x, b.bb_type, b.events, b.launch_speed, b.launch_angle, b.release_speed
            FROM base b
            JOIN ranked_games g
              ON b.batter = g.batter
             AND b.game_date = g.game_date
             AND b.game_pk = g.game_pk
            WHERE g.recent_game_rank <= 25
            ORDER BY b.game_date DESC, b.game_pk DESC, b.at_bat_number DESC, b.pitch_number DESC
            """,
            params,
        )
    if frame.empty:
        return pd.DataFrame(columns=columns + ["team", "opponent", "game_label"])
    work = frame.copy()
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    work["launch_speed"] = pd.to_numeric(work["launch_speed"], errors="coerce")
    work["launch_angle"] = pd.to_numeric(work["launch_angle"], errors="coerce")
    work["release_speed"] = pd.to_numeric(work["release_speed"], errors="coerce")
    work["zone"] = pd.to_numeric(work["zone"], errors="coerce")
    work["hc_x"] = pd.to_numeric(work["hc_x"], errors="coerce")
    work["batter"] = pd.to_numeric(work["batter"], errors="coerce")
    work["at_bat_number"] = pd.to_numeric(work["at_bat_number"], errors="coerce")
    work["pitch_number"] = pd.to_numeric(work["pitch_number"], errors="coerce")
    work = work.loc[
        work["game_date"].notna()
        & work["launch_speed"].notna()
        & work["launch_angle"].notna()
        & work["batter"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=columns + ["team", "opponent", "game_label"])
    work["batter"] = work["batter"].astype(int)
    work["team"] = work["away_team"].where(work["inning_topbot"].eq("Top"), work["home_team"])
    work["opponent"] = work["home_team"].where(work["team"].eq(work["away_team"]), work["away_team"])
    work["game_label"] = work["away_team"].fillna("").astype(str) + " @ " + work["home_team"].fillna("").astype(str)
    return work[
        [
            "game_date",
            "game_pk",
            "away_team",
            "home_team",
            "inning_topbot",
            "batter",
            "batter_name",
            "pitcher_name",
            "player_name",
            "team",
            "opponent",
            "game_label",
            "at_bat_number",
            "pitch_number",
            "pitch_type",
            "pitch_name",
            "zone",
            "stand",
            "hc_x",
            "bb_type",
            "events",
            "launch_speed",
            "launch_angle",
            "release_speed",
        ]
    ].reset_index(drop=True)


def read_recent_batter_name_lookup(
    config: AppConfig,
    *,
    end_date: date | None = None,
    lookback_days: int = 45,
) -> pd.DataFrame:
    _ensure_driver()
    if not config.database_url:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    database_url = _normalize_database_url(config.database_url)
    params: dict[str, object] = {
        "season_start": date(2026, 1, 1),
        "lookback_days": max(int(lookback_days), 1),
    }
    end_filter = ""
    if end_date is not None:
        params["end_date"] = end_date
        end_filter = "AND CAST(game_date AS DATE) <= %(end_date)s"
    with psycopg.connect(database_url, autocommit=True) as conn:
        frame = _read_frame(
            conn,
            f"""
            WITH base AS (
                SELECT
                    batter,
                    CASE
                        WHEN inning_topbot = 'Top' THEN away_team
                        ELSE home_team
                    END AS team,
                    COALESCE(NULLIF(BTRIM(batter_name), ''), NULLIF(BTRIM(player_name), '')) AS player_name,
                    CAST(game_date AS DATE) AS game_date
                FROM {config.cockroach_live_pitch_mix_table}
                WHERE batter IS NOT NULL
                  AND COALESCE(NULLIF(BTRIM(batter_name), ''), NULLIF(BTRIM(player_name), '')) IS NOT NULL
                  AND CAST(game_date AS DATE) >= %(season_start)s
                  {end_filter}
            ),
            max_date AS (
                SELECT COALESCE(MAX(game_date), %(season_start)s) AS latest_date
                FROM base
            ),
            trimmed AS (
                SELECT b.*
                FROM base b
                CROSS JOIN max_date m
                WHERE b.game_date >= m.latest_date - (%(lookback_days)s::INT * INTERVAL '1 day')
            ),
            counted AS (
                SELECT
                    batter,
                    team,
                    player_name,
                    MAX(game_date) AS last_seen_date,
                    COUNT(*) AS name_count
                FROM trimmed
                GROUP BY batter, team, player_name
            ),
            ranked AS (
                SELECT
                    batter,
                    team,
                    player_name,
                    ROW_NUMBER() OVER (
                        PARTITION BY batter, team
                        ORDER BY name_count DESC, last_seen_date DESC, player_name
                    ) AS choice_rank
                FROM counted
            )
            SELECT batter, team, player_name
            FROM ranked
            WHERE choice_rank = 1
            """,
            params,
        )
    if frame.empty:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    work = frame.copy()
    work["batter"] = pd.to_numeric(work["batter"], errors="coerce")
    work = work.loc[work["batter"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    work["batter"] = work["batter"].astype(int)
    work["team"] = work["team"].fillna("").astype(str).str.strip().str.upper()
    work["player_name"] = work["player_name"].fillna("").astype(str).str.strip()
    work = work.loc[work["player_name"].ne("")].drop_duplicates(["batter", "team"])
    return work.reset_index(drop=True)


def read_source_freshness_report(
    config: AppConfig,
    *,
    target_date: date | None = None,
    lookback_days: int = 7,
) -> list[dict[str, object]]:
    _ensure_driver()
    if not config.database_url:
        return []
    check_date = target_date or date.today()
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url, autocommit=True) as conn:
        summaries = [
            _read_source_freshness_summary(
                conn,
                source_name="live_pitch_mix",
                table_name=config.cockroach_live_pitch_mix_table,
                target_date=check_date,
                lookback_days=lookback_days,
            ),
            _read_source_freshness_summary(
                conn,
                source_name="pitcher_baseline_event_rows",
                table_name=config.cockroach_pitcher_baseline_event_table,
                target_date=check_date,
                lookback_days=lookback_days,
            ),
        ]
    return [asdict(summary) | {"is_fresh": summary.is_fresh} for summary in summaries]


def write_props_odds_snapshot(config: AppConfig, frame: pd.DataFrame) -> None:
    _ensure_driver()
    if not config.database_url or frame.empty:
        return
    work = frame.copy()
    if "sportsbook" not in work.columns and "book_title" in work.columns:
        work["sportsbook"] = work["book_title"]
    if "sportsbook_key" not in work.columns and "book_key" in work.columns:
        work["sportsbook_key"] = work["book_key"]
    for column in [
        "sportsbook",
        "sportsbook_key",
        "player_name_raw",
        "player_name",
        "selection_label",
        "selection_scope",
        "selection_side",
        "market_family",
        "market_variant",
        "display_label",
        "row_source_type",
        "coverage_completion_status",
        "hr_books_requested",
        "hr_books_present",
        "hr_books_missing",
    ]:
        if column not in work.columns:
            work[column] = pd.NA
    for column in ["fetched_at", "cache_key", "provider", "event_id", "commence_time", "away_team", "home_team", "market_key", "market"]:
        if column not in work.columns:
            raise KeyError(f"Missing required odds snapshot column: {column}")
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url, autocommit=True) as conn:
        _create_props_odds_table(conn, config)
        _insert_frame(
            conn,
            config.cockroach_props_odds_table,
            work[
                [
                    "fetched_at",
                    "cache_key",
                    "provider",
                    "event_id",
                    "commence_time",
                    "away_team",
                    "home_team",
                    "sportsbook",
                    "sportsbook_key",
                    "market_key",
                    "market",
                    "player_name_raw",
                    "player_name",
                    "odds_american",
                    "line",
                    "selection_label",
                    "selection_scope",
                    "selection_side",
                    "market_family",
                    "market_variant",
                    "threshold",
                    "display_label",
                    "is_primary_line",
                    "is_modeled",
                    "player_event_market_key",
                    "row_source_type",
                    "coverage_completion_status",
                    "hr_books_requested",
                    "hr_books_present",
                    "hr_books_missing",
                ]
            ].copy(),
        )


def replace_live_event_payload(
    config: AppConfig,
    live_pitch_mix: pd.DataFrame,
    baseline_event_rows: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
) -> None:
    _ensure_driver()
    if not config.database_url:
        raise RuntimeError("DATABASE_URL must be set to write live event payloads.")
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url) as conn:
        _create_live_event_tables(conn, config)
        live_game_pks = sorted({int(value) for value in pd.to_numeric(live_pitch_mix.get("game_pk"), errors="coerce").dropna().astype(int).tolist()}) if not live_pitch_mix.empty else []
        baseline_game_pks = sorted({int(value) for value in pd.to_numeric(baseline_event_rows.get("game_pk"), errors="coerce").dropna().astype(int).tolist()}) if not baseline_event_rows.empty else []
        if live_game_pks:
            conn.execute(
                f"DELETE FROM {config.cockroach_live_pitch_mix_table} WHERE game_pk = ANY(%s)",
                (live_game_pks,),
            )
        else:
            conn.execute(
                f"DELETE FROM {config.cockroach_live_pitch_mix_table} WHERE CAST(game_date AS DATE) BETWEEN %s AND %s",
                (start_date, end_date),
            )
        if baseline_game_pks:
            conn.execute(
                f"DELETE FROM {config.cockroach_pitcher_baseline_event_table} WHERE game_pk = ANY(%s)",
                (baseline_game_pks,),
            )
        else:
            conn.execute(
                f"DELETE FROM {config.cockroach_pitcher_baseline_event_table} WHERE CAST(game_date AS DATE) BETWEEN %s AND %s",
                (start_date, end_date),
            )
        _insert_frame(conn, config.cockroach_live_pitch_mix_table, live_pitch_mix)
        baseline_columns = [column for column in baseline_event_rows.columns if column != "rowid"]
        _insert_frame(conn, config.cockroach_pitcher_baseline_event_table, baseline_event_rows.loc[:, baseline_columns].copy())
        conn.commit()


def load_cockroach_payload(config: AppConfig) -> CockroachPayload:
    _ensure_driver()
    if not config.database_url:
        raise RuntimeError("DATABASE_URL must be set to pull 2026 Cockroach data.")
    database_url = _normalize_database_url(config.database_url)
    with psycopg.connect(database_url, autocommit=True) as conn:
        live_pitch_mix = _read_query(conn, f"SELECT * FROM {config.cockroach_live_pitch_mix_table}")
        pitcher_baseline_event_rows = _read_query(conn, f"SELECT * FROM {config.cockroach_pitcher_baseline_event_table}")
        hitter_rolling = _read_query(conn, f"SELECT * FROM {config.cockroach_hitter_rolling_table}")
        pitcher_rolling = _read_query(conn, f"SELECT * FROM {config.cockroach_pitcher_rolling_table}")
        batter_zone_profiles = _read_query(conn, f"SELECT * FROM {config.cockroach_batter_zone_table}")
        pitcher_zone_profiles = _read_query(conn, f"SELECT * FROM {config.cockroach_pitcher_zone_table}")
        batter_family_zone_profiles = _read_query(conn, f"SELECT * FROM {config.cockroach_batter_family_zone_table}")
    live_pitch_mix = _normalize_live_pitch_mix(live_pitch_mix)
    pitcher_baseline_event_rows = _normalize_pitcher_baseline_event_rows(pitcher_baseline_event_rows)
    hitter_rolling_mapped = pd.concat([_map_hitter_rolling(hitter_rolling), _compute_hitter_rolling_15(live_pitch_mix)], ignore_index=True, sort=False)
    pitcher_rolling_mapped = _map_pitcher_rolling(pitcher_rolling, live_pitch_mix)
    batter_zone_profiles = _normalize_batter_zone_profiles(batter_zone_profiles)
    pitcher_zone_profiles = _normalize_pitcher_zone_profiles(pitcher_zone_profiles)
    batter_family_zone_profiles = _normalize_batter_family_zone_profiles(batter_family_zone_profiles)
    return CockroachPayload(
        live_pitch_mix=live_pitch_mix,
        pitcher_baseline_event_rows=pitcher_baseline_event_rows,
        hitter_rolling=hitter_rolling_mapped,
        pitcher_rolling=pitcher_rolling_mapped,
        batter_zone_profiles=batter_zone_profiles,
        pitcher_zone_profiles=pitcher_zone_profiles,
        batter_family_zone_profiles=batter_family_zone_profiles,
    )
