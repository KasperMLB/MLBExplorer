from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .branding import apply_branding_head, page_icon_path
from .config import AppConfig
from .local_store import read_hitter_exit_velo_events, read_recent_batter_name_lookup
from .dashboard_views import latest_built_date
from .query_engine import StatcastQueryEngine, load_remote_parquet
from .ui_components import render_custom_metric_table, render_exit_velo_summary_grid


WINDOWS = [1, 3, 5, 10, 15, 25]
SORT_DEPTH = 20
EVENT_LOG_LIMIT = 250
REMOTE_EXIT_VELO_LOOKBACK_DAYS = 75
EXIT_VELO_SOURCE_CACHE_VERSION = "2026-04-14-source-v2"
EXIT_VELO_EVENT_COLUMNS = [
    "game_date", "game_pk", "away_team", "home_team", "inning_topbot", "batter", "batter_name", "pitcher_name",
    "player_name", "team", "opponent", "game_label", "at_bat_number", "pitch_number", "pitch_type", "pitch_name",
    "p_throws", "zone", "plate_x", "plate_z", "stand", "hc_x", "bb_type", "events", "launch_speed", "launch_angle",
    "release_speed",
]
REMOTE_STATCAST_COLUMNS = [
    "game_date", "game_pk", "away_team", "home_team", "inning_topbot", "batter", "batter_name", "pitcher_name",
    "player_name", "at_bat_number", "pitch_number", "pitch_type", "pitch_name", "p_throws", "zone", "plate_x",
    "plate_z", "stand", "hc_x", "bb_type", "events", "launch_speed", "launch_angle", "release_speed",
]
ZONE_DISPLAY_MAP = {
    1: "Upper Inside",
    2: "Upper Middle",
    3: "Upper Outside",
    4: "Middle Inside",
    5: "Middle Middle",
    6: "Middle Outside",
    7: "Lower Inside",
    8: "Lower Middle",
    9: "Lower Outside",
    11: "Above Zone",
    12: "Inside Off Plate",
    13: "Outside Off Plate",
    14: "Below Zone",
}
EXIT_VELO_METRIC_STYLES = {
    "Exit Velo": {"mode": "high", "low": 65.0, "high": 112.0},
    "EV": {"mode": "high", "low": 65.0, "high": 112.0},
    "Pitch Velo": {"mode": "high", "low": 80.0, "high": 100.0},
    "LA": {"mode": "target", "low": -30.0, "ideal": 18.0, "high": 50.0},
    "HH%": {"mode": "high", "low": 0.0, "high": 100.0},
    "PFB%": {"mode": "high", "low": 0.0, "high": 100.0},
    "FB%": {"mode": "high", "low": 0.0, "high": 100.0},
    "Brl": {"mode": "high", "low": 0.0, "high": 6.0},
    "Avg EV": {"mode": "high", "low": 65.0, "high": 112.0},
    "Max EV": {"mode": "high", "low": 65.0, "high": 112.0},
}


def _hosted_base_url() -> str:
    import os

    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


def _empty_board() -> pd.DataFrame:
    return pd.DataFrame(columns=["Player", "Team"])


def _normalize_name(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().split())


def _normalize_name_series(series: pd.Series) -> pd.Series:
    safe = pd.Series(series, index=series.index, dtype="object")
    safe = safe.where(pd.notna(safe), None)
    return safe.map(_normalize_name)


def _normalize_pitch_label(pitch_name: object, pitch_type: object) -> str:
    name = _normalize_name(pitch_name)
    if name.casefold() == "fastball":
        return "Four-Seam Fastball"
    if name:
        return name
    code = _normalize_name(pitch_type)
    if code.casefold() == "ff":
        return "Four-Seam Fastball"
    return code if code else "Unknown"


def _zone_filter_label(zone: object) -> str:
    zone_value = pd.to_numeric(pd.Series([zone]), errors="coerce").iloc[0]
    if pd.isna(zone_value):
        return "Unknown"
    return ZONE_DISPLAY_MAP.get(int(zone_value), "Unknown")


def _zone_filter_sort_key(value: object) -> tuple[int, int | str]:
    text = _normalize_name(value)
    reverse_map = {label: zone_id for zone_id, label in ZONE_DISPLAY_MAP.items()}
    if text in reverse_map:
        return (0, reverse_map[text])
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return (0, int(numeric))
    return (1, text)


def _handedness_filter_label(p_throws: object) -> str:
    hand = _normalize_name(p_throws).upper()
    if hand == "L":
        return "vs LHP"
    if hand == "R":
        return "vs RHP"
    return "Unknown"


def _is_statcast_barrel(exit_velocity: object, launch_angle: object) -> bool:
    ev = pd.to_numeric(pd.Series([exit_velocity]), errors="coerce").iloc[0]
    la = pd.to_numeric(pd.Series([launch_angle]), errors="coerce").iloc[0]
    if pd.isna(ev) or pd.isna(la) or float(ev) < 98.0:
        return False
    ev_value = min(float(ev), 116.0)
    half_window = min(22.0, 4.0 + ((ev_value - 98.0) * (22.0 / 18.0)))
    lower_bound = 26.0 - half_window
    upper_bound = 26.0 + half_window
    return lower_bound <= float(la) <= upper_bound


def _is_pulled_fly_ball(bb_type: object, stand: object, hc_x: object) -> bool:
    batted_ball_type = "" if bb_type is None else str(bb_type).strip().lower()
    batter_stand = "" if stand is None else str(stand).strip().upper()
    hit_coord_x = pd.to_numeric(pd.Series([hc_x]), errors="coerce").iloc[0]
    if batted_ball_type != "fly_ball" or pd.isna(hit_coord_x):
        return False
    return (batter_stand == "R" and float(hit_coord_x) < 125.0) or (batter_stand == "L" and float(hit_coord_x) > 125.0)


@st.cache_data(show_spinner=False, ttl=300)
def _load_exit_velo_events_cached(end_date_value: str | None) -> pd.DataFrame:
    config = AppConfig()
    parsed_end_date = date.fromisoformat(end_date_value) if end_date_value else None
    return read_hitter_exit_velo_events(config, end_date=parsed_end_date)


def _shape_exit_velo_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=EXIT_VELO_EVENT_COLUMNS)
    work = events.loc[
        pd.to_numeric(events.get("launch_speed"), errors="coerce").notna()
        & pd.to_numeric(events.get("launch_angle"), errors="coerce").notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=EXIT_VELO_EVENT_COLUMNS)
    for column in REMOTE_STATCAST_COLUMNS:
        if column not in work.columns:
            work[column] = pd.NA
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    away_team = work.get("away_team", pd.Series("", index=work.index)).fillna("").astype(str).str.upper()
    home_team = work.get("home_team", pd.Series("", index=work.index)).fillna("").astype(str).str.upper()
    topbot = work.get("inning_topbot", pd.Series("", index=work.index)).fillna("").astype(str)
    work["team"] = away_team.where(topbot.eq("Top"), home_team)
    work["opponent"] = home_team.where(work["team"].eq(away_team), away_team)
    work["game_label"] = away_team + " @ " + home_team
    ranked_games = (
        work[["batter", "game_date", "game_pk"]]
        .drop_duplicates()
        .sort_values(["batter", "game_date", "game_pk"], ascending=[True, False, False], na_position="last")
    )
    ranked_games["recent_game_rank"] = ranked_games.groupby("batter").cumcount() + 1
    work = work.merge(ranked_games.loc[ranked_games["recent_game_rank"].le(25)], on=["batter", "game_date", "game_pk"], how="inner")
    for column in EXIT_VELO_EVENT_COLUMNS:
        if column not in work.columns:
            work[column] = pd.NA
    return work[EXIT_VELO_EVENT_COLUMNS].sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60)
def _load_remote_exit_velo_events_cached(base_url: str, end_date_value: str | None, cache_version: str) -> pd.DataFrame:
    end = date.fromisoformat(end_date_value) if end_date_value else date.today()
    days = [end - timedelta(days=offset) for offset in range(REMOTE_EXIT_VELO_LOOKBACK_DAYS)]
    frames: list[pd.DataFrame] = []

    def _load_day(day: date) -> pd.DataFrame:
        return load_remote_parquet(
            f"{base_url.rstrip('/')}/sources/statcast_events/year={day.year}",
            f"game_date={day.isoformat()}.parquet",
            columns=REMOTE_STATCAST_COLUMNS,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_load_day, day): day for day in days}
        for future in as_completed(futures):
            try:
                frame = future.result()
            except Exception:
                continue
            if frame.empty:
                continue
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=EXIT_VELO_EVENT_COLUMNS)
    return _shape_exit_velo_events(pd.concat(frames, ignore_index=True, sort=False))


@st.cache_data(show_spinner=False, ttl=60)
def _probe_remote_exit_velo_date_cached(base_url: str, probe_date_value: str, cache_version: str) -> tuple[int, str]:
    probe_date = date.fromisoformat(probe_date_value)
    try:
        frame = load_remote_parquet(
            f"{base_url.rstrip('/')}/sources/statcast_events/year={probe_date.year}",
            f"game_date={probe_date.isoformat()}.parquet",
            columns=["game_date", "launch_speed", "launch_angle"],
        )
    except Exception as exc:
        return 0, f"error:{type(exc).__name__}"
    tracked = int(
        (
            pd.to_numeric(frame.get("launch_speed"), errors="coerce").notna()
            & pd.to_numeric(frame.get("launch_angle"), errors="coerce").notna()
        ).sum()
    )
    return int(len(frame)), f"tracked={tracked}"


def _max_event_date_label(frame: pd.DataFrame) -> str:
    if frame.empty or "game_date" not in frame.columns:
        return "-"
    max_date = pd.to_datetime(frame["game_date"], errors="coerce").max()
    if pd.isna(max_date):
        return "-"
    return max_date.date().isoformat()


@st.cache_data(show_spinner=False, ttl=300)
def _load_recent_batter_names_cached(end_date_value: str | None) -> pd.DataFrame:
    config = AppConfig()
    parsed_end_date = date.fromisoformat(end_date_value) if end_date_value else None
    return read_recent_batter_name_lookup(config, end_date=parsed_end_date)


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    work = events.copy()
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    games = (
        work[["batter", "game_pk", "game_date"]]
        .drop_duplicates()
        .sort_values(["batter", "game_date", "game_pk"], ascending=[True, False, False], na_position="last")
    )
    games["recent_game_rank"] = games.groupby("batter").cumcount() + 1
    work = work.merge(games, on=["batter", "game_pk", "game_date"], how="left")
    work["player_label"] = work["player_name"].fillna("").astype(str)
    work["event_pair"] = (
        pd.to_numeric(work["launch_speed"], errors="coerce").round(1).map(lambda value: "" if pd.isna(value) else f"{float(value):.1f}")
        + "/"
        + pd.to_numeric(work["launch_angle"], errors="coerce").round(1).map(lambda value: "" if pd.isna(value) else f"{float(value):.1f}")
    )
    work["pa_number"] = pd.to_numeric(work["at_bat_number"], errors="coerce")
    work["is_hard_hit"] = pd.to_numeric(work["launch_speed"], errors="coerce").ge(95.0)
    work["is_barrel"] = [
        _is_statcast_barrel(exit_velocity, launch_angle)
        for exit_velocity, launch_angle in zip(work["launch_speed"], work["launch_angle"])
    ]
    work["is_pulled_fly_ball"] = [
        _is_pulled_fly_ball(bb_type, stand, hc_x)
        for bb_type, stand, hc_x in zip(work["bb_type"], work.get("stand", pd.Series(index=work.index)), work.get("hc_x", pd.Series(index=work.index)))
    ]
    work["pitch_filter_label"] = [
        _normalize_pitch_label(pitch_name, pitch_type)
        for pitch_name, pitch_type in zip(
            work.get("pitch_name", pd.Series(index=work.index)),
            work.get("pitch_type", pd.Series(index=work.index)),
        )
    ]
    work["zone_filter_label"] = [
        _zone_filter_label(zone)
        for zone in work.get("zone", pd.Series(index=work.index))
    ]
    work["handedness_filter_label"] = [
        _handedness_filter_label(p_throws)
        for p_throws in work.get("p_throws", pd.Series(index=work.index))
    ]
    work["barrel_count"] = pd.Series(work["is_barrel"], index=work.index).fillna(False).astype(bool).astype(int)
    work["inning_number"] = pd.NA
    if "inning" in work.columns:
        work["inning_number"] = pd.to_numeric(work["inning"], errors="coerce")
    return work


@st.cache_data(show_spinner=False, ttl=300)
def _load_rosters(config: AppConfig, end_date: date | None) -> pd.DataFrame:
    target_date = end_date or latest_built_date(config.daily_dir) or date.today()
    local_path = config.daily_dir / target_date.isoformat() / "rosters.parquet"
    if local_path.exists():
        engine = StatcastQueryEngine(config)
        return engine.load_daily_rosters(target_date)
    base_url = _hosted_base_url()
    if not base_url:
        return pd.DataFrame(columns=["team", "player_id", "player_name"])
    for offset in range(21):
        candidate = target_date - timedelta(days=offset)
        try:
            rosters = load_remote_parquet(f"{base_url}/daily/{candidate.isoformat()}", "rosters.parquet")
            if not rosters.empty:
                return rosters
        except Exception:
            continue
    return pd.DataFrame(columns=["team", "player_id", "player_name"])


@st.cache_data(show_spinner=False, ttl=300)
def _load_hitter_artifact_names(config: AppConfig, end_date: date | None) -> pd.DataFrame:
    local_path = config.reusable_dir / "hitter_metrics.parquet"
    frame = pd.DataFrame()
    if local_path.exists():
        try:
            frame = pd.read_parquet(local_path)
        except Exception:
            frame = pd.DataFrame()
    elif _hosted_base_url():
        try:
            frame = load_remote_parquet(f"{_hosted_base_url()}/reusable", "hitter_metrics.parquet")
        except Exception:
            frame = pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    batter_column = next((column for column in ["hitter_id", "batter", "player_id"] if column in frame.columns), None)
    name_column = next((column for column in ["hitter_name", "player_name"] if column in frame.columns), None)
    if batter_column is None or name_column is None:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    lookup_columns = [batter_column, name_column]
    if "team" in frame.columns:
        lookup_columns.append("team")
    lookup = frame.loc[:, lookup_columns].copy()
    lookup = lookup.rename(columns={batter_column: "batter", name_column: "player_name"})
    lookup["batter"] = pd.to_numeric(lookup["batter"], errors="coerce")
    lookup = lookup.loc[lookup["batter"].notna()].copy()
    if lookup.empty:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    lookup["batter"] = lookup["batter"].astype(int)
    lookup["player_name"] = _normalize_name_series(lookup["player_name"])
    lookup["team"] = lookup.get("team", pd.Series("", index=lookup.index)).fillna("").astype(str).str.strip().str.upper()
    lookup = lookup.loc[lookup["player_name"].ne("")].drop_duplicates(["batter", "team"])
    return lookup.reset_index(drop=True)


def _prepare_name_lookup(frame: pd.DataFrame, *, id_column: str, team_column: str | None = None, name_column: str = "player_name") -> pd.DataFrame:
    if frame.empty or id_column not in frame.columns or name_column not in frame.columns:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    lookup = frame.copy()
    lookup["batter"] = pd.to_numeric(lookup[id_column], errors="coerce")
    lookup = lookup.loc[lookup["batter"].notna()].copy()
    if lookup.empty:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    lookup["batter"] = lookup["batter"].astype(int)
    lookup["player_name"] = _normalize_name_series(lookup[name_column])
    lookup["team"] = ""
    if team_column and team_column in lookup.columns:
        lookup["team"] = lookup[team_column].fillna("").astype(str).str.strip().str.upper()
    lookup = lookup.loc[lookup["player_name"].ne("")][["batter", "team", "player_name"]]
    return lookup.drop_duplicates(["batter", "team"])


def _apply_name_source(work: pd.DataFrame, lookup: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
    if work.empty or lookup.empty:
        return work
    merged = work.merge(lookup, on=["batter", "team"], how="left", suffixes=("", "_candidate"))
    unresolved = merged["resolved_name"].eq("")
    candidate = _normalize_name_series(merged["player_name_candidate"])
    merged.loc[unresolved & candidate.ne(""), "resolved_name"] = candidate[unresolved & candidate.ne("")]
    merged.loc[unresolved & candidate.ne(""), "name_source"] = source_label
    merged = merged.drop(columns=["player_name_candidate"], errors="ignore")
    unresolved = merged["resolved_name"].eq("")
    if unresolved.any():
        teamless = lookup.loc[_normalize_name_series(lookup["player_name"]).ne("") & lookup["team"].eq(""), ["batter", "player_name"]].drop_duplicates("batter")
        if not teamless.empty:
            merged = merged.merge(teamless, on="batter", how="left", suffixes=("", "_teamless"))
            fallback = _normalize_name_series(merged["player_name_teamless"])
            mask = merged["resolved_name"].eq("") & fallback.ne("")
            merged.loc[mask, "resolved_name"] = fallback[mask]
            merged.loc[mask, "name_source"] = source_label
            merged = merged.drop(columns=["player_name_teamless"], errors="ignore")
    return merged


def _apply_layered_names(
    events: pd.DataFrame,
    rosters: pd.DataFrame,
    hitter_artifact_names: pd.DataFrame,
    recent_live_names: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return events
    work = events.copy()
    work["team"] = work["team"].fillna("").astype(str).str.strip().str.upper()
    work["resolved_name"] = ""
    work["name_source"] = ""

    if "batter_name" in work.columns:
        canonical = _normalize_name_series(work["batter_name"])
        mask = canonical.ne("")
        work.loc[mask, "resolved_name"] = canonical[mask]
        work.loc[mask, "name_source"] = "event"

    roster_lookup = pd.DataFrame(columns=["batter", "team", "player_name"])
    if not rosters.empty and "player_id" in rosters.columns and "player_name" in rosters.columns:
        roster_lookup = _prepare_name_lookup(rosters, id_column="player_id", team_column="team")

    work = _apply_name_source(work, roster_lookup, source_label="roster")
    work = _apply_name_source(work, hitter_artifact_names, source_label="artifact")
    work = _apply_name_source(work, recent_live_names, source_label="live")

    unresolved = work["resolved_name"].eq("")
    work.loc[unresolved, "resolved_name"] = work.loc[unresolved, "batter"].apply(lambda value: f"ID {int(value)}" if pd.notna(value) else "Unknown")
    work.loc[unresolved, "name_source"] = "id"
    work["player_name"] = work["resolved_name"]
    return work.drop(columns=["resolved_name"], errors="ignore")


def _build_sort_columns(frame: pd.DataFrame) -> pd.DataFrame:
    recent = frame.loc[frame["recent_game_rank"] <= 5].copy()
    if recent.empty:
        return pd.DataFrame(columns=["batter"] + [f"sort_ev_{idx}" for idx in range(1, SORT_DEPTH + 1)])
    ordered = recent.sort_values(
        ["batter", "game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    rows: list[dict[str, object]] = []
    for batter, group in ordered.groupby("batter", sort=False):
        values = pd.to_numeric(group["launch_speed"], errors="coerce").dropna().tolist()[:SORT_DEPTH]
        row: dict[str, object] = {"batter": batter}
        for idx in range(SORT_DEPTH):
            row[f"sort_ev_{idx + 1}"] = float(values[idx]) if idx < len(values) else -1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _build_window_strings(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["batter", "window", "window_value"])
    ordered = frame.sort_values(
        ["batter", "game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[True, False, False, False, False],
        na_position="last",
    ).copy()
    outputs: list[pd.DataFrame] = []
    for window in WINDOWS:
        subset = ordered.loc[ordered["recent_game_rank"] <= window, ["batter", "event_pair"]].copy()
        grouped = subset.groupby("batter", sort=False)["event_pair"].agg(lambda values: " | ".join(part for part in values if part)).reset_index()
        grouped["window"] = f"Last {window}"
        grouped["window_value"] = grouped["event_pair"].replace("", "-").fillna("-")
        outputs.append(grouped[["batter", "window", "window_value"]])
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=["batter", "window", "window_value"])


@st.cache_data(show_spinner=False, ttl=300)
def _build_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_board()
    player_base = (
        frame.sort_values(["game_date", "game_pk"], ascending=[False, False], na_position="last")
        .groupby("batter", as_index=False)
        .first()[["batter", "player_name", "team"]]
        .rename(columns={"player_name": "Player", "team": "Team"})
    )
    window_strings = _build_window_strings(frame)
    board = player_base.copy()
    if not window_strings.empty:
        pivoted = window_strings.pivot(index="batter", columns="window", values="window_value").reset_index()
        board = board.merge(pivoted, on="batter", how="left")
    for window in WINDOWS:
        column = f"Last {window}"
        if column not in board.columns:
            board[column] = "-"
        else:
            board[column] = board[column].fillna("-")
    sort_frame = _build_sort_columns(frame)
    if not sort_frame.empty:
        board = board.merge(sort_frame, on="batter", how="left")
        sort_columns = [f"sort_ev_{idx}" for idx in range(1, SORT_DEPTH + 1)]
        board = board.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
        board = board.drop(columns=sort_columns, errors="ignore")
    return board.reset_index(drop=True)


def _result_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace("_", " ").str.title()


def _checkbox_flag(series: pd.Series) -> pd.Series:
    values = pd.Series(series).fillna(False).astype(bool)
    return values.map(lambda flag: "[x]" if flag else "[ ]")


def _build_event_log(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Batter", "Team", "Game", "Date", "PA", "isHH", "isBarrel", "Result", "Exit Velo", "LA", "Pitch Velo"])
    ordered = frame.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[False, False, False, False],
        na_position="last",
    ).head(EVENT_LOG_LIMIT).copy()
    ordered["Batter"] = ordered["player_name"].fillna("")
    ordered["Team"] = ordered["team"].fillna("")
    ordered["Game"] = ordered["game_label"].fillna("")
    ordered["Date"] = ordered["game_date"].dt.date.astype(str)
    ordered["PA"] = pd.to_numeric(ordered["at_bat_number"], errors="coerce").fillna(0).astype(int)
    ordered["isHH"] = _checkbox_flag(ordered.get("is_hard_hit", pd.Series(False, index=ordered.index)))
    ordered["isBarrel"] = _checkbox_flag(ordered.get("is_barrel", pd.Series(False, index=ordered.index)))
    ordered["Result"] = _result_text(ordered["events"])
    ordered["Exit Velo"] = pd.to_numeric(ordered["launch_speed"], errors="coerce").round(1)
    ordered["LA"] = pd.to_numeric(ordered["launch_angle"], errors="coerce").round(1)
    ordered["Pitch Velo"] = pd.to_numeric(ordered["release_speed"], errors="coerce").round(1)
    return ordered[["Batter", "Team", "Game", "Date", "PA", "isHH", "isBarrel", "Result", "Exit Velo", "LA", "Pitch Velo"]].reset_index(drop=True)


def _sort_event_log(frame: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if sort_mode == "Team":
        return frame.sort_values(
            ["Team", "Date", "Game", "PA"],
            ascending=[True, False, True, False],
            na_position="last",
        ).reset_index(drop=True)
    if sort_mode == "Game":
        return frame.sort_values(
            ["Game", "Date", "Team", "PA"],
            ascending=[True, False, True, False],
            na_position="last",
        ).reset_index(drop=True)
    return frame.reset_index(drop=True)


def _apply_summary_event_filters(
    frame: pd.DataFrame,
    team_filters: list[str],
    handedness_filter: str,
    pitch_filters: list[str],
    zone_filters: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.copy()
    if team_filters:
        filtered = filtered.loc[filtered["team"].fillna("").astype(str).isin(team_filters)].copy()
    if handedness_filter in {"vs LHP", "vs RHP"}:
        filtered = filtered.loc[filtered["handedness_filter_label"].fillna("").astype(str).eq(handedness_filter)].copy()
    if pitch_filters:
        filtered = filtered.loc[filtered["pitch_filter_label"].fillna("").astype(str).isin(pitch_filters)].copy()
    if zone_filters:
        filtered = filtered.loc[filtered["zone_filter_label"].fillna("").astype(str).isin(zone_filters)].copy()
    return filtered


def _filter_and_sort_summary(frame: pd.DataFrame, player_search: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.copy()
    needle = player_search.strip().casefold()
    if needle:
        filtered = filtered.loc[
            filtered["Player"].fillna("").astype(str).str.casefold().str.contains(needle)
        ].copy()
    filtered = filtered.sort_values(["Team", "Player"], ascending=[True, True], na_position="last")
    return filtered.reset_index(drop=True)


def _build_player_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_board()
    base = (
        frame.sort_values(["game_date", "game_pk"], ascending=[False, False], na_position="last")
        .groupby("batter", as_index=False)
        .first()[["batter", "player_name", "team"]]
        .rename(columns={"player_name": "Player", "team": "Team"})
    )
    summaries: list[pd.DataFrame] = []
    for window in WINDOWS:
        subset = frame.loc[frame["recent_game_rank"] <= window].copy()
        grouped = (
            subset.groupby("batter", as_index=False)
            .agg(
                avg_ev=("launch_speed", lambda s: pd.to_numeric(s, errors="coerce").mean()),
                max_ev=("launch_speed", lambda s: pd.to_numeric(s, errors="coerce").max()),
                fly_balls=("bb_type", lambda s: pd.Series(s).fillna("").astype(str).str.strip().str.lower().eq("fly_ball").sum()),
                pulled_flyballs=("is_pulled_fly_ball", lambda s: pd.Series(s).fillna(False).astype(bool).sum()),
                tracked_bbe=("launch_speed", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
                hard_hits=("is_hard_hit", lambda s: pd.Series(s).fillna(False).astype(bool).sum()),
                barrel_count=("barrel_count", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
            )
        )
        grouped["fly_ball_pct"] = grouped["fly_balls"] / grouped["tracked_bbe"].replace(0, pd.NA)
        grouped["pulled_flyball_pct"] = grouped["pulled_flyballs"] / grouped["tracked_bbe"].replace(0, pd.NA)
        grouped["hard_hit_pct"] = grouped["hard_hits"] / grouped["tracked_bbe"].replace(0, pd.NA)
        window_prefix = f"L{window}"
        grouped[f"{window_prefix} BBE"] = pd.to_numeric(grouped["tracked_bbe"], errors="coerce").fillna(0).astype(int)
        grouped[f"{window_prefix} Avg EV"] = pd.to_numeric(grouped["avg_ev"], errors="coerce").round(1)
        grouped[f"{window_prefix} Max EV"] = pd.to_numeric(grouped["max_ev"], errors="coerce").round(1)
        grouped[f"{window_prefix} PFB%"] = (pd.to_numeric(grouped["pulled_flyball_pct"], errors="coerce") * 100.0).round(0)
        grouped[f"{window_prefix} FB%"] = (pd.to_numeric(grouped["fly_ball_pct"], errors="coerce") * 100.0).round(0)
        grouped[f"{window_prefix} HH%"] = (pd.to_numeric(grouped["hard_hit_pct"], errors="coerce") * 100.0).round(0)
        grouped[f"{window_prefix} Brl"] = pd.to_numeric(grouped["barrel_count"], errors="coerce").fillna(0).astype(int)
        summaries.append(
            grouped[
                [
                    "batter",
                    f"{window_prefix} BBE",
                    f"{window_prefix} Avg EV",
                    f"{window_prefix} Max EV",
                    f"{window_prefix} PFB%",
                    f"{window_prefix} FB%",
                    f"{window_prefix} HH%",
                    f"{window_prefix} Brl",
                ]
            ]
        )
    board = base.copy()
    for summary in summaries:
        board = board.merge(summary, on="batter", how="left")
    sort_frame = _build_sort_columns(frame)
    if not sort_frame.empty:
        board = board.merge(sort_frame, on="batter", how="left")
        sort_columns = [f"sort_ev_{idx}" for idx in range(1, SORT_DEPTH + 1)]
        board = board.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
        board = board.drop(columns=sort_columns, errors="ignore")
    return board.reset_index(drop=True)


def _select_summary_columns(frame: pd.DataFrame, selected_windows: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected = selected_windows or ["L1", "L5"]
    columns = ["Player", "Team", "batter"]
    metric_suffixes = ["BBE", "Avg EV", "Max EV", "PFB%", "FB%", "HH%", "Brl"]
    for window in selected:
        for suffix in metric_suffixes:
            column_name = f"{window} {suffix}"
            if column_name in frame.columns:
                columns.append(column_name)
    return frame.loc[:, [column for column in columns if column in frame.columns]].copy()


def _format_detail_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Date", "Game", "PA", "isHH", "isBarrel", "Result", "EV", "LA", "Pitch Velo", "Brl"])
    detail = frame.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[False, False, False, False],
        na_position="last",
    ).copy()
    detail["Date"] = pd.to_datetime(detail["game_date"], errors="coerce").dt.date.astype(str)
    detail["Game"] = detail["game_label"].fillna("")
    detail["PA"] = pd.to_numeric(detail["at_bat_number"], errors="coerce").fillna(0).astype(int)
    detail["isHH"] = _checkbox_flag(detail.get("is_hard_hit", pd.Series(False, index=detail.index)))
    detail["isBarrel"] = _checkbox_flag(detail.get("is_barrel", pd.Series(False, index=detail.index)))
    detail["Result"] = _result_text(detail["events"])
    detail["EV"] = pd.to_numeric(detail["launch_speed"], errors="coerce").round(1)
    detail["LA"] = pd.to_numeric(detail["launch_angle"], errors="coerce").round(1)
    detail["Pitch Velo"] = pd.to_numeric(detail["release_speed"], errors="coerce").round(1)
    detail["Brl"] = pd.to_numeric(detail.get("barrel_count"), errors="coerce").fillna(0).astype(int)
    return detail[["Date", "Game", "PA", "isHH", "isBarrel", "Result", "EV", "LA", "Pitch Velo", "Brl"]].reset_index(drop=True)


def _build_detail_rollup(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Date", "Game", "BBE", "Avg EV", "Max EV", "HH%", "Brl"])
    grouped = (
        frame.groupby(["game_date", "game_pk", "game_label"], as_index=False)
        .agg(
            bbe=("launch_speed", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
            avg_ev=("launch_speed", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            max_ev=("launch_speed", lambda s: pd.to_numeric(s, errors="coerce").max()),
            hard_hits=("is_hard_hit", lambda s: pd.Series(s).fillna(False).astype(bool).sum()),
            barrel_count=("barrel_count", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
        )
    )
    grouped = grouped.sort_values(["game_date", "game_pk"], ascending=[False, False], na_position="last").copy()
    grouped["Date"] = pd.to_datetime(grouped["game_date"], errors="coerce").dt.date.astype(str)
    grouped["Game"] = grouped["game_label"].fillna("")
    grouped["BBE"] = pd.to_numeric(grouped["bbe"], errors="coerce").fillna(0).astype(int)
    grouped["Avg EV"] = pd.to_numeric(grouped["avg_ev"], errors="coerce").round(1)
    grouped["Max EV"] = pd.to_numeric(grouped["max_ev"], errors="coerce").round(1)
    grouped["HH%"] = (
        (pd.to_numeric(grouped["hard_hits"], errors="coerce").fillna(0) / grouped["BBE"].replace(0, pd.NA)) * 100.0
    ).round(0)
    grouped["Brl"] = pd.to_numeric(grouped["barrel_count"], errors="coerce").fillna(0).astype(int)
    return grouped[["Date", "Game", "BBE", "Avg EV", "Max EV", "HH%", "Brl"]].reset_index(drop=True)


def _render_side_detail(frame: pd.DataFrame, board: pd.DataFrame) -> None:
    st.subheader("Player Detail")
    if board.empty or frame.empty:
        st.info("No hitter EV results are available for the current filters.")
        return
    options = board[["Player", "Team", "batter"]].copy()
    options["label"] = options["Player"].fillna("").astype(str) + " (" + options["Team"].fillna("").astype(str) + ")"
    labels = options["label"].tolist()
    selected_label = st.selectbox("Select hitter", labels, index=0, key="exit-velo-player")
    selected = options.loc[options["label"] == selected_label].iloc[0]
    player_frame = frame.loc[frame["batter"] == int(selected["batter"])].copy()
    st.caption(f"{selected['Player']} | {selected['Team']}")
    tabs = st.tabs([f"Last {window}" for window in WINDOWS])
    for tab, window in zip(tabs, WINDOWS):
        with tab:
            window_frame = player_frame.loc[player_frame["recent_game_rank"] <= window].copy()
            games = int(window_frame["game_pk"].nunique()) if not window_frame.empty else 0
            st.caption(f"{games} tracked game(s)")
            rollup = _build_detail_rollup(window_frame)
            if not rollup.empty:
                render_custom_metric_table(
                    rollup,
                    key=f"exit-velo-rollup-{int(selected['batter'])}-{window}",
                    height=min(220, 70 + len(rollup) * 35),
                    metric_styles=EXIT_VELO_METRIC_STYLES,
                )
            detail = _format_detail_table(window_frame)
            if detail.empty:
                st.info("No tracked batted-ball events in this window.")
            else:
                render_custom_metric_table(detail, key=f"exit-velo-detail-{int(selected['batter'])}-{window}", height=420, metric_styles=EXIT_VELO_METRIC_STYLES)


def main() -> None:
    st.set_page_config(page_title="Exit Velo Log", page_icon=page_icon_path(), layout="wide")
    apply_branding_head()
    st.title("Exit Velo Log")
    st.caption("Recent hitter exit velocity and launch angle results from tracked batted-ball events.")

    config = AppConfig()
    use_end_date = st.sidebar.checkbox("Use end-date override", value=False)
    end_date = st.sidebar.date_input("End date", value=date.today(), disabled=not use_end_date)
    base_url = _hosted_base_url()
    raw = pd.DataFrame()
    local_raw = pd.DataFrame()
    remote_raw = pd.DataFrame()
    source_used = "local"
    local_error: Exception | None = None
    if base_url:
        try:
            remote_raw = _load_remote_exit_velo_events_cached(
                base_url,
                end_date.isoformat() if use_end_date else None,
                EXIT_VELO_SOURCE_CACHE_VERSION,
            )
            raw = remote_raw
            source_used = "published"
        except Exception as exc:
            st.warning(f"Published Statcast source could not be loaded; trying local files. Detail: {exc}")
    if raw.empty:
        try:
            local_raw = _load_exit_velo_events_cached(end_date.isoformat() if use_end_date else None)
            raw = local_raw
            source_used = "local"
        except Exception as exc:
            local_error = exc
    elif not base_url:
        local_raw = raw
    if base_url and remote_raw.empty and local_raw.empty:
        try:
            local_raw = _load_exit_velo_events_cached(end_date.isoformat() if use_end_date else None)
        except Exception:
            local_raw = pd.DataFrame()
    probe_rows, probe_status = (
        _probe_remote_exit_velo_date_cached(base_url, "2026-04-14", EXIT_VELO_SOURCE_CACHE_VERSION)
        if base_url
        else (0, "base_url_missing")
    )
    st.caption(
        "EV source diagnostic: "
        f"source={source_used}; "
        f"base_url={'set' if base_url else 'missing'}; "
        f"published_max={_max_event_date_label(remote_raw)}; "
        f"local_max={_max_event_date_label(local_raw)}; "
        f"rows={len(raw):,}; "
        f"probe_2026_04_14_rows={probe_rows:,}; "
        f"probe_2026_04_14={probe_status}"
    )
    if raw.empty and local_error is not None:
        st.error(f"Unable to load hitter exit velocity results from local Statcast files: {local_error}")
        return
    if raw.empty:
        st.info("No hitter exit velocity results were found in the local or published Statcast event source.")
        return
    rosters = _load_rosters(config, end_date if use_end_date else None)
    hitter_artifact_names = _load_hitter_artifact_names(config, end_date if use_end_date else None)
    recent_live_names = _load_recent_batter_names_cached(end_date.isoformat() if use_end_date else None)
    raw = _apply_layered_names(raw, rosters, hitter_artifact_names, recent_live_names)

    latest_tracked = pd.to_datetime(raw["game_date"], errors="coerce").max()
    if pd.notna(latest_tracked):
        if use_end_date:
            st.caption(f"Using tracked hitter EV/LA events through {pd.Timestamp(end_date).date().isoformat()}. Latest available event date: {latest_tracked.date().isoformat()}.")
        else:
            st.caption(f"Using latest tracked hitter EV/LA events through {latest_tracked.date().isoformat()}.")

    events = _prepare_events(raw)
    team_options = sorted(value for value in events["team"].dropna().astype(str).unique().tolist())
    selected_teams = st.sidebar.multiselect("Teams", team_options)
    player_search = st.sidebar.text_input("Player search", value="")
    filtered = events.copy()
    if selected_teams:
        filtered = filtered.loc[filtered["team"].isin(selected_teams)].copy()
    if player_search.strip():
        needle = player_search.strip().casefold()
        filtered = filtered.loc[filtered["player_name"].fillna("").astype(str).str.casefold().str.contains(needle)].copy()

    if filtered.empty:
        st.info("No hitters matched the current filters.")
        return

    event_log = _build_event_log(filtered)
    summary_board = _build_player_summary(filtered)
    left, right = st.columns([2.0, 1.1])
    with left:
        st.subheader("Recent Event Log")
        sort_mode = st.segmented_control(
            "Sort by",
            options=["Default", "Team", "Game"],
            default="Default",
            key="exit-velo-event-log-sort",
        )
        event_log = _sort_event_log(event_log, str(sort_mode or "Default"))
        st.caption(f"{len(event_log):,} recent tracked batted-ball events")
        render_custom_metric_table(event_log, key="exit-velo-event-log", height=520, metric_styles=EXIT_VELO_METRIC_STYLES)
    with right:
        _render_side_detail(filtered, summary_board)
    st.subheader("Player Summary")
    summary_controls = st.columns([1.1, 0.8, 0.8, 1.0, 1.0, 0.9])
    with summary_controls[0]:
        summary_player_search = st.text_input("Player search", value="", key="exit-velo-summary-search")
    with summary_controls[1]:
        summary_team_options = sorted(
            value for value in filtered["team"].dropna().astype(str).unique().tolist() if value
        )
        summary_team_filters = st.multiselect(
            "Team",
            options=summary_team_options,
            default=[],
            key="exit-velo-summary-team",
        )
    with summary_controls[2]:
        summary_handedness_filter = st.selectbox(
            "Handedness",
            options=["Both", "vs LHP", "vs RHP"],
            index=0,
            key="exit-velo-summary-handedness",
        )
    with summary_controls[3]:
        summary_pitch_options = sorted(
            value for value in filtered["pitch_filter_label"].dropna().astype(str).unique().tolist() if value
        )
        summary_pitch_filters = st.multiselect(
            "Pitch Type",
            options=summary_pitch_options,
            default=[],
            key="exit-velo-summary-pitch-type",
        )
    with summary_controls[4]:
        zone_values = {
            value for value in filtered["zone_filter_label"].dropna().astype(str).unique().tolist() if value
        }
        summary_zone_options = sorted(zone_values, key=_zone_filter_sort_key)
        summary_zone_filters = st.multiselect(
            "Zone",
            options=summary_zone_options,
            default=[],
            key="exit-velo-summary-zone",
        )
    with summary_controls[5]:
        summary_window_filters = st.multiselect(
            "Rolling",
            options=["L1", "L3", "L5", "L10", "L15", "L25"],
            default=["L1", "L5"],
            key="exit-velo-summary-windows",
        )
    summary_filtered_events = _apply_summary_event_filters(
        filtered,
        summary_team_filters,
        summary_handedness_filter,
        summary_pitch_filters,
        summary_zone_filters,
    )
    summary_board = _build_player_summary(summary_filtered_events)
    display_summary_board = _filter_and_sort_summary(summary_board, summary_player_search)
    display_summary_board = _select_summary_columns(display_summary_board, summary_window_filters)
    st.caption(f"{len(display_summary_board):,} hitters")
    render_exit_velo_summary_grid(
        display_summary_board.drop(columns=["batter"], errors="ignore"),
        key="exit-velo-summary-board",
        height=520,
    )


if __name__ == "__main__":
    main()
