from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .cockroach_loader import read_hitter_exit_velo_events, read_recent_batter_name_lookup
from .config import AppConfig
from .dashboard_views import latest_built_date
from .query_engine import StatcastQueryEngine, load_remote_parquet
from .ui_components import render_custom_metric_table


WINDOWS = [1, 3, 5, 10, 15, 25]
SORT_DEPTH = 20
EVENT_LOG_LIMIT = 250
EXIT_VELO_METRIC_STYLES = {
    "Exit Velo": {"mode": "high", "low": 65.0, "high": 112.0},
    "EV": {"mode": "high", "low": 65.0, "high": 112.0},
    "Pitch Velo": {"mode": "high", "low": 80.0, "high": 100.0},
    "LA": {"mode": "target", "low": -30.0, "ideal": 18.0, "high": 50.0},
    "HH%": {"mode": "high", "low": 0.0, "high": 100.0},
    "Avg EV": {"mode": "high", "low": 65.0, "high": 112.0},
    "Max EV": {"mode": "high", "low": 65.0, "high": 112.0},
}


def _hosted_base_url() -> str:
    import os

    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


def _empty_board() -> pd.DataFrame:
    return pd.DataFrame(columns=["Player", "Team", "Last 1", "Last 3", "Last 5", "Last 10", "Last 15", "Last 25"])


def _normalize_name(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().split())


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


@st.cache_data(show_spinner=False, ttl=300)
def _load_exit_velo_events_cached(end_date_value: str | None) -> pd.DataFrame:
    config = AppConfig()
    parsed_end_date = date.fromisoformat(end_date_value) if end_date_value else None
    return read_hitter_exit_velo_events(config, end_date=parsed_end_date)


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
    lookup["player_name"] = lookup["player_name"].map(_normalize_name)
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
    lookup["player_name"] = lookup[name_column].map(_normalize_name)
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
    candidate = merged["player_name_candidate"].fillna("").map(_normalize_name)
    merged.loc[unresolved & candidate.ne(""), "resolved_name"] = candidate[unresolved & candidate.ne("")]
    merged.loc[unresolved & candidate.ne(""), "name_source"] = source_label
    merged = merged.drop(columns=["player_name_candidate"], errors="ignore")
    unresolved = merged["resolved_name"].eq("")
    if unresolved.any():
        teamless = lookup.loc[lookup["player_name"].map(_normalize_name).ne("") & lookup["team"].eq(""), ["batter", "player_name"]].drop_duplicates("batter")
        if not teamless.empty:
            merged = merged.merge(teamless, on="batter", how="left", suffixes=("", "_teamless"))
            fallback = merged["player_name_teamless"].fillna("").map(_normalize_name)
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
        canonical = work["batter_name"].fillna("").map(_normalize_name)
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
                avg_la=("launch_angle", lambda s: pd.to_numeric(s, errors="coerce").mean()),
                tracked_bbe=("launch_speed", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
                hard_hits=("is_hard_hit", lambda s: pd.Series(s).fillna(False).astype(bool).sum()),
                barrel_count=("barrel_count", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
            )
        )
        grouped["hard_hit_pct"] = grouped["hard_hits"] / grouped["tracked_bbe"].replace(0, pd.NA)
        grouped[f"Last {window}"] = grouped.apply(
            lambda row: "-"
            if pd.isna(row["avg_ev"])
            else (
                f"{float(row['avg_ev']):.1f} avg | "
                f"{float(row['max_ev']):.1f} max | "
                f"{float(row['avg_la']):.1f} LA | "
                f"{float(row['hard_hit_pct']) * 100.0:.0f} HH% | "
                f"{int(row['barrel_count'])} Brl"
            ),
            axis=1,
        )
        summaries.append(grouped[["batter", f"Last {window}"]])
    board = base.copy()
    for summary in summaries:
        board = board.merge(summary, on="batter", how="left")
    sort_frame = _build_sort_columns(frame)
    if not sort_frame.empty:
        board = board.merge(sort_frame, on="batter", how="left")
        sort_columns = [f"sort_ev_{idx}" for idx in range(1, SORT_DEPTH + 1)]
        board = board.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
        board = board.drop(columns=sort_columns, errors="ignore")
    for window in WINDOWS:
        column = f"Last {window}"
        board[column] = board[column].fillna("-")
    return board.reset_index(drop=True)


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
    st.set_page_config(page_title="Exit Velo Log", layout="wide")
    st.title("Exit Velo Log")
    st.caption("Recent hitter exit velocity and launch angle results from tracked batted-ball events.")

    config = AppConfig()
    use_end_date = st.sidebar.checkbox("Use end-date override", value=False)
    end_date = st.sidebar.date_input("End date", value=date.today(), disabled=not use_end_date)
    try:
        raw = _load_exit_velo_events_cached(end_date.isoformat() if use_end_date else None)
    except Exception as exc:
        st.error(f"Unable to load hitter exit velocity results from Cockroach: {exc}")
        return
    if raw.empty:
        st.info("No hitter exit velocity results were found in the Cockroach event source.")
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
    st.caption(f"{len(summary_board):,} hitters")
    st.dataframe(summary_board.drop(columns=["batter"], errors="ignore"), hide_index=True, use_container_width=True, height=520)


if __name__ == "__main__":
    main()
