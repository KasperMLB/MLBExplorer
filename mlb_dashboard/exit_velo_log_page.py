from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .cockroach_loader import read_hitter_exit_velo_events
from .config import AppConfig
from .dashboard_views import latest_built_date
from .query_engine import StatcastQueryEngine, load_remote_parquet


WINDOWS = [1, 3, 5, 10, 15, 25]
SORT_DEPTH = 20
EVENT_LOG_LIMIT = 250


def _hosted_base_url() -> str:
    import os

    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


def _empty_board() -> pd.DataFrame:
    return pd.DataFrame(columns=["Player", "Team", "Last 1", "Last 3", "Last 5", "Last 10", "Last 15", "Last 25"])


def _format_event_pair(launch_speed: object, launch_angle: object) -> str:
    ev = pd.to_numeric(pd.Series([launch_speed]), errors="coerce").iloc[0]
    la = pd.to_numeric(pd.Series([launch_angle]), errors="coerce").iloc[0]
    if pd.isna(ev) or pd.isna(la):
        return ""
    return f"{float(ev):.1f}/{float(la):.1f}"


@st.cache_data(show_spinner=False, ttl=300)
def _load_exit_velo_events_cached(end_date_value: str | None) -> pd.DataFrame:
    config = AppConfig()
    parsed_end_date = date.fromisoformat(end_date_value) if end_date_value else None
    return read_hitter_exit_velo_events(config, end_date=parsed_end_date)


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


def _apply_roster_names(events: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    work = events.copy()
    if rosters.empty or "player_id" not in rosters.columns or "player_name" not in rosters.columns:
        work["player_name"] = work["batter"].apply(lambda value: f"ID {int(value)}" if pd.notna(value) else "Unknown")
        return work
    lookup = rosters.loc[:, [column for column in ["team", "player_id", "player_name"] if column in rosters.columns]].copy()
    lookup["player_id"] = pd.to_numeric(lookup["player_id"], errors="coerce")
    lookup = lookup.loc[lookup["player_id"].notna()].copy()
    lookup["player_id"] = lookup["player_id"].astype(int)
    lookup = lookup.drop_duplicates(["team", "player_id"])
    work = work.merge(lookup, left_on=["team", "batter"], right_on=["team", "player_id"], how="left", suffixes=("", "_roster"))
    work["player_name"] = work["player_name_roster"]
    unresolved = work["player_name"].isna() | work["player_name"].astype(str).str.strip().eq("")
    work.loc[unresolved, "player_name"] = work.loc[unresolved, "batter"].apply(lambda value: f"ID {int(value)}" if pd.notna(value) else "Unknown")
    return work.drop(columns=["player_id", "player_name_roster"], errors="ignore")


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


def _build_event_log(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Batter", "Team", "Date", "PA", "In.", "Result", "Exit Velo", "LA", "Pitch Velo"])
    ordered = frame.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[False, False, False, False],
        na_position="last",
    ).head(EVENT_LOG_LIMIT).copy()
    ordered["Batter"] = ordered["player_name"].fillna("")
    ordered["Team"] = ordered["team"].fillna("")
    ordered["Date"] = ordered["game_date"].dt.date.astype(str)
    ordered["PA"] = pd.to_numeric(ordered["at_bat_number"], errors="coerce").fillna(0).astype(int)
    if "inning" in ordered.columns:
        ordered["In."] = pd.to_numeric(ordered["inning"], errors="coerce").fillna(0).astype(int)
    else:
        ordered["In."] = ""
    ordered["Result"] = _result_text(ordered["events"])
    ordered["Exit Velo"] = pd.to_numeric(ordered["launch_speed"], errors="coerce").round(1)
    ordered["LA"] = pd.to_numeric(ordered["launch_angle"], errors="coerce").round(1)
    ordered["Pitch Velo"] = pd.to_numeric(ordered["release_speed"], errors="coerce").round(1)
    return ordered[["Batter", "Team", "Date", "PA", "In.", "Result", "Exit Velo", "LA", "Pitch Velo"]].reset_index(drop=True)


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
                bbe=("launch_speed", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
            )
        )
        grouped[f"Last {window}"] = grouped.apply(
            lambda row: "-"
            if pd.isna(row["avg_ev"])
            else f"{float(row['avg_ev']):.1f} avg | {float(row['max_ev']):.1f} max | {float(row['avg_la']):.1f} LA | {int(row['bbe'])} BBE",
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
        return pd.DataFrame(columns=["Date", "Game", "PA", "In.", "Result", "EV", "LA", "Pitch Velo"])
    detail = frame.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[False, False, False, False],
        na_position="last",
    ).copy()
    detail["Date"] = pd.to_datetime(detail["game_date"], errors="coerce").dt.date.astype(str)
    detail["Game"] = detail["game_label"].fillna("")
    detail["PA"] = pd.to_numeric(detail["at_bat_number"], errors="coerce").fillna(0).astype(int)
    if "inning" in detail.columns:
        detail["In."] = pd.to_numeric(detail["inning"], errors="coerce").fillna(0).astype(int)
    else:
        detail["In."] = ""
    detail["Result"] = _result_text(detail["events"])
    detail["EV"] = pd.to_numeric(detail["launch_speed"], errors="coerce").round(1)
    detail["LA"] = pd.to_numeric(detail["launch_angle"], errors="coerce").round(1)
    detail["Pitch Velo"] = pd.to_numeric(detail["release_speed"], errors="coerce").round(1)
    return detail[["Date", "Game", "PA", "In.", "Result", "EV", "LA", "Pitch Velo"]].reset_index(drop=True)


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
            detail = _format_detail_table(window_frame)
            if detail.empty:
                st.info("No tracked batted-ball events in this window.")
            else:
                st.dataframe(detail, hide_index=True, use_container_width=True, height=420)


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
    raw = _apply_roster_names(raw, rosters)

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
        st.caption(f"{len(event_log):,} recent tracked batted-ball events")
        st.dataframe(event_log, hide_index=True, use_container_width=True, height=520)
    with right:
        _render_side_detail(filtered, summary_board)
    st.subheader("Player Summary")
    st.caption(f"{len(summary_board):,} hitters")
    st.dataframe(summary_board.drop(columns=["batter"], errors="ignore"), hide_index=True, use_container_width=True, height=520)


if __name__ == "__main__":
    main()
