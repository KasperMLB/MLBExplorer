from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .cockroach_loader import read_hitter_exit_velo_events
from .config import AppConfig


WINDOWS = [1, 3, 5, 10, 15, 25]
SORT_DEPTH = 20


def _empty_board() -> pd.DataFrame:
    return pd.DataFrame(columns=["Player", "Team", "Last 1", "Last 3", "Last 5", "Last 10", "Last 15", "Last 25"])


def _format_event_pair(launch_speed: object, launch_angle: object) -> str:
    ev = pd.to_numeric(pd.Series([launch_speed]), errors="coerce").iloc[0]
    la = pd.to_numeric(pd.Series([launch_angle]), errors="coerce").iloc[0]
    if pd.isna(ev) or pd.isna(la):
        return ""
    return f"{float(ev):.1f}/{float(la):.1f}"


def _window_cell(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "-"
    ordered = frame.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    parts = [_format_event_pair(row.launch_speed, row.launch_angle) for row in ordered.itertuples(index=False)]
    parts = [part for part in parts if part]
    return " | ".join(parts) if parts else "-"


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    work = events.copy()
    games = (
        work[["batter", "game_pk", "game_date"]]
        .drop_duplicates()
        .sort_values(["batter", "game_date", "game_pk"], ascending=[True, False, False], na_position="last")
    )
    games["recent_game_rank"] = games.groupby("batter").cumcount() + 1
    work = work.merge(games, on=["batter", "game_pk", "game_date"], how="left")
    work["player_label"] = work["player_name"].fillna("").astype(str)
    return work


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


def _build_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_board()
    rows: list[dict[str, object]] = []
    player_base = (
        frame.sort_values(["game_date", "game_pk"], ascending=[False, False], na_position="last")
        .groupby("batter", as_index=False)
        .first()[["batter", "player_name", "team"]]
    )
    for row in player_base.itertuples(index=False):
        player_events = frame.loc[frame["batter"] == row.batter].copy()
        board_row: dict[str, object] = {"batter": row.batter, "Player": row.player_name, "Team": row.team}
        for window in WINDOWS:
            window_frame = player_events.loc[player_events["recent_game_rank"] <= window].copy()
            board_row[f"Last {window}"] = _window_cell(window_frame)
        rows.append(board_row)
    board = pd.DataFrame(rows)
    sort_frame = _build_sort_columns(frame)
    if not sort_frame.empty:
        board = board.merge(sort_frame, on="batter", how="left")
        sort_columns = [f"sort_ev_{idx}" for idx in range(1, SORT_DEPTH + 1)]
        board = board.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
        board = board.drop(columns=sort_columns, errors="ignore")
    return board.reset_index(drop=True)


def _format_detail_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Date", "Game", "Result", "EV", "LA"])
    detail = frame.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[False, False, False, False],
        na_position="last",
    ).copy()
    detail["Date"] = pd.to_datetime(detail["game_date"], errors="coerce").dt.date.astype(str)
    detail["Game"] = detail["game_label"].fillna("")
    detail["Result"] = detail["events"].fillna("").astype(str).str.replace("_", " ").str.title()
    detail["EV"] = pd.to_numeric(detail["launch_speed"], errors="coerce").round(1)
    detail["LA"] = pd.to_numeric(detail["launch_angle"], errors="coerce").round(1)
    return detail[["Date", "Game", "Result", "EV", "LA"]].reset_index(drop=True)


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
    raw = read_hitter_exit_velo_events(config, end_date=end_date if use_end_date else None)
    if raw.empty:
        st.info("No hitter exit velocity results are available from Cockroach.")
        return

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

    board = _build_leaderboard(filtered)
    left, right = st.columns([2.4, 1.2])
    with left:
        st.subheader("Leaderboard")
        st.caption(f"{len(board):,} hitters")
        display = board.drop(columns=["batter"], errors="ignore")
        st.dataframe(display, hide_index=True, use_container_width=True, height=860)
    with right:
        _render_side_detail(filtered, board)


if __name__ == "__main__":
    main()
