from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .config import AppConfig
from .dashboard_views import latest_built_date
from .odds_service import PropsBoardPayload, load_live_props_board
from .query_engine import StatcastQueryEngine
from .ui_components import render_metric_grid


def _hosted_base_url() -> str:
    import os

    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


@st.cache_data(show_spinner=False)
def _load_remote_daily(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    slate = pd.read_parquet(f"{base_url}/daily/{day}/slate.parquet")
    rosters = pd.read_parquet(f"{base_url}/daily/{day}/rosters.parquet")
    return slate, rosters


def _load_context(config: AppConfig, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    local_slate_path = config.daily_dir / target_date.isoformat() / "slate.parquet"
    if local_slate_path.exists():
        engine = StatcastQueryEngine(config)
        slate = pd.DataFrame(engine.load_daily_slate(target_date))
        rosters = engine.load_daily_rosters(target_date)
        return slate, rosters, "local"
    base_url = _hosted_base_url()
    if not base_url:
        return pd.DataFrame(), pd.DataFrame(), "none"
    slate, rosters = _load_remote_daily(base_url, target_date)
    return slate, rosters, "hosted"


def _default_date(config: AppConfig) -> date:
    latest = latest_built_date(config.daily_dir)
    return latest or date.today()


def _apply_filters(frame: pd.DataFrame, book_details: pd.DataFrame, sportsbooks: tuple[str, ...]) -> pd.DataFrame:
    work = frame
    st.sidebar.title("Odds Filters")
    prop_types = sorted(value for value in work["prop_type"].dropna().astype(str).unique().tolist())
    games = sorted(value for value in work["game"].dropna().astype(str).unique().tolist())
    teams = sorted(value for value in work["team"].dropna().astype(str).unique().tolist())
    players = sorted(value for value in work["player"].dropna().astype(str).unique().tolist())

    selected_props = st.sidebar.multiselect("Prop types", prop_types, default=prop_types)
    selected_games = st.sidebar.multiselect("Games", games)
    selected_teams = st.sidebar.multiselect("Teams", teams)
    player_search = st.sidebar.text_input("Player search", value="")
    selected_books = st.sidebar.multiselect("Sportsbooks", list(sportsbooks))

    if selected_props:
        work = work.loc[work["prop_type"].isin(selected_props)]
    if selected_games:
        work = work.loc[work["game"].isin(selected_games)]
    if selected_teams:
        work = work.loc[work["team"].isin(selected_teams)]
    if player_search.strip():
        needle = player_search.strip().casefold()
        work = work.loc[work["player"].fillna("").astype(str).str.casefold().str.contains(needle)]
    if selected_books:
        matching_ids = book_details.loc[book_details["sportsbook"].isin(selected_books), "row_id"].drop_duplicates()
        work = work.loc[work["row_id"].isin(matching_ids)]
    sort_column = st.sidebar.selectbox(
        "Sort by",
        ["best_price", "market_width", "largest_discrepancy", "prop_type", "player"],
        index=0,
    )
    ascending = st.sidebar.checkbox("Ascending sort", value=False)
    return work.sort_values(sort_column, ascending=ascending, na_position="last")


def _format_american(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{int(round(float(value))):+d}"
    except (TypeError, ValueError):
        return ""


def _format_pct(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def _format_discrepancy(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return ""


def _display_board(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.loc[
        :,
        [
            "row_id",
            "game",
            "team",
            "player",
            "prop_type",
            "side",
            "line",
            "best_books",
            "best_price",
            "market_width",
            "largest_discrepancy",
        ],
    ].copy()
    display["best_price"] = display["best_price"].apply(_format_american)
    display["market_width"] = display["market_width"].apply(_format_pct)
    display["largest_discrepancy"] = display["largest_discrepancy"].apply(_format_discrepancy)
    display["model_odds"] = ""
    display["edge_pct"] = ""
    display["ev_pct"] = ""
    return display.rename(
        columns={
            "game": "Game",
            "team": "Team",
            "player": "Player",
            "prop_type": "Prop Type",
            "side": "Side",
            "line": "Line",
            "best_books": "Best Books",
            "best_price": "Best Price",
            "all_books": "All Books",
            "market_width": "Market Width",
            "largest_discrepancy": "Largest Discrepancy",
            "model_odds": "Model Odds",
            "edge_pct": "Edge%",
            "ev_pct": "EV%",
        }
    )


def _build_detail_labels(frame: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in frame.iterrows():
        line_text = "" if pd.isna(row.get("line")) else f" {row.get('line')}"
        labels.append(
            f"{row.get('player', '')} | {row.get('prop_type', '')} | {row.get('side', '')}{line_text} | {row.get('game', '')}"
        )
    return labels


def _render_book_details(filtered_board: pd.DataFrame, book_details: pd.DataFrame, target_date: date, source: str) -> None:
    st.subheader("All Books")
    if filtered_board.empty:
        st.info("No props match the current filters.")
        return
    labels = _build_detail_labels(filtered_board)
    selected_label = st.selectbox("Prop detail", options=labels, key=f"props-detail-{target_date.isoformat()}")
    selected_row = filtered_board.iloc[labels.index(selected_label)]
    st.caption(
        f"{selected_row.get('player', '')} | {selected_row.get('prop_type', '')} | {selected_row.get('side', '')}"
        + ("" if pd.isna(selected_row.get("line")) else f" {selected_row.get('line')}")
    )
    detail_frame = book_details.loc[book_details["row_id"] == selected_row["row_id"], ["sportsbook", "price_display"]].rename(
        columns={"sportsbook": "Sportsbook", "price_display": "Price"}
    )
    render_metric_grid(
        detail_frame,
        key=f"props-book-detail-{selected_row['row_id']}",
        height=min(420, max(140, 42 * (len(detail_frame) + 1))),
        use_lightweight=False,
    )


def main() -> None:
    st.set_page_config(page_title="Props Board", layout="wide")
    st.title("Props Board")
    st.caption("Live odds page. This page loads separately so the main matchup workflow stays untouched.")

    config = AppConfig()
    target_date = st.sidebar.date_input("Slate date", value=_default_date(config))
    slate, rosters, source = _load_context(config, target_date)
    if source == "none":
        st.error("No local artifacts or hosted artifact base URL were found for loading slate context.")
        return
    st.caption(f"Context source: {source}")

    refresh_token = st.session_state.get("props_refresh_token", 0)
    if st.sidebar.button("Refresh Odds"):
        refresh_token += 1
        st.session_state["props_refresh_token"] = refresh_token
    state_key = f"props_board::{target_date.isoformat()}::{refresh_token}"
    if state_key not in st.session_state:
        try:
            st.session_state[state_key] = load_live_props_board(config, target_date, rosters)
        except Exception as exc:
            st.error(f"Unable to load live props: {exc}")
            return
    payload = st.session_state[state_key]
    if not isinstance(payload, PropsBoardPayload):
        st.error("Unexpected props payload shape.")
        return
    board = payload.board
    book_details = payload.book_details

    if board.empty:
        st.info("No props were returned for the selected slate date.")
        return

    filtered = _apply_filters(board, book_details, payload.sportsbooks)
    display = _display_board(filtered)
    st.caption(f"{len(display):,} prop rows")
    render_metric_grid(display.drop(columns=["row_id"], errors="ignore"), key=f"props-board-{target_date.isoformat()}", height=720, use_lightweight=False)
    _render_book_details(filtered, book_details, target_date, source)
