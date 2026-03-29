from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .config import AppConfig
from .dashboard_views import latest_built_date
from .odds_service import load_live_props_board
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


def _apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    st.sidebar.title("Odds Filters")
    prop_types = sorted(value for value in work["prop_type"].dropna().astype(str).unique().tolist())
    games = sorted(value for value in work["game"].dropna().astype(str).unique().tolist())
    teams = sorted(value for value in work["team"].dropna().astype(str).unique().tolist())
    players = sorted(value for value in work["player"].dropna().astype(str).unique().tolist())
    books = sorted({book for text in work["book_titles"].fillna("").astype(str) for book in text.split("|") if book})

    selected_props = st.sidebar.multiselect("Prop types", prop_types, default=prop_types)
    selected_games = st.sidebar.multiselect("Games", games)
    selected_teams = st.sidebar.multiselect("Teams", teams)
    player_search = st.sidebar.text_input("Player search", value="")
    selected_books = st.sidebar.multiselect("Sportsbooks", books)

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
        work = work.loc[
            work["book_titles"].fillna("").astype(str).apply(
                lambda value: any(book in value.split("|") for book in selected_books)
            )
        ]
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
    display = frame.copy()
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
    board = st.session_state[state_key]

    if board.empty:
        st.info("No props were returned for the selected slate date.")
        return

    filtered = _apply_filters(board)
    display = _display_board(filtered.drop(columns=["book_list", "book_titles"], errors="ignore"))
    st.caption(f"{len(display):,} prop rows")
    render_metric_grid(display, key=f"props-board-{target_date.isoformat()}", height=720, use_lightweight=(source == "hosted"))
