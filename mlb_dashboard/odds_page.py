from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .components import render_props_board
from .config import AppConfig
from .dashboard_views import latest_built_date
from .odds_service import PropsBoardPayload, load_live_props_board
from .query_engine import StatcastQueryEngine


def _hosted_base_url() -> str:
    import os

    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


@st.cache_data(show_spinner=False)
def _load_remote_daily(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    slate = pd.read_parquet(f"{base_url}/daily/{day}/slate.parquet")
    rosters = pd.read_parquet(f"{base_url}/daily/{day}/rosters.parquet")
    return slate, rosters


def _load_context(config: AppConfig, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, str, date]:
    local_slate_path = config.daily_dir / target_date.isoformat() / "slate.parquet"
    if local_slate_path.exists():
        engine = StatcastQueryEngine(config)
        slate = pd.DataFrame(engine.load_daily_slate(target_date))
        rosters = engine.load_daily_rosters(target_date)
        return slate, rosters, "local", target_date
    base_url = _hosted_base_url()
    if not base_url:
        return pd.DataFrame(), pd.DataFrame(), "none", target_date
    last_error: Exception | None = None
    for offset in range(8):
        candidate = target_date - timedelta(days=offset)
        try:
            slate, rosters = _load_remote_daily(base_url, candidate)
            return slate, rosters, "hosted", candidate
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.DataFrame(), pd.DataFrame(), "none", target_date


def _default_date(config: AppConfig) -> date:
    latest = latest_built_date(config.daily_dir)
    return latest or date.today()


def _apply_filters(frame: pd.DataFrame, book_details: pd.DataFrame, sportsbooks: tuple[str, ...]) -> tuple[pd.DataFrame, str, bool]:
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
    return work.sort_values(sort_column, ascending=ascending, na_position="last"), sort_column, ascending


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


def _render_instruction_banner() -> None:
    st.markdown(
        """
        <div style="
            margin: 0.35rem 0 0.9rem 0;
            padding: 0.75rem 1rem;
            border: 1px solid #d9e0ea;
            border-radius: 14px;
            background: linear-gradient(90deg, #f4f7fb 0%, #fbf6ea 100%);
            color: #17304f;
            font-size: 0.95rem;
            font-weight: 600;
            box-shadow: 0 6px 20px rgba(16, 37, 66, 0.04);
        ">
            Click any prop row to view all sportsbook odds.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Props Board", layout="wide")
    st.title("Props Board")
    st.caption("Live odds page. This page loads separately so the main matchup workflow stays untouched.")

    config = AppConfig()
    target_date = st.sidebar.date_input("Slate date", value=_default_date(config))
    slate, rosters, source, loaded_date = _load_context(config, target_date)
    if source == "none":
        st.error("No local artifacts or hosted artifact base URL were found for loading slate context.")
        return
    st.caption(f"Context source: {source}")
    if loaded_date != target_date:
        st.caption(f"Using most recent available published slate: {loaded_date.isoformat()}")

    refresh_token = st.session_state.get("props_refresh_token", 0)
    if st.sidebar.button("Refresh Odds"):
        refresh_token += 1
        st.session_state["props_refresh_token"] = refresh_token
    state_key = f"props_board::{loaded_date.isoformat()}::{refresh_token}"
    if state_key not in st.session_state:
        try:
            st.session_state[state_key] = load_live_props_board(config, loaded_date, rosters)
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

    filtered, sort_column, ascending = _apply_filters(board, book_details, payload.sportsbooks)
    display = _display_board(filtered)
    st.caption(f"{len(display):,} prop rows")
    _render_instruction_banner()
    sort_label_map = {
        "best_price": "Best Price",
        "market_width": "Market Width",
        "largest_discrepancy": "Largest Discrepancy",
        "prop_type": "Prop Type",
        "player": "Player",
    }
    render_props_board(
        display,
        book_details,
        height=760,
        initial_sort_column=sort_label_map.get(sort_column, "Best Price"),
        initial_sort_ascending=ascending,
    )
