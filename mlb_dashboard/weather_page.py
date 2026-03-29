from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .config import AppConfig
from .dashboard_views import latest_built_date
from .query_engine import StatcastQueryEngine
from .ui_components import render_metric_grid
from .weather_service import build_slate_weather_rows


def _hosted_base_url() -> str:
    import os

    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


@st.cache_data(show_spinner=False)
def _load_remote_daily(base_url: str, target_date: date) -> pd.DataFrame:
    return pd.read_parquet(f"{base_url}/daily/{target_date.isoformat()}/slate.parquet")


def _load_slate(config: AppConfig, target_date: date) -> tuple[pd.DataFrame, str]:
    local_slate_path = config.daily_dir / target_date.isoformat() / "slate.parquet"
    if local_slate_path.exists():
        engine = StatcastQueryEngine(config)
        return pd.DataFrame(engine.load_daily_slate(target_date)), "local"
    base_url = _hosted_base_url()
    if not base_url:
        return pd.DataFrame(), "none"
    return _load_remote_daily(base_url, target_date), "hosted"


def _default_date(config: AppConfig) -> date:
    latest = latest_built_date(config.daily_dir)
    return latest or date.today()


def _fmt_temp(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.0f} F"


def _fmt_wind(speed: object, direction: object) -> str:
    if speed is None or pd.isna(speed):
        return "--"
    direction_text = str(direction).strip()
    return f"{float(speed):.0f} mph {direction_text}".strip()


def _fmt_humidity(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.0f}%"


def _render_cards(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    columns = st.columns(3)
    for idx, row in enumerate(frame.to_dict("records")):
        with columns[idx % 3]:
            status = str(row.get("status") or "Unavailable")
            border = "#2f8f4e" if status == "Available" else "#c94b4b"
            st.markdown(
                f"""
                <div style="border:1px solid {border}; border-radius:12px; padding:14px; margin-bottom:12px; background:#ffffff;">
                  <div style="font-size:0.9rem; color:#666;">{row.get("game", "")}</div>
                  <div style="font-size:1.05rem; font-weight:700; margin-top:4px;">{row.get("venue", "") or "Unknown venue"}</div>
                  <div style="font-size:0.85rem; color:#666; margin-bottom:8px;">{row.get("location", "")}</div>
                  <div style="font-size:1.6rem; font-weight:700;">{_fmt_temp(row.get("temperature_f"))}</div>
                  <div style="font-size:0.95rem; margin-top:6px;">Wind: {_fmt_wind(row.get("wind_speed_mph"), row.get("wind_direction"))}</div>
                  <div style="font-size:0.95rem;">Humidity: {_fmt_humidity(row.get("humidity"))}</div>
                  <div style="font-size:0.95rem;">Conditions: {row.get("conditions", "Unavailable")}</div>
                  <div style="font-size:0.85rem; margin-top:8px; color:#666;">Status: {status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _display_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    display["Temperature"] = pd.to_numeric(display["temperature_f"], errors="coerce").round(1)
    display["Wind Speed"] = pd.to_numeric(display["wind_speed_mph"], errors="coerce").round(1)
    display["Humidity"] = pd.to_numeric(display["humidity"], errors="coerce").round(1)
    return display[
        [
            "game",
            "venue",
            "location",
            "Temperature",
            "Wind Speed",
            "wind_direction",
            "Humidity",
            "conditions",
            "status",
            "weather_mode",
        ]
    ].rename(
        columns={
            "game": "Game",
            "venue": "Venue",
            "location": "Location",
            "wind_direction": "Wind Direction",
            "conditions": "Conditions",
            "status": "Status",
            "weather_mode": "Weather Mode",
        }
    )


def main() -> None:
    st.set_page_config(page_title="Weather", layout="wide")
    st.title("Weather")
    st.caption("Live weather page for slate parks only. This page stays separate from matchup scoring and the main explorer load path.")

    config = AppConfig()
    target_date = st.sidebar.date_input("Slate date", value=_default_date(config))
    slate, source = _load_slate(config, target_date)
    if source == "none":
        st.error("No local artifacts or hosted artifact base URL were found for loading slate context.")
        return
    st.caption(f"Context source: {source}")

    weather_mode = st.sidebar.radio("Weather view", ["Current", "Game-Time"], index=0, horizontal=False)
    refresh_token = st.session_state.get("weather_refresh_token", 0)
    if st.sidebar.button("Refresh Weather"):
        refresh_token += 1
        st.session_state["weather_refresh_token"] = refresh_token

    state_key = f"weather_board::{target_date.isoformat()}::{refresh_token}"
    if state_key not in st.session_state:
        with st.spinner("Loading weather..."):
            st.session_state[state_key] = build_slate_weather_rows(slate)
    board = st.session_state[state_key]

    if board.empty:
        st.info("No slate games were found for the selected date.")
        return

    active = board.loc[board["weather_mode"] == weather_mode].copy()
    st.caption(f"{len(active):,} slate parks")
    _render_cards(active)
    render_metric_grid(
        _display_table(active),
        key=f"weather-board-{target_date.isoformat()}-{weather_mode}",
        height=520,
        use_lightweight=(source == "hosted"),
    )

