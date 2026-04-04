from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .branding import apply_branding_head, page_icon_path
from .config import AppConfig
from .dashboard_views import latest_built_date
from .query_engine import StatcastQueryEngine
from .ui_components import render_metric_grid, render_weather_field
from .weather_service import build_slate_weather_rows


def _hosted_base_url() -> str:
    import os

    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


@st.cache_data(show_spinner=False)
def _load_remote_daily(base_url: str, target_date: date) -> pd.DataFrame:
    return pd.read_parquet(f"{base_url}/daily/{target_date.isoformat()}/slate.parquet")


def _load_slate(config: AppConfig, target_date: date) -> tuple[pd.DataFrame, str, date]:
    local_slate_path = config.daily_dir / target_date.isoformat() / "slate.parquet"
    if local_slate_path.exists():
        engine = StatcastQueryEngine(config)
        return pd.DataFrame(engine.load_daily_slate(target_date)), "local", target_date
    base_url = _hosted_base_url()
    if not base_url:
        return pd.DataFrame(), "none", target_date
    last_error: Exception | None = None
    for offset in range(8):
        candidate = target_date - timedelta(days=offset)
        try:
            return _load_remote_daily(base_url, candidate), "hosted", candidate
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.DataFrame(), "none", target_date


def _default_date(config: AppConfig) -> date:
    latest = latest_built_date(config.daily_dir)
    return latest or date.today()


def _fmt_temp(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.0f} F"


def _fmt_humidity(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.0f}%"


def _wind_direction_label(wind_direction_deg: object) -> str:
    if wind_direction_deg is None or pd.isna(wind_direction_deg):
        return "DIR --"
    toward_deg = (float(wind_direction_deg) + 180.0) % 360.0
    if 25 <= toward_deg < 65:
        return "OUT TOWARDS RF"
    if 65 <= toward_deg < 115:
        return "OUT"
    if 115 <= toward_deg < 155:
        return "OUT TOWARDS LF"
    if 155 <= toward_deg < 205:
        return "NEUTRAL"
    if 205 <= toward_deg < 245:
        return "IN TOWARDS LF"
    if 245 <= toward_deg < 295:
        return "IN"
    if 295 <= toward_deg < 335:
        return "IN TOWARDS RF"
    return "NEUTRAL"


def _wind_speed_text(wind_speed_mph: object) -> str:
    if wind_speed_mph is None or pd.isna(wind_speed_mph):
        return "MPH --"
    return f"{int(round(float(wind_speed_mph)))} MPH"


def _wind_direction_pill_style(wind_direction_deg: object) -> tuple[str, str]:
    if wind_direction_deg is None or pd.isna(wind_direction_deg):
        return "#eef2f7", "#52606d"
    label = _wind_direction_label(wind_direction_deg)
    if label.startswith("OUT"):
        return "#cfeeda", "#14532d"
    if label.startswith("IN"):
        return "#f8d4d4", "#8b1e1e"
    return "#ffefad", "#715400"


def _wind_speed_pill_style(wind_speed_mph: object) -> tuple[str, str]:
    if wind_speed_mph is None or pd.isna(wind_speed_mph):
        return "#eef2f7", "#52606d"
    speed = float(wind_speed_mph)
    if speed <= 5:
        return "#eef5ff", "#2c5282"
    if speed <= 10:
        return "#d7e8ff", "#1d4f91"
    return "#bdd7ff", "#123b78"


def _render_cards(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    columns = st.columns(2)
    for idx, row in enumerate(frame.to_dict("records")):
        with columns[idx % 2]:
            status = str(row.get("status") or "Unavailable")
            status_fill = "#e6f6ec" if status == "Available" else "#fde8e8"
            status_text = "#166534" if status == "Available" else "#b91c1c"
            wind_direction_text = _wind_direction_label(row.get("wind_direction_deg"))
            wind_direction_fill, wind_direction_text_color = _wind_direction_pill_style(row.get("wind_direction_deg"))
            wind_speed_text = _wind_speed_text(row.get("wind_speed_mph"))
            wind_speed_fill, wind_speed_text_color = _wind_speed_pill_style(row.get("wind_speed_mph"))
            with st.container(border=True):
                st.markdown(
                    f"""
                    <style>
                    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {{
                        min-height: 330px;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                info_col, field_col = st.columns([1.0, 1.05], vertical_alignment="top")
                with info_col:
                    st.markdown(f"<div style='font-size:1.0rem; color:#666; margin-bottom:18px;'>{row.get('game', '')}</div>", unsafe_allow_html=True)
                    venue = str(row.get("venue", "") or "Unknown venue")
                    location = str(row.get("location", "") or "")
                    conditions = str(row.get("conditions", "Unavailable") or "Unavailable")
                    short_conditions = conditions if len(conditions) <= 22 else conditions[:19] + "..."
                    st.markdown(f"<div style='font-size:1.2rem; font-weight:700; line-height:1.25; min-height:3.0rem;'>{venue}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:0.95rem; color:#666; min-height:1.4rem; margin-bottom:8px;'>{location}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:2.25rem; font-weight:800; line-height:1.0; margin:10px 0 16px 0;'>{_fmt_temp(row.get('temperature_f'))}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:1.0rem; margin-bottom:8px;'><strong>Humidity:</strong> {_fmt_humidity(row.get('humidity'))}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:1.0rem; min-height:2.6rem;'><strong>Conditions:</strong> {short_conditions}</div>", unsafe_allow_html=True)
                with field_col:
                    st.markdown(
                        (
                            "<div style='display:flex; justify-content:flex-end; gap:6px; flex-wrap:nowrap; margin-bottom:10px; overflow-x:auto;'>"
                            f"<span style='background:{wind_direction_fill}; color:{wind_direction_text_color}; padding:3px 8px; border-radius:999px; font-size:0.70rem; font-weight:700; letter-spacing:0.01em; white-space:nowrap;'>{wind_direction_text}</span>"
                            f"<span style='background:{wind_speed_fill}; color:{wind_speed_text_color}; padding:3px 8px; border-radius:999px; font-size:0.70rem; font-weight:700; letter-spacing:0.01em; white-space:nowrap;'>{wind_speed_text}</span>"
                            f"<span style='background:{status_fill}; color:{status_text}; padding:3px 8px; border-radius:999px; font-size:0.70rem; font-weight:700; white-space:nowrap;'>{status}</span>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    field_image = render_weather_field(
                        venue_name=str(row.get("venue") or ""),
                        lf_distance_ft=row.get("lf_distance_ft"),
                        cf_distance_ft=row.get("cf_distance_ft"),
                        rf_distance_ft=row.get("rf_distance_ft"),
                        wind_speed_mph=row.get("wind_speed_mph"),
                        wind_direction_deg=row.get("wind_direction_deg"),
                    )
                    if field_image is not None:
                        image_spacer, image_col = st.columns([0.06, 0.94], vertical_alignment="top")
                        with image_col:
                            st.image(field_image, use_container_width=True)
                    else:
                        st.caption("Field graphic unavailable")


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
        ]
    ].rename(
        columns={
            "game": "Game",
            "venue": "Venue",
            "location": "Location",
            "wind_direction": "Wind Direction",
            "conditions": "Conditions",
            "status": "Status",
        }
    )


def main() -> None:
    st.set_page_config(page_title="Weather", page_icon=page_icon_path(), layout="wide")
    apply_branding_head()
    st.title("Weather")
    st.caption("Live weather page for slate parks only. This page stays separate from matchup scoring and the main explorer load path.")

    config = AppConfig()
    target_date = st.sidebar.date_input("Slate date", value=_default_date(config))
    slate, source, loaded_date = _load_slate(config, target_date)
    if source == "none":
        st.error("No local artifacts or hosted artifact base URL were found for loading slate context.")
        return
    st.caption(f"Context source: {source}")
    if loaded_date != target_date:
        st.caption(f"Using most recent available published slate: {loaded_date.isoformat()}")

    refresh_token = st.session_state.get("weather_refresh_token", 0)
    if st.sidebar.button("Refresh Weather"):
        refresh_token += 1
        st.session_state["weather_refresh_token"] = refresh_token

    state_key = f"weather_board::{loaded_date.isoformat()}::{refresh_token}"
    if state_key not in st.session_state:
        with st.spinner("Loading weather..."):
            st.session_state[state_key] = build_slate_weather_rows(slate)
    board = st.session_state[state_key]

    if board.empty:
        st.info("No slate games were found for the selected date.")
        return

    active = board.copy()
    available_count = int((active["status"] == "Available").sum()) if "status" in active else 0
    unavailable_count = int((active["status"] != "Available").sum()) if "status" in active else 0
    if available_count == 0 and unavailable_count > 0:
        st.warning("Live game-time weather is temporarily unavailable.")
    st.caption(f"{len(active):,} slate parks | {available_count} available | {unavailable_count} unavailable")
    _render_cards(active)
    render_metric_grid(
        _display_table(active),
        key=f"weather-board-{loaded_date.isoformat()}",
        height=520,
        use_lightweight=(source == "hosted"),
    )
