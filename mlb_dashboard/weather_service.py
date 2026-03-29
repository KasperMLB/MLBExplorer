from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .stadium_registry import STADIUMS_BY_TEAM, Stadium


OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_LABELS = {
    0: "Clear",
    1: "Mostly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime Fog",
    51: "Light Drizzle",
    53: "Drizzle",
    55: "Heavy Drizzle",
    56: "Freezing Drizzle",
    57: "Heavy Freezing Drizzle",
    61: "Light Rain",
    63: "Rain",
    65: "Heavy Rain",
    66: "Freezing Rain",
    67: "Heavy Freezing Rain",
    71: "Light Snow",
    73: "Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    80: "Rain Showers",
    81: "Heavy Showers",
    82: "Violent Showers",
    85: "Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorm",
    96: "Thunderstorm + Hail",
    99: "Severe Thunderstorm + Hail",
}


@dataclass(frozen=True)
class WeatherSnapshot:
    weather_time: str
    temperature_f: float | None
    humidity: float | None
    wind_speed_mph: float | None
    wind_direction_deg: float | None
    conditions: str
    status: str


def _weather_label(code: object) -> str:
    if code is None or pd.isna(code):
        return "Unavailable"
    return WEATHER_CODE_LABELS.get(int(code), f"Code {int(code)}")


def _parse_iso_datetime(value: object) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _nearest_index(times: list[str], target: datetime, tz_name: str) -> int | None:
    if not times:
        return None
    zone = ZoneInfo(tz_name)
    target_local = target.astimezone(zone)
    best_idx = None
    best_delta = None
    for idx, value in enumerate(times):
        try:
            current = datetime.fromisoformat(value).replace(tzinfo=zone)
        except ValueError:
            continue
        delta = abs((current - target_local).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx
    return best_idx


def _wind_direction_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    degrees = float(value) % 360.0
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(degrees / 45.0) % len(labels)
    return labels[idx]


def _safe_get(series: list[object], idx: int | None) -> object:
    if idx is None:
        return None
    if idx < 0 or idx >= len(series):
        return None
    return series[idx]


def _fetch_open_meteo_payload(stadium: Stadium) -> dict:
    response = requests.get(
        OPEN_METEO_BASE_URL,
        params={
            "latitude": stadium.latitude,
            "longitude": stadium.longitude,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": stadium.timezone,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m",
            "hourly": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m",
            "forecast_days": 2,
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _current_snapshot(payload: dict) -> WeatherSnapshot:
    current = payload.get("current") or {}
    return WeatherSnapshot(
        weather_time=str(current.get("time") or ""),
        temperature_f=current.get("temperature_2m"),
        humidity=current.get("relative_humidity_2m"),
        wind_speed_mph=current.get("wind_speed_10m"),
        wind_direction_deg=current.get("wind_direction_10m"),
        conditions=_weather_label(current.get("weather_code")),
        status="Available" if current else "Unavailable",
    )


def _game_time_snapshot(payload: dict, first_pitch: datetime | None, stadium: Stadium) -> WeatherSnapshot:
    hourly = payload.get("hourly") or {}
    if first_pitch is None or not hourly:
        return WeatherSnapshot("", None, None, None, None, "Unavailable", "Unavailable")
    times = list(hourly.get("time") or [])
    idx = _nearest_index(times, first_pitch, stadium.timezone)
    return WeatherSnapshot(
        weather_time=str(_safe_get(times, idx) or ""),
        temperature_f=_safe_get(list(hourly.get("temperature_2m") or []), idx),
        humidity=_safe_get(list(hourly.get("relative_humidity_2m") or []), idx),
        wind_speed_mph=_safe_get(list(hourly.get("wind_speed_10m") or []), idx),
        wind_direction_deg=_safe_get(list(hourly.get("wind_direction_10m") or []), idx),
        conditions=_weather_label(_safe_get(list(hourly.get("weather_code") or []), idx)),
        status="Available" if idx is not None else "Unavailable",
    )


def build_slate_weather_rows(slate: pd.DataFrame) -> pd.DataFrame:
    if slate.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for game in slate.to_dict("records"):
        home_team = str(game.get("home_team") or "").strip()
        away_team = str(game.get("away_team") or "").strip()
        game_label = f"{away_team} @ {home_team}" if away_team and home_team else ""
        stadium = STADIUMS_BY_TEAM.get(home_team)
        first_pitch = _parse_iso_datetime(game.get("game_date"))

        base = {
            "game_pk": game.get("game_pk"),
            "game": game_label,
            "home_team": home_team,
            "away_team": away_team,
            "venue": stadium.venue_name if stadium else "",
            "location": stadium.location_name if stadium else "",
            "roof_type": stadium.roof_type if stadium else "",
            "weather_mode": "",
            "weather_time": "",
            "temperature_f": None,
            "wind_speed_mph": None,
            "wind_direction_deg": None,
            "wind_direction": "",
            "humidity": None,
            "conditions": "Unavailable",
            "status": "Unavailable",
        }

        if stadium is None:
            for mode in ("Current", "Game-Time"):
                row = base.copy()
                row["weather_mode"] = mode
                records.append(row)
            continue

        try:
            payload = _fetch_open_meteo_payload(stadium)
            current = _current_snapshot(payload)
            game_time = _game_time_snapshot(payload, first_pitch, stadium)
        except Exception:
            current = WeatherSnapshot("", None, None, None, None, "Unavailable", "Unavailable")
            game_time = WeatherSnapshot("", None, None, None, None, "Unavailable", "Unavailable")

        for mode, snapshot in (("Current", current), ("Game-Time", game_time)):
            row = base.copy()
            row["weather_mode"] = mode
            row["weather_time"] = snapshot.weather_time
            row["temperature_f"] = snapshot.temperature_f
            row["wind_speed_mph"] = snapshot.wind_speed_mph
            row["wind_direction_deg"] = snapshot.wind_direction_deg
            row["wind_direction"] = _wind_direction_label(snapshot.wind_direction_deg)
            row["humidity"] = snapshot.humidity
            row["conditions"] = snapshot.conditions
            row["status"] = snapshot.status
            records.append(row)

    return pd.DataFrame.from_records(records)

