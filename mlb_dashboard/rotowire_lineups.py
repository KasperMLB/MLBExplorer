from __future__ import annotations

import re
import unicodedata
from datetime import date

import pandas as pd
import requests
import streamlit as st
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:  # pragma: no cover
    BeautifulSoup = None
    HAS_BS4 = False

ROTOWIRE_LINEUPS_URL = "https://www.rotowire.com/baseball/daily-lineups.php"
LINEUP_STATUS_MAP = {
    "Confirmed Lineup": "confirmed",
    "Expected Lineup": "expected",
    "Unknown Lineup": "unknown",
}
POSITION_TOKENS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "P"}
STOP_TOKENS = {"Home Run Odds", "Starting Pitcher Intel"}
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s(?:AM|PM)\sET$")
PRICE_RE = re.compile(r"^\$\d[\d,]*$")


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _next_non_price_token(tokens: list[str], start: int) -> tuple[str | None, int]:
    index = start
    while index < len(tokens):
        token = tokens[index]
        if PRICE_RE.match(token):
            index += 1
            continue
        return token, index
    return None, index


def _parse_lineup_side(tokens: list[str], start_index: int) -> tuple[dict[str, object], int]:
    raw_status = tokens[start_index]
    status = LINEUP_STATUS_MAP.get(raw_status, "unknown")
    index = start_index + 1
    players: list[dict[str, object]] = []

    if status == "unknown":
        while index < len(tokens):
            token = tokens[index]
            if token in LINEUP_STATUS_MAP or token in STOP_TOKENS or token.startswith("Umpire:") or TIME_RE.match(token):
                break
            index += 1
        return {"status": status, "players": players}, index

    slot = 1
    while index < len(tokens):
        token = tokens[index]
        if token in LINEUP_STATUS_MAP or token in STOP_TOKENS or token.startswith("Umpire:") or TIME_RE.match(token):
            break
        if token in POSITION_TOKENS:
            name_token, name_index = _next_non_price_token(tokens, index + 1)
            if name_token and name_token not in POSITION_TOKENS and name_token not in STOP_TOKENS and name_token not in LINEUP_STATUS_MAP:
                players.append(
                    {
                        "slot": slot,
                        "player_name": name_token,
                        "position": token,
                    }
                )
                slot += 1
                index = name_index + 1
                continue
        index += 1

    return {"status": status, "players": players}, index


@st.cache_data(show_spinner=False, ttl=300)
def fetch_rotowire_lineups(target_date: date, valid_teams: tuple[str, ...]) -> dict[str, dict[str, object]]:
    if not HAS_BS4:
        return {}

    response = requests.get(
        ROTOWIRE_LINEUPS_URL,
        params={"date": target_date.isoformat()},
        headers={"User-Agent": "MLB Dashboard/1.0"},
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    tokens = [token.strip() for token in soup.stripped_strings if token and token.strip()]
    valid_team_set = {str(team).upper() for team in valid_teams}
    lineups: dict[str, dict[str, object]] = {}
    index = 0

    while index < len(tokens):
        if not TIME_RE.match(tokens[index]):
            index += 1
            continue

        team_tokens: list[str] = []
        scan_index = index + 1
        while scan_index < len(tokens) and scan_index < index + 25 and len(team_tokens) < 2:
            token = tokens[scan_index].upper()
            if token in valid_team_set and token not in team_tokens:
                team_tokens.append(token)
            scan_index += 1

        if len(team_tokens) != 2:
            index += 1
            continue

        away_team, home_team = team_tokens
        first_status_index = next((i for i in range(scan_index, min(len(tokens), scan_index + 80)) if tokens[i] in LINEUP_STATUS_MAP), None)
        if first_status_index is None:
            index += 1
            continue

        away_lineup, next_index = _parse_lineup_side(tokens, first_status_index)
        second_status_index = next((i for i in range(next_index, min(len(tokens), next_index + 80)) if tokens[i] in LINEUP_STATUS_MAP), None)
        if second_status_index is None:
            index = next_index
            continue
        home_lineup, next_index = _parse_lineup_side(tokens, second_status_index)

        lineups[away_team] = away_lineup
        lineups[home_team] = home_lineup
        index = next_index

    return lineups


def resolve_rotowire_lineups(lineups: dict[str, dict[str, object]], rosters: pd.DataFrame) -> dict[str, dict[str, object]]:
    if not lineups:
        return {}

    roster_frame = rosters.copy() if rosters is not None else pd.DataFrame()
    if roster_frame.empty:
        return lineups

    resolved: dict[str, dict[str, object]] = {}
    for team, payload in lineups.items():
        team_roster = roster_frame.loc[roster_frame["team"] == team].copy()
        exact_lookup = {
            str(row["player_name"]).casefold(): row["player_id"]
            for _, row in team_roster.dropna(subset=["player_id", "player_name"]).drop_duplicates("player_id").iterrows()
        }
        normalized_lookup = {
            _normalize_name(row["player_name"]): row["player_id"]
            for _, row in team_roster.dropna(subset=["player_id", "player_name"]).drop_duplicates("player_id").iterrows()
        }

        players: list[dict[str, object]] = []
        for player in payload.get("players", []):
            player_name = str(player.get("player_name") or "")
            player_id = exact_lookup.get(player_name.casefold())
            if player_id is None:
                player_id = normalized_lookup.get(_normalize_name(player_name))
            players.append(
                {
                    **player,
                    "player_id": player_id,
                }
            )
        resolved[team] = {"status": payload.get("status", "unknown"), "players": players}
    return resolved
