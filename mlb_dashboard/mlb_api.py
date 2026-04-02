from __future__ import annotations

from datetime import date, timedelta

import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"


def _get_json(path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_schedule(target_date: date) -> list[dict]:
    payload = _get_json(
        "/schedule",
        {
            "sportId": 1,
            "date": target_date.isoformat(),
            "hydrate": "probablePitcher,team,linescore",
        },
    )
    games: list[dict] = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            teams = game["teams"]
            home = teams["home"]
            away = teams["away"]
            games.append(
                {
                    "game_pk": game["gamePk"],
                    "game_date": game["gameDate"],
                    "status": game.get("status", {}).get("detailedState"),
                    "home_team": home["team"]["abbreviation"],
                    "away_team": away["team"]["abbreviation"],
                    "home_team_id": home["team"]["id"],
                    "away_team_id": away["team"]["id"],
                    "home_probable_pitcher_id": home.get("probablePitcher", {}).get("id"),
                    "home_probable_pitcher_name": home.get("probablePitcher", {}).get("fullName"),
                    "away_probable_pitcher_id": away.get("probablePitcher", {}).get("id"),
                    "away_probable_pitcher_name": away.get("probablePitcher", {}).get("fullName"),
                }
            )
    return games


def fetch_team_roster(team_id: int, team_abbreviation: str, target_date: date) -> list[dict]:
    payload = _get_json(
        f"/teams/{team_id}/roster",
        {"rosterType": "active", "date": target_date.isoformat(), "hydrate": "person"},
    )
    return [
        {
            "team_id": team_id,
            "team": team_abbreviation,
            "player_id": row["person"]["id"],
            "player_name": row["person"]["fullName"],
            "position_type": row.get("position", {}).get("type"),
            "position_code": row.get("position", {}).get("abbreviation"),
            "status": row.get("status", {}).get("description"),
        }
        for row in payload.get("roster", [])
    ]


def fetch_team_rosters_for_schedule(schedule: list[dict], target_date: date) -> list[dict]:
    rosters: list[dict] = []
    seen: set[int] = set()
    for game in schedule:
        for team_id, team_abbreviation in (
            (game["home_team_id"], game["home_team"]),
            (game["away_team_id"], game["away_team"]),
        ):
            if team_id in seen:
                continue
            seen.add(team_id)
            rosters.extend(fetch_team_roster(team_id, team_abbreviation, target_date))
    return rosters


def fetch_tomorrow_schedule(today: date) -> list[dict]:
    return fetch_schedule(today + timedelta(days=1))


def fetch_game_feed(game_pk: int) -> dict:
    return _get_json(f"/../v1.1/game/{game_pk}/feed/live")
