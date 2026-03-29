from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import AppConfig


@dataclass(frozen=True)
class OddsConfig:
    api_key: str
    base_url: str
    sport: str
    regions: str
    odds_format: str
    date_format: str
    markets: tuple[str, ...]


@dataclass(frozen=True)
class PropsBoardPayload:
    board: pd.DataFrame
    book_details: pd.DataFrame
    sportsbooks: tuple[str, ...]


MARKET_LABELS = {
    "batter_home_runs": "Home Runs",
    "batter_hits": "Hits",
    "batter_total_bases": "Total Bases",
    "batter_rbis": "RBIs",
    "batter_runs_scored": "Runs",
    "batter_stolen_bases": "Stolen Bases",
    "batter_walks": "Walks",
    "batter_hits_runs_rbis": "Hits+Runs+RBIs",
    "pitcher_strikeouts": "Strikeouts",
}


def odds_config_from_app(config: AppConfig) -> OddsConfig:
    markets = tuple(market.strip() for market in config.odds_api_markets.split(",") if market.strip())
    return OddsConfig(
        api_key=config.odds_api_key,
        base_url=config.odds_api_base_url.rstrip("/"),
        sport=config.odds_api_sport,
        regions=config.odds_api_regions,
        odds_format=config.odds_api_odds_format,
        date_format=config.odds_api_date_format,
        markets=markets,
    )


def _american_to_decimal(price: float | int | None) -> float | None:
    if price is None:
        return None
    try:
        price_value = float(price)
    except (TypeError, ValueError):
        return None
    if price_value == 0:
        return None
    if price_value > 0:
        return 1.0 + (price_value / 100.0)
    return 1.0 + (100.0 / abs(price_value))


def _decimal_to_implied_prob(decimal_price: float | None) -> float | None:
    if decimal_price is None or decimal_price <= 0:
        return None
    return 1.0 / decimal_price


def _best_price(prices: Iterable[float | int | None]) -> float | None:
    ranked: list[tuple[float, float]] = []
    for price in prices:
        decimal_price = _american_to_decimal(price)
        if decimal_price is None:
            continue
        ranked.append((decimal_price, float(price)))
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[0])[1]


def _market_width_implied(prices: Iterable[float | int | None]) -> float | None:
    implied_values = []
    for price in prices:
        implied = _decimal_to_implied_prob(_american_to_decimal(price))
        if implied is not None:
            implied_values.append(implied)
    if len(implied_values) < 2:
        return None
    return max(implied_values) - min(implied_values)


def _largest_discrepancy_points(prices: Iterable[float | int | None]) -> float | None:
    numeric = []
    for price in prices:
        try:
            numeric.append(float(price))
        except (TypeError, ValueError):
            continue
    if len(numeric) < 2:
        return None
    return max(numeric) - min(numeric)


def _normalize_side(outcome: dict) -> tuple[str, float | None]:
    name = str(outcome.get("name", "")).strip()
    description = str(outcome.get("description", "")).strip()
    point = outcome.get("point")
    if point is None:
        if name.lower() in {"yes", "no"}:
            return name.title(), None
        if description:
            return description, None
        return name, None
    lowered = name.lower()
    if lowered in {"over", "under", "yes", "no"}:
        return name.title(), float(point)
    return name or description, float(point)


def _infer_team(player_name: str, rosters: pd.DataFrame) -> str | None:
    if rosters.empty or "player_name" not in rosters.columns:
        return None
    exact = rosters.loc[rosters["player_name"].fillna("").astype(str).str.casefold() == player_name.casefold()]
    if exact.empty:
        return None
    teams = exact["team"].dropna().astype(str).unique().tolist()
    if len(teams) == 1:
        return teams[0]
    return teams[0] if teams else None


def fetch_event_list(config: OddsConfig) -> list[dict]:
    if not config.api_key:
        raise RuntimeError("ODDS_API_KEY must be set to load props.")
    response = requests.get(
        f"{config.base_url}/sports/{config.sport}/events",
        params={
            "apiKey": config.api_key,
            "dateFormat": config.date_format,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def fetch_event_odds(config: OddsConfig, event_id: str) -> dict:
    response = requests.get(
        f"{config.base_url}/sports/{config.sport}/events/{event_id}/odds",
        params={
            "apiKey": config.api_key,
            "regions": config.regions,
            "markets": ",".join(config.markets),
            "oddsFormat": config.odds_format,
            "dateFormat": config.date_format,
        },
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def load_live_props_board(config: AppConfig, target_date: date, rosters: pd.DataFrame) -> PropsBoardPayload:
    odds_config = odds_config_from_app(config)
    events = fetch_event_list(odds_config)
    rows: list[dict] = []
    chicago = ZoneInfo("America/Chicago")
    for event in events:
        event_time = pd.to_datetime(event.get("commence_time"), errors="coerce", utc=True)
        if pd.isna(event_time):
            continue
        if event_time.tz_convert(chicago).date() != target_date:
            continue
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue
        details = fetch_event_odds(odds_config, event_id)
        game_label = f"{details.get('away_team', event.get('away_team', ''))} @ {details.get('home_team', event.get('home_team', ''))}".strip()
        bookmakers = details.get("bookmakers", []) or []
        for bookmaker in bookmakers:
            book_key = str(bookmaker.get("key", "")).strip()
            book_title = str(bookmaker.get("title", book_key)).strip()
            for market in bookmaker.get("markets", []) or []:
                market_key = str(market.get("key", "")).strip()
                if not market_key:
                    continue
                prop_type = MARKET_LABELS.get(market_key, market_key.replace("_", " ").title())
                for outcome in market.get("outcomes", []) or []:
                    player_name = str(outcome.get("description") or outcome.get("participant") or outcome.get("player_name") or "").strip()
                    side, line = _normalize_side(outcome)
                    price = outcome.get("price")
                    if not player_name or price is None:
                        continue
                    team = _infer_team(player_name, rosters)
                    rows.append(
                        {
                            "event_id": event_id,
                            "game": game_label,
                            "team": team,
                            "player": player_name,
                            "prop_type": prop_type,
                            "side": side,
                            "line": line,
                            "book_key": book_key,
                            "book_title": book_title,
                            "price": float(price),
                        }
                    )
    if not rows:
        return PropsBoardPayload(
            board=pd.DataFrame(
                columns=[
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
                    "model_odds",
                    "edge_pct",
                    "ev_pct",
                ]
            ),
            book_details=pd.DataFrame(columns=["row_id", "sportsbook", "book_key", "price", "price_display"]),
            sportsbooks=tuple(),
        )
    raw = pd.DataFrame(rows)
    grouped_rows: list[dict] = []
    detail_rows: list[dict] = []
    for idx, (keys, group) in enumerate(raw.groupby(["game", "team", "player", "prop_type", "side", "line"], dropna=False, sort=False), start=1):
        prices = group["price"].tolist()
        best_price = _best_price(prices)
        best_books = group.loc[group["price"] == best_price, "book_title"].astype(str).tolist() if best_price is not None else []
        row_id = f"prop-{idx}"
        for _, row in group.sort_values(["book_title", "price"], ascending=[True, False]).iterrows():
            detail_rows.append(
                {
                    "row_id": row_id,
                    "sportsbook": str(row["book_title"]),
                    "book_key": str(row["book_key"]),
                    "price": float(row["price"]),
                    "price_display": f"{int(row['price']):+d}",
                    "sort_decimal": _american_to_decimal(row["price"]),
                }
            )
        grouped_rows.append(
            {
                "row_id": row_id,
                "game": keys[0],
                "team": keys[1],
                "player": keys[2],
                "prop_type": keys[3],
                "side": keys[4],
                "line": keys[5],
                "best_books": ", ".join(best_books),
                "best_price": best_price,
                "market_width": _market_width_implied(prices),
                "largest_discrepancy": _largest_discrepancy_points(prices),
                "model_odds": None,
                "edge_pct": None,
                "ev_pct": None,
            }
        )
    result = pd.DataFrame(grouped_rows).sort_values(
        ["prop_type", "player", "line", "best_price"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    details = pd.DataFrame(detail_rows).sort_values(["row_id", "sort_decimal", "sportsbook"], ascending=[True, False, True], na_position="last").reset_index(drop=True)
    sportsbooks = tuple(sorted(details["sportsbook"].dropna().astype(str).unique().tolist())) if not details.empty else tuple()
    return PropsBoardPayload(board=result, book_details=details, sportsbooks=sportsbooks)
