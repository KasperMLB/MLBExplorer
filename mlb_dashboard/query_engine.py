from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import pandas as pd

from .config import AppConfig

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None


@dataclass
class QueryFilters:
    split: str = "overall"
    recent_window: str = "season"
    weighted_mode: str = "weighted"
    min_pitch_count: int = 100
    min_bip: int = 25
    likely_starters_only: bool = False


class StatcastQueryEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self._conn = None

    @property
    def conn(self):
        if duckdb is None:
            raise RuntimeError("duckdb is required for the local explorer. Install requirements.txt first.")
        if self._conn is None:
            self._conn = duckdb.connect(str(self.config.db_path), read_only=True)
        return self._conn

    def load_daily_slate(self, target_date: date) -> list[dict]:
        path = self.config.daily_dir / target_date.isoformat() / "slate.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def load_daily_rosters(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "rosters.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["team", "player_id", "player_name"])
        return pd.read_parquet(path)

    def load_daily_hitter_rolling(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_hitter_rolling.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_pitcher_rolling(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_pitcher_rolling.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_batter_zone_profiles(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_batter_zone_profiles.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_pitcher_zone_profiles(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_pitcher_zone_profiles.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def get_pitcher_cards(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_metrics
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_pitcher_arsenal(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_arsenal
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name, usage_pct DESC
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_pitcher_summary_by_hand(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_summary_by_hand
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name, batter_side_key
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_pitcher_arsenal_by_hand(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_arsenal_by_hand
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name, batter_side_key, usage_pct DESC
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_pitcher_usage_by_count(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_usage_by_count
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name, batter_side_key, count_bucket, usage_pct DESC
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_team_hitter_pool(self, team: str, opposing_pitcher_hand: str | None, filters: QueryFilters) -> pd.DataFrame:
        split_key = filters.split
        if split_key == "overall" and opposing_pitcher_hand == "R":
            split_key = "vs_rhp"
        elif split_key == "overall" and opposing_pitcher_hand == "L":
            split_key = "vs_lhp"
        data = self.conn.execute(
            """
            SELECT *
            FROM hitter_metrics
            WHERE team = $team
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
              AND pitch_count >= $min_pitch_count
              AND bip >= $min_bip
            ORDER BY likely_starter_score DESC NULLS LAST, xwoba DESC NULLS LAST
            """,
            {
                "team": team,
                "split_key": split_key,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
                "min_pitch_count": filters.min_pitch_count,
                "min_bip": filters.min_bip,
            },
        ).df()
        if filters.likely_starters_only and "likely_starter_score" in data:
            data = data.loc[data["likely_starter_score"].fillna(0) > 0]
        return data

    def run_explorer_query(
        self,
        entity_type: str,
        search_text: str = "",
        split_key: str = "overall",
        recent_window: str = "season",
        weighted_mode: str = "weighted",
        limit: int = 100,
    ) -> pd.DataFrame:
        table = "hitter_metrics" if entity_type == "hitters" else "pitcher_metrics"
        name_col = "hitter_name" if entity_type == "hitters" else "pitcher_name"
        search_pattern = f"%{search_text.lower()}%"
        return self.conn.execute(
            f"""
            SELECT *
            FROM {table}
            WHERE split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
              AND LOWER({name_col}) LIKE $search_pattern
            ORDER BY xwoba DESC NULLS LAST
            LIMIT $limit
            """,
            {
                "split_key": split_key,
                "recent_window": recent_window,
                "weighted_mode": weighted_mode,
                "search_pattern": search_pattern,
                "limit": limit,
            },
        ).df()


def load_remote_parquet(base_path: str, filename: str) -> pd.DataFrame:
    return pd.read_parquet(f"{base_path.rstrip('/')}/{filename}")
