from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_YEAR_WEIGHTS = {
    2021: 0.35,
    2022: 0.5,
    2023: 0.7,
    2024: 0.9,
    2025: 1.0,
    2026: 1.35,
}

DEFAULT_ZONE_YEAR_WEIGHTS = {
    2021: 0.20,
    2022: 0.35,
    2023: 0.55,
    2024: 0.85,
    2025: 1.10,
    2026: 1.75,
}

DEFAULT_MOVEMENT_YEAR_WEIGHTS = {
    2021: 0.01,
    2022: 0.03,
    2023: 0.08,
    2024: 0.20,
    2025: 0.52,
    2026: 1.00,
}

DEFAULT_RECENT_WINDOWS = ("season", "last_45_days", "last_14_days")
DEFAULT_SPLITS = ("overall", "vs_rhp", "vs_lhp", "home", "away")
DEFAULT_PITCH_GROUPS = {
    "Fastballs": ("4-Seam Fastball", "Sinker", "Cutter"),
    "Breaking": ("Slider", "Sweeper", "Curveball", "Knuckle Curve", "Slurve"),
    "Offspeed": ("Changeup", "Split-Finger", "Forkball", "Screwball"),
}


@dataclass(frozen=True)
class AppConfig:
    workspace: Path = field(default_factory=lambda: Path(os.getenv("MLB_DASHBOARD_WORKSPACE", Path.cwd())))
    csv_dir: Path = field(default_factory=lambda: Path(os.getenv("MLB_CSV_DIR", Path.cwd())))
    artifacts_dir: Path = field(default_factory=lambda: Path(os.getenv("MLB_ARTIFACTS_DIR", Path.cwd() / "artifacts")))
    db_path: Path = field(default_factory=lambda: Path(os.getenv("MLB_DB_PATH", Path.cwd() / "artifacts" / "statcast.duckdb")))
    hf_repo_id: str = field(default_factory=lambda: os.getenv("HF_REPO_ID", ""))
    hf_repo_type: str = field(default_factory=lambda: os.getenv("HF_REPO_TYPE", "dataset"))
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    cockroach_live_pitch_mix_table: str = field(default_factory=lambda: os.getenv("COCKROACH_LIVE_PITCH_MIX_TABLE", "public.live_pitch_mix_2026"))
    cockroach_pitcher_baseline_event_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_BASELINE_EVENT_TABLE", "public.shared_pitcher_baseline_event_rows"))
    cockroach_hitter_rolling_table: str = field(default_factory=lambda: os.getenv("COCKROACH_HITTER_ROLLING_TABLE", "public.shared_hitter_rolling_summary"))
    cockroach_pitcher_rolling_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_ROLLING_TABLE", "public.shared_pitcher_rolling_summary"))
    cockroach_batter_zone_table: str = field(default_factory=lambda: os.getenv("COCKROACH_BATTER_ZONE_TABLE", "public.batter_zone_damage_profiles"))
    cockroach_pitcher_zone_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_ZONE_TABLE", "public.pitcher_zone_profiles"))
    cockroach_batter_family_zone_table: str = field(default_factory=lambda: os.getenv("COCKROACH_BATTER_FAMILY_ZONE_TABLE", "public.batter_family_zone_profiles"))
    cockroach_hitter_snapshot_table: str = field(default_factory=lambda: os.getenv("COCKROACH_HITTER_SNAPSHOT_TABLE", "public.hitter_model_snapshots"))
    cockroach_hitter_outcome_table: str = field(default_factory=lambda: os.getenv("COCKROACH_HITTER_OUTCOME_TABLE", "public.hitter_game_outcomes"))
    cockroach_hitter_board_table: str = field(default_factory=lambda: os.getenv("COCKROACH_HITTER_BOARD_TABLE", "public.hitter_board_winners"))
    cockroach_pitcher_snapshot_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_SNAPSHOT_TABLE", "public.pitcher_model_snapshots"))
    cockroach_pitcher_outcome_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_OUTCOME_TABLE", "public.pitcher_game_outcomes"))
    cockroach_pitcher_board_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_BOARD_TABLE", "public.pitcher_board_winners"))
    cockroach_pitcher_arsenal_snapshot_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_ARSENAL_SNAPSHOT_TABLE", "public.pitcher_arsenal_snapshots"))
    cockroach_pitcher_count_snapshot_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_COUNT_SNAPSHOT_TABLE", "public.pitcher_count_usage_snapshots"))
    cockroach_props_odds_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PROPS_ODDS_TABLE", "public.cached_upcoming_props_rows"))
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    odds_api_base_url: str = field(default_factory=lambda: os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4"))
    odds_api_sport: str = field(default_factory=lambda: os.getenv("ODDS_API_SPORT", "baseball_mlb"))
    odds_api_regions: str = field(default_factory=lambda: os.getenv("ODDS_API_REGIONS", "us"))
    odds_api_odds_format: str = field(default_factory=lambda: os.getenv("ODDS_API_ODDS_FORMAT", "american"))
    odds_api_date_format: str = field(default_factory=lambda: os.getenv("ODDS_API_DATE_FORMAT", "iso"))
    odds_api_markets: str = field(default_factory=lambda: os.getenv("ODDS_API_MARKETS", "batter_home_runs,batter_hits,batter_total_bases,batter_rbis,batter_runs_scored,batter_stolen_bases,batter_walks,batter_hits_runs_rbis,pitcher_strikeouts"))
    metrics_version: str = field(default_factory=lambda: os.getenv("MLB_METRICS_VERSION", "v1"))
    year_weights: dict[int, float] = field(default_factory=lambda: DEFAULT_YEAR_WEIGHTS.copy())
    zone_year_weights: dict[int, float] = field(default_factory=lambda: DEFAULT_ZONE_YEAR_WEIGHTS.copy())
    movement_year_weights: dict[int, float] = field(default_factory=lambda: DEFAULT_MOVEMENT_YEAR_WEIGHTS.copy())

    @property
    def reusable_dir(self) -> Path:
        return self.artifacts_dir / "reusable"

    @property
    def daily_dir(self) -> Path:
        return self.artifacts_dir / "daily"

    @property
    def historical_cache_dir(self) -> Path:
        return self.artifacts_dir / "historical_cache"

    @property
    def historical_statcast_cache_path(self) -> Path:
        return self.historical_cache_dir / "historical_statcast.parquet"

    @property
    def historical_cache_manifest_path(self) -> Path:
        return self.historical_cache_dir / "manifest.json"


def ensure_directories(config: AppConfig) -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.reusable_dir.mkdir(parents=True, exist_ok=True)
    config.daily_dir.mkdir(parents=True, exist_ok=True)
    config.historical_cache_dir.mkdir(parents=True, exist_ok=True)
