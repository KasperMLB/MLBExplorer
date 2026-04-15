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
    def historical_aggregate_dir(self) -> Path:
        return self.historical_cache_dir / "aggregates"

    @property
    def sources_dir(self) -> Path:
        return self.artifacts_dir / "sources"

    @property
    def statcast_source_dir(self) -> Path:
        return self.sources_dir / "statcast_events"

    @property
    def tracking_dir(self) -> Path:
        return self.artifacts_dir / "tracking"

    @property
    def odds_dir(self) -> Path:
        return self.artifacts_dir / "odds"

    @property
    def odds_history_path(self) -> Path:
        return self.odds_dir / "props_odds_history.parquet"

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
    config.historical_aggregate_dir.mkdir(parents=True, exist_ok=True)
    config.sources_dir.mkdir(parents=True, exist_ok=True)
    config.statcast_source_dir.mkdir(parents=True, exist_ok=True)
    config.tracking_dir.mkdir(parents=True, exist_ok=True)
    config.odds_dir.mkdir(parents=True, exist_ok=True)
