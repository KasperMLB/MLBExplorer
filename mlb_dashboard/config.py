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
    cockroach_hitter_rolling_table: str = field(default_factory=lambda: os.getenv("COCKROACH_HITTER_ROLLING_TABLE", "public.shared_hitter_rolling_summary"))
    cockroach_pitcher_rolling_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_ROLLING_TABLE", "public.shared_pitcher_rolling_summary"))
    cockroach_batter_zone_table: str = field(default_factory=lambda: os.getenv("COCKROACH_BATTER_ZONE_TABLE", "public.batter_zone_damage_profiles"))
    cockroach_pitcher_zone_table: str = field(default_factory=lambda: os.getenv("COCKROACH_PITCHER_ZONE_TABLE", "public.pitcher_zone_profiles"))
    metrics_version: str = field(default_factory=lambda: os.getenv("MLB_METRICS_VERSION", "v1"))
    year_weights: dict[int, float] = field(default_factory=lambda: DEFAULT_YEAR_WEIGHTS.copy())
    zone_year_weights: dict[int, float] = field(default_factory=lambda: DEFAULT_ZONE_YEAR_WEIGHTS.copy())

    @property
    def reusable_dir(self) -> Path:
        return self.artifacts_dir / "reusable"

    @property
    def daily_dir(self) -> Path:
        return self.artifacts_dir / "daily"


def ensure_directories(config: AppConfig) -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.reusable_dir.mkdir(parents=True, exist_ok=True)
    config.daily_dir.mkdir(parents=True, exist_ok=True)
