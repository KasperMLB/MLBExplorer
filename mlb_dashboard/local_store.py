from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import AppConfig
from .metrics import add_metric_flags


SOURCE_COLUMNS = [
    "game_date",
    "game_year",
    "game_pk",
    "home_team",
    "away_team",
    "inning_topbot",
    "pitcher",
    "player_name",
    "pitcher_name",
    "batter_name",
    "pitch_name",
    "pitch_type",
    "p_throws",
    "stand",
    "batter",
    "estimated_woba_using_speedangle",
    "bb_type",
    "launch_speed_angle",
    "description",
    "events",
    "launch_angle",
    "launch_speed",
    "release_speed",
    "release_spin_rate",
    "effective_speed",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "plate_x",
    "plate_z",
    "at_bat_number",
    "pitch_number",
    "balls",
    "strikes",
    "outs_when_up",
    "hc_x",
    "hc_y",
    "zone",
    "bat_score",
    "fld_score",
    "post_bat_score",
    "post_fld_score",
    "hit_distance_sc",
    "delta_home_win_exp",
    "delta_run_exp",
    "woba_value",
    "woba_denom",
    "team",
    "fielding_team",
]


ROLLING_HITTER_COLUMNS = [
    "player_name",
    "rolling_window",
    "games_in_window",
    "pulled_barrel_pct",
    "barrel_bip_pct",
    "hard_hit_pct",
    "fb_pct",
    "avg_launch_angle",
    "xwoba",
]

ROLLING_PITCHER_COLUMNS = [
    "player_name",
    "rolling_window",
    "games_in_window",
    "avg_release_speed",
    "barrel_bip_pct",
    "hard_hit_pct",
    "fb_pct",
    "avg_launch_angle",
]

ODDS_COLUMNS = [
    "fetched_at",
    "cache_key",
    "provider",
    "event_id",
    "commence_time",
    "away_team",
    "home_team",
    "sportsbook",
    "sportsbook_key",
    "market_key",
    "market",
    "player_name_raw",
    "player_name",
    "odds_american",
    "line",
    "selection_label",
    "selection_scope",
    "selection_side",
    "market_family",
    "market_variant",
    "threshold",
    "display_label",
    "is_primary_line",
    "is_modeled",
    "player_event_market_key",
    "row_source_type",
    "coverage_completion_status",
    "hr_books_requested",
    "hr_books_present",
    "hr_books_missing",
]


@dataclass
class SourcePayload:
    statcast_events: pd.DataFrame
    hitter_rolling: pd.DataFrame
    pitcher_rolling: pd.DataFrame

    @property
    def live_pitch_mix(self) -> pd.DataFrame:
        return self.statcast_events

    @property
    def pitcher_baseline_event_rows(self) -> pd.DataFrame:
        return self.statcast_events

    @property
    def batter_zone_profiles(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def pitcher_zone_profiles(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def batter_family_zone_profiles(self) -> pd.DataFrame:
        return pd.DataFrame()


@dataclass(frozen=True)
class SourceFreshnessSummary:
    source_name: str
    table_name: str
    target_date: date
    max_event_date: date | None
    lag_days: int | None
    row_count: int
    recent_daily_counts: list[dict[str, object]]
    missing_dates: list[str]

    @property
    def is_fresh(self) -> bool:
        return self.max_event_date is not None and self.lag_days is not None and self.lag_days <= 1


def _date_value(value: object) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index)
    return pd.to_datetime(frame[column], errors="coerce").dt.date


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    frame.to_parquet(temp_path, index=False)
    temp_path.replace(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def normalize_statcast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=SOURCE_COLUMNS)
    work = frame.copy()
    if "game_date" not in work.columns and "game_date_str" in work.columns:
        work["game_date"] = work["game_date_str"]
    work["game_date"] = pd.to_datetime(work.get("game_date"), errors="coerce")
    work = work.loc[work["game_date"].notna()].copy()
    game_year = work["game_year"] if "game_year" in work.columns else pd.Series(pd.NA, index=work.index)
    work["game_year"] = pd.to_numeric(game_year, errors="coerce")
    work.loc[work["game_year"].isna(), "game_year"] = work.loc[work["game_year"].isna(), "game_date"].dt.year
    if "pitcher_name" not in work.columns:
        work["pitcher_name"] = work["player_name"] if "player_name" in work.columns else pd.NA
    if "player_name" not in work.columns:
        work["player_name"] = work["pitcher_name"] if "pitcher_name" in work.columns else pd.NA
    if "batter_name" not in work.columns:
        work["batter_name"] = pd.NA
    away_team = work["away_team"] if "away_team" in work.columns else pd.Series(pd.NA, index=work.index)
    home_team = work["home_team"] if "home_team" in work.columns else pd.Series(pd.NA, index=work.index)
    inning_topbot = work["inning_topbot"] if "inning_topbot" in work.columns else pd.Series(pd.NA, index=work.index)
    if "team" not in work.columns:
        work["team"] = away_team.where(inning_topbot.eq("Top"), home_team)
    if "fielding_team" not in work.columns:
        work["fielding_team"] = home_team.where(inning_topbot.eq("Top"), away_team)
    for column in SOURCE_COLUMNS:
        if column not in work.columns:
            work[column] = pd.NA
    work = add_metric_flags(work)
    return work


def source_partition_path(config: AppConfig, event_date: date) -> Path:
    return config.statcast_source_dir / f"year={event_date.year}" / f"game_date={event_date.isoformat()}.parquet"


def write_statcast_events(config: AppConfig, frame: pd.DataFrame, *, start_date: date | None = None, end_date: date | None = None) -> None:
    work = normalize_statcast_frame(frame)
    if work.empty:
        if start_date is not None and end_date is not None:
            for day in pd.date_range(start=start_date, end=end_date, freq="D"):
                _atomic_write_parquet(pd.DataFrame(columns=SOURCE_COLUMNS), source_partition_path(config, day.date()))
        return
    work["_event_date"] = work["game_date"].dt.date
    for event_date, day_frame in work.groupby("_event_date", sort=True):
        output = day_frame.drop(columns=["_event_date"], errors="ignore").drop_duplicates(
            subset=[column for column in ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"] if column in day_frame.columns],
            keep="last",
        )
        _atomic_write_parquet(output, source_partition_path(config, event_date))


def read_statcast_events(config: AppConfig, *, target_date: date | None = None, start_date: date | None = None, end_date: date | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(config.statcast_source_dir.glob("year=*/game_date=*.parquet")):
        try:
            event_date = date.fromisoformat(path.stem.split("=", 1)[1])
        except Exception:
            continue
        if target_date is not None and event_date > target_date:
            continue
        if start_date is not None and event_date < start_date:
            continue
        if end_date is not None and event_date > end_date:
            continue
        frame = _read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=SOURCE_COLUMNS)
    return normalize_statcast_frame(pd.concat(frames, ignore_index=True, sort=False))


def fetch_pybaseball_statcast(start_date: date, end_date: date) -> pd.DataFrame:
    try:
        from pybaseball import cache, statcast
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pybaseball is required for local Statcast ingest. Install requirements.txt first.") from exc
    try:
        cache.enable()
    except Exception:
        pass
    frame = statcast(start_dt=start_date.isoformat(), end_dt=end_date.isoformat())
    return normalize_statcast_frame(frame)


def ingest_pybaseball_range(
    config: AppConfig,
    *,
    start_date: date,
    end_date: date,
    fallback_csv: Path | None = None,
) -> pd.DataFrame:
    frame = fetch_pybaseball_statcast(start_date, end_date)
    if frame.empty and fallback_csv is not None and fallback_csv.exists():
        frame = normalize_statcast_frame(pd.read_csv(fallback_csv, low_memory=False))
    write_statcast_events(config, frame, start_date=start_date, end_date=end_date)
    return frame


def _empty_hitter_rolling_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ROLLING_HITTER_COLUMNS)


def _empty_pitcher_rolling_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ROLLING_PITCHER_COLUMNS)


def _rate(numerator: pd.Series, denominator: pd.Series) -> float:
    denom = float(denominator.sum())
    if denom <= 0:
        return math.nan
    return float(numerator.sum()) / denom


def compute_hitter_rolling(events: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 15)) -> pd.DataFrame:
    if events.empty:
        return _empty_hitter_rolling_frame()
    work = normalize_statcast_frame(events).sort_values(["batter", "game_date", "game_pk"])
    rows: list[dict[str, object]] = []
    for batter, group in work.dropna(subset=["batter"]).groupby("batter", sort=False):
        game_keys = group[["game_date", "game_pk"]].drop_duplicates().sort_values(["game_date", "game_pk"])
        player_name = group.get("batter_name", pd.Series(index=group.index, dtype="object")).dropna().astype(str)
        if player_name.empty:
            player_name = group.get("player_name", pd.Series(index=group.index, dtype="object")).dropna().astype(str)
        display_name = player_name.value_counts().idxmax() if not player_name.empty else str(int(batter))
        for window in windows:
            recent_games = game_keys.tail(window)
            recent = group.merge(recent_games, on=["game_date", "game_pk"], how="inner")
            if recent.empty:
                continue
            tracked = recent["is_tracked_bbe"].astype(bool)
            bip = recent["is_batted_ball"].astype(bool)
            rows.append(
                {
                    "player_name": display_name,
                    "rolling_window": f"Rolling {window}",
                    "games_in_window": int(len(recent_games)),
                    "pulled_barrel_pct": _rate(recent.loc[tracked, "is_pulled_barrel"].astype(int), pd.Series(1, index=recent.loc[tracked].index)),
                    "barrel_bip_pct": _rate(recent["is_barrel"].astype(int), bip.astype(int)),
                    "hard_hit_pct": _rate(recent["is_hard_hit"].astype(int), bip.astype(int)),
                    "fb_pct": _rate(recent["is_fly_ball"].astype(int), bip.astype(int)),
                    "avg_launch_angle": pd.to_numeric(recent.loc[tracked, "launch_angle_value"], errors="coerce").mean(),
                    "xwoba": pd.to_numeric(recent["xwoba_value"], errors="coerce").mean(),
                }
            )
    return pd.DataFrame(rows, columns=ROLLING_HITTER_COLUMNS)


def compute_pitcher_rolling(events: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 15)) -> pd.DataFrame:
    if events.empty:
        return _empty_pitcher_rolling_frame()
    work = normalize_statcast_frame(events).sort_values(["pitcher", "game_date", "game_pk"])
    rows: list[dict[str, object]] = []
    for pitcher, group in work.dropna(subset=["pitcher"]).groupby("pitcher", sort=False):
        game_keys = group[["game_date", "game_pk"]].drop_duplicates().sort_values(["game_date", "game_pk"])
        names = group.get("pitcher_name", group.get("player_name", pd.Series(index=group.index, dtype="object"))).dropna().astype(str)
        display_name = names.value_counts().idxmax() if not names.empty else str(int(pitcher))
        for window in windows:
            recent_games = game_keys.tail(window)
            recent = group.merge(recent_games, on=["game_date", "game_pk"], how="inner")
            if recent.empty:
                continue
            bip = recent["is_batted_ball"].astype(bool)
            tracked = recent["is_tracked_bbe"].astype(bool)
            rows.append(
                {
                    "player_name": display_name,
                    "rolling_window": f"Rolling {window}",
                    "games_in_window": int(len(recent_games)),
                    "avg_release_speed": pd.to_numeric(recent["release_speed_value"], errors="coerce").mean(),
                    "barrel_bip_pct": _rate(recent["is_barrel"].astype(int), bip.astype(int)),
                    "hard_hit_pct": _rate(recent["is_hard_hit"].astype(int), bip.astype(int)),
                    "fb_pct": _rate(recent["is_fly_ball"].astype(int), bip.astype(int)),
                    "avg_launch_angle": pd.to_numeric(recent.loc[tracked, "launch_angle_value"], errors="coerce").mean(),
                }
            )
    return pd.DataFrame(rows, columns=ROLLING_PITCHER_COLUMNS)


def load_local_source_payload(config: AppConfig, *, target_date: date | None = None) -> SourcePayload:
    events = read_statcast_events(config, target_date=target_date)
    return SourcePayload(
        statcast_events=events,
        hitter_rolling=compute_hitter_rolling(events),
        pitcher_rolling=compute_pitcher_rolling(events),
    )


def _tracking_path(config: AppConfig, name: str) -> Path:
    return config.tracking_dir / f"{name}.parquet"


def _upsert_parquet(config: AppConfig, name: str, frame: pd.DataFrame, keys: list[str]) -> None:
    if frame is None or frame.empty:
        return
    path = _tracking_path(config, name)
    existing = _read_parquet(path)
    combined = pd.concat([existing, frame], ignore_index=True, sort=False) if not existing.empty else frame.copy()
    available_keys = [key for key in keys if key in combined.columns]
    if available_keys:
        combined = combined.drop_duplicates(subset=available_keys, keep="last")
    _atomic_write_parquet(combined, path)


def write_tracking_payload(
    config: AppConfig,
    snapshots: pd.DataFrame,
    outcomes: pd.DataFrame,
    board_winners: pd.DataFrame,
    pitcher_snapshots: pd.DataFrame | None = None,
    pitcher_outcomes: pd.DataFrame | None = None,
    pitcher_board_winners: pd.DataFrame | None = None,
    pitcher_arsenal_snapshots: pd.DataFrame | None = None,
    pitcher_count_snapshots: pd.DataFrame | None = None,
) -> None:
    _upsert_parquet(config, "hitter_model_snapshots", snapshots, ["slate_date", "game_pk", "batter_id", "split_key", "recent_window", "weighted_mode"])
    _upsert_parquet(config, "hitter_game_outcomes", outcomes, ["slate_date", "game_pk", "batter_id"])
    _upsert_parquet(config, "hitter_board_winners", board_winners, ["slate_date", "board_name", "board_rank"])
    _upsert_parquet(config, "pitcher_model_snapshots", pitcher_snapshots if pitcher_snapshots is not None else pd.DataFrame(), ["slate_date", "game_pk", "pitcher_id", "split_key", "recent_window", "weighted_mode"])
    _upsert_parquet(config, "pitcher_game_outcomes", pitcher_outcomes if pitcher_outcomes is not None else pd.DataFrame(), ["slate_date", "game_pk", "pitcher_id"])
    _upsert_parquet(config, "pitcher_board_winners", pitcher_board_winners if pitcher_board_winners is not None else pd.DataFrame(), ["slate_date", "board_name", "board_rank"])
    _upsert_parquet(config, "pitcher_arsenal_snapshots", pitcher_arsenal_snapshots if pitcher_arsenal_snapshots is not None else pd.DataFrame(), ["slate_date", "game_pk", "pitcher_id", "split_key", "recent_window", "weighted_mode", "batter_side_key", "pitch_name"])
    _upsert_parquet(config, "pitcher_count_usage_snapshots", pitcher_count_snapshots if pitcher_count_snapshots is not None else pd.DataFrame(), ["slate_date", "game_pk", "pitcher_id", "split_key", "recent_window", "weighted_mode", "batter_side_key", "pitch_name", "count_bucket"])


def _read_tracking_range(config: AppConfig, name: str, start_date, end_date, *, date_column: str = "slate_date") -> pd.DataFrame:
    frame = _read_parquet(_tracking_path(config, name))
    if frame.empty or date_column not in frame.columns:
        return pd.DataFrame()
    dates = _date_series(frame, date_column)
    return frame.loc[dates.between(start_date, end_date, inclusive="both")].copy()


def _filter_backtest_snapshots(frame: pd.DataFrame, split_key: str | None, recent_window: str | None, weighted_mode: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    if split_key and "split_key" in work.columns:
        work = work.loc[work["split_key"].eq(split_key)]
    if recent_window and "recent_window" in work.columns:
        work = work.loc[work["recent_window"].eq(recent_window)]
    if weighted_mode and "weighted_mode" in work.columns:
        work = work.loc[work["weighted_mode"].eq(weighted_mode)]
    return work


def read_hitter_backtest_data(config: AppConfig, start_date, end_date, split_key: str | None = None, recent_window: str | None = None, weighted_mode: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshots = _filter_backtest_snapshots(_read_tracking_range(config, "hitter_model_snapshots", start_date, end_date), split_key, recent_window, weighted_mode)
    outcomes = _read_tracking_range(config, "hitter_game_outcomes", start_date, end_date)
    boards = _read_tracking_range(config, "hitter_board_winners", start_date, end_date)
    if not snapshots.empty and "matchup_score" in snapshots.columns:
        snapshots = snapshots.sort_values(["slate_date", "game_pk", "matchup_score"], ascending=[False, True, False], na_position="last")
    return snapshots, outcomes, boards


def read_pitcher_backtest_data(config: AppConfig, start_date, end_date, split_key: str | None = None, recent_window: str | None = None, weighted_mode: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshots = _filter_backtest_snapshots(_read_tracking_range(config, "pitcher_model_snapshots", start_date, end_date), split_key, recent_window, weighted_mode)
    outcomes = _read_tracking_range(config, "pitcher_game_outcomes", start_date, end_date)
    boards = _read_tracking_range(config, "pitcher_board_winners", start_date, end_date)
    if not snapshots.empty and "pitcher_score" in snapshots.columns:
        snapshots = snapshots.sort_values(["slate_date", "game_pk", "pitcher_score"], ascending=[False, True, False], na_position="last")
    return snapshots, outcomes, boards


def read_hitter_snapshots_for_date(config: AppConfig, slate_date: date) -> pd.DataFrame:
    return _read_tracking_range(config, "hitter_model_snapshots", slate_date, slate_date)


def read_pitcher_snapshots_for_date(config: AppConfig, slate_date: date) -> pd.DataFrame:
    return _read_tracking_range(config, "pitcher_model_snapshots", slate_date, slate_date)


def purge_backtest_outcomes_before(config: AppConfig, cutoff_date, *, include_pitchers: bool = True) -> tuple[int, int]:
    deleted: list[int] = []
    for name in ["hitter_game_outcomes", "pitcher_game_outcomes"] if include_pitchers else ["hitter_game_outcomes"]:
        path = _tracking_path(config, name)
        frame = _read_parquet(path)
        if frame.empty or "slate_date" not in frame.columns:
            deleted.append(0)
            continue
        dates = _date_series(frame, "slate_date")
        keep = dates.ge(cutoff_date)
        deleted.append(int((~keep).sum()))
        _atomic_write_parquet(frame.loc[keep].copy(), path)
    return deleted[0] if deleted else 0, deleted[1] if len(deleted) > 1 else 0


def write_props_odds_snapshot(config: AppConfig, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    work = frame.copy()
    if "sportsbook" not in work.columns and "book_title" in work.columns:
        work["sportsbook"] = work["book_title"]
    if "sportsbook_key" not in work.columns and "book_key" in work.columns:
        work["sportsbook_key"] = work["book_key"]
    for column in ODDS_COLUMNS:
        if column not in work.columns:
            work[column] = pd.NA
    path = config.odds_history_path
    existing = _read_parquet(path)
    combined = pd.concat([existing, work[ODDS_COLUMNS]], ignore_index=True, sort=False) if not existing.empty else work[ODDS_COLUMNS].copy()
    keys = ["fetched_at", "cache_key", "provider", "event_id", "market_key", "player_event_market_key", "sportsbook_key", "line"]
    combined = combined.drop_duplicates(subset=[key for key in keys if key in combined.columns], keep="last")
    _atomic_write_parquet(combined, path)


def read_prop_odds_history(config: AppConfig, start_date, end_date, markets: tuple[str, ...] = ("batter_home_runs", "pitcher_strikeouts")) -> pd.DataFrame:
    frame = _read_parquet(config.odds_history_path)
    if frame.empty:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    commence = frame["commence_time"] if "commence_time" in frame.columns else pd.Series(pd.NaT, index=frame.index)
    dates = pd.to_datetime(commence, errors="coerce").dt.date
    work = frame.loc[dates.between(start_date, end_date, inclusive="both")].copy()
    if markets and "market_key" in work.columns:
        work = work.loc[work["market_key"].isin(markets)]
    return work.sort_values(["fetched_at", "player_name", "market_key", "sportsbook"], ascending=[False, True, True, True], na_position="last")


def read_latest_prop_odds_snapshot(config: AppConfig, target_date, markets: tuple[str, ...] | None = None) -> pd.DataFrame:
    frame = read_prop_odds_history(config, target_date, target_date, markets or tuple())
    if frame.empty or "fetched_at" not in frame.columns:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    latest = frame.sort_values("fetched_at", ascending=False, na_position="last").iloc[0]
    work = frame.loc[frame["fetched_at"].eq(latest["fetched_at"])]
    if "cache_key" in work.columns:
        work = work.loc[work["cache_key"].eq(latest.get("cache_key"))]
    return work.sort_values(["player_name", "market_key", "sportsbook"], na_position="last")


def read_hitter_exit_velo_events(config: AppConfig, end_date=None) -> pd.DataFrame:
    events = read_statcast_events(config, target_date=end_date)
    columns = [
        "game_date", "game_pk", "away_team", "home_team", "inning_topbot", "batter", "batter_name", "pitcher_name",
        "player_name", "team", "opponent", "game_label", "at_bat_number", "pitch_number", "pitch_type", "pitch_name",
        "p_throws", "zone", "plate_x", "plate_z", "stand", "hc_x", "bb_type", "events", "launch_speed", "launch_angle",
        "release_speed",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    work = events.loc[pd.to_numeric(events.get("launch_speed"), errors="coerce").notna() & pd.to_numeric(events.get("launch_angle"), errors="coerce").notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    work["team"] = work.get("away_team").where(work.get("inning_topbot").eq("Top"), work.get("home_team"))
    work["opponent"] = work.get("home_team").where(work["team"].eq(work.get("away_team")), work.get("away_team"))
    work["game_label"] = work.get("away_team").fillna("").astype(str) + " @ " + work.get("home_team").fillna("").astype(str)
    ranked_games = (
        work[["batter", "game_date", "game_pk"]]
        .drop_duplicates()
        .sort_values(["batter", "game_date", "game_pk"], ascending=[True, False, False])
    )
    ranked_games["recent_game_rank"] = ranked_games.groupby("batter").cumcount() + 1
    work = work.merge(ranked_games.loc[ranked_games["recent_game_rank"].le(25)], on=["batter", "game_date", "game_pk"], how="inner")
    for column in columns:
        if column not in work.columns:
            work[column] = pd.NA
    return work[columns].sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], ascending=[False, False, False, False], na_position="last").reset_index(drop=True)


def read_recent_batter_name_lookup(config: AppConfig, *, end_date: date | None = None, lookback_days: int = 45) -> pd.DataFrame:
    events = read_statcast_events(config, target_date=end_date)
    if events.empty:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    max_date = pd.to_datetime(events["game_date"], errors="coerce").max()
    if pd.isna(max_date):
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    cutoff = max_date.date() - timedelta(days=max(int(lookback_days), 1))
    work = events.loc[_date_series(events, "game_date").ge(cutoff)].copy()
    name_col = work.get("batter_name")
    if name_col is None or name_col.dropna().empty:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    work["player_name"] = name_col.fillna("").astype(str).str.strip()
    work = work.loc[work["player_name"].ne("") & work["batter"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["batter", "team", "player_name"])
    work["team"] = work["team"].fillna("").astype(str).str.upper()
    return work.sort_values("game_date").drop_duplicates(["batter", "team"], keep="last")[["batter", "team", "player_name"]]


def read_source_freshness_report(config: AppConfig, *, target_date: date | None = None, lookback_days: int = 7) -> list[dict[str, object]]:
    check_date = target_date or date.today()
    events = read_statcast_events(config, target_date=check_date, start_date=check_date - timedelta(days=max(int(lookback_days), 1) - 1))
    if events.empty:
        expected = [check_date - timedelta(days=offset) for offset in reversed(range(max(int(lookback_days), 1)))]
        summary = SourceFreshnessSummary("statcast_events", str(config.statcast_source_dir), check_date, None, None, 0, [], [day.isoformat() for day in expected])
        return [asdict(summary) | {"is_fresh": summary.is_fresh}]
    event_dates = _date_series(events, "game_date")
    max_event_date = event_dates.max()
    recent = (
        pd.DataFrame({"event_date": event_dates})
        .dropna()
        .groupby("event_date", as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )
    expected = [check_date - timedelta(days=offset) for offset in reversed(range(max(int(lookback_days), 1)))]
    observed = set(recent["event_date"].tolist())
    summary = SourceFreshnessSummary(
        source_name="statcast_events",
        table_name=str(config.statcast_source_dir),
        target_date=check_date,
        max_event_date=max_event_date,
        lag_days=None if max_event_date is None else (check_date - max_event_date).days,
        row_count=int(len(events)),
        recent_daily_counts=recent.to_dict("records"),
        missing_dates=[day.isoformat() for day in expected if day not in observed],
    )
    return [asdict(summary) | {"is_fresh": summary.is_fresh}]
