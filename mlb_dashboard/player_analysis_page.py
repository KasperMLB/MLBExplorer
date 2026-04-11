from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from .branding import apply_branding_head, page_icon_path
from .config import AppConfig
from .dashboard_views import (
    ARSENAL_COLUMNS,
    BATTER_SIDE_LABELS,
    BATTER_ZONE_COLUMNS,
    FAMILY_FIT_DETAIL_COLUMNS,
    FAMILY_ZONE_CONTEXT_COLUMNS,
    HITTER_ROLLING_COLUMNS,
    MOVEMENT_ARSENAL_COLUMNS,
    PITCHER_HIGHER_IS_BETTER,
    PITCHER_LOWER_IS_BETTER,
    PITCHER_ROLLING_COLUMNS,
    PITCHER_ZONE_COLUMNS,
    add_hitter_matchup_score,
    add_pitcher_rank_score,
    aggregate_batter_zone_map,
    aggregate_pitcher_zone_map,
    apply_roster_names,
    apply_projected_lineup,
    build_pitcher_matchup_key,
    build_zone_overlay_map,
    build_family_zone_fit_detail,
    compute_family_fit_score,
    filter_excluded_pitchers_from_hitter_pool,
    hitter_columns_for_preset,
    latest_built_date,
    pivot_count_usage,
    sort_arsenal_frame,
)
from .query_engine import QueryFilters, StatcastQueryEngine, load_remote_parquet
from .rotowire_lineups import fetch_rotowire_lineups, resolve_rotowire_lineups
from .ui_components import render_metric_grid, render_zone_heatmap

try:
    import altair as alt

    HAS_ALTAIR = True
except ImportError:  # pragma: no cover
    alt = None
    HAS_ALTAIR = False


ROLLING_WINDOW_ORDER = ["Rolling 5", "Rolling 10", "Rolling 15"]
HITTER_OVERVIEW_CARDS = [
    ("Matchup", "matchup_score"),
    ("Ceiling", "ceiling_score"),
    ("Zone Fit", "zone_fit_score"),
    ("Likely", "likely_starter_score"),
    ("xwOBA", "xwoba"),
    ("PulledBrl%", "pulled_barrel_pct"),
    ("SweetSpot%", "sweet_spot_pct"),
    ("Brl/BIP%", "barrel_bip_pct"),
    ("HH%", "hard_hit_pct"),
]
PITCHER_OVERVIEW_CARDS = [
    ("Pitch Score", "pitcher_score"),
    ("Strikeout Score", "strikeout_score"),
    ("xwOBA", "xwoba"),
    ("CSW%", "csw_pct"),
    ("SwStr%", "swstr_pct"),
    ("PutAway%", "putaway_pct"),
    ("Ball%", "ball_pct"),
    ("SIERA", "siera"),
    ("GB%", "gb_pct"),
    ("GB/FB", "gb_fb_ratio"),
    ("Brl/BIP%", "barrel_bip_pct"),
]
HITTER_OVERVIEW_COLUMNS = hitter_columns_for_preset("All stats")
PITCHER_OVERVIEW_COLUMNS = [
    "pitcher_name",
    "p_throws",
    "pitcher_score",
    "strikeout_score",
    "raw_pitcher_score",
    "raw_strikeout_score",
    "pitcher_matchup_adjustment",
    "strikeout_matchup_adjustment",
    "pitch_count",
    "bip",
    "xwoba",
    "called_strike_pct",
    "csw_pct",
    "swstr_pct",
    "putaway_pct",
    "ball_pct",
    "siera",
    "pulled_barrel_pct",
    "barrel_bip_pct",
    "gb_pct",
    "gb_fb_ratio",
    "fb_pct",
    "hard_hit_pct",
    "avg_launch_angle",
]
MATCHUP_HITTER_COLUMNS = [
    "hitter_name",
    "team",
    "matchup_score",
    "test_score",
    "ceiling_score",
    "zone_fit_score",
    "likely_starter_score",
    "xwoba",
    "pulled_barrel_pct",
    "sweet_spot_pct",
    "barrel_bip_pct",
    "hard_hit_pct",
]
MATCHUP_PITCHER_COLUMNS = [
    "pitcher_name",
    "p_throws",
    "pitcher_score",
    "strikeout_score",
    "raw_pitcher_score",
    "raw_strikeout_score",
    "pitcher_matchup_adjustment",
    "strikeout_matchup_adjustment",
    "opponent_lineup_quality",
    "opponent_contact_threat",
    "opponent_whiff_tendency",
    "opponent_family_fit_allowed",
    "lineup_source",
    "lineup_hitter_count",
    "xwoba",
    "csw_pct",
    "swstr_pct",
    "putaway_pct",
    "ball_pct",
    "siera",
    "gb_pct",
    "gb_fb_ratio",
    "barrel_bip_pct",
    "hard_hit_pct",
]
OPPONENT_HITTER_COLUMNS = [
    "hitter_name",
    "team",
    "matchup_score",
    "test_score",
    "ceiling_score",
    "xwoba",
    "pulled_barrel_pct",
    "barrel_bip_pct",
]


def _hitter_table_columns(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], set[str]]:
    present = [column for column in columns if column in frame.columns]
    hidden: set[str] = set()
    if "hitter_name" in present:
        for sample_column in ("pitch_count", "bip"):
            if sample_column in frame.columns and sample_column not in present:
                present.append(sample_column)
                hidden.add(sample_column)
    return present, hidden


def _render_hitter_confidence_legend() -> None:
    chips = [
        ("High", "#166534"),
        ("Medium", "#1f2937"),
        ("Thin", "#b45309"),
        ("Very Thin", "#b91c1c"),
    ]
    chip_html = "".join(
        (
            "<span style='display:inline-flex;align-items:center;gap:6px;"
            "padding:2px 8px;border-radius:12px;border:1px solid rgba(15,23,42,0.15);"
            "background:#f8fafc;margin-right:6px;font-size:12px;'>"
            f"<span style='width:8px;height:8px;border-radius:999px;background:{color};display:inline-block;'></span>"
            f"{label}</span>"
        )
        for label, color in chips
    )
    st.markdown(
        "<div style='font-size:12px;color:#475569;margin-bottom:4px;'>"
        "<strong>Player name sample size</strong> "
        "<span style='color:#64748b'>(legend applies to player name text color)</span>"
        "</div>"
        f"<div>{chip_html}</div>",
        unsafe_allow_html=True,
    )


def _hosted_base_url() -> str:
    import os

    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


def _read_local_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_remote_daily_bundle(base_url: str, target_date: date) -> dict[str, pd.DataFrame]:
    day = target_date.isoformat()
    bundle = {
        "slate": load_remote_parquet(f"{base_url}/daily/{day}", "slate.parquet"),
        "rosters": load_remote_parquet(f"{base_url}/daily/{day}", "rosters.parquet"),
        "daily_hitters": load_remote_parquet(f"{base_url}/daily/{day}", "daily_hitter_metrics.parquet"),
        "daily_pitchers": load_remote_parquet(f"{base_url}/daily/{day}", "daily_pitcher_metrics.parquet"),
        "daily_pitcher_summary_by_hand": load_remote_parquet(f"{base_url}/daily/{day}", "daily_pitcher_summary_by_hand.parquet"),
        "daily_pitcher_arsenal": load_remote_parquet(f"{base_url}/daily/{day}", "daily_pitcher_arsenal.parquet"),
        "daily_pitcher_arsenal_by_hand": load_remote_parquet(f"{base_url}/daily/{day}", "daily_pitcher_arsenal_by_hand.parquet"),
        "daily_pitcher_usage_by_count": load_remote_parquet(f"{base_url}/daily/{day}", "daily_pitcher_usage_by_count.parquet"),
        "daily_hitter_rolling": load_remote_parquet(f"{base_url}/daily/{day}", "daily_hitter_rolling.parquet"),
        "daily_pitcher_rolling": load_remote_parquet(f"{base_url}/daily/{day}", "daily_pitcher_rolling.parquet"),
        "daily_batter_zone_profiles": load_remote_parquet(f"{base_url}/daily/{day}", "daily_batter_zone_profiles.parquet"),
        "daily_pitcher_zone_profiles": load_remote_parquet(f"{base_url}/daily/{day}", "daily_pitcher_zone_profiles.parquet"),
        "daily_batter_family_zone_profiles": load_remote_parquet(f"{base_url}/daily/{day}", "daily_batter_family_zone_profiles.parquet"),
        "daily_pitcher_family_zone_context": load_remote_parquet(f"{base_url}/daily/{day}", "daily_pitcher_family_zone_context.parquet"),
        "daily_pitcher_movement_arsenal": load_remote_parquet(f"{base_url}/daily/{day}", "daily_pitcher_movement_arsenal.parquet"),
    }
    try:
        bundle["hitter_pitcher_exclusions"] = load_remote_parquet(f"{base_url}/daily/{day}", "hitter_pitcher_exclusions.parquet")
    except Exception:
        bundle["hitter_pitcher_exclusions"] = pd.DataFrame(columns=["player_id", "exclude_from_hitter_tables"])
    return bundle


@st.cache_data(show_spinner=False)
def _load_remote_reusable_bundle(base_url: str) -> dict[str, pd.DataFrame]:
    reusable_base = f"{base_url}/reusable"
    return {
        "hitters": load_remote_parquet(reusable_base, "hitter_metrics.parquet"),
        "pitchers": load_remote_parquet(reusable_base, "pitcher_metrics.parquet"),
        "pitcher_summary_by_hand": load_remote_parquet(reusable_base, "pitcher_summary_by_hand.parquet"),
        "pitcher_arsenal": load_remote_parquet(reusable_base, "pitcher_arsenal.parquet"),
        "pitcher_arsenal_by_hand": load_remote_parquet(reusable_base, "pitcher_arsenal_by_hand.parquet"),
        "pitcher_usage_by_count": load_remote_parquet(reusable_base, "pitcher_usage_by_count.parquet"),
        "hitter_rolling": load_remote_parquet(reusable_base, "hitter_rolling.parquet"),
        "pitcher_rolling": load_remote_parquet(reusable_base, "pitcher_rolling.parquet"),
        "batter_zone_profiles": load_remote_parquet(reusable_base, "batter_zone_profiles.parquet"),
        "pitcher_zone_profiles": load_remote_parquet(reusable_base, "pitcher_zone_profiles.parquet"),
        "batter_family_zone_profiles": load_remote_parquet(reusable_base, "batter_family_zone_profiles.parquet"),
        "pitcher_family_zone_context": load_remote_parquet(reusable_base, "pitcher_family_zone_context.parquet"),
        "pitcher_movement_arsenal": load_remote_parquet(reusable_base, "pitcher_movement_arsenal.parquet"),
    }


def _load_local_daily_bundle(config: AppConfig, target_date: date) -> dict[str, pd.DataFrame]:
    engine = StatcastQueryEngine(config)
    daily_dir = config.daily_dir / target_date.isoformat()
    return {
        "slate": pd.DataFrame(engine.load_daily_slate(target_date)),
        "rosters": engine.load_daily_rosters(target_date),
        "daily_hitters": _read_local_parquet(daily_dir / "daily_hitter_metrics.parquet"),
        "daily_pitchers": _read_local_parquet(daily_dir / "daily_pitcher_metrics.parquet"),
        "daily_pitcher_summary_by_hand": _read_local_parquet(daily_dir / "daily_pitcher_summary_by_hand.parquet"),
        "daily_pitcher_arsenal": _read_local_parquet(daily_dir / "daily_pitcher_arsenal.parquet"),
        "daily_pitcher_arsenal_by_hand": _read_local_parquet(daily_dir / "daily_pitcher_arsenal_by_hand.parquet"),
        "daily_pitcher_usage_by_count": _read_local_parquet(daily_dir / "daily_pitcher_usage_by_count.parquet"),
        "daily_hitter_rolling": _read_local_parquet(daily_dir / "daily_hitter_rolling.parquet"),
        "daily_pitcher_rolling": _read_local_parquet(daily_dir / "daily_pitcher_rolling.parquet"),
        "daily_batter_zone_profiles": _read_local_parquet(daily_dir / "daily_batter_zone_profiles.parquet"),
        "daily_pitcher_zone_profiles": _read_local_parquet(daily_dir / "daily_pitcher_zone_profiles.parquet"),
        "daily_batter_family_zone_profiles": _read_local_parquet(daily_dir / "daily_batter_family_zone_profiles.parquet"),
        "daily_pitcher_family_zone_context": _read_local_parquet(daily_dir / "daily_pitcher_family_zone_context.parquet"),
        "daily_pitcher_movement_arsenal": _read_local_parquet(daily_dir / "daily_pitcher_movement_arsenal.parquet"),
        "hitter_pitcher_exclusions": _read_local_parquet(daily_dir / "hitter_pitcher_exclusions.parquet"),
    }


def _load_local_reusable_bundle(config: AppConfig) -> dict[str, pd.DataFrame]:
    reusable_dir = config.reusable_dir
    return {
        "hitters": _read_local_parquet(reusable_dir / "hitter_metrics.parquet"),
        "pitchers": _read_local_parquet(reusable_dir / "pitcher_metrics.parquet"),
        "pitcher_summary_by_hand": _read_local_parquet(reusable_dir / "pitcher_summary_by_hand.parquet"),
        "pitcher_arsenal": _read_local_parquet(reusable_dir / "pitcher_arsenal.parquet"),
        "pitcher_arsenal_by_hand": _read_local_parquet(reusable_dir / "pitcher_arsenal_by_hand.parquet"),
        "pitcher_usage_by_count": _read_local_parquet(reusable_dir / "pitcher_usage_by_count.parquet"),
        "hitter_rolling": _read_local_parquet(reusable_dir / "hitter_rolling.parquet"),
        "pitcher_rolling": _read_local_parquet(reusable_dir / "pitcher_rolling.parquet"),
        "batter_zone_profiles": _read_local_parquet(reusable_dir / "batter_zone_profiles.parquet"),
        "pitcher_zone_profiles": _read_local_parquet(reusable_dir / "pitcher_zone_profiles.parquet"),
        "batter_family_zone_profiles": _read_local_parquet(reusable_dir / "batter_family_zone_profiles.parquet"),
        "pitcher_family_zone_context": _read_local_parquet(reusable_dir / "pitcher_family_zone_context.parquet"),
        "pitcher_movement_arsenal": _read_local_parquet(reusable_dir / "pitcher_movement_arsenal.parquet"),
    }


def _load_context(config: AppConfig, target_date: date) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], str, date]:
    local_slate_json = config.daily_dir / target_date.isoformat() / "slate.json"
    if local_slate_json.exists():
        return _load_local_daily_bundle(config, target_date), _load_local_reusable_bundle(config), "local", target_date
    base_url = _hosted_base_url()
    if not base_url:
        return {}, {}, "none", target_date
    reusable = _load_remote_reusable_bundle(base_url)
    last_error: Exception | None = None
    for offset in range(8):
        candidate = target_date - timedelta(days=offset)
        try:
            daily = _load_remote_daily_bundle(base_url, candidate)
            return daily, reusable, "hosted", candidate
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return {}, {}, "none", target_date


def _default_date(config: AppConfig) -> date:
    latest = latest_built_date(config.daily_dir)
    return latest or date.today()


def _sidebar(config: AppConfig) -> tuple[date, QueryFilters]:
    st.sidebar.title("Player Analysis Filters")
    target_date = st.sidebar.date_input("Slate date", value=_default_date(config))
    split = st.sidebar.selectbox("Split", ["overall", "vs_rhp", "vs_lhp", "home", "away"])
    recent_window = st.sidebar.selectbox("Recent window", ["season", "last_45_days", "last_14_days"])
    weighted_mode = st.sidebar.radio("Weighting", ["weighted", "unweighted"], horizontal=True)
    return target_date, QueryFilters(split=split, recent_window=recent_window, weighted_mode=weighted_mode)


def _filter_hitter_metrics(frame: pd.DataFrame, filters: QueryFilters) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.loc[
        (frame["split_key"] == filters.split)
        & (frame["recent_window"] == filters.recent_window)
        & (frame["weighted_mode"] == filters.weighted_mode)
    ].copy()


def _filter_pitcher_metrics(frame: pd.DataFrame, filters: QueryFilters) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.loc[
        (frame["split_key"] == filters.split)
        & (frame["recent_window"] == filters.recent_window)
        & (frame["weighted_mode"] == filters.weighted_mode)
    ].copy()


def _filter_team_hitters(
    frame: pd.DataFrame,
    rosters: pd.DataFrame,
    hitter_pitcher_exclusions: pd.DataFrame,
    rotowire_lineups: dict[str, dict[str, object]],
    team: str,
    opposing_hand: str | None,
    filters: QueryFilters,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    split_key = filters.split
    if split_key == "overall" and opposing_hand == "R":
        split_key = "vs_rhp"
    elif split_key == "overall" and opposing_hand == "L":
        split_key = "vs_lhp"
    roster_lookup = rosters.loc[rosters["team"] == team, ["player_id"]].dropna().drop_duplicates()
    roster_player_ids = pd.to_numeric(roster_lookup["player_id"], errors="coerce").dropna().astype(int)
    filtered = frame.loc[
        (frame["split_key"] == split_key)
        & (frame["recent_window"] == filters.recent_window)
        & (frame["weighted_mode"] == filters.weighted_mode)
    ].copy()
    if not roster_player_ids.empty and "batter" in filtered.columns:
        filtered = filtered.loc[filtered["batter"].isin(roster_player_ids)]
    filtered["team"] = team
    filtered = apply_roster_names(filtered, rosters, team)
    filtered = filter_excluded_pitchers_from_hitter_pool(filtered, hitter_pitcher_exclusions)
    return apply_projected_lineup(filtered, team, rotowire_lineups)


def _filter_pitcher_detail(frame: pd.DataFrame, pitcher_id: int | None, filters: QueryFilters) -> pd.DataFrame:
    if frame.empty or pitcher_id is None:
        return pd.DataFrame()
    work = frame.loc[frame["pitcher_id"] == pitcher_id].copy()
    if "split_key" in work.columns:
        work = work.loc[work["split_key"] == filters.split]
    if "recent_window" in work.columns:
        work = work.loc[work["recent_window"] == filters.recent_window]
    if "weighted_mode" in work.columns:
        work = work.loc[work["weighted_mode"] == filters.weighted_mode]
    return work


def _player_key(entity_type: str, player_id: object) -> str:
    return f"{entity_type}:{int(player_id)}"


def _build_slate_lookup(
    daily: dict[str, pd.DataFrame],
    filters: QueryFilters,
    resolved_date: date,
) -> tuple[dict[str, dict[str, object]], str | None]:
    slate_lookup: dict[str, dict[str, object]] = {}
    slate = daily.get("slate", pd.DataFrame())
    rosters = daily.get("rosters", pd.DataFrame())
    daily_hitters = daily.get("daily_hitters", pd.DataFrame())
    daily_pitchers = daily.get("daily_pitchers", pd.DataFrame())
    hitter_pitcher_exclusions = daily.get("hitter_pitcher_exclusions", pd.DataFrame())
    batter_zone_profiles = daily.get("daily_batter_zone_profiles", pd.DataFrame())
    pitcher_zone_profiles = daily.get("daily_pitcher_zone_profiles", pd.DataFrame())
    batter_family_zone_profiles = daily.get("daily_batter_family_zone_profiles", pd.DataFrame())
    pitcher_family_zone_context = daily.get("daily_pitcher_family_zone_context", pd.DataFrame())
    if slate.empty:
        return slate_lookup, None
    valid_teams = tuple(sorted(set(slate["away_team"].dropna().astype(str)) | set(slate["home_team"].dropna().astype(str))))
    try:
        rotowire_lineups = resolve_rotowire_lineups(fetch_rotowire_lineups(resolved_date, valid_teams), rosters)
    except Exception:
        rotowire_lineups = {}

    top_key: str | None = None
    top_score: float | None = None
    pitcher_frame = _filter_pitcher_metrics(daily_pitchers, filters)
    slate_pitcher_frames: list[pd.DataFrame] = []
    opponent_hitters_by_key: dict[tuple[object, object, object, object, object], pd.DataFrame] = {}
    raw_pitchers_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for game in slate.to_dict("records"):
        away_pitcher_id = game.get("away_probable_pitcher_id")
        home_pitcher_id = game.get("home_probable_pitcher_id")
        away_pitcher = pitcher_frame.loc[pitcher_frame["pitcher_id"] == away_pitcher_id].copy()
        home_pitcher = pitcher_frame.loc[pitcher_frame["pitcher_id"] == home_pitcher_id].copy()
        away_hand = home_pitcher["p_throws"].iloc[0] if not home_pitcher.empty else None
        home_hand = away_pitcher["p_throws"].iloc[0] if not away_pitcher.empty else None

        away_hitters = _filter_team_hitters(daily_hitters, rosters, hitter_pitcher_exclusions, rotowire_lineups, str(game.get("away_team") or ""), away_hand, filters)
        home_hitters = _filter_team_hitters(daily_hitters, rosters, hitter_pitcher_exclusions, rotowire_lineups, str(game.get("home_team") or ""), home_hand, filters)
        away_hitters = add_hitter_matchup_score(
            away_hitters,
            batter_zone_profiles=batter_zone_profiles,
            pitcher_zone_profiles=pitcher_zone_profiles,
            opposing_pitcher_id=home_pitcher_id,
            opposing_pitcher_hand=away_hand,
        )
        home_hitters = add_hitter_matchup_score(
            home_hitters,
            batter_zone_profiles=batter_zone_profiles,
            pitcher_zone_profiles=pitcher_zone_profiles,
            opposing_pitcher_id=away_pitcher_id,
            opposing_pitcher_hand=home_hand,
        )
        if away_pitcher_id:
            opponent_hitters_by_key[
                build_pitcher_matchup_key(game.get("game_pk"), away_pitcher_id, filters.split, filters.recent_window, filters.weighted_mode)
            ] = home_hitters
        if home_pitcher_id:
            opponent_hitters_by_key[
                build_pitcher_matchup_key(game.get("game_pk"), home_pitcher_id, filters.split, filters.recent_window, filters.weighted_mode)
            ] = away_hitters
        raw_pitchers_by_game[int(game.get("game_pk"))] = (away_pitcher, home_pitcher)
        if not away_pitcher.empty:
            slate_pitcher_frames.append(away_pitcher)
        if not home_pitcher.empty:
            slate_pitcher_frames.append(home_pitcher)
        game_label = f"{game.get('away_team', '')} @ {game.get('home_team', '')}"

    scored_slate_pitchers = (
        add_pitcher_rank_score(
            pd.concat(slate_pitcher_frames, ignore_index=True, sort=False).drop_duplicates(
                subset=["pitcher_id", "split_key", "recent_window", "weighted_mode"],
                keep="last",
            ),
            opponent_hitters_by_key=opponent_hitters_by_key or None,
            batter_family_zone_profiles=batter_family_zone_profiles,
            pitcher_family_zone_context=pitcher_family_zone_context,
        )
        if slate_pitcher_frames
        else pd.DataFrame()
    )
    scored_pitchers_by_id = {
        pitcher_id: group.copy()
        for pitcher_id, group in scored_slate_pitchers.groupby("pitcher_id", sort=False)
    } if not scored_slate_pitchers.empty else {}

    for game in slate.to_dict("records"):
        away_pitcher_id = game.get("away_probable_pitcher_id")
        home_pitcher_id = game.get("home_probable_pitcher_id")
        away_pitcher, home_pitcher = raw_pitchers_by_game.get(int(game.get("game_pk")), (pd.DataFrame(), pd.DataFrame()))
        away_pitcher = scored_pitchers_by_id.get(away_pitcher_id, away_pitcher).copy()
        home_pitcher = scored_pitchers_by_id.get(home_pitcher_id, home_pitcher).copy()
        away_hand = home_pitcher["p_throws"].iloc[0] if not home_pitcher.empty else None
        home_hand = away_pitcher["p_throws"].iloc[0] if not away_pitcher.empty else None
        away_hitters = add_hitter_matchup_score(
            _filter_team_hitters(daily_hitters, rosters, hitter_pitcher_exclusions, rotowire_lineups, str(game.get("away_team") or ""), away_hand, filters),
            batter_zone_profiles=batter_zone_profiles,
            pitcher_zone_profiles=pitcher_zone_profiles,
            opposing_pitcher_id=home_pitcher_id,
            opposing_pitcher_hand=away_hand,
        )
        home_hitters = add_hitter_matchup_score(
            _filter_team_hitters(daily_hitters, rosters, hitter_pitcher_exclusions, rotowire_lineups, str(game.get("home_team") or ""), home_hand, filters),
            batter_zone_profiles=batter_zone_profiles,
            pitcher_zone_profiles=pitcher_zone_profiles,
            opposing_pitcher_id=away_pitcher_id,
            opposing_pitcher_hand=home_hand,
        )
        game_label = f"{game.get('away_team', '')} @ {game.get('home_team', '')}"
        for hitter_frame, opponent_pitcher, team_label, opponent_team in [
            (away_hitters, home_pitcher, str(game.get("away_team") or ""), str(game.get("home_team") or "")),
            (home_hitters, away_pitcher, str(game.get("home_team") or ""), str(game.get("away_team") or "")),
        ]:
            opponent_pitcher_name = opponent_pitcher["pitcher_name"].iloc[0] if not opponent_pitcher.empty else ""
            opponent_pitcher_hand = opponent_pitcher["p_throws"].iloc[0] if not opponent_pitcher.empty else None
            opponent_pitcher_id = int(opponent_pitcher["pitcher_id"].iloc[0]) if not opponent_pitcher.empty else None
            for _, row in hitter_frame.iterrows():
                player_id = row.get("batter")
                if pd.isna(player_id):
                    continue
                key = _player_key("hitter", player_id)
                score = float(pd.to_numeric(pd.Series([row.get("matchup_score")]), errors="coerce").fillna(0).iloc[0])
                slate_lookup[key] = {
                    "entity_type": "hitter",
                    "player_id": int(player_id),
                    "player_name": str(row.get("hitter_name") or ""),
                    "team": team_label,
                    "game": game_label,
                    "opponent_team": opponent_team,
                    "opposing_pitcher_name": opponent_pitcher_name,
                    "opposing_pitcher_hand": opponent_pitcher_hand,
                    "opposing_pitcher_id": opponent_pitcher_id,
                    "row": row.to_dict(),
                    "current_score": score,
                }
                if top_score is None or score > top_score:
                    top_score = score
                    top_key = key

        for pitcher_single, opponent_hitters, team_label, opponent_team in [
            (away_pitcher, home_hitters, str(game.get("away_team") or ""), str(game.get("home_team") or "")),
            (home_pitcher, away_hitters, str(game.get("home_team") or ""), str(game.get("away_team") or "")),
        ]:
            if pitcher_single.empty:
                continue
            row = pitcher_single.iloc[0]
            player_id = row.get("pitcher_id")
            if pd.isna(player_id):
                continue
            key = _player_key("pitcher", player_id)
            score = float(pd.to_numeric(pd.Series([row.get("pitcher_score")]), errors="coerce").fillna(0).iloc[0])
            slate_lookup[key] = {
                "entity_type": "pitcher",
                "player_id": int(player_id),
                "player_name": str(row.get("pitcher_name") or ""),
                "team": team_label,
                "game": game_label,
                "opponent_team": opponent_team,
                "opponent_hitters": opponent_hitters.copy(),
                "row": row.to_dict(),
                "current_score": score,
            }
            if top_score is None or score > top_score:
                top_score = score
                top_key = key

    return slate_lookup, top_key


def _build_player_index(
    reusable: dict[str, pd.DataFrame],
    filters: QueryFilters,
    slate_lookup: dict[str, dict[str, object]],
) -> pd.DataFrame:
    hitter_frame = _filter_hitter_metrics(reusable.get("hitters", pd.DataFrame()), filters)
    pitcher_frame = add_pitcher_rank_score(_filter_pitcher_metrics(reusable.get("pitchers", pd.DataFrame()), filters))
    hitter_team = hitter_frame["team"] if "team" in hitter_frame.columns else pd.Series("", index=hitter_frame.index, dtype="object")
    pitcher_team = pitcher_frame["team"] if "team" in pitcher_frame.columns else pd.Series("", index=pitcher_frame.index, dtype="object")

    hitter_index = pd.DataFrame()
    if not hitter_frame.empty:
        hitter_index = (
            hitter_frame.assign(
                entity_type="Hitter",
                player_id=hitter_frame["batter"].astype(int),
                player_name=hitter_frame["hitter_name"].astype(str),
                team=hitter_team.fillna(""),
                hand=hitter_frame.get("stand"),
                key=hitter_frame["batter"].astype(int).map(lambda value: _player_key("hitter", value)),
                sort_value=pd.to_numeric(hitter_frame["xwoba"], errors="coerce"),
            )[["entity_type", "player_id", "player_name", "team", "hand", "key", "sort_value"]]
            .drop_duplicates("key")
        )

    pitcher_index = pd.DataFrame()
    if not pitcher_frame.empty:
        pitcher_index = (
            pitcher_frame.assign(
                entity_type="Pitcher",
                player_id=pitcher_frame["pitcher_id"].astype(int),
                player_name=pitcher_frame["pitcher_name"].astype(str),
                team=pitcher_team.fillna(""),
                hand=pitcher_frame.get("p_throws"),
                key=pitcher_frame["pitcher_id"].astype(int).map(lambda value: _player_key("pitcher", value)),
                sort_value=pd.to_numeric(pitcher_frame["pitcher_score"], errors="coerce"),
            )[["entity_type", "player_id", "player_name", "team", "hand", "key", "sort_value"]]
            .drop_duplicates("key")
        )

    combined = pd.concat([hitter_index, pitcher_index], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    combined["on_slate"] = combined["key"].isin(slate_lookup)
    combined["label"] = combined.apply(lambda row: f"{row['player_name']} | {row['entity_type']} | {row.get('team', '')}", axis=1)
    return combined.sort_values(["on_slate", "sort_value", "player_name"], ascending=[False, False, True], na_position="last").reset_index(drop=True)


def _select_player(player_index: pd.DataFrame, default_key: str | None) -> tuple[str | None, pd.Series | None]:
    if player_index.empty:
        return None, None
    search_value = st.text_input("Player search", value=st.session_state.get("player-analysis-search", ""))
    st.session_state["player-analysis-search"] = search_value
    work = player_index.copy()
    if search_value.strip():
        needle = search_value.strip().casefold()
        work = work.loc[work["label"].str.casefold().str.contains(needle)]
    if work.empty:
        st.info("No players matched the current search.")
        return None, None

    current_key = st.session_state.get("player-analysis-selected-key", default_key or work["key"].iloc[0])
    if current_key not in set(work["key"]):
        current_key = work["key"].iloc[0]
    labels = work["label"].tolist()
    label_by_key = dict(zip(work["key"], labels))
    selected_label = st.selectbox("Player", options=labels, index=labels.index(label_by_key[current_key]))
    selected_row = work.loc[work["label"] == selected_label].iloc[0]
    st.session_state["player-analysis-selected-key"] = selected_row["key"]
    return str(selected_row["key"]), selected_row


def _format_metric_value(value: object, column: str) -> str:
    if value is None or pd.isna(value):
        return "--"
    if column in {"xwoba", "xwoba_con"}:
        return f"{float(value):.3f}"
    if column in {"matchup_score", "test_score", "ceiling_score", "zone_fit_score", "pitcher_score", "strikeout_score", "likely_starter_score", "gb_fb_ratio", "siera"}:
        return f"{float(value):.1f}"
    if column in {"pulled_barrel_pct", "barrel_bip_pct", "fb_pct", "hard_hit_pct", "swstr_pct", "called_strike_pct", "csw_pct", "putaway_pct", "ball_pct", "gb_pct"}:
        return f"{float(value) * 100:.1f}%"
    return str(value)


def _render_overview_cards(cards: list[tuple[str, str]], row: pd.Series) -> None:
    columns = st.columns(3)
    for idx, (label, column) in enumerate(cards):
        with columns[idx % 3]:
            st.metric(label, _format_metric_value(row.get(column), column))


def _render_profile_header(selected_row: pd.Series, slate_entry: dict[str, object] | None, resolved_date: date) -> None:
    badge = "Hitter" if selected_row["entity_type"] == "Hitter" else "Pitcher"
    hand_label = str(selected_row.get("hand") or "-")
    parts = [str(selected_row.get("team") or "-"), hand_label, resolved_date.isoformat()]
    if slate_entry is not None:
        parts.append(str(slate_entry.get("game") or "On slate"))
    st.title(str(selected_row["player_name"]))
    st.caption(f"{badge} | " + " | ".join(parts))


def _render_overview_tab(selected_row: pd.Series, profile_row: pd.Series, source: str) -> None:
    st.subheader("Profile")
    if selected_row["entity_type"] == "Hitter":
        _render_overview_cards(HITTER_OVERVIEW_CARDS, profile_row)
        display = pd.DataFrame([profile_row]).reindex(columns=[column for column in HITTER_OVERVIEW_COLUMNS if column in profile_row.index])
        display_columns, display_hidden = _hitter_table_columns(display, list(display.columns))
        _render_hitter_confidence_legend()
        render_metric_grid(
            display[display_columns],
            key=f"player-overview-h-{selected_row['player_id']}",
            height=130,
            use_lightweight=(source == "hosted"),
            hidden_columns=display_hidden,
            color_hitter_confidence=True,
        )
    else:
        _render_overview_cards(PITCHER_OVERVIEW_CARDS, profile_row)
        display = pd.DataFrame([profile_row]).reindex(columns=[column for column in PITCHER_OVERVIEW_COLUMNS if column in profile_row.index])
        render_metric_grid(
            display,
            key=f"player-overview-p-{selected_row['player_id']}",
            height=130,
            lower_is_better=PITCHER_LOWER_IS_BETTER,
            higher_is_better=PITCHER_HIGHER_IS_BETTER,
            use_lightweight=(source == "hosted"),
        )


def _render_rolling_tab(selected_row: pd.Series, reusable: dict[str, pd.DataFrame], source: str) -> None:
    entity_type = str(selected_row["entity_type"])
    player_name = str(selected_row["player_name"])
    if entity_type == "Hitter":
        rolling = reusable.get("hitter_rolling", pd.DataFrame())
        metrics = ["xwoba", "pulled_barrel_pct", "hard_hit_pct", "fb_pct"]
        columns = HITTER_ROLLING_COLUMNS
        lower_is_better = None
        higher_is_better = None
    else:
        rolling = reusable.get("pitcher_rolling", pd.DataFrame())
        metrics = ["avg_release_speed", "barrel_bip_pct", "hard_hit_pct", "fb_pct"]
        columns = PITCHER_ROLLING_COLUMNS
        lower_is_better = PITCHER_LOWER_IS_BETTER | {"barrel_bip_pct"}
        higher_is_better = PITCHER_HIGHER_IS_BETTER | {"avg_release_speed"}

    frame = rolling.loc[rolling["player_name"] == player_name].copy() if not rolling.empty else pd.DataFrame()
    if frame.empty:
        st.info("No rolling data available for this player.")
        return
    frame["rolling_window"] = pd.Categorical(frame["rolling_window"], categories=ROLLING_WINDOW_ORDER, ordered=True)
    frame = frame.sort_values("rolling_window")
    metric = st.selectbox("Rolling metric", metrics, key=f"rolling-metric-{selected_row['key']}")
    chart_frame = frame[["rolling_window", metric]].dropna().set_index("rolling_window")
    if not chart_frame.empty:
        st.line_chart(chart_frame)
    render_metric_grid(
        frame[[column for column in columns if column in frame.columns]],
        key=f"rolling-grid-{selected_row['key']}",
        height=210,
        lower_is_better=lower_is_better,
        higher_is_better=higher_is_better,
        use_lightweight=(source == "hosted"),
    )


def _render_hitter_zones_tab(selected_row: pd.Series, reusable: dict[str, pd.DataFrame], slate_entry: dict[str, object] | None, source: str) -> None:
    zone_profiles = reusable.get("batter_zone_profiles", pd.DataFrame())
    player_id = int(selected_row["player_id"])
    frame = zone_profiles.loc[zone_profiles["batter_id"] == player_id].copy() if not zone_profiles.empty else pd.DataFrame()
    if frame.empty:
        st.info("No hitter zone data available for this player.")
        return
    options = [value for value in ["overall", "vs_rhp", "vs_lhp"] if value in frame.get("pitcher_hand_key", pd.Series(dtype="object")).dropna().unique().tolist()] or ["overall"]
    default_hand = "overall"
    if slate_entry is not None:
        default_hand = {"R": "vs_rhp", "L": "vs_lhp"}.get(str(slate_entry.get("opposing_pitcher_hand") or ""), "overall")
    hand_key = st.selectbox("Pitcher hand context", options, index=options.index(default_hand) if default_hand in options else 0, key=f"h-zone-hand-{selected_row['key']}")
    detail = frame.loc[frame["pitcher_hand_key"] == hand_key].copy() if "pitcher_hand_key" in frame.columns else frame.copy()
    detail["damage_rate"] = (
        pd.to_numeric(detail.get("hit_rate"), errors="coerce").fillna(0) * 0.6
        + pd.to_numeric(detail.get("hr_rate"), errors="coerce").fillna(0) * 0.4
    )
    pitch_options = ["All pitches"] + sorted(detail["pitch_type"].dropna().astype(str).unique().tolist())
    selected_pitch = st.selectbox("Pitch type", pitch_options, key=f"h-zone-pitch-{selected_row['key']}")
    hitter_map = aggregate_batter_zone_map(detail, selected_pitch)
    render_zone_heatmap(f"{selected_row['player_name']} Zone Damage", f"{selected_pitch} | {hand_key}", hitter_map)
    render_metric_grid(
        detail[[column for column in BATTER_ZONE_COLUMNS if column in detail.columns]],
        key=f"h-zone-grid-{selected_row['key']}",
        height=230,
        higher_is_better={"hit_rate", "hr_rate", "damage_rate"},
        use_lightweight=(source == "hosted"),
    )


def _render_pitcher_zones_tab(selected_row: pd.Series, reusable: dict[str, pd.DataFrame], source: str) -> None:
    zone_profiles = reusable.get("pitcher_zone_profiles", pd.DataFrame())
    player_id = int(selected_row["player_id"])
    frame = zone_profiles.loc[zone_profiles["pitcher_id"] == player_id].copy() if not zone_profiles.empty else pd.DataFrame()
    if frame.empty:
        st.info("No pitcher zone data available for this player.")
        return
    options = [value for value in ["overall", "vs_lhh", "vs_rhh"] if value in frame.get("batter_side_key", pd.Series(dtype="object")).dropna().unique().tolist()] or ["overall"]
    side_key = st.selectbox("Batter side context", options, key=f"p-zone-side-{selected_row['key']}")
    detail = frame.loc[frame["batter_side_key"] == side_key].copy() if "batter_side_key" in frame.columns else frame.copy()
    pitch_options = ["All pitches"] + sorted(detail["pitch_type"].dropna().astype(str).unique().tolist())
    selected_pitch = st.selectbox("Pitch type", pitch_options, key=f"p-zone-pitch-{selected_row['key']}")
    pitcher_map = aggregate_pitcher_zone_map(detail, selected_pitch)
    render_zone_heatmap(f"{selected_row['player_name']} Zone Attack", f"{selected_pitch} | {side_key}", pitcher_map)
    render_metric_grid(
        detail[[column for column in PITCHER_ZONE_COLUMNS if column in detail.columns]],
        key=f"p-zone-grid-{selected_row['key']}",
        height=230,
        higher_is_better={"usage_rate"},
        use_lightweight=(source == "hosted"),
    )


def _render_arsenal_split_chart(by_hand: pd.DataFrame) -> None:
    if by_hand.empty:
        st.info("No arsenal split data available for this pitcher.")
        return
    chart_source = by_hand.loc[by_hand["batter_side_key"].isin(["vs_lhh", "vs_rhh"]), ["pitch_name", "batter_side_key", "usage_pct"]].copy()
    if chart_source.empty or not HAS_ALTAIR:
        st.info("Mirrored arsenal chart unavailable. Showing the arsenal grid below.")
        return
    pivot = chart_source.pivot_table(index="pitch_name", columns="batter_side_key", values="usage_pct", aggfunc="max").fillna(0)
    pivot["sort_value"] = pivot.max(axis=1)
    pitch_order = pivot.sort_values("sort_value", ascending=False).index.tolist()
    max_usage = max(float(pivot[["vs_lhh", "vs_rhh"]].max().max()), 0.01)
    left_frame = pd.DataFrame({"pitch_name": pitch_order, "usage_pct": [float(pivot.loc[pitch, "vs_lhh"]) for pitch in pitch_order]})
    right_frame = pd.DataFrame({"pitch_name": pitch_order, "usage_pct": [float(pivot.loc[pitch, "vs_rhh"]) for pitch in pitch_order]})
    label_frame = pd.DataFrame({"pitch_name": pitch_order})

    left_chart = (
        alt.Chart(left_frame)
        .mark_bar(color="#8bb7f0", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("pitch_name:N", sort=pitch_order, axis=None),
            x=alt.X("usage_pct:Q", scale=alt.Scale(domain=[max_usage, 0]), axis=alt.Axis(format=".0%", title="vs LHH")),
            tooltip=[alt.Tooltip("pitch_name:N", title="Pitch"), alt.Tooltip("usage_pct:Q", title="Usage", format=".1%")],
        )
        .properties(width=220, height=max(180, len(pitch_order) * 28))
    )
    right_chart = (
        alt.Chart(right_frame)
        .mark_bar(color="#5f95e8", cornerRadiusTopLeft=3, cornerRadiusBottomLeft=3)
        .encode(
            y=alt.Y("pitch_name:N", sort=pitch_order, axis=None),
            x=alt.X("usage_pct:Q", scale=alt.Scale(domain=[0, max_usage]), axis=alt.Axis(format=".0%", title="vs RHH")),
            tooltip=[alt.Tooltip("pitch_name:N", title="Pitch"), alt.Tooltip("usage_pct:Q", title="Usage", format=".1%")],
        )
        .properties(width=220, height=max(180, len(pitch_order) * 28))
    )
    center_chart = (
        alt.Chart(label_frame)
        .mark_text(align="center", baseline="middle", fontSize=12, fontWeight="bold", color="#334155")
        .encode(y=alt.Y("pitch_name:N", sort=pitch_order, axis=None), text="pitch_name:N")
        .properties(width=120, height=max(180, len(pitch_order) * 28))
    )
    st.altair_chart(alt.hconcat(left_chart, center_chart, right_chart, spacing=8), use_container_width=True)


def _render_pitcher_arsenal_tab(selected_row: pd.Series, reusable: dict[str, pd.DataFrame], filters: QueryFilters, source: str) -> None:
    player_id = int(selected_row["player_id"])
    overall = sort_arsenal_frame(_filter_pitcher_detail(reusable.get("pitcher_arsenal", pd.DataFrame()), player_id, filters))
    by_hand = sort_arsenal_frame(_filter_pitcher_detail(reusable.get("pitcher_arsenal_by_hand", pd.DataFrame()), player_id, filters))
    _render_arsenal_split_chart(by_hand)
    if overall.empty:
        st.info("No overall arsenal data available for this pitcher.")
        return
    render_metric_grid(
        overall[[column for column in ARSENAL_COLUMNS if column in overall.columns]],
        key=f"arsenal-grid-{selected_row['key']}",
        height=260,
        lower_is_better={"hard_hit_pct", "xwoba_con"},
        higher_is_better={"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
        use_lightweight=(source == "hosted"),
    )


def _render_pitcher_count_usage_tab(selected_row: pd.Series, reusable: dict[str, pd.DataFrame], filters: QueryFilters, source: str) -> None:
    player_id = int(selected_row["player_id"])
    count_usage = _filter_pitcher_detail(reusable.get("pitcher_usage_by_count", pd.DataFrame()), player_id, filters)
    overall = _filter_pitcher_detail(reusable.get("pitcher_arsenal", pd.DataFrame()), player_id, filters)
    by_hand = _filter_pitcher_detail(reusable.get("pitcher_arsenal_by_hand", pd.DataFrame()), player_id, filters)
    tabs = st.tabs([BATTER_SIDE_LABELS[key] for key in BATTER_SIDE_LABELS])
    for side_key, tab in zip(BATTER_SIDE_LABELS, tabs):
        with tab:
            side_count = count_usage.loc[count_usage["batter_side_key"] == side_key] if "batter_side_key" in count_usage.columns else count_usage
            if side_key == "all":
                side_arsenal = overall[["pitch_name", "usage_pct"]] if not overall.empty else pd.DataFrame(columns=["pitch_name", "usage_pct"])
            else:
                side_arsenal = by_hand.loc[by_hand["batter_side_key"] == side_key, ["pitch_name", "usage_pct"]] if not by_hand.empty else pd.DataFrame(columns=["pitch_name", "usage_pct"])
            table = pivot_count_usage(side_count, side_arsenal)
            if table.empty:
                st.info("No count usage data available.")
            else:
                render_metric_grid(
                    table,
                    key=f"count-usage-{selected_row['key']}-{side_key}",
                    height=230,
                    higher_is_better=set(table.columns) - {"pitch_name"},
                    use_lightweight=(source == "hosted"),
                )


def _render_pitch_shape_tab(selected_row: pd.Series, reusable: dict[str, pd.DataFrame], source: str) -> None:
    player_id = int(selected_row["player_id"])
    movement = reusable.get("pitcher_movement_arsenal", pd.DataFrame())
    family_context = reusable.get("pitcher_family_zone_context", pd.DataFrame())
    movement_detail = movement.loc[movement["pitcher_id"] == player_id].copy() if not movement.empty else pd.DataFrame()
    family_detail = family_context.loc[family_context["pitcher_id"] == player_id].copy() if not family_context.empty else pd.DataFrame()

    cols = st.columns(2)
    with cols[0]:
        st.markdown("##### Movement Arsenal")
        if movement_detail.empty:
            st.info("No movement arsenal context available.")
        else:
            render_metric_grid(
                movement_detail[[column for column in MOVEMENT_ARSENAL_COLUMNS if column in movement_detail.columns]].sort_values(
                    ["usage_rate", "weighted_sample_size"], ascending=[False, False], na_position="last"
                ),
                key=f"pitch-shape-arsenal-{selected_row['key']}",
                height=260,
                higher_is_better={"usage_rate", "avg_velocity", "avg_spin_rate", "avg_extension", "avg_pfx_x", "avg_pfx_z"},
                use_lightweight=(source == "hosted"),
            )
    with cols[1]:
        st.markdown("##### Family-Zone Tendencies")
        if family_detail.empty:
            st.info("No family-zone context available.")
        else:
            render_metric_grid(
                family_detail[[column for column in FAMILY_ZONE_CONTEXT_COLUMNS if column in family_detail.columns]].sort_values(
                    ["usage_rate_overall", "weighted_sample_size"], ascending=[False, False], na_position="last"
                ),
                key=f"pitch-shape-family-{selected_row['key']}",
                height=260,
                lower_is_better={"prior_weight_share", "damage_allowed_rate", "xwoba_allowed"},
                higher_is_better={"usage_rate_overall", "whiff_rate", "called_strike_rate"},
                use_lightweight=(source == "hosted"),
            )


def _render_hitter_matchup_tab(selected_row: pd.Series, slate_entry: dict[str, object], daily: dict[str, pd.DataFrame], source: str) -> None:
    current_row = pd.DataFrame([slate_entry["row"]])
    st.caption(f"{slate_entry['game']} | vs {slate_entry.get('opposing_pitcher_name') or 'opposing starter'}")
    matchup_columns, matchup_hidden = _hitter_table_columns(current_row, MATCHUP_HITTER_COLUMNS)
    _render_hitter_confidence_legend()
    render_metric_grid(
        current_row[matchup_columns],
        key=f"h-matchup-row-{selected_row['key']}",
        height=120,
        use_lightweight=(source == "hosted"),
        hidden_columns=matchup_hidden,
        color_hitter_confidence=True,
    )
    batter_frame = daily.get("daily_batter_zone_profiles", pd.DataFrame())
    pitcher_frame = daily.get("daily_pitcher_zone_profiles", pd.DataFrame())
    batter_detail = batter_frame.loc[batter_frame["batter_id"] == int(selected_row["player_id"])].copy() if not batter_frame.empty else pd.DataFrame()
    pitcher_detail = pitcher_frame.loc[pitcher_frame["pitcher_id"] == slate_entry.get("opposing_pitcher_id")].copy() if not pitcher_frame.empty else pd.DataFrame()
    if batter_detail.empty or pitcher_detail.empty:
        st.info("No matchup zone overlay data available.")
    else:
        pitch_types = ["All pitches"] + sorted(
            set(batter_detail.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
            | set(pitcher_detail.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
        )
        selected_pitch = st.selectbox("Pitch type", pitch_types, key=f"h-matchup-pitch-{selected_row['key']}")
        batter_map = aggregate_batter_zone_map(batter_detail, selected_pitch)
        pitcher_map = aggregate_pitcher_zone_map(pitcher_detail, selected_pitch)
        overlay_map = build_zone_overlay_map(batter_map, pitcher_map)
        cols = st.columns(2)
        with cols[0]:
            render_zone_heatmap(f"{selected_row['player_name']} Damage", f"{selected_pitch} | Hitter quality", batter_map)
        with cols[1]:
            render_zone_heatmap(f"Overlay vs {slate_entry.get('opposing_pitcher_name') or 'Opposing Pitcher'}", f"{selected_pitch} | Damage x usage", overlay_map)

    batter_family = daily.get("daily_batter_family_zone_profiles", pd.DataFrame())
    pitcher_family = daily.get("daily_pitcher_family_zone_context", pd.DataFrame())
    family_fit = compute_family_fit_score(
        batter_family,
        pitcher_family,
        int(selected_row["player_id"]),
        slate_entry.get("opposing_pitcher_id"),
    )
    detail = build_family_zone_fit_detail(
        batter_family,
        pitcher_family,
        int(selected_row["player_id"]),
        slate_entry.get("opposing_pitcher_id"),
    )
    st.markdown("##### Family-Zone Context")
    st.metric("Family Fit", f"{family_fit * 100:.1f}" if family_fit is not None else "N/A")
    if detail.empty:
        st.info("No family-zone matchup context available.")
    else:
        render_metric_grid(
            detail[[column for column in FAMILY_FIT_DETAIL_COLUMNS if column in detail.columns]],
            key=f"h-family-fit-{selected_row['key']}",
            height=220,
            lower_is_better={"prior_weight_share"},
            higher_is_better={"usage_rate_overall", "damage_rate", "xwoba", "fit_score"},
            use_lightweight=(source == "hosted"),
        )


def _render_pitcher_matchup_tab(selected_row: pd.Series, slate_entry: dict[str, object], source: str) -> None:
    current_row = pd.DataFrame([slate_entry["row"]])
    st.caption(f"{slate_entry['game']} | vs {slate_entry.get('opponent_team')}")
    render_metric_grid(
        current_row[[column for column in MATCHUP_PITCHER_COLUMNS if column in current_row.columns]],
        key=f"p-matchup-row-{selected_row['key']}",
        height=170,
        lower_is_better=PITCHER_LOWER_IS_BETTER,
        higher_is_better=PITCHER_HIGHER_IS_BETTER,
        use_lightweight=(source == "hosted"),
    )
    opponent_hitters = slate_entry.get("opponent_hitters", pd.DataFrame())
    if isinstance(opponent_hitters, pd.DataFrame) and not opponent_hitters.empty:
        st.markdown("##### Top opposing hitters")
        opponent_columns, opponent_hidden = _hitter_table_columns(opponent_hitters, OPPONENT_HITTER_COLUMNS)
        _render_hitter_confidence_legend()
        render_metric_grid(
            opponent_hitters[opponent_columns].head(6),
            key=f"p-matchup-hitters-{selected_row['key']}",
            height=220,
            use_lightweight=(source == "hosted"),
            hidden_columns=opponent_hidden,
            color_hitter_confidence=True,
        )
    else:
        st.info("No opposing hitter context available.")


def main() -> None:
    st.set_page_config(page_title="Player Analysis", page_icon=page_icon_path(), layout="wide")
    apply_branding_head()
    config = AppConfig()
    target_date, filters = _sidebar(config)
    daily, reusable, source, resolved_date = _load_context(config, target_date)
    if source == "none":
        st.error("No local artifacts or hosted artifact base URL were found for loading player context.")
        return
    if resolved_date != target_date:
        st.caption(f"Using most recent available published slate: {resolved_date.isoformat()}")

    slate_lookup, top_slate_key = _build_slate_lookup(daily, filters, resolved_date)
    player_index = _build_player_index(reusable, filters, slate_lookup)
    if player_index.empty:
        st.info("No player data available for the selected filters.")
        return
    selected_key, selected_row = _select_player(player_index, top_slate_key)
    if selected_row is None or selected_key is None:
        return

    slate_entry = slate_lookup.get(selected_key)
    entity_type = str(selected_row["entity_type"])
    if entity_type == "Hitter":
        profile_frame = _filter_hitter_metrics(reusable.get("hitters", pd.DataFrame()), filters)
        profile_frame = profile_frame.loc[profile_frame["batter"] == int(selected_row["player_id"])].copy()
        profile_row = profile_frame.iloc[0] if not profile_frame.empty else pd.Series(dtype="object")
        if slate_entry is not None and not profile_row.empty:
            for column in ["matchup_score", "test_score", "ceiling_score", "zone_fit_score", "likely_starter_score"]:
                profile_row[column] = slate_entry["row"].get(column)
    else:
        profile_frame = add_pitcher_rank_score(_filter_pitcher_metrics(reusable.get("pitchers", pd.DataFrame()), filters))
        profile_frame = profile_frame.loc[profile_frame["pitcher_id"] == int(selected_row["player_id"])].copy()
        profile_row = profile_frame.iloc[0] if not profile_frame.empty else pd.Series(dtype="object")
        if slate_entry is not None and not profile_row.empty:
            for column in [
                "pitcher_score",
                "strikeout_score",
                "raw_pitcher_score",
                "raw_strikeout_score",
                "pitcher_matchup_adjustment",
                "strikeout_matchup_adjustment",
                "opponent_lineup_quality",
                "opponent_contact_threat",
                "opponent_whiff_tendency",
                "opponent_family_fit_allowed",
                "lineup_source",
                "lineup_hitter_count",
            ]:
                profile_row[column] = slate_entry["row"].get(column)

    if profile_row.empty:
        st.info("No profile row was found for the selected player under the current filters.")
        return

    _render_profile_header(selected_row, slate_entry, resolved_date)

    tab_names = ["Overview", "Rolling", "Zones"]
    if slate_entry is not None:
        tab_names.append("Matchup Context")
    if entity_type == "Pitcher":
        tab_names.extend(["Pitch Shape", "Arsenal", "Count Usage"])
    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_names, tabs))

    with tab_map["Overview"]:
        _render_overview_tab(selected_row, profile_row, source)
    with tab_map["Rolling"]:
        _render_rolling_tab(selected_row, reusable, source)
    with tab_map["Zones"]:
        if entity_type == "Hitter":
            _render_hitter_zones_tab(selected_row, reusable, slate_entry, source)
        else:
            _render_pitcher_zones_tab(selected_row, reusable, source)
    if "Matchup Context" in tab_map:
        with tab_map["Matchup Context"]:
            if entity_type == "Hitter":
                _render_hitter_matchup_tab(selected_row, slate_entry, daily, source)
            else:
                _render_pitcher_matchup_tab(selected_row, slate_entry, source)
    if entity_type == "Pitcher":
        with tab_map["Pitch Shape"]:
            _render_pitch_shape_tab(selected_row, reusable, source)
        with tab_map["Arsenal"]:
            _render_pitcher_arsenal_tab(selected_row, reusable, filters, source)
        with tab_map["Count Usage"]:
            _render_pitcher_count_usage_tab(selected_row, reusable, filters, source)
