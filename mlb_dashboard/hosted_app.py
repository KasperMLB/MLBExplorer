from __future__ import annotations

import os
from datetime import date, timedelta
from time import perf_counter

import pandas as pd
import streamlit as st

from .branding import page_icon_path
from .dashboard_views import (
    ARSENAL_COLUMNS,
    BATTER_SIDE_LABELS,
    BATTER_ZONE_COLUMNS,
    BEST_MATCHUP_COLUMNS,
    COUNT_BUCKET_ORDER,
    COUNT_USAGE_COLUMNS,
    FAMILY_ZONE_CONTEXT_COLUMNS,
    HITTER_PRESETS,
    HITTER_ROLLING_COLUMNS,
    MOVEMENT_ARSENAL_COLUMNS,
    PITCHER_HIGHER_IS_BETTER,
    PITCHER_LOWER_IS_BETTER,
    PITCHER_ROLLING_COLUMNS,
    PITCHER_SUMMARY_COLUMNS,
    PITCHER_ZONE_COLUMNS,
    TOP_PITCHER_COLUMNS,
    aggregate_batter_zone_map,
    aggregate_pitcher_zone_map,
    add_hitter_matchup_score,
    add_pitcher_rank_score,
    apply_roster_names,
    build_pitcher_matchup_key,
    build_zone_overlay_map,
    build_best_matchups,
    build_full_slate_export_bundles,
    build_slate_export_options,
    compute_family_fit_score,
    build_game_export_options,
    filter_excluded_pitchers_from_hitter_pool,
    apply_projected_lineup,
    hitter_columns_for_preset,
    pivot_count_usage,
    sort_arsenal_frame,
    with_game_label,
)
from .query_engine import load_remote_parquet
from .rotowire_lineups import fetch_rotowire_lineups, resolve_rotowire_lineups
from .ui_components import (
    build_pitcher_summary_table,
    render_export_hub,
    render_matchup_header,
    render_metric_grid,
    render_slate_export_controls,
    render_zone_heatmap,
)


def _base_url() -> str:
    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


def _perf_enabled() -> bool:
    return os.getenv("MLB_PERF_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _record_perf(events: list[tuple[str, float]], label: str, start_time: float) -> None:
    if _perf_enabled():
        events.append((label, perf_counter() - start_time))


def _render_perf(events: list[tuple[str, float]]) -> None:
    if not _perf_enabled() or not events:
        return
    st.caption("Perf: " + " | ".join(f"{label} {duration:.2f}s" for label, duration in events))


def _is_mobile_request() -> bool:
    try:
        context = getattr(st, "context", None)
        headers = getattr(context, "headers", None)
        if headers is None:
            return False
        user_agent = headers.get("User-Agent", "") or headers.get("user-agent", "")
        agent = str(user_agent).lower()
        return any(token in agent for token in ("iphone", "android", "ipad", "mobile", "blackberry", "opera mini", "windows phone"))
    except Exception:
        return False


def _render_hosted_grid(
    frame: pd.DataFrame,
    key: str,
    mobile_safe: bool,
    height: int = 320,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> pd.DataFrame:
    return render_metric_grid(
        frame,
        key=key,
        height=height,
        lower_is_better=lower_is_better,
        higher_is_better=higher_is_better,
        use_lightweight=True,
    )


def _present_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.head(0).copy() if not frame.empty else frame.copy()


def _build_family_fit_board(
    hitters: pd.DataFrame,
    batter_family_zone_profiles: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    pitcher_id: int | None,
) -> pd.DataFrame:
    if hitters.empty or pitcher_id is None or batter_family_zone_profiles.empty or pitcher_family_zone_context.empty:
        return pd.DataFrame(columns=["hitter_name", "team", "family_fit_score", "matchup_score", "ceiling_score", "xwoba"])
    rows: list[dict] = []
    for _, row in hitters.iterrows():
        batter_id = pd.to_numeric(pd.Series([row.get("batter")]), errors="coerce").iloc[0]
        if pd.isna(batter_id):
            continue
        fit_score = compute_family_fit_score(
            batter_family_zone_profiles,
            pitcher_family_zone_context,
            int(batter_id),
            int(pitcher_id),
        )
        rows.append(
            {
                "hitter_name": row.get("hitter_name"),
                "team": row.get("team"),
                "family_fit_score": None if fit_score is None else float(fit_score) * 100.0,
                "matchup_score": row.get("matchup_score"),
                "ceiling_score": row.get("ceiling_score"),
                "xwoba": row.get("xwoba"),
            }
        )
    return pd.DataFrame(rows).sort_values(["family_fit_score", "matchup_score"], ascending=[False, False], na_position="last")


def _render_pitch_shape_context(
    game_pk: int,
    team_label: str,
    pitcher_row: pd.DataFrame,
    movement_arsenal: pd.DataFrame,
    family_context: pd.DataFrame,
    opposing_hitters: pd.DataFrame,
    batter_family_zone_profiles: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    mobile_safe: bool,
) -> None:
    if pitcher_row.empty:
        st.info("No pitch-shape context available.")
        return
    pitcher_id = int(pitcher_row["pitcher_id"].iloc[0])
    movement_detail = movement_arsenal.loc[movement_arsenal["pitcher_id"] == pitcher_id].copy() if not movement_arsenal.empty else pd.DataFrame()
    family_detail = family_context.loc[family_context["pitcher_id"] == pitcher_id].copy() if not family_context.empty else pd.DataFrame()
    family_board = _build_family_fit_board(opposing_hitters, batter_family_zone_profiles, pitcher_family_zone_context, pitcher_id)

    st.markdown(f"##### {team_label} Pitch Shape")
    if movement_detail.empty:
        st.info("No weighted movement arsenal data available.")
    else:
        _render_hosted_grid(
            movement_detail[[column for column in MOVEMENT_ARSENAL_COLUMNS if column in movement_detail.columns]],
            key=f"movement-arsenal-hosted-{game_pk}-{team_label}",
            mobile_safe=mobile_safe,
            height=230,
            higher_is_better={"usage_rate", "avg_velocity", "avg_spin_rate", "avg_extension", "avg_pfx_x", "avg_pfx_z"},
        )
    if family_detail.empty:
        st.info("No weighted family-zone profile available.")
    else:
        _render_hosted_grid(
            family_detail[[column for column in FAMILY_ZONE_CONTEXT_COLUMNS if column in family_detail.columns]],
            key=f"family-context-hosted-{game_pk}-{team_label}",
            mobile_safe=mobile_safe,
            height=210,
            lower_is_better={"prior_weight_share", "damage_allowed_rate", "xwoba_allowed"},
            higher_is_better={"usage_rate_overall", "whiff_rate", "called_strike_rate"},
        )
    if family_board.empty:
        st.info("No opposing-hitter family fit context available.")
    else:
        _render_hosted_grid(
            family_board.head(6),
            key=f"family-fit-board-hosted-{game_pk}-{team_label}",
            mobile_safe=mobile_safe,
            height=210,
            higher_is_better={"family_fit_score", "matchup_score", "ceiling_score", "xwoba"},
        )


def _frame_by_pitcher_id(frame: pd.DataFrame) -> dict[object, pd.DataFrame]:
    if frame.empty or "pitcher_id" not in frame.columns:
        return {}
    return {pitcher_id: group.copy() for pitcher_id, group in frame.groupby("pitcher_id", sort=False)}


@st.cache_data(show_spinner=False)
def _load_core_artifacts(
    base_url: str,
    target_date: date,
) -> tuple[pd.DataFrame, ...]:
    day = target_date.isoformat()
    slate = pd.read_parquet(f"{base_url}/daily/{day}/slate.parquet")
    rosters = pd.read_parquet(f"{base_url}/daily/{day}/rosters.parquet")
    hitters = pd.read_parquet(f"{base_url}/daily/{day}/daily_hitter_metrics.parquet")
    pitchers = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_metrics.parquet")
    pitcher_summary_by_hand = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_summary_by_hand.parquet")
    arsenal = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_arsenal.parquet")
    arsenal_by_hand = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_arsenal_by_hand.parquet")
    usage_by_count = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_usage_by_count.parquet")
    batter_zone_profiles = pd.read_parquet(f"{base_url}/daily/{day}/daily_batter_zone_profiles.parquet")
    pitcher_zone_profiles = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_zone_profiles.parquet")
    try:
        hitter_pitcher_exclusions = pd.read_parquet(f"{base_url}/daily/{day}/hitter_pitcher_exclusions.parquet")
    except Exception:
        hitter_pitcher_exclusions = pd.DataFrame(columns=["player_id", "exclude_from_hitter_tables"])
    return slate, rosters, hitters, pitchers, pitcher_summary_by_hand, arsenal, arsenal_by_hand, usage_by_count, batter_zone_profiles, pitcher_zone_profiles, hitter_pitcher_exclusions


@st.cache_data(show_spinner=False)
def _load_rolling_artifacts(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    hitter_rolling = pd.read_parquet(f"{base_url}/daily/{day}/daily_hitter_rolling.parquet")
    pitcher_rolling = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_rolling.parquet")
    return hitter_rolling, pitcher_rolling


@st.cache_data(show_spinner=False)
def _load_pitch_shape_artifacts(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    batter_family_zone_profiles = pd.read_parquet(f"{base_url}/daily/{day}/daily_batter_family_zone_profiles.parquet")
    pitcher_family_zone_context = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_family_zone_context.parquet")
    pitcher_movement_arsenal = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_movement_arsenal.parquet")
    return batter_family_zone_profiles, pitcher_family_zone_context, pitcher_movement_arsenal


def _sidebar() -> tuple[date, str, str, str, int, int, bool, str]:
    st.sidebar.title("Hosted Filters")
    target_date = st.sidebar.date_input("Slate date", value=date.today())
    split = st.sidebar.selectbox("Split", ["overall", "vs_rhp", "vs_lhp", "home", "away"])
    recent_window = st.sidebar.selectbox("Recent window", ["season", "last_45_days", "last_14_days"])
    weighted_mode = st.sidebar.radio("Weighting", ["weighted", "unweighted"], horizontal=True)
    min_pitch_count = st.sidebar.slider("Min pitches", 0, 3000, 0, 25)
    min_bip = st.sidebar.slider("Min BIP", 0, 500, 0, 5)
    likely_only = st.sidebar.checkbox("Likely starters only", value=False)
    preset_names = list(HITTER_PRESETS.keys())
    hitter_preset = st.sidebar.selectbox("Hitter view", preset_names, index=preset_names.index("All stats"))
    return target_date, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only, hitter_preset


def _load_artifacts_with_fallback(
    base_url: str,
    target_date: date,
    lookback_days: int = 7,
) -> tuple[date, tuple[pd.DataFrame, ...]]:
    last_error: Exception | None = None
    for offset in range(lookback_days + 1):
        candidate = target_date - timedelta(days=offset)
        try:
            return candidate, _load_core_artifacts(base_url, candidate)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No published hosted artifacts were found.")


def _game_selection(slate: list[dict]) -> list[dict]:
    if not slate:
        return []
    options = ["All games"] + [f"{game['away_team']} @ {game['home_team']}" for game in slate]
    selection = st.sidebar.selectbox("Game", options, index=0)
    if selection == "All games":
        return slate
    return [game for game in slate if f"{game['away_team']} @ {game['home_team']}" == selection]


def _filter_hitters(
    hitters: pd.DataFrame,
    rosters: pd.DataFrame,
    hitter_pitcher_exclusions: pd.DataFrame,
    rotowire_lineups: dict[str, dict[str, object]],
    team: str,
    opposing_hand: str | None,
    split: str,
    recent_window: str,
    weighted_mode: str,
    min_pitch_count: int,
    min_bip: int,
    likely_only: bool,
) -> pd.DataFrame:
    split_key = split
    if split == "overall" and opposing_hand == "R":
        split_key = "vs_rhp"
    elif split == "overall" and opposing_hand == "L":
        split_key = "vs_lhp"
    roster_lookup = rosters.loc[rosters["team"] == team, ["player_id"]].dropna().drop_duplicates()
    roster_player_ids = pd.to_numeric(roster_lookup["player_id"], errors="coerce").dropna().astype(int)
    frame = hitters.loc[
        (hitters["split_key"] == split_key)
        & (hitters["recent_window"] == recent_window)
        & (hitters["weighted_mode"] == weighted_mode)
    ].copy()
    if not roster_player_ids.empty and "batter" in frame.columns:
        frame = frame.loc[frame["batter"].isin(roster_player_ids)]
    frame["team"] = team
    if likely_only and "likely_starter_score" in frame:
        frame = frame.loc[frame["likely_starter_score"].fillna(0) > 0]
    frame = filter_excluded_pitchers_from_hitter_pool(apply_roster_names(frame, rosters, team), hitter_pitcher_exclusions)
    return apply_projected_lineup(frame, team, rotowire_lineups)


def _filter_pitcher_frame(
    frame: pd.DataFrame,
    split: str,
    recent_window: str,
    weighted_mode: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.copy()
    if "split_key" in filtered.columns:
        filtered = filtered.loc[filtered["split_key"] == split]
    if "recent_window" in filtered.columns:
        filtered = filtered.loc[filtered["recent_window"] == recent_window]
    if "weighted_mode" in filtered.columns:
        filtered = filtered.loc[filtered["weighted_mode"] == weighted_mode]
    return filtered


def _render_pitcher_tab(
    game_pk: int,
    team_label: str,
    pitcher_summary_by_hand: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    pitcher_by_hand: pd.DataFrame,
    pitcher_count_usage: pd.DataFrame,
    pitcher_row: pd.DataFrame,
    movement_arsenal: pd.DataFrame,
    family_context: pd.DataFrame,
    opposing_hitters: pd.DataFrame,
    batter_family_zone_profiles: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    mobile_safe: bool,
) -> list[dict]:
    export_sections: list[dict] = []
    tab_summary, tab_arsenal, tab_count, tab_shape = st.tabs(["Summary", "Arsenal", "Count Usage", "Pitch Shape"])

    with tab_summary:
        summary_table = build_pitcher_summary_table(pitcher_summary_by_hand)
        _render_hosted_grid(
            summary_table,
            key=f"summary-{game_pk}-{team_label}",
            mobile_safe=mobile_safe,
            height=132,
            lower_is_better=PITCHER_LOWER_IS_BETTER,
            higher_is_better=PITCHER_HIGHER_IS_BETTER,
        )
        if "batter_side_key" in pitcher_summary_by_hand.columns:
            for side_key in BATTER_SIDE_LABELS:
                summary_frame = pitcher_summary_by_hand.loc[pitcher_summary_by_hand["batter_side_key"] == side_key]
                if not summary_frame.empty:
                    summary_columns = _present_columns(summary_frame, PITCHER_SUMMARY_COLUMNS)
                    export_sections.append(
                        {
                            "title": f"{team_label} Summary {BATTER_SIDE_LABELS[side_key]}",
                            "frame": summary_frame[summary_columns],
                            "lower_is_better": PITCHER_LOWER_IS_BETTER,
                            "higher_is_better": PITCHER_HIGHER_IS_BETTER,
                        }
                    )

    with tab_arsenal:
        arsenal_tabs = st.tabs([BATTER_SIDE_LABELS[key] for key in BATTER_SIDE_LABELS])
        for side_key, side_tab in zip(BATTER_SIDE_LABELS, arsenal_tabs):
            with side_tab:
                if side_key == "all":
                    side_frame = sort_arsenal_frame(pitcher_arsenal)
                elif "batter_side_key" in pitcher_by_hand.columns:
                    side_frame = sort_arsenal_frame(pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key])
                else:
                    side_frame = _empty_like(pitcher_by_hand)
                if side_frame.empty:
                    st.info("No arsenal data available.")
                else:
                    arsenal_columns = _present_columns(side_frame, ARSENAL_COLUMNS)
                    arsenal_grid = _render_hosted_grid(
                        side_frame[arsenal_columns],
                        key=f"arsenal-{game_pk}-{team_label}-{side_key}",
                        mobile_safe=mobile_safe,
                        height=250,
                        lower_is_better={"hard_hit_pct", "xwoba_con"},
                        higher_is_better={"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                    )
                    export_sections.append(
                        {
                            "title": f"{team_label} Arsenal {BATTER_SIDE_LABELS[side_key]}",
                            "frame": arsenal_grid[_present_columns(arsenal_grid, ARSENAL_COLUMNS)],
                            "lower_is_better": {"hard_hit_pct", "xwoba_con"},
                            "higher_is_better": {"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                        }
                    )

    with tab_count:
        sub_tabs = st.tabs([BATTER_SIDE_LABELS[key] for key in BATTER_SIDE_LABELS])
        for side_key, side_tab in zip(BATTER_SIDE_LABELS, sub_tabs):
            with side_tab:
                if "batter_side_key" in pitcher_count_usage.columns:
                    side_count = pitcher_count_usage.loc[pitcher_count_usage["batter_side_key"] == side_key]
                else:
                    side_count = _empty_like(pitcher_count_usage)
                if side_key == "all":
                    side_arsenal = pitcher_arsenal[["pitch_name", "usage_pct"]] if not pitcher_arsenal.empty else pd.DataFrame(columns=["pitch_name", "usage_pct"])
                elif "batter_side_key" in pitcher_by_hand.columns:
                    side_arsenal = pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key, ["pitch_name", "usage_pct"]]
                else:
                    side_arsenal = pd.DataFrame(columns=["pitch_name", "usage_pct"])
                count_frame = pivot_count_usage(side_count, side_arsenal)
                if count_frame.empty:
                    st.info("No count-state usage data available.")
                else:
                    count_columns = _present_columns(count_frame, COUNT_USAGE_COLUMNS)
                    count_grid = _render_hosted_grid(
                        count_frame[count_columns],
                        key=f"count-{game_pk}-{team_label}-{side_key}",
                        mobile_safe=mobile_safe,
                        height=250,
                        higher_is_better=set(COUNT_BUCKET_ORDER),
                    )
                    export_sections.append(
                        {
                            "title": f"{team_label} Count Usage {BATTER_SIDE_LABELS[side_key]}",
                            "frame": count_grid[_present_columns(count_grid, COUNT_USAGE_COLUMNS)],
                            "higher_is_better": set(COUNT_BUCKET_ORDER),
                        }
                    )

    with tab_shape:
        _render_pitch_shape_context(
            game_pk,
            team_label,
            pitcher_row,
            movement_arsenal,
            family_context,
            opposing_hitters,
            batter_family_zone_profiles,
            pitcher_family_zone_context,
            mobile_safe,
        )

    return export_sections


def _build_pitcher_export_sections(
    team_label: str,
    pitcher_summary_by_hand: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    pitcher_by_hand: pd.DataFrame,
    pitcher_count_usage: pd.DataFrame,
) -> list[dict]:
    export_sections: list[dict] = []
    for side_key, side_label in BATTER_SIDE_LABELS.items():
        if "batter_side_key" in pitcher_summary_by_hand.columns:
            summary_frame = pitcher_summary_by_hand.loc[pitcher_summary_by_hand["batter_side_key"] == side_key]
            if not summary_frame.empty:
                summary_columns = _present_columns(summary_frame, PITCHER_SUMMARY_COLUMNS)
                export_sections.append(
                    {
                        "title": f"{team_label} Summary {side_label}",
                        "frame": summary_frame[summary_columns],
                        "lower_is_better": PITCHER_LOWER_IS_BETTER,
                        "higher_is_better": PITCHER_HIGHER_IS_BETTER,
                    }
                )

        if side_key == "all":
            side_arsenal = sort_arsenal_frame(pitcher_arsenal)
        elif "batter_side_key" in pitcher_by_hand.columns:
            side_arsenal = sort_arsenal_frame(pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key])
        else:
            side_arsenal = _empty_like(pitcher_by_hand)
        if not side_arsenal.empty:
            arsenal_columns = _present_columns(side_arsenal, ARSENAL_COLUMNS)
            export_sections.append(
                {
                    "title": f"{team_label} Arsenal {side_label}",
                    "frame": side_arsenal[arsenal_columns],
                    "lower_is_better": {"hard_hit_pct", "xwoba_con"},
                    "higher_is_better": {"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                }
            )

        if "batter_side_key" in pitcher_count_usage.columns:
            side_count = pitcher_count_usage.loc[pitcher_count_usage["batter_side_key"] == side_key]
        else:
            side_count = _empty_like(pitcher_count_usage)
        if side_key == "all":
            side_usage = pitcher_arsenal[["pitch_name", "usage_pct"]] if not pitcher_arsenal.empty else pd.DataFrame(columns=["pitch_name", "usage_pct"])
        elif "batter_side_key" in pitcher_by_hand.columns:
            side_usage = pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key, ["pitch_name", "usage_pct"]]
        else:
            side_usage = pd.DataFrame(columns=["pitch_name", "usage_pct"])
        count_frame = pivot_count_usage(side_count, side_usage)
        if not count_frame.empty:
            count_columns = _present_columns(count_frame, COUNT_USAGE_COLUMNS)
            export_sections.append(
                {
                    "title": f"{team_label} Count Usage {side_label}",
                    "frame": count_frame[count_columns],
                    "higher_is_better": set(COUNT_BUCKET_ORDER),
                }
            )
    return export_sections


def main() -> None:
    st.set_page_config(page_title="MLB Hosted Slate Companion", page_icon=page_icon_path(), layout="wide")
    st.title("MLB Hosted Slate Companion")
    perf_events: list[tuple[str, float]] = []
    base_url = _base_url()
    if not base_url:
        st.error("Set MLB_HOSTED_BASE_URL to your Hugging Face dataset file base URL before running this app.")
        return

    target_date, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only, hitter_preset = _sidebar()
    mobile_safe = True
    try:
        load_start = perf_counter()
        resolved_date, artifacts = _load_artifacts_with_fallback(base_url, target_date)
        (
            slate,
            rosters,
            hitters,
            pitchers,
            pitcher_summary_by_hand,
            arsenal,
            arsenal_by_hand,
            usage_by_count,
            batter_zone_profiles,
            pitcher_zone_profiles,
            hitter_pitcher_exclusions,
        ) = artifacts
        _record_perf(perf_events, "core load", load_start)
    except Exception as exc:  # pragma: no cover
        st.error(f"Unable to load hosted artifacts for {target_date.isoformat()}: {exc}")
        return
    if resolved_date != target_date:
        st.warning(
            f"No published artifacts were found for {target_date.isoformat()}. "
            f"Showing the latest available published slate from {resolved_date.isoformat()} instead."
        )

    all_games = slate.to_dict(orient="records")
    selected_games = _game_selection(all_games)
    active_sections = {st.session_state.get(f"section-{game['game_pk']}", "Matchup") for game in selected_games}
    st.caption(f"Showing {len(selected_games)} of {len(all_games)} games")
    st.caption("PulledBrl% tracks pulled barrels on tracked batted-ball events. Brl/BIP% uses all balls in play.")
    batter_join_col = "batter_id" if "batter_id" in batter_zone_profiles.columns else "batter"
    pitcher_join_col = "pitcher_id" if "pitcher_id" in pitcher_zone_profiles.columns else "pitcher"
    valid_teams = tuple(sorted({game["away_team"] for game in all_games} | {game["home_team"] for game in all_games}))
    try:
        rotowire_start = perf_counter()
        rotowire_lineups = resolve_rotowire_lineups(fetch_rotowire_lineups(resolved_date, valid_teams), rosters)
        _record_perf(perf_events, "rotowire", rotowire_start)
    except Exception:
        rotowire_lineups = {}

    hitter_rolling = pd.DataFrame()
    pitcher_rolling = pd.DataFrame()
    batter_family_zone_profiles = pd.DataFrame()
    pitcher_family_zone_context = pd.DataFrame()
    pitcher_movement_arsenal = pd.DataFrame()
    if "Rolling" in active_sections:
        rolling_start = perf_counter()
        hitter_rolling, pitcher_rolling = _load_rolling_artifacts(base_url, resolved_date)
        _record_perf(perf_events, "rolling load", rolling_start)
    if "Matchup" in active_sections:
        shape_start = perf_counter()
        (
            batter_family_zone_profiles,
            pitcher_family_zone_context,
            pitcher_movement_arsenal,
        ) = _load_pitch_shape_artifacts(base_url, resolved_date)
        _record_perf(perf_events, "pitch-shape load", shape_start)

    batter_zone_named = pd.DataFrame()
    if {"Pitcher Zones", "Hitter Zones"} & active_sections:
        zone_named_start = perf_counter()
        roster_lookup = rosters[["team", "player_id", "player_name"]].drop_duplicates("player_id")
        batter_zone_named = batter_zone_profiles.merge(roster_lookup, left_on=batter_join_col, right_on="player_id", how="left")
        _record_perf(perf_events, "zone merge", zone_named_start)

    hitters_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitchers_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    best_matchups_by_game: dict[int, pd.DataFrame] = {}
    pitcher_summary_by_hand_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitcher_arsenal_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitcher_by_hand_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitcher_count_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    opponent_hitters_by_key: dict[tuple[object, object, object, object, object], pd.DataFrame] = {}
    filtered_pitchers = _filter_pitcher_frame(pitchers, split, recent_window, weighted_mode)
    filtered_summary = _filter_pitcher_frame(pitcher_summary_by_hand, split, recent_window, weighted_mode)
    filtered_arsenal = _filter_pitcher_frame(arsenal, split, recent_window, weighted_mode)
    filtered_arsenal_by_hand = _filter_pitcher_frame(arsenal_by_hand, split, recent_window, weighted_mode)
    filtered_count = _filter_pitcher_frame(usage_by_count, split, recent_window, weighted_mode)
    filtered_family_context = _filter_pitcher_frame(pitcher_family_zone_context, split, recent_window, weighted_mode)
    filtered_movement_arsenal = _filter_pitcher_frame(pitcher_movement_arsenal, split, recent_window, weighted_mode)
    summary_by_pitcher = _frame_by_pitcher_id(filtered_summary)
    arsenal_by_pitcher = _frame_by_pitcher_id(filtered_arsenal)
    arsenal_by_hand_by_pitcher = _frame_by_pitcher_id(filtered_arsenal_by_hand)
    count_by_pitcher = _frame_by_pitcher_id(filtered_count)
    family_context_by_pitcher = _frame_by_pitcher_id(filtered_family_context)
    movement_arsenal_by_pitcher = _frame_by_pitcher_id(filtered_movement_arsenal)

    game_loop_start = perf_counter()
    for game in selected_games:
        away_pitcher = filtered_pitchers.loc[filtered_pitchers["pitcher_id"] == game.get("away_probable_pitcher_id")].copy()
        home_pitcher = filtered_pitchers.loc[filtered_pitchers["pitcher_id"] == game.get("home_probable_pitcher_id")].copy()
        away_hand = home_pitcher["p_throws"].iloc[0] if not home_pitcher.empty else None
        home_hand = away_pitcher["p_throws"].iloc[0] if not away_pitcher.empty else None

        away_hitters = _filter_hitters(hitters, rosters, hitter_pitcher_exclusions, rotowire_lineups, game["away_team"], away_hand, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only)
        home_hitters = _filter_hitters(hitters, rosters, hitter_pitcher_exclusions, rotowire_lineups, game["home_team"], home_hand, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only)
        hitters_by_game[game["game_pk"]] = (
            add_hitter_matchup_score(
                away_hitters,
                batter_zone_profiles=batter_zone_profiles,
                pitcher_zone_profiles=pitcher_zone_profiles,
                opposing_pitcher_id=game.get("home_probable_pitcher_id"),
                opposing_pitcher_hand=away_hand,
            ),
            add_hitter_matchup_score(
                home_hitters,
                batter_zone_profiles=batter_zone_profiles,
                pitcher_zone_profiles=pitcher_zone_profiles,
                opposing_pitcher_id=game.get("away_probable_pitcher_id"),
                opposing_pitcher_hand=home_hand,
            ),
        )
        away_scored_hitters, home_scored_hitters = hitters_by_game[game["game_pk"]]
        if game.get("away_probable_pitcher_id"):
            opponent_hitters_by_key[
                build_pitcher_matchup_key(game["game_pk"], game.get("away_probable_pitcher_id"), split, recent_window, weighted_mode)
            ] = home_scored_hitters
        if game.get("home_probable_pitcher_id"):
            opponent_hitters_by_key[
                build_pitcher_matchup_key(game["game_pk"], game.get("home_probable_pitcher_id"), split, recent_window, weighted_mode)
            ] = away_scored_hitters
        best_matchups_by_game[game["game_pk"]] = build_best_matchups(*hitters_by_game[game["game_pk"]])
        pitcher_summary_by_hand_map[game["game_pk"]] = (
            summary_by_pitcher.get(game.get("away_probable_pitcher_id"), filtered_summary.head(0)).copy(),
            summary_by_pitcher.get(game.get("home_probable_pitcher_id"), filtered_summary.head(0)).copy(),
        )
        pitcher_arsenal_map[game["game_pk"]] = (
            arsenal_by_pitcher.get(game.get("away_probable_pitcher_id"), filtered_arsenal.head(0)).copy(),
            arsenal_by_pitcher.get(game.get("home_probable_pitcher_id"), filtered_arsenal.head(0)).copy(),
        )
        pitcher_by_hand_map[game["game_pk"]] = (
            arsenal_by_hand_by_pitcher.get(game.get("away_probable_pitcher_id"), filtered_arsenal_by_hand.head(0)).copy(),
            arsenal_by_hand_by_pitcher.get(game.get("home_probable_pitcher_id"), filtered_arsenal_by_hand.head(0)).copy(),
        )
        pitcher_count_map[game["game_pk"]] = (
            count_by_pitcher.get(game.get("away_probable_pitcher_id"), filtered_count.head(0)).copy(),
            count_by_pitcher.get(game.get("home_probable_pitcher_id"), filtered_count.head(0)).copy(),
        )
    slate_pitcher_ids = {
        pitcher_id
        for game in selected_games
        for pitcher_id in [game.get("away_probable_pitcher_id"), game.get("home_probable_pitcher_id")]
        if pitcher_id
    }
    scored_slate_pitchers = (
        add_pitcher_rank_score(
            filtered_pitchers.loc[filtered_pitchers["pitcher_id"].isin(slate_pitcher_ids)].copy(),
            opponent_hitters_by_key=opponent_hitters_by_key or None,
            batter_family_zone_profiles=batter_family_zone_profiles,
            pitcher_family_zone_context=pitcher_family_zone_context,
        )
        if slate_pitcher_ids
        else filtered_pitchers.head(0).copy()
    )
    scored_pitchers_by_id = _frame_by_pitcher_id(scored_slate_pitchers)
    for game in selected_games:
        pitchers_by_game[game["game_pk"]] = (
            scored_pitchers_by_id.get(game.get("away_probable_pitcher_id"), filtered_pitchers.head(0)).copy(),
            scored_pitchers_by_id.get(game.get("home_probable_pitcher_id"), filtered_pitchers.head(0)).copy(),
        )
    _record_perf(perf_events, "game prep", game_loop_start)

    hitter_rows: list[pd.DataFrame] = []
    pitcher_rows: list[pd.DataFrame] = []
    for game in selected_games:
        label = f"{game['away_team']} @ {game['home_team']}"
        away_hitters, home_hitters = hitters_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_pitcher, home_pitcher = pitchers_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        if not away_hitters.empty:
            hitter_rows.append(with_game_label(away_hitters, label))
        if not home_hitters.empty:
            hitter_rows.append(with_game_label(home_hitters, label))
        if not away_pitcher.empty:
            pitcher_rows.append(with_game_label(away_pitcher, label))
        if not home_pitcher.empty:
            pitcher_rows.append(with_game_label(home_pitcher, label))

    export_options_by_game: dict[int, dict[str, list[dict]]] = {}
    for game in selected_games:
        away_hitters, home_hitters = hitters_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_summary_by_hand, home_summary_by_hand = pitcher_summary_by_hand_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_arsenal, home_arsenal = pitcher_arsenal_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_by_hand, home_by_hand = pitcher_by_hand_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_count, home_count = pitcher_count_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_export_sections = _build_pitcher_export_sections(
            game["away_team"],
            away_summary_by_hand,
            away_arsenal,
            away_by_hand,
            away_count,
        )
        home_export_sections = _build_pitcher_export_sections(
            game["home_team"],
            home_summary_by_hand,
            home_arsenal,
            home_by_hand,
            home_count,
        )
        export_options_by_game[game["game_pk"]] = build_game_export_options(
            game_title=f"{game['away_team']} @ {game['home_team']}",
            away_team=game["away_team"],
            home_team=game["home_team"],
            best_matchups=best_matchups_by_game.get(game["game_pk"], pd.DataFrame()).copy(),
            away_sections=away_export_sections,
            home_sections=home_export_sections,
            away_hitters=away_hitters,
            home_hitters=home_hitters,
        )
    full_slate_export_bundles = build_full_slate_export_bundles(selected_games, export_options_by_game)

    st.header("Top Slate Hitters")
    top_hitters_start = perf_counter()
    all_hitters = pd.concat(hitter_rows, ignore_index=True, sort=False) if hitter_rows else pd.DataFrame()
    all_pitchers = pd.concat(pitcher_rows, ignore_index=True, sort=False) if pitcher_rows else pd.DataFrame()
    ranked_pitchers = all_pitchers.sort_values(["pitcher_score", "xwoba"], ascending=[False, True], na_position="last") if not all_pitchers.empty else pd.DataFrame()
    if all_hitters.empty:
        st.info("No hitter data available for this slate.")
    else:
        preset_columns = hitter_columns_for_preset(hitter_preset)
        ranked_hitters = all_hitters.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last")
        export_options = build_slate_export_options(
            ranked_hitters,
            preset_columns,
            ranked_pitchers,
        )
        render_slate_export_controls(
            "top-matchups-export-hosted",
            "Top Slate Export",
            export_options,
            full_slate_export_bundles,
        )
        _render_hosted_grid(
            ranked_hitters[["game"] + [column for column in preset_columns if column in all_hitters.columns]].head(10),
            key="top-slate-hitters-hosted",
            mobile_safe=mobile_safe,
            height=320,
        )
    _record_perf(perf_events, "top hitters", top_hitters_start)

    st.header("Top Slate Pitchers")
    top_pitchers_start = perf_counter()
    if all_pitchers.empty:
        st.info("No pitcher data available for this slate.")
    else:
        pitcher_columns = _present_columns(ranked_pitchers, TOP_PITCHER_COLUMNS)
        if pitcher_columns:
            _render_hosted_grid(
                ranked_pitchers[pitcher_columns].head(10),
                key="top-slate-pitchers-hosted",
                mobile_safe=mobile_safe,
                height=320,
                lower_is_better=PITCHER_LOWER_IS_BETTER,
                higher_is_better=PITCHER_HIGHER_IS_BETTER,
            )
        else:
            st.info("No pitcher table columns available for this slate.")
    _record_perf(perf_events, "top pitchers", top_pitchers_start)
    _render_perf(perf_events)

    st.divider()
    hitter_columns = hitter_columns_for_preset(hitter_preset)

    for idx, game in enumerate(selected_games):
        away_hitters, home_hitters = hitters_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_pitcher, home_pitcher = pitchers_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_pitcher_id = game.get("away_probable_pitcher_id")
        home_pitcher_id = game.get("home_probable_pitcher_id")
        away_summary_by_hand = summary_by_pitcher.get(away_pitcher_id, filtered_summary.head(0)).copy()
        home_summary_by_hand = summary_by_pitcher.get(home_pitcher_id, filtered_summary.head(0)).copy()
        away_arsenal = arsenal_by_pitcher.get(away_pitcher_id, filtered_arsenal.head(0)).copy()
        home_arsenal = arsenal_by_pitcher.get(home_pitcher_id, filtered_arsenal.head(0)).copy()
        away_by_hand = arsenal_by_hand_by_pitcher.get(away_pitcher_id, filtered_arsenal_by_hand.head(0)).copy()
        home_by_hand = arsenal_by_hand_by_pitcher.get(home_pitcher_id, filtered_arsenal_by_hand.head(0)).copy()
        away_count = count_by_pitcher.get(away_pitcher_id, filtered_count.head(0)).copy()
        home_count = count_by_pitcher.get(home_pitcher_id, filtered_count.head(0)).copy()
        away_family_context = family_context_by_pitcher.get(away_pitcher_id, filtered_family_context.head(0)).copy()
        home_family_context = family_context_by_pitcher.get(home_pitcher_id, filtered_family_context.head(0)).copy()
        away_movement = movement_arsenal_by_pitcher.get(away_pitcher_id, filtered_movement_arsenal.head(0)).copy()
        home_movement = movement_arsenal_by_pitcher.get(home_pitcher_id, filtered_movement_arsenal.head(0)).copy()
        best_matchups = best_matchups_by_game.get(game["game_pk"], pd.DataFrame()).copy()

        with st.expander(f"{game['away_team']} @ {game['home_team']}", expanded=idx == 0):
            render_matchup_header(game)
            active_section = st.radio(
                "Section",
                ["Matchup", "Rolling", "Pitcher Zones", "Hitter Zones", "Exports"],
                horizontal=True,
                key=f"section-{game['game_pk']}",
                label_visibility="collapsed",
            )
            if active_section == "Matchup":
                st.markdown("#### Best Matchups")
                matchup_columns = _present_columns(best_matchups, BEST_MATCHUP_COLUMNS)
                if matchup_columns:
                    best_matchups = _render_hosted_grid(
                        best_matchups[matchup_columns],
                        key=f"best-hosted-{game['game_pk']}",
                        mobile_safe=mobile_safe,
                        height=170,
                    )
                else:
                    st.info("No matchup rows available for this game.")

                st.markdown("#### Pitchers")
                pitcher_cols = st.columns(2)
                with pitcher_cols[0]:
                    st.markdown(f"##### {game['away_team']} starter")
                    away_export_sections = _render_pitcher_tab(
                        game["game_pk"],
                        game["away_team"],
                        away_summary_by_hand,
                        away_arsenal,
                        away_by_hand,
                        away_count,
                        away_pitcher,
                        away_movement,
                        away_family_context,
                        home_hitters,
                        batter_family_zone_profiles,
                        filtered_family_context,
                        mobile_safe,
                    )
                with pitcher_cols[1]:
                    st.markdown(f"##### {game['home_team']} starter")
                    home_export_sections = _render_pitcher_tab(
                        game["game_pk"],
                        game["home_team"],
                        home_summary_by_hand,
                        home_arsenal,
                        home_by_hand,
                        home_count,
                        home_pitcher,
                        home_movement,
                        home_family_context,
                        away_hitters,
                        batter_family_zone_profiles,
                        filtered_family_context,
                        mobile_safe,
                    )

                st.markdown("#### Hitters")
                hitter_cols = st.columns(2)
                with hitter_cols[0]:
                    st.caption(f"{game['away_team']} vs {game.get('home_probable_pitcher_name') or 'opposing starter'}")
                    away_hitters = _render_hosted_grid(
                        away_hitters[[column for column in hitter_columns if column in away_hitters.columns]],
                        key=f"away-hitters-hosted-{game['game_pk']}",
                        mobile_safe=mobile_safe,
                        height=360,
                    )
                with hitter_cols[1]:
                    st.caption(f"{game['home_team']} vs {game.get('away_probable_pitcher_name') or 'opposing starter'}")
                    home_hitters = _render_hosted_grid(
                        home_hitters[[column for column in hitter_columns if column in home_hitters.columns]],
                        key=f"home-hitters-hosted-{game['game_pk']}",
                        mobile_safe=mobile_safe,
                        height=360,
                    )

            elif active_section == "Rolling":
                roll_tabs = st.tabs(["Rolling 5", "Rolling 10", "Rolling 15"])
                away_hitter_names = set(away_hitters.get("hitter_name", pd.Series(dtype="object")).dropna().tolist())
                home_hitter_names = set(home_hitters.get("hitter_name", pd.Series(dtype="object")).dropna().tolist())
                away_pitcher_name = away_pitcher["pitcher_name"].iloc[0] if not away_pitcher.empty else None
                home_pitcher_name = home_pitcher["pitcher_name"].iloc[0] if not home_pitcher.empty else None
                for label, tab in zip(["Rolling 5", "Rolling 10", "Rolling 15"], roll_tabs):
                    with tab:
                        cols = st.columns(2)
                        with cols[0]:
                            hitter_frame = hitter_rolling.loc[
                                hitter_rolling["rolling_window"].eq(label)
                                & hitter_rolling["player_name"].isin(sorted(away_hitter_names | home_hitter_names))
                            ]
                            _render_hosted_grid(
                                hitter_frame[[column for column in HITTER_ROLLING_COLUMNS if column in hitter_frame.columns]],
                                key=f"h-roll-hosted-{game['game_pk']}-{label}",
                                mobile_safe=mobile_safe,
                                height=260,
                            )
                        with cols[1]:
                            pitcher_frame = pitcher_rolling.loc[
                                pitcher_rolling["rolling_window"].eq(label)
                                & pitcher_rolling["player_name"].isin([name for name in [away_pitcher_name, home_pitcher_name] if name])
                            ]
                            _render_hosted_grid(
                                pitcher_frame[[column for column in PITCHER_ROLLING_COLUMNS if column in pitcher_frame.columns]],
                                key=f"p-roll-hosted-{game['game_pk']}-{label}",
                                mobile_safe=mobile_safe,
                                height=260,
                                lower_is_better=PITCHER_LOWER_IS_BETTER | {"barrel_bip_pct"},
                                higher_is_better=PITCHER_HIGHER_IS_BETTER | {"avg_release_speed"},
                            )

            elif active_section == "Pitcher Zones":
                zone_tabs = st.tabs([game["away_team"], game["home_team"]])
                for pitcher_row, tab, team_label in [
                    (away_pitcher, zone_tabs[0], game["away_team"]),
                    (home_pitcher, zone_tabs[1], game["home_team"]),
                ]:
                    with tab:
                        if pitcher_row.empty:
                            st.info("No pitcher zone data available.")
                        else:
                            pitcher_id = pitcher_row["pitcher_id"].iloc[0]
                            zone_frame = (
                                pitcher_zone_profiles.loc[pitcher_zone_profiles[pitcher_join_col] == pitcher_id]
                                .sort_values("sample_size", ascending=False, na_position="last")
                                .head(200)
                            )
                            opposing_hitters = home_hitters if team_label == game["away_team"] else away_hitters
                            opponent_names = sorted(opposing_hitters.get("hitter_name", pd.Series(dtype="object")).dropna().unique().tolist())
                            selected_hitter = (
                                st.selectbox("Overlay hitter", opponent_names, key=f"p-zone-hosted-hitter-{game['game_pk']}-{team_label}")
                                if opponent_names
                                else None
                            )
                            hitter_detail = batter_zone_named.loc[batter_zone_named["player_name"] == selected_hitter].copy() if selected_hitter else pd.DataFrame()
                            pitch_types = ["All pitches"] + sorted(
                                set(zone_frame.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                                | set(hitter_detail.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                            )
                            selected_pitch = st.selectbox("Pitch type", pitch_types, key=f"p-zone-hosted-pitch-{game['game_pk']}-{team_label}")
                            pitcher_map = aggregate_pitcher_zone_map(zone_frame, selected_pitch)
                            hitter_map = aggregate_batter_zone_map(hitter_detail, selected_pitch)
                            overlay_map = build_zone_overlay_map(hitter_map, pitcher_map)
                            heatmap_cols = st.columns(2)
                            with heatmap_cols[0]:
                                render_zone_heatmap(
                                    f"{pitcher_row['pitcher_name'].iloc[0]} Usage",
                                    f"{selected_pitch} | Pitcher zone attack",
                                    pitcher_map,
                                )
                            with heatmap_cols[1]:
                                render_zone_heatmap(
                                    f"Overlay vs {selected_hitter or 'Opposing Hitter'}",
                                    f"{selected_pitch} | Hitter damage x pitcher usage",
                                    overlay_map,
                                )
                            _render_hosted_grid(
                                zone_frame[[column for column in PITCHER_ZONE_COLUMNS if column in zone_frame.columns]],
                                key=f"p-zone-hosted-{game['game_pk']}-{team_label}",
                                mobile_safe=mobile_safe,
                                height=240,
                                higher_is_better={"usage_rate"},
                            )

            elif active_section == "Hitter Zones":
                zone_tabs = st.tabs([game["away_team"], game["home_team"]])
                for team_label, tab in [(game["away_team"], zone_tabs[0]), (game["home_team"], zone_tabs[1])]:
                    with tab:
                        zone_frame = (
                            batter_zone_named.loc[batter_zone_named["team"] == team_label]
                            .sort_values("sample_size", ascending=False, na_position="last")
                            .head(250)
                        )
                        hitter_options = sorted(zone_frame.get("player_name", pd.Series(dtype="object")).dropna().unique().tolist())
                        selected_hitter = (
                            st.selectbox("Hitter", hitter_options, key=f"h-zone-hosted-hitter-{game['game_pk']}-{team_label}")
                            if hitter_options
                            else None
                        )
                        hitter_detail = zone_frame.loc[zone_frame["player_name"] == selected_hitter].copy() if selected_hitter else pd.DataFrame()
                        hitter_detail["damage_rate"] = (
                            pd.to_numeric(hitter_detail.get("hit_rate"), errors="coerce").fillna(0) * 0.6
                            + pd.to_numeric(hitter_detail.get("hr_rate"), errors="coerce").fillna(0) * 0.4
                        )
                        opposing_pitcher = home_pitcher if team_label == game["away_team"] else away_pitcher
                        pitcher_detail = (
                            pitcher_zone_profiles.loc[pitcher_zone_profiles[pitcher_join_col] == opposing_pitcher["pitcher_id"].iloc[0]].copy()
                            if not opposing_pitcher.empty
                            else pd.DataFrame()
                        )
                        pitch_types = ["All pitches"] + sorted(
                            set(hitter_detail.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                            | set(pitcher_detail.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                        )
                        selected_pitch = st.selectbox("Pitch type", pitch_types, key=f"h-zone-hosted-pitch-{game['game_pk']}-{team_label}")
                        hitter_map = aggregate_batter_zone_map(hitter_detail, selected_pitch)
                        pitcher_map = aggregate_pitcher_zone_map(pitcher_detail, selected_pitch)
                        overlay_map = build_zone_overlay_map(hitter_map, pitcher_map)
                        heatmap_cols = st.columns(2)
                        with heatmap_cols[0]:
                            render_zone_heatmap(
                                f"{selected_hitter or team_label} Damage",
                                f"{selected_pitch} | Hitter zone quality",
                                hitter_map,
                            )
                        with heatmap_cols[1]:
                            opposing_name = opposing_pitcher["pitcher_name"].iloc[0] if not opposing_pitcher.empty else "Opposing Pitcher"
                            render_zone_heatmap(
                                f"Overlay vs {opposing_name}",
                                f"{selected_pitch} | Hitter damage x pitcher usage",
                                overlay_map,
                            )
                        _render_hosted_grid(
                            hitter_detail[[column for column in BATTER_ZONE_COLUMNS if column in hitter_detail.columns]],
                            key=f"h-zone-hosted-{game['game_pk']}-{team_label}",
                            mobile_safe=mobile_safe,
                            height=240,
                            higher_is_better={"hit_rate", "hr_rate", "damage_rate"},
                        )

            elif active_section == "Exports":
                away_export_sections = _build_pitcher_export_sections(
                    game["away_team"],
                    away_summary_by_hand,
                    away_arsenal,
                    away_by_hand,
                    away_count,
                )
                home_export_sections = _build_pitcher_export_sections(
                    game["home_team"],
                    home_summary_by_hand,
                    home_arsenal,
                    home_by_hand,
                    home_count,
                )
                export_options = export_options_by_game.get(game["game_pk"])
                if export_options is None:
                    export_options = build_game_export_options(
                        game_title=f"{game['away_team']} @ {game['home_team']}",
                        away_team=game["away_team"],
                        home_team=game["home_team"],
                        best_matchups=best_matchups,
                        away_sections=away_export_sections,
                        home_sections=home_export_sections,
                        away_hitters=away_hitters,
                        home_hitters=home_hitters,
                    )
                render_export_hub(
                    key=f"export-hosted-{game['game_pk']}",
                    title=f"{game['away_team']} @ {game['home_team']}",
                    export_options=export_options,
                )
        st.divider()
