from __future__ import annotations

import os
from datetime import date

import pandas as pd
import streamlit as st

from .dashboard_views import (
    ARSENAL_COLUMNS,
    BATTER_SIDE_LABELS,
    BATTER_ZONE_COLUMNS,
    BEST_MATCHUP_COLUMNS,
    COUNT_BUCKET_ORDER,
    COUNT_USAGE_COLUMNS,
    HITTER_PRESETS,
    HITTER_ROLLING_COLUMNS,
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
    build_zone_overlay_map,
    build_best_matchups,
    build_game_export_options,
    hitter_columns_for_preset,
    pivot_count_usage,
    with_game_label,
)
from .query_engine import load_remote_parquet
from .ui_components import render_export_hub, render_matchup_header, render_metric_grid, render_pitcher_summary_strip, render_zone_heatmap


def _base_url() -> str:
    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


@st.cache_data(show_spinner=False)
def _load_artifacts(
    base_url: str,
    target_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    slate = pd.read_parquet(f"{base_url}/daily/{day}/slate.parquet")
    rosters = pd.read_parquet(f"{base_url}/daily/{day}/rosters.parquet")
    hitters = pd.read_parquet(f"{base_url}/daily/{day}/daily_hitter_metrics.parquet")
    pitchers = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_metrics.parquet")
    pitcher_summary_by_hand = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_summary_by_hand.parquet")
    arsenal = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_arsenal.parquet")
    arsenal_by_hand = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_arsenal_by_hand.parquet")
    usage_by_count = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_usage_by_count.parquet")
    hitter_rolling = pd.read_parquet(f"{base_url}/daily/{day}/daily_hitter_rolling.parquet")
    pitcher_rolling = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_rolling.parquet")
    batter_zone_profiles = pd.read_parquet(f"{base_url}/daily/{day}/daily_batter_zone_profiles.parquet")
    pitcher_zone_profiles = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_zone_profiles.parquet")
    return slate, rosters, hitters, pitchers, pitcher_summary_by_hand, arsenal, arsenal_by_hand, usage_by_count, hitter_rolling, pitcher_rolling, batter_zone_profiles, pitcher_zone_profiles


def _sidebar() -> tuple[date, str, str, str, int, int, bool, str]:
    st.sidebar.title("Hosted Filters")
    target_date = st.sidebar.date_input("Slate date", value=date.today())
    split = st.sidebar.selectbox("Split", ["overall", "vs_rhp", "vs_lhp", "home", "away"])
    recent_window = st.sidebar.selectbox("Recent window", ["season", "last_45_days", "last_14_days"])
    weighted_mode = st.sidebar.radio("Weighting", ["weighted", "unweighted"], horizontal=True)
    min_pitch_count = st.sidebar.slider("Min pitches", 0, 3000, 100, 25)
    min_bip = st.sidebar.slider("Min BIP", 0, 500, 25, 5)
    likely_only = st.sidebar.checkbox("Likely starters only", value=False)
    preset_names = list(HITTER_PRESETS.keys())
    hitter_preset = st.sidebar.selectbox("Hitter view", preset_names, index=preset_names.index("All stats"))
    return target_date, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only, hitter_preset


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
    frame = hitters.loc[
        (hitters["team"] == team)
        & (hitters["split_key"] == split_key)
        & (hitters["recent_window"] == recent_window)
        & (hitters["weighted_mode"] == weighted_mode)
        & (hitters["pitch_count"] >= min_pitch_count)
        & (hitters["bip"] >= min_bip)
    ].copy()
    if likely_only and "likely_starter_score" in frame:
        frame = frame.loc[frame["likely_starter_score"].fillna(0) > 0]
    return apply_roster_names(frame, rosters, team)


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
) -> list[dict]:
    export_sections: list[dict] = []
    tab_summary, tab_arsenal, tab_count = st.tabs(["Summary", "Arsenal", "Count Usage"])

    with tab_summary:
        summary_tabs = st.tabs([BATTER_SIDE_LABELS[key] for key in BATTER_SIDE_LABELS])
        for side_key, side_tab in zip(BATTER_SIDE_LABELS, summary_tabs):
            with side_tab:
                summary_frame = pitcher_summary_by_hand.loc[pitcher_summary_by_hand["batter_side_key"] == side_key]
                if summary_frame.empty:
                    st.info("No pitcher summary available.")
                else:
                    render_pitcher_summary_strip(
                        summary_frame.iloc[0],
                        lower_is_better=PITCHER_LOWER_IS_BETTER,
                        higher_is_better=PITCHER_HIGHER_IS_BETTER,
                    )
                    export_sections.append(
                        {
                            "title": f"{team_label} Summary {BATTER_SIDE_LABELS[side_key]}",
                            "frame": summary_frame[PITCHER_SUMMARY_COLUMNS],
                            "lower_is_better": PITCHER_LOWER_IS_BETTER,
                            "higher_is_better": PITCHER_HIGHER_IS_BETTER,
                        }
                    )

    with tab_arsenal:
        arsenal_tabs = st.tabs([BATTER_SIDE_LABELS[key] for key in BATTER_SIDE_LABELS])
        for side_key, side_tab in zip(BATTER_SIDE_LABELS, arsenal_tabs):
            with side_tab:
                side_frame = pitcher_arsenal if side_key == "all" else pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key]
                if side_frame.empty:
                    st.info("No arsenal data available.")
                else:
                    arsenal_grid = render_metric_grid(
                        side_frame[ARSENAL_COLUMNS],
                        key=f"arsenal-{game_pk}-{team_label}-{side_key}",
                        height=250,
                        lower_is_better={"hard_hit_pct", "xwoba_con"},
                        higher_is_better={"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                    )
                    export_sections.append(
                        {
                            "title": f"{team_label} Arsenal {BATTER_SIDE_LABELS[side_key]}",
                            "frame": arsenal_grid[ARSENAL_COLUMNS],
                            "lower_is_better": {"hard_hit_pct", "xwoba_con"},
                            "higher_is_better": {"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                        }
                    )

    with tab_count:
        sub_tabs = st.tabs([BATTER_SIDE_LABELS[key] for key in BATTER_SIDE_LABELS])
        for side_key, side_tab in zip(BATTER_SIDE_LABELS, sub_tabs):
            with side_tab:
                side_count = pitcher_count_usage.loc[pitcher_count_usage["batter_side_key"] == side_key]
                if side_key == "all":
                    side_arsenal = pitcher_arsenal[["pitch_name", "usage_pct"]] if not pitcher_arsenal.empty else pd.DataFrame(columns=["pitch_name", "usage_pct"])
                else:
                    side_arsenal = pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key, ["pitch_name", "usage_pct"]]
                count_frame = pivot_count_usage(side_count, side_arsenal)
                if count_frame.empty:
                    st.info("No count-state usage data available.")
                else:
                    count_grid = render_metric_grid(
                        count_frame[COUNT_USAGE_COLUMNS],
                        key=f"count-{game_pk}-{team_label}-{side_key}",
                        height=250,
                        higher_is_better=set(COUNT_BUCKET_ORDER),
                    )
                    export_sections.append(
                        {
                            "title": f"{team_label} Count Usage {BATTER_SIDE_LABELS[side_key]}",
                            "frame": count_grid[COUNT_USAGE_COLUMNS],
                            "higher_is_better": set(COUNT_BUCKET_ORDER),
                        }
                    )

    return export_sections


def main() -> None:
    st.set_page_config(page_title="MLB Hosted Slate Companion", layout="wide")
    st.title("MLB Hosted Slate Companion")
    base_url = _base_url()
    if not base_url:
        st.error("Set MLB_HOSTED_BASE_URL to your Hugging Face dataset file base URL before running this app.")
        return

    target_date, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only, hitter_preset = _sidebar()
    try:
        slate, rosters, hitters, pitchers, pitcher_summary_by_hand, arsenal, arsenal_by_hand, usage_by_count, hitter_rolling, pitcher_rolling, batter_zone_profiles, pitcher_zone_profiles = _load_artifacts(base_url, target_date)
    except Exception as exc:  # pragma: no cover
        st.error(f"Unable to load hosted artifacts for {target_date.isoformat()}: {exc}")
        return

    all_games = slate.to_dict(orient="records")
    selected_games = _game_selection(all_games)
    st.caption(f"Showing {len(selected_games)} of {len(all_games)} games")
    st.caption("PulledBrl% tracks pulled barrels on tracked batted-ball events. Brl/BIP% uses all balls in play.")
    batter_join_col = "batter_id" if "batter_id" in batter_zone_profiles.columns else "batter"
    pitcher_join_col = "pitcher_id" if "pitcher_id" in pitcher_zone_profiles.columns else "pitcher"
    roster_lookup = rosters[["team", "player_id", "player_name"]].drop_duplicates("player_id")
    batter_zone_named = batter_zone_profiles.merge(roster_lookup, left_on=batter_join_col, right_on="player_id", how="left")

    hitters_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitchers_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for game in selected_games:
        filtered_pitchers = _filter_pitcher_frame(pitchers, split, recent_window, weighted_mode)
        away_pitcher = filtered_pitchers.loc[filtered_pitchers["pitcher_id"] == game.get("away_probable_pitcher_id")].copy()
        home_pitcher = filtered_pitchers.loc[filtered_pitchers["pitcher_id"] == game.get("home_probable_pitcher_id")].copy()
        away_hand = home_pitcher["p_throws"].iloc[0] if not home_pitcher.empty else None
        home_hand = away_pitcher["p_throws"].iloc[0] if not away_pitcher.empty else None

        away_hitters = _filter_hitters(hitters, rosters, game["away_team"], away_hand, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only)
        home_hitters = _filter_hitters(hitters, rosters, game["home_team"], home_hand, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only)
        hitters_by_game[game["game_pk"]] = (add_hitter_matchup_score(away_hitters), add_hitter_matchup_score(home_hitters))
        pitchers_by_game[game["game_pk"]] = (away_pitcher, home_pitcher)

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

    st.header("Top Slate Hitters")
    all_hitters = pd.concat(hitter_rows, ignore_index=True, sort=False) if hitter_rows else pd.DataFrame()
    if all_hitters.empty:
        st.info("No hitter data available for this slate.")
    else:
        preset_columns = hitter_columns_for_preset(hitter_preset)
        ranked_hitters = all_hitters.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last")
        render_metric_grid(
            ranked_hitters[["game"] + [column for column in preset_columns if column in all_hitters.columns]].head(10),
            key="top-slate-hitters-hosted",
            height=320,
        )

    st.header("Top Slate Pitchers")
    all_pitchers = pd.concat(pitcher_rows, ignore_index=True, sort=False) if pitcher_rows else pd.DataFrame()
    if all_pitchers.empty:
        st.info("No pitcher data available for this slate.")
    else:
        ranked_pitchers = add_pitcher_rank_score(all_pitchers)
        render_metric_grid(
            ranked_pitchers[TOP_PITCHER_COLUMNS].head(10),
            key="top-slate-pitchers-hosted",
            height=320,
            lower_is_better=PITCHER_LOWER_IS_BETTER,
            higher_is_better=PITCHER_HIGHER_IS_BETTER,
        )

    st.divider()
    hitter_columns = hitter_columns_for_preset(hitter_preset)

    for idx, game in enumerate(selected_games):
        away_hitters, home_hitters = hitters_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_pitcher, home_pitcher = pitchers_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        filtered_summary = _filter_pitcher_frame(pitcher_summary_by_hand, split, recent_window, weighted_mode)
        filtered_arsenal = _filter_pitcher_frame(arsenal, split, recent_window, weighted_mode)
        filtered_arsenal_by_hand = _filter_pitcher_frame(arsenal_by_hand, split, recent_window, weighted_mode)
        filtered_count = _filter_pitcher_frame(usage_by_count, split, recent_window, weighted_mode)
        away_summary_by_hand = filtered_summary.loc[filtered_summary["pitcher_id"] == game.get("away_probable_pitcher_id")].copy()
        home_summary_by_hand = filtered_summary.loc[filtered_summary["pitcher_id"] == game.get("home_probable_pitcher_id")].copy()
        away_arsenal = filtered_arsenal.loc[filtered_arsenal["pitcher_id"] == game.get("away_probable_pitcher_id")].copy()
        home_arsenal = filtered_arsenal.loc[filtered_arsenal["pitcher_id"] == game.get("home_probable_pitcher_id")].copy()
        away_by_hand = filtered_arsenal_by_hand.loc[filtered_arsenal_by_hand["pitcher_id"] == game.get("away_probable_pitcher_id")].copy()
        home_by_hand = filtered_arsenal_by_hand.loc[filtered_arsenal_by_hand["pitcher_id"] == game.get("home_probable_pitcher_id")].copy()
        away_count = filtered_count.loc[filtered_count["pitcher_id"] == game.get("away_probable_pitcher_id")].copy()
        home_count = filtered_count.loc[filtered_count["pitcher_id"] == game.get("home_probable_pitcher_id")].copy()
        best_matchups = build_best_matchups(away_hitters, home_hitters)

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
                best_matchups = render_metric_grid(best_matchups[BEST_MATCHUP_COLUMNS], key=f"best-hosted-{game['game_pk']}", height=170)

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
                    )

                st.markdown("#### Hitters")
                hitter_cols = st.columns(2)
                with hitter_cols[0]:
                    st.caption(f"{game['away_team']} vs {game.get('home_probable_pitcher_name') or 'opposing starter'}")
                    away_hitters = render_metric_grid(
                        away_hitters[[column for column in hitter_columns if column in away_hitters.columns]],
                        key=f"away-hitters-hosted-{game['game_pk']}",
                        height=360,
                    )
                with hitter_cols[1]:
                    st.caption(f"{game['home_team']} vs {game.get('away_probable_pitcher_name') or 'opposing starter'}")
                    home_hitters = render_metric_grid(
                        home_hitters[[column for column in hitter_columns if column in home_hitters.columns]],
                        key=f"home-hitters-hosted-{game['game_pk']}",
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
                            render_metric_grid(
                                hitter_frame[[column for column in HITTER_ROLLING_COLUMNS if column in hitter_frame.columns]],
                                key=f"h-roll-hosted-{game['game_pk']}-{label}",
                                height=260,
                            )
                        with cols[1]:
                            pitcher_frame = pitcher_rolling.loc[
                                pitcher_rolling["rolling_window"].eq(label)
                                & pitcher_rolling["player_name"].isin([name for name in [away_pitcher_name, home_pitcher_name] if name])
                            ]
                            render_metric_grid(
                                pitcher_frame[[column for column in PITCHER_ROLLING_COLUMNS if column in pitcher_frame.columns]],
                                key=f"p-roll-hosted-{game['game_pk']}-{label}",
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
                            render_metric_grid(
                                zone_frame[[column for column in PITCHER_ZONE_COLUMNS if column in zone_frame.columns]],
                                key=f"p-zone-hosted-{game['game_pk']}-{team_label}",
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
                        render_metric_grid(
                            hitter_detail[[column for column in BATTER_ZONE_COLUMNS if column in hitter_detail.columns]],
                            key=f"h-zone-hosted-{game['game_pk']}-{team_label}",
                            height=240,
                            higher_is_better={"hit_rate", "hr_rate", "damage_rate"},
                        )

            elif active_section == "Exports":
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
