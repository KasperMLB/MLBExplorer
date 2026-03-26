from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .query_engine import load_remote_parquet
from .ui_components import render_dataframe, render_matchup_header

PITCHER_LOWER_IS_BETTER = {"xwoba", "barrel_pct", "fb_pct", "hard_hit_pct", "avg_launch_angle"}
PITCHER_HIGHER_IS_BETTER = {"swstr_pct", "gb_pct", "gb_fb_ratio", "avg_release_speed", "avg_spin_rate", "usage_pct"}
BEST_MATCHUP_COLUMNS = ["hitter_name", "team", "matchup_score", "swstr_pct", "barrel_pct", "avg_launch_angle", "xwoba", "hard_hit_pct"]


def _base_url() -> str:
    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


@st.cache_data(show_spinner=False)
def _load_artifacts(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    slate = pd.read_parquet(f"{base_url}/daily/{day}/slate.parquet")
    rosters = pd.read_parquet(f"{base_url}/daily/{day}/rosters.parquet")
    hitters = load_remote_parquet(f"{base_url}/reusable", "hitter_metrics.parquet")
    pitchers = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_metrics.parquet")
    arsenal = pd.read_parquet(f"{base_url}/daily/{day}/daily_pitcher_arsenal.parquet")
    return slate, rosters, hitters, pitchers, arsenal


def _sidebar() -> tuple[date, str, str, str, int, int, bool]:
    st.sidebar.title("Hosted filters")
    target_date = st.sidebar.date_input("Slate date", value=date.today())
    split = st.sidebar.selectbox("Split", ["overall", "vs_rhp", "vs_lhp", "home", "away"])
    recent_window = st.sidebar.selectbox("Recent window", ["season", "last_45_days", "last_14_days"])
    weighted_mode = st.sidebar.radio("Weighting", ["weighted", "unweighted"], horizontal=True)
    min_pitch_count = st.sidebar.slider("Min pitch count", 0, 3000, 100, 25)
    min_bip = st.sidebar.slider("Min BIP", 0, 500, 25, 5)
    likely_only = st.sidebar.checkbox("Likely starters only", value=False)
    return target_date, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only


def _game_options(games: list[dict]) -> list[str]:
    return ["All games"] + [f"{game['away_team']} @ {game['home_team']}" for game in games]


def _filter_games(games: list[dict]) -> list[dict]:
    if not games:
        return games
    options = _game_options(games)
    selection = st.sidebar.selectbox("Game", options, index=0)
    if selection == "All games":
        return games
    return [game for game in games if f"{game['away_team']} @ {game['home_team']}" == selection]


def _normalize_series(series: pd.Series, inverse: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(0.5, index=series.index)
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value) or abs(max_value - min_value) < 1e-9:
        normalized = pd.Series(0.5, index=series.index)
    else:
        normalized = (numeric - min_value) / (max_value - min_value)
    if inverse:
        normalized = 1.0 - normalized
    return normalized.fillna(0.5)


def _launch_angle_score(series: pd.Series, low: float = 24.0, ideal: float = 28.5, high: float = 33.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    def score(value: float | int | None) -> float:
        if pd.isna(value):
            return 0.5
        value = float(value)
        if low <= value <= high:
            if value <= ideal:
                return 0.8 + 0.2 * ((value - low) / max(ideal - low, 1e-9))
            return 0.8 + 0.2 * ((high - value) / max(high - ideal, 1e-9))
        if value < low:
            return max(0.0, 0.8 - ((low - value) / max(low, 1e-9)) * 0.8)
        return max(0.0, 0.8 - ((value - high) / max(high, 1e-9)) * 0.8)
    return numeric.apply(score)


def _add_matchup_score(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy()
    swstr_score = _normalize_series(enriched["swstr_pct"], inverse=True)
    barrel_score = _normalize_series(enriched["barrel_pct"])
    launch_angle_score = _launch_angle_score(enriched["avg_launch_angle"])
    enriched["matchup_score"] = ((swstr_score * 0.4) + (barrel_score * 0.35) + (launch_angle_score * 0.25)) * 100.0
    return enriched.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last")


def _export_frame(frame: pd.DataFrame, base_name: str, label: str) -> None:
    if frame.empty:
        return
    csv_data = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Export {label} CSV",
        data=csv_data,
        file_name=f"{base_name}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _with_game_label(frame: pd.DataFrame, game_label: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy()
    enriched.insert(0, "game", game_label)
    return enriched


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
    lookup = rosters.loc[rosters["team"] == team, ["player_id", "player_name"]].drop_duplicates("player_id")
    if not lookup.empty:
        frame = frame.merge(lookup, left_on="batter", right_on="player_id", how="inner")
        frame["hitter_name"] = frame["player_name"]
        frame = frame.drop(columns=["player_id", "player_name"], errors="ignore")
    return frame.sort_values(["likely_starter_score", "xwoba"], ascending=[False, False], na_position="last")


def main() -> None:
    st.set_page_config(page_title="MLB Hosted Slate Companion", layout="wide")
    st.title("MLB Hosted Slate Companion")
    base_url = _base_url()
    if not base_url:
        st.error("Set MLB_HOSTED_BASE_URL to your Hugging Face dataset file base URL before running this app.")
        return

    target_date, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only = _sidebar()
    slate, rosters, hitters, pitchers, arsenal = _load_artifacts(base_url, target_date)
    all_games = slate.to_dict(orient="records")
    selected_games = _filter_games(all_games)
    st.caption(f"Showing {len(selected_games)} of {len(all_games)} games")
    slate_best_matchups: list[pd.DataFrame] = []
    for game in selected_games:
        away_pitcher = pitchers.loc[pitchers["pitcher_id"] == game.get("away_probable_pitcher_id")]
        home_pitcher = pitchers.loc[pitchers["pitcher_id"] == game.get("home_probable_pitcher_id")]
        away_hand = home_pitcher["p_throws"].iloc[0] if not home_pitcher.empty else None
        home_hand = away_pitcher["p_throws"].iloc[0] if not away_pitcher.empty else None
        away_hitters = _add_matchup_score(
            _filter_hitters(hitters, rosters, game["away_team"], away_hand, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only)
        )
        home_hitters = _add_matchup_score(
            _filter_hitters(hitters, rosters, game["home_team"], home_hand, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only)
        )
        slate_best_matchups.append(_with_game_label(away_hitters[BEST_MATCHUP_COLUMNS], f"{game['away_team']} @ {game['home_team']}"))
        slate_best_matchups.append(_with_game_label(home_hitters[BEST_MATCHUP_COLUMNS], f"{game['away_team']} @ {game['home_team']}"))
        best_matchups = pd.concat([away_hitters, home_hitters], ignore_index=True, sort=False)
        best_matchups = best_matchups.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last").head(3)

        with st.expander(f"{game['away_team']} @ {game['home_team']}", expanded=game == selected_games[0]):
            render_matchup_header(game)
            st.caption("Top in-game hitter profiles weighted by low SwStr%, high Brl%, and launch angle in the 24-33 degree damage band.")
            render_dataframe(best_matchups[BEST_MATCHUP_COLUMNS], key=f"best-{game['game_pk']}", height=180, title=f"Best Matchups {game['away_team']} at {game['home_team']}")
            pitchers_tab, away_tab, home_tab = st.tabs(["Pitchers", f"{game['away_team']} Hitters", f"{game['home_team']} Hitters"])

            with pitchers_tab:
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"#### {game['away_team']} starter")
                    render_dataframe(
                        away_pitcher[["pitcher_name", "p_throws", "pitch_count", "bip", "xwoba", "swstr_pct", "barrel_pct", "fb_pct", "gb_fb_ratio", "hard_hit_pct", "avg_launch_angle"]],
                        key=f"away-p-{game['game_pk']}",
                        height=90,
                        lower_is_better=PITCHER_LOWER_IS_BETTER,
                        higher_is_better=PITCHER_HIGHER_IS_BETTER,
                        title=f"{game['away_team']} Starter Summary",
                    )
                    render_dataframe(
                        arsenal.loc[arsenal["pitcher_id"] == game.get("away_probable_pitcher_id")][["pitch_name", "usage_pct", "swstr_pct", "hard_hit_pct", "avg_release_speed", "avg_spin_rate", "xwoba_con"]],
                        key=f"away-a-{game['game_pk']}",
                        height=220,
                        lower_is_better={"hard_hit_pct", "xwoba_con"},
                        higher_is_better={"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                        title=f"{game['away_team']} Starter Arsenal",
                    )
                with cols[1]:
                    st.markdown(f"#### {game['home_team']} starter")
                    render_dataframe(
                        home_pitcher[["pitcher_name", "p_throws", "pitch_count", "bip", "xwoba", "swstr_pct", "barrel_pct", "fb_pct", "gb_fb_ratio", "hard_hit_pct", "avg_launch_angle"]],
                        key=f"home-p-{game['game_pk']}",
                        height=90,
                        lower_is_better=PITCHER_LOWER_IS_BETTER,
                        higher_is_better=PITCHER_HIGHER_IS_BETTER,
                        title=f"{game['home_team']} Starter Summary",
                    )
                    render_dataframe(
                        arsenal.loc[arsenal["pitcher_id"] == game.get("home_probable_pitcher_id")][["pitch_name", "usage_pct", "swstr_pct", "hard_hit_pct", "avg_release_speed", "avg_spin_rate", "xwoba_con"]],
                        key=f"home-a-{game['game_pk']}",
                        height=220,
                        lower_is_better={"hard_hit_pct", "xwoba_con"},
                        higher_is_better={"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                        title=f"{game['home_team']} Starter Arsenal",
                    )

            with away_tab:
                st.caption(f"{game['away_team']} vs {game.get('home_probable_pitcher_name') or 'opposing starter'}")
                render_dataframe(
                    away_hitters[
                        ["hitter_name", "team", "matchup_score", "pitch_count", "bip", "xwoba", "xwoba_con", "swstr_pct", "barrel_pct", "fb_pct", "hard_hit_pct", "avg_launch_angle", "likely_starter_score"]
                    ],
                    key=f"away-h-{game['game_pk']}",
                    height=360,
                    title=f"{game['away_team']} Hitters vs {game.get('home_probable_pitcher_name') or 'Opposing Starter'}",
                )

            with home_tab:
                st.caption(f"{game['home_team']} vs {game.get('away_probable_pitcher_name') or 'opposing starter'}")
                render_dataframe(
                    home_hitters[
                        ["hitter_name", "team", "matchup_score", "pitch_count", "bip", "xwoba", "xwoba_con", "swstr_pct", "barrel_pct", "fb_pct", "hard_hit_pct", "avg_launch_angle", "likely_starter_score"]
                    ],
                    key=f"home-h-{game['game_pk']}",
                    height=360,
                    title=f"{game['home_team']} Hitters vs {game.get('away_probable_pitcher_name') or 'Opposing Starter'}",
                )
        st.divider()
    combined_best = pd.concat(slate_best_matchups, ignore_index=True, sort=False) if slate_best_matchups else pd.DataFrame()
    if not combined_best.empty:
        top_slate = combined_best.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last").head(10)
        st.header("Top Slate Matchups")
        st.caption("Best hitter profiles across the selected slate, weighted by low SwStr%, high Brl%, and ideal launch angle.")
        render_dataframe(top_slate[["game"] + BEST_MATCHUP_COLUMNS], key=f"top-slate-{target_date.isoformat()}", height=360, title=f"Top Slate Matchups {target_date.isoformat()}")
        _export_frame(top_slate[["game"] + BEST_MATCHUP_COLUMNS], f"top_slate_matchups_{target_date.isoformat()}", "Top Slate Matchups")
