from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .config import AppConfig
from .query_engine import QueryFilters, StatcastQueryEngine
from .ui_components import render_dataframe, render_matchup_header, render_pitcher_card

PITCHER_LOWER_IS_BETTER = {"xwoba", "barrel_pct", "fb_pct", "hard_hit_pct", "avg_launch_angle"}
PITCHER_HIGHER_IS_BETTER = {"swstr_pct", "gb_pct", "gb_fb_ratio", "avg_release_speed", "avg_spin_rate", "usage_pct"}
BEST_MATCHUP_COLUMNS = ["hitter_name", "team", "matchup_score", "swstr_pct", "barrel_pct", "avg_launch_angle", "xwoba", "hard_hit_pct"]


def _default_local_date(config: AppConfig) -> date:
    daily_dir = config.daily_dir
    today = date.today()
    if (daily_dir / today.isoformat()).exists():
        return today
    if daily_dir.exists():
        dated_dirs = sorted([path.name for path in daily_dir.iterdir() if path.is_dir()])
        if dated_dirs:
            return date.fromisoformat(dated_dirs[-1])
    return date.today()


def _sidebar_filters(default_date: date) -> tuple[date, QueryFilters]:
    st.sidebar.title("Filters")
    target_date = st.sidebar.date_input("Slate date", value=default_date)
    split = st.sidebar.selectbox("Split", ["overall", "vs_rhp", "vs_lhp", "home", "away"])
    recent_window = st.sidebar.selectbox("Recent window", ["season", "last_45_days", "last_14_days"])
    weighted_mode = st.sidebar.radio("Weighting", ["weighted", "unweighted"], horizontal=True)
    min_pitch_count = st.sidebar.slider("Min pitch count", 0, 3000, 100, 25)
    min_bip = st.sidebar.slider("Min BIP", 0, 500, 25, 5)
    likely_starters_only = st.sidebar.checkbox("Likely starters only", value=False)
    return target_date, QueryFilters(
        split=split,
        recent_window=recent_window,
        weighted_mode=weighted_mode,
        min_pitch_count=min_pitch_count,
        min_bip=min_bip,
        likely_starters_only=likely_starters_only,
    )


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


def _attach_roster_names(frame, roster_frame, team):
    if frame.empty:
        return frame
    if roster_frame.empty:
        return frame
    lookup = roster_frame.loc[roster_frame["team"] == team, ["player_id", "player_name"]].drop_duplicates("player_id")
    if lookup.empty:
        return frame
    enriched = frame.merge(lookup, left_on="batter", right_on="player_id", how="inner")
    enriched["hitter_name"] = enriched["player_name"]
    return enriched.drop(columns=["player_id", "player_name"], errors="ignore")


def _render_game(engine: StatcastQueryEngine, roster_frame, game: dict, filters: QueryFilters) -> None:
    pitcher_ids = [pitcher_id for pitcher_id in [game.get("away_probable_pitcher_id"), game.get("home_probable_pitcher_id")] if pitcher_id]
    cards = engine.get_pitcher_cards(pitcher_ids, filters)
    arsenal = engine.get_pitcher_arsenal(pitcher_ids, filters)
    away_hand = cards.loc[cards["pitcher_id"] == game.get("home_probable_pitcher_id"), "p_throws"]
    home_hand = cards.loc[cards["pitcher_id"] == game.get("away_probable_pitcher_id"), "p_throws"]
    away_hand_value = away_hand.iloc[0] if not away_hand.empty else None
    home_hand_value = home_hand.iloc[0] if not home_hand.empty else None

    away_hitters = _attach_roster_names(engine.get_team_hitter_pool(game["away_team"], away_hand_value, filters), roster_frame, game["away_team"])
    home_hitters = _attach_roster_names(engine.get_team_hitter_pool(game["home_team"], home_hand_value, filters), roster_frame, game["home_team"])
    away_hitters = _add_matchup_score(away_hitters)
    home_hitters = _add_matchup_score(home_hitters)
    best_matchups = pd.concat([away_hitters, home_hitters], ignore_index=True, sort=False)
    best_matchups = best_matchups.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last").head(3)

    with st.expander(f"{game['away_team']} @ {game['home_team']}", expanded=len(pitcher_ids) > 0):
        render_matchup_header(game)
        st.caption("Top in-game hitter profiles weighted by low SwStr%, high Brl%, and launch angle in the 24-33 degree damage band.")
        render_dataframe(best_matchups[BEST_MATCHUP_COLUMNS], key=f"best-{game['game_pk']}", height=180, title=f"Best Matchups {game['away_team']} at {game['home_team']}")
        pitchers_tab, away_tab, home_tab = st.tabs(["Pitchers", f"{game['away_team']} Hitters", f"{game['home_team']} Hitters"])

        with pitchers_tab:
            columns = st.columns(2)
            for idx, pitcher_id in enumerate(pitcher_ids[:2]):
                with columns[idx]:
                    pitcher_row = cards.loc[cards["pitcher_id"] == pitcher_id]
                    if pitcher_row.empty:
                        st.info("No probable starter metrics available yet.")
                    else:
                        render_pitcher_card(pitcher_row.iloc[0])
                        render_dataframe(
                            pitcher_row[
                                ["pitcher_name", "p_throws", "pitch_count", "bip", "xwoba", "swstr_pct", "barrel_pct", "fb_pct", "gb_fb_ratio", "hard_hit_pct", "avg_launch_angle"]
                            ],
                            key=f"pitcher-summary-{game['game_pk']}-{pitcher_id}",
                            height=90,
                            lower_is_better=PITCHER_LOWER_IS_BETTER,
                            higher_is_better=PITCHER_HIGHER_IS_BETTER,
                            title=f"{pitcher_row.iloc[0]['pitcher_name']} Summary",
                        )
                        st.caption("Pitch arsenal")
                        render_dataframe(
                            arsenal.loc[arsenal["pitcher_id"] == pitcher_id][
                                ["pitch_name", "usage_pct", "swstr_pct", "hard_hit_pct", "avg_release_speed", "avg_spin_rate", "xwoba_con"]
                            ],
                            key=f"arsenal-{game['game_pk']}-{pitcher_id}",
                            height=220,
                            lower_is_better={"hard_hit_pct", "xwoba_con"},
                            higher_is_better={"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                            title=f"{pitcher_row.iloc[0]['pitcher_name']} Arsenal",
                        )

        with away_tab:
            st.caption(f"{game['away_team']} vs {game.get('home_probable_pitcher_name') or 'opposing starter'}")
            render_dataframe(
                away_hitters[
                    ["hitter_name", "team", "matchup_score", "pitch_count", "bip", "xwoba", "xwoba_con", "swstr_pct", "barrel_pct", "fb_pct", "hard_hit_pct", "avg_launch_angle", "likely_starter_score"]
                ],
                key=f"away-{game['game_pk']}",
                height=360,
                title=f"{game['away_team']} Hitters vs {game.get('home_probable_pitcher_name') or 'Opposing Starter'}",
            )

        with home_tab:
            st.caption(f"{game['home_team']} vs {game.get('away_probable_pitcher_name') or 'opposing starter'}")
            render_dataframe(
                home_hitters[
                    ["hitter_name", "team", "matchup_score", "pitch_count", "bip", "xwoba", "xwoba_con", "swstr_pct", "barrel_pct", "fb_pct", "hard_hit_pct", "avg_launch_angle", "likely_starter_score"]
                ],
                key=f"home-{game['game_pk']}",
                height=360,
                title=f"{game['home_team']} Hitters vs {game.get('away_probable_pitcher_name') or 'Opposing Starter'}",
            )
    st.divider()


def _render_explorer(engine: StatcastQueryEngine, filters: QueryFilters) -> None:
    st.header("Explorer")
    entity_type = st.radio("Entity", ["hitters", "pitchers"], horizontal=True)
    search_text = st.text_input("Search")
    results = engine.run_explorer_query(
        entity_type=entity_type,
        search_text=search_text,
        split_key=filters.split,
        recent_window=filters.recent_window,
        weighted_mode=filters.weighted_mode,
        limit=100,
    )
    render_dataframe(results, key=f"explorer-{entity_type}")


def main() -> None:
    st.set_page_config(page_title="MLB Matchup Explorer", layout="wide")
    st.title("MLB Matchup Explorer")
    config = AppConfig()
    engine = StatcastQueryEngine(config)
    target_date, filters = _sidebar_filters(_default_local_date(config))
    games = engine.load_daily_slate(target_date)
    rosters = engine.load_daily_rosters(target_date)
    if not games:
        st.warning("No slate file found for this date. Run the build pipeline first.")
    else:
        selected_games = _filter_games(games)
        st.caption(f"Showing {len(selected_games)} of {len(games)} games")
        slate_best_matchups: list[pd.DataFrame] = []
        for game in selected_games:
            pitcher_ids = [pitcher_id for pitcher_id in [game.get("away_probable_pitcher_id"), game.get("home_probable_pitcher_id")] if pitcher_id]
            cards = engine.get_pitcher_cards(pitcher_ids, filters)
            away_hand = cards.loc[cards["pitcher_id"] == game.get("home_probable_pitcher_id"), "p_throws"]
            home_hand = cards.loc[cards["pitcher_id"] == game.get("away_probable_pitcher_id"), "p_throws"]
            away_hand_value = away_hand.iloc[0] if not away_hand.empty else None
            home_hand_value = home_hand.iloc[0] if not home_hand.empty else None
            away_hitters = _add_matchup_score(_attach_roster_names(engine.get_team_hitter_pool(game["away_team"], away_hand_value, filters), rosters, game["away_team"]))
            home_hitters = _add_matchup_score(_attach_roster_names(engine.get_team_hitter_pool(game["home_team"], home_hand_value, filters), rosters, game["home_team"]))
            slate_best_matchups.append(_with_game_label(away_hitters[BEST_MATCHUP_COLUMNS], f"{game['away_team']} @ {game['home_team']}"))
            slate_best_matchups.append(_with_game_label(home_hitters[BEST_MATCHUP_COLUMNS], f"{game['away_team']} @ {game['home_team']}"))
            _render_game(engine, rosters, game, filters)

        combined_best = pd.concat(slate_best_matchups, ignore_index=True, sort=False) if slate_best_matchups else pd.DataFrame()
        if not combined_best.empty:
            top_slate = combined_best.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last").head(10)
            st.header("Top Slate Matchups")
            st.caption("Best hitter profiles across the selected slate, weighted by low SwStr%, high Brl%, and ideal launch angle.")
            render_dataframe(top_slate[["game"] + BEST_MATCHUP_COLUMNS], key=f"top-slate-{target_date.isoformat()}", height=360, title=f"Top Slate Matchups {target_date.isoformat()}")
            _export_frame(top_slate[["game"] + BEST_MATCHUP_COLUMNS], f"top_slate_matchups_{target_date.isoformat()}", "Top Slate Matchups")
    _render_explorer(engine, filters)
