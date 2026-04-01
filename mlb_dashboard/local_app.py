from __future__ import annotations

import os
from datetime import date, timedelta
from time import perf_counter

import pandas as pd
import streamlit as st

from .backtesting_view import render_backtesting_tab
from .cockroach_loader import read_hitter_backtest_data, read_pitcher_backtest_data, read_prop_odds_history
from .config import AppConfig
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
    build_zone_overlay_map,
    build_best_matchups,
    build_full_slate_export_bundles,
    compute_family_fit_score,
    build_game_export_options,
    build_slate_export_options,
    filter_excluded_pitchers_from_hitter_pool,
    apply_projected_lineup,
    hitter_columns_for_preset,
    latest_built_date,
    pivot_count_usage,
    sort_arsenal_frame,
    with_game_label,
)
from .odds_service import american_to_implied_prob
from .query_engine import QueryFilters, StatcastQueryEngine
from .rotowire_lineups import fetch_rotowire_lineups, resolve_rotowire_lineups
from .ui_components import (
    build_pitcher_summary_table,
    render_export_hub,
    render_matchup_header,
    render_metric_grid,
    render_slate_export_controls,
    render_zone_tool,
)

try:
    import altair as alt

    HAS_ALTAIR = True
except ImportError:  # pragma: no cover
    alt = None
    HAS_ALTAIR = False


BACKTEST_PRESETS = {
    "Last 7": 7,
    "Last 14": 14,
    "Last 30": 30,
    "Season to date": None,
    "Custom": "custom",
}

HITTER_SCORE_OPTIONS = {
    "Matchup Score": "matchup_score",
    "Ceiling Score (Ladder / 2+ HR Lens)": "ceiling_score",
}

PITCHER_SCORE_OPTIONS = {
    "Pitch Score": "pitcher_score",
    "Strikeout Score": "strikeout_score",
}

HITTER_BACKTEST_OUTCOMES = {
    "Hit Rate": "hit_flag",
    "Start Rate": "started_flag",
    "PA Rate": "had_pa_flag",
    "HR Rate": "home_run_flag",
    "Avg Total Bases": "total_bases",
    "Avg Runs+RBI": "runs_rbi",
}

PITCHER_BACKTEST_OUTCOMES = {
    "Start Rate": "started_flag",
    "Avg Strikeouts": "strikeouts",
    "K Rate per BF": "k_rate_per_bf",
    "5+ K Rate": "k5_flag",
    "Avg Batters Faced": "batters_faced",
    "Avg Hits Allowed": "hits_allowed",
    "Avg Runs Allowed": "runs_allowed",
}


def _default_target_date(config: AppConfig) -> date:
    latest = latest_built_date(config.daily_dir)
    return latest or date.today()


def _existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _perf_enabled() -> bool:
    return os.getenv("MLB_PERF_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _record_perf(events: list[tuple[str, float]], label: str, start_time: float) -> None:
    if _perf_enabled():
        events.append((label, perf_counter() - start_time))


def _render_perf(events: list[tuple[str, float]]) -> None:
    if not _perf_enabled() or not events:
        return
    st.caption("Perf: " + " | ".join(f"{label} {duration:.2f}s" for label, duration in events))


@st.cache_data(show_spinner=False, ttl=300)
def _cached_hitter_backtest_data(
    database_url: str,
    start_date: date,
    end_date: date,
    split_key: str,
    recent_window: str,
    weighted_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = AppConfig(database_url=database_url)
    return read_hitter_backtest_data(config, start_date, end_date, split_key, recent_window, weighted_mode)


@st.cache_data(show_spinner=False, ttl=300)
def _cached_pitcher_backtest_data(
    database_url: str,
    start_date: date,
    end_date: date,
    split_key: str,
    recent_window: str,
    weighted_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = AppConfig(database_url=database_url)
    return read_pitcher_backtest_data(config, start_date, end_date, split_key, recent_window, weighted_mode)


@st.cache_data(show_spinner=False, ttl=300)
def _cached_prop_odds_history(
    database_url: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    config = AppConfig(database_url=database_url)
    return read_prop_odds_history(config, start_date, end_date)


def _sidebar(config: AppConfig) -> tuple[date, QueryFilters, str]:
    st.sidebar.title("Explorer Filters")
    target_date = st.sidebar.date_input("Slate date", value=_default_target_date(config))
    split = st.sidebar.selectbox("Split", ["overall", "vs_rhp", "vs_lhp", "home", "away"])
    recent_window = st.sidebar.selectbox("Recent window", ["season", "last_45_days", "last_14_days"])
    weighted_mode = st.sidebar.radio("Weighting", ["weighted", "unweighted"], horizontal=True)
    min_pitch_count = st.sidebar.slider("Min pitches", 0, 3000, 0, 25)
    min_bip = st.sidebar.slider("Min BIP", 0, 500, 0, 5)
    likely_only = st.sidebar.checkbox("Likely starters only", value=False)
    preset_names = list(HITTER_PRESETS.keys())
    hitter_preset = st.sidebar.selectbox("Hitter view", preset_names, index=preset_names.index("All stats"))
    filters = QueryFilters(
        split=split,
        recent_window=recent_window,
        weighted_mode=weighted_mode,
        min_pitch_count=min_pitch_count,
        min_bip=min_bip,
        likely_starters_only=likely_only,
    )
    return target_date, filters, hitter_preset


def _backtesting_filters() -> dict[str, object]:
    st.subheader("Backtesting")
    row_one = st.columns([1.15, 1.0, 1.0, 1.0, 1.0, 0.9])
    entity = row_one[0].radio("Entity", ["Hitters", "Pitchers", "Boards"], horizontal=True)
    board_entity = row_one[1].radio("Board Type", ["Hitters", "Pitchers"], horizontal=True) if entity == "Boards" else None
    preset = row_one[2].selectbox("Date Preset", list(BACKTEST_PRESETS.keys()), index=1)
    split_key = row_one[3].selectbox("Split", ["overall", "vs_rhp", "vs_lhp", "home", "away"], index=0)
    recent_window = row_one[4].selectbox("Recent Window", ["season", "last_45_days", "last_14_days"], index=0)
    weighted_mode = row_one[5].radio("Weighting", ["weighted", "unweighted"], horizontal=True)

    today = date.today()
    season_start = date(today.year, 1, 1)
    if preset == "Custom":
        default_start = today - timedelta(days=14)
        start_date = st.date_input("Start date", value=default_start, key="backtest-start")
        end_date = st.date_input("End date", value=today, key="backtest-end")
    else:
        days = BACKTEST_PRESETS[preset]
        start_date = season_start if days is None else today - timedelta(days=int(days) - 1)
        end_date = today
        range_cols = st.columns(2)
        range_cols[0].date_input("Start date", value=start_date, disabled=True, key="backtest-start-fixed")
        range_cols[1].date_input("End date", value=end_date, disabled=True, key="backtest-end-fixed")

    row_two = st.columns([1.0, 1.0, 1.0, 1.2])
    min_rows = row_two[0].slider("Min rows", 1, 100, 5)
    bucket_count = row_two[1].selectbox("Calibration Buckets", [10, 20], index=0)
    topn_max = row_two[2].selectbox("Top-N Depth", [3, 5, 10], index=2)
    score_choice = None
    outcome_choice = None
    if entity == "Hitters":
        score_label = row_two[3].selectbox("Score", list(HITTER_SCORE_OPTIONS.keys()), index=0)
        score_choice = HITTER_SCORE_OPTIONS[score_label]
        outcome_choice = st.selectbox("Outcome", list(HITTER_BACKTEST_OUTCOMES.keys()), index=list(HITTER_BACKTEST_OUTCOMES.keys()).index("HR Rate"))
    elif entity == "Pitchers":
        score_label = row_two[3].selectbox("Score", list(PITCHER_SCORE_OPTIONS.keys()), index=1)
        score_choice = PITCHER_SCORE_OPTIONS[score_label]
        outcome_choice = st.selectbox("Outcome", list(PITCHER_BACKTEST_OUTCOMES.keys()), index=list(PITCHER_BACKTEST_OUTCOMES.keys()).index("Avg Strikeouts"))

    return {
        "entity": entity,
        "board_entity": board_entity,
        "start_date": start_date,
        "end_date": end_date,
        "split_key": split_key,
        "recent_window": recent_window,
        "weighted_mode": weighted_mode,
        "min_rows": min_rows,
        "bucket_count": bucket_count,
        "topn_max": topn_max,
        "score_choice": score_choice,
        "outcome_choice": outcome_choice,
    }


def _render_backtesting_explainer(entity: str) -> None:
    if entity == "Hitters":
        st.markdown(
            """
**How To Read This Backtest**

- `Matchup Score` is the default HR betting lens. Higher-ranked hitters should homer more often than the slate baseline.
- `Ceiling Score (Ladder / 2+ HR Lens)` is a power-ceiling lens for ladders and multi-HR style exploration, not the primary single-HR ranking.
- `Top-N HR Success` is the most actionable chart: if the model is useful, Top 1 / Top 3 / Top 5 / Top 10 hitters should materially beat the overall slate HR rate.
- `Calibration by Score Bucket` should climb from low buckets to high buckets. If it does, the score is directionally ranking HR candidates correctly.
- `Odds / ROI` matters more than raw hit rate. A model can look accurate but still lose money if the captured book price is too efficient.
- Dates without stored odds remain model-only. Profitability metrics only begin from the point live odds snapshots were captured.
            """
        )
    elif entity == "Pitchers":
        st.markdown(
            """
**How To Read This Backtest**

- `Strikeout Score` is the default strikeout betting lens. Higher-ranked pitchers should deliver more Ks and beat slate-average K outcomes.
- `Top-N K Success` should show that your highest-ranked pitchers outperform the broader pool on strikeout metrics and milestone rates like 5+ Ks.
- `Calibration by Score Bucket` should rise as score buckets improve. That means the ranking is correctly ordering K upside.
- `Odds / ROI` evaluates whether the captured strikeout prices were actually beatable, not just whether the pitcher projection was directionally good.
- A predictive model is not automatically profitable. Profit requires realized edge over the market’s implied probability after price.
            """
        )
    else:
        st.markdown(
            """
**How To Read This Backtest**

- `Board Rank Performance` tells you whether Rank 1 actually deserves to be treated better than Rank 2 or Rank 3.
- If a board is useful, higher-ranked board winners should beat both lower board ranks and the broader slate baseline.
- Odds-aware results only apply to dates where live prices were captured and stored.
            """
        )


def _load_backtest_data(config: AppConfig, filters: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if filters["entity"] == "Hitters" or filters["board_entity"] == "Hitters":
        return _cached_hitter_backtest_data(
            config.database_url,
            filters["start_date"],
            filters["end_date"],
            filters["split_key"],
            filters["recent_window"],
            filters["weighted_mode"],
        )
    return _cached_pitcher_backtest_data(
        config.database_url,
        filters["start_date"],
        filters["end_date"],
        filters["split_key"],
        filters["recent_window"],
        filters["weighted_mode"],
    )


def _prepare_hitter_backtest_frame(snapshots: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    joined = snapshots.merge(
        outcomes,
        on=["slate_date", "game_pk", "team", "batter_id"],
        how="left",
        suffixes=("", "_outcome"),
    )
    joined["hitter_name"] = joined["hitter_name"].fillna(joined.get("hitter_name_outcome"))
    joined["slate_date"] = pd.to_datetime(joined["slate_date"], errors="coerce")
    joined["started_flag"] = pd.to_numeric(joined.get("started"), errors="coerce").fillna(0).astype(float)
    joined["had_pa_flag"] = pd.to_numeric(joined.get("had_plate_appearance"), errors="coerce").fillna(0).astype(float)
    joined["hit_flag"] = pd.to_numeric(joined.get("hits"), errors="coerce").fillna(0).gt(0).astype(float)
    joined["home_run_flag"] = pd.to_numeric(joined.get("home_runs"), errors="coerce").fillna(0).gt(0).astype(float)
    joined["total_bases"] = pd.to_numeric(joined.get("total_bases"), errors="coerce").fillna(0.0)
    joined["runs_rbi"] = (
        pd.to_numeric(joined.get("runs"), errors="coerce").fillna(0.0)
        + pd.to_numeric(joined.get("rbi"), errors="coerce").fillna(0.0)
    )
    return joined


def _prepare_pitcher_backtest_frame(snapshots: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    joined = snapshots.merge(
        outcomes,
        on=["slate_date", "game_pk", "team", "pitcher_id"],
        how="left",
        suffixes=("", "_outcome"),
    )
    joined["pitcher_name"] = joined["pitcher_name"].fillna(joined.get("pitcher_name_outcome"))
    joined["slate_date"] = pd.to_datetime(joined["slate_date"], errors="coerce")
    joined["started_flag"] = pd.to_numeric(joined.get("started"), errors="coerce").fillna(0).astype(float)
    joined["strikeouts"] = pd.to_numeric(joined.get("strikeouts"), errors="coerce").fillna(0.0)
    joined["batters_faced"] = pd.to_numeric(joined.get("batters_faced"), errors="coerce").fillna(0.0)
    joined["hits_allowed"] = pd.to_numeric(joined.get("hits_allowed"), errors="coerce").fillna(0.0)
    joined["runs_allowed"] = pd.to_numeric(joined.get("runs_allowed"), errors="coerce").fillna(0.0)
    joined["k_rate_per_bf"] = joined["strikeouts"] / joined["batters_faced"].replace(0, pd.NA)
    joined["k_rate_per_bf"] = pd.to_numeric(joined["k_rate_per_bf"], errors="coerce").fillna(0.0)
    joined["k5_flag"] = joined["strikeouts"].ge(5).astype(float)
    return joined


def _normalize_prop_odds_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["fetched_at"] = pd.to_datetime(work.get("fetched_at"), errors="coerce", utc=True)
    work["commence_time"] = pd.to_datetime(work.get("commence_time"), errors="coerce", utc=True)
    local_commence = work["commence_time"].dt.tz_convert("America/Chicago").dt.tz_localize(None)
    work["slate_date"] = local_commence.dt.normalize()
    work["player_name"] = work.get("player_name", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
    work["market_key"] = work.get("market_key", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
    work["selection_side"] = work.get("selection_side", pd.Series(dtype="object")).fillna("").astype(str).str.strip().str.lower()
    work["selection_label"] = work.get("selection_label", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
    work["odds_american"] = pd.to_numeric(work.get("odds_american"), errors="coerce")
    work["line"] = pd.to_numeric(work.get("line"), errors="coerce")
    work["implied_prob"] = work["odds_american"].apply(american_to_implied_prob)
    return work


def _latest_best_price(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    if "is_primary_line" in work.columns and work["is_primary_line"].notna().any():
        primary = work.loc[work["is_primary_line"] == True].copy()  # noqa: E712
        if not primary.empty:
            work = primary
    latest = work.groupby(group_columns, as_index=False)["fetched_at"].max().rename(columns={"fetched_at": "latest_fetched_at"})
    work = work.merge(latest, on=group_columns, how="inner")
    work = work.loc[work["fetched_at"] == work["latest_fetched_at"]].copy()
    work["decimal_price"] = work["odds_american"].apply(lambda price: None if pd.isna(price) else (1.0 + (price / 100.0) if price > 0 else 1.0 + (100.0 / abs(price))))
    work = work.sort_values(["decimal_price", "odds_american"], ascending=[False, False], na_position="last")
    return work.groupby(group_columns, as_index=False).head(1).reset_index(drop=True)


def _join_hitter_prop_odds(frame: pd.DataFrame, odds_history: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or odds_history.empty:
        return pd.DataFrame()
    odds = odds_history.loc[
        odds_history["market_key"].eq("batter_home_runs")
        & odds_history["selection_side"].eq("over")
        & odds_history["line"].isin([0.5, 1.5])
    ].copy()
    if odds.empty:
        return pd.DataFrame()
    group_columns = ["slate_date", "player_name", "market_key", "line"]
    best = _latest_best_price(odds, group_columns)
    merged = frame.merge(
        best.rename(columns={"player_name": "hitter_name"}),
        on=["slate_date", "hitter_name"],
        how="left",
        suffixes=("", "_odds"),
    )
    return merged


def _join_pitcher_prop_odds(frame: pd.DataFrame, odds_history: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or odds_history.empty:
        return pd.DataFrame()
    odds = odds_history.loc[
        odds_history["market_key"].eq("pitcher_strikeouts")
        & odds_history["selection_side"].isin(["over", "under"])
    ].copy()
    if odds.empty:
        return pd.DataFrame()
    primary = odds.loc[odds["selection_side"].eq("over")].copy()
    if primary.empty:
        return pd.DataFrame()
    if "is_primary_line" in primary.columns and primary["is_primary_line"].fillna(False).any():
        primary = primary.loc[primary["is_primary_line"].fillna(False)].copy()
    else:
        line_rank = (
            primary.groupby(["slate_date", "player_name", "line"], as_index=False)
            .agg(book_count=("sportsbook", "nunique"))
            .sort_values(["slate_date", "player_name", "book_count", "line"], ascending=[True, True, False, True], na_position="last")
        )
        line_rank["line_rank"] = line_rank.groupby(["slate_date", "player_name"]).cumcount() + 1
        primary = primary.merge(
            line_rank.loc[line_rank["line_rank"] == 1, ["slate_date", "player_name", "line"]],
            on=["slate_date", "player_name", "line"],
            how="inner",
        )
    group_columns = ["slate_date", "player_name", "market_key", "selection_side"]
    best = _latest_best_price(primary, group_columns)
    merged = frame.merge(
        best.rename(columns={"player_name": "pitcher_name"}),
        on=["slate_date", "pitcher_name"],
        how="left",
        suffixes=("", "_odds"),
    )
    return merged


def _compute_hr_ladder_flags(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["hr2_flag"] = pd.to_numeric(work.get("home_runs"), errors="coerce").fillna(0).ge(2).astype(float)
    return work


def _compute_pitcher_over_flag(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["over_line_flag"] = (
        pd.to_numeric(work.get("strikeouts"), errors="coerce").fillna(0)
        > pd.to_numeric(work.get("line"), errors="coerce").fillna(999)
    ).astype(float)
    return work


def _build_topn_event_summary(frame: pd.DataFrame, score_column: str, outcome_column: str, top_ns: list[int]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["top_n", "events", "event_rate", "baseline_rate", "lift_vs_baseline"])
    ranked = frame.sort_values(["slate_date", "game_pk", score_column], ascending=[True, True, False], na_position="last").copy()
    ranked["rank"] = ranked.groupby(["slate_date", "game_pk"]).cumcount() + 1
    baseline = float(pd.to_numeric(frame[outcome_column], errors="coerce").fillna(0).mean())
    rows: list[dict] = []
    for top_n in top_ns:
        subset = ranked.loc[ranked["rank"] <= top_n].copy()
        if subset.empty:
            continue
        event_series = pd.to_numeric(subset[outcome_column], errors="coerce").fillna(0)
        event_count = float(event_series.sum()) if event_series.max() > 1 else int(event_series.sum())
        event_rate = float(event_series.mean())
        rows.append(
            {
                "top_n": f"Top {top_n}",
                "events": event_count,
                "event_rate": event_rate,
                "baseline_rate": baseline,
                "lift_vs_baseline": event_rate - baseline,
            }
        )
    return pd.DataFrame(rows)


def _build_pitcher_k_summary(frame: pd.DataFrame, score_column: str, top_ns: list[int]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["top_n", "avg_strikeouts", "k5_rate", "baseline_avg_strikeouts", "baseline_k5_rate"])
    ranked = frame.sort_values(["slate_date", "game_pk", score_column], ascending=[True, True, False], na_position="last").copy()
    ranked["rank"] = ranked.groupby(["slate_date", "game_pk"]).cumcount() + 1
    baseline_avg_k = float(pd.to_numeric(frame["strikeouts"], errors="coerce").fillna(0).mean())
    baseline_k5 = float(pd.to_numeric(frame["k5_flag"], errors="coerce").fillna(0).mean())
    rows: list[dict] = []
    for top_n in top_ns:
        subset = ranked.loc[ranked["rank"] <= top_n].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "top_n": f"Top {top_n}",
                "avg_strikeouts": pd.to_numeric(subset["strikeouts"], errors="coerce").fillna(0).mean(),
                "k5_rate": pd.to_numeric(subset["k5_flag"], errors="coerce").fillna(0).mean(),
                "baseline_avg_strikeouts": baseline_avg_k,
                "baseline_k5_rate": baseline_k5,
            }
        )
    return pd.DataFrame(rows)


def _build_hit_miss_table(frame: pd.DataFrame, score_column: str, outcome_column: str, name_column: str, hits: bool, limit: int = 25) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    outcome = pd.to_numeric(frame.get(outcome_column), errors="coerce").fillna(0)
    mask = outcome.gt(0) if hits else outcome.eq(0)
    detail_columns = [column for column in [name_column, "slate_date", "game_label", "team", score_column, outcome_column, "odds_american", "implied_prob", "line"] if column in frame.columns]
    return frame.loc[mask, detail_columns].sort_values(["slate_date", score_column], ascending=[False, False], na_position="last").head(limit)


def _compute_odds_grading(frame: pd.DataFrame, event_column: str) -> pd.DataFrame:
    if frame.empty or "odds_american" not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    work["event_result"] = pd.to_numeric(work.get(event_column), errors="coerce").fillna(0).astype(float)
    work = work.loc[work["odds_american"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["decimal_price"] = work["odds_american"].apply(lambda price: None if pd.isna(price) else (1.0 + (price / 100.0) if price > 0 else 1.0 + (100.0 / abs(price))))
    work["units"] = work.apply(
        lambda row: (row["decimal_price"] - 1.0) if row["event_result"] > 0 else -1.0,
        axis=1,
    )
    work["roi"] = work["units"]
    return work


def _render_event_summary_cards(summary: pd.DataFrame, title_prefix: str) -> None:
    if summary.empty:
        st.info(f"No {title_prefix.lower()} summary available.")
        return
    cards = st.columns(len(summary))
    for card, (_, row) in zip(cards, summary.iterrows()):
        card.metric(
            row["top_n"],
            f"{row['event_rate'] * 100:.1f}%",
            f"{row['lift_vs_baseline'] * 100:+.1f} pts vs baseline",
        )


def _bucket_scores(frame: pd.DataFrame, score_column: str, bucket_count: int) -> pd.DataFrame:
    work = frame.loc[pd.to_numeric(frame.get(score_column), errors="coerce").notna()].copy()
    if work.empty:
        return work
    bucket_total = min(bucket_count, max(int(work[score_column].nunique()), 1))
    labels = [f"B{i}" for i in range(bucket_total, 0, -1)]
    try:
        work["score_bucket"] = pd.qcut(work[score_column], q=bucket_total, labels=labels, duplicates="drop")
    except ValueError:
        work["score_bucket"] = "B1"
    return work


def _build_calibration_table(frame: pd.DataFrame, score_column: str, outcome_column: str, bucket_count: int) -> pd.DataFrame:
    bucketed = _bucket_scores(frame, score_column, bucket_count)
    if bucketed.empty:
        return pd.DataFrame(columns=["score_bucket", "predictions", "avg_score", "avg_outcome"])
    calibration = (
        bucketed.groupby("score_bucket", observed=False, as_index=False)
        .agg(
            predictions=(score_column, "size"),
            avg_score=(score_column, "mean"),
            avg_outcome=(outcome_column, "mean"),
        )
        .sort_values("avg_score", ascending=False, na_position="last")
    )
    return calibration


def _build_histogram(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    numeric = pd.to_numeric(frame.get(score_column), errors="coerce").dropna()
    if numeric.empty:
        return pd.DataFrame(columns=["score_range", "count"])
    bins = min(12, max(int(numeric.nunique()), 1))
    counts, edges = pd.cut(numeric, bins=bins, include_lowest=True, retbins=True)
    histogram = counts.value_counts(sort=False).reset_index()
    histogram.columns = ["score_range", "count"]
    histogram["score_range"] = histogram["score_range"].astype(str)
    return histogram


def _build_rolling_table(frame: pd.DataFrame, score_column: str, outcome_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["slate_date", "avg_score", "avg_outcome", "predictions"])
    rolling = (
        frame.groupby("slate_date", as_index=False)
        .agg(
            avg_score=(score_column, "mean"),
            avg_outcome=(outcome_column, "mean"),
            predictions=(score_column, "size"),
        )
        .sort_values("slate_date")
    )
    return rolling


def _build_topn_table(frame: pd.DataFrame, score_column: str, outcome_column: str, topn_max: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["top_n", "avg_outcome", "observations"])
    ranked = frame.sort_values(["slate_date", "game_pk", score_column], ascending=[True, True, False], na_position="last").copy()
    ranked["within_game_rank"] = ranked.groupby(["slate_date", "game_pk"]).cumcount() + 1
    rows: list[dict] = []
    for top_n in [1, 3, 5, topn_max]:
        subset = ranked.loc[ranked["within_game_rank"] <= top_n]
        if subset.empty:
            continue
        rows.append(
            {
                "top_n": f"Top {top_n}",
                "avg_outcome": float(pd.to_numeric(subset[outcome_column], errors="coerce").fillna(0).mean()),
                "observations": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def _build_leaderboard(frame: pd.DataFrame, name_column: str, score_column: str, outcome_column: str, min_rows: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[name_column, "predictions", "avg_score", "avg_outcome"])
    leaderboard = (
        frame.groupby(name_column, as_index=False)
        .agg(
            predictions=(score_column, "size"),
            avg_score=(score_column, "mean"),
            avg_outcome=(outcome_column, "mean"),
        )
        .loc[lambda df: df["predictions"] >= min_rows]
        .sort_values(["avg_outcome", "avg_score"], ascending=[False, False], na_position="last")
    )
    return leaderboard


def _render_simple_chart(data: pd.DataFrame, x: str, y: str, chart_type: str, title: str, color: str | None = None) -> None:
    if data.empty:
        st.info(f"No data for {title.lower()}.")
        return
    if HAS_ALTAIR:
        if chart_type == "bar":
            chart = alt.Chart(data).mark_bar().encode(x=alt.X(x, sort=None), y=alt.Y(y), color=alt.value(color or "#1f77b4"))
        elif chart_type == "line":
            chart = alt.Chart(data).mark_line(point=True).encode(x=alt.X(x), y=alt.Y(y), color=alt.value(color or "#1f77b4"))
        else:
            chart = alt.Chart(data).mark_circle(size=55, opacity=0.65).encode(x=alt.X(x), y=alt.Y(y), tooltip=list(data.columns))
        st.altair_chart(chart.properties(height=280, title=title), use_container_width=True)
        return
    st.markdown(f"##### {title}")
    chart_frame = data[[x, y]].copy()
    if chart_type == "line":
        st.line_chart(chart_frame.set_index(x))
    elif chart_type == "scatter":
        st.scatter_chart(chart_frame, x=x, y=y)
    else:
        st.bar_chart(chart_frame.set_index(x))


def _render_backtest_kpis(frame: pd.DataFrame, score_column: str, outcome_column: str, label: str) -> None:
    raw_score = pd.to_numeric(frame.get(score_column), errors="coerce")
    raw_outcome = pd.to_numeric(frame.get(outcome_column), errors="coerce")
    numeric_score = raw_score.fillna(0.0)
    numeric_outcome = raw_outcome.fillna(0.0)
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Predictions", f"{len(frame):,}")
    kpi_cols[1].metric("Matched Outcomes", f"{raw_outcome.notna().sum():,}")
    kpi_cols[2].metric("Avg Score", f"{numeric_score.mean():.1f}")
    kpi_cols[3].metric(label, f"{numeric_outcome.mean():.3f}" if numeric_outcome.max() <= 1.0 else f"{numeric_outcome.mean():.2f}")


def _render_backtest_entity_view(
    frame: pd.DataFrame,
    score_column: str,
    outcome_column: str,
    outcome_label: str,
    name_column: str,
    filters: dict[str, object],
    entity: str,
    odds_history: pd.DataFrame,
) -> None:
    if frame.empty:
        st.info("No historical rows matched the selected filters.")
        return
    _render_backtesting_explainer(entity)

    if entity == "Hitters":
        frame = _compute_hr_ladder_flags(frame)
        event_summary = _build_topn_event_summary(frame, score_column, outcome_column, [1, 3, 5, 10])
        st.markdown("#### HR Event Summary")
        _render_event_summary_cards(event_summary, "HR")
        render_metric_grid(event_summary, key=f"hr-topn-summary-{score_column}", height=180, use_lightweight=True)
        if score_column == "ceiling_score":
            st.caption("Ceiling Score is being used as a ladder / 2+ HR lens here. Matchup Score remains the default single-HR ranking.")
        odds_joined = _join_hitter_prop_odds(frame, odds_history)
        hr_market = odds_joined.loc[odds_joined["line"].eq(0.5)].copy() if not odds_joined.empty else pd.DataFrame()
        ladder_market = odds_joined.loc[odds_joined["line"].eq(1.5)].copy() if not odds_joined.empty else pd.DataFrame()
        st.markdown("#### Top HR Hits")
        render_metric_grid(_build_hit_miss_table(frame, score_column, outcome_column, name_column, True), key=f"hr-hit-table-{score_column}", height=240, use_lightweight=True)
        st.markdown("#### Top HR Misses")
        render_metric_grid(_build_hit_miss_table(frame, score_column, outcome_column, name_column, False), key=f"hr-miss-table-{score_column}", height=240, use_lightweight=True)
        st.markdown("#### Odds / ROI")
        if hr_market.empty:
            st.info("Model-only mode for this date range. No captured HR odds were found yet, so ROI metrics are unavailable.")
        else:
            graded = _compute_odds_grading(hr_market, "home_run_flag")
            if not graded.empty:
                odds_cards = st.columns(4)
                odds_cards[0].metric("Captured HR Bets", f"{len(graded):,}")
                odds_cards[1].metric("Avg Implied HR%", f"{graded['implied_prob'].fillna(0).mean() * 100:.1f}%")
                odds_cards[2].metric("Realized HR%", f"{graded['event_result'].mean() * 100:.1f}%")
                odds_cards[3].metric("Units", f"{graded['units'].sum():+.2f}")
                roi_summary = pd.DataFrame(
                    [
                        {
                            "bets": len(graded),
                            "avg_implied_prob": graded["implied_prob"].fillna(0).mean(),
                            "realized_rate": graded["event_result"].mean(),
                            "units": graded["units"].sum(),
                            "roi_per_bet": graded["units"].mean(),
                        }
                    ]
                )
                render_metric_grid(roi_summary, key=f"hr-roi-summary-{score_column}", height=130, use_lightweight=True)
                render_metric_grid(
                    graded[[column for column in [name_column, "slate_date", "game_label", "team", score_column, "line", "odds_american", "implied_prob", "event_result", "units"] if column in graded.columns]].sort_values(["slate_date", score_column], ascending=[False, False], na_position="last").head(200),
                    key=f"hr-roi-detail-{score_column}",
                    height=260,
                    use_lightweight=True,
                )
            else:
                st.info("Captured HR odds were found, but no rows were gradeable for the selected filters.")
        if score_column == "ceiling_score":
            st.markdown("#### Ladder / 2+ HR Lens")
            ladder_summary = _build_topn_event_summary(frame, score_column, "hr2_flag", [1, 3, 5, 10])
            render_metric_grid(ladder_summary, key="ladder-summary", height=180, use_lightweight=True)
            if not ladder_market.empty:
                ladder_graded = _compute_odds_grading(
                    ladder_market.rename(columns={"hr2_flag": "event_result"}),
                    "event_result",
                )
                if not ladder_graded.empty:
                    render_metric_grid(
                        ladder_graded[[column for column in [name_column, "slate_date", "game_label", "team", score_column, "line", "odds_american", "implied_prob", "event_result", "units"] if column in ladder_graded.columns]].head(100),
                        key="ladder-roi-detail",
                        height=220,
                        use_lightweight=True,
                    )
    else:
        frame = _compute_pitcher_over_flag(frame)
        event_summary = _build_topn_event_summary(frame, score_column, "k5_flag", [1, 3, 5])
        k_summary = _build_pitcher_k_summary(frame, score_column, [1, 3, 5])
        st.markdown("#### Strikeout Event Summary")
        _render_event_summary_cards(event_summary, "K")
        render_metric_grid(event_summary, key=f"k-topn-summary-{score_column}", height=180, use_lightweight=True)
        render_metric_grid(k_summary, key=f"k-avg-summary-{score_column}", height=180, use_lightweight=True)
        st.markdown("#### Top Strikeout Hits")
        render_metric_grid(_build_hit_miss_table(frame, score_column, "k5_flag", name_column, True), key=f"k-hit-table-{score_column}", height=220, use_lightweight=True)
        st.markdown("#### Top Strikeout Misses")
        render_metric_grid(_build_hit_miss_table(frame, score_column, "k5_flag", name_column, False), key=f"k-miss-table-{score_column}", height=220, use_lightweight=True)
        st.markdown("#### Odds / ROI")
        odds_joined = _join_pitcher_prop_odds(frame, odds_history)
        if odds_joined.empty:
            st.info("Model-only mode for this date range. No captured strikeout odds were found yet, so ROI metrics are unavailable.")
        else:
            graded = _compute_odds_grading(_compute_pitcher_over_flag(odds_joined), "over_line_flag")
            if not graded.empty:
                odds_cards = st.columns(4)
                odds_cards[0].metric("Captured K Bets", f"{len(graded):,}")
                odds_cards[1].metric("Avg Implied Over%", f"{graded['implied_prob'].fillna(0).mean() * 100:.1f}%")
                odds_cards[2].metric("Realized Over%", f"{graded['event_result'].mean() * 100:.1f}%")
                odds_cards[3].metric("Units", f"{graded['units'].sum():+.2f}")
                render_metric_grid(
                    graded[[column for column in [name_column, "slate_date", "game_label", "team", score_column, "line", "odds_american", "implied_prob", "event_result", "units", "strikeouts"] if column in graded.columns]].sort_values(["slate_date", score_column], ascending=[False, False], na_position="last").head(200),
                    key=f"k-roi-detail-{score_column}",
                    height=260,
                    use_lightweight=True,
                )
            else:
                st.info("Captured strikeout odds were found, but no rows were gradeable for the selected filters.")

    _render_backtest_kpis(frame, score_column, outcome_column, outcome_label)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        _render_simple_chart(_build_histogram(frame, score_column), "score_range", "count", "bar", "Score Distribution")
    with chart_cols[1]:
        _render_simple_chart(_build_calibration_table(frame, score_column, outcome_column, int(filters["bucket_count"])), "score_bucket", "avg_outcome", "bar", "Calibration by Score Bucket")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        _render_simple_chart(_build_rolling_table(frame, score_column, outcome_column), "slate_date", "avg_outcome", "line", f"Rolling {outcome_label}")
    with chart_cols[1]:
        _render_simple_chart(_build_topn_table(frame, score_column, outcome_column, int(filters["topn_max"])), "top_n", "avg_outcome", "bar", "Top-N Success")

    _render_simple_chart(
        frame[[score_column, outcome_column]].dropna().head(400),
        score_column,
        outcome_column,
        "scatter",
        "Score vs Realized Outcome",
    )

    st.markdown("#### Calibration Table")
    calibration = _build_calibration_table(frame, score_column, outcome_column, int(filters["bucket_count"]))
    render_metric_grid(calibration, key=f"calibration-{score_column}-{outcome_column}", height=220, use_lightweight=True)

    st.markdown("#### Player Leaderboard")
    leaderboard = _build_leaderboard(frame, name_column, score_column, outcome_column, int(filters["min_rows"]))
    render_metric_grid(leaderboard.head(30), key=f"leaderboard-{name_column}-{score_column}", height=320, use_lightweight=True)

    st.markdown("#### Per-Date Summary")
    render_metric_grid(
        _build_rolling_table(frame, score_column, outcome_column),
        key=f"rolling-table-{score_column}-{outcome_column}",
        height=260,
        use_lightweight=True,
    )

    st.markdown("#### Raw Joined Rows")
    detail_columns = [column for column in [name_column, "slate_date", "game_label", "team", score_column, outcome_column] if column in frame.columns]
    render_metric_grid(
        frame[detail_columns].sort_values(["slate_date", score_column], ascending=[False, False], na_position="last").head(250),
        key=f"detail-{name_column}-{score_column}",
        height=360,
        use_lightweight=True,
    )


def _prepare_board_backtest_frame(
    board_entity: str,
    snapshots: pd.DataFrame,
    outcomes: pd.DataFrame,
    boards: pd.DataFrame,
) -> pd.DataFrame:
    if boards.empty:
        return pd.DataFrame()
    if board_entity == "Hitters":
        joined = boards.merge(
            outcomes,
            on=["slate_date", "game_pk", "team", "batter_id"],
            how="left",
            suffixes=("", "_outcome"),
        )
        joined["hit_flag"] = pd.to_numeric(joined.get("hits"), errors="coerce").fillna(0).gt(0).astype(float)
        joined["home_run_flag"] = pd.to_numeric(joined.get("home_runs"), errors="coerce").fillna(0).gt(0).astype(float)
        joined["runs_rbi"] = (
            pd.to_numeric(joined.get("runs"), errors="coerce").fillna(0.0)
            + pd.to_numeric(joined.get("rbi"), errors="coerce").fillna(0.0)
        )
        joined["total_bases"] = pd.to_numeric(joined.get("total_bases"), errors="coerce").fillna(0.0)
        joined["started_flag"] = pd.to_numeric(joined.get("started"), errors="coerce").fillna(0).astype(float)
        joined["pool_avg_source_score"] = None
        if not snapshots.empty:
            for metric in boards["source_metric"].dropna().unique():
                mask = joined["source_metric"].eq(metric)
                if metric in snapshots.columns:
                    pool = snapshots.groupby(["slate_date", "game_pk"], as_index=False).agg(pool_avg_source_score=(metric, "mean"))
                    joined.loc[mask, "pool_avg_source_score"] = joined.loc[mask].merge(pool, on=["slate_date", "game_pk"], how="left")["pool_avg_source_score"].to_numpy()
        joined["slate_date"] = pd.to_datetime(joined["slate_date"], errors="coerce")
        return joined

    joined = boards.merge(
        outcomes,
        on=["slate_date", "game_pk", "team", "pitcher_id"],
        how="left",
        suffixes=("", "_outcome"),
    )
    joined["started_flag"] = pd.to_numeric(joined.get("started"), errors="coerce").fillna(0).astype(float)
    joined["strikeouts"] = pd.to_numeric(joined.get("strikeouts"), errors="coerce").fillna(0.0)
    joined["runs_allowed"] = pd.to_numeric(joined.get("runs_allowed"), errors="coerce").fillna(0.0)
    joined["hits_allowed"] = pd.to_numeric(joined.get("hits_allowed"), errors="coerce").fillna(0.0)
    joined["k5_flag"] = joined["strikeouts"].ge(5).astype(float)
    joined["slate_date"] = pd.to_datetime(joined["slate_date"], errors="coerce")
    return joined


def _render_board_backtest_view(
    board_entity: str,
    board_frame: pd.DataFrame,
    available_boards: list[str],
) -> None:
    if board_frame.empty:
        st.info("No board winner history matched the selected range.")
        return
    board_name = st.selectbox("Board", available_boards, index=0)
    filtered = board_frame.loc[board_frame["board_name"] == board_name].copy()
    if board_entity == "Hitters":
        outcome_map = {
            "Hit Rate": "hit_flag",
            "HR Rate": "home_run_flag",
            "Avg Total Bases": "total_bases",
            "Avg Runs+RBI": "runs_rbi",
            "Start Rate": "started_flag",
        }
        name_col = "hitter_name"
    else:
        outcome_map = {
            "Start Rate": "started_flag",
            "Avg Strikeouts": "strikeouts",
            "5+ K Rate": "k5_flag",
            "Avg Runs Allowed": "runs_allowed",
            "Avg Hits Allowed": "hits_allowed",
        }
        name_col = "pitcher_name"
    outcome_label = st.selectbox("Board Outcome", list(outcome_map.keys()), index=0)
    outcome_column = outcome_map[outcome_label]

    _render_backtest_kpis(filtered, "board_score", outcome_column, outcome_label)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        rank_table = (
            filtered.groupby("board_rank", as_index=False)
            .agg(avg_outcome=(outcome_column, "mean"), picks=("board_score", "size"))
            .sort_values("board_rank")
        )
        _render_simple_chart(rank_table, "board_rank", "avg_outcome", "bar", "Board Rank Performance")
    with chart_cols[1]:
        _render_simple_chart(
            filtered.groupby("slate_date", as_index=False).agg(avg_outcome=(outcome_column, "mean")).sort_values("slate_date"),
            "slate_date",
            "avg_outcome",
            "line",
            "Rolling Board Performance",
        )

    st.markdown("#### Board Summary")
    summary = (
        board_frame.groupby(["board_name", "board_rank"], as_index=False)
        .agg(
            picks=("board_score", "size"),
            avg_board_score=("board_score", "mean"),
            avg_outcome=(outcome_column, "mean"),
        )
        .sort_values(["board_name", "board_rank"])
    )
    render_metric_grid(summary, key=f"board-summary-{board_entity}", height=260, use_lightweight=True)

    st.markdown("#### Board History")
    detail_columns = [column for column in ["slate_date", "board_name", "board_rank", name_col, "team", "board_score", outcome_column, "source_metric"] if column in filtered.columns]
    render_metric_grid(
        filtered[detail_columns].sort_values(["slate_date", "board_rank"], ascending=[False, True], na_position="last"),
        key=f"board-history-{board_entity}",
        height=360,
        use_lightweight=True,
    )


def _render_backtesting_tab(config: AppConfig) -> None:
    if not config.database_url:
        st.error("DATABASE_URL is required for Backtesting because this view reads historical model results directly from Cockroach.")
        return

    filters = _backtesting_filters()
    if filters["start_date"] > filters["end_date"]:
        st.error("Start date must be on or before end date.")
        return

    try:
        snapshots, outcomes, boards = _load_backtest_data(config, filters)
        odds_history = _normalize_prop_odds_history(
            _cached_prop_odds_history(config.database_url, filters["start_date"], filters["end_date"])
        )
    except Exception as exc:
        st.error(f"Unable to load backtest history from Cockroach: {exc}")
        return

    if filters["entity"] == "Hitters":
        frame = _prepare_hitter_backtest_frame(snapshots, outcomes)
        _render_backtest_entity_view(
            frame,
            str(filters["score_choice"]),
            HITTER_BACKTEST_OUTCOMES[str(filters["outcome_choice"])],
            str(filters["outcome_choice"]),
            "hitter_name",
            filters,
            "Hitters",
            odds_history,
        )
        return

    if filters["entity"] == "Pitchers":
        frame = _prepare_pitcher_backtest_frame(snapshots, outcomes)
        _render_backtest_entity_view(
            frame,
            str(filters["score_choice"]),
            PITCHER_BACKTEST_OUTCOMES[str(filters["outcome_choice"])],
            str(filters["outcome_choice"]),
            "pitcher_name",
            filters,
            "Pitchers",
            odds_history,
        )
        return

    board_entity = str(filters["board_entity"])
    _render_backtesting_explainer("Boards")
    board_frame = _prepare_board_backtest_frame(board_entity, snapshots, outcomes, boards)
    available_boards = sorted(board_frame.get("board_name", pd.Series(dtype="object")).dropna().unique().tolist())
    if not available_boards:
        st.info("No board history matched the selected filters.")
        return
    _render_board_backtest_view(board_entity, board_frame, available_boards)


def _game_selection(slate: list[dict]) -> list[dict]:
    if not slate:
        return []
    options = ["All games"] + [f"{game['away_team']} @ {game['home_team']}" for game in slate]
    selection = st.sidebar.selectbox("Game", options, index=0)
    if selection == "All games":
        return slate
    return [game for game in slate if f"{game['away_team']} @ {game['home_team']}" == selection]


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
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    export_sections: list[dict] = []
    summary_all = pitcher_summary_by_hand.loc[
        pitcher_summary_by_hand["batter_side_key"] == "all",
        _existing_columns(pitcher_summary_by_hand, PITCHER_SUMMARY_COLUMNS),
    ].copy()
    arsenal_grid = pd.DataFrame(columns=ARSENAL_COLUMNS)
    hand_grids: dict[str, pd.DataFrame] = {}
    count_grids: dict[str, pd.DataFrame] = {}

    for side_key in BATTER_SIDE_LABELS:
        summary_frame = pitcher_summary_by_hand.loc[pitcher_summary_by_hand["batter_side_key"] == side_key]
        if not summary_frame.empty:
            export_sections.append(
                {
                    "title": f"{team_label} Summary {BATTER_SIDE_LABELS[side_key]}",
                    "frame": summary_frame[_existing_columns(summary_frame, PITCHER_SUMMARY_COLUMNS)],
                    "lower_is_better": PITCHER_LOWER_IS_BETTER,
                    "higher_is_better": PITCHER_HIGHER_IS_BETTER,
                }
            )

        side_arsenal = sort_arsenal_frame(pitcher_arsenal) if side_key == "all" else sort_arsenal_frame(
            pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key]
        )
        hand_grids[side_key] = side_arsenal[ARSENAL_COLUMNS].copy() if not side_arsenal.empty else pd.DataFrame(columns=ARSENAL_COLUMNS)
        if not hand_grids[side_key].empty:
            export_sections.append(
                {
                    "title": f"{team_label} Arsenal {BATTER_SIDE_LABELS[side_key]}",
                    "frame": hand_grids[side_key],
                    "lower_is_better": {"hard_hit_pct", "xwoba_con"},
                    "higher_is_better": {"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                }
            )

        side_count = pitcher_count_usage.loc[pitcher_count_usage["batter_side_key"] == side_key]
        if side_key == "all":
            side_usage = pitcher_arsenal[["pitch_name", "usage_pct"]] if not pitcher_arsenal.empty else pd.DataFrame(columns=["pitch_name", "usage_pct"])
        else:
            side_usage = pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key, ["pitch_name", "usage_pct"]]
        count_frame = pivot_count_usage(side_count, side_usage)
        count_grids[side_key] = count_frame[COUNT_USAGE_COLUMNS].copy() if not count_frame.empty else pd.DataFrame(columns=COUNT_USAGE_COLUMNS)
        if not count_grids[side_key].empty:
            export_sections.append(
                {
                    "title": f"{team_label} Count Usage {BATTER_SIDE_LABELS[side_key]}",
                    "frame": count_grids[side_key],
                    "higher_is_better": set(COUNT_BUCKET_ORDER),
                }
            )

    active_table = st.radio(
        "Pitcher Table",
        ["Summary", "Arsenal", "Count Usage", "Pitch Shape"],
        horizontal=True,
        key=f"pitcher-table-{game_pk}-{team_label}",
        label_visibility="collapsed",
    )

    if active_table == "Summary":
        summary_table = build_pitcher_summary_table(pitcher_summary_by_hand)
        if summary_table.empty:
            st.info("No summary data available.")
        else:
            render_metric_grid(
                summary_table,
                key=f"summary-{game_pk}-{team_label}",
                height=132,
                lower_is_better=PITCHER_LOWER_IS_BETTER,
                higher_is_better=PITCHER_HIGHER_IS_BETTER,
                use_lightweight=True,
            )
    else:
        active_side_label = st.radio(
            "Pitcher Split",
            [BATTER_SIDE_LABELS[key] for key in BATTER_SIDE_LABELS],
            horizontal=True,
            key=f"pitcher-side-{game_pk}-{team_label}-{active_table.lower().replace(' ', '-')}",
            label_visibility="collapsed",
        )
        active_side_key = next(key for key, label in BATTER_SIDE_LABELS.items() if label == active_side_label)
        if active_table == "Arsenal":
            active_grid = hand_grids[active_side_key]
            if active_grid.empty:
                st.info("No arsenal data available.")
            else:
                arsenal_grid = render_metric_grid(
                    active_grid,
                    key=f"arsenal-{game_pk}-{team_label}-{active_side_key}",
                    height=250,
                    lower_is_better={"hard_hit_pct", "xwoba_con"},
                    higher_is_better={"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                    use_lightweight=True,
                )
                hand_grids[active_side_key] = arsenal_grid
        elif active_table == "Count Usage":
            active_grid = count_grids[active_side_key]
            if active_grid.empty:
                st.info("No count-state usage data available.")
            else:
                count_grids[active_side_key] = render_metric_grid(
                    active_grid,
                    key=f"count-{game_pk}-{team_label}-{active_side_key}",
                    height=250,
                    higher_is_better=set(COUNT_BUCKET_ORDER),
                    use_lightweight=True,
                )
    if active_table == "Pitch Shape":
        _render_pitch_shape_context(
            game_pk,
            team_label,
            pitcher_row,
            movement_arsenal,
            family_context,
            opposing_hitters,
            batter_family_zone_profiles,
            pitcher_family_zone_context,
        )

    return export_sections, summary_all, arsenal_grid, hand_grids, count_grids


def _render_top_sections(
    selected_games: list[dict],
    hitters_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    pitchers_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    hitter_preset: str,
    best_matchups_by_game: dict[int, pd.DataFrame] | None = None,
    full_slate_export_bundles: list[dict] | None = None,
) -> None:
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
    all_pitchers = pd.concat(pitcher_rows, ignore_index=True, sort=False) if pitcher_rows else pd.DataFrame()
    ranked_pitchers = add_pitcher_rank_score(all_pitchers) if not all_pitchers.empty else pd.DataFrame()
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
            "top-matchups-export",
            "Top Slate Export",
            export_options,
            full_slate_export_bundles,
        )
        render_metric_grid(
            ranked_hitters[["game"] + [column for column in preset_columns if column in all_hitters.columns]].head(10),
            key="top-slate-hitters",
            height=320,
            use_lightweight=True,
        )

    st.header("Top Slate Pitchers")
    if all_pitchers.empty:
        st.info("No pitcher data available for this slate.")
    else:
        render_metric_grid(
            ranked_pitchers[_existing_columns(ranked_pitchers, TOP_PITCHER_COLUMNS)].head(10),
            key="top-slate-pitchers",
            height=320,
            lower_is_better=PITCHER_LOWER_IS_BETTER,
            higher_is_better=PITCHER_HIGHER_IS_BETTER,
            use_lightweight=True,
        )


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
) -> None:
    if pitcher_row.empty:
        st.info("No pitch-shape context available.")
        return
    pitcher_id = int(pitcher_row["pitcher_id"].iloc[0])
    movement_detail = movement_arsenal.loc[movement_arsenal["pitcher_id"] == pitcher_id].copy() if not movement_arsenal.empty else pd.DataFrame()
    family_detail = family_context.loc[family_context["pitcher_id"] == pitcher_id].copy() if not family_context.empty else pd.DataFrame()
    family_board = _build_family_fit_board(opposing_hitters, batter_family_zone_profiles, pitcher_family_zone_context, pitcher_id)

    st.markdown(f"##### {team_label} Pitch Shape")
    render_metric_grid(
        movement_detail[[column for column in MOVEMENT_ARSENAL_COLUMNS if column in movement_detail.columns]],
        key=f"movement-arsenal-{game_pk}-{team_label}",
        height=230,
        higher_is_better={"usage_rate", "avg_velocity", "avg_spin_rate", "avg_extension", "avg_pfx_x", "avg_pfx_z"},
        use_lightweight=True,
    ) if not movement_detail.empty else st.info("No weighted movement arsenal data available.")
    render_metric_grid(
        family_detail[[column for column in FAMILY_ZONE_CONTEXT_COLUMNS if column in family_detail.columns]],
        key=f"family-context-{game_pk}-{team_label}",
        height=210,
        lower_is_better={"prior_weight_share", "damage_allowed_rate", "xwoba_allowed"},
        higher_is_better={"usage_rate_overall", "whiff_rate", "called_strike_rate"},
        use_lightweight=True,
    ) if not family_detail.empty else st.info("No weighted family-zone profile available.")
    render_metric_grid(
        family_board.head(6),
        key=f"family-fit-board-{game_pk}-{team_label}",
        height=210,
        higher_is_better={"family_fit_score", "matchup_score", "ceiling_score", "xwoba"},
        use_lightweight=True,
    ) if not family_board.empty else st.info("No opposing-hitter family fit context available.")


def _build_pitcher_export_sections(
    team_label: str,
    pitcher_summary_by_hand: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    pitcher_by_hand: pd.DataFrame,
    pitcher_count_usage: pd.DataFrame,
) -> list[dict]:
    export_sections: list[dict] = []
    for side_key, side_label in BATTER_SIDE_LABELS.items():
        summary_frame = pitcher_summary_by_hand.loc[pitcher_summary_by_hand["batter_side_key"] == side_key]
        if not summary_frame.empty:
            export_sections.append(
                {
                    "title": f"{team_label} Summary {side_label}",
                    "frame": summary_frame[_existing_columns(summary_frame, PITCHER_SUMMARY_COLUMNS)],
                    "lower_is_better": PITCHER_LOWER_IS_BETTER,
                    "higher_is_better": PITCHER_HIGHER_IS_BETTER,
                }
            )

        side_arsenal = sort_arsenal_frame(pitcher_arsenal) if side_key == "all" else sort_arsenal_frame(
            pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key]
        )
        if not side_arsenal.empty:
            export_sections.append(
                {
                    "title": f"{team_label} Arsenal {side_label}",
                    "frame": side_arsenal[ARSENAL_COLUMNS],
                    "lower_is_better": {"hard_hit_pct", "xwoba_con"},
                    "higher_is_better": {"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                }
            )

        side_count = pitcher_count_usage.loc[pitcher_count_usage["batter_side_key"] == side_key]
        if side_key == "all":
            side_usage = pitcher_arsenal[["pitch_name", "usage_pct"]] if not pitcher_arsenal.empty else pd.DataFrame(columns=["pitch_name", "usage_pct"])
        else:
            side_usage = pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key, ["pitch_name", "usage_pct"]]
        count_frame = pivot_count_usage(side_count, side_usage)
        if not count_frame.empty:
            export_sections.append(
                {
                    "title": f"{team_label} Count Usage {side_label}",
                    "frame": count_frame[COUNT_USAGE_COLUMNS],
                    "higher_is_better": set(COUNT_BUCKET_ORDER),
                }
            )
    return export_sections


def main() -> None:
    st.set_page_config(page_title="MLB Local Explorer", layout="wide")
    st.title("MLB Local Explorer")
    perf_events: list[tuple[str, float]] = []
    config = AppConfig()
    top_level_view = st.sidebar.radio("View", ["Slate Explorer", "Backtesting"], key="local-top-view")
    if top_level_view == "Backtesting":
        render_backtesting_tab(config)
        return
    engine = StatcastQueryEngine(config)
    target_date, filters, hitter_preset = _sidebar(config)
    slate = engine.load_daily_slate(target_date)
    if not slate:
        st.error(f"No built slate found for {target_date.isoformat()}. Run the build command for that date first.")
        return
    rosters = engine.load_daily_rosters(target_date)
    selected_games = _game_selection(slate)
    active_sections = {st.session_state.get(f"section-local-{game['game_pk']}", "Matchup") for game in selected_games}
    st.caption(f"Showing {len(selected_games)} of {len(slate)} games")
    st.caption("PulledBrl% tracks pulled barrels on tracked batted-ball events. Brl/BIP% uses all balls in play.")

    hitters_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitchers_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    best_matchups_by_game: dict[int, pd.DataFrame] = {}
    pitcher_summary_by_hand_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitcher_arsenal_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitcher_by_hand_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitcher_count_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    base_load_start = perf_counter()
    batter_zone_profiles = engine.load_daily_batter_zone_profiles(target_date)
    pitcher_zone_profiles = engine.load_daily_pitcher_zone_profiles(target_date)
    hitter_pitcher_exclusions = engine.load_daily_hitter_pitcher_exclusions(target_date)
    _record_perf(perf_events, "core load", base_load_start)

    hitter_rolling = pd.DataFrame()
    pitcher_rolling = pd.DataFrame()
    if "Rolling" in active_sections:
        rolling_start = perf_counter()
        hitter_rolling = engine.load_daily_hitter_rolling(target_date)
        pitcher_rolling = engine.load_daily_pitcher_rolling(target_date)
        _record_perf(perf_events, "rolling load", rolling_start)

    batter_family_zone_profiles = pd.DataFrame()
    pitcher_family_zone_context = pd.DataFrame()
    pitcher_movement_arsenal = pd.DataFrame()
    if "Matchup" in active_sections:
        shape_start = perf_counter()
        batter_family_zone_profiles = engine.load_daily_batter_family_zone_profiles(target_date)
        pitcher_family_zone_context = engine.load_daily_pitcher_family_zone_context(target_date)
        pitcher_movement_arsenal = engine.load_daily_pitcher_movement_arsenal(target_date)
        _record_perf(perf_events, "pitch-shape load", shape_start)

    valid_teams = tuple(sorted({game["away_team"] for game in slate} | {game["home_team"] for game in slate}))
    try:
        rotowire_start = perf_counter()
        rotowire_lineups = resolve_rotowire_lineups(fetch_rotowire_lineups(target_date, valid_teams), rosters)
        _record_perf(perf_events, "rotowire", rotowire_start)
    except Exception:
        rotowire_lineups = {}
    batter_join_col = "batter_id" if "batter_id" in batter_zone_profiles.columns else "batter"
    pitcher_join_col = "pitcher_id" if "pitcher_id" in pitcher_zone_profiles.columns else "pitcher"
    batter_zone_named = pd.DataFrame()
    if {"Pitcher Zones", "Hitter Zones"} & active_sections:
        zone_named_start = perf_counter()
        roster_lookup = rosters[["team", "player_id", "player_name"]].drop_duplicates("player_id")
        batter_zone_named = batter_zone_profiles.merge(roster_lookup, left_on=batter_join_col, right_on="player_id", how="left")
        _record_perf(perf_events, "zone merge", zone_named_start)

    game_prep_start = perf_counter()
    for game in selected_games:
        pitcher_ids = [pitcher_id for pitcher_id in [game.get("away_probable_pitcher_id"), game.get("home_probable_pitcher_id")] if pitcher_id]
        pitchers = engine.get_pitcher_cards(pitcher_ids, filters)
        pitcher_summaries_by_hand = engine.get_pitcher_summary_by_hand(pitcher_ids, filters)
        arsenals = engine.get_pitcher_arsenal(pitcher_ids, filters)
        arsenals_by_hand = engine.get_pitcher_arsenal_by_hand(pitcher_ids, filters)
        usages_by_count = engine.get_pitcher_usage_by_count(pitcher_ids, filters)

        away_pitcher = pitchers.loc[pitchers["pitcher_id"] == game.get("away_probable_pitcher_id")].copy()
        home_pitcher = pitchers.loc[pitchers["pitcher_id"] == game.get("home_probable_pitcher_id")].copy()
        away_hand = home_pitcher["p_throws"].iloc[0] if not home_pitcher.empty else None
        home_hand = away_pitcher["p_throws"].iloc[0] if not away_pitcher.empty else None
        away_roster_ids = rosters.loc[rosters["team"] == game["away_team"], "player_id"].dropna().astype(int).unique().tolist() if not rosters.empty else []
        home_roster_ids = rosters.loc[rosters["team"] == game["home_team"], "player_id"].dropna().astype(int).unique().tolist() if not rosters.empty else []

        away_hitters = apply_roster_names(engine.get_team_hitter_pool(game["away_team"], away_hand, filters, away_roster_ids), rosters, game["away_team"])
        home_hitters = apply_roster_names(engine.get_team_hitter_pool(game["home_team"], home_hand, filters, home_roster_ids), rosters, game["home_team"])
        away_hitters = filter_excluded_pitchers_from_hitter_pool(away_hitters, hitter_pitcher_exclusions)
        home_hitters = filter_excluded_pitchers_from_hitter_pool(home_hitters, hitter_pitcher_exclusions)
        away_hitters = apply_projected_lineup(away_hitters, game["away_team"], rotowire_lineups)
        home_hitters = apply_projected_lineup(home_hitters, game["home_team"], rotowire_lineups)
        away_hitters = add_hitter_matchup_score(
            away_hitters,
            batter_zone_profiles=batter_zone_profiles,
            pitcher_zone_profiles=pitcher_zone_profiles,
            opposing_pitcher_id=game.get("home_probable_pitcher_id"),
            opposing_pitcher_hand=away_hand,
        )
        home_hitters = add_hitter_matchup_score(
            home_hitters,
            batter_zone_profiles=batter_zone_profiles,
            pitcher_zone_profiles=pitcher_zone_profiles,
            opposing_pitcher_id=game.get("away_probable_pitcher_id"),
            opposing_pitcher_hand=home_hand,
        )

        hitters_by_game[game["game_pk"]] = (away_hitters, home_hitters)
        best_matchups_by_game[game["game_pk"]] = build_best_matchups(away_hitters, home_hitters)
        pitchers_by_game[game["game_pk"]] = (away_pitcher, home_pitcher)
        pitcher_summary_by_hand_map[game["game_pk"]] = (
            pitcher_summaries_by_hand.loc[pitcher_summaries_by_hand["pitcher_id"] == game.get("away_probable_pitcher_id")].copy(),
            pitcher_summaries_by_hand.loc[pitcher_summaries_by_hand["pitcher_id"] == game.get("home_probable_pitcher_id")].copy(),
        )
        pitcher_arsenal_map[game["game_pk"]] = (
            arsenals.loc[arsenals["pitcher_id"] == game.get("away_probable_pitcher_id")].copy(),
            arsenals.loc[arsenals["pitcher_id"] == game.get("home_probable_pitcher_id")].copy(),
        )
        pitcher_by_hand_map[game["game_pk"]] = (
            arsenals_by_hand.loc[arsenals_by_hand["pitcher_id"] == game.get("away_probable_pitcher_id")].copy(),
            arsenals_by_hand.loc[arsenals_by_hand["pitcher_id"] == game.get("home_probable_pitcher_id")].copy(),
        )
        pitcher_count_map[game["game_pk"]] = (
            usages_by_count.loc[usages_by_count["pitcher_id"] == game.get("away_probable_pitcher_id")].copy(),
            usages_by_count.loc[usages_by_count["pitcher_id"] == game.get("home_probable_pitcher_id")].copy(),
        )
    _record_perf(perf_events, "game prep", game_prep_start)

    export_options_by_game: dict[int, dict[str, list[dict]]] = {}
    for game in selected_games:
        away_hitters, home_hitters = hitters_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_summary_by_hand, home_summary_by_hand = pitcher_summary_by_hand_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_arsenal, home_arsenal = pitcher_arsenal_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_by_hand, home_by_hand = pitcher_by_hand_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_count, home_count = pitcher_count_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_export_sections = _build_pitcher_export_sections(game["away_team"], away_summary_by_hand, away_arsenal, away_by_hand, away_count)
        home_export_sections = _build_pitcher_export_sections(game["home_team"], home_summary_by_hand, home_arsenal, home_by_hand, home_count)
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

    _render_top_sections(
        selected_games,
        hitters_by_game,
        pitchers_by_game,
        hitter_preset,
        best_matchups_by_game,
        full_slate_export_bundles,
    )
    _render_perf(perf_events)
    st.divider()

    hitter_columns = hitter_columns_for_preset(hitter_preset)

    for idx, game in enumerate(selected_games):
        away_hitters, home_hitters = hitters_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_pitcher, home_pitcher = pitchers_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_summary_by_hand, home_summary_by_hand = pitcher_summary_by_hand_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_arsenal, home_arsenal = pitcher_arsenal_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_by_hand, home_by_hand = pitcher_by_hand_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_count, home_count = pitcher_count_map.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        best_matchups = best_matchups_by_game.get(game["game_pk"], pd.DataFrame()).copy()
        away_export_sections = _build_pitcher_export_sections(game["away_team"], away_summary_by_hand, away_arsenal, away_by_hand, away_count)
        home_export_sections = _build_pitcher_export_sections(game["home_team"], home_summary_by_hand, home_arsenal, home_by_hand, home_count)

        with st.expander(f"{game['away_team']} @ {game['home_team']}", expanded=idx == 0):
            render_matchup_header(game)
            active_section = st.radio(
                "Section",
                ["Matchup", "Rolling", "Pitcher Zones", "Hitter Zones"],
                horizontal=True,
                key=f"section-local-{game['game_pk']}",
                label_visibility="collapsed",
            )
            if active_section == "Matchup":
                st.markdown("#### Best Matchups")
                best_matchups = render_metric_grid(
                    best_matchups[BEST_MATCHUP_COLUMNS],
                    key=f"best-{game['game_pk']}",
                    height=170,
                    use_lightweight=True,
                )

                st.markdown("#### Pitchers")
                pitcher_cols = st.columns(2)
                with pitcher_cols[0]:
                    st.markdown(f"##### {game['away_team']} starter")
                    _render_pitcher_tab(
                        game["game_pk"],
                        game["away_team"],
                        away_summary_by_hand,
                        away_arsenal,
                        away_by_hand,
                        away_count,
                        away_pitcher,
                        pitcher_movement_arsenal,
                        pitcher_family_zone_context,
                        home_hitters,
                        batter_family_zone_profiles,
                        pitcher_family_zone_context,
                    )
                with pitcher_cols[1]:
                    st.markdown(f"##### {game['home_team']} starter")
                    _render_pitcher_tab(
                        game["game_pk"],
                        game["home_team"],
                        home_summary_by_hand,
                        home_arsenal,
                        home_by_hand,
                        home_count,
                        home_pitcher,
                        pitcher_movement_arsenal,
                        pitcher_family_zone_context,
                        away_hitters,
                        batter_family_zone_profiles,
                        pitcher_family_zone_context,
                    )

                st.markdown("#### Hitters")
                hitter_cols = st.columns(2)
                with hitter_cols[0]:
                    st.caption(f"{game['away_team']} vs {game.get('home_probable_pitcher_name') or 'opposing starter'}")
                    away_hitters = render_metric_grid(
                        away_hitters[[column for column in hitter_columns if column in away_hitters.columns]],
                        key=f"away-hitters-{game['game_pk']}",
                        height=360,
                        use_lightweight=True,
                    )
                with hitter_cols[1]:
                    st.caption(f"{game['home_team']} vs {game.get('away_probable_pitcher_name') or 'opposing starter'}")
                    home_hitters = render_metric_grid(
                        home_hitters[[column for column in hitter_columns if column in home_hitters.columns]],
                        key=f"home-hitters-{game['game_pk']}",
                        height=360,
                        use_lightweight=True,
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
                            st.markdown("##### Hitters")
                            hitter_frame = hitter_rolling.loc[
                                hitter_rolling["rolling_window"].eq(label)
                                & hitter_rolling["player_name"].isin(sorted(away_hitter_names | home_hitter_names))
                            ]
                            render_metric_grid(
                                hitter_frame[[column for column in HITTER_ROLLING_COLUMNS if column in hitter_frame.columns]],
                                key=f"h-roll-{game['game_pk']}-{label}",
                                height=260,
                                use_lightweight=True,
                            )
                        with cols[1]:
                            st.markdown("##### Pitchers")
                            pitcher_frame = pitcher_rolling.loc[
                                pitcher_rolling["rolling_window"].eq(label)
                                & pitcher_rolling["player_name"].isin([name for name in [away_pitcher_name, home_pitcher_name] if name])
                            ]
                            render_metric_grid(
                                pitcher_frame[[column for column in PITCHER_ROLLING_COLUMNS if column in pitcher_frame.columns]],
                                key=f"p-roll-{game['game_pk']}-{label}",
                                height=260,
                                lower_is_better=PITCHER_LOWER_IS_BETTER | {"barrel_bip_pct"},
                                higher_is_better=PITCHER_HIGHER_IS_BETTER | {"avg_release_speed"},
                                use_lightweight=True,
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
                                st.selectbox("Overlay hitter", opponent_names, key=f"p-zone-hitter-{game['game_pk']}-{team_label}")
                                if opponent_names
                                else None
                            )
                            hitter_detail = batter_zone_named.loc[batter_zone_named["player_name"] == selected_hitter].copy() if selected_hitter else pd.DataFrame()
                            pitch_types = ["All pitches"] + sorted(
                                set(zone_frame.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                                | set(hitter_detail.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                            )
                            selected_pitch = st.selectbox("Pitch type", pitch_types, key=f"p-zone-pitch-{game['game_pk']}-{team_label}")
                            pitcher_map = aggregate_pitcher_zone_map(zone_frame, selected_pitch)
                            hitter_map = aggregate_batter_zone_map(hitter_detail, selected_pitch)
                            overlay_map = build_zone_overlay_map(hitter_map, pitcher_map)
                            heatmap_cols = st.columns(2)
                            with heatmap_cols[0]:
                                render_zone_tool(
                                    f"{pitcher_row['pitcher_name'].iloc[0]} Usage",
                                    f"{selected_pitch} | Pitcher zone attack",
                                    pitcher_map,
                                    key=f"p-zone-map-{game['game_pk']}-{team_label}-{selected_pitch}",
                                    map_kind="pitcher",
                                )
                            with heatmap_cols[1]:
                                render_zone_tool(
                                    f"Overlay vs {selected_hitter or 'Opposing Hitter'}",
                                    f"{selected_pitch} | Hitter damage x pitcher usage",
                                    overlay_map,
                                    key=f"p-zone-overlay-{game['game_pk']}-{team_label}-{selected_pitch}",
                                    map_kind="overlay",
                                )
                            render_metric_grid(
                                zone_frame[[column for column in PITCHER_ZONE_COLUMNS if column in zone_frame.columns]],
                                key=f"p-zone-{game['game_pk']}-{team_label}",
                                height=240,
                                higher_is_better={"usage_rate"},
                                use_lightweight=True,
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
                            st.selectbox("Hitter", hitter_options, key=f"h-zone-hitter-{game['game_pk']}-{team_label}")
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
                        selected_pitch = st.selectbox("Pitch type", pitch_types, key=f"h-zone-pitch-{game['game_pk']}-{team_label}")
                        hitter_map = aggregate_batter_zone_map(hitter_detail, selected_pitch)
                        pitcher_map = aggregate_pitcher_zone_map(pitcher_detail, selected_pitch)
                        overlay_map = build_zone_overlay_map(hitter_map, pitcher_map)
                        heatmap_cols = st.columns(2)
                        with heatmap_cols[0]:
                            render_zone_tool(
                                f"{selected_hitter or team_label} Damage",
                                f"{selected_pitch} | Hitter zone quality",
                                hitter_map,
                                key=f"h-zone-map-{game['game_pk']}-{team_label}-{selected_pitch}",
                                map_kind="hitter",
                            )
                        with heatmap_cols[1]:
                            opposing_name = opposing_pitcher["pitcher_name"].iloc[0] if not opposing_pitcher.empty else "Opposing Pitcher"
                            render_zone_tool(
                                f"Overlay vs {opposing_name}",
                                f"{selected_pitch} | Hitter damage x pitcher usage",
                                overlay_map,
                                key=f"h-zone-overlay-{game['game_pk']}-{team_label}-{selected_pitch}",
                                map_kind="overlay",
                            )
                        render_metric_grid(
                            hitter_detail[[column for column in BATTER_ZONE_COLUMNS if column in hitter_detail.columns]],
                            key=f"h-zone-{game['game_pk']}-{team_label}",
                            height=240,
                            higher_is_better={"hit_rate", "hr_rate", "damage_rate"},
                            use_lightweight=True,
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
                key=f"export-{game['game_pk']}",
                title=f"{game['away_team']} @ {game['home_team']}",
                export_options=export_options,
            )
        st.divider()
