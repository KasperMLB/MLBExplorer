from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .cockroach_loader import read_hitter_backtest_data, read_pitcher_backtest_data, read_prop_odds_history
from .config import AppConfig
from .odds_service import american_to_implied_prob
from .ui_components import render_metric_grid

try:
    import altair as alt

    HAS_ALTAIR = True
except ImportError:  # pragma: no cover
    alt = None
    HAS_ALTAIR = False


BACKTEST_PRESETS = {
    "Since Tracking Began": "tracking_start",
    "Season to date": None,
    "Last 7": 7,
    "Last 14": 14,
    "Last 30": 30,
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


def _normalize_key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _render_simple_chart(data: pd.DataFrame, x: str, y: str, chart_type: str, title: str) -> None:
    if data.empty:
        st.info(f"No data for {title.lower()}.")
        return
    if HAS_ALTAIR:
        if chart_type == "line":
            chart = alt.Chart(data).mark_line(point=True).encode(x=alt.X(x), y=alt.Y(y))
        elif chart_type == "scatter":
            chart = alt.Chart(data).mark_circle(size=55, opacity=0.65).encode(x=alt.X(x), y=alt.Y(y), tooltip=list(data.columns))
        else:
            chart = alt.Chart(data).mark_bar().encode(x=alt.X(x, sort=None), y=alt.Y(y))
        st.altair_chart(chart.properties(height=280, title=title), use_container_width=True)
        return
    chart_frame = data[[x, y]].copy()
    if chart_type == "line":
        st.line_chart(chart_frame.set_index(x))
    elif chart_type == "scatter":
        st.scatter_chart(chart_frame, x=x, y=y)
    else:
        st.bar_chart(chart_frame.set_index(x))


def _render_metric_cards(cards: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(cards))
    for col, (label, value, delta) in zip(cols, cards):
        col.metric(label, value, delta)


def _filters() -> dict[str, object]:
    st.subheader("Backtesting")
    row_one = st.columns([1.2, 1.0, 1.0, 1.0, 1.0, 0.9])
    mode = row_one[0].radio("Mode", ["Slate Audit", "Historical Backtest"], horizontal=True, index=1)
    entity = row_one[1].radio("Entity", ["Hitters", "Pitchers"], horizontal=True)
    split_key = row_one[2].selectbox("Split", ["overall", "vs_rhp", "vs_lhp", "home", "away"], index=0)
    recent_window = row_one[3].selectbox("Recent Window", ["season", "last_45_days", "last_14_days"], index=0)
    weighted_mode = row_one[4].radio("Weighting", ["weighted", "unweighted"], horizontal=True)
    score_choice = (
        HITTER_SCORE_OPTIONS[row_one[5].selectbox("Score", list(HITTER_SCORE_OPTIONS.keys()), index=0)]
        if entity == "Hitters"
        else PITCHER_SCORE_OPTIONS[row_one[5].selectbox("Score", list(PITCHER_SCORE_OPTIONS.keys()), index=1)]
    )

    today = date.today()
    season_start = date(today.year, 1, 1)
    if mode == "Slate Audit":
        audit_date = st.date_input("Audit Date", value=today, key="bt-audit-date")
        start_date = audit_date
        end_date = audit_date
    else:
        row_two = st.columns(3)
        preset = row_two[0].selectbox("Date Preset", list(BACKTEST_PRESETS.keys()), index=0)
        if preset == "Custom":
            start_date = row_two[1].date_input("Start date", value=today - timedelta(days=14), key="bt-start")
            end_date = row_two[2].date_input("End date", value=today, key="bt-end")
        else:
            days = BACKTEST_PRESETS[preset]
            end_date = row_two[2].date_input("End date", value=today, key="bt-end-historical")
            start_date = date(2026, 1, 1) if days == "tracking_start" else season_start if days is None else end_date - timedelta(days=int(days) - 1)
            row_two[1].date_input("Start date", value=start_date, disabled=True, key="bt-start-fixed")

    row_three = st.columns(4)
    top_n = row_three[0].selectbox("Top Cohort", [3, 5, 10, 15, 20], index=2)
    bucket_count = row_three[1].selectbox("Score Buckets", [5, 10, 20], index=1)
    min_rows = row_three[2].slider("Min Rows", 1, 100, 5)
    show_ladder = row_three[3].checkbox("Show Ladder Lens", value=(entity == "Hitters" and score_choice == "ceiling_score"))
    return {
        "mode": mode,
        "entity": entity,
        "start_date": start_date,
        "end_date": end_date,
        "split_key": split_key,
        "recent_window": recent_window,
        "weighted_mode": weighted_mode,
        "score_choice": score_choice,
        "top_n": int(top_n),
        "bucket_count": int(bucket_count),
        "min_rows": int(min_rows),
        "show_ladder": bool(show_ladder),
    }


def _render_explainer(entity: str, mode: str) -> None:
    history_text = "selected slate" if mode == "Slate Audit" else "graded forward-tracked history through the selected end date"
    if entity == "Hitters":
        st.markdown(
            f"""
**What This Page Is Measuring**

- This page audits **pregame hitter model rows** against what actually happened on the {history_text}.
- `Matchup Score` is the main single-HR lens. `Ceiling Score` is the ladder / 2+ HR lens.
- `Captured Winners`: hitter homered and ranked inside your chosen Top cohort.
- `Missed Winners`: hitter homered but ranked outside your Top cohort.
- `False Positives`: hitter ranked highly but did not homer.
- `CLV`: closing implied probability minus captured implied probability. Positive means your captured price beat the stored close.
- Profitability is the goal, so this page is checking both **result capture** and **price quality**.
            """
        )
    else:
        st.markdown(
            f"""
**What This Page Is Measuring**

- This page audits **pregame pitcher model rows** plus **all captured over strikeout lines including alt lines** on the {history_text}.
- `Strikeout Score` is the primary K-upside lens. `Pitch Score` is the broader quality lens.
- `Captured K Props`: line cleared and pitcher ranked inside your Top cohort.
- `Missed K Props`: line cleared but pitcher ranked too low.
- `False Positives`: model liked the pitcher but the captured line did not clear.
- The page is trying to tell you whether the model is finding the right arms, beating close, and actually making money.
            """
        )


def _load_data(config: AppConfig, entity: str, start_date: date, end_date: date, split_key: str, recent_window: str, weighted_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if entity == "Hitters":
        snapshots, outcomes, _ = read_hitter_backtest_data(config, start_date, end_date, split_key, recent_window, weighted_mode)
    else:
        snapshots, outcomes, _ = read_pitcher_backtest_data(config, start_date, end_date, split_key, recent_window, weighted_mode)
    odds = read_prop_odds_history(config, start_date, end_date)
    return snapshots, outcomes, odds


def _prepare_hitter_frame(snapshots: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    frame = snapshots.merge(outcomes, on=["slate_date", "game_pk", "team", "batter_id"], how="left", suffixes=("", "_outcome")).copy()
    frame["slate_date"] = pd.to_datetime(frame["slate_date"], errors="coerce")
    frame["hitter_name"] = frame["hitter_name"].fillna(frame.get("hitter_name_outcome"))
    frame["name_key"] = _normalize_key(frame["hitter_name"])
    outcome_status = frame.get("outcome_status")
    frame["outcome_status"] = outcome_status.fillna(pd.NA) if outcome_status is not None else pd.NA
    outcome_complete = frame.get("outcome_complete")
    frame["outcome_complete"] = outcome_complete.fillna(pd.NA) if outcome_complete is not None else pd.NA
    frame["source_max_event_date"] = pd.to_datetime(frame.get("source_max_event_date"), errors="coerce")
    for column in ["matchup_score", "ceiling_score", "zone_fit_score", "xwoba", "xwoba_con", "swstr_pct", "pulled_barrel_pct", "sweet_spot_pct", "avg_launch_angle", "hard_hit_pct", "hits", "home_runs", "total_bases", "runs", "rbi"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["hit_flag"] = frame.get("hits", 0).fillna(0).gt(0).astype(float)
    frame["home_run_flag"] = frame.get("home_runs", 0).fillna(0).gt(0).astype(float)
    frame["hr2_flag"] = frame.get("home_runs", 0).fillna(0).ge(2).astype(float)
    frame["runs_rbi"] = pd.to_numeric(frame.get("runs"), errors="coerce").fillna(0) + pd.to_numeric(frame.get("rbi"), errors="coerce").fillna(0)
    return frame


def _prepare_pitcher_frame(snapshots: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    frame = snapshots.merge(outcomes, on=["slate_date", "game_pk", "team", "pitcher_id"], how="left", suffixes=("", "_outcome")).copy()
    frame["slate_date"] = pd.to_datetime(frame["slate_date"], errors="coerce")
    frame["pitcher_name"] = frame["pitcher_name"].fillna(frame.get("pitcher_name_outcome"))
    frame["name_key"] = _normalize_key(frame["pitcher_name"])
    outcome_status = frame.get("outcome_status")
    frame["outcome_status"] = outcome_status.fillna(pd.NA) if outcome_status is not None else pd.NA
    outcome_complete = frame.get("outcome_complete")
    frame["outcome_complete"] = outcome_complete.fillna(pd.NA) if outcome_complete is not None else pd.NA
    frame["source_max_event_date"] = pd.to_datetime(frame.get("source_max_event_date"), errors="coerce")
    for column in ["pitcher_score", "strikeout_score", "xwoba", "called_strike_pct", "csw_pct", "swstr_pct", "putaway_pct", "ball_pct", "siera", "strikeouts", "batters_faced", "hits_allowed", "runs_allowed"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["k5_flag"] = frame.get("strikeouts", 0).fillna(0).ge(5).astype(float)
    return frame


def _legacy_stale_dates(frame: pd.DataFrame) -> set[pd.Timestamp]:
    if frame.empty or "slate_date" not in frame.columns:
        return set()
    work = frame.copy()
    work["slate_date"] = pd.to_datetime(work["slate_date"], errors="coerce")
    work = work.loc[work["slate_date"].notna()].copy()
    if work.empty:
        return set()
    if "home_runs" in work.columns:
        realized = pd.to_numeric(work.get("home_runs"), errors="coerce").fillna(0)
    elif "strikeouts" in work.columns:
        realized = pd.to_numeric(work.get("strikeouts"), errors="coerce").fillna(0)
    else:
        return set()
    work["realized_total_value"] = realized
    status_series = work.get("outcome_status", pd.Series(pd.NA, index=work.index, dtype="object"))
    complete_series = work.get("outcome_complete", pd.Series(pd.NA, index=work.index, dtype="object"))
    source_date = pd.to_datetime(work.get("source_max_event_date"), errors="coerce")
    summary = (
        pd.DataFrame(
            {
                "slate_date": work["slate_date"],
                "status_null": status_series.isna(),
                "complete_null": complete_series.isna(),
                "source_null": source_date.isna(),
                "realized_total_value": work["realized_total_value"],
            }
        )
        .groupby("slate_date", as_index=False)
        .agg(
            all_status_null=("status_null", "all"),
            all_complete_null=("complete_null", "all"),
            all_source_null=("source_null", "all"),
            realized_total=("realized_total_value", "sum"),
        )
    )
    if summary.empty:
        return set()
    recent_cutoff = summary["slate_date"].max() - timedelta(days=14)
    stale_dates = summary.loc[
        summary["slate_date"].ge(recent_cutoff)
        & summary["all_status_null"]
        & summary["all_complete_null"]
        & summary["all_source_null"]
        & summary["realized_total"].eq(0),
        "slate_date",
    ]
    return set(stale_dates.tolist())


def _graded_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype="bool")
    work = frame.copy()
    work["slate_date"] = pd.to_datetime(work["slate_date"], errors="coerce")
    status_series = work.get("outcome_status", pd.Series(pd.NA, index=work.index, dtype="object"))
    complete_series = work.get("outcome_complete", pd.Series(pd.NA, index=work.index, dtype="object"))
    explicit_dates = set(work.loc[status_series.notna() | complete_series.notna(), "slate_date"].dropna().unique().tolist())
    stale_dates = _legacy_stale_dates(work)
    mask = pd.Series(True, index=frame.index, dtype="bool")
    if stale_dates:
        mask.loc[work["slate_date"].isin(stale_dates)] = False
    if explicit_dates:
        explicit_rows = work["slate_date"].isin(explicit_dates)
        mask.loc[explicit_rows] = complete_series.loc[explicit_rows].fillna(False).astype(bool)
    return mask


def _render_outcome_status(frame: pd.DataFrame, entity: str) -> pd.Series:
    graded_mask = _graded_mask(frame)
    if frame.empty:
        return graded_mask
    stale_dates = sorted(day.date() for day in _legacy_stale_dates(frame))
    explicit = frame.loc[
        frame.get("outcome_status", pd.Series(pd.NA, index=frame.index, dtype="object")).notna()
        | frame.get("outcome_complete", pd.Series(pd.NA, index=frame.index, dtype="object")).notna(),
        ["slate_date", "outcome_status", "source_max_event_date"],
    ].copy()
    label = "hitter" if entity == "Hitters" else "pitcher"
    if stale_dates:
        shown = ", ".join(day.isoformat() for day in stale_dates[:5])
        if len(stale_dates) > 5:
            shown += ", ..."
        st.warning(
            f"Some {label} outcome rows look stale from earlier builds and are excluded from summaries. "
            f"Excluded legacy-stale dates: {shown}."
        )
    if explicit.empty:
        if not stale_dates:
            graded_dates = frame.loc[graded_mask, "slate_date"].dropna().dt.date.nunique()
            if graded_dates:
                st.caption(f"Using graded results for {graded_dates} date(s) in this view.")
        return graded_mask
    explicit["slate_date"] = pd.to_datetime(explicit["slate_date"], errors="coerce")
    incomplete_dates = explicit.loc[explicit["outcome_status"].ne("complete"), "slate_date"].dropna().dt.date.drop_duplicates().tolist()
    if incomplete_dates:
        latest_source = pd.to_datetime(explicit["source_max_event_date"], errors="coerce").dropna()
        latest_text = latest_source.max().date().isoformat() if not latest_source.empty else "unknown"
        shown = ", ".join(day.isoformat() for day in incomplete_dates[:5])
        if len(incomplete_dates) > 5:
            shown += ", ..."
        st.warning(
            f"Some {label} outcomes are incomplete and excluded from summaries. "
            f"Incomplete dates: {shown}. Latest available live event date: {latest_text}."
        )
    elif not stale_dates:
        graded_dates = frame.loc[graded_mask, "slate_date"].dropna().dt.date.nunique()
        if graded_dates:
            st.caption(f"Using graded results for {graded_dates} date(s) in this view.")
    return graded_mask


def _prepare_odds_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["fetched_at"] = pd.to_datetime(work.get("fetched_at"), errors="coerce", utc=True)
    work["commence_time"] = pd.to_datetime(work.get("commence_time"), errors="coerce", utc=True)
    work["slate_date"] = work["commence_time"].dt.tz_convert("America/Chicago").dt.tz_localize(None).dt.normalize()
    work["name_key"] = _normalize_key(work.get("player_name", pd.Series(dtype="object")))
    work["selection_side"] = work.get("selection_side", pd.Series(dtype="object")).fillna("").astype(str).str.lower()
    work["market_key"] = work.get("market_key", pd.Series(dtype="object")).fillna("").astype(str)
    work["line"] = pd.to_numeric(work.get("line"), errors="coerce")
    work["odds_american"] = pd.to_numeric(work.get("odds_american"), errors="coerce")
    work["implied_prob"] = work["odds_american"].apply(american_to_implied_prob)
    work["decimal_price"] = work["odds_american"].apply(lambda price: None if pd.isna(price) else (1.0 + (price / 100.0) if price > 0 else 1.0 + (100.0 / abs(price))))
    return work


def _add_slate_ranks(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.sort_values(["slate_date", score_column], ascending=[True, False], na_position="last").copy()
    ranked["slate_rank"] = ranked.groupby("slate_date").cumcount() + 1
    ranked["rank_bucket"] = pd.cut(ranked["slate_rank"], bins=[0, 5, 10, 20, 50, 999999], labels=["Top 5", "6-10", "11-20", "21-50", "51+"], include_lowest=True).astype(str)
    return ranked


def _best_snapshot_price(frame: pd.DataFrame, group_columns: list[str], which: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    agg = "min" if which == "first" else "max"
    picked = frame.groupby(group_columns, as_index=False)["fetched_at"].agg(agg).rename(columns={"fetched_at": "picked_at"})
    chosen = frame.merge(picked, on=group_columns, how="inner")
    chosen = chosen.loc[chosen["fetched_at"] == chosen["picked_at"]].copy()
    chosen = chosen.sort_values(["decimal_price", "odds_american"], ascending=[False, False], na_position="last")
    return chosen.groupby(group_columns, as_index=False).head(1).reset_index(drop=True)


def _price_units(price: float | int | None, win_flag: float | int | None) -> float | None:
    if price is None or pd.isna(price):
        return None
    if win_flag and float(win_flag) > 0:
        return float(price) / 100.0 if float(price) > 0 else 100.0 / abs(float(price))
    return -1.0


def _merge_hitter_prices(frame: pd.DataFrame, odds_history: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if odds_history.empty:
        out["captured_price"] = pd.NA
        out["captured_implied_prob"] = pd.NA
        out["closing_price"] = pd.NA
        out["closing_implied_prob"] = pd.NA
        out["clv"] = pd.NA
        out["units"] = pd.NA
        out["fair_odds"] = pd.NA
        return out
    main = odds_history.loc[
        odds_history["market_key"].eq("batter_home_runs")
        & odds_history["selection_side"].eq("over")
        & odds_history["line"].eq(0.5)
    ].copy()
    if main.empty:
        out["captured_price"] = pd.NA
        out["captured_implied_prob"] = pd.NA
        out["closing_price"] = pd.NA
        out["closing_implied_prob"] = pd.NA
        out["clv"] = pd.NA
        out["units"] = pd.NA
        out["fair_odds"] = pd.NA
        return out
    group_cols = ["slate_date", "name_key", "market_key", "line"]
    first = _best_snapshot_price(main, group_cols, "first").rename(columns={"odds_american": "captured_price", "implied_prob": "captured_implied_prob"})
    last = _best_snapshot_price(main, group_cols, "last").rename(columns={"odds_american": "closing_price", "implied_prob": "closing_implied_prob"})
    out = out.merge(first[["slate_date", "name_key", "captured_price", "captured_implied_prob"]], on=["slate_date", "name_key"], how="left")
    out = out.merge(last[["slate_date", "name_key", "closing_price", "closing_implied_prob"]], on=["slate_date", "name_key"], how="left")
    out["clv"] = pd.to_numeric(out["closing_implied_prob"], errors="coerce") - pd.to_numeric(out["captured_implied_prob"], errors="coerce")
    out["units"] = out.apply(lambda row: _price_units(row.get("captured_price"), row.get("home_run_flag")), axis=1)
    out["fair_odds"] = pd.NA
    return out


def _build_pitcher_line_frame(base_frame: pd.DataFrame, odds_history: pd.DataFrame) -> pd.DataFrame:
    if base_frame.empty or odds_history.empty:
        return pd.DataFrame()
    odds = odds_history.loc[
        odds_history["market_key"].eq("pitcher_strikeouts")
        & odds_history["selection_side"].eq("over")
    ].copy()
    if odds.empty:
        return pd.DataFrame()
    group_cols = ["slate_date", "name_key", "market_key", "line", "selection_side"]
    first = _best_snapshot_price(odds, group_cols, "first").rename(columns={"odds_american": "captured_price", "implied_prob": "captured_implied_prob"})
    last = _best_snapshot_price(odds, group_cols, "last").rename(columns={"odds_american": "closing_price", "implied_prob": "closing_implied_prob"})
    lines = first.merge(
        last[["slate_date", "name_key", "market_key", "line", "selection_side", "closing_price", "closing_implied_prob"]],
        on=["slate_date", "name_key", "market_key", "line", "selection_side"],
        how="left",
    )
    lines = lines.merge(base_frame, on=["slate_date", "name_key"], how="left", suffixes=("", "_model"))
    lines["line_clear_flag"] = pd.to_numeric(lines.get("strikeouts"), errors="coerce").fillna(0).gt(pd.to_numeric(lines.get("line"), errors="coerce").fillna(999)).astype(float)
    lines["clv"] = pd.to_numeric(lines["closing_implied_prob"], errors="coerce") - pd.to_numeric(lines["captured_implied_prob"], errors="coerce")
    lines["units"] = lines.apply(lambda row: _price_units(row.get("captured_price"), row.get("line_clear_flag")), axis=1)
    lines["fair_odds"] = pd.NA
    return lines


def _top_n_summary(frame: pd.DataFrame, event_column: str, top_ns: list[int]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    baseline = pd.to_numeric(frame.get(event_column), errors="coerce").fillna(0).mean()
    rows = []
    for top_n in top_ns:
        subset = frame.loc[frame["slate_rank"] <= top_n].copy()
        if subset.empty:
            continue
        outcome = pd.to_numeric(subset.get(event_column), errors="coerce").fillna(0)
        rows.append(
            {
                "Top Cohort": f"Top {top_n}",
                "Event Count": float(outcome.sum()),
                "Realized Rate": float(outcome.mean()),
                "Baseline Rate": float(baseline),
                "Lift vs Baseline": float(outcome.mean() - baseline),
            }
        )
    return pd.DataFrame(rows)


def _pitcher_top_n_summary(frame: pd.DataFrame, top_ns: list[int]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    baseline_k = pd.to_numeric(frame.get("strikeouts"), errors="coerce").fillna(0).mean()
    baseline_k5 = pd.to_numeric(frame.get("k5_flag"), errors="coerce").fillna(0).mean()
    rows = []
    for top_n in top_ns:
        subset = frame.loc[frame["slate_rank"] <= top_n].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "Top Cohort": f"Top {top_n}",
                "Average Ks": pd.to_numeric(subset.get("strikeouts"), errors="coerce").fillna(0).mean(),
                "5+ K Rate": pd.to_numeric(subset.get("k5_flag"), errors="coerce").fillna(0).mean(),
                "Baseline Ks": baseline_k,
                "Baseline 5+ K Rate": baseline_k5,
            }
        )
    return pd.DataFrame(rows)


def _score_bucket_table(frame: pd.DataFrame, score_column: str, event_column: str, bucket_count: int, roi_column: str | None = None) -> pd.DataFrame:
    work = frame.loc[pd.to_numeric(frame.get(score_column), errors="coerce").notna()].copy()
    if work.empty:
        return pd.DataFrame()
    bucket_total = min(bucket_count, max(int(work[score_column].nunique()), 1))
    labels = [f"Bucket {i}" for i in range(bucket_total, 0, -1)]
    try:
        work["Score Bucket"] = pd.qcut(work[score_column], q=bucket_total, labels=labels, duplicates="drop")
    except ValueError:
        work["Score Bucket"] = "Bucket 1"
    table = (
        work.groupby("Score Bucket", observed=False, as_index=False)
        .agg(
            Sample_Size=(score_column, "size"),
            Average_Model_Score=(score_column, "mean"),
            Event_Count=(event_column, "sum"),
            Realized_Rate=(event_column, "mean"),
        )
        .sort_values("Average_Model_Score", ascending=False, na_position="last")
    )
    if roi_column and roi_column in work.columns:
        roi = work.groupby("Score Bucket", observed=False, as_index=False).agg(ROI=(roi_column, "mean"), Units=(roi_column, "sum"))
        table = table.merge(roi, on="Score Bucket", how="left")
    return table


def _rolling_top_n(frame: pd.DataFrame, event_column: str, top_n: int) -> pd.DataFrame:
    rows = []
    for slate_date, group in frame.groupby("slate_date", sort=True):
        top_group = group.loc[group["slate_rank"] <= top_n].copy()
        if top_group.empty:
            continue
        rows.append(
            {
                "slate_date": slate_date,
                "Realized Rate": pd.to_numeric(top_group.get(event_column), errors="coerce").fillna(0).mean(),
            }
        )
    return pd.DataFrame(rows)


def _rank_bucket_table(frame: pd.DataFrame, event_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("rank_bucket", as_index=False)
        .agg(Winners=(event_column, "sum"), Sample_Size=(event_column, "size"), Realized_Rate=(event_column, "mean"))
        .sort_values("rank_bucket")
    )


def _signal_table(frame: pd.DataFrame, signal_columns: list[str], winner_mask: pd.Series, compare_mask: pd.Series) -> pd.DataFrame:
    rows = []
    for column in signal_columns:
        if column not in frame.columns:
            continue
        win_mean = pd.to_numeric(frame.loc[winner_mask, column], errors="coerce").mean()
        compare_mean = pd.to_numeric(frame.loc[compare_mask, column], errors="coerce").mean()
        if pd.isna(win_mean) and pd.isna(compare_mean):
            continue
        rows.append({"Signal": column, "Winner Cohort Avg": win_mean, "Comparison Cohort Avg": compare_mean, "Delta": win_mean - compare_mean if pd.notna(win_mean) and pd.notna(compare_mean) else pd.NA})
    return pd.DataFrame(rows).sort_values("Delta", ascending=False, na_position="last") if rows else pd.DataFrame()


def _render_verdict(title: str, capture_rate: float | None, roi: float | None, clv: float | None) -> None:
    st.markdown(f"#### {title}")
    verdicts: list[str] = []
    verdicts.append("No captured-odds rows exist yet for this view." if capture_rate is None else ("The model is capturing enough real winners inside the chosen Top cohort." if capture_rate >= 0.5 else "The model is missing too many real winners inside the chosen Top cohort."))
    verdicts.append("ROI is not available yet." if roi is None or pd.isna(roi) else ("Captured prices are profitable so far." if roi > 0 else "Captured prices are not profitable so far."))
    verdicts.append("Closing-line comparison is not available yet." if clv is None or pd.isna(clv) else ("Captured prices are beating the stored close on average." if clv > 0 else "Captured prices are not beating the stored close on average."))
    for verdict in verdicts:
        st.markdown(f"- {verdict}")


def _render_hitter_view(frame: pd.DataFrame, score_column: str, filters: dict[str, object]) -> None:
    graded_mask = _render_outcome_status(frame, "Hitters")
    graded_frame = frame.loc[graded_mask].copy()
    top_n = int(filters["top_n"])
    winners = graded_frame["home_run_flag"].fillna(0).gt(0)
    captured = winners & graded_frame["slate_rank"].le(top_n)
    missed = winners & graded_frame["slate_rank"].gt(top_n)
    false_positive = graded_frame["slate_rank"].le(top_n) & graded_frame["home_run_flag"].fillna(0).eq(0)
    priced = graded_frame.loc[graded_frame["captured_price"].notna()].copy()
    capture_rate = None if winners.sum() == 0 else float(captured.sum() / max(int(winners.sum()), 1))
    roi = pd.to_numeric(priced.get("units"), errors="coerce").mean() if not priced.empty else None
    clv = pd.to_numeric(priced.get("clv"), errors="coerce").mean() if not priced.empty else None

    _render_metric_cards([
        ("Tracked Rows", f"{len(frame):,}", None),
        ("Graded Rows", f"{len(graded_frame):,}", None),
        ("HR Winners", f"{int(winners.sum()):,}", None),
        ("Captured Winners", f"{int(captured.sum()):,}", None),
        ("Missed Winners", f"{int(missed.sum()):,}", None),
        ("Units", "N/A" if roi is None or pd.isna(roi) else f"{pd.to_numeric(priced.get('units'), errors='coerce').fillna(0).sum():+.2f}", None),
    ])
    _render_verdict("Model Verdict Today" if filters["mode"] == "Slate Audit" else "Model Verdict Since Tracking Began", capture_rate, roi, clv)

    render_metric_grid(_top_n_summary(graded_frame, "home_run_flag", [1, 3, 5, top_n]), key=f"bt-h-topn-{score_column}", height=180, use_lightweight=True)
    cols = st.columns(2)
    with cols[0]:
        _render_simple_chart(_rolling_top_n(graded_frame, "home_run_flag", top_n), "slate_date", "Realized Rate", "line", "Top-N HR Hit Rate Over Time")
    with cols[1]:
        _render_simple_chart(_rank_bucket_table(graded_frame, "home_run_flag"), "rank_bucket", "Winners", "bar", "Captured vs Missed Winners by Rank Bucket")
    bucket = _score_bucket_table(priced if not priced.empty else graded_frame, score_column, "home_run_flag", int(filters["bucket_count"]), "units" if not priced.empty else None)
    cols = st.columns(2)
    with cols[0]:
        _render_simple_chart(bucket, "Score Bucket", "Realized_Rate", "bar", "Score Bucket vs Realized HR Rate")
    with cols[1]:
        _render_simple_chart(bucket, "Score Bucket", "ROI" if "ROI" in bucket.columns else "Average_Model_Score", "bar", "Score Bucket vs ROI" if "ROI" in bucket.columns else "Average Model Score by Bucket")

    detail_cols = _existing_columns(frame, ["slate_date", "hitter_name", "team", "game_label", "slate_rank", score_column, "home_runs", "total_bases", "runs_rbi", "captured_price", "closing_price", "clv", "units", "fair_odds"])
    st.markdown("#### Biggest Captured HRs")
    render_metric_grid(graded_frame.loc[captured, detail_cols].sort_values([score_column], ascending=[False], na_position="last"), key=f"bt-h-cap-{score_column}", height=240, use_lightweight=True)
    st.markdown("#### Biggest Missed HRs")
    render_metric_grid(graded_frame.loc[missed, detail_cols].sort_values([score_column], ascending=[False], na_position="last"), key=f"bt-h-miss-{score_column}", height=240, use_lightweight=True)
    st.markdown("#### Bad High-Rank Plays")
    render_metric_grid(graded_frame.loc[false_positive, detail_cols].sort_values(["slate_rank"]), key=f"bt-h-fp-{score_column}", height=240, use_lightweight=True)

    signals = ["matchup_score", "ceiling_score", "zone_fit_score", "xwoba", "xwoba_con", "swstr_pct", "pulled_barrel_pct", "sweet_spot_pct", "avg_launch_angle", "hard_hit_pct"]
    st.markdown("#### Signal Attribution Summary")
    render_metric_grid(_signal_table(graded_frame, signals, captured, missed if missed.any() else false_positive), key=f"bt-h-signals-{score_column}", height=240, use_lightweight=True)


def _render_pitcher_view(base_frame: pd.DataFrame, line_frame: pd.DataFrame, score_column: str, filters: dict[str, object]) -> None:
    graded_mask = _render_outcome_status(base_frame, "Pitchers")
    graded_base = base_frame.loc[graded_mask].copy()
    graded_keys = set(zip(graded_base["slate_date"], graded_base["name_key"]))
    graded_line = line_frame.loc[line_frame.apply(lambda row: (row.get("slate_date"), row.get("name_key")) in graded_keys, axis=1)].copy() if not line_frame.empty else line_frame
    top_n = int(filters["top_n"])
    clear_lines = graded_line["line_clear_flag"].gt(0) if not graded_line.empty else pd.Series(dtype="bool")
    captured = clear_lines & graded_line["slate_rank"].le(top_n) if not graded_line.empty else pd.Series(dtype="bool")
    missed = clear_lines & graded_line["slate_rank"].gt(top_n) if not graded_line.empty else pd.Series(dtype="bool")
    false_positive = graded_line["slate_rank"].le(top_n) & graded_line["line_clear_flag"].eq(0) if not graded_line.empty else pd.Series(dtype="bool")
    roi = pd.to_numeric(graded_line.get("units"), errors="coerce").mean() if not graded_line.empty else None
    clv = pd.to_numeric(graded_line.get("clv"), errors="coerce").mean() if not graded_line.empty else None
    capture_rate = None if graded_line.empty or clear_lines.sum() == 0 else float(captured.sum() / max(int(clear_lines.sum()), 1))

    _render_metric_cards([
        ("Tracked Pitchers", f"{len(base_frame):,}", None),
        ("Graded Pitchers", f"{len(graded_base):,}", None),
        ("Cleared K Props", f"{int(clear_lines.sum()):,}" if not graded_line.empty else "0", None),
        ("Captured K Props", f"{int(captured.sum()):,}" if not graded_line.empty else "0", None),
        ("Missed K Props", f"{int(missed.sum()):,}" if not graded_line.empty else "0", None),
        ("Units", "N/A" if roi is None or pd.isna(roi) else f"{pd.to_numeric(graded_line.get('units'), errors='coerce').fillna(0).sum():+.2f}", None),
    ])
    _render_verdict("Model Verdict Today" if filters["mode"] == "Slate Audit" else "Model Verdict Since Tracking Began", capture_rate, roi, clv)

    render_metric_grid(_pitcher_top_n_summary(graded_base, [1, 3, 5, top_n]), key=f"bt-p-topn-{score_column}", height=180, use_lightweight=True)
    cols = st.columns(2)
    with cols[0]:
        _render_simple_chart(_rolling_top_n(graded_base, "k5_flag", top_n), "slate_date", "Realized Rate", "line", "Top-N Pitcher 5+ K Rate Over Time")
    with cols[1]:
        _render_simple_chart(_score_bucket_table(graded_base, score_column, "k5_flag", int(filters["bucket_count"])), "Score Bucket", "Realized_Rate", "bar", "Score Bucket vs 5+ K Rate")

    st.markdown("#### K Leaders")
    render_metric_grid(graded_base[_existing_columns(graded_base, ["slate_date", "pitcher_name", "team", "game_label", "slate_rank", score_column, "strikeouts", "k5_flag", "pitcher_score", "strikeout_score", "csw_pct", "swstr_pct", "putaway_pct", "ball_pct", "siera"])].sort_values(["strikeouts", score_column], ascending=[False, False], na_position="last"), key=f"bt-p-leaders-{score_column}", height=240, use_lightweight=True)

    if graded_line.empty:
        st.info("No captured strikeout lines were stored for this view yet.")
        return
    line_cols = _existing_columns(graded_line, ["slate_date", "pitcher_name", "team", "game_label", "slate_rank", score_column, "line", "strikeouts", "line_clear_flag", "captured_price", "closing_price", "clv", "units", "fair_odds"])
    st.markdown("#### Biggest Captured K Props")
    render_metric_grid(graded_line.loc[captured, line_cols].sort_values([score_column, "line"], ascending=[False, True], na_position="last"), key=f"bt-p-cap-{score_column}", height=240, use_lightweight=True)
    st.markdown("#### Biggest Missed K Props")
    render_metric_grid(graded_line.loc[missed, line_cols].sort_values([score_column, "line"], ascending=[False, True], na_position="last"), key=f"bt-p-miss-{score_column}", height=240, use_lightweight=True)
    st.markdown("#### Bad High-Rank K Props")
    render_metric_grid(graded_line.loc[false_positive, line_cols].sort_values(["slate_rank", "line"]), key=f"bt-p-fp-{score_column}", height=240, use_lightweight=True)

    threshold = graded_line.groupby("line", as_index=False).agg(Clear_Rate=("line_clear_flag", "mean"), ROI=("units", "mean")).sort_values("line")
    cols = st.columns(2)
    with cols[0]:
        _render_simple_chart(threshold, "line", "Clear_Rate", "bar", "Pitcher K Line-Level Performance")
    with cols[1]:
        _render_simple_chart(threshold, "line", "ROI", "bar", "Pitcher K Line ROI by Threshold")

    signals = ["strikeout_score", "pitcher_score", "csw_pct", "swstr_pct", "putaway_pct", "ball_pct", "siera", "xwoba"]
    st.markdown("#### Signal Attribution Summary")
    render_metric_grid(_signal_table(graded_line, signals, captured, missed if missed.any() else false_positive), key=f"bt-p-signals-{score_column}", height=240, use_lightweight=True)


def render_backtesting_tab(config: AppConfig) -> None:
    if not config.database_url:
        st.error("DATABASE_URL is required for Backtesting because this page reads tracked model, outcome, and odds history from Cockroach.")
        return

    filters = _filters()
    if filters["start_date"] > filters["end_date"]:
        st.error("Start date must be on or before end date.")
        return

    try:
        snapshots, outcomes, odds = _load_data(
            config,
            str(filters["entity"]),
            filters["start_date"],
            filters["end_date"],
            str(filters["split_key"]),
            str(filters["recent_window"]),
            str(filters["weighted_mode"]),
        )
    except Exception as exc:
        st.error(f"Unable to load backtesting data from Cockroach: {exc}")
        return

    _render_explainer(str(filters["entity"]), str(filters["mode"]))
    odds_history = _prepare_odds_history(odds)

    if str(filters["entity"]) == "Hitters":
        frame = _merge_hitter_prices(_add_slate_ranks(_prepare_hitter_frame(snapshots, outcomes), str(filters["score_choice"])), odds_history)
        if frame.empty:
            st.info("No hitter rows matched the selected filters.")
            return
        _render_hitter_view(frame, str(filters["score_choice"]), filters)
        return

    base_frame = _add_slate_ranks(_prepare_pitcher_frame(snapshots, outcomes), str(filters["score_choice"]))
    if base_frame.empty:
        st.info("No pitcher rows matched the selected filters.")
        return
    line_frame = _build_pitcher_line_frame(base_frame, odds_history)
    _render_pitcher_view(base_frame, line_frame, str(filters["score_choice"]), filters)
