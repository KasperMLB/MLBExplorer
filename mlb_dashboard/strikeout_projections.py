"""Strikeout and walk projection logic for Kasper's Strikeouts page.

Projection chain: avg pitch count per start → pitches per batter → batters faced → K/BB totals.
Matchup blend:    70% pitcher historical rate + 30% pitch-mix-and-zone-aware lineup component.
Year weighting:   game outcome starts are weighted by the same year schedule used in metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #

# Year weights matching DEFAULT_YEAR_WEIGHTS in config.py
OUTCOME_YEAR_WEIGHTS: dict[int, float] = {
    2026: 1.35,
    2025: 1.00,
    2024: 0.90,
    2023: 0.70,
    2022: 0.50,
}
DEFAULT_YEAR_WEIGHT = 0.35  # fallback for seasons older than 2022

# League-average fallbacks (used when sample is insufficient)
LEAGUE_AVG_PITCH_COUNT = 88.0
LEAGUE_AVG_P_PER_BF = 3.88
LEAGUE_AVG_K_RATE = 0.230   # K per PA, league average
LEAGUE_AVG_BB_RATE = 0.083
LEAGUE_AVG_SWSTR = 0.110

# Swinging-strike → K% calibration: K% ≈ swstr_pct × 2.1
SWSTR_TO_K_CALIBRATION = 2.1
# ball_pct → BB% calibration: BB% ≈ ball_pct × 0.245
BALL_TO_BB_CALIBRATION = 0.245

# Minimum weighted-equivalent starts before switching to proxy fallback
MIN_WEIGHTED_STARTS = 5.0


# --------------------------------------------------------------------------- #
# Result dataclass                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class ProjectionResult:
    pitcher_id: int
    pitcher_name: str
    team: str
    p_throws: str

    projected_k: float
    projected_bb: float
    projected_bf: float
    avg_pitch_count: float
    avg_p_per_bf: float

    pitcher_k_rate: float
    pitcher_bb_rate: float
    lineup_k_rate: float          # pitch-mix-aware 30% component
    blended_k_rate: float

    sample_starts: int            # raw count of starts found
    weighted_starts: float        # year-weighted equivalent starts
    confidence: str               # "High" / "Medium" / "Low"
    using_proxy: bool             # True if proxy stats were used as fallback

    # Pitch-mix detail
    pitcher_mix_whiff: float = 0.0   # pitcher's weighted whiff rate from family-zone context
    hitter_k_probs: list[dict] = field(default_factory=list)  # per-hitter K probability rows


# --------------------------------------------------------------------------- #
# Year-weighting helpers                                                        #
# --------------------------------------------------------------------------- #

def _year_weight(game_year: int) -> float:
    return OUTCOME_YEAR_WEIGHTS.get(game_year, DEFAULT_YEAR_WEIGHT)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    w = weights[values.notna() & weights.gt(0)]
    v = values[values.notna() & weights.gt(0)]
    if w.sum() == 0:
        return None
    return float((v * w).sum() / w.sum())


def _add_year_weights(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Add a `_yw` column to an outcomes frame using slate_date year."""
    out = outcomes.copy()
    if "slate_date" in out.columns:
        years = pd.to_datetime(out["slate_date"], errors="coerce").dt.year.fillna(0).astype(int)
    else:
        years = pd.Series(0, index=out.index)
    out["_yw"] = years.map(lambda y: _year_weight(y))
    return out


# --------------------------------------------------------------------------- #
# BF / rate computation from game outcomes                                      #
# --------------------------------------------------------------------------- #

def _compute_outcome_rates(
    outcomes: pd.DataFrame,
) -> tuple[float, float, float, float, int, float]:
    """Return (avg_pitch_count, avg_p_per_bf, k_rate, bb_rate, n_starts, weighted_starts)."""
    if outcomes.empty:
        return LEAGUE_AVG_PITCH_COUNT, LEAGUE_AVG_P_PER_BF, LEAGUE_AVG_K_RATE, LEAGUE_AVG_BB_RATE, 0, 0.0

    df = _add_year_weights(outcomes)
    # Keep only starts with meaningful pitch/BF data
    usable = df.loc[df["batters_faced"].gt(0)].copy()

    n_starts = len(usable)
    w_starts = float(usable["_yw"].sum())

    if w_starts < MIN_WEIGHTED_STARTS:
        return LEAGUE_AVG_PITCH_COUNT, LEAGUE_AVG_P_PER_BF, LEAGUE_AVG_K_RATE, LEAGUE_AVG_BB_RATE, n_starts, w_starts

    # Pitch count: only rows where pitch_count > 0 (pre-pipeline rows will be 0)
    pc_usable = usable.loc[usable.get("pitch_count", pd.Series(0, index=usable.index)).gt(0)]
    if len(pc_usable) >= 3 and "pitch_count" in pc_usable.columns:
        avg_pitch_count = _weighted_mean(
            pd.to_numeric(pc_usable["pitch_count"], errors="coerce"),
            pc_usable["_yw"],
        ) or LEAGUE_AVG_PITCH_COUNT

        p_per_bf_series = pd.to_numeric(pc_usable["pitch_count"], errors="coerce") / pd.to_numeric(pc_usable["batters_faced"], errors="coerce").replace(0, pd.NA)
        avg_p_per_bf = _weighted_mean(p_per_bf_series, pc_usable["_yw"]) or LEAGUE_AVG_P_PER_BF
    else:
        avg_pitch_count = LEAGUE_AVG_PITCH_COUNT
        avg_p_per_bf = LEAGUE_AVG_P_PER_BF

    k_per_bf = pd.to_numeric(usable["strikeouts"], errors="coerce") / pd.to_numeric(usable["batters_faced"], errors="coerce").replace(0, pd.NA)
    bb_per_bf = pd.to_numeric(usable["walks"], errors="coerce") / pd.to_numeric(usable["batters_faced"], errors="coerce").replace(0, pd.NA)

    k_rate = _weighted_mean(k_per_bf, usable["_yw"]) or LEAGUE_AVG_K_RATE
    bb_rate = _weighted_mean(bb_per_bf, usable["_yw"]) or LEAGUE_AVG_BB_RATE

    return avg_pitch_count, avg_p_per_bf, k_rate, bb_rate, n_starts, w_starts


# --------------------------------------------------------------------------- #
# Pitch-mix and zone-aware lineup K rate                                        #
# --------------------------------------------------------------------------- #

def compute_pitcher_mix_whiff(
    pitcher_id: int,
    pitcher_fzc: pd.DataFrame,
) -> float:
    """Compute pitcher's pitch-mix-weighted expected whiff rate from family-zone context.

    Returns Σ(usage_rate_overall × whiff_rate) across all (pitch_family, zone_bucket) rows.
    Shadow zones contribute most of the whiff mass.  Result is typically 0.08–0.15.
    """
    if pitcher_fzc.empty or "pitcher_id" not in pitcher_fzc.columns:
        return LEAGUE_AVG_SWSTR
    rows = pitcher_fzc.loc[pitcher_fzc["pitcher_id"] == pitcher_id].copy()
    if rows.empty:
        return LEAGUE_AVG_SWSTR
    usage = pd.to_numeric(rows.get("usage_rate_overall", pd.Series(dtype=float)), errors="coerce").fillna(0)
    whiff = pd.to_numeric(rows.get("whiff_rate", pd.Series(dtype=float)), errors="coerce").fillna(0)
    total = float((usage * whiff).sum())
    return total if total > 0 else LEAGUE_AVG_SWSTR


def _family_vulnerability_scalar(
    batter_id: int,
    p_throws: str,
    pitcher_fzc_rows: pd.DataFrame,
    batter_fzp: pd.DataFrame,
) -> float:
    """Return a multiplier (centered at 1.0) reflecting hitter's vulnerability to this pitcher's mix.

    Logic: for each pitch family the pitcher uses heavily, check if the hitter has low xwoba
    against that family (= struggles to do damage = more K-prone).  Returns 1.0 on sparse data.
    """
    if batter_fzp.empty or pitcher_fzc_rows.empty:
        return 1.0

    hand_key = "vs_rhp" if str(p_throws).upper() == "R" else "vs_lhp"
    hitter_rows = batter_fzp.loc[
        (batter_fzp["batter_id"] == batter_id)
        & (batter_fzp.get("pitcher_hand_key", pd.Series("overall", index=batter_fzp.index)) == hand_key)
    ].copy()
    if hitter_rows.empty:
        # try "overall" split
        hitter_rows = batter_fzp.loc[
            (batter_fzp["batter_id"] == batter_id)
            & (batter_fzp.get("pitcher_hand_key", pd.Series("overall", index=batter_fzp.index)) == "overall")
        ].copy()
    if hitter_rows.empty:
        return 1.0

    # Pitcher family usage (normalized)
    pitcher_usage = (
        pitcher_fzc_rows.groupby("pitch_family")["usage_rate_overall"]
        .sum()
        .reset_index()
    )
    total_usage = pitcher_usage["usage_rate_overall"].sum()
    if total_usage == 0:
        return 1.0
    pitcher_usage["share"] = pitcher_usage["usage_rate_overall"] / total_usage

    # Hitter xwoba per family (lower = more vulnerable = higher K scalar)
    hitter_family = (
        hitter_rows.groupby("pitch_family")
        .apply(
            lambda g: pd.Series({
                "xwoba": _weighted_mean(
                    pd.to_numeric(g.get("xwoba", pd.Series(dtype=float)), errors="coerce"),
                    pd.to_numeric(g.get("weighted_sample_size", pd.Series(1.0, index=g.index)), errors="coerce").fillna(1),
                ) or 0.320,
                "sample": float(pd.to_numeric(g.get("weighted_sample_size", pd.Series(0, index=g.index)), errors="coerce").sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # League-average xwoba reference
    LEAGUE_XWOBA = 0.315
    scalar_sum = 0.0
    weight_sum = 0.0
    for _, pu_row in pitcher_usage.iterrows():
        fam = pu_row["pitch_family"]
        share = float(pu_row["share"])
        hfam = hitter_family.loc[hitter_family["pitch_family"] == fam]
        if hfam.empty or float(hfam.iloc[0]["sample"]) < 20:
            # insufficient sample — assume league average (scalar = 1.0)
            scalar_sum += share * 1.0
        else:
            hxwoba = float(hfam.iloc[0]["xwoba"])
            # invert: lower hitter xwoba vs. this family → higher K vulnerability
            vuln = max(0.5, min(1.5, 1.0 + (LEAGUE_XWOBA - hxwoba) / LEAGUE_XWOBA * 0.5))
            scalar_sum += share * vuln
        weight_sum += share

    return float(scalar_sum / weight_sum) if weight_sum > 0 else 1.0


def compute_pitch_mix_lineup_k_rate(
    pitcher_id: int,
    p_throws: str,
    opp_hitters: pd.DataFrame,
    pitcher_fzc: pd.DataFrame,
    batter_fzp: pd.DataFrame,
    pitcher_mix_whiff: float,
) -> tuple[float, list[dict]]:
    """Compute the pitch-mix-and-zone-aware lineup K rate (30% component).

    Returns (lineup_k_rate, per_hitter_detail_rows).
    """
    if opp_hitters.empty:
        return LEAGUE_AVG_K_RATE, []

    pitcher_rows = pitcher_fzc.loc[pitcher_fzc["pitcher_id"] == pitcher_id] if not pitcher_fzc.empty else pd.DataFrame()

    hitter_k_probs: list[dict] = []
    k_rates: list[float] = []

    for _, hitter in opp_hitters.iterrows():
        swstr = float(pd.to_numeric(hitter.get("swstr_pct", LEAGUE_AVG_SWSTR), errors="coerce") or LEAGUE_AVG_SWSTR)
        swstr_scale = swstr / LEAGUE_AVG_SWSTR

        batter_id = hitter.get("batter_id") or hitter.get("player_id")
        vuln_scalar = 1.0
        if batter_id is not None and not pitcher_rows.empty and not batter_fzp.empty:
            try:
                vuln_scalar = _family_vulnerability_scalar(int(batter_id), p_throws, pitcher_rows, batter_fzp)
            except Exception:
                vuln_scalar = 1.0

        hitter_k_rate = pitcher_mix_whiff * swstr_scale * vuln_scalar * SWSTR_TO_K_CALIBRATION
        hitter_k_rate = max(0.05, min(0.55, hitter_k_rate))
        k_rates.append(hitter_k_rate)

        hitter_k_probs.append({
            "hitter_name": hitter.get("hitter_name", ""),
            "team": hitter.get("team", ""),
            "bats": hitter.get("bats", ""),
            "swstr_pct": swstr,
            "swstr_scale": round(swstr_scale, 3),
            "family_vuln": round(vuln_scalar, 3),
            "k_prob": round(hitter_k_rate, 3),
        })

    lineup_k_rate = float(sum(k_rates) / len(k_rates)) if k_rates else LEAGUE_AVG_K_RATE
    return lineup_k_rate, hitter_k_probs


# --------------------------------------------------------------------------- #
# Main projection function                                                      #
# --------------------------------------------------------------------------- #

def _confidence_label(weighted_starts: float) -> str:
    if weighted_starts >= 15:
        return "High"
    if weighted_starts >= MIN_WEIGHTED_STARTS:
        return "Medium"
    return "Low"


def compute_pitcher_projection(
    pitcher_metrics_row: pd.Series,
    outcomes: pd.DataFrame,
    opp_hitters: pd.DataFrame,
    pitcher_fzc: pd.DataFrame,
    batter_fzp: pd.DataFrame,
) -> ProjectionResult:
    """Compute a full K/BB projection for one starting pitcher.

    Chain:
      avg_pitch_count (year-weighted) → avg_p_per_bf → projected_bf
      pitcher_k_rate (year-weighted historical OR proxy) × 0.70
      + pitch-mix-lineup_k_rate × 0.30
      → projected_k

      pitcher_bb_rate (year-weighted historical OR proxy) → projected_bb
    """
    pitcher_id = int(pitcher_metrics_row.get("pitcher_id", 0) or 0)
    pitcher_name = str(pitcher_metrics_row.get("pitcher_name", "") or "")
    team = str(pitcher_metrics_row.get("team", "") or "")
    p_throws = str(pitcher_metrics_row.get("p_throws", "R") or "R")

    # --- BF chain & historical rates ---
    avg_pitch_count, avg_p_per_bf, hist_k_rate, hist_bb_rate, n_starts, w_starts = _compute_outcome_rates(outcomes)
    using_proxy = w_starts < MIN_WEIGHTED_STARTS
    projected_bf = avg_pitch_count / avg_p_per_bf if avg_p_per_bf > 0 else LEAGUE_AVG_PITCH_COUNT / LEAGUE_AVG_P_PER_BF

    if using_proxy:
        swstr = float(pd.to_numeric(pitcher_metrics_row.get("swstr_pct", LEAGUE_AVG_SWSTR), errors="coerce") or LEAGUE_AVG_SWSTR)
        ball_pct = float(pd.to_numeric(pitcher_metrics_row.get("ball_pct", 0.34), errors="coerce") or 0.34)
        pitcher_k_rate = swstr * SWSTR_TO_K_CALIBRATION
        pitcher_bb_rate = ball_pct * BALL_TO_BB_CALIBRATION
    else:
        pitcher_k_rate = hist_k_rate
        pitcher_bb_rate = hist_bb_rate

    # --- Pitch-mix lineup component (30%) ---
    pitcher_mix_whiff = compute_pitcher_mix_whiff(pitcher_id, pitcher_fzc)
    lineup_k_rate, hitter_k_probs = compute_pitch_mix_lineup_k_rate(
        pitcher_id, p_throws, opp_hitters, pitcher_fzc, batter_fzp, pitcher_mix_whiff,
    )

    blended_k_rate = pitcher_k_rate * 0.70 + lineup_k_rate * 0.30

    projected_k = projected_bf * blended_k_rate
    projected_bb = projected_bf * pitcher_bb_rate

    return ProjectionResult(
        pitcher_id=pitcher_id,
        pitcher_name=pitcher_name,
        team=team,
        p_throws=p_throws,
        projected_k=round(projected_k, 1),
        projected_bb=round(projected_bb, 1),
        projected_bf=round(projected_bf, 1),
        avg_pitch_count=round(avg_pitch_count, 1),
        avg_p_per_bf=round(avg_p_per_bf, 2),
        pitcher_k_rate=round(pitcher_k_rate, 3),
        pitcher_bb_rate=round(pitcher_bb_rate, 3),
        lineup_k_rate=round(lineup_k_rate, 3),
        blended_k_rate=round(blended_k_rate, 3),
        sample_starts=n_starts,
        weighted_starts=round(w_starts, 1),
        confidence=_confidence_label(w_starts),
        using_proxy=using_proxy,
        pitcher_mix_whiff=round(pitcher_mix_whiff, 3),
        hitter_k_probs=hitter_k_probs,
    )


# --------------------------------------------------------------------------- #
# Slate-level builder                                                           #
# --------------------------------------------------------------------------- #

def build_slate_projections(
    games: list[dict],
    pitcher_metrics: pd.DataFrame,
    outcomes_frame: pd.DataFrame,
    hitters_by_team: dict[str, pd.DataFrame],
    pitcher_fzc: pd.DataFrame,
    batter_fzp: pd.DataFrame,
) -> pd.DataFrame:
    """Build one projection row per starting pitcher for the full slate.

    games: list of game dicts (away_team, home_team, away_probable_pitcher_id, etc.)
    pitcher_metrics: all pitcher metric rows (filtered to relevant pitchers)
    outcomes_frame: all pitcher game outcomes (all pitchers, last N starts)
    hitters_by_team: dict of team → hitter DataFrame (for opposing lineup)
    pitcher_fzc: pitcher_family_zone_context for the day
    batter_fzp: batter_family_zone_profiles for the day
    """
    rows: list[dict] = []

    metrics_by_id: dict[int, pd.Series] = {}
    if not pitcher_metrics.empty and "pitcher_id" in pitcher_metrics.columns:
        for _, row in pitcher_metrics.iterrows():
            pid = int(row.get("pitcher_id", 0) or 0)
            if pid:
                metrics_by_id[pid] = row

    outcomes_by_id: dict[int, pd.DataFrame] = {}
    if not outcomes_frame.empty and "pitcher_id" in outcomes_frame.columns:
        for pid, grp in outcomes_frame.groupby("pitcher_id"):
            outcomes_by_id[int(pid)] = grp

    for game in games:
        game_pk = game.get("game_pk")
        away_team = str(game.get("away_team", "") or "")
        home_team = str(game.get("home_team", "") or "")

        for role, team, opp_team, pitcher_id_key, pitcher_name_key in [
            ("away", away_team, home_team, "away_probable_pitcher_id", "away_probable_pitcher_name"),
            ("home", home_team, away_team, "home_probable_pitcher_id", "home_probable_pitcher_name"),
        ]:
            pitcher_id = game.get(pitcher_id_key)
            pitcher_name = game.get(pitcher_name_key, "TBD") or "TBD"
            try:
                if pitcher_id is None or pitcher_id != pitcher_id:  # NaN != NaN
                    continue
                pitcher_id = int(pitcher_id)
            except (TypeError, ValueError):
                continue
            metrics_row = metrics_by_id.get(pitcher_id, pd.Series({
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher_name,
                "team": team,
                "p_throws": "R",
                "swstr_pct": LEAGUE_AVG_SWSTR,
                "ball_pct": 0.34,
            }))

            opp_hitters = hitters_by_team.get(opp_team, pd.DataFrame())
            outcomes = outcomes_by_id.get(pitcher_id, pd.DataFrame())

            proj = compute_pitcher_projection(metrics_row, outcomes, opp_hitters, pitcher_fzc, batter_fzp)

            rows.append({
                "game_pk": game_pk,
                "role": role,
                "team": team,
                "opp_team": opp_team,
                "away_team": away_team,
                "home_team": home_team,
                "pitcher_id": pitcher_id,
                "pitcher_name": proj.pitcher_name or pitcher_name,
                "p_throws": proj.p_throws,
                "projected_k": proj.projected_k,
                "projected_bb": proj.projected_bb,
                "projected_bf": proj.projected_bf,
                "avg_pitch_count": proj.avg_pitch_count,
                "avg_p_per_bf": proj.avg_p_per_bf,
                "pitcher_k_rate": proj.pitcher_k_rate,
                "pitcher_bb_rate": proj.pitcher_bb_rate,
                "lineup_k_rate": proj.lineup_k_rate,
                "blended_k_rate": proj.blended_k_rate,
                "pitcher_mix_whiff": proj.pitcher_mix_whiff,
                "sample_starts": proj.sample_starts,
                "weighted_starts": proj.weighted_starts,
                "confidence": proj.confidence,
                "using_proxy": proj.using_proxy,
                "_hitter_k_probs": proj.hitter_k_probs,
            })

    return pd.DataFrame(rows)
