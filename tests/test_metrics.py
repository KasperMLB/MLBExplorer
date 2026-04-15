import pandas as pd

from mlb_dashboard.dashboard_views import add_hitter_matchup_score, filter_excluded_pitchers_from_hitter_pool, normalize_series
from mlb_dashboard.metrics import add_metric_flags, is_barrel


def test_add_metric_flags_marks_core_metrics():
    frame = pd.DataFrame(
        [
            {
                "launch_speed": 101.0,
                "launch_angle": 27.0,
                "bb_type": "fly_ball",
                "description": "swinging_strike",
                "estimated_woba_using_speedangle": 0.7,
                "release_speed": 95.0,
                "release_spin_rate": 2300.0,
            }
        ]
    )
    enriched = add_metric_flags(frame)
    row = enriched.iloc[0]
    assert bool(row["is_barrel"])
    assert bool(row["is_fly_ball"])
    assert bool(row["is_hard_hit"])
    assert bool(row["is_swinging_strike"])


def test_official_barrel_bands_expand_with_exit_velocity():
    frame = pd.DataFrame(
        [
            {"launch_speed": 98.0, "launch_angle": 26.0},
            {"launch_speed": 98.4, "launch_angle": 27.0},
            {"launch_speed": 98.0, "launch_angle": 25.0},
            {"launch_speed": 100.0, "launch_angle": 33.0},
            {"launch_speed": 100.0, "launch_angle": 34.0},
            {"launch_speed": 112.0, "launch_angle": 50.0},
            {"launch_speed": 112.0, "launch_angle": 51.0},
            {"launch_speed": 116.0, "launch_angle": 8.0},
            {"launch_speed": 116.0, "launch_angle": 51.0},
        ]
    )
    result = is_barrel(frame).tolist()
    assert result == [True, True, False, True, False, True, False, True, False]


def test_ohtani_two_way_exception_stays_in_hitter_pool():
    hitters = pd.DataFrame(
        [
            {"batter": 660271, "hitter_name": "Shohei Ohtani"},
            {"batter": 123456, "hitter_name": "Pitcher Hitter"},
            {"batter": 999999, "hitter_name": "Everyday Batter"},
        ]
    )
    exclusions = pd.DataFrame(
        [
            {"player_id": 660271, "pitcher_name": "Ohtani, Shohei", "exclude_from_hitter_tables": True},
            {"player_id": 123456, "pitcher_name": "Pitcher Hitter", "exclude_from_hitter_tables": True},
        ]
    )

    filtered = filter_excluded_pitchers_from_hitter_pool(hitters, exclusions)

    assert filtered["batter"].tolist() == [660271, 999999]


def test_test_score_boosts_zone_fit_without_changing_shape_formula():
    hitters = pd.DataFrame(
        [
            {
                "batter": 1,
                "hitter_name": "Launch Band Hitter",
                "swstr_pct": 0.12,
                "barrel_bbe_pct": 0.20,
                "pulled_barrel_pct": 0.18,
                "sweet_spot_pct": 0.38,
                "avg_launch_angle": 22.0,
                "barrel_bip_pct": 0.15,
                "hard_hit_pct": 0.48,
                "xwoba": 0.410,
            },
            {
                "batter": 2,
                "hitter_name": "Low Launch Hitter",
                "swstr_pct": 0.18,
                "barrel_bbe_pct": 0.10,
                "pulled_barrel_pct": 0.08,
                "sweet_spot_pct": 0.24,
                "avg_launch_angle": 4.0,
                "barrel_bip_pct": 0.08,
                "hard_hit_pct": 0.36,
                "xwoba": 0.330,
            },
        ]
    )

    scored = add_hitter_matchup_score(hitters)
    pulled_barrel_scale = normalize_series(scored["pulled_barrel_pct"])
    pulled_barrel_bonus = ((pulled_barrel_scale - 0.5).clip(lower=0.0) / 0.5) * 0.08
    expected_test_score = ((
        (normalize_series(scored["swstr_pct"], inverse=True) * 0.325)
        + (normalize_series(scored["barrel_bbe_pct"]) * 0.30)
        + (scored["shape_score"] * 0.20)
        + (scored["zone_fit_score"] * 0.175)
    ) * 100.0 * (1.0 + pulled_barrel_bonus)).clip(lower=0.0, upper=100.0)

    pd.testing.assert_series_equal(
        scored["test_shape_score"],
        scored["shape_score"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        scored["test_score"],
        expected_test_score,
        check_names=False,
    )
