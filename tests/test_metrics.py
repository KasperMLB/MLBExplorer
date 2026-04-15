import pandas as pd

from mlb_dashboard.dashboard_views import filter_excluded_pitchers_from_hitter_pool
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
