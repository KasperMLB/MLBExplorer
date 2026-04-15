import pandas as pd

from mlb_dashboard.config import AppConfig
from mlb_dashboard.build import BuildContext, _save_per_game_files
from mlb_dashboard.dashboard_views import add_hitter_matchup_score, filter_excluded_pitchers_from_hitter_pool, normalize_series
from mlb_dashboard.local_app import _game_selection as _local_game_selection
from mlb_dashboard.local_store import (
    compute_hitter_rolling,
    compute_pitcher_rolling,
    load_local_source_payload,
    read_hitter_exit_velo_events,
    read_latest_prop_odds_snapshot,
    read_prop_odds_history,
    write_props_odds_snapshot,
    write_statcast_events,
    write_tracking_payload,
    read_hitter_snapshots_for_date,
)
from mlb_dashboard.metrics import add_metric_flags, is_barrel


def test_game_selection_defaults_to_first_game_detail():
    slate = [
        {"away_team": "AAA", "home_team": "BBB", "game_pk": 1},
        {"away_team": "CCC", "home_team": "DDD", "game_pk": 2},
    ]
    import mlb_dashboard.local_app as local_app

    original = local_app.st.sidebar.selectbox
    try:
        seen = {}

        def fake_selectbox(label, options, index=0):
            seen["options"] = options
            seen["index"] = index
            return options[index]

        local_app.st.sidebar.selectbox = fake_selectbox
        selection, selected_games = _local_game_selection(slate)
    finally:
        local_app.st.sidebar.selectbox = original

    assert seen["options"] == ["Slate Summary", "AAA @ BBB", "CCC @ DDD"]
    assert seen["index"] == 1
    assert selection == "AAA @ BBB"
    assert [game["game_pk"] for game in selected_games] == [1]


def test_game_selection_summary_prepares_no_detail_games():
    slate = [{"away_team": "AAA", "home_team": "BBB", "game_pk": 1}]
    import mlb_dashboard.local_app as local_app

    original = local_app.st.sidebar.selectbox
    try:
        local_app.st.sidebar.selectbox = lambda label, options, index=0: "Slate Summary"
        selection, selected_games = _local_game_selection(slate)
    finally:
        local_app.st.sidebar.selectbox = original

    assert selection == "Slate Summary"
    assert selected_games == []


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
                "hc_x": 90.0,
                "stand": "R",
                "balls": 1,
                "strikes": 2,
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


def test_pitch_level_rolling_windows_are_complete():
    rows = []
    for game_idx in range(1, 17):
        rows.append(
            {
                "batter": 100,
                "player_name": "Rolling Hitter",
                "pitcher": 200,
                "pitcher_name": "Rolling Pitcher",
                "game_pk": game_idx,
                "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=game_idx),
                "game_year": 2026,
                "stand": "R",
                "p_throws": "R",
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "bb_type": "fly_ball" if game_idx % 3 == 0 else "ground_ball",
                "description": "swinging_strike" if game_idx % 5 == 0 else "called_strike",
                "launch_speed": 101.0 if game_idx % 4 == 0 else 96.0,
                "launch_angle": 26.0 if game_idx % 4 == 0 else 10.0 + game_idx,
                "hc_x": 90.0,
                "estimated_woba_using_speedangle": 0.300 + (game_idx / 1000.0),
                "release_speed": 91.0 + (game_idx / 10.0),
                "release_spin_rate": 2300.0,
                "at_bat_number": game_idx,
                "pitch_number": 1,
                "balls": 0,
                "strikes": 1,
                "is_tracked_bbe": True,
                "is_barrel": game_idx % 4 == 0,
                "is_pulled_barrel": game_idx % 8 == 0,
                "is_hard_hit": game_idx % 2 == 0,
                "is_fly_ball": game_idx % 3 == 0,
                "is_batted_ball": True,
                "launch_angle_value": 10.0 + game_idx,
                "xwoba_value": 0.300 + (game_idx / 1000.0),
                "release_speed_value": 91.0 + (game_idx / 10.0),
            }
        )
    live_pitch_mix = pd.DataFrame(rows)

    hitter_rolling = compute_hitter_rolling(live_pitch_mix)
    pitcher_rolling = compute_pitcher_rolling(live_pitch_mix)

    assert hitter_rolling["rolling_window"].tolist() == ["Rolling 5", "Rolling 10", "Rolling 15"]
    assert pitcher_rolling["rolling_window"].tolist() == ["Rolling 5", "Rolling 10", "Rolling 15"]
    assert set(hitter_rolling["games_in_window"].tolist()) == {5, 10, 15}
    assert set(pitcher_rolling["games_in_window"].tolist()) == {5, 10, 15}
    assert hitter_rolling[["pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle", "xwoba"]].notna().all().all()
    assert pitcher_rolling[["avg_release_speed", "barrel_bip_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle"]].notna().all().all()
    assert pitcher_rolling["player_name"].eq("Rolling Pitcher").all()


def test_local_source_payload_respects_target_date(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    rows = [
        {"game_date": "2026-04-01", "game_year": 2026, "game_pk": 1, "batter": 10, "pitcher": 20, "player_name": "Pitcher A", "pitch_type": "FF", "pitch_name": "4-Seam Fastball", "stand": "R", "p_throws": "R", "description": "swinging_strike", "launch_speed": 100.0, "launch_angle": 25.0, "bb_type": "fly_ball", "estimated_woba_using_speedangle": 0.6, "at_bat_number": 1, "pitch_number": 1},
        {"game_date": "2026-04-02", "game_year": 2026, "game_pk": 2, "batter": 10, "pitcher": 20, "player_name": "Pitcher A", "pitch_type": "FF", "pitch_name": "4-Seam Fastball", "stand": "R", "p_throws": "R", "description": "called_strike", "launch_speed": 101.0, "launch_angle": 26.0, "bb_type": "fly_ball", "estimated_woba_using_speedangle": 0.7, "at_bat_number": 1, "pitch_number": 1},
    ]
    write_statcast_events(config, pd.DataFrame(rows))

    payload = load_local_source_payload(config, target_date=pd.Timestamp("2026-04-01").date())

    assert payload.live_pitch_mix["game_pk"].tolist() == [1]
    assert payload.hitter_rolling["games_in_window"].max() == 1


def test_local_tracking_upsert_is_idempotent(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    snapshots = pd.DataFrame(
        [
            {"slate_date": pd.Timestamp("2026-04-01").date(), "game_pk": 1, "batter_id": 10, "split_key": "overall", "recent_window": "season", "weighted_mode": "weighted", "matchup_score": 50.0},
            {"slate_date": pd.Timestamp("2026-04-01").date(), "game_pk": 1, "batter_id": 10, "split_key": "overall", "recent_window": "season", "weighted_mode": "weighted", "matchup_score": 60.0},
        ]
    )

    write_tracking_payload(config, snapshots, pd.DataFrame(), pd.DataFrame())
    write_tracking_payload(config, snapshots.tail(1), pd.DataFrame(), pd.DataFrame())
    loaded = read_hitter_snapshots_for_date(config, pd.Timestamp("2026-04-01").date())

    assert len(loaded) == 1
    assert float(loaded["matchup_score"].iloc[0]) == 60.0


def test_local_odds_cache_history_and_latest(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    odds = pd.DataFrame(
        [
            {"fetched_at": "2026-04-01T10:00:00Z", "cache_key": "old", "provider": "test", "event_id": "e1", "commence_time": "2026-04-02T00:00:00Z", "away_team": "A", "home_team": "B", "sportsbook": "Book", "sportsbook_key": "book", "market_key": "batter_home_runs", "market": "HR", "player_name": "Batter One", "odds_american": 300, "line": 0.5, "player_event_market_key": "p1"},
            {"fetched_at": "2026-04-01T11:00:00Z", "cache_key": "new", "provider": "test", "event_id": "e1", "commence_time": "2026-04-02T00:00:00Z", "away_team": "A", "home_team": "B", "sportsbook": "Book", "sportsbook_key": "book", "market_key": "batter_home_runs", "market": "HR", "player_name": "Batter One", "odds_american": 280, "line": 0.5, "player_event_market_key": "p1"},
        ]
    )

    write_props_odds_snapshot(config, odds)
    history = read_prop_odds_history(config, pd.Timestamp("2026-04-02").date(), pd.Timestamp("2026-04-02").date(), ("batter_home_runs",))
    latest = read_latest_prop_odds_snapshot(config, pd.Timestamp("2026-04-02").date(), ("batter_home_runs",))

    assert len(history) == 2
    assert latest["cache_key"].unique().tolist() == ["new"]


def test_exit_velo_reader_keeps_last_25_games_per_batter(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    rows = []
    for idx in range(30):
        rows.append(
            {
                "game_date": (pd.Timestamp("2026-04-01") + pd.Timedelta(days=idx)).date().isoformat(),
                "game_year": 2026,
                "game_pk": idx + 1,
                "batter": 10,
                "batter_name": "EV Hitter",
                "pitcher": 20,
                "player_name": "EV Pitcher",
                "pitcher_name": "EV Pitcher",
                "home_team": "H",
                "away_team": "A",
                "inning_topbot": "Top",
                "stand": "R",
                "p_throws": "L",
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "launch_speed": 95.0 + idx,
                "launch_angle": 10.0 + idx,
                "bb_type": "line_drive",
                "estimated_woba_using_speedangle": 0.4,
                "at_bat_number": 1,
                "pitch_number": 1,
            }
        )
    write_statcast_events(config, pd.DataFrame(rows))

    events = read_hitter_exit_velo_events(config)

    assert events["game_pk"].nunique() == 25
    assert events["game_pk"].min() == 6
    assert events["team"].eq("A").all()


def test_per_game_artifacts_are_scoped_to_game(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    context = BuildContext(config=config, target_date=pd.Timestamp("2026-04-15").date(), csv_dir=tmp_path)
    schedule = [
        {"game_pk": 1, "away_team": "AAA", "home_team": "BBB", "away_probable_pitcher_id": 20, "home_probable_pitcher_id": 21},
        {"game_pk": 2, "away_team": "CCC", "home_team": "DDD", "away_probable_pitcher_id": 30, "home_probable_pitcher_id": 31},
    ]
    rosters = [
        {"team": "AAA", "player_id": 10, "player_name": "Away Hitter"},
        {"team": "BBB", "player_id": 11, "player_name": "Home Hitter"},
        {"team": "CCC", "player_id": 12, "player_name": "Other Hitter"},
    ]
    hitter_snapshots = pd.DataFrame(
        [
            {"game_pk": 1, "team": "AAA", "batter_id": 10, "hitter_name": "Away Hitter", "matchup_score": 70},
            {"game_pk": 2, "team": "CCC", "batter_id": 12, "hitter_name": "Other Hitter", "matchup_score": 60},
        ]
    )
    pitcher_snapshots = pd.DataFrame(
        [
            {"game_pk": 1, "team": "AAA", "pitcher_id": 20, "pitcher_name": "Away Pitcher", "pitcher_score": 55},
            {"game_pk": 2, "team": "CCC", "pitcher_id": 30, "pitcher_name": "Other Pitcher", "pitcher_score": 50},
        ]
    )
    board_winners = pd.DataFrame([{"game_pk": 1, "board_name": "Best", "board_rank": 1, "batter_id": 10}])
    hitter_rolling = pd.DataFrame([{"player_name": "Away Hitter", "rolling_window": "Rolling 5"}])
    pitcher_rolling = pd.DataFrame([{"player_name": "Away Pitcher", "rolling_window": "Rolling 5"}])
    pitcher_summary = pd.DataFrame([{"pitcher_id": 20, "batter_side_key": "all"}, {"pitcher_id": 30, "batter_side_key": "all"}])
    pitcher_arsenal = pd.DataFrame([{"pitcher_id": 20, "pitch_name": "Fastball"}, {"pitcher_id": 30, "pitch_name": "Slider"}])
    pitcher_arsenal_by_hand = pd.DataFrame([{"pitcher_id": 20, "batter_side_key": "vs_lhh", "pitch_name": "Fastball"}])
    pitcher_usage = pd.DataFrame([{"pitcher_id": 20, "batter_side_key": "all", "pitch_name": "Fastball"}])
    batter_zones = pd.DataFrame([{"batter_id": 10, "zone": 5}, {"batter_id": 12, "zone": 6}])
    pitcher_zones = pd.DataFrame([{"pitcher_id": 20, "zone": 5}, {"pitcher_id": 30, "zone": 6}])
    batter_family = pd.DataFrame([{"batter_id": 10, "pitch_family": "fastball"}, {"batter_id": 12, "pitch_family": "breaking"}])
    pitcher_family = pd.DataFrame([{"pitcher_id": 20, "pitch_family": "fastball"}, {"pitcher_id": 30, "pitch_family": "breaking"}])
    pitcher_movement = pd.DataFrame([{"pitcher_id": 20, "pitch_name": "Fastball"}, {"pitcher_id": 30, "pitch_name": "Slider"}])

    _save_per_game_files(
        context,
        schedule,
        rosters,
        hitter_snapshots,
        pitcher_snapshots,
        board_winners,
        hitter_rolling,
        pitcher_rolling,
        pitcher_summary,
        pitcher_arsenal,
        pitcher_arsenal_by_hand,
        pitcher_usage,
        batter_zones,
        pitcher_zones,
        batter_family,
        pitcher_family,
        pitcher_movement,
    )

    game_dir = config.daily_dir / "2026-04-15" / "games" / "1"
    matchup = pd.read_parquet(game_dir / "matchup.parquet")
    zones = pd.read_parquet(game_dir / "zones.parquet")
    detail = pd.read_parquet(game_dir / "pitcher_detail.parquet")

    assert {"matchup.parquet", "rolling.parquet", "pitcher_detail.parquet", "zones.parquet", "pitch_shape.parquet", "exports.parquet"}.issubset({path.name for path in game_dir.iterdir()})
    assert set(matchup["game_pk"].dropna().astype(int)) == {1}
    assert 12 not in set(pd.to_numeric(zones.get("batter_id"), errors="coerce").dropna().astype(int))
    assert 30 not in set(pd.to_numeric(detail.get("pitcher_id"), errors="coerce").dropna().astype(int))
