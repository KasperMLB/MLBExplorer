from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PILLOW = True
except ImportError:  # pragma: no cover
    HAS_PILLOW = False

try:
    from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode, JsCode
    HAS_AGGRID = True
except ImportError:  # pragma: no cover
    HAS_AGGRID = False

PERCENT_COLUMNS = {
    "swstr_pct",
    "pulled_barrel_pct",
    "barrel_bbe_pct",
    "barrel_bip_pct",
    "fb_pct",
    "hard_hit_pct",
    "usage_pct",
    "gb_pct",
    "All counts",
    "Early count",
    "Even count",
    "Pitcher ahead",
    "Pitcher behind",
    "Two-strike",
    "Pre two-strike",
    "Full count",
}
RATE_COLUMNS = {"xwoba", "xwoba_con", "avg_launch_angle", "avg_release_speed", "avg_spin_rate", "gb_fb_ratio", "matchup_score", "ceiling_score", "pitcher_score", "zone_fit_score"}
LOWER_IS_BETTER = {"swstr_pct"}
HIGHER_IS_BETTER = {
    "pulled_barrel_pct",
    "barrel_bbe_pct",
    "barrel_bip_pct",
    "fb_pct",
    "hard_hit_pct",
    "usage_pct",
    "xwoba",
    "xwoba_con",
    "avg_release_speed",
    "avg_spin_rate",
    "matchup_score",
    "ceiling_score",
    "zone_fit_score",
    "gb_pct",
    "gb_fb_ratio",
}
TARGET_COLUMNS = {"avg_launch_angle": (20.0, 27.5, 35.0)}
DISPLAY_LABELS = {
    "game": "Game",
    "split_label": "Split",
    "hitter_name": "Hitter",
    "pitcher_name": "Pitcher",
    "team": "Team",
    "pitch_count": "Pitches",
    "bip": "BIP",
    "xwoba": "xwOBA",
    "xwoba_con": "xwOBAcon",
    "swstr_pct": "SwStr%",
    "pulled_barrel_pct": "PulledBrl%",
    "barrel_bbe_pct": "Brl/BBE%",
    "barrel_bip_pct": "Brl/BIP%",
    "fb_pct": "FB%",
    "gb_pct": "GB%",
    "gb_fb_ratio": "GB/FB",
    "hard_hit_pct": "HH%",
    "avg_launch_angle": "LA",
    "avg_release_speed": "Velo",
    "avg_spin_rate": "Spin",
    "usage_pct": "Usage%",
    "pitch_name": "Pitch",
    "p_throws": "Throws",
    "likely_starter_score": "Likely",
    "matchup_score": "Matchup",
    "ceiling_score": "Ceiling",
    "zone_fit_score": "Zone Fit",
    "pitcher_score": "Pitch Score",
    "player_name": "Player",
    "rolling_window": "Window",
    "games_in_window": "Games",
    "sample_size": "Sample",
    "pitch_type": "Pitch Type",
    "zone": "Zone",
    "pitch_family": "Family",
    "zone_bucket": "Zone",
    "whiff_rate": "Whiff%",
    "ball_in_play_rate": "BIP%",
    "hit_rate": "Hit%",
    "hr_rate": "HR%",
    "damage_rate": "Damage%",
    "usage_rate_overall": "Usage%",
    "hit_allowed_rate": "Hit Allowed%",
    "hr_allowed_rate": "HR Allowed%",
    "damage_allowed_rate": "Damage Allowed%",
    "xwoba_allowed": "xwOBA Allowed",
    "batter_id": "Batter",
    "pitcher_id": "Pitcher ID",
    "usage_rate": "Usage%",
}
INTEGER_COLUMNS = {"pitch_count", "bip", "likely_starter_score"}

SHORT_COLUMNS = {"team", "p_throws", "bip", "pitch_count", "likely_starter_score"}
MEDIUM_COLUMNS = {
    "split_label",
    "xwoba",
    "xwoba_con",
    "swstr_pct",
    "pulled_barrel_pct",
    "barrel_bbe_pct",
    "barrel_bip_pct",
    "fb_pct",
    "gb_pct",
    "hard_hit_pct",
    "avg_launch_angle",
    "avg_release_speed",
    "avg_spin_rate",
    "usage_pct",
    "matchup_score",
    "ceiling_score",
    "pitcher_score",
    "whiff_rate",
    "ball_in_play_rate",
    "hit_rate",
    "hr_rate",
    "damage_rate",
    "usage_rate_overall",
    "hit_allowed_rate",
    "hr_allowed_rate",
    "damage_allowed_rate",
    "xwoba_allowed",
}

PITCHER_SUMMARY_TABLE_COLUMNS = [
    "split_label",
    "p_throws",
    "pitch_count",
    "bip",
    "xwoba",
    "swstr_pct",
    "pulled_barrel_pct",
    "barrel_bip_pct",
    "fb_pct",
    "hard_hit_pct",
    "avg_launch_angle",
]
LONG_COLUMNS = {"hitter_name", "pitcher_name", "game", "pitch_name"}

ZONE_RECTANGLES = {
    11: (1, 0, 3, 1),
    12: (1, 4, 3, 1),
    13: (0, 1, 1, 3),
    14: (4, 1, 1, 3),
    1: (1, 1, 1, 1),
    2: (2, 1, 1, 1),
    3: (3, 1, 1, 1),
    4: (1, 2, 1, 1),
    5: (2, 2, 1, 1),
    6: (3, 2, 1, 1),
    7: (1, 3, 1, 1),
    8: (2, 3, 1, 1),
    9: (3, 3, 1, 1),
}


def _format_value(column: str, value: object, export_mode: bool = False) -> str:
    if pd.isna(value):
        return "-"
    if column in INTEGER_COLUMNS:
        return f"{int(float(value)):,}"
    if column in PERCENT_COLUMNS:
        decimals = 3 if export_mode else 1
        return f"{float(value):.{decimals}%}"
    if column == "avg_spin_rate":
        decimals = 0 if not export_mode else 3
        return f"{float(value):.{decimals}f}"
    if column in RATE_COLUMNS:
        decimals = 3 if export_mode else (3 if "xwoba" in column or column in {"matchup_score", "ceiling_score", "zone_fit_score"} else 1)
        return f"{float(value):.{decimals}f}"
    if isinstance(value, float):
        decimals = 3 if export_mode else 2
        return f"{value:.{decimals}f}"
    return str(value)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, channel)):02x}" for channel in rgb)


def _blend_color(start: str, end: str, ratio: float) -> str:
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    clamped = max(0.0, min(1.0, ratio))
    blended = tuple(round(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * clamped) for i in range(3))
    return _rgb_to_hex(blended)


def _ease_ratio(ratio: float) -> float:
    clamped = max(0.0, min(1.0, ratio))
    centered = (clamped - 0.5) * 2.0
    curved = 0.5 + 0.5 * (abs(centered) ** 0.8) * (1 if centered >= 0 else -1)
    return max(0.0, min(1.0, curved))


def _diverging_heatmap_hex(ratio: float) -> str:
    bad = "#c94b4b"
    mid = "#f2dc62"
    good = "#2f8f4e"
    curved = _ease_ratio(ratio)
    if curved <= 0.5:
        inner_ratio = curved / 0.5
        return _blend_color(bad, mid, inner_ratio)
    inner_ratio = (curved - 0.5) / 0.5
    return _blend_color(mid, good, inner_ratio)


def _zone_heatmap_hex(ratio: float) -> str:
    cold = "#58a7e4"
    mid = "#a8b0ad"
    hot = "#ef8d32"
    curved = _ease_ratio(ratio)
    if curved <= 0.5:
        return _blend_color(cold, mid, curved / 0.5)
    return _blend_color(mid, hot, (curved - 0.5) / 0.5)


def _two_sided_target_ratio(value: float, low: float, ideal: float, high: float) -> float:
    if value <= ideal:
        return 1.0 - min(1.0, max(0.0, (ideal - value) / max(ideal - low, 1e-9)))
    return 1.0 - min(1.0, max(0.0, (value - ideal) / max(high - ideal, 1e-9)))


def _background_hex(
    column: str,
    value: object,
    series: pd.Series,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> str | None:
    if pd.isna(value):
        return None
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return None

    numeric_value = float(value)
    if column in TARGET_COLUMNS:
        low, ideal, high = TARGET_COLUMNS[column]
        ratio = _two_sided_target_ratio(numeric_value, low, ideal, high)
    else:
        col_min = float(numeric_series.min())
        col_max = float(numeric_series.max())
        if abs(col_max - col_min) < 1e-9:
            ratio = 0.5
        else:
            ratio = (numeric_value - col_min) / (col_max - col_min)
        if lower_is_better and column in lower_is_better:
            ratio = 1.0 - ratio
        elif higher_is_better and column not in higher_is_better:
            ratio = 1.0 - ratio

    return _diverging_heatmap_hex(ratio)


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=DISPLAY_LABELS)


def _prepare_grid_frame(
    frame: pd.DataFrame,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> pd.DataFrame:
    prepared = frame.copy()
    for column in frame.columns:
        if column in PERCENT_COLUMNS or column in RATE_COLUMNS:
            prepared[f"__style_{column}"] = [
                _background_hex(
                    column,
                    value,
                    frame[column],
                    lower_is_better=lower_is_better or LOWER_IS_BETTER,
                    higher_is_better=higher_is_better or HIGHER_IS_BETTER,
                ) or ""
                for value in frame[column]
            ]
    return prepared


def _formatted_length(column: str, value: object) -> int:
    return len(_format_value(column, value))


def _column_width(column: str, series: pd.Series) -> tuple[int, int, int]:
    header_name = DISPLAY_LABELS.get(column, column)
    content_lengths = [_formatted_length(column, value) for value in series.head(50)]
    max_len = max([len(header_name), *content_lengths, 4])
    if column in SHORT_COLUMNS:
        min_width = 70
        max_width = 110
        scale = 8
    elif column in LONG_COLUMNS:
        min_width = 140
        max_width = 260
        scale = 9
    elif column in MEDIUM_COLUMNS or column in PERCENT_COLUMNS or column in RATE_COLUMNS:
        min_width = 92
        max_width = 135
        scale = 8
    else:
        min_width = 100
        max_width = 180
        scale = 8
    width = max(min_width, min(max_width, max_len * scale + 24))
    return width, min_width, max_width


def render_metric_grid(
    frame: pd.DataFrame,
    key: str,
    height: int = 320,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
    use_lightweight: bool = False,
) -> pd.DataFrame:
    if frame.empty:
        st.info("No data available for this selection.")
        return frame
    if use_lightweight or not HAS_AGGRID:
        display_frame = _display_frame(frame).copy()
        source_by_label = {DISPLAY_LABELS.get(column, column): column for column in frame.columns}
        formatters = {DISPLAY_LABELS.get(column, column): (lambda value, source=column: _format_value(source, value)) for column in frame.columns}
        styles = pd.DataFrame("", index=display_frame.index, columns=display_frame.columns)
        for source_column in frame.columns:
            display_column = DISPLAY_LABELS.get(source_column, source_column)
            if source_column in PERCENT_COLUMNS or source_column in RATE_COLUMNS:
                styles[display_column] = [
                    (
                        f"background-color: {_background_hex(source_column, value, frame[source_column], lower_is_better=lower_is_better or LOWER_IS_BETTER, higher_is_better=higher_is_better or HIGHER_IS_BETTER)}; color: #1f1f1f"
                        if _background_hex(source_column, value, frame[source_column], lower_is_better=lower_is_better or LOWER_IS_BETTER, higher_is_better=higher_is_better or HIGHER_IS_BETTER)
                        else ""
                    )
                    for value in frame[source_column]
                ]
        st.dataframe(
            display_frame.style.format(formatters).apply(lambda _: styles, axis=None),
            hide_index=True,
            use_container_width=True,
            height=height,
        )
        return frame

    if not HAS_AGGRID:
        st.error("Install `streamlit-aggrid` to render the matchup tables.")
        return frame

    prepared = _prepare_grid_frame(frame, lower_is_better=lower_is_better, higher_is_better=higher_is_better)
    builder = GridOptionsBuilder.from_dataframe(prepared)
    builder.configure_default_column(sortable=True, resizable=True, filter=False)
    builder.configure_grid_options(
        tooltipShowDelay=0,
    )

    cell_style = JsCode(
        """
        function(params) {
          const styleKey = "__style_" + params.colDef.field;
          const bg = params.data ? params.data[styleKey] : "";
          if (bg) {
            return {backgroundColor: bg, color: "#1f1f1f"};
          }
          return {color: "#1f1f1f"};
        }
        """
    )
    percent_formatter = JsCode(
        """
        function(params) {
          if (params.value === null || params.value === undefined || params.value === "") { return "-"; }
          return (Number(params.value) * 100).toFixed(1) + "%";
        }
        """
    )
    rate_formatter = JsCode(
        """
        function(params) {
          if (params.value === null || params.value === undefined || params.value === "") { return "-"; }
          return Number(params.value).toFixed(3);
        }
        """
    )
    one_decimal_formatter = JsCode(
        """
        function(params) {
          if (params.value === null || params.value === undefined || params.value === "") { return "-"; }
          return Number(params.value).toFixed(1);
        }
        """
    )
    integer_formatter = JsCode(
        """
        function(params) {
          if (params.value === null || params.value === undefined || params.value === "") { return "-"; }
          return Math.round(Number(params.value)).toLocaleString();
        }
        """
    )

    for column in frame.columns:
        header_name = DISPLAY_LABELS.get(column, column)
        formatter = None
        width, min_width, max_width = _column_width(column, frame[column])
        if column in PERCENT_COLUMNS:
            formatter = percent_formatter
        elif column in {"xwoba", "xwoba_con", "matchup_score", "ceiling_score", "pitcher_score"}:
            formatter = rate_formatter
        elif column in {"avg_launch_angle", "avg_release_speed", "gb_fb_ratio"}:
            formatter = one_decimal_formatter
        elif column in {"avg_spin_rate", *INTEGER_COLUMNS}:
            formatter = integer_formatter
        builder.configure_column(
            column,
            header_name=header_name,
            cellStyle=cell_style if column in PERCENT_COLUMNS or column in RATE_COLUMNS else None,
            valueFormatter=formatter,
            width=width,
            minWidth=min_width,
            maxWidth=max_width,
        )
        style_col = f"__style_{column}"
        if style_col in prepared.columns:
            builder.configure_column(style_col, hide=True)

    grid_options = builder.build()
    response = AgGrid(
        prepared,
        gridOptions=grid_options,
        height=height,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme="streamlit",
        key=key,
    )
    returned = pd.DataFrame(response["data"])
    if returned.empty:
        return frame
    return returned.loc[:, frame.columns]


def build_pitcher_summary_table(pitcher_summary_by_hand: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    split_map = [
        ("all", "All"),
        ("vs_rhh", "vs RHH"),
        ("vs_lhh", "vs LHH"),
    ]
    for side_key, label in split_map:
        side_frame = pitcher_summary_by_hand.loc[pitcher_summary_by_hand["batter_side_key"] == side_key].copy()
        if side_frame.empty:
            row = {column: None for column in PITCHER_SUMMARY_TABLE_COLUMNS if column != "split_label"}
            row["split_label"] = label
            rows.append(row)
            continue
        first_row = side_frame.iloc[0]
        row = {"split_label": label}
        for column in PITCHER_SUMMARY_TABLE_COLUMNS:
            if column == "split_label":
                continue
            row[column] = first_row.get(column)
        rows.append(row)
    return pd.DataFrame(rows, columns=PITCHER_SUMMARY_TABLE_COLUMNS)


def render_matchup_header(game: dict) -> None:
    st.subheader(f"{game['away_team']} @ {game['home_team']}")
    st.caption(f"{game.get('status', 'Scheduled')} | Game PK: {game['game_pk']}")


def _zone_map_lookup(zone_map: pd.DataFrame) -> dict[int, dict[str, float]]:
    if zone_map.empty:
        return {}
    lookup: dict[int, dict[str, float]] = {}
    for _, row in zone_map.iterrows():
        zone = row.get("zone")
        if pd.isna(zone):
            continue
        lookup[int(float(zone))] = {
            "zone_value": row.get("zone_value"),
            "display_value": row.get("display_value"),
            "sample_size": row.get("sample_size"),
        }
    return lookup


def _draw_home_plate(draw: ImageDraw.ImageDraw, center_x: int, top_y: int, fill: str, outline: str) -> None:
    points = [
        (center_x - 34, top_y),
        (center_x + 34, top_y),
        (center_x + 48, top_y + 24),
        (center_x, top_y + 42),
        (center_x - 48, top_y + 24),
    ]
    draw.polygon(points, fill=fill, outline=outline)


def _build_zone_heatmap_image(title: str, subtitle: str, zone_map: pd.DataFrame, value_mode: str = "percent") -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for heatmap rendering.")
    font = ImageFont.load_default()
    width = 620
    height = 700
    image = Image.new("RGB", (width, height), "#1f2d31")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 10, width - 10, height - 10), radius=22, fill="#2c3f45", outline="#41565d", width=2)
    draw.text((34, 28), title, fill="#f4f6f7", font=font)
    draw.text((34, 50), subtitle, fill="#b8c4c8", font=font)

    legend_x = width - 112
    legend_top = 118
    legend_labels = [("Best", "#ef8d32"), ("", "#c39656"), ("", "#a8b0ad"), ("", "#79a7ca"), ("Worst", "#58a7e4")]
    draw.text((legend_x, legend_top - 24), "Zone Score", fill="#d5dcde", font=font)
    for idx, (label, color) in enumerate(legend_labels):
        y = legend_top + idx * 52
        draw.rounded_rectangle((legend_x, y, legend_x + 36, y + 36), radius=7, fill=color)
        if label:
            draw.text((legend_x + 50, y + 8), label, fill="#f4f6f7", font=font)

    grid_origin_x = 64
    grid_origin_y = 110
    unit = 86
    gap = 6
    lookup = _zone_map_lookup(zone_map)
    zone_values = [value["zone_value"] for value in lookup.values() if value.get("zone_value") is not None and not pd.isna(value.get("zone_value"))]
    min_val = min(zone_values) if zone_values else 0.0
    max_val = max(zone_values) if zone_values else 1.0

    def zone_ratio(value: float | None) -> float:
        if value is None or pd.isna(value):
            return 0.5
        if abs(max_val - min_val) < 1e-9:
            return 0.5
        return (float(value) - min_val) / (max_val - min_val)

    for zone, (grid_x, grid_y, span_x, span_y) in ZONE_RECTANGLES.items():
        x0 = grid_origin_x + grid_x * (unit + gap)
        y0 = grid_origin_y + grid_y * (unit + gap)
        x1 = x0 + span_x * unit + (span_x - 1) * gap
        y1 = y0 + span_y * unit + (span_y - 1) * gap
        zone_info = lookup.get(zone, {})
        color = _zone_heatmap_hex(zone_ratio(zone_info.get("zone_value")))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=color, outline="#d5d8d9", width=2)
        sample_size = zone_info.get("sample_size")
        display_value = zone_info.get("display_value")
        label = "-"
        if display_value is not None and not pd.isna(display_value):
            if value_mode == "percent":
                label = f"{float(display_value) * 100:.0f}"
            else:
                label = f"{float(display_value) * 100:.0f}"
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_x = x0 + ((x1 - x0) - (label_bbox[2] - label_bbox[0])) / 2
        label_y = y0 + ((y1 - y0) - (label_bbox[3] - label_bbox[1])) / 2 - 6
        draw.text((label_x, label_y), label, fill="#111618", font=font)
        if sample_size is not None and not pd.isna(sample_size):
            sample_text = f"{int(float(sample_size))}"
            sample_bbox = draw.textbbox((0, 0), sample_text, font=font)
            sample_x = x0 + ((x1 - x0) - (sample_bbox[2] - sample_bbox[0])) / 2
            sample_y = label_y + 18
            draw.text((sample_x, sample_y), sample_text, fill="#263338", font=font)

    strike_left = grid_origin_x + (unit + gap)
    strike_top = grid_origin_y + (unit + gap)
    strike_right = strike_left + 3 * unit + 2 * gap
    strike_bottom = strike_top + 3 * unit + 2 * gap
    draw.rectangle((strike_left - 3, strike_top - 3, strike_right + 3, strike_bottom + 3), outline="#f3f7f8", width=3)
    _draw_home_plate(draw, (strike_left + strike_right) // 2, strike_bottom + 32, fill="#d9dedf", outline="#f3f7f8")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_zone_heatmap(title: str, subtitle: str, zone_map: pd.DataFrame, value_mode: str = "percent") -> None:
    if zone_map.empty:
        st.info("No zone heatmap data available.")
        return
    st.image(_build_zone_heatmap_image(title, subtitle, zone_map, value_mode=value_mode), use_container_width=False)


def _table_section_height(title: str, frame: pd.DataFrame) -> int:
    return 32 + 28 + 26 * (len(frame) + 1)


def _export_section_height_estimate(title: str, frame: pd.DataFrame) -> int:
    lowered = title.lower()
    if "summary" in lowered:
        return _table_section_height(title, frame) + 24
    return 54 + len(frame) * 62 + 36


def _summary_card_color(column: str, row: pd.Series, lower_is_better: set[str] | None, higher_is_better: set[str] | None) -> str:
    if column not in PERCENT_COLUMNS and column not in RATE_COLUMNS:
        return "#f6f8fb"
    value = row.get(column)
    if pd.isna(value):
        return "#f6f8fb"
    if column in TARGET_COLUMNS:
        return _background_hex(column, value, pd.Series([value]), lower_is_better=lower_is_better, higher_is_better=higher_is_better) or "#f6f8fb"
    thresholds = {
        "xwoba": (0.24, 0.30, True),
        "swstr_pct": (0.10, 0.14, False),
        "pulled_barrel_pct": (0.03, 0.08, True),
        "barrel_bbe_pct": (0.05, 0.10, True),
        "barrel_bip_pct": (0.04, 0.09, True),
        "fb_pct": (0.20, 0.33, True),
        "hard_hit_pct": (0.34, 0.48, True),
    }
    if column not in thresholds:
        return "#f6f8fb"
    low, high, invert = thresholds[column]
    ratio = (float(value) - low) / max(high - low, 1e-9)
    ratio = max(0.0, min(1.0, ratio))
    if invert:
        ratio = 1.0 - ratio
    return _diverging_heatmap_hex(ratio)


def _infer_summary_stats(frame: pd.DataFrame) -> list[tuple[str, str]]:
    preferred = [
        ("Throws", "p_throws"),
        ("Pitches", "pitch_count"),
        ("BIP", "bip"),
        ("xwOBA", "xwoba"),
        ("SwStr%", "swstr_pct"),
        ("PulledBrl%", "pulled_barrel_pct"),
        ("Brl/BIP%", "barrel_bip_pct"),
        ("FB%", "fb_pct"),
        ("HH%", "hard_hit_pct"),
        ("LA", "avg_launch_angle"),
    ]
    return [(label, column) for label, column in preferred if column in frame.columns]


def _draw_summary_cards(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    font: ImageFont.ImageFont,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> int:
    if frame.empty:
        return top
    row = frame.iloc[0]
    stats = _infer_summary_stats(frame)
    draw.text((left, top), title, fill="#102542", font=font)
    y = top + 22
    cols = 5
    card_gap = 10
    card_width = max(120, int((width - (card_gap * (cols - 1))) / cols))
    card_height = 62
    for idx, (label, column_name) in enumerate(stats):
        col_idx = idx % cols
        row_idx = idx // cols
        x0 = left + col_idx * (card_width + card_gap)
        y0 = y + row_idx * (card_height + 10)
        x1 = x0 + card_width
        y1 = y0 + card_height
        fill = _summary_card_color(column_name, row, lower_is_better, higher_is_better)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=fill, outline="#d9e2ec", width=1)
        draw.text((x0 + 10, y0 + 10), label, fill="#516273", font=font)
        draw.text((x0 + 10, y0 + 34), _format_value(column_name, row.get(column_name), export_mode=True), fill="#1f1f1f", font=font)
    rows = (len(stats) + cols - 1) // cols
    return y + rows * (card_height + 10)


def _limit_frame(frame: pd.DataFrame, rows: int, columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    limited = frame.copy()
    if columns is not None:
        limited = limited[[column for column in columns if column in limited.columns]]
    return limited.head(rows)


def _section_type(title: str) -> str:
    lowered = title.lower()
    if "best matchups" in lowered:
        return "best"
    if "summary" in lowered:
        return "summary"
    if "arsenal" in lowered:
        return "arsenal"
    if "count usage" in lowered:
        return "count"
    if "hitters" in lowered:
        return "hitters"
    return "table"


def _extract_sections(sections: list[dict], kind: str) -> list[dict]:
    return [section for section in sections if _section_type(section["title"]) == kind]


def _extract_first_matching(sections: list[dict], keyword: str) -> dict | None:
    lowered = keyword.lower()
    for section in sections:
        if lowered in section["title"].lower():
            return section
    return None


def _draw_side_by_side_tables(
    draw: ImageDraw.ImageDraw,
    top: int,
    width: int,
    left_title: str,
    left_frame: pd.DataFrame,
    right_title: str,
    right_frame: pd.DataFrame,
    font: ImageFont.ImageFont,
) -> int:
    gap = 24
    table_width = (width - gap) // 2
    left_bottom = _draw_section(draw, top, 18, table_width, left_title, left_frame, None, None, font)
    right_bottom = _draw_section(draw, top, 18 + table_width + gap, table_width, right_title, right_frame, None, None, font)
    return max(left_bottom, right_bottom)


REPORT_BG = "#ffffff"
REPORT_PANEL = "#ffffff"
REPORT_PANEL_ALT = "#f6f8fb"
REPORT_BORDER = "#cfd8e3"
REPORT_TEXT = "#111111"
REPORT_MUTED = "#4b5563"
REPORT_ACCENT = "#1f4e79"


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)
    except Exception:  # pragma: no cover
        return ImageFont.load_default()


def _section_team(title: str) -> str:
    return title.split(" ", 1)[0] if title else ""


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str = REPORT_PANEL, radius: int = 18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=REPORT_BORDER, width=2)


def _text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    font: ImageFont.ImageFont,
    fill: str = REPORT_TEXT,
    stroke_width: int = 0,
    stroke_fill: str = REPORT_TEXT,
) -> None:
    draw.text(position, value, fill=fill, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)


def _measure(draw: ImageDraw.ImageDraw, value: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), value, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _report_table_height(frame: pd.DataFrame, row_height: int = 24, header_height: int = 26) -> int:
    return header_height + row_height * max(len(frame), 1) + 20


def _report_summary_height(frame: pd.DataFrame) -> int:
    stats = _infer_summary_stats(frame)
    rows = max(1, (len(stats) + 1) // 2)
    return 64 + rows * 84


def _filter_section_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()


def _draw_report_header(
    draw: ImageDraw.ImageDraw,
    width: int,
    title: str,
    subtitle: str,
    away_summary: dict | None,
    home_summary: dict | None,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> int:
    hero_box = (24, 24, width - 24, 166)
    _panel(draw, hero_box, fill="#f5f7fa", radius=24)
    _text(draw, (46, 42), "KASPER SCOUTING REPORT", body_font, REPORT_ACCENT)
    _text(draw, (46, 68), title, title_font, REPORT_TEXT)
    _text(draw, (46, 110), subtitle, body_font, REPORT_TEXT)
    starter_y = 132
    if away_summary is not None and not away_summary["frame"].empty:
        row = away_summary["frame"].iloc[0]
        _text(draw, (46, starter_y), f"Away: {_section_team(away_summary['title'])} | {row.get('pitcher_name', 'Starter')} ({row.get('p_throws', '-')})", body_font, REPORT_TEXT)
    if home_summary is not None and not home_summary["frame"].empty:
        row = home_summary["frame"].iloc[0]
        _text(draw, (width // 2 + 40, starter_y), f"Home: {_section_team(home_summary['title'])} | {row.get('pitcher_name', 'Starter')} ({row.get('p_throws', '-')})", body_font, REPORT_TEXT)
    return hero_box[3] + 18


def _draw_matchup_board(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    frame: pd.DataFrame,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> int:
    board_height = 258
    _panel(draw, (left, top, left + width, top + board_height), fill=REPORT_PANEL_ALT)
    _text(draw, (left + 18, top + 16), "Best Matchups", title_font, REPORT_TEXT)
    work = _filter_section_columns(frame.head(3), ["hitter_name", "team", "matchup_score", "xwoba", "swstr_pct", "pulled_barrel_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle"])
    card_top = top + 56
    card_height = 58
    for idx, (_, row) in enumerate(work.iterrows(), start=1):
        y0 = card_top + (idx - 1) * (card_height + 10)
        _panel(draw, (left + 14, y0, left + width - 14, y0 + card_height), fill="#ffffff", radius=14)
        _text(draw, (left + 24, y0 + 14), f"{idx}. {row.get('hitter_name', '-')}", body_font, REPORT_TEXT)
        _text(draw, (left + width - 90, y0 + 20), row.get("team", "-"), body_font, REPORT_TEXT)
        chip_specs = [
            ("matchup_score", "Matchup"),
            ("xwoba", "xwOBA"),
            ("swstr_pct", "SwStr%"),
            ("pulled_barrel_pct", "PulledBrl%"),
            ("hard_hit_pct", "HH%"),
            ("fb_pct", "FB%"),
            ("avg_launch_angle", "LA"),
        ]
        start_x = left + 250
        chip_width = 110
        chip_gap = 10
        for offset, (column, label) in enumerate(chip_specs):
            chip_x = start_x + offset * (chip_width + chip_gap)
            chip_fill = _background_hex(
                column,
                row.get(column),
                work[column],
                lower_is_better=LOWER_IS_BETTER,
                higher_is_better=HIGHER_IS_BETTER,
            ) or "#e8eef5"
            draw.rounded_rectangle((chip_x, y0 + 10, chip_x + chip_width, y0 + 40), radius=12, fill=chip_fill)
            chip_text = f"{label} {_format_value(column, row.get(column), export_mode=True)}"
            _text(draw, (chip_x + 8, y0 + 18), chip_text, body_font, REPORT_TEXT)
    return top + board_height


def _draw_report_summary_cards(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> int:
    panel_height = _report_summary_height(frame)
    _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL)
    _text(draw, (left + 18, top + 14), title, title_font, REPORT_TEXT)
    if frame.empty:
        _text(draw, (left + 18, top + 50), "No summary available", body_font, REPORT_MUTED)
        return top + panel_height
    row = frame.iloc[0]
    stats = _infer_summary_stats(frame)
    cols = 2
    gap = 12
    card_width = (width - 36 - gap) // cols
    card_height = 68
    start_y = top + 48
    for idx, (label, column_name) in enumerate(stats):
        col_idx = idx % cols
        row_idx = idx // cols
        x0 = left + 18 + col_idx * (card_width + gap)
        y0 = start_y + row_idx * (card_height + 10)
        x1 = x0 + card_width
        y1 = y0 + card_height
        fill = _summary_card_color(column_name, row, lower_is_better, higher_is_better)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=fill, outline="#d0d8e0", width=1)
        _text(draw, (x0 + 10, y0 + 10), label, body_font, REPORT_TEXT)
        _text(draw, (x0 + 10, y0 + 36), _format_value(column_name, row.get(column_name), export_mode=True), body_font, REPORT_TEXT)
    return top + panel_height


def _draw_dark_table(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> int:
    if frame.empty:
        panel_height = 76
        _panel(draw, (left, top, left + width, top + panel_height))
        _text(draw, (left + 16, top + 16), title, font, REPORT_TEXT)
        _text(draw, (left + 16, top + 42), "No data available", small_font, REPORT_MUTED)
        return top + panel_height

    header_height = 34
    row_height = 32
    padding_x = 12
    padding_y = 8
    panel_height = 46 + _report_table_height(frame, row_height=row_height, header_height=header_height)
    _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL)
    _text(draw, (left + 16, top + 14), title, font, REPORT_TEXT)

    y = top + 46
    display_frame = _display_frame(frame).copy()
    headers = list(display_frame.columns)
    source_by_label = {DISPLAY_LABELS.get(col, col): col for col in frame.columns}
    formatted_rows = [[_format_value(source_by_label.get(column, column), row[column], export_mode=True) for column in headers] for _, row in display_frame.iterrows()]

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    col_widths: list[int] = []
    for col_idx, header in enumerate(headers):
        max_width = _measure(probe, header, small_font)[0]
        for row in formatted_rows:
            max_width = max(max_width, _measure(probe, row[col_idx], small_font)[0])
        col_widths.append(max_width + padding_x * 2)
    total_width = sum(col_widths)
    if total_width < width - 24:
        extra = (width - 24 - total_width) // max(len(col_widths), 1)
        col_widths = [size + extra for size in col_widths]

    x = left + 12
    for idx, header in enumerate(headers):
        draw.rounded_rectangle((x, y, x + col_widths[idx], y + header_height), radius=8, fill="#dce6f2")
        _text(draw, (x + padding_x, y + 7), header, small_font, REPORT_TEXT)
        x += col_widths[idx]

    for row_idx, row in enumerate(formatted_rows, start=1):
        current_y = y + header_height + (row_idx - 1) * row_height
        x = left + 12
        for col_idx, value in enumerate(row):
            display_label = headers[col_idx]
            source_column = source_by_label.get(display_label, display_label)
            fill = "#ffffff" if row_idx % 2 else "#f7f9fc"
            if source_column in frame.columns and (source_column in PERCENT_COLUMNS or source_column in RATE_COLUMNS):
                fill = _background_hex(
                    source_column,
                    frame.iloc[row_idx - 1][source_column],
                    frame[source_column],
                    lower_is_better=lower_is_better or LOWER_IS_BETTER,
                    higher_is_better=higher_is_better or HIGHER_IS_BETTER,
                ) or fill
            draw.rounded_rectangle((x, current_y, x + col_widths[col_idx], current_y + row_height - 2), radius=6, fill=fill)
            _text(draw, (x + padding_x, current_y + padding_y), value, small_font, REPORT_TEXT)
            x += col_widths[col_idx]
    return top + panel_height


def _identity_columns_for_section(frame: pd.DataFrame, title: str) -> list[str]:
    lowered = title.lower()
    if "hitters" in lowered or "best matchups" in lowered:
        columns = ["hitter_name", "team"]
    elif "arsenal" in lowered or "count usage" in lowered:
        columns = ["pitch_name"]
    elif "summary" in lowered:
        columns = ["pitcher_name", "p_throws"]
    else:
        columns = [frame.columns[0]] if len(frame.columns) else []
    return [column for column in columns if column in frame.columns]


def _metric_columns_for_section(frame: pd.DataFrame, title: str) -> list[str]:
    lowered = title.lower()
    if "best matchups" in lowered:
        columns = ["matchup_score", "xwoba", "swstr_pct", "pulled_barrel_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle"]
    elif "hitters" in lowered:
        columns = ["matchup_score", "xwoba", "pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "avg_launch_angle"]
    elif "arsenal" in lowered:
        columns = ["usage_pct", "swstr_pct", "hard_hit_pct", "avg_release_speed", "avg_spin_rate", "xwoba_con"]
    elif "count usage" in lowered:
        columns = ["All counts", "Early count", "Even count", "Pitcher ahead", "Pitcher behind", "Two-strike", "Pre two-strike", "Full count"]
    elif "summary" in lowered:
        columns = ["pitch_count", "bip", "xwoba", "swstr_pct", "pulled_barrel_pct", "barrel_bip_pct", "fb_pct", "hard_hit_pct", "avg_launch_angle"]
    else:
        columns = [column for column in frame.columns if column not in _identity_columns_for_section(frame, title)]
    return [column for column in columns if column in frame.columns]


def _draw_scouting_rows_section(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> int:
    if frame.empty:
        panel_height = 82
        _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL)
        _text(draw, (left + 16, top + 14), title, font, REPORT_TEXT)
        _text(draw, (left + 16, top + 48), "No data available", small_font, REPORT_TEXT)
        return top + panel_height

    identity_columns = _identity_columns_for_section(frame, title)
    metric_columns = _metric_columns_for_section(frame, title)
    row_height = 52
    panel_height = 54 + len(frame) * (row_height + 10) + 12
    _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL)
    _text(draw, (left + 16, top + 14), title, font, REPORT_TEXT)

    y = top + 52
    identity_width = max(170, min(290, int(width * 0.27)))
    chip_area_left = left + 18 + identity_width + 14
    chip_width = 92
    chip_gap = 8

    for _, row in frame.iterrows():
        row_top = y
        row_bottom = row_top + row_height
        _panel(draw, (left + 12, row_top, left + width - 12, row_bottom), fill="#ffffff", radius=14)

        primary = _format_value(identity_columns[0], row.get(identity_columns[0]), export_mode=True) if identity_columns else "-"
        secondary_parts = [
            _format_value(column, row.get(column), export_mode=True)
            for column in identity_columns[1:]
            if str(_format_value(column, row.get(column), export_mode=True)).strip() != "-"
        ]
        _text(draw, (left + 26, row_top + 10), primary, small_font, REPORT_TEXT)
        if secondary_parts:
            _text(draw, (left + 26, row_top + 28), " | ".join(secondary_parts), small_font, REPORT_TEXT)

        for idx, column in enumerate(metric_columns):
            chip_x = chip_area_left + idx * (chip_width + chip_gap)
            if chip_x + chip_width > left + width - 24:
                break
            chip_fill = _background_hex(
                column,
                row.get(column),
                frame[column],
                lower_is_better=lower_is_better or LOWER_IS_BETTER,
                higher_is_better=higher_is_better or HIGHER_IS_BETTER,
            ) or "#e8eef5"
            draw.rounded_rectangle((chip_x, row_top + 10, chip_x + chip_width, row_top + 40), radius=12, fill=chip_fill)
            chip_label = DISPLAY_LABELS.get(column, column)
            chip_text = f"{chip_label} {_format_value(column, row.get(column), export_mode=True)}"
            _text(draw, (chip_x + 8, row_top + 18), chip_text, small_font, REPORT_TEXT)
        y += row_height + 10

    return top + panel_height


def _draw_export_section(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    section: dict,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> int:
    title = section["title"]
    frame = section["frame"]
    return _draw_dark_table(
        draw,
        top,
        left,
        width,
        title,
        frame,
        font,
        small_font,
        section.get("lower_is_better"),
        section.get("higher_is_better"),
    )


def _draw_report_stacked_sections(
    draw: ImageDraw.ImageDraw,
    top: int,
    width: int,
    sections: list[dict],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> int:
    y = top
    for section in sections:
        y = _draw_export_section(draw, y, 24, width - 48, section, font, small_font) + 12
    return y


def _draw_report_two_column_sections(
    draw: ImageDraw.ImageDraw,
    top: int,
    width: int,
    left_sections: list[dict],
    right_sections: list[dict],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> int:
    gap = 22
    col_width = (width - 48 - gap) // 2
    left_x = 24
    right_x = left_x + col_width + gap
    left_y = top
    right_y = top
    for section in left_sections:
        left_y = _draw_export_section(draw, left_y, left_x, col_width, section, font, small_font) + 12
    for section in right_sections:
        right_y = _draw_export_section(draw, right_y, right_x, col_width, section, font, small_font) + 12
    return max(left_y, right_y)


def _collect_team_sections(sections: list[dict], kind: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for section in _extract_sections(sections, kind):
        grouped.setdefault(_section_team(section["title"]), []).append(section)
    return grouped


def _exclude_all_split_sections(sections: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    for section in sections:
        lowered = section["title"].lower()
        if "summary all" in lowered or "arsenal all" in lowered or "count usage all" in lowered:
            continue
        filtered.append(section)
    return filtered


def _split_sections_by_hand(sections: list[dict]) -> tuple[list[dict], list[dict]]:
    vs_rhh = [section for section in sections if "vs rhh" in section["title"].lower()]
    vs_lhh = [section for section in sections if "vs lhh" in section["title"].lower()]
    return vs_rhh, vs_lhh


def _build_compact_poster_image(title: str, subtitle: str, sections: list[dict]) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    title_font = _load_font(56, bold=True)
    section_font = _load_font(34, bold=True)
    body_font = _load_font(28, bold=True)
    small_font = _load_font(24, bold=True)
    hitter_title_font = _load_font(38, bold=True)
    hitter_body_font = _load_font(28, bold=True)
    width = 1600
    image = Image.new("RGB", (width, 4200), REPORT_BG)
    draw = ImageDraw.Draw(image)

    best_section = next((section for section in sections if _section_type(section["title"]) == "best"), None)
    summary_sections = _extract_sections(sections, "summary")
    arsenal_by_team = _collect_team_sections(_exclude_all_split_sections(sections), "arsenal")
    hitter_sections = _extract_sections(sections, "hitters")
    away_summary = next((section for section in summary_sections if "summary all" in section["title"].lower()), None)
    home_summary = next((section for section in summary_sections if section is not away_summary and "summary all" in section["title"].lower()), None)

    y = _draw_report_header(draw, width, title, subtitle, away_summary, home_summary, title_font, body_font)

    if best_section is not None:
        best_frame = _filter_section_columns(best_section["frame"], ["hitter_name", "team", "matchup_score", "xwoba", "swstr_pct", "pulled_barrel_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle"])
        y = _draw_export_section(
            draw,
            y,
            24,
            width - 48,
            {
                **best_section,
                "title": "Best Matchups",
                "frame": best_frame,
            },
            section_font,
            body_font,
        ) + 18

    team_order = list(arsenal_by_team.keys())
    for team in team_order:
        team_sections = arsenal_by_team.get(team, [])
        if not team_sections:
            continue
        _text(draw, (24, y), f"{team} Pitcher", title_font, REPORT_TEXT)
        y += 42
        vs_rhh, vs_lhh = _split_sections_by_hand(team_sections)
        y = _draw_report_two_column_sections(draw, y, width, vs_rhh[:1], vs_lhh[:1], section_font, small_font) + 12

    if len(hitter_sections) >= 2:
        for section in hitter_sections[:2]:
            frame = _filter_section_columns(section["frame"], ["hitter_name", "matchup_score", "xwoba", "pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "avg_launch_angle"])
            _text(draw, (24, y), section["title"], title_font, REPORT_TEXT)
            y += 42
            y = _draw_export_section(
                draw,
                y,
                24,
                width - 48,
                {**section, "frame": frame},
                hitter_title_font,
                hitter_body_font,
            ) + 14

    cropped = image.crop((0, 0, width, min(max(y + 30, 1600), image.height)))
    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_carousel_images(title: str, subtitle: str, sections: list[dict]) -> list[bytes]:
    slides: list[bytes] = []
    best_section = next((section for section in sections if _section_type(section["title"]) == "best"), None)
    summary_sections = _extract_sections(sections, "summary")
    arsenal_sections = _exclude_all_split_sections(_extract_sections(sections, "arsenal"))
    count_sections = _exclude_all_split_sections(_extract_sections(sections, "count"))
    hitter_sections = _extract_sections(sections, "hitters")

    overview_sections: list[dict] = []
    if best_section is not None:
        overview_sections.append(
            {
                "title": "Best Matchups",
                "frame": _filter_section_columns(best_section["frame"], ["hitter_name", "team", "matchup_score", "xwoba", "swstr_pct", "pulled_barrel_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle"]),
            }
        )
    overview_sections.extend([section for section in summary_sections if "summary all" in section["title"].lower()])
    slides.append(_build_compact_poster_image(title, f"{subtitle} | Slide 1: Overview", overview_sections))

    pitching_sections: list[dict] = []
    pitching_sections.extend(summary_sections)
    pitching_sections.extend(
        {
            **section,
            "frame": _filter_section_columns(section["frame"], ["pitch_name", "usage_pct", "swstr_pct", "hard_hit_pct", "avg_release_speed", "xwoba_con"]),
        }
        for section in arsenal_sections
    )
    slides.append(build_branded_report_image(title, f"{subtitle} | Slide 2: Pitching Detail", pitching_sections))

    if len(hitter_sections) >= 2:
        hitter_slide_sections = [
            {"title": hitter_sections[0]["title"], "frame": _filter_section_columns(hitter_sections[0]["frame"], ["hitter_name", "matchup_score", "xwoba", "pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "avg_launch_angle"])},
            {"title": hitter_sections[1]["title"], "frame": _filter_section_columns(hitter_sections[1]["frame"], ["hitter_name", "matchup_score", "xwoba", "pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "avg_launch_angle"])},
        ]
        slides.append(build_branded_report_image(title, f"{subtitle} | Slide 3: Hitter Matchups", hitter_slide_sections))

    if count_sections:
        count_slide_sections = [
            {
                **section,
                "frame": _filter_section_columns(section["frame"], ["pitch_name", "All counts", "Early count", "Even count", "Pitcher ahead", "Pitcher behind", "Two-strike", "Pre two-strike", "Full count"]),
            }
            for section in count_sections
        ]
        slides.append(build_branded_report_image(title, f"{subtitle} | Slide 4: Count Usage", count_slide_sections))

    return slides


def _draw_section(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    lower_is_better: set[str] | None,
    higher_is_better: set[str] | None,
    font: ImageFont.ImageFont,
) -> int:
    padding_x = 12
    padding_y = 8
    row_height = 26
    small_font = _load_font(13)
    panel_height = 46 + _report_table_height(frame, row_height=row_height, header_height=26)
    _panel(draw, (left + 8, top, left + width - 8, top + panel_height), fill=REPORT_PANEL)
    _text(draw, (left + 22, top + 14), title, font, REPORT_TEXT)
    y = top + 46

    display_frame = _display_frame(frame).copy()
    headers = list(display_frame.columns)
    formatted_rows = []
    source_by_label = {DISPLAY_LABELS.get(col, col): col for col in frame.columns}
    for _, row in display_frame.iterrows():
        formatted_rows.append([_format_value(source_by_label.get(column, column), row[column], export_mode=True) for column in headers])

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    col_widths = []
    for col_idx, header in enumerate(headers):
        bbox = probe.textbbox((0, 0), header, font=small_font)
        max_width = bbox[2] - bbox[0]
        for row in formatted_rows:
            bbox = probe.textbbox((0, 0), row[col_idx], font=small_font)
            max_width = max(max_width, bbox[2] - bbox[0])
        col_widths.append(max_width + padding_x * 2)

    usable_width = width - 32
    total_width = sum(col_widths)
    if total_width < usable_width:
        extra = (usable_width - total_width) // max(len(col_widths), 1)
        col_widths = [size + extra for size in col_widths]

    x = left + 16
    for idx, header in enumerate(headers):
        draw.rounded_rectangle((x, y, x + col_widths[idx], y + row_height), radius=8, fill="#dce6f2")
        _text(draw, (x + padding_x, y + padding_y), header, small_font, REPORT_TEXT)
        x += col_widths[idx]

    for row_idx, row in enumerate(formatted_rows, start=1):
        current_y = y + row_height * row_idx
        x = left + 16
        for col_idx, value in enumerate(row):
            display_label = headers[col_idx]
            source_column = source_by_label.get(display_label, display_label)
            fill = "#ffffff" if row_idx % 2 else "#f7f9fc"
            if source_column in frame.columns and (source_column in PERCENT_COLUMNS or source_column in RATE_COLUMNS):
                fill = _background_hex(
                    source_column,
                    frame.iloc[row_idx - 1][source_column],
                    frame[source_column],
                    lower_is_better=lower_is_better or LOWER_IS_BETTER,
                    higher_is_better=higher_is_better or HIGHER_IS_BETTER,
                ) or fill
            draw.rounded_rectangle((x, current_y, x + col_widths[col_idx], current_y + row_height - 2), radius=6, fill=fill)
            _text(draw, (x + padding_x, current_y + padding_y), value, small_font, REPORT_TEXT)
            x += col_widths[col_idx]

    return top + panel_height + 12


def build_branded_report_image(title: str, subtitle: str, sections: list[dict]) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    font = _load_font(36, bold=True)
    body_font = _load_font(26, bold=True)
    width = 1500
    branding_height = 110
    total_height = branding_height + 24
    for section in sections:
        total_height += _export_section_height_estimate(section["title"], section["frame"])
    image = Image.new("RGB", (width, total_height), REPORT_BG)
    draw = ImageDraw.Draw(image)
    _panel(draw, (20, 20, width - 20, branding_height), fill="#f5f7fa", radius=24)
    _text(draw, (42, 34), "KASPER SCOUTING REPORT", body_font, REPORT_ACCENT)
    _text(draw, (42, 58), title, font, REPORT_TEXT)
    _text(draw, (42, 88), subtitle, body_font, REPORT_TEXT)

    y = branding_height + 18
    for section in sections:
        if section["frame"].empty:
            continue
        y = _draw_export_section(draw, y, 8, width - 16, section, font, body_font) + 12

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_export_bundle(title: str, subtitle: str, sections: list[dict], layout_mode: str) -> list[bytes]:
    if layout_mode == "Compact single poster":
        return [_build_compact_poster_image(title, subtitle, sections)]
    if layout_mode == "Carousel":
        return _build_carousel_images(title, subtitle, sections)
    return [build_branded_report_image(title, subtitle, sections)]


def png_to_jpg_bytes(png_bytes: bytes) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    source = Image.open(BytesIO(png_bytes)).convert("RGB")
    buffer = BytesIO()
    source.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _zip_export_slides(slides: list[bytes], base_filename: str, extension: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for idx, slide_bytes in enumerate(slides, start=1):
            archive.writestr(f"{base_filename}-{idx}.{extension}", slide_bytes)
    return buffer.getvalue()


def render_export_hub(key: str, title: str, export_options: dict[str, list[dict]]) -> None:
    if not export_options:
        return
    st.markdown("#### Game Exports")
    if not HAS_PILLOW:
        st.caption("Install `Pillow` to enable PNG/JPG export.")
        return
    option_labels = list(export_options.keys())
    selected_option = st.selectbox("Export scope", option_labels, key=f"{key}-scope", label_visibility="collapsed")
    layout_options = ["Compact single poster", "Carousel"] if selected_option == "Full game" else ["Standard report"]
    selected_layout = st.selectbox("Export layout", layout_options, key=f"{key}-layout", label_visibility="collapsed")
    sections = export_options[selected_option]
    prep_key = f"{key}-prepared"
    current_signature = (selected_option, selected_layout)
    if st.session_state.get(f"{prep_key}-signature") != current_signature:
        st.session_state[prep_key] = False
        st.session_state[f"{prep_key}-signature"] = current_signature

    if not st.session_state.get(prep_key, False):
        st.button("Prepare exports", key=f"{key}-prepare", use_container_width=True, on_click=lambda: st.session_state.__setitem__(prep_key, True))
        st.caption("Exports are generated on demand to keep the hosted app fast and stable.")
        return

    subtitle = f"{selected_option} | Generated from the Kasper matchup dashboard"
    png_slides = _build_export_bundle(title=title, subtitle=subtitle, sections=sections, layout_mode=selected_layout)
    jpg_slides = [png_to_jpg_bytes(slide) for slide in png_slides]
    safe_name = f"{key}-{selected_option.lower().replace(' ', '_').replace('/', '_')}-{selected_layout.lower().replace(' ', '_')}"
    col1, col2 = st.columns(2)
    with col1:
        if len(png_slides) == 1:
            st.download_button(
                label="Download PNG",
                data=png_slides[0],
                file_name=f"{safe_name}.png",
                mime="image/png",
                use_container_width=True,
                key=f"{key}-png",
            )
        else:
            st.download_button(
                label="Download PNG Carousel (.zip)",
                data=_zip_export_slides(png_slides, safe_name, "png"),
                file_name=f"{safe_name}.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"{key}-png-zip",
            )
    with col2:
        if len(jpg_slides) == 1:
            st.download_button(
                label="Download JPG",
                data=jpg_slides[0],
                file_name=f"{safe_name}.jpg",
                mime="image/jpeg",
                use_container_width=True,
                key=f"{key}-jpg",
            )
        else:
            st.download_button(
                label="Download JPG Carousel (.zip)",
                data=_zip_export_slides(jpg_slides, safe_name, "jpg"),
                file_name=f"{safe_name}-jpg.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"{key}-jpg-zip",
            )
