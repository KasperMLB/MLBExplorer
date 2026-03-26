from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PILLOW = True
except ImportError:  # pragma: no cover
    HAS_PILLOW = False

PERCENT_COLUMNS = {"swstr_pct", "barrel_pct", "fb_pct", "hard_hit_pct", "usage_pct"}
RATE_COLUMNS = {"xwoba", "xwoba_con", "avg_launch_angle", "avg_release_speed", "avg_spin_rate", "gb_fb_ratio", "gb_pct"}
LOWER_IS_BETTER = {"swstr_pct"}
HIGHER_IS_BETTER = {"barrel_pct", "fb_pct", "hard_hit_pct", "usage_pct", "xwoba", "xwoba_con", "avg_release_speed", "avg_spin_rate", "matchup_score"}
TARGET_COLUMNS = {"avg_launch_angle": (8.0, 18.0, 28.0)}
DISPLAY_LABELS = {
    "hitter_name": "Hitter",
    "pitcher_name": "Pitcher",
    "team": "Team",
    "pitch_count": "Pitches",
    "bip": "BIP",
    "xwoba": "xwOBA",
    "xwoba_con": "xwOBAcon",
    "swstr_pct": "SwStr%",
    "barrel_pct": "Brl%",
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
}


def _format_value(column: str, value: object) -> str:
    if pd.isna(value):
        return "-"
    if column in PERCENT_COLUMNS:
        return f"{float(value):.1%}"
    if column == "avg_spin_rate":
        return f"{float(value):.0f}"
    if column in RATE_COLUMNS:
        return f"{float(value):.3f}" if "xwoba" in column else f"{float(value):.1f}"
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


def _two_sided_target_ratio(value: float, low: float, ideal: float, high: float) -> float:
    if value <= ideal:
        return 1.0 - min(1.0, max(0.0, (ideal - value) / max(ideal - low, 1e-9)))
    return 1.0 - min(1.0, max(0.0, (value - ideal) / max(high - ideal, 1e-9)))


def _background_style(
    column: str,
    value: object,
    series: pd.Series,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> str:
    if pd.isna(value):
        return ""
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return ""

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

    background = _blend_color("#f4b6b6", "#b8f2a6", ratio)
    return f"background-color: {background}; color: #1f1f1f;"


def style_metric_table(
    frame: pd.DataFrame,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> "pd.io.formats.style.Styler":
    display_frame = frame.rename(columns=DISPLAY_LABELS)
    reverse_labels = {label: source for source, label in DISPLAY_LABELS.items()}
    styler = display_frame.style.format(
        {
            column: (
                lambda value, c=reverse_labels.get(column, column): _format_value(c, value)
            )
            for column in display_frame.columns
        }
    )
    color_columns = [col for col in frame.columns if col in PERCENT_COLUMNS or col in RATE_COLUMNS]
    for column in color_columns:
        label = DISPLAY_LABELS.get(column, column)
        styler = styler.map(
            lambda value, c=column: _background_style(
                c,
                value,
                frame[c],
                lower_is_better=lower_is_better or LOWER_IS_BETTER,
                higher_is_better=higher_is_better or HIGHER_IS_BETTER,
            ),
            subset=[label],
        )
    return styler


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=DISPLAY_LABELS)


def _sorted_frame(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    sort_options = ["Current order"] + list(frame.columns)
    control_col1, control_col2, control_col3, control_col4 = st.columns([2.2, 1.2, 1.1, 1.1])
    with control_col1:
        sort_column = st.selectbox("Sort", sort_options, index=0, key=f"{key}-sort-col", label_visibility="collapsed")
    with control_col2:
        ascending = st.toggle("Ascending", value=False, key=f"{key}-sort-dir")
    sorted_frame = frame
    if sort_column != "Current order":
        sorted_frame = frame.sort_values(sort_column, ascending=ascending, na_position="last")
    return sorted_frame


def _render_branded_table_image(frame: pd.DataFrame, title: str, subtitle: str) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    display_frame = _display_frame(frame).copy()
    formatted_rows = []
    for _, row in display_frame.iterrows():
        formatted_rows.append([str(value) if pd.notna(value) else "-" for value in row.tolist()])
    headers = [str(column) for column in display_frame.columns.tolist()]

    font = ImageFont.load_default()
    padding_x = 12
    padding_y = 8
    row_height = 26
    branding_height = 74

    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    col_widths = []
    for col_idx, header in enumerate(headers):
        max_width = draw_probe.textbbox((0, 0), header, font=font)[2] - draw_probe.textbbox((0, 0), header, font=font)[0]
        for row in formatted_rows:
            text = row[col_idx]
            bbox = draw_probe.textbbox((0, 0), text, font=font)
            max_width = max(max_width, bbox[2] - bbox[0])
        col_widths.append(max_width + padding_x * 2)

    width = sum(col_widths)
    height = branding_height + row_height * (len(formatted_rows) + 1) + padding_y * 2
    image = Image.new("RGB", (width, height), "#fffdf8")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, branding_height), fill="#102542")
    draw.text((padding_x, 12), "KASPER", fill="#f7c04a", font=font)
    draw.text((padding_x, 34), title, fill="#ffffff", font=font)
    draw.text((padding_x, 52), subtitle, fill="#d7e4f2", font=font)

    y = branding_height
    x = 0
    for idx, header in enumerate(headers):
        draw.rectangle((x, y, x + col_widths[idx], y + row_height), fill="#d8e6f3")
        draw.text((x + padding_x, y + padding_y), header, fill="#102542", font=font)
        x += col_widths[idx]

    for row_idx, row in enumerate(formatted_rows, start=1):
        y = branding_height + row_height * row_idx
        x = 0
        fill = "#ffffff" if row_idx % 2 else "#f6f8fb"
        for col_idx, value in enumerate(row):
            draw.rectangle((x, y, x + col_widths[col_idx], y + row_height), fill=fill)
            draw.text((x + padding_x, y + padding_y), value, fill="#1f1f1f", font=font)
            x += col_widths[col_idx]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_to_jpg_bytes(png_bytes: bytes) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    source = Image.open(BytesIO(png_bytes)).convert("RGB")
    buffer = BytesIO()
    source.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _export_image_buttons(frame: pd.DataFrame, key: str, title: str) -> None:
    if not HAS_PILLOW:
        st.caption("Install `Pillow` to enable PNG/JPG export for this table.")
        return
    png_bytes = _render_branded_table_image(frame, title=title, subtitle="Generated from the Kasper matchup dashboard")
    jpg_bytes = _png_to_jpg_bytes(png_bytes)
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            label="Export PNG",
            data=png_bytes,
            file_name=f"{key}.png",
            mime="image/png",
            key=f"{key}-png",
            use_container_width=True,
        )
    with export_col2:
        st.download_button(
            label="Export JPG",
            data=jpg_bytes,
            file_name=f"{key}.jpg",
            mime="image/jpeg",
            key=f"{key}-jpg",
            use_container_width=True,
        )


def render_pitcher_card(pitcher_row: pd.Series) -> None:
    st.markdown(
        f"""
        ### {pitcher_row.get('pitcher_name', 'Unknown Pitcher')}
        **Throws:** {pitcher_row.get('p_throws', '-')}  
        **Pitches:** {int(pitcher_row.get('pitch_count', 0)):,}  
        **BIP:** {int(pitcher_row.get('bip', 0)):,}  
        **xwOBA:** {_format_value('xwoba', pitcher_row.get('xwoba'))}  
        **SwStr%:** {_format_value('swstr_pct', pitcher_row.get('swstr_pct'))}  
        **Barrel% Allowed:** {_format_value('barrel_pct', pitcher_row.get('barrel_pct'))}  
        **FB% Allowed:** {_format_value('fb_pct', pitcher_row.get('fb_pct'))}  
        **GB/FB:** {_format_value('gb_fb_ratio', pitcher_row.get('gb_fb_ratio'))}
        """
    )


def render_matchup_header(game: dict) -> None:
    st.subheader(f"{game['away_team']} @ {game['home_team']}")
    st.caption(f"{game.get('status', 'Scheduled')} | Game PK: {game['game_pk']}")


def render_dataframe(
    frame: pd.DataFrame,
    key: str,
    height: int = 320,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
    title: str | None = None,
) -> None:
    if frame.empty:
        st.info("No data available for this selection.")
        return
    sorted_frame = _sorted_frame(frame, key)
    _export_image_buttons(sorted_frame, key=key, title=title or key.replace("-", " ").title())
    st.dataframe(
        style_metric_table(sorted_frame, lower_is_better=lower_is_better, higher_is_better=higher_is_better),
        use_container_width=True,
        key=key,
        hide_index=True,
        height=height,
    )
