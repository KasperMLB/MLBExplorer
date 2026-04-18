from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .config import AppConfig
from .team_logos import team_logo_path

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PILLOW = True
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None
    HAS_PILLOW = False


TWITTER_EXPORT_DIRNAME = "exports"
TWITTER_EXPORT_GAMES_DIRNAME = "games"
TWITTER_EXPORT_ZIP = "full_slate_game_cards.zip"
TWITTER_EXPORT_SPLIT = "overall"
TWITTER_EXPORT_RECENT_WINDOW = "season"
TWITTER_EXPORT_WEIGHTED_MODE = "weighted"

_BG_TOP = "#0d1021"
_BG_BOTTOM = "#182441"
_PANEL = "#f8fafc"
_PANEL_DARK = "#141b31"
_PANEL_MID = "#1d2947"
_TEXT = "#f8fafc"
_MUTED = "#cbd5e1"
_DARK_TEXT = "#111827"
_ACCENT = "#a78bfa"
_BORDER = "#c86bf2"
_GOOD = "#2f8f46"
_MID = "#e7cf63"
_BAD = "#c94b4b"
_NEUTRAL = "#aeb7b4"


@dataclass(frozen=True)
class TwitterExportResult:
    zip_path: Path
    image_paths: list[Path]


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if bold else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        try:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[idx:idx + 2], 16) for idx in (0, 2, 4))


def _blend(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    return f"#{round(ar + (br - ar) * t):02x}{round(ag + (bg - ag) * t):02x}{round(ab + (bb - ab) * t):02x}"


def _paint_gradient(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    height = max(image.height - 1, 1)
    for y in range(image.height):
        draw.line([(0, y), (image.width, y)], fill=_blend(_BG_TOP, _BG_BOTTOM, y / height))


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: object, font: ImageFont.ImageFont, fill: str, *, max_width: int | None = None) -> None:
    text = "-" if value is None or pd.isna(value) else str(value)
    if max_width is not None:
        ellipsis = "..."
        while draw.textbbox((0, 0), text, font=font)[2] > max_width and len(text) > 3:
            text = text[:-1]
        if str(value) != text and len(text) > 3:
            text = text[:-3] + ellipsis
    draw.text(xy, text, font=font, fill=fill)


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, fill: str = _PANEL_DARK, outline: str = _BORDER, radius: int = 24, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _fmt_num(value: object, decimals: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    return f"{float(numeric):.{decimals}f}"


def _fmt_pct(value: object, decimals: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    if abs(float(numeric)) <= 1.5:
        numeric = float(numeric) * 100
    return f"{float(numeric):.{decimals}f}%"


def _safe_display(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _higher_heat(value: object, low: float, high: float) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return _NEUTRAL
    t = (float(numeric) - low) / max(high - low, 1e-9)
    if t < 0.5:
        return _blend(_BAD, _MID, t * 2)
    return _blend(_MID, _GOOD, (t - 0.5) * 2)


def _la_heat(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return _NEUTRAL
    distance = abs(float(numeric) - 22.0)
    score = max(0.0, 1.0 - distance / 24.0)
    if score < 0.5:
        return _blend(_BAD, _MID, score * 2)
    return _blend(_MID, _GOOD, (score - 0.5) * 2)


def _hr_form_heat(value: object) -> str:
    if value is None:
        return _NEUTRAL
    text = str(value).strip()
    if not text or text == "-":
        return _NEUTRAL
    pct = pd.to_numeric(pd.Series([text.split("%")[0]]), errors="coerce").iloc[0]
    if pd.isna(pct):
        return _NEUTRAL
    pct = float(pct)
    if pct < 40:
        return _blend(_BAD, _NEUTRAL, pct / 40)
    if pct <= 60:
        return _NEUTRAL
    return _blend(_NEUTRAL, _GOOD, min((pct - 60) / 35, 1.0))


def _metric_spec(row: pd.Series) -> list[tuple[str, str, str]]:
    barrel_value = row.get("barrel_bip_pct", row.get("barrel_bbe_pct", row.get("barrel_pct")))
    return [
        ("Matchup", _fmt_num(row.get("matchup_score"), 1), _higher_heat(row.get("matchup_score"), 35, 90)),
        ("Test", _fmt_num(row.get("test_score"), 1), _higher_heat(row.get("test_score"), 35, 90)),
        ("Ceiling", _fmt_num(row.get("ceiling_score"), 1), _higher_heat(row.get("ceiling_score"), 35, 100)),
        ("Zone", _fmt_num(row.get("zone_fit_score"), 3), _higher_heat(row.get("zone_fit_score"), 0.02, 0.14)),
        ("HR Form", _safe_display(row.get("hr_form")), _hr_form_heat(row.get("hr_form"))),
        ("PulledBarrel", _fmt_pct(row.get("pulled_barrel_pct"), 1), _higher_heat(row.get("pulled_barrel_pct"), 0.02, 0.14)),
        ("Barrel", _fmt_pct(barrel_value, 1), _higher_heat(barrel_value, 0.04, 0.18)),
        ("HH", _fmt_pct(row.get("hard_hit_pct"), 1), _higher_heat(row.get("hard_hit_pct"), 0.30, 0.58)),
        ("FB%", _fmt_pct(row.get("fb_pct"), 1), _higher_heat(row.get("fb_pct"), 0.20, 0.55)),
        ("LA", _fmt_num(row.get("avg_launch_angle"), 1), _la_heat(row.get("avg_launch_angle"))),
    ]


def _draw_logo(draw: ImageDraw.ImageDraw, image: Image.Image, team: str, xy: tuple[int, int], size: int) -> None:
    path = team_logo_path(team)
    if path is not None:
        try:
            logo = Image.open(path).convert("RGBA")
            logo.thumbnail((size, size), Image.Resampling.LANCZOS)
            x = xy[0] + (size - logo.width) // 2
            y = xy[1] + (size - logo.height) // 2
            image.paste(logo, (x, y), logo)
            return
        except Exception:
            pass
    font = _load_font(40, bold=True)
    _text(draw, (xy[0] + 8, xy[1] + size // 3), team, font, _TEXT, max_width=size - 16)


def _draw_header(draw: ImageDraw.ImageDraw, image: Image.Image, game: dict) -> None:
    away = str(game.get("away_team", "") or "")
    home = str(game.get("home_team", "") or "")
    _panel(draw, (28, 24, image.width - 28, 150), fill=_PANEL_DARK, radius=26)
    draw.rounded_rectangle((54, 46, 220, 82), radius=14, fill=_ACCENT)
    _text(draw, (72, 54), "KASPER", _load_font(19, bold=True), _TEXT)
    _text(draw, (54, 92), "HR MATCHUP CARD", _load_font(34, bold=True), _TEXT)
    _text(draw, (54, 126), f"Game PK {game.get('game_pk', '-')}", _load_font(18, bold=True), _MUTED)

    center_x = image.width // 2 + 185
    _draw_logo(draw, image, away, (center_x - 180, 38), 82)
    _text(draw, (center_x - 50, 64), "@", _load_font(48, bold=True), _TEXT)
    _draw_logo(draw, image, home, (center_x + 44, 38), 82)
    _text(draw, (center_x - 210, 124), f"{away} @ {home}", _load_font(21, bold=True), _MUTED, max_width=430)


def _draw_pitcher_strip(draw: ImageDraw.ImageDraw, game: dict, top: int, width: int) -> int:
    font = _load_font(22, bold=True)
    small = _load_font(20, bold=True)
    _panel(draw, (28, top, width - 28, top + 72), fill=_PANEL_MID, radius=18)
    away = str(game.get("away_team", "") or "")
    home = str(game.get("home_team", "") or "")
    away_pitcher = str(game.get("away_probable_pitcher_name") or "Away starter TBD")
    home_pitcher = str(game.get("home_probable_pitcher_name") or "Home starter TBD")
    _text(draw, (52, top + 15), "Probable Starters", font, _TEXT)
    _text(draw, (300, top + 17), f"{away}: {away_pitcher}", small, _MUTED, max_width=520)
    _text(draw, (880, top + 17), f"{home}: {home_pitcher}", small, _MUTED, max_width=520)
    return top + 88


def _draw_targets(draw: ImageDraw.ImageDraw, frame: pd.DataFrame, top: int, width: int) -> int:
    title_font = _load_font(28, bold=True)
    row_font = _load_font(22, bold=True)
    score_font = _load_font(24, bold=True)
    _panel(draw, (28, top, width - 28, top + 136), fill=_PANEL_DARK, radius=20)
    _text(draw, (52, top + 16), "Top HR Targets", title_font, _TEXT)
    card_w = (width - 104) // 3
    y = top + 60
    for idx, (_, row) in enumerate(frame.head(3).iterrows(), start=1):
        x = 52 + (idx - 1) * (card_w + 12)
        draw.rounded_rectangle((x, y, x + card_w, y + 50), radius=13, fill=_PANEL)
        _text(draw, (x + 14, y + 12), f"{idx}. {row.get('hitter_name', '-')}", row_font, _DARK_TEXT, max_width=card_w - 120)
        _text(draw, (x + card_w - 104, y + 14), str(row.get("team", "-")), _load_font(18, bold=True), _DARK_TEXT, max_width=44)
        score = _fmt_num(row.get("matchup_score"), 1)
        draw.rounded_rectangle((x + card_w - 58, y + 8, x + card_w - 12, y + 42), radius=10, fill=_higher_heat(row.get("matchup_score"), 35, 90))
        _text(draw, (x + card_w - 52, y + 11), score, score_font, _TEXT, max_width=38)
    return top + 154


def _metric_map(row: pd.Series) -> dict[str, tuple[str, str]]:
    return {label: (value, fill) for label, value, fill in _metric_spec(row)}


def _draw_hitter_panel(draw: ImageDraw.ImageDraw, frame: pd.DataFrame, title: str, top: int, left: int, width: int, height: int) -> int:
    title_font = _load_font(29, bold=True)
    header_font = _load_font(15, bold=True)
    name_font = _load_font(24, bold=True)
    value_font = _load_font(20, bold=True)
    _panel(draw, (left, top, left + width, top + height), fill=_PANEL_DARK, radius=26)
    _text(draw, (left + 24, top + 14), title, title_font, _TEXT, max_width=width - 48)
    if frame.empty:
        _text(draw, (left + 24, top + 76), "No hitter rows available", name_font, _MUTED)
        return top + height
    columns = [
        ("Player", 258),
        ("Matchup", 86),
        ("Test", 76),
        ("Ceiling", 84),
        ("Zone", 76),
        ("HR Form", 88),
        ("PulledBarrel", 116),
        ("Barrel", 82),
        ("HH", 70),
        ("FB%", 70),
        ("LA", 58),
    ]
    scale = min(1.0, (width - 48) / sum(col_width for _, col_width in columns))
    columns = [(label, max(48, int(col_width * scale))) for label, col_width in columns]
    x = left + 24
    header_y = top + 58
    for label, col_width in columns:
        _text(draw, (x + 6, header_y), label, header_font, _MUTED, max_width=col_width - 10)
        x += col_width
    row_h = 58
    y = top + 84
    for _, row in frame.head(5).iterrows():
        draw.rounded_rectangle((left + 18, y, left + width - 18, y + row_h - 5), radius=14, fill=_PANEL)
        metric_values = _metric_map(row)
        x = left + 24
        for label, col_width in columns:
            if label == "Player":
                _text(draw, (x + 8, y + 14), row.get("hitter_name", "-"), name_font, _DARK_TEXT, max_width=col_width - 14)
            else:
                value, fill = metric_values.get(label, ("-", _NEUTRAL))
                draw.rounded_rectangle((x + 3, y + 8, x + col_width - 4, y + row_h - 13), radius=10, fill=fill)
                _text(draw, (x + 8, y + 17), value, value_font, _TEXT, max_width=col_width - 12)
            x += col_width
        y += row_h
    return top + height


def build_twitter_game_card(game: dict, hitters: pd.DataFrame) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required to build Twitter export cards.")
    width, height = 1600, 1200
    image = Image.new("RGB", (width, height), _BG_TOP)
    _paint_gradient(image)
    draw = ImageDraw.Draw(image)
    _draw_header(draw, image, game)
    y = _draw_pitcher_strip(draw, game, 168, width)
    top_targets = hitters.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last") if not hitters.empty else hitters
    y = _draw_targets(draw, top_targets, y, width)

    away = str(game.get("away_team", "") or "")
    home = str(game.get("home_team", "") or "")
    away_hitters = top_targets.loc[top_targets.get("team", pd.Series(dtype="object")).astype(str).eq(away)].copy() if not top_targets.empty and "team" in top_targets.columns else pd.DataFrame()
    home_hitters = top_targets.loc[top_targets.get("team", pd.Series(dtype="object")).astype(str).eq(home)].copy() if not top_targets.empty and "team" in top_targets.columns else pd.DataFrame()
    panel_height = 390
    y = _draw_hitter_panel(draw, away_hitters, f"{away} Hitters", y + 4, 28, width - 56, panel_height) + 14
    _draw_hitter_panel(draw, home_hitters, f"{home} Hitters", y, 28, width - 56, panel_height)
    _text(draw, (42, height - 24), "Generated from Kasper matchup artifacts", _load_font(14, bold=True), _MUTED)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _safe_game_name(game: dict) -> str:
    game_pk = str(game.get("game_pk", "game"))
    away = str(game.get("away_team", "away")).replace("/", "_").replace(" ", "_")
    home = str(game.get("home_team", "home")).replace("/", "_").replace(" ", "_")
    return f"{game_pk}_{away}_at_{home}.png"


def write_full_slate_twitter_exports(
    config: AppConfig,
    target_date,
    slate: pd.DataFrame | list[dict],
    hitter_board: pd.DataFrame,
) -> TwitterExportResult | None:
    if not HAS_PILLOW:
        return None
    games = slate.to_dict(orient="records") if isinstance(slate, pd.DataFrame) else list(slate)
    export_dir = config.daily_dir / target_date.isoformat() / TWITTER_EXPORT_DIRNAME
    games_dir = export_dir / TWITTER_EXPORT_GAMES_DIRNAME
    games_dir.mkdir(parents=True, exist_ok=True)
    zip_path = export_dir / TWITTER_EXPORT_ZIP
    board = hitter_board.copy()
    for column, value in {
        "split_key": TWITTER_EXPORT_SPLIT,
        "recent_window": TWITTER_EXPORT_RECENT_WINDOW,
        "weighted_mode": TWITTER_EXPORT_WEIGHTED_MODE,
    }.items():
        if column in board.columns:
            board = board.loc[board[column].eq(value)].copy()
    image_paths: list[Path] = []
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for game in games:
            game_pk = game.get("game_pk")
            if game_pk is None:
                continue
            game_hitters = board.loc[board.get("game_pk", pd.Series(dtype="object")).astype(str).eq(str(game_pk))].copy() if not board.empty and "game_pk" in board.columns else pd.DataFrame()
            image_bytes = build_twitter_game_card(game, game_hitters)
            filename = _safe_game_name(game)
            image_path = games_dir / filename
            image_path.write_bytes(image_bytes)
            archive.writestr(filename, image_bytes)
            image_paths.append(image_path)
    return TwitterExportResult(zip_path=zip_path, image_paths=image_paths)
