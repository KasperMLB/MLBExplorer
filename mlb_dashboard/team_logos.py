from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg", ".svg")

TEAM_LOGO_SLUGS = {
    "ARI": "arizona-diamondbacks",
    "AZ": "arizona-diamondbacks",
    "ATH": "athletics",
    "OAK": "athletics",
    "ATL": "atlanta-braves",
    "BAL": "baltimore-orioles",
    "BOS": "boston-red-sox",
    "CHC": "chicago-cubs",
    "CWS": "chicago-white-sox",
    "CHW": "chicago-white-sox",
    "CIN": "cincinnati-reds",
    "CLE": "cleveland-guardians",
    "COL": "colorado-rockies",
    "DET": "detroit-tigers",
    "HOU": "houston-astros",
    "KC": "kansas-city-royals",
    "KCR": "kansas-city-royals",
    "LAA": "los-angeles-angeles",
    "ANA": "los-angeles-angeles",
    "LAD": "los-angeles-dodgers",
    "MIA": "miami-marlins",
    "MIL": "milwaukee-brewers",
    "MIN": "minnesota-twins",
    "NYM": "new-york-mets",
    "NYY": "new-york-yankees",
    "PHI": "philadelphia-phillies",
    "PIT": "pittsburgh-pirates",
    "SD": "san-diego-padres",
    "SDP": "san-diego-padres",
    "SF": "san-francisco-giants",
    "SFG": "san-francisco-giants",
    "SEA": "seattle-mariners",
    "STL": "st-louis-cardinals",
    "TB": "tampa-bay-rays",
    "TBR": "tampa-bay-rays",
    "TEX": "texas-rangers",
    "TOR": "toronto-blue-jays",
    "WSH": "washington-nationals",
    "WAS": "washington-nationals",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def logo_directories() -> list[Path]:
    root = _repo_root()
    return [
        root / "logos",
        root / "mlb_dashboard" / "assets" / "logos",
        root / "mlb_dashboard" / "assets",
    ]


def _filename_candidates(team: str) -> list[str]:
    raw = str(team or "").strip()
    if not raw:
        return []
    slug = TEAM_LOGO_SLUGS.get(raw.upper())
    stems = [
        raw,
        raw.lower(),
        raw.upper(),
        f"{raw}_logo",
        f"{raw.lower()}_logo",
        f"{raw.upper()}_logo",
        f"{raw}-logo",
        f"{raw.lower()}-logo",
        f"{raw.upper()}-logo",
    ]
    if slug:
        stems = [slug, f"{slug}-logo", *stems]
    return [f"{stem}{extension}" for stem in stems for extension in IMAGE_EXTENSIONS]


@lru_cache(maxsize=64)
def team_logo_path(team: str) -> Path | None:
    for directory in logo_directories():
        if not directory.exists():
            continue
        for filename in _filename_candidates(team):
            path = directory / filename
            if path.exists():
                return path
    return None


@lru_cache(maxsize=64)
def team_logo_data_uri(team: str) -> str | None:
    path = team_logo_path(team)
    if path is None:
        return None
    suffix = path.suffix.lower()
    mime = "image/svg+xml" if suffix == ".svg" else f"image/{'jpeg' if suffix in {'.jpg', '.jpeg'} else suffix.lstrip('.')}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def team_logo_img_html(team: str, *, size: int = 34) -> str:
    uri = team_logo_data_uri(team)
    label = str(team or "").strip()
    if not uri:
        return f"<span class='team-logo-fallback'>{label}</span>"
    return f"<img class='team-logo-img' src='{uri}' alt='{label}' style='width:{size}px;height:{size}px;object-fit:contain;'/>"


def matchup_logo_html(away_team: str, home_team: str, *, size: int = 34) -> str:
    return (
        "<div class='matchup-logo-row'>"
        f"{team_logo_img_html(away_team, size=size)}"
        "<span class='matchup-at'>@</span>"
        f"{team_logo_img_html(home_team, size=size)}"
        "</div>"
    )


def add_matchup_logo_columns(frame, game_column: str = "game"):
    if frame.empty or game_column not in frame.columns:
        return frame
    output = frame.copy()
    parts = output[game_column].fillna("").astype(str).str.split("@", n=1, expand=True)
    if parts.shape[1] < 2:
        return output
    away = parts[0].str.strip()
    home = parts[1].str.strip()
    output.insert(0, "away_logo", [team_logo_data_uri(team) for team in away])
    output.insert(1, "matchup_at", "@")
    output.insert(2, "home_logo", [team_logo_data_uri(team) for team in home])
    return output
