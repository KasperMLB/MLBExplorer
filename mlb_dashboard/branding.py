from __future__ import annotations

from pathlib import Path


def page_icon_path() -> str:
    return str(Path(__file__).resolve().parent / "assets" / "kasper.jpg")
