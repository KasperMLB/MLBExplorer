from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit.components.v1 as components


_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "props_board_component" / "dist"
_component_func = components.declare_component("props_board_component", path=str(_FRONTEND_DIR))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    prepared = frame.where(pd.notna(frame), None).copy()
    return prepared.to_dict("records")


def render_props_board(
    board_rows: pd.DataFrame,
    book_details: pd.DataFrame,
    height: int = 760,
    initial_sort_column: str = "Best Price",
    initial_sort_ascending: bool = False,
) -> None:
    _component_func(
        boardRows=_records(board_rows),
        bookDetails=_records(book_details),
        initialSort={"column": initial_sort_column, "ascending": initial_sort_ascending},
        default=0,
        height=height,
    )
