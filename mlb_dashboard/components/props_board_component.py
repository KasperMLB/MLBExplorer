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
    prepared = frame.astype(object).where(pd.notna(frame), None).copy()
    return _normalize_component_value(prepared.to_dict("records"))


def _normalize_component_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_component_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_component_value(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _validate_table_payload(columns: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_columns = _normalize_component_value(columns)
    normalized_rows = _normalize_component_value(rows)

    if not isinstance(normalized_columns, list):
        raise ValueError("React table payload error: columns must be a list.")
    if not isinstance(normalized_rows, list):
        raise ValueError("React table payload error: rows must be a list.")

    column_keys = [str(column.get("key")) for column in normalized_columns if isinstance(column, dict) and column.get("key")]
    if len(column_keys) != len(normalized_columns):
        raise ValueError("React table payload error: every column must define a key.")

    for idx, row in enumerate(normalized_rows):
        if not isinstance(row, dict):
            raise ValueError(f"React table payload error: row {idx} is not an object.")
        if "row_id" not in row:
            raise ValueError(f"React table payload error: row {idx} is missing row_id.")
        cells = row.get("cells")
        if not isinstance(cells, dict):
            raise ValueError(f"React table payload error: row {idx} cells must be an object.")
        missing = [key for key in column_keys if key not in cells]
        if missing:
            raise ValueError(f"React table payload error: row {idx} is missing cells for {missing[:3]}.")
        for key in column_keys:
            cell = cells.get(key)
            if cell is None:
                cells[key] = {"display": "", "sort": None, "background": None}
                continue
            if not isinstance(cell, dict):
                raise ValueError(f"React table payload error: row {idx} cell '{key}' is not an object.")
            cells[key] = {
                "display": _normalize_component_value(cell.get("display", "")),
                "sort": _normalize_component_value(cell.get("sort")),
                "background": _normalize_component_value(cell.get("background")),
            }

    return normalized_columns, normalized_rows


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
        initialSort=_normalize_component_value({"column": initial_sort_column, "ascending": initial_sort_ascending}),
        default=0,
        height=height,
    )


def render_react_data_table(
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    key: str,
    height: int = 320,
    title: str | None = None,
    subtitle: str | None = None,
    empty_message: str = "No data available for this selection.",
    initial_sort_column: str | None = None,
    initial_sort_ascending: bool = False,
) -> list[str]:
    safe_columns, safe_rows = _validate_table_payload(columns, rows)
    return _component_func(
        componentType="data_table",
        tableColumns=safe_columns,
        tableRows=safe_rows,
        initialSort=_normalize_component_value({"column": initial_sort_column, "ascending": initial_sort_ascending}) if initial_sort_column else None,
        title=title,
        subtitle=subtitle,
        emptyMessage=empty_message,
        default=[],
        key=key,
        height=height,
    )


def render_game_selector(
    cards: list[dict[str, Any]],
    selected_key: str,
    key: str,
    height: int = 250,
) -> str | None:
    return _component_func(
        componentType="game_selector",
        cards=_normalize_component_value(cards),
        selectedKey=str(selected_key),
        default=str(selected_key),
        key=key,
        height=height,
    )


def render_zone_tool(
    zone_rows: pd.DataFrame,
    key: str,
    title: str,
    subtitle: str,
    height: int = 560,
    value_mode: str = "percent",
    map_kind: str = "zone",
) -> None:
    _component_func(
        componentType="zone_tool",
        zoneRows=_records(zone_rows),
        title=title,
        subtitle=subtitle,
        valueMode=value_mode,
        mapKind=map_kind,
        default=0,
        key=key,
        height=height,
    )
