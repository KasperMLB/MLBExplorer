import React, { useEffect } from "react";
import { Streamlit } from "streamlit-component-lib";

export function normalizeSortValue(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const cleaned = value.replace(/[%+,]/g, "").trim();
    const numeric = Number(cleaned);
    if (!Number.isNaN(numeric) && cleaned !== "") return numeric;
    return value.toLowerCase();
  }
  return String(value).toLowerCase();
}

export function sortRows(rows, sortState, accessor) {
  const next = [...rows];
  if (!sortState?.column) return next;
  next.sort((left, right) => {
    const a = accessor(left, sortState.column);
    const b = accessor(right, sortState.column);
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    if (typeof a === "number" && typeof b === "number") {
      return sortState.ascending ? a - b : b - a;
    }
    if (a < b) return sortState.ascending ? -1 : 1;
    if (a > b) return sortState.ascending ? 1 : -1;
    return 0;
  });
  return next;
}

export function useAutoHeight(...deps) {
  useEffect(() => {
    const timeout = window.setTimeout(() => {
      Streamlit.setFrameHeight(document.documentElement.scrollHeight);
    }, 0);
    return () => window.clearTimeout(timeout);
  }, deps);
}

export function sharedStyles() {
  return `
    :root {
      color-scheme: light;
      --bg: #fffdf8;
      --panel: #fbf7ef;
      --panel-2: #fffdfa;
      --border: #e6dcc8;
      --header: #1c2b45;
      --text: #1f1f1f;
      --muted: #667085;
      --accent: #102542;
      --accent-soft: #e8eef7;
      --row-hover: #f2efe7;
      --row-alt: rgba(251, 247, 239, 0.7);
      --modal-bg: rgba(16, 23, 38, 0.42);
      --price: #183f76;
      --hero-top: #fffefb;
      --hero-bottom: #f8f2e4;
      --shadow: 0 16px 38px rgba(16, 37, 66, 0.08);
      --header-shadow: inset 0 -1px 0 #30456c, 0 10px 18px rgba(11, 25, 46, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: transparent;
      color: var(--text);
      font-family: "Segoe UI", system-ui, sans-serif;
    }
    .shell-card {
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 18px 42px rgba(16, 37, 66, 0.09);
    }
    .shell-hero {
      padding: 12px 14px 10px;
      border-bottom: 1px solid rgba(230, 220, 200, 0.85);
      background: linear-gradient(180deg, var(--hero-top) 0%, var(--hero-bottom) 100%);
    }
    .shell-kicker {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
      font-weight: 700;
    }
    .shell-title {
      font-size: 16px;
      font-weight: 800;
      line-height: 1.15;
      color: #12233f;
      margin: 0;
    }
    .shell-subtitle {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .table-wrap {
      max-height: 760px;
      overflow: auto;
      scrollbar-width: thin;
      scrollbar-color: #c8b99d #f8f2e6;
      background: linear-gradient(180deg, rgba(255, 253, 248, 0.78) 0%, rgba(251, 247, 239, 0.58) 100%);
      border-top: 1px solid rgba(255, 255, 255, 0.65);
    }
    .table-wrap::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    .table-wrap::-webkit-scrollbar-thumb {
      background: #cfbea2;
      border-radius: 999px;
      border: 2px solid #f7f1e6;
    }
    .table-wrap::-webkit-scrollbar-track {
      background: #f7f1e6;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 11.5px;
    }
    tbody td:first-child {
      position: sticky;
      left: 0;
      z-index: 2;
      background: #faf6ee;
    }
    thead th:first-child {
      position: sticky;
      left: 0;
      z-index: 3;
      background: linear-gradient(180deg, #21385d 0%, #182b48 100%);
    }
    thead th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: linear-gradient(180deg, #21385d 0%, #182b48 100%);
      color: #f8f7f3;
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      text-align: left;
      padding: 9px 11px;
      border-bottom: 1px solid #30456c;
      white-space: nowrap;
      user-select: none;
      box-shadow: var(--header-shadow);
    }
    thead th.sortable { cursor: pointer; }
    tbody td {
      padding: 8px 11px;
      border-bottom: 1px solid rgba(230, 220, 200, 0.85);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      vertical-align: middle;
    }
    tbody tr {
      background: transparent;
      transition: background 120ms ease;
    }
    tbody tr:hover {
      background: var(--row-hover);
    }
    tbody tr:nth-child(even) {
      background: var(--row-alt);
    }
    tbody tr:nth-child(even):hover {
      background: var(--row-hover);
    }
    .numeric {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .header-label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      width: 100%;
    }
    .header-sort {
      margin-left: auto;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 15px;
      height: 15px;
      border-radius: 999px;
      font-size: 9px;
      color: rgba(248, 247, 243, 0.88);
      background: rgba(255, 255, 255, 0.08);
    }
    .empty {
      padding: 26px 18px;
      color: var(--muted);
      font-size: 14px;
      background: linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(248,242,228,0.8) 100%);
    }
    .metric-cell {
      border-radius: 9px;
      padding: 5px 8px;
      font-weight: 700;
      color: #1f1f1f;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.22), 0 1px 0 rgba(16, 37, 66, 0.05);
    }
    .cell-plain {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric-cell.cell-plain {
      display: block;
    }
    .table-wrap table tr:last-child td {
      border-bottom-color: transparent;
    }
  `;
}

export function HeaderCell({ column, sortState, onSort }) {
  const sortGlyph = sortState?.column === column.key ? (sortState.ascending ? "↑" : "↓") : "·";
  return (
    <th
      className={`${column.className || ""} ${onSort ? "sortable" : ""} ${column.numeric ? "numeric" : ""}`}
      style={column.width ? { width: `${column.width}px` } : undefined}
      onClick={onSort}
    >
      <span className="header-label">
        {column.label}
        <span className="header-sort">{sortGlyph}</span>
      </span>
    </th>
  );
}
