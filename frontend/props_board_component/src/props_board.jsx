import React, { useEffect, useMemo, useState } from "react";
import { HeaderCell, normalizeSortValue, sharedStyles, sortRows, useAutoHeight } from "./shared_ui.jsx";

const columns = [
  { key: "Game", label: "Game", className: "col-game" },
  { key: "Team", label: "Team", className: "col-team" },
  { key: "Player", label: "Player", className: "col-player" },
  { key: "Prop Type", label: "Prop Type", className: "col-prop" },
  { key: "Side", label: "Side", className: "col-side" },
  { key: "Line", label: "Line", className: "col-line numeric", numeric: true },
  { key: "Best Books", label: "Best Books", className: "col-books" },
  { key: "Best Price", label: "Best Price", className: "col-price numeric strong", numeric: true },
  { key: "Market Width", label: "Market Width", className: "col-width numeric", numeric: true },
  { key: "Largest Discrepancy", label: "Largest Discrepancy", className: "col-disc numeric", numeric: true },
  { key: "Model Odds", label: "Model Odds", className: "col-model numeric muted", numeric: true },
  { key: "Edge%", label: "Edge%", className: "col-edge numeric muted", numeric: true },
  { key: "EV%", label: "EV%", className: "col-ev numeric muted", numeric: true },
];

function formatDetails(details) {
  return [...details].sort((a, b) => {
    const left = typeof a.sort_decimal === "number" ? a.sort_decimal : -Infinity;
    const right = typeof b.sort_decimal === "number" ? b.sort_decimal : -Infinity;
    if (left !== right) return right - left;
    return String(a.sportsbook || "").localeCompare(String(b.sportsbook || ""));
  });
}

function styleText() {
  return `
    ${sharedStyles()}
    .board-shell {
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(16, 37, 66, 0.05);
    }
    .strong {
      color: var(--price);
      font-weight: 700;
    }
    .muted {
      color: var(--muted);
    }
    .col-game { width: 110px; }
    .col-team { width: 72px; }
    .col-player { width: 150px; }
    .col-prop { width: 115px; }
    .col-side { width: 92px; }
    .col-line { width: 70px; }
    .col-books { width: 145px; }
    .col-price { width: 92px; }
    .col-width { width: 102px; }
    .col-disc { width: 140px; }
    .col-model, .col-edge, .col-ev { width: 84px; }
    .modal-layer {
      position: fixed;
      inset: 0;
      background: var(--modal-bg);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
      z-index: 9999;
    }
    .modal-card {
      width: min(820px, 100%);
      max-height: min(76vh, 820px);
      overflow: hidden;
      background: #fffdfa;
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: 0 24px 60px rgba(16, 23, 38, 0.28);
      display: flex;
      flex-direction: column;
    }
    .modal-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      padding: 18px 20px 14px;
      border-bottom: 1px solid rgba(230, 220, 200, 0.9);
      background: linear-gradient(180deg, #fffefb 0%, #f9f4e8 100%);
    }
    .modal-kicker {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }
    .modal-title {
      font-size: 22px;
      font-weight: 800;
      line-height: 1.15;
      margin-bottom: 6px;
      color: #12233f;
    }
    .modal-subtitle {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .close-btn {
      border: none;
      background: #f1ebde;
      color: #203452;
      border-radius: 999px;
      padding: 8px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    .modal-body {
      padding: 16px 20px 20px;
      overflow: auto;
    }
    .detail-table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }
    .detail-table th,
    .detail-table td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(230, 220, 200, 0.85);
    }
    .detail-table th {
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.06em;
      color: var(--muted);
      text-align: left;
      background: #faf6ee;
      position: sticky;
      top: 0;
    }
    .detail-table .numeric {
      text-align: right;
      color: var(--price);
      font-weight: 700;
    }
  `;
}

export function PropsBoard({ args }) {
  const boardRows = args?.boardRows || [];
  const bookDetails = args?.bookDetails || [];
  const initialSort = args?.initialSort || { column: "Best Price", ascending: false };
  const detailLookup = useMemo(() => {
    const lookup = new Map();
    for (const row of bookDetails) {
      const current = lookup.get(row.row_id) || [];
      current.push(row);
      lookup.set(row.row_id, current);
    }
    return lookup;
  }, [bookDetails]);
  const [selectedRowId, setSelectedRowId] = useState(null);
  const [sortState, setSortState] = useState(initialSort);

  useEffect(() => {
    setSortState(initialSort);
  }, [initialSort?.column, initialSort?.ascending]);

  const sortedRows = useMemo(
    () => sortRows(boardRows, sortState, (row, columnKey) => normalizeSortValue(row[columnKey])),
    [boardRows, sortState],
  );

  const selectedRow = useMemo(
    () => sortedRows.find((row) => row.row_id === selectedRowId) || null,
    [sortedRows, selectedRowId],
  );
  const selectedDetails = useMemo(() => {
    if (!selectedRowId) return [];
    return formatDetails(detailLookup.get(selectedRowId) || []);
  }, [detailLookup, selectedRowId]);

  useAutoHeight(boardRows.length, selectedRowId, sortState?.column, sortState?.ascending);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setSelectedRowId(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const toggleSort = (columnKey) => {
    setSortState((current) => {
      if (current?.column === columnKey) {
        return { column: columnKey, ascending: !current.ascending };
      }
      const ascendingDefault = ["Game", "Team", "Player", "Prop Type", "Side"].includes(columnKey);
      return { column: columnKey, ascending: ascendingDefault };
    });
  };

  return (
    <>
      <style>{styleText()}</style>
      <div className="board-shell">
        {boardRows.length === 0 ? (
          <div className="empty">No props match the current filters.</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {columns.map((column) => (
                    <HeaderCell key={column.key} column={column} sortState={sortState} onSort={() => toggleSort(column.key)} />
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((row) => (
                  <tr key={row.row_id} onClick={() => setSelectedRowId(row.row_id)}>
                    {columns.map((column) => (
                      <td key={column.key} className={column.className}>
                        {row[column.key] ?? ""}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      {selectedRow ? (
        <div className="modal-layer" onClick={() => setSelectedRowId(null)}>
          <div className="modal-card" onClick={(event) => event.stopPropagation()}>
            <div className="modal-head">
              <div>
                <div className="modal-kicker">All Books</div>
                <div className="modal-title">
                  {selectedRow.Player} | {selectedRow["Prop Type"]}
                </div>
                <div className="modal-subtitle">
                  {selectedRow.Side}
                  {selectedRow.Line ? ` ${selectedRow.Line}` : ""}
                  {" | "}
                  {selectedRow.Game}
                </div>
              </div>
              <button className="close-btn" type="button" onClick={() => setSelectedRowId(null)}>
                Close
              </button>
            </div>
            <div className="modal-body">
              <table className="detail-table">
                <thead>
                  <tr>
                    <th>Sportsbook</th>
                    <th className="numeric">Odds</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedDetails.map((detail) => (
                    <tr key={`${detail.row_id}-${detail.book_key}`}>
                      <td>{detail.sportsbook}</td>
                      <td className="numeric">{detail.price_display}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}

