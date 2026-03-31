import React, { useEffect, useMemo, useRef, useState } from "react";
import { Streamlit } from "streamlit-component-lib";
import { HeaderCell, normalizeSortValue, sharedStyles, sortRows, useAutoHeight } from "./shared_ui.jsx";

function tableStyles() {
  return `
    ${sharedStyles()}
    .explorer-shell {
      position: relative;
    }
    .explorer-shell::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(180deg, rgba(255,255,255,0.28) 0%, rgba(255,255,255,0) 18%);
    }
    .explorer-table tbody td {
      padding: 7px 10px;
    }
    .explorer-table thead th {
      font-size: 10px;
      letter-spacing: 0.08em;
    }
    .explorer-table .numeric .metric-cell,
    .explorer-table td.numeric .metric-cell {
      text-align: right;
    }
    .explorer-table .long {
      font-weight: 700;
      color: #1b2431;
    }
    .explorer-table .short {
      color: #5f6f85;
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    .explorer-table td {
      background: transparent;
    }
    .explorer-table .metric-cell {
      min-height: 28px;
      display: inline-flex;
      align-items: center;
      justify-content: flex-end;
      min-width: 68px;
    }
    .explorer-table .cell-plain {
      padding: 2px 0;
    }
    .explorer-table td.numeric {
      font-variant-numeric: tabular-nums;
    }
    .explorer-table tbody tr:hover .metric-cell {
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.28), 0 2px 6px rgba(16, 37, 66, 0.08);
    }
    .diagnostic {
      padding: 18px;
      color: #8a1c1c;
      background: linear-gradient(180deg, rgba(255,241,241,0.96) 0%, rgba(255,248,248,0.98) 100%);
      border-top: 1px solid rgba(199, 71, 71, 0.16);
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
  `;
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function describePayload(args, columns, rows) {
  const firstRow = rows[0];
  const firstCellKeys = firstRow && typeof firstRow === "object" && firstRow.cells ? Object.keys(firstRow.cells).slice(0, 8) : [];
  return JSON.stringify(
    {
      componentType: args?.componentType,
      columnCount: columns.length,
      rowCount: rows.length,
      firstColumnKeys: columns.slice(0, 8).map((column) => column?.key),
      firstRowType: typeof firstRow,
      firstRowKeys: firstRow && typeof firstRow === "object" ? Object.keys(firstRow).slice(0, 8) : [],
      firstCellKeys,
    },
    null,
    2,
  );
}

class TableErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    console.error("React data_table render failed", error, this.props.debugInfo);
  }

  render() {
    if (this.state.error) {
      return (
        <>
          <style>{tableStyles()}</style>
          <div className="shell-card explorer-shell">
            <div className="shell-hero">
              <div className="shell-kicker">Slate Explorer</div>
              <h3 className="shell-title">React Table Error</h3>
              <div className="shell-subtitle">The shared data_table component failed before rendering.</div>
            </div>
            <div className="diagnostic">
              {String(this.state.error)}
              {"\n\n"}
              {this.props.debugInfo}
            </div>
          </div>
        </>
      );
    }
    return this.props.children;
  }
}

function DataTableInner({ args }) {
  const columns = safeArray(args?.tableColumns).filter((column) => column && typeof column === "object" && column.key);
  const rows = safeArray(args?.tableRows).filter((row) => row && typeof row === "object");
  const title = args?.title;
  const subtitle = args?.subtitle;
  const emptyMessage = args?.emptyMessage || "No data available for this selection.";
  const initialSort = args?.initialSort || null;
  const [sortState, setSortState] = useState(initialSort);
  const userSortedRef = useRef(false);

  useEffect(() => {
    console.debug("React data_table args", {
      componentType: args?.componentType,
      columns: columns.length,
      rows: rows.length,
      firstColumn: columns[0],
      firstRow: rows[0],
    });
  }, [args, columns, rows]);

  useEffect(() => {
    setSortState(initialSort);
  }, [initialSort?.column, initialSort?.ascending]);

  const sortedRows = useMemo(
    () => sortRows(rows, sortState, (row, columnKey) => normalizeSortValue(row.cells?.[columnKey]?.sort)),
    [rows, sortState],
  );

  useEffect(() => {
    if (!userSortedRef.current) return;
    Streamlit.setComponentValue(sortedRows.map((row) => row.row_id));
  }, [sortedRows]);

  useAutoHeight(rows.length, sortState?.column, sortState?.ascending, title, subtitle);

  const toggleSort = (columnKey) => {
    userSortedRef.current = true;
    setSortState((current) => {
      if (current?.column === columnKey) {
        return { column: columnKey, ascending: !current.ascending };
      }
      const column = columns.find((item) => item.key === columnKey);
      return { column: columnKey, ascending: !column?.numeric };
    });
  };

  return (
    <>
      <style>{tableStyles()}</style>
      <div className="shell-card explorer-shell">
        {title || subtitle ? (
          <div className="shell-hero">
            <div className="shell-kicker">Slate Explorer</div>
            {title ? <h3 className="shell-title">{title}</h3> : null}
            {subtitle ? <div className="shell-subtitle">{subtitle}</div> : null}
          </div>
        ) : null}
        {rows.length === 0 ? (
          <div className="empty">{emptyMessage}</div>
        ) : (
          <div className="table-wrap">
            <table className="explorer-table">
              <thead>
                <tr>
                  {columns.map((column) => (
                    <HeaderCell
                      key={column.key}
                      column={column}
                      sortState={sortState}
                      onSort={() => toggleSort(column.key)}
                    />
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((row) => (
                  <tr key={row.row_id}>
                    {columns.map((column) => {
                      const cell = row.cells?.[column.key] || {};
                      const hasHeat = column.heat && cell.background;
                      return (
                        <td
                          key={column.key}
                          className={`${column.numeric ? "numeric" : ""} ${column.kind || ""}`}
                          style={column.width ? { width: `${column.width}px` } : undefined}
                        >
                          <span
                            className={hasHeat ? "metric-cell" : "cell-plain"}
                            style={hasHeat ? { backgroundColor: cell.background } : undefined}
                          >
                            {cell.display ?? ""}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}

export function DataTable({ args }) {
  const columns = safeArray(args?.tableColumns).filter((column) => column && typeof column === "object" && column.key);
  const rows = safeArray(args?.tableRows).filter((row) => row && typeof row === "object");
  const debugInfo = describePayload(args, columns, rows);
  return (
    <TableErrorBoundary debugInfo={debugInfo}>
      <DataTableInner args={args} />
    </TableErrorBoundary>
  );
}
