import React, { useEffect, useMemo, useRef, useState } from "react";
import { sharedStyles, useAutoHeight } from "./shared_ui.jsx";

const zoneRects = {
  11: { x: 1, y: 0, w: 3, h: 1 },
  12: { x: 1, y: 4, w: 3, h: 1 },
  13: { x: 0, y: 1, w: 1, h: 3 },
  14: { x: 4, y: 1, w: 1, h: 3 },
  1: { x: 1, y: 1, w: 1, h: 1 },
  2: { x: 2, y: 1, w: 1, h: 1 },
  3: { x: 3, y: 1, w: 1, h: 1 },
  4: { x: 1, y: 2, w: 1, h: 1 },
  5: { x: 2, y: 2, w: 1, h: 1 },
  6: { x: 3, y: 2, w: 1, h: 1 },
  7: { x: 1, y: 3, w: 1, h: 1 },
  8: { x: 2, y: 3, w: 1, h: 1 },
  9: { x: 3, y: 3, w: 1, h: 1 },
};

function zoneStyles() {
  return `
    ${sharedStyles()}
    .zone-shell {
      position: relative;
      background: linear-gradient(180deg, #10212e 0%, #1d3442 100%);
      border: 1px solid #324b5a;
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 18px 42px rgba(8, 18, 28, 0.22);
      color: #edf4f7;
      max-width: 100%;
    }
    .zone-frame {
      width: 100%;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      overflow: hidden;
    }
    .zone-scale {
      will-change: transform;
    }
    .zone-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px 7px;
      border-bottom: 1px solid rgba(132, 162, 179, 0.18);
      background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    }
    .zone-kicker {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #9db0bd;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .zone-title {
      font-size: 14px;
      font-weight: 800;
      line-height: 1.1;
      margin: 0;
    }
    .zone-subtitle {
      margin-top: 4px;
      font-size: 9px;
      color: #b9c8d2;
      line-height: 1.35;
    }
    .zone-mapkind {
      align-self: flex-start;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(186, 205, 217, 0.22);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #edf4f7;
      font-weight: 700;
    }
    .zone-body {
      position: relative;
      padding: 9px 9px 8px;
    }
    .zone-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 102px;
      gap: 10px;
      align-items: start;
    }
    .zone-svg {
      width: 100%;
      height: auto;
      display: block;
      max-width: 100%;
    }
    .legend-card {
      border-radius: 14px;
      border: 1px solid rgba(186, 205, 217, 0.18);
      background: rgba(10, 20, 28, 0.3);
      padding: 8px 8px 6px;
    }
    .legend-title {
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #9db0bd;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .legend-subtitle {
      font-size: 9px;
      color: #c7d5dd;
      line-height: 1.35;
      margin-bottom: 8px;
    }
    .legend-row {
      display: grid;
      grid-template-columns: 16px minmax(0, 1fr);
      gap: 6px;
      align-items: center;
      margin-bottom: 5px;
      font-size: 9px;
      color: #eef4f7;
    }
    .legend-chip {
      height: 12px;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.2);
    }
    .tooltip {
      position: absolute;
      min-width: 138px;
      max-width: 220px;
      pointer-events: none;
      transform: translate(-50%, calc(-100% - 12px));
      background: rgba(7, 14, 20, 0.94);
      border: 1px solid rgba(173, 197, 210, 0.24);
      border-radius: 14px;
      padding: 8px 10px;
      box-shadow: 0 18px 28px rgba(0, 0, 0, 0.28);
      z-index: 5;
    }
    .tooltip.tooltip-below {
      transform: translate(-50%, 12px);
    }
    .tooltip-zone {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #9db0bd;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .tooltip-value {
      font-size: 20px;
      font-weight: 800;
      line-height: 1;
      margin-bottom: 4px;
    }
    .tooltip-meta {
      font-size: 12px;
      color: #d7e2e8;
      line-height: 1.35;
    }
    .tooltip-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #9db0bd;
      font-weight: 700;
      margin-bottom: 4px;
    }
    @media (max-width: 620px) {
      .zone-layout {
        grid-template-columns: 1fr;
      }
      .legend-card {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
      }
      .legend-title {
        grid-column: 1 / -1;
        margin-bottom: 0;
      }
    }
  `;
}

function displayValue(cell, valueMode) {
  const raw = cell?.display_value;
  if (raw === null || raw === undefined || Number.isNaN(Number(raw))) return "-";
  if (valueMode === "percent") return `${Math.round(Number(raw) * 100)}`;
  return `${Number(raw).toFixed(3)}`;
}

function pitchesLabel(cell) {
  const sample = cell?.sample_size;
  if (sample === null || sample === undefined || Number.isNaN(Number(sample))) return "No data";
  return `${Number(sample).toLocaleString()} pitches`;
}

function valueLabel(mapKind) {
  if (mapKind === "pitcher") return "Usage%";
  if (mapKind === "hitter") return "Damage";
  if (mapKind === "overlay") return "Overlay Score";
  return "Value";
}

function kindLabel(mapKind) {
  if (mapKind === "pitcher") return "Pitcher Map";
  if (mapKind === "hitter") return "Hitter Map";
  if (mapKind === "overlay") return "Overlay Map";
  return "Zone Map";
}

export function ZoneTool({ args }) {
  const rows = args?.zoneRows || [];
  const title = args?.title || "Zone Tool";
  const subtitle = args?.subtitle || "";
  const valueMode = args?.valueMode || "percent";
  const mapKind = args?.mapKind || "zone";
  const [hoveredZone, setHoveredZone] = useState(null);
  const [lockedZone, setLockedZone] = useState(null);
  const frameRef = useRef(null);
  const scaleRef = useRef(null);
  const [frameWidth, setFrameWidth] = useState(0);
  const [contentHeight, setContentHeight] = useState(0);

  const designWidth = 520;
  const baseScale = 0.9;

  useEffect(() => {
    if (!frameRef.current) return undefined;
    const node = frameRef.current;
    const update = () => setFrameWidth(node.clientWidth || 0);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!scaleRef.current) return undefined;
    const node = scaleRef.current;
    const update = () => setContentHeight(node.offsetHeight || 0);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(node);
    return () => observer.disconnect();
  }, [title, subtitle, rows.length, mapKind, valueMode]);

  const scale = frameWidth ? Math.min(baseScale, Math.max(0.58, ((frameWidth - 4) / designWidth) * baseScale)) : baseScale;
  const scaledHeight = Math.ceil((contentHeight || 470) * scale);

  useAutoHeight(rows.length, hoveredZone?.zone, lockedZone?.zone, title, subtitle, frameWidth, scale);

  const zoneLookup = useMemo(() => {
    const next = new Map();
    rows.forEach((row) => {
      next.set(Number(row.zone), row);
    });
    return next;
  }, [rows]);

  const zoneValues = rows
    .map((row) => Number(row.zone_value))
    .filter((value) => !Number.isNaN(value));
  const minValue = zoneValues.length ? Math.min(...zoneValues) : 0;
  const maxValue = zoneValues.length ? Math.max(...zoneValues) : 1;

  const activeZone = lockedZone || hoveredZone;
  const cellSize = 60;
  const gap = 6;
  const zoneOriginX = 18;
  const zoneOriginY = 18;
  const boardWidth = cellSize * 5 + gap * 4 + 44;
  const boardHeight = cellSize * 5 + gap * 4 + 56;
  const tooltipLeft = activeZone ? activeZone.tooltipLeft : 0;
  const tooltipTop = activeZone ? activeZone.tooltipTop : 0;

  const ratioFor = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 0.5;
    if (Math.abs(maxValue - minValue) < 1e-9) return 0.5;
    return (Number(value) - minValue) / (maxValue - minValue);
  };

  const fillFor = (value) => {
    const ratio = ratioFor(value);
    if (ratio <= 0.25) return "#58a7e4";
    if (ratio <= 0.45) return "#88b1c8";
    if (ratio <= 0.55) return "#adb5b9";
    if (ratio <= 0.75) return "#c99b59";
    return "#ef8d32";
  };

  return (
    <>
      <style>{zoneStyles()}</style>
      <div ref={frameRef} className="zone-frame" style={{ height: `${scaledHeight}px` }}>
        <div
          ref={scaleRef}
          className="zone-scale"
          style={{
            width: `${designWidth}px`,
            transform: `scale(${scale})`,
            transformOrigin: "top center",
          }}
        >
          <div className="zone-shell">
            <div className="zone-head">
              <div>
                <div className="zone-kicker">Zone Tool</div>
                <h3 className="zone-title">{title}</h3>
                <div className="zone-subtitle">{subtitle}</div>
              </div>
              <div className="zone-mapkind">{kindLabel(mapKind)}</div>
            </div>
            <div className="zone-body">
              <div className="zone-layout">
                <div style={{ position: "relative" }}>
                  {activeZone ? (
                    <div className={`tooltip${activeZone.below ? " tooltip-below" : ""}`} style={{ left: `${tooltipLeft}px`, top: `${tooltipTop}px` }}>
                      <div className="tooltip-zone">Zone {activeZone.zone}</div>
                      <div className="tooltip-label">{valueLabel(mapKind)}</div>
                      <div className="tooltip-value">{displayValue(activeZone.row, valueMode)}</div>
                      <div className="tooltip-meta">{pitchesLabel(activeZone.row)}</div>
                    </div>
                  ) : null}
                  <svg className="zone-svg" viewBox={`0 0 ${boardWidth} ${boardHeight}`} role="img" aria-label={title}>
                    <defs>
                      <linearGradient id="zoneBoardBg" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#112431" />
                        <stop offset="100%" stopColor="#1d3542" />
                      </linearGradient>
                    </defs>
                    <rect x="6" y="6" width={boardWidth - 12} height={boardHeight - 12} rx="20" fill="url(#zoneBoardBg)" stroke="#4b6672" strokeWidth="2" />
                    {Object.entries(zoneRects).map(([zoneKey, rect]) => {
                      const zone = Number(zoneKey);
                      const row = zoneLookup.get(zone) || {};
                      const x = zoneOriginX + rect.x * (cellSize + gap);
                      const y = zoneOriginY + rect.y * (cellSize + gap);
                      const width = rect.w * cellSize + (rect.w - 1) * gap;
                      const height = rect.h * cellSize + (rect.h - 1) * gap;
                      const active = activeZone?.zone === zone;
                      const below = y < 150;
                      return (
                        <g
                          key={zone}
                          onMouseEnter={() => setHoveredZone({ zone, row, tooltipLeft: x + width / 2, tooltipTop: below ? y + height : y, below })}
                          onMouseLeave={() => setHoveredZone((current) => (current?.zone === zone ? null : current))}
                          onClick={() => setLockedZone((current) => (current?.zone === zone ? null : { zone, row, tooltipLeft: x + width / 2, tooltipTop: below ? y + height : y, below }))}
                          style={{ cursor: "pointer" }}
                        >
                          <rect x={x + 2} y={y + 3} width={width} height={height} rx="10" fill="#081218" opacity="0.72" />
                          <rect x={x} y={y} width={width} height={height} rx="10" fill={fillFor(row.zone_value)} stroke={active ? "#ffffff" : "#edf4f7"} strokeWidth={active ? 3 : 2} style={{ transition: "fill 0.3s ease, stroke 0.15s ease" }} />
                          {row && row.sample_size !== null && row.sample_size !== undefined ? (
                            <>
                              <text x={x + width / 2} y={y + 13} textAnchor="middle" fontSize="8" fontWeight="800" fill="#24363d">
                                {valueLabel(mapKind)}
                              </text>
                              <text x={x + width / 2} y={y + height / 2} textAnchor="middle" fontSize="17" fontWeight="800" fill="#10171b">
                                {displayValue(row, valueMode)}
                              </text>
                              <text x={x + width / 2} y={y + height - 10} textAnchor="middle" fontSize="8.5" fontWeight="700" fill="#22343b">
                                {pitchesLabel(row)}
                              </text>
                            </>
                          ) : (
                            <>
                              <text x={x + width / 2} y={y + height / 2 - 2} textAnchor="middle" fontSize="16" fontWeight="800" fill="#22343b">
                                -
                              </text>
                              <text x={x + width / 2} y={y + height / 2 + 14} textAnchor="middle" fontSize="8.5" fontWeight="700" fill="#22343b">
                                No data
                              </text>
                            </>
                          )}
                        </g>
                      );
                    })}
                    <rect
                      x={zoneOriginX + cellSize + gap - 6}
                      y={zoneOriginY + cellSize + gap - 6}
                      width={cellSize * 3 + gap * 2 + 12}
                      height={cellSize * 3 + gap * 2 + 12}
                      rx="14"
                      fill="none"
                      stroke="#f5f8fa"
                      strokeWidth="3"
                    />
                    <path
                      d={`M ${zoneOriginX + cellSize * 2.5 + gap * 2} ${zoneOriginY + cellSize * 4 + gap * 4 + 30}
                      l 22 0 l 8 14 l -30 12 l -30 -12 l 8 -14 z`}
                      fill="#eef4f7"
                      stroke="#dce7eb"
                      strokeWidth="2"
                    />
                  </svg>
                </div>
                <div className="legend-card">
                  <div className="legend-title">Legend</div>
                  <div className="legend-subtitle">{valueLabel(mapKind)} relative strength</div>
                  {[
                    ["Best", "#ef8d32"],
                    ["Above Avg", "#c99b59"],
                    ["Neutral", "#adb5b9"],
                    ["Below Avg", "#88b1c8"],
                    ["Worst", "#58a7e4"],
                  ].map(([label, color]) => (
                    <div className="legend-row" key={label}>
                      <div className="legend-chip" style={{ backgroundColor: color }} />
                      <div>{label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
