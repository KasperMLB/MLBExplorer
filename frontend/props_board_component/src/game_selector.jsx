import React, { useEffect, useMemo, useState } from "react";
import { Streamlit } from "streamlit-component-lib";

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function selectorStyles() {
  return `
    :root {
      color-scheme: light;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      background: transparent;
      color: #1f2937;
      font-family: "Segoe UI", system-ui, sans-serif;
    }
    .game-logo-selector-label {
      font-size: 1rem;
      font-weight: 650;
      margin: 0 0 8px 0;
    }
    .game-logo-selector-grid {
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 8px;
      width: 100%;
      margin: 0;
    }
    .game-logo-card {
      appearance: none;
      box-sizing: border-box;
      width: 100%;
      min-height: 92px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      border: 1px solid rgba(31, 41, 55, 0.16);
      border-radius: 8px;
      background: #f8fafc;
      padding: 10px 12px;
      color: #1f2937;
      text-decoration: none;
      cursor: pointer;
      font: inherit;
    }
    .game-logo-card:focus-visible {
      outline: 2px solid rgba(31, 41, 55, 0.55);
      outline-offset: 2px;
    }
    .game-logo-card.is-active {
      border-color: #1f2937;
      background: #eef3f8;
      box-shadow: inset 0 0 0 1px rgba(31, 41, 55, 0.20);
    }
    .matchup-logo-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-width: 0;
    }
    .team-logo-img {
      width: 42px;
      height: 42px;
      object-fit: contain;
      flex: 0 0 auto;
    }
    .matchup-at {
      color: #1f2937;
      font-weight: 800;
      font-size: 1.05rem;
      line-height: 1;
    }
    .team-logo-fallback {
      color: #1f2937;
      font-weight: 800;
      font-size: 0.78rem;
      line-height: 1;
      min-width: 28px;
      text-align: center;
    }
    .summary-card {
      color: #1f2937;
      font-weight: 750;
      letter-spacing: 0;
    }
    .game-logo-card-summary-title {
      font-size: 1rem;
      font-weight: 800;
      text-decoration: underline;
      text-underline-offset: 4px;
      text-align: center;
      line-height: 1.15;
    }
    .game-logo-card-status {
      color: #6b7280;
      font-size: 0.88rem;
      line-height: 1;
      margin-top: 10px;
      text-decoration: underline;
      text-underline-offset: 3px;
    }
    .game-logo-card.is-active .game-logo-card-status {
      color: #1f2937;
      font-weight: 750;
    }
    @media (max-width: 1180px) {
      .game-logo-selector-grid {
        grid-template-columns: repeat(auto-fit, minmax(126px, 1fr));
      }
    }
    @media (max-width: 760px) {
      .game-logo-selector-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }
      .game-logo-card {
        min-height: 86px;
        padding: 8px 8px;
      }
      .team-logo-img {
        width: 34px;
        height: 34px;
      }
      .matchup-logo-row {
        gap: 7px;
      }
      .matchup-at {
        font-size: 1rem;
      }
      .team-logo-fallback {
        font-size: 0.72rem;
        min-width: 24px;
      }
      .game-logo-card-summary-title {
        font-size: 0.95rem;
      }
      .game-logo-card-status {
        font-size: 0.82rem;
        margin-top: 8px;
      }
    }
  `;
}

function TeamLogo({ src, team }) {
  if (!src) {
    return <span className="team-logo-fallback">{team || ""}</span>;
  }
  return <img className="team-logo-img" src={src} alt={team || ""} />;
}

function SelectorCard({ card, active, onSelect }) {
  const status = active ? "Selected" : "Open";
  if (card?.isSummary) {
    return (
      <button
        type="button"
        className={`game-logo-card summary-card${active ? " is-active" : ""}`}
        onClick={() => onSelect(card.selectionKey)}
      >
        <span className="game-logo-card-summary-title">Slate Summary</span>
        <span className="game-logo-card-status">{status}</span>
      </button>
    );
  }

  return (
    <button
      type="button"
      className={`game-logo-card${active ? " is-active" : ""}`}
      onClick={() => onSelect(card.selectionKey)}
      aria-label={`${card?.awayTeam || ""} at ${card?.homeTeam || ""}`}
    >
      <span className="matchup-logo-row">
        <TeamLogo src={card?.awayLogo} team={card?.awayTeam} />
        <span className="matchup-at">@</span>
        <TeamLogo src={card?.homeLogo} team={card?.homeTeam} />
      </span>
      <span className="game-logo-card-status">{status}</span>
    </button>
  );
}

export function GameSelector({ args }) {
  const cards = useMemo(() => safeArray(args?.cards), [args?.cards]);
  const selectedKey = String(args?.selectedKey || cards[0]?.selectionKey || "");
  const [localSelected, setLocalSelected] = useState(selectedKey);

  useEffect(() => {
    setLocalSelected(selectedKey);
  }, [selectedKey]);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      Streamlit.setFrameHeight(document.documentElement.scrollHeight);
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [cards.length, localSelected]);

  const onSelect = (selectionKey) => {
    if (!selectionKey) return;
    const next = String(selectionKey);
    setLocalSelected(next);
    Streamlit.setComponentValue(next);
  };

  return (
    <>
      <style>{selectorStyles()}</style>
      <div className="game-logo-selector-label">Game</div>
      <div className="game-logo-selector-grid">
        {cards.map((card) => (
          <SelectorCard
            key={card.selectionKey}
            card={card}
            active={String(card.selectionKey) === localSelected}
            onSelect={onSelect}
          />
        ))}
      </div>
    </>
  );
}
