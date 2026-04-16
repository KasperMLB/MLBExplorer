import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Streamlit } from "streamlit-component-lib";

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function rowAtStart(el) {
  return el.scrollLeft <= 4;
}

function rowAtEnd(el) {
  return el.scrollLeft + el.clientWidth >= el.scrollWidth - 4;
}

function gameStatusBadge(status) {
  if (!status) return null;
  const s = status.toLowerCase();
  if (s.includes("progress") || s === "live") return { label: "Live", color: "#16a34a" };
  if (s.includes("final") || s.includes("game over") || s.includes("completed")) return { label: "Final", color: "#6b7280" };
  if (s.includes("postponed") || s.includes("suspended")) return { label: "PPD", color: "#d97706" };
  return null;
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

function stickySelectorStyles() {
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
    .sticky-game-nav-frame {
      width: 100%;
    }
    .sticky-section-row {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 10px 0 0 0;
      scrollbar-width: thin;
    }
    .section-chip {
      appearance: none;
      border: 1px solid rgba(31, 41, 55, 0.16);
      border-radius: 8px;
      background: #f8fafc;
      color: #374151;
      cursor: pointer;
      flex: 0 0 auto;
      font: inherit;
      font-size: 0.9rem;
      font-weight: 650;
      padding: 8px 12px;
      white-space: nowrap;
    }
    .section-chip.is-active {
      border-color: #1f2937;
      background: #eef3f8;
      color: #111827;
      box-shadow: inset 0 0 0 1px rgba(31, 41, 55, 0.16);
    }
    .section-chip:disabled {
      cursor: default;
      opacity: 0.45;
    }
    .kasper-sticky-nav {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 999999;
      background: rgba(255, 255, 252, 0.96);
      border-bottom: 1px solid rgba(31, 41, 55, 0.14);
      box-shadow: 0 10px 26px rgba(15, 23, 42, 0.10);
      padding: 8px 16px 10px 16px;
      backdrop-filter: blur(10px);
      transform: translateY(-110%);
      transition: transform 0.22s ease;
      pointer-events: none;
    }
    .kasper-sticky-nav.is-visible {
      transform: translateY(0);
      pointer-events: auto;
    }
    .kasper-sticky-inner {
      max-width: 1760px;
      margin: 0 auto;
    }
    .sticky-game-row-outer {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .chip-row-fade-wrap {
      position: relative;
      flex: 1 1 0;
      min-width: 0;
    }
    .chip-row-fade-wrap::before,
    .chip-row-fade-wrap::after {
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      width: 36px;
      pointer-events: none;
      z-index: 1;
      opacity: 0;
      transition: opacity 0.15s;
    }
    .chip-row-fade-wrap::before {
      left: 0;
      background: linear-gradient(to right, rgba(255, 255, 252, 0.96), transparent);
    }
    .chip-row-fade-wrap::after {
      right: 0;
      background: linear-gradient(to left, rgba(255, 255, 252, 0.96), transparent);
    }
    .chip-row-fade-wrap.show-left::before { opacity: 1; }
    .chip-row-fade-wrap.show-right::after { opacity: 1; }
    .sticky-game-chip-row,
    .sticky-section-chip-row {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 2px;
      scrollbar-width: thin;
    }
    .sticky-section-chip-row {
      margin-top: 7px;
    }
    .sticky-game-chip {
      appearance: none;
      border: 1px solid rgba(31, 41, 55, 0.16);
      border-radius: 8px;
      background: #f8fafc;
      color: #1f2937;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      flex: 0 0 auto;
      font: inherit;
      font-weight: 750;
      gap: 4px;
      min-height: 44px;
      min-width: 86px;
      padding: 5px 9px;
      white-space: nowrap;
    }
    .sticky-game-chip.is-active,
    .sticky-section-chip.is-active {
      border-color: #1f2937;
      background: #eef3f8;
      color: #111827;
      box-shadow: inset 0 0 0 1px rgba(31, 41, 55, 0.16);
    }
    .sticky-game-chip.summary-chip {
      min-width: 116px;
      font-size: 0.9rem;
    }
    .sticky-chip-logo-row {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .sticky-game-chip .team-logo-img {
      width: 28px;
      height: 28px;
    }
    .sticky-game-chip .matchup-at {
      font-size: 0.88rem;
    }
    .sticky-game-chip .team-logo-fallback {
      min-width: 22px;
      font-size: 0.68rem;
    }
    .sticky-chip-status-badge {
      font-size: 0.64rem;
      font-weight: 700;
      line-height: 1;
      letter-spacing: 0.03em;
    }
    .sticky-section-chip {
      appearance: none;
      border: 1px solid rgba(31, 41, 55, 0.16);
      border-radius: 8px;
      background: #ffffff;
      color: #374151;
      cursor: pointer;
      flex: 0 0 auto;
      font: inherit;
      font-size: 0.82rem;
      font-weight: 700;
      min-height: 34px;
      padding: 6px 10px;
      white-space: nowrap;
    }
    .sticky-scroll-top-btn {
      appearance: none;
      background: transparent;
      border: 1px solid rgba(31, 41, 55, 0.16);
      border-radius: 8px;
      color: #374151;
      cursor: pointer;
      flex: 0 0 auto;
      font: inherit;
      font-size: 1rem;
      font-weight: 700;
      min-height: 44px;
      min-width: 36px;
      padding: 4px 8px;
    }
    .sticky-scroll-top-btn:hover {
      background: rgba(31, 41, 55, 0.06);
    }
    @media (max-width: 760px) {
      .kasper-sticky-nav {
        padding: 7px 10px 8px 10px;
      }
      .sticky-game-chip {
        min-height: 40px;
        min-width: 74px;
        padding: 4px 7px;
      }
      .sticky-game-chip.summary-chip {
        min-width: 104px;
      }
      .sticky-game-chip .team-logo-img {
        width: 24px;
        height: 24px;
      }
      .sticky-section-chip {
        font-size: 0.78rem;
        min-height: 32px;
      }
      .sticky-scroll-top-btn {
        min-height: 40px;
        min-width: 32px;
      }
    }
  `;
}

function SectionRow({ sections, selectedSection, disabled, onSelect }) {
  return (
    <div className="sticky-section-row" aria-label="Section">
      {sections.map((section) => (
        <button
          type="button"
          className={`section-chip${section === selectedSection ? " is-active" : ""}`}
          disabled={disabled}
          onClick={() => onSelect(section)}
          key={section}
        >
          {section}
        </button>
      ))}
    </div>
  );
}

function StickyPortalBar({ parentDocument, visible, cards, selectedKey, selectedSection, sections, onGameSelect, onSectionSelect }) {
  const activeChipRef = useRef(null);
  const gameRowRef = useRef(null);
  const sectionRowRef = useRef(null);
  const selectedIsSummary = selectedKey === "__slate_summary__";

  // Auto-scroll active chip into view when bar becomes visible or selection changes
  useEffect(() => {
    if (visible && activeChipRef.current) {
      activeChipRef.current.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    }
  }, [selectedKey, visible]);

  // Overflow fade indicators for game chip row
  useEffect(() => {
    const row = gameRowRef.current;
    if (!row) return;
    const wrapper = row.parentElement;
    const update = () => {
      const hasOverflow = row.scrollWidth > row.clientWidth + 4;
      wrapper.classList.toggle("show-left", !rowAtStart(row) && hasOverflow);
      wrapper.classList.toggle("show-right", !rowAtEnd(row) && hasOverflow);
    };
    const t = setTimeout(update, 60);
    row.addEventListener("scroll", update, { passive: true });
    return () => {
      clearTimeout(t);
      row.removeEventListener("scroll", update);
    };
  }, [cards.length, visible]);

  // Overflow fade indicators for section chip row
  useEffect(() => {
    const row = sectionRowRef.current;
    if (!row) return;
    const wrapper = row.parentElement;
    const update = () => {
      const hasOverflow = row.scrollWidth > row.clientWidth + 4;
      wrapper.classList.toggle("show-left", !rowAtStart(row) && hasOverflow);
      wrapper.classList.toggle("show-right", !rowAtEnd(row) && hasOverflow);
    };
    const t = setTimeout(update, 60);
    row.addEventListener("scroll", update, { passive: true });
    return () => {
      clearTimeout(t);
      row.removeEventListener("scroll", update);
    };
  }, [selectedKey, sections.length]);

  if (!parentDocument) return null;

  const scrollToTop = () => {
    try { window.parent.scrollTo({ top: 0, behavior: "smooth" }); } catch {}
  };

  return createPortal(
    <>
      <style>{stickySelectorStyles()}</style>
      <div className={`kasper-sticky-nav${visible ? " is-visible" : ""}`}>
        <div className="kasper-sticky-inner">
          <div className="sticky-game-row-outer">
            <div className="chip-row-fade-wrap">
              <div className="sticky-game-chip-row" ref={gameRowRef} aria-label="Sticky game selector">
                {cards.map((card) => {
                  const active = String(card.selectionKey) === selectedKey;
                  if (card?.isSummary) {
                    return (
                      <button
                        type="button"
                        ref={active ? activeChipRef : null}
                        className={`sticky-game-chip summary-chip${active ? " is-active" : ""}`}
                        onClick={() => onGameSelect(card.selectionKey)}
                        key={card.selectionKey}
                      >
                        Slate Summary
                      </button>
                    );
                  }
                  const badge = gameStatusBadge(card?.status);
                  return (
                    <button
                      type="button"
                      ref={active ? activeChipRef : null}
                      className={`sticky-game-chip${active ? " is-active" : ""}`}
                      onClick={() => onGameSelect(card.selectionKey)}
                      aria-label={`${card?.awayTeam || ""} at ${card?.homeTeam || ""}`}
                      key={card.selectionKey}
                    >
                      <span className="sticky-chip-logo-row">
                        <TeamLogo src={card?.awayLogo} team={card?.awayTeam} />
                        <span className="matchup-at">@</span>
                        <TeamLogo src={card?.homeLogo} team={card?.homeTeam} />
                      </span>
                      {badge ? (
                        <span className="sticky-chip-status-badge" style={{ color: badge.color }}>
                          {badge.label}
                        </span>
                      ) : null}
                    </button>
                  );
                })}
              </div>
            </div>
            <button
              type="button"
              className="sticky-scroll-top-btn"
              onClick={scrollToTop}
              aria-label="Scroll to top"
            >
              ↑
            </button>
          </div>
          {!selectedIsSummary ? (
            <div className="chip-row-fade-wrap" style={{ marginTop: "7px" }}>
              <div className="sticky-section-chip-row" ref={sectionRowRef} aria-label="Sticky section selector">
                {sections.map((section) => (
                  <button
                    type="button"
                    className={`sticky-section-chip${section === selectedSection ? " is-active" : ""}`}
                    onClick={() => onSectionSelect(section)}
                    key={section}
                  >
                    {section}
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </>,
    parentDocument.body
  );
}

export function StickyGameNav({ args }) {
  const cards = useMemo(() => safeArray(args?.cards), [args?.cards]);
  const sections = useMemo(() => safeArray(args?.sections).map(String), [args?.sections]);
  const initialSelectedKey = String(args?.selectedKey || cards[0]?.selectionKey || "");
  const initialSection = String(args?.selectedSection || sections[0] || "");
  const [selectedKey, setSelectedKey] = useState(initialSelectedKey);
  const [selectedSection, setSelectedSection] = useState(initialSection);
  const [stickyVisible, setStickyVisible] = useState(false);
  const [parentDocument, setParentDocument] = useState(null);

  useEffect(() => {
    setSelectedKey(initialSelectedKey);
  }, [initialSelectedKey]);

  useEffect(() => {
    setSelectedSection(initialSection);
  }, [initialSection]);

  useEffect(() => {
    try {
      setParentDocument(window.parent?.document || null);
    } catch {
      setParentDocument(null);
    }
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      Streamlit.setFrameHeight(document.documentElement.scrollHeight);
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [cards.length, selectedKey, selectedSection, sections.length]);

  useEffect(() => {
    let frameEl = null;
    try {
      frameEl = window.frameElement || null;
    } catch {
      frameEl = null;
    }
    if (!frameEl) return undefined;
    let ParentIO = null;
    try {
      ParentIO = window.parent?.IntersectionObserver || null;
    } catch {
      ParentIO = null;
    }
    if (!ParentIO) return undefined;
    const observer = new ParentIO(
      ([entry]) => {
        setStickyVisible(!entry.isIntersecting && entry.boundingClientRect.top < 0);
      },
      { threshold: 0 }
    );
    observer.observe(frameEl);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    return () => {
      setStickyVisible(false);
    };
  }, []);

  const emitSelection = (nextKey, nextSection) => {
    Streamlit.setComponentValue({
      selectionKey: String(nextKey || selectedKey),
      section: String(nextSection || selectedSection || sections[0] || ""),
    });
  };

  const onGameSelect = (selectionKey) => {
    if (!selectionKey) return;
    const next = String(selectionKey);
    setSelectedKey(next);
    emitSelection(next, selectedSection);
  };

  const onSectionSelect = (section) => {
    if (!section) return;
    const next = String(section);
    setSelectedSection(next);
    emitSelection(selectedKey, next);
  };

  const selectedIsSummary = selectedKey === "__slate_summary__";

  return (
    <>
      <style>{`${selectorStyles()}\n${stickySelectorStyles()}`}</style>
      <div className="sticky-game-nav-frame">
        <div className="game-logo-selector-label">Game</div>
        <div className="game-logo-selector-grid">
          {cards.map((card) => (
            <SelectorCard
              key={card.selectionKey}
              card={card}
              active={String(card.selectionKey) === selectedKey}
              onSelect={onGameSelect}
            />
          ))}
        </div>
        {!selectedIsSummary ? (
          <SectionRow
            sections={sections}
            selectedSection={selectedSection}
            disabled={false}
            onSelect={onSectionSelect}
          />
        ) : null}
      </div>
      {parentDocument ? (
        <StickyPortalBar
          parentDocument={parentDocument}
          visible={stickyVisible}
          cards={cards}
          selectedKey={selectedKey}
          selectedSection={selectedSection}
          sections={sections}
          onGameSelect={onGameSelect}
          onSectionSelect={onSectionSelect}
        />
      ) : null}
    </>
  );
}
