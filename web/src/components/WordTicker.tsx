import { useState, useRef, useEffect, useCallback, useLayoutEffect } from "react";

interface WordTickerProps {
  foundMadeWords: string[];
  totalCards: number;
  totalWords: number;
  wordLengths: number[];
  usedCards: number;
}

/**
 * Animation flow for newly added words:
 *  1. "push"   — new word wrappers animate width from 0 → measured px,
 *                pushing existing words right. Content is invisible.
 *  2. "reveal" — new words pop in (scale + opacity) coloured blue,
 *                then the blue fades to the normal text colour over 2 s.
 *  3. "idle"   — no animation; all words render normally.
 */
type Phase = "idle" | "push" | "reveal";

export function WordTicker({ foundMadeWords, totalCards, totalWords, wordLengths, usedCards }: WordTickerProps) {
  const [expanded, setExpanded] = useState(false);
  const displayWords = foundMadeWords.filter((w) => w.length >= 4);

  const prevWordsRef = useRef<string[]>([]);
  const [newWords, setNewWords] = useState<Set<string>>(new Set());
  const [phase, setPhase] = useState<Phase>("idle");
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  // Refs for measuring new-word natural widths
  const wordElRefs = useRef<Map<string, HTMLSpanElement>>(new Map());
  const [measuredWidths, setMeasuredWidths] = useState<Map<string, number>>(new Map());
  // Flipped after measurement to kick off the CSS width transition
  const [expanding, setExpanding] = useState(false);

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  useEffect(() => clearTimers, [clearTimers]);

  // Detect new words → start animation
  useEffect(() => {
    const prevSet = new Set(prevWordsRef.current);
    const justAdded = displayWords.filter((w) => !prevSet.has(w));

    if (justAdded.length > 0 && prevWordsRef.current.length > 0) {
      clearTimers();
      setNewWords(new Set(justAdded));
      setMeasuredWidths(new Map());
      setExpanding(false);
      setPhase("push");
    } else if (justAdded.length > 0) {
      // Very first words — skip push, go straight to reveal
      clearTimers();
      setNewWords(new Set(justAdded));
      setPhase("reveal");

      // Clean up after animations finish (pop 350ms + colour fade 2s)
      const t = setTimeout(() => {
        setNewWords(new Set());
        setPhase("idle");
      }, 2500);
      timersRef.current.push(t);
    }

    prevWordsRef.current = displayWords;
  }, [displayWords.join(","), clearTimers]);

  // After "push" renders (words in DOM at width 0), measure then expand
  useLayoutEffect(() => {
    if (phase !== "push") return;

    // Measure natural widths of the hidden new-word inner spans
    const widths = new Map<string, number>();
    for (const word of newWords) {
      const el = wordElRefs.current.get(word);
      if (el) {
        widths.set(word, el.offsetWidth);
      }
    }
    setMeasuredWidths(widths);

    // Next frame: flip expanding so CSS transitions width from 0 → measured
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setExpanding(true);

        // After the width transition (400ms), switch to reveal
        const t1 = setTimeout(() => {
          setPhase("reveal");
          setExpanding(false);

          // Clean up after animations finish (pop 350ms + colour fade 2s)
          const t2 = setTimeout(() => {
            setNewWords(new Set());
            setPhase("idle");
            setMeasuredWidths(new Map());
          }, 2500);
          timersRef.current.push(t2);
        }, 420);
        timersRef.current.push(t1);
      });
    });
  }, [phase, newWords]);

  // ───────── histogram helpers ─────────
  const foundByLength = new Map<number, number>();
  for (const w of displayWords) {
    foundByLength.set(w.length, (foundByLength.get(w.length) ?? 0) + 1);
  }
  const usedByLength = new Map<number, number>();

  const MIN_LEN = 4;
  const MAX_LEN = 10;
  const MIN_H = 30;
  const MAX_H = 100;

  return (
    <div className="w-full overflow-hidden">
      <div className="flex justify-between items-end text-sm text-muted-foreground mb-2">
        <span className="shrink-0">{displayWords.length} / {totalWords} words found</span>
        <div className="flex items-end h-5 mx-3 w-48">
          {wordLengths.map((len, i) => {
            const clamped = Math.min(Math.max(len, MIN_LEN), MAX_LEN);
            const t = (clamped - MIN_LEN) / (MAX_LEN - MIN_LEN);
            const hPct = MIN_H + t * (MAX_H - MIN_H);
            const used = usedByLength.get(len) ?? 0;
            const found = foundByLength.get(len) ?? 0;
            const isFound = used < found;
            usedByLength.set(len, used + 1);
            return (
              <div
                key={i}
                className="flex-1"
                style={{
                  height: `${hPct}%`,
                  backgroundColor: isFound ? "#000" : "#d4d4d4",
                }}
              />
            );
          })}
        </div>
        <span className="shrink-0">{usedCards}/{totalCards} cards used</span>
      </div>

      {/* ─── scrolling word ribbon ─── */}
      <div
        onClick={() => setExpanded(true)}
        className="relative h-10 border-t border-b border-border cursor-pointer"
      >
        <div className="absolute inset-0 flex gap-1.5 items-center overflow-hidden">
          {displayWords.map((word) => {
            const isNew = newWords.has(word);
            const isPush = isNew && phase === "push";
            const isReveal = isNew && phase === "reveal";

            if (isPush) {
              const targetW = expanding ? measuredWidths.get(word) ?? "auto" : 0;
              return (
                <span
                  key={word}
                  className="ticker-push-wrapper shrink-0"
                  style={{ width: typeof targetW === "number" ? `${targetW}px` : targetW }}
                >
                  <span
                    ref={(el) => { if (el) wordElRefs.current.set(word, el); }}
                    className="px-2 py-0.5 rounded-md bg-muted text-sm font-medium whitespace-nowrap"
                    style={{ visibility: "hidden" }}
                  >
                    {word}
                  </span>
                </span>
              );
            }

            return (
              <span
                key={word}
                className={[
                  "px-2 py-0.5 rounded-md bg-muted text-sm font-medium shrink-0",
                  isReveal ? "ticker-word-reveal" : "",
                ].join(" ")}
              >
                {word}
              </span>
            );
          })}
        </div>
      </div>

      {/* ─── expanded overlay ─── */}
      {expanded && (
        <div
          className="fixed inset-0 z-50 flex items-end justify-center bg-black/30"
          onClick={() => setExpanded(false)}
        >
          <div
            className="w-full max-w-lg bg-card rounded-t-2xl p-6 pb-8 max-h-[70vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-4">
              <h2 className="font-bold text-lg">
                {displayWords.length} / {totalWords} words found
              </h2>
              <button
                onClick={() => setExpanded(false)}
                className="text-muted-foreground text-2xl leading-none cursor-pointer"
              >
                &times;
              </button>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {displayWords.map((word) => (
                <span
                  key={word}
                  className="px-2 py-0.5 rounded-md bg-muted text-sm font-medium"
                >
                  {word}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
