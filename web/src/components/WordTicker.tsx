import { useState, useRef, useEffect, useCallback } from "react";

interface WordTickerProps {
  foundMadeWords: string[];
  totalCards: number;
  totalWords: number;
  wordLengths: number[];
  usedCards: number;
}

// Animation phases for newly added words:
// 1. "slide"  — wrapper grows width from 0→target over 350ms, pushing siblings right. Content invisible.
// 2. "reveal" — content pops in (scale 0→1) in blue.
// 3. "fade"   — blue transitions to normal text color over 2s.
// null        — animation complete, word rendered normally.
type AnimPhase = "slide" | "reveal" | "fade" | null;

export function WordTicker({ foundMadeWords, totalCards, totalWords, wordLengths, usedCards }: WordTickerProps) {
  const [expanded, setExpanded] = useState(false);
  const displayWords = foundMadeWords.filter((w) => w.length >= 4);

  const [animatingWords, setAnimatingWords] = useState<Set<string>>(new Set());
  const [phase, setPhase] = useState<AnimPhase>(null);
  const prevWordsRef = useRef<Set<string>>(new Set());
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  // We measure widths using a hidden off-screen container
  const measureRef = useRef<HTMLDivElement>(null);
  const [targetWidths, setTargetWidths] = useState<Map<string, number>>(new Map());

  const clearTimers = useCallback(() => {
    for (const t of timersRef.current) clearTimeout(t);
    timersRef.current = [];
  }, []);

  // Detect newly added words and kick off animation sequence
  useEffect(() => {
    const added = displayWords.filter((w) => !prevWordsRef.current.has(w));
    prevWordsRef.current = new Set(displayWords);

    if (added.length === 0) return;

    clearTimers();

    // Measure widths using the hidden container
    const widths = new Map<string, number>();
    if (measureRef.current) {
      // Create temporary span to measure each word
      for (const word of added) {
        const span = document.createElement("span");
        span.className = "px-2 py-0.5 rounded-md bg-muted text-sm font-medium whitespace-nowrap";
        span.textContent = word;
        measureRef.current.appendChild(span);
        widths.set(word, span.offsetWidth);
        measureRef.current.removeChild(span);
      }
    }

    const addedSet = new Set(added);
    setAnimatingWords(addedSet);
    setTargetWidths(widths);
    setPhase("slide");

    // After slide animation completes → reveal
    const t1 = setTimeout(() => {
      setPhase("reveal");

      // After pop-in → fade
      const t2 = setTimeout(() => {
        setPhase("fade");

        // After fade → cleanup
        const t3 = setTimeout(() => {
          setAnimatingWords(new Set());
          setPhase(null);
          setTargetWidths(new Map());
        }, 2000);
        timersRef.current.push(t3);
      }, 400);
      timersRef.current.push(t2);
    }, 400);
    timersRef.current.push(t1);

    return clearTimers;
  }, [displayWords.join(",")]);

  // Build a map of how many words of each length are found vs total
  const foundByLength = new Map<number, number>();
  for (const w of displayWords) {
    foundByLength.set(w.length, (foundByLength.get(w.length) ?? 0) + 1);
  }
  const usedByLength = new Map<number, number>();

  const MIN_LEN = 4;
  const MAX_LEN = 10;
  const MIN_H = 30;
  const MAX_H = 100;

  function renderWord(word: string) {
    const isAnimating = animatingWords.has(word);
    const baseClass = "px-2 py-0.5 rounded-md bg-muted text-sm font-medium whitespace-nowrap";

    if (!isAnimating) {
      return (
        <span key={word} className={`${baseClass} shrink-0`}>
          {word}
        </span>
      );
    }

    const w = targetWidths.get(word) ?? 0;

    // Slide phase: wrapper uses CSS animation to grow width from 0 → target.
    // The gap (6px = gap-1.5) is also included via margin animation.
    if (phase === "slide") {
      return (
        <span
          key={word}
          className="shrink-0 inline-flex overflow-hidden word-ticker-slide"
          style={{ "--target-w": `${w}px` } as React.CSSProperties}
        >
          <span className={baseClass} style={{ opacity: 0 }}>
            {word}
          </span>
        </span>
      );
    }

    // Reveal phase: pop in with blue color
    if (phase === "reveal") {
      return (
        <span key={word} className={`${baseClass} shrink-0 word-ticker-pop`} style={{ color: "#2563eb" }}>
          {word}
        </span>
      );
    }

    // Fade phase: blue → normal color over 2s via CSS animation
    if (phase === "fade") {
      return (
        <span key={word} className={`${baseClass} shrink-0 word-ticker-fade`}>
          {word}
        </span>
      );
    }

    return (
      <span key={word} className={`${baseClass} shrink-0`}>
        {word}
      </span>
    );
  }

  return (
    <div className="w-full overflow-hidden">
      {/* Hidden measurement container */}
      <div
        ref={measureRef}
        aria-hidden
        style={{ position: "absolute", visibility: "hidden", pointerEvents: "none", whiteSpace: "nowrap" }}
      />

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
      <div
        onClick={() => setExpanded(true)}
        className="relative h-10 border-t border-b border-border cursor-pointer"
      >
        <div className="absolute inset-0 flex gap-1.5 items-center overflow-hidden">
          {displayWords.map((word) => renderWord(word))}
        </div>
      </div>

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
