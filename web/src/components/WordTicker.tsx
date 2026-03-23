import { useState, useRef, useEffect, type CSSProperties } from "react";

interface WordTickerProps {
  foundMadeWords: string[];
  totalCards: number;
  totalWords: number;
  wordLengths: number[];
  usedCards: number;
}

export function WordTicker({ foundMadeWords, totalCards, totalWords, wordLengths, usedCards }: WordTickerProps) {
  const [expanded, setExpanded] = useState(false);
  const displayWords = foundMadeWords.filter((w) => w.length >= 4);

  // Animation phases for newly added words:
  // "hidden"   — in DOM with max-width:0, invisible, no space taken
  // "pushing"  — max-width expanding, pushes existing words right, still invisible
  // "popping"  — pop-in animation plays, word becomes visible with blue color
  // "fading"   — blue color transitions to gray over 2s
  // (removed)  — word is fully settled, no special state
  const prevWordsRef = useRef<string[] | null>(null);
  const [hiddenWords, setHiddenWords] = useState<Set<string>>(new Set());
  const [pushingWords, setPushingWords] = useState<Set<string>>(new Set());
  const [poppingWords, setPoppingWords] = useState<Set<string>>(new Set());
  const [fadingWords, setFadingWords] = useState<Set<string>>(new Set());
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    // On first render, just record current words — don't animate them
    if (prevWordsRef.current === null) {
      prevWordsRef.current = displayWords;
      return;
    }

    const prevSet = new Set(prevWordsRef.current);
    const added = displayWords.filter((w) => !prevSet.has(w));
    prevWordsRef.current = displayWords;

    if (added.length === 0) return;

    // Phase 1: Insert as hidden (max-width:0, no padding, invisible)
    setHiddenWords((prev) => new Set([...prev, ...added]));
    setPushingWords((prev) => { const n = new Set(prev); for (const w of added) n.delete(w); return n; });
    setPoppingWords((prev) => { const n = new Set(prev); for (const w of added) n.delete(w); return n; });

    // Phase 2: After one frame, start the push (expand max-width, still invisible)
    const pushTimer = requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setHiddenWords((prev) => {
          const next = new Set(prev);
          for (const w of added) next.delete(w);
          return next;
        });
        setPushingWords((prev) => new Set([...prev, ...added]));
      });
    });

    // Phase 3: After push transition completes (350ms), pop in the words
    const popTimer = setTimeout(() => {
      setPushingWords((prev) => {
        const next = new Set(prev);
        for (const w of added) next.delete(w);
        return next;
      });
      setPoppingWords((prev) => new Set([...prev, ...added]));

      // Phase 4: After pop animation (350ms), transition to fading state
      const fadeStartTimer = setTimeout(() => {
        setPoppingWords((prev) => {
          const next = new Set(prev);
          for (const w of added) next.delete(w);
          return next;
        });
        setFadingWords((prev) => new Set([...prev, ...added]));

        // Phase 5: After 2s fade, remove all special state
        const fadeEndTimer = setTimeout(() => {
          setFadingWords((prev) => {
            const next = new Set(prev);
            for (const w of added) next.delete(w);
            return next;
          });
        }, 2000);
        timersRef.current.push(fadeEndTimer);
      }, 350);
      timersRef.current.push(fadeStartTimer);
    }, 400);
    timersRef.current.push(popTimer);

    return () => {
      cancelAnimationFrame(pushTimer);
    };
  }, [displayWords.join(",")]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      for (const timer of timersRef.current) {
        clearTimeout(timer);
      }
    };
  }, []);

  // Build a map of how many words of each length are found vs total
  const foundByLength = new Map<number, number>();
  for (const w of displayWords) {
    foundByLength.set(w.length, (foundByLength.get(w.length) ?? 0) + 1);
  }

  // Track which word lengths have been "used up" as we render lines
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
      <div
        onClick={() => setExpanded(true)}
        className="relative h-10 border-t border-b border-border cursor-pointer"
      >
        <div className="absolute inset-0 flex gap-1.5 items-center overflow-hidden">
          {displayWords.map((word) => {
            const isHidden = hiddenWords.has(word);
            const isPushing = pushingWords.has(word);
            const isPopping = poppingWords.has(word);
            const isFading = fadingWords.has(word);

            let className = "ticker-word px-2 py-0.5 rounded-md text-sm font-medium shrink-0";
            if (isHidden) {
              className += " ticker-word-hidden";
            } else if (isPushing) {
              className += " ticker-word-pushing";
            } else if (isPopping) {
              className += " ticker-word-pop";
            }

            // Color/bg: blue for popping, transitions to gray for fading, default for settled
            const style: CSSProperties = {};
            if (isPopping) {
              style.backgroundColor = "#dbeafe";
              style.color = "#2563eb";
            } else if (isFading) {
              style.backgroundColor = "#f5f5f5";
              style.color = "inherit";
              style.transition = "background-color 2s ease, color 2s ease";
            }

            return (
              <span key={word} className={className} style={style}>
                {word}
              </span>
            );
          })}
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
