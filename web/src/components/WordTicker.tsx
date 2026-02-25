import { useState } from "react";

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
  const foundSet = new Set(displayWords.map((w) => w.length));

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
          {displayWords.map((word) => (
            <span
              key={word}
              className="px-2 py-0.5 rounded-md bg-muted text-sm font-medium shrink-0"
            >
              {word}
            </span>
          ))}
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
