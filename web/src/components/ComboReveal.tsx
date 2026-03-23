import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type { ComboResult } from "@/lib/puzzle";

interface ComboRevealProps {
  combo: ComboResult;
  cards: string[];
  previouslyFoundWords: Set<string>;
  onDismiss: () => void;
  onRevealWord?: (word: string) => void;
}

type Phase = "cards" | "merged" | "split";

function computeBoundaries(words: string[]): Set<number> {
  const b = new Set<number>();
  let pos = 0;
  for (let i = 0; i < words.length - 1; i++) {
    pos += words[i]!.length;
    b.add(pos);
  }
  return b;
}

type WordClass = "original" | "already-found" | "new";

/** Classify each word in the segmentation */
function classifyWord(
  word: string,
  wordStartPos: number,
  cards: string[],
  previouslyFoundWords: Set<string>
): WordClass {
  let isOriginal = false;
  let cardPos = 0;
  for (const card of cards) {
    if (word === card && wordStartPos === cardPos) {
      isOriginal = true;
      break;
    }
    cardPos += card.length;
  }

  if (isOriginal || word.length < 4) {
    return "original";
  } else if (previouslyFoundWords.has(word)) {
    return "already-found";
  } else {
    return "new";
  }
}

/** Build word groups from segmentation with their classifications and positions */
function buildWordGroups(
  seg: string[],
  cards: string[],
  previouslyFoundWords: Set<string>
): { word: string; cls: WordClass; startPos: number }[] {
  const groups: { word: string; cls: WordClass; startPos: number }[] = [];
  let pos = 0;
  for (const word of seg) {
    const cls = classifyWord(word, pos, cards, previouslyFoundWords);
    groups.push({ word, cls, startPos: pos });
    pos += word.length;
  }
  return groups;
}

export function ComboReveal({ combo, cards, previouslyFoundWords, onDismiss, onRevealWord }: ComboRevealProps) {
  // Fall back to segmentations if bestSegmentations is missing (e.g., from older saved progress)
  const segs = combo.bestSegmentations?.length ? combo.bestSegmentations : combo.segmentations;
  const [segIndex, setSegIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("cards");
  const [dismissed, setDismissed] = useState(false);
  // Increment spiritKey each time we enter split phase to retrigger ghost animations
  const [spiritKey, setSpiritKey] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const revealTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const currentSeg = segs[segIndex] ?? segs[0]!;
  const concat = combo.concat.toUpperCase();

  // Reset on new combo
  useEffect(() => {
    setPhase("cards");
    setSegIndex(0);
    setDismissed(false);
    setSpiritKey(0);
  }, [combo]);

  // Drive animation phases
  useEffect(() => {
    if (dismissed) return;

    if (phase === "cards") {
      timerRef.current = setTimeout(() => setPhase("merged"), 1200);
    } else if (phase === "merged") {
      timerRef.current = setTimeout(() => {
        setSpiritKey((k) => k + 1);
        setPhase("split");
      }, 700);
    } else if (phase === "split") {
      timerRef.current = setTimeout(() => {
        if (segIndex < segs.length - 1) {
          // Collapse back to merged before showing next split
          setPhase("merged");
          setSegIndex((i) => i + 1);
        }
      }, 2800);
    }

    return () => clearTimeout(timerRef.current);
  }, [phase, segIndex, segs.length, dismissed]);

  // Fire onRevealWord for each new word in sync with the ghost animation delays
  useEffect(() => {
    if (phase !== "split" || !onRevealWord) return;

    const GHOST_BASE_DELAY = 500;
    const GHOST_STAGGER = 250;

    const groups = buildWordGroups(currentSeg, cards, previouslyFoundWords);
    const newWords = groups.filter((g) => g.cls === "new");

    const timers: ReturnType<typeof setTimeout>[] = [];
    newWords.forEach((g, idx) => {
      const delay = GHOST_BASE_DELAY + idx * GHOST_STAGGER;
      const t = setTimeout(() => onRevealWord(g.word), delay);
      timers.push(t);
    });
    revealTimersRef.current = timers;

    return () => {
      for (const t of timers) clearTimeout(t);
    };
  }, [phase, spiritKey]); // spiritKey changes each time we re-enter split

  const handleDismiss = useCallback(() => {
    setDismissed(true);
    clearTimeout(timerRef.current);
    // Clear pending reveal timers and flush all remaining new words immediately
    for (const t of revealTimersRef.current) clearTimeout(t);
    revealTimersRef.current = [];
    if (onRevealWord) {
      // Reveal any words from ALL segmentations that haven't been revealed yet
      for (const seg of segs) {
        const groups = buildWordGroups(seg, cards, previouslyFoundWords);
        for (const g of groups) {
          if (g.cls === "new") onRevealWord(g.word);
        }
      }
    }
    onDismiss();
  }, [onDismiss, onRevealWord, segs, cards, previouslyFoundWords]);

  // Compute boundaries
  const cardBoundaries = computeBoundaries(cards);
  const segBoundaries = computeBoundaries(currentSeg);

  // Build word groups with classifications
  const wordGroups = useMemo(
    () => buildWordGroups(currentSeg, cards, previouslyFoundWords),
    [currentSeg, cards, previouslyFoundWords]
  );

  // Count new words for stagger logic
  const newWordIndices: number[] = [];
  wordGroups.forEach((g, i) => {
    if (g.cls === "new") newWordIndices.push(i);
  });

  // All possible gap positions (between any two adjacent letters)
  const gapWidths: number[] = [];
  for (let i = 0; i < concat.length - 1; i++) {
    const boundary = i + 1;
    if (phase === "cards" && cardBoundaries.has(boundary)) {
      gapWidths.push(10);
    } else if (phase === "split" && segBoundaries.has(boundary)) {
      gapWidths.push(10);
    } else {
      gapWidths.push(0);
    }
  }

  return (
    <div
      className="combo-reveal w-full cursor-pointer"
      onClick={handleDismiss}
    >
      <div className="flex flex-col items-center gap-1">
        {/* Animated letter display - stable DOM, gaps animate via CSS */}
        <div className="combo-reveal-words">
          {concat.split("").map((char, i) => {
            // Determine if this letter starts a new word that should get a ghost
            const isNewWordStart =
              phase === "split" &&
              wordGroups.some(
                (g) => g.cls === "new" && g.startPos === i
              );

            // Find the word group this letter belongs to (for classification)
            const group = wordGroups.find(
              (g) => i >= g.startPos && i < g.startPos + g.word.length
            );
            const letterClass = group?.cls ?? "original";

            // If this is the start of a new word, compute stagger delay.
            // Base delay lets the split/decomposition gaps open first,
            // then each successive ghost word staggers by 250ms.
            const GHOST_BASE_DELAY = 500;
            const GHOST_STAGGER = 250;
            let staggerDelay = 0;
            if (isNewWordStart) {
              const newIdx = newWordIndices.indexOf(
                wordGroups.findIndex(
                  (g) => g.cls === "new" && g.startPos === i
                )
              );
              staggerDelay = GHOST_BASE_DELAY + newIdx * GHOST_STAGGER;
            }

            // Find the new word text for ghost rendering
            const newWord = isNewWordStart ? group : null;

            return (
              <span key={i} className="inline-flex items-center">
                {i > 0 && (
                  <span
                    className="combo-gap"
                    style={{ width: `${gapWidths[i - 1]}px` }}
                  />
                )}
                <span className="combo-letter-wrapper">
                  <span
                    className={
                      phase === "split"
                        ? `combo-letter ${
                            letterClass === "new"
                              ? "combo-letter-new"
                              : letterClass === "already-found"
                                ? "combo-letter-found"
                                : ""
                          }`
                        : "combo-letter"
                    }
                  >
                    {char}
                  </span>
                  {/* Ghost "spirit" copy for the first letter of each new word */}
                  {newWord && (
                    <span
                      key={`spirit-${spiritKey}-${i}`}
                      className="combo-spirit"
                      style={{ animationDelay: `${staggerDelay}ms` }}
                      aria-hidden="true"
                    >
                      {newWord.word.toUpperCase()}
                    </span>
                  )}
                </span>
              </span>
            );
          })}
        </div>

        {/* Segmentation dots indicator if multiple segs */}
        {segs.length > 1 && (
          <div className="flex gap-1 mt-0.5">
            {segs.map((_, i) => (
              <div
                key={i}
                className={`w-1.5 h-1.5 rounded-full transition-colors duration-300 ${
                  i <= segIndex ? "bg-foreground" : "bg-border"
                }`}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
