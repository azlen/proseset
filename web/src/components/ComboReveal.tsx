import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type { ComboResult } from "@/lib/puzzle";

interface ComboRevealProps {
  combo: ComboResult;
  cards: string[];
  previouslyFoundWords: Set<string>;
  onDismiss: () => void;
  onWordRevealed?: (word: string) => void;
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

const FADE_OUT_DURATION = 500; // ms for the fade-out animation

export function ComboReveal({ combo, cards, previouslyFoundWords, onDismiss, onWordRevealed }: ComboRevealProps) {
  // Fall back to segmentations if bestSegmentations is missing (e.g., from older saved progress)
  const segs = combo.bestSegmentations?.length ? combo.bestSegmentations : combo.segmentations;
  const [segIndex, setSegIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("cards");
  const [dismissed, setDismissed] = useState(false);
  const [fadingOut, setFadingOut] = useState(false);
  // Increment spiritKey each time we enter split phase to retrigger ghost animations
  const [spiritKey, setSpiritKey] = useState(0);
  // Track words revealed during this combo reveal so subsequent segmentations show them as gray
  const [revealedDuringCombo, setRevealedDuringCombo] = useState<Set<string>>(new Set());
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const fadeTimerRef = useRef<ReturnType<typeof setTimeout>>();

  const currentSeg = segs[segIndex] ?? segs[0]!;
  const concat = combo.concat.toUpperCase();

  // Combine previouslyFoundWords with words revealed during this combo animation
  const effectiveFoundWords = useMemo(() => {
    if (revealedDuringCombo.size === 0) return previouslyFoundWords;
    const combined = new Set(previouslyFoundWords);
    for (const w of revealedDuringCombo) combined.add(w);
    return combined;
  }, [previouslyFoundWords, revealedDuringCombo]);

  // Reset on new combo
  useEffect(() => {
    setPhase("cards");
    setSegIndex(0);
    setDismissed(false);
    setFadingOut(false);
    setSpiritKey(0);
    setRevealedDuringCombo(new Set());
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
          // Before advancing, record all "new" words from this segmentation
          // so the next segmentation will show them as already-found (gray)
          const currentGroups = buildWordGroups(currentSeg, cards, effectiveFoundWords);
          const newWords = currentGroups.filter((g) => g.cls === "new").map((g) => g.word);
          if (newWords.length > 0) {
            setRevealedDuringCombo((prev) => {
              const next = new Set(prev);
              for (const w of newWords) next.add(w);
              return next;
            });
          }
          // Collapse back to merged before showing next split
          setPhase("merged");
          setSegIndex((i) => i + 1);
        } else {
          // Last segmentation finished — start fade out
          setFadingOut(true);
        }
      }, 2800);
    }

    return () => clearTimeout(timerRef.current);
  }, [phase, segIndex, segs.length, dismissed]);

  // When fade-out starts, wait for animation then dismiss
  useEffect(() => {
    if (!fadingOut) return;
    fadeTimerRef.current = setTimeout(() => {
      onDismiss();
    }, FADE_OUT_DURATION);
    return () => clearTimeout(fadeTimerRef.current);
  }, [fadingOut, onDismiss]);

  const handleDismiss = useCallback(() => {
    setDismissed(true);
    clearTimeout(timerRef.current);
    clearTimeout(fadeTimerRef.current);
    // Clear pending word-reveal timers
    for (const t of wordRevealTimersRef.current) clearTimeout(t);
    wordRevealTimersRef.current = [];
    // Flush any unrevealed new words immediately
    // Use a running set so we don't double-report words across segmentations
    if (onWordRevealed) {
      const seen = new Set(effectiveFoundWords);
      for (const seg of segs) {
        const groups = buildWordGroups(seg, cards, seen);
        for (const g of groups) {
          if (g.cls === "new") {
            onWordRevealed(g.word);
            seen.add(g.word);
          }
        }
      }
    }
    onDismiss();
  }, [onDismiss, onWordRevealed, segs, cards, effectiveFoundWords]);

  // Compute boundaries
  const cardBoundaries = computeBoundaries(cards);
  const segBoundaries = computeBoundaries(currentSeg);

  // Build word groups with classifications (using effectiveFoundWords to gray-out words revealed earlier in this combo)
  const wordGroups = useMemo(
    () => buildWordGroups(currentSeg, cards, effectiveFoundWords),
    [currentSeg, cards, effectiveFoundWords]
  );

  // Count new words for stagger logic
  const newWordIndices: number[] = [];
  wordGroups.forEach((g, i) => {
    if (g.cls === "new") newWordIndices.push(i);
  });

  // Timing constants for ghost word stagger
  const GHOST_INITIAL_DELAY = 400; // ms after split before first ghost appears
  const GHOST_STAGGER = 250; // ms between each ghost word

  // Track timers for word-revealed callbacks
  const wordRevealTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  // Fire onWordRevealed for each new word, timed to match ghost animation
  useEffect(() => {
    // Clear any pending word-reveal timers
    for (const t of wordRevealTimersRef.current) clearTimeout(t);
    wordRevealTimersRef.current = [];

    if (phase !== "split" || !onWordRevealed) return;

    const newWords = wordGroups.filter((g) => g.cls === "new");
    newWords.forEach((g, idx) => {
      const delay = GHOST_INITIAL_DELAY + idx * GHOST_STAGGER;
      const t = setTimeout(() => {
        onWordRevealed(g.word);
      }, delay);
      wordRevealTimersRef.current.push(t);
    });

    return () => {
      for (const t of wordRevealTimersRef.current) clearTimeout(t);
      wordRevealTimersRef.current = [];
    };
  }, [phase, spiritKey]); // spiritKey changes each time we re-enter split

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
      className={`combo-reveal w-full cursor-pointer${fadingOut ? " combo-reveal-fade-out" : ""}`}
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

            // If this is the start of a new word, compute stagger delay
            let staggerDelay = 0;
            if (isNewWordStart) {
              const newIdx = newWordIndices.indexOf(
                wordGroups.findIndex(
                  (g) => g.cls === "new" && g.startPos === i
                )
              );
              staggerDelay = GHOST_INITIAL_DELAY + newIdx * GHOST_STAGGER;
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
