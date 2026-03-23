import { useState, useEffect, useRef, useCallback } from "react";
import type { ComboResult } from "@/lib/puzzle";

export interface WordRect {
  word: string;
  rect: DOMRect;
}

interface ComboRevealProps {
  combo: ComboResult;
  cards: string[];
  onDismiss: () => void;
  onNewWordsPositioned?: (wordRects: WordRect[]) => void;
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

/** Check which seg words are "new" (not matching an original card at the same position) */
function getNewWordRanges(seg: string[], cards: string[]): Set<number> {
  // Returns set of letter indices that belong to a "new" 4+ letter word
  const newIndices = new Set<number>();
  let segPos = 0;
  for (const word of seg) {
    let isOriginal = false;
    let cardPos = 0;
    for (const card of cards) {
      if (word === card && segPos === cardPos) {
        isOriginal = true;
        break;
      }
      cardPos += card.length;
    }
    if (!isOriginal && word.length >= 4) {
      for (let i = segPos; i < segPos + word.length; i++) {
        newIndices.add(i);
      }
    }
    segPos += word.length;
  }
  return newIndices;
}

export function ComboReveal({ combo, cards, onDismiss, onNewWordsPositioned }: ComboRevealProps) {
  // Fall back to segmentations if bestSegmentations is missing (e.g., from older saved progress)
  const segs = combo.bestSegmentations?.length ? combo.bestSegmentations : combo.segmentations;
  const [segIndex, setSegIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("cards");
  const [dismissed, setDismissed] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const wordSpanRefs = useRef<Map<string, HTMLSpanElement>>(new Map());
  const wordsReportedRef = useRef(false);

  const currentSeg = segs[segIndex] ?? segs[0]!;
  const concat = combo.concat.toUpperCase();

  // Reset on new combo
  useEffect(() => {
    setPhase("cards");
    setSegIndex(0);
    setDismissed(false);
    wordsReportedRef.current = false;
  }, [combo]);

  // Drive animation phases
  useEffect(() => {
    if (dismissed) return;

    if (phase === "cards") {
      timerRef.current = setTimeout(() => setPhase("merged"), 1200);
    } else if (phase === "merged") {
      timerRef.current = setTimeout(() => setPhase("split"), 700);
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

  // When split phase is active, report new word positions after a brief delay for layout
  useEffect(() => {
    if (phase !== "split" || wordsReportedRef.current || !onNewWordsPositioned) return;

    const timer = setTimeout(() => {
      if (wordsReportedRef.current) return;
      wordsReportedRef.current = true;

      const wordRects: WordRect[] = [];
      wordSpanRefs.current.forEach((el, word) => {
        if (el) {
          wordRects.push({ word, rect: el.getBoundingClientRect() });
        }
      });
      if (wordRects.length > 0) {
        onNewWordsPositioned(wordRects);
      }
    }, 600); // Delay to let the split animate into place

    return () => clearTimeout(timer);
  }, [phase, segIndex, onNewWordsPositioned]);

  const handleDismiss = useCallback(() => {
    setDismissed(true);
    clearTimeout(timerRef.current);
    onDismiss();
  }, [onDismiss]);

  // Compute boundaries
  const cardBoundaries = computeBoundaries(cards);
  const segBoundaries = computeBoundaries(currentSeg);
  const newWordIndices = getNewWordRanges(currentSeg, cards);

  // Build a map: letter index → word string (for new words only)
  const newWordAtIndex = new Map<number, string>();
  {
    let pos = 0;
    for (const word of currentSeg) {
      let isOriginal = false;
      let cardPos = 0;
      for (const card of cards) {
        if (word === card && pos === cardPos) {
          isOriginal = true;
          break;
        }
        cardPos += card.length;
      }
      if (!isOriginal && word.length >= 4) {
        newWordAtIndex.set(pos, word);
      }
      pos += word.length;
    }
  }

  // All possible gap positions (between any two adjacent letters)
  // For each position i (after letter i, before letter i+1), determine gap width
  const gapWidths: number[] = [];
  for (let i = 0; i < concat.length - 1; i++) {
    const boundary = i + 1;
    if (phase === "cards" && cardBoundaries.has(boundary)) {
      gapWidths.push(10); // gap visible
    } else if (phase === "split" && segBoundaries.has(boundary)) {
      gapWidths.push(10); // gap visible
    } else {
      gapWidths.push(0); // no gap
    }
  }

  // Build groups of consecutive letters, grouping new-word letters together
  // Each group: { letters: {char, globalIdx}[], isNewWord: boolean, word?: string }
  type LetterGroup = {
    letters: { char: string; globalIdx: number }[];
    isNewWord: boolean;
    word?: string;
  };
  const groups: LetterGroup[] = [];
  {
    let i = 0;
    while (i < concat.length) {
      if (newWordAtIndex.has(i)) {
        const word = newWordAtIndex.get(i)!;
        const letters: { char: string; globalIdx: number }[] = [];
        for (let j = 0; j < word.length; j++) {
          letters.push({ char: concat[i + j]!, globalIdx: i + j });
        }
        groups.push({ letters, isNewWord: true, word });
        i += word.length;
      } else {
        groups.push({
          letters: [{ char: concat[i]!, globalIdx: i }],
          isNewWord: false,
        });
        i++;
      }
    }
  }

  // Clear refs for this render
  wordSpanRefs.current.clear();

  return (
    <div
      className="combo-reveal w-full cursor-pointer"
      onClick={handleDismiss}
    >
      <div className="flex flex-col items-center gap-1">
        {/* Animated letter display - stable DOM, gaps animate via CSS */}
        <div className="combo-reveal-words">
          {groups.map((group) => {
            if (group.isNewWord && phase === "split") {
              // Wrap the whole new word in a measurable span
              return (
                <span
                  key={`word-${group.letters[0]!.globalIdx}`}
                  ref={(el) => {
                    if (el && group.word) wordSpanRefs.current.set(group.word, el);
                  }}
                  className="combo-word-group inline-flex items-center"
                >
                  {group.letters.map(({ char, globalIdx }) => (
                    <span key={globalIdx} className="inline-flex items-center">
                      {globalIdx > 0 && (
                        <span
                          className="combo-gap"
                          style={{ width: `${gapWidths[globalIdx - 1]}px` }}
                        />
                      )}
                      <span className="combo-letter combo-letter-new">{char}</span>
                    </span>
                  ))}
                </span>
              );
            }
            // Regular letters (one per group when not a new word)
            return group.letters.map(({ char, globalIdx }) => (
              <span key={globalIdx} className="inline-flex items-center">
                {globalIdx > 0 && (
                  <span
                    className="combo-gap"
                    style={{ width: `${gapWidths[globalIdx - 1]}px` }}
                  />
                )}
                <span
                  className={
                    phase === "split" && newWordIndices.has(globalIdx)
                      ? "combo-letter combo-letter-new"
                      : "combo-letter"
                  }
                >
                  {char}
                </span>
              </span>
            ));
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
