import { useState, useEffect, useRef, useCallback } from "react";
import type { ComboResult } from "@/lib/puzzle";

interface ComboRevealProps {
  combo: ComboResult;
  cards: string[];
  previouslyFoundWords: Set<string>;
  onDismiss: () => void;
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

/** Classify each letter index as original card, already-found word, or new word */
function classifyLetters(
  seg: string[],
  cards: string[],
  previouslyFoundWords: Set<string>
): Map<number, WordClass> {
  const classification = new Map<number, WordClass>();
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

    let cls: WordClass;
    if (isOriginal || word.length < 4) {
      cls = "original";
    } else if (previouslyFoundWords.has(word)) {
      cls = "already-found";
    } else {
      cls = "new";
    }

    for (let i = segPos; i < segPos + word.length; i++) {
      classification.set(i, cls);
    }
    segPos += word.length;
  }
  return classification;
}

export function ComboReveal({ combo, cards, previouslyFoundWords, onDismiss }: ComboRevealProps) {
  // Fall back to segmentations if bestSegmentations is missing (e.g., from older saved progress)
  const segs = combo.bestSegmentations?.length ? combo.bestSegmentations : combo.segmentations;
  const [segIndex, setSegIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("cards");
  const [dismissed, setDismissed] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  const currentSeg = segs[segIndex] ?? segs[0]!;
  const concat = combo.concat.toUpperCase();

  // Reset on new combo
  useEffect(() => {
    setPhase("cards");
    setSegIndex(0);
    setDismissed(false);
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

  const handleDismiss = useCallback(() => {
    setDismissed(true);
    clearTimeout(timerRef.current);
    onDismiss();
  }, [onDismiss]);

  // Compute boundaries
  const cardBoundaries = computeBoundaries(cards);
  const segBoundaries = computeBoundaries(currentSeg);
  const letterClasses = classifyLetters(currentSeg, cards, previouslyFoundWords);

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

  return (
    <div
      className="combo-reveal w-full cursor-pointer"
      onClick={handleDismiss}
    >
      <div className="flex flex-col items-center gap-1">
        {/* Animated letter display - stable DOM, gaps animate via CSS */}
        <div className="combo-reveal-words">
          {concat.split("").map((char, i) => (
            <span key={i} className="inline-flex items-center">
              {i > 0 && (
                <span
                  className="combo-gap"
                  style={{ width: `${gapWidths[i - 1]}px` }}
                />
              )}
              <span
                className={
                  phase === "split"
                    ? `combo-letter ${
                        letterClasses.get(i) === "new"
                          ? "combo-letter-new"
                          : letterClasses.get(i) === "already-found"
                            ? "combo-letter-found"
                            : ""
                      }`
                    : "combo-letter"
                }
              >
                {char}
              </span>
            </span>
          ))}
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
