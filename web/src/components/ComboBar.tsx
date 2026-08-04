import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { getWordBoundaries, SplitChip } from "./SplitChip";

interface CardSlotsProps {
  selectedCards: string[];
  shake: boolean;
}

const CHIP_RESIZE_DURATION = 420;

export function CardSlots({ selectedCards, shake }: CardSlotsProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const resizerRef = useRef<HTMLDivElement>(null);
  const previousCardsRef = useRef(selectedCards);
  const exitTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const chipWidthRef = useRef<number>();
  const completedExitKeyRef = useRef<string>();
  const [, rerenderAfterExit] = useState(0);
  const previousCards = previousCardsRef.current;
  const isClearing = selectedCards.length === 0 && previousCards.length > 0;
  const renderedCards = isClearing ? previousCards : selectedCards;
  const selectionKey = selectedCards.join("\u0000");
  const renderedSelectionKey = renderedCards.join("\u0000");
  const exitComplete = selectedCards.length === 0
    && renderedSelectionKey === completedExitKeyRef.current;
  const enteringCards = new Set(
    selectedCards.filter((card) => !previousCards.includes(card)),
  );
  const resizeDirection = selectedCards.length > previousCards.length
    ? "expanding"
    : selectedCards.length < previousCards.length
      ? "contracting"
      : undefined;

  useLayoutEffect(() => {
    clearTimeout(exitTimerRef.current);

    if (selectedCards.length === 0) {
      if (previousCards.length > 0) {
        chipWidthRef.current = 0;
        if (resizerRef.current) resizerRef.current.style.width = "0px";
        const exitDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? 0
          : CHIP_RESIZE_DURATION;
        exitTimerRef.current = setTimeout(() => {
          completedExitKeyRef.current = renderedSelectionKey;
          rerenderAfterExit((version) => version + 1);
        }, exitDelay);
      } else {
        chipWidthRef.current = undefined;
      }

      return () => clearTimeout(exitTimerRef.current);
    }

    completedExitKeyRef.current = undefined;
    const nextWidth = contentRef.current?.getBoundingClientRect().width;
    if (nextWidth !== undefined) {
      chipWidthRef.current = nextWidth;
      if (resizerRef.current) resizerRef.current.style.width = `${nextWidth}px`;
    }

    return () => clearTimeout(exitTimerRef.current);
  }, [selectionKey, selectedCards.length, previousCards.length]);

  useEffect(() => {
    previousCardsRef.current = selectedCards;
  }, [selectionKey, selectedCards]);

  return (
    <div
      className={cn(
        "flex justify-center items-center min-h-12 transition-all",
        shake && "animate-shake",
      )}
    >
      {renderedCards.length > 0 && (!exitComplete || selectedCards.length > 0)
        ? (
            <div
              className="selection-chip-resizer"
              ref={resizerRef}
              style={{ width: chipWidthRef.current }}
            >
              <div
                className={cn(
                  "selection-chip-shell",
                  resizeDirection && `selection-chip-shell-${resizeDirection}`,
                )}
                key={renderedSelectionKey}
                aria-hidden={selectedCards.length === 0 || undefined}
              >
                <div className="selection-chip-content" ref={contentRef}>
                  <SplitChip
                    text={renderedCards.join("")}
                    boundaries={getWordBoundaries(renderedCards)}
                    segments={renderedCards}
                    segmentClassName={(segment) => enteringCards.has(segment)
                      ? "selection-chip-word-entering"
                      : undefined}
                    className="selection-chip-chip"
                    ariaLabel={`Selected words: ${renderedCards.join(", ")}`}
                  />
                </div>
              </div>
            </div>
          )
        : Array.from({ length: 2 }, (_, i) => (
            <div
              key={`empty-${i}`}
              className="mx-1 px-6 py-2 rounded-lg border-2 border-dashed border-border/40 min-h-10 min-w-16"
            />
          ))}
    </div>
  );
}

interface ActionButtonsProps {
  selectedCards: string[];
  onClear: () => void;
  onShuffle: () => void;
  onSubmit: () => void;
  submitting?: boolean;
}

export function ActionButtons({ selectedCards, onClear, onShuffle, onSubmit, submitting }: ActionButtonsProps) {
  const canSubmit = selectedCards.length >= 2;

  return (
    <div className="flex gap-4 justify-center items-center">
      <button
        onClick={onClear}
        disabled={selectedCards.length === 0}
        className="px-6 py-2.5 rounded-full border-2 border-border text-sm font-medium disabled:opacity-30 hover:bg-muted transition-colors cursor-pointer disabled:cursor-default"
      >
        Clear
      </button>
      <button
        onClick={onShuffle}
        className="w-11 h-11 rounded-full border-2 border-border flex items-center justify-center hover:bg-muted transition-colors cursor-pointer"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21.5 2v6h-6" />
          <path d="M2.5 22v-6h6" />
          <path d="M2.5 11.5a10 10 0 0 1 18.8-4.3" />
          <path d="M21.5 12.5a10 10 0 0 1-18.8 4.2" />
        </svg>
      </button>
      <button
        onClick={onSubmit}
        disabled={!canSubmit || submitting}
        className="px-6 py-2.5 rounded-full border-2 border-border text-sm font-medium disabled:opacity-30 hover:bg-muted transition-colors cursor-pointer disabled:cursor-default"
      >
        {submitting ? "..." : "Enter"}
      </button>
    </div>
  );
}
