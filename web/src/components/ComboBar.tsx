import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { cn } from "@/lib/utils";
import { getWordBoundaries, SplitChip } from "./SplitChip";

interface CardSlotsProps {
  selectedCards: string[];
  shake: boolean;
}

const CHIP_RESIZE_DURATION = 500;

export function CardSlots({ selectedCards, shake }: CardSlotsProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const targetContentRef = useRef<HTMLDivElement>(null);
  const resizerRef = useRef<HTMLDivElement>(null);
  const previousCardsRef = useRef(selectedCards);
  const exitTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const chipWidthRef = useRef<number>();
  const segmentWidthsRef = useRef(new Map<string, number>());
  const [, rerenderAfterExit] = useState(0);
  const previousCards = previousCardsRef.current;
  const removedCards = previousCards.filter((card) => !selectedCards.includes(card));
  const removedCardSet = new Set(removedCards);
  const isRemoving = removedCards.length > 0;
  const removedIndex = removedCards.length === 1
    ? previousCards.indexOf(removedCards[0]!)
    : -1;
  const removalEdge = selectedCards.length > 0 && removedIndex === 0
    ? "leading"
    : selectedCards.length > 0 && removedIndex === previousCards.length - 1
      ? "trailing"
      : undefined;
  // Keep every old segment mounted until its precise slot has collapsed.
  const renderedCards = isRemoving ? previousCards : selectedCards;
  const selectionKey = selectedCards.join("\u0000");
  const renderedSelectionKey = renderedCards.join("\u0000");
  // Separators live on the segment to their right. Only a separator becoming
  // an outside edge should disappear; separators around an interior removal
  // remain full-size and simply overlap as the empty word slot closes.
  const edgeBoundaryExitIndex = removalEdge === "leading"
    ? 1
    : removalEdge === "trailing"
      ? removedIndex
      : -1;
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

    const renderedUnits = contentRef.current?.querySelectorAll<HTMLElement>(".split-chip-unit");
    renderedUnits?.forEach((unit, index) => {
      const card = renderedCards[index];
      // offsetWidth intentionally ignores the word's scale/pop transform.
      if (card) segmentWidthsRef.current.set(card, unit.offsetWidth);
    });

    const nextWidth = selectedCards.length === 0
      ? 0
      : targetContentRef.current?.getBoundingClientRect().width;
    if (nextWidth !== undefined) {
      const currentWidth = resizerRef.current?.getBoundingClientRect().width;
      if (currentWidth !== undefined && removalEdge) {
        const direction = removalEdge === "leading" ? 1 : -1;
        const anchorShift = Math.max(0, currentWidth - nextWidth) * direction / 2;
        resizerRef.current?.style.setProperty(
          "--selection-chip-anchor-shift",
          `${anchorShift}px`,
        );
      }
      chipWidthRef.current = nextWidth;
      if (resizerRef.current) resizerRef.current.style.width = `${nextWidth}px`;
    }

    if (isRemoving) {
      const exitDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? 0
        : CHIP_RESIZE_DURATION;
      exitTimerRef.current = setTimeout(() => {
        rerenderAfterExit((version) => version + 1);
      }, exitDelay);
    }

    return () => clearTimeout(exitTimerRef.current);
  }, [isRemoving, removalEdge, renderedSelectionKey, selectionKey, selectedCards.length]);

  useEffect(() => {
    previousCardsRef.current = selectedCards;
  }, [selectionKey, selectedCards]);
  const emptySlotCount = Math.max(0, 2 - selectedCards.length);

  return (
    <div
      className={cn(
        "flex justify-center items-center min-h-12 transition-all",
        shake && "animate-shake",
      )}
    >
      {renderedCards.length > 0 && (
        <div
          className={cn(
            "selection-chip-resizer",
            emptySlotCount > 0 && "mx-1",
            isRemoving && "selection-chip-resizer-removing",
            removalEdge && `selection-chip-resizer-removing-${removalEdge}`,
          )}
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
                segmentClassName={(segment) => removedCardSet.has(segment)
                  ? "selection-chip-word-exiting"
                  : enteringCards.has(segment)
                    ? "selection-chip-word-entering"
                    : undefined}
                segmentUnitClassName={(segment, index) => cn(
                  removedCardSet.has(segment) && "selection-chip-unit-exiting",
                  index === edgeBoundaryExitIndex
                    && cn(
                      "selection-chip-edge-boundary-exiting",
                      removalEdge && `selection-chip-edge-boundary-exiting-${removalEdge}`,
                    ),
                )}
                segmentUnitStyle={(segment) => {
                  const width = segmentWidthsRef.current.get(segment);
                  if (!removedCardSet.has(segment) || width === undefined) return undefined;
                  return { "--selection-chip-exit-width": `${width}px` } as CSSProperties;
                }}
                className="selection-chip-chip"
                ariaLabel={`Selected words: ${selectedCards.join(", ")}`}
              />
            </div>
          </div>
        </div>
      )}

      {Array.from({
        length: renderedCards.length > 0 && selectedCards.length === 0
          ? 0
          : emptySlotCount,
      }, (_, i) => (
        <div
          key={`empty-${i}`}
          aria-hidden="true"
          className="mx-1 px-6 py-2 rounded-lg border-2 border-dashed border-border/40 min-h-10 min-w-16"
        />
      ))}

      {selectedCards.length > 0 && (
        <div className="selection-chip-target-sizer" ref={targetContentRef} aria-hidden="true">
          <SplitChip
            text={selectedCards.join("")}
            boundaries={getWordBoundaries(selectedCards)}
            segments={selectedCards}
            className="selection-chip-chip"
            ariaLabel=""
          />
        </div>
      )}
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
