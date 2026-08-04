import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { cn } from "@/lib/utils";
import { getWordBoundaries, SplitChip } from "./SplitChip";

interface CardSlotsProps {
  selectedCards: string[];
  shake: boolean;
}

const CHIP_RESIZE_DURATION = 500;
const EMPTY_SLOT_WIDTH = 64;
const EMPTY_SLOT_GAP = 8;
const SLOT_FILL_DURATION = 520;
const SLOT_EMPTY_HANDOFF_DELAY = SLOT_FILL_DURATION + 20;

export function CardSlots({ selectedCards, shake }: CardSlotsProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const targetContentRef = useRef<HTMLDivElement>(null);
  const resizerRef = useRef<HTMLDivElement>(null);
  const resizeAnimationRef = useRef<Animation>();
  const transitionRestoreFrameRef = useRef<number>();
  const previousCardsRef = useRef(selectedCards);
  const entryDeadlinesRef = useRef(new Map<string, number>());
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
  const fillingSlotIndex = !isRemoving
    && selectedCards.length === previousCards.length + 1
    && previousCards.length < 2
    ? previousCards.length
    : undefined;
  const isFirstSlotFill = fillingSlotIndex === 0;
  const isSecondSlotFill = fillingSlotIndex === 1;
  const isSecondSlotEmpty = isRemoving
    && previousCards.length === 2
    && selectedCards.length === 1;
  const isFirstSlotEmpty = isRemoving
    && previousCards.length === 1
    && selectedCards.length === 0;
  const isReversingEntry = removedCards.some(
    (card) => (entryDeadlinesRef.current.get(card) ?? 0) > Date.now(),
  );
  const isReversingGenericEntry = isReversingEntry
    && !isSecondSlotEmpty
    && !isFirstSlotEmpty;
  const resizeDirection = selectedCards.length > previousCards.length
    ? "expanding"
    : selectedCards.length < previousCards.length
      ? "contracting"
      : undefined;

  useLayoutEffect(() => {
    clearTimeout(exitTimerRef.current);
    cancelAnimationFrame(transitionRestoreFrameRef.current ?? 0);
    transitionRestoreFrameRef.current = undefined;
    resizerRef.current?.style.removeProperty("transition");

    const entryDeadline = Date.now() + SLOT_FILL_DURATION;
    enteringCards.forEach((card) => entryDeadlinesRef.current.set(card, entryDeadline));
    removedCards.forEach((card) => entryDeadlinesRef.current.delete(card));

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
      const previousWidth = currentWidth ?? EMPTY_SLOT_WIDTH;
      const morphTargetWidth = isSecondSlotEmpty
        ? nextWidth + EMPTY_SLOT_WIDTH + EMPTY_SLOT_GAP
        : isFirstSlotEmpty
          ? EMPTY_SLOT_WIDTH
          : nextWidth;
      resizeAnimationRef.current?.cancel();
      resizeAnimationRef.current = undefined;

      /*
       * If a trailing card is removed before its entry has finished, only part
       * of its unit has made it inside the shell. Collapse that visible part,
       * not the unit's full intrinsic width. This keeps the outgoing seam on
       * the live right edge as the outer width reverses, so it cannot strand a
       * detached piece of the black surface.
       */
      if (
        isReversingGenericEntry
        && removalEdge === "trailing"
        && removedIndex >= 0
        && currentWidth !== undefined
      ) {
        const exitingUnit = renderedUnits?.[removedIndex];
        const fullExitWidth = exitingUnit?.offsetWidth ?? 0;
        const visibleExitWidth = Math.min(
          fullExitWidth,
          Math.max(0, currentWidth - nextWidth),
        );
        exitingUnit?.style.setProperty(
          "--selection-chip-exit-width",
          `${visibleExitWidth}px`,
        );
      }

      if (currentWidth !== undefined && removalEdge && !isSecondSlotEmpty) {
        const direction = removalEdge === "leading" ? 1 : -1;
        const anchorShift = Math.max(0, currentWidth - nextWidth) * direction / 2;
        resizerRef.current?.style.setProperty(
          "--selection-chip-anchor-shift",
          `${anchorShift}px`,
        );
      }
      chipWidthRef.current = nextWidth;
      if (resizerRef.current) {
        resizerRef.current.style.width = `${morphTargetWidth}px`;
        resizerRef.current.style.setProperty(
          "--selection-chip-previous-width",
          `${previousWidth}px`,
        );
        resizerRef.current.style.setProperty(
          "--selection-chip-target-width",
          `${nextWidth}px`,
        );

        const shouldMorphSlot = fillingSlotIndex !== undefined
          || isSecondSlotEmpty
          || isFirstSlotEmpty;
        if (
          shouldMorphSlot
          && !window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ) {
          // Keep the entire changing chip/slot group inside one measured
          // footprint. That lets a placeholder be absorbed or released
          // without the centered row jumping when its DOM handoff completes.
          const startWidth = fillingSlotIndex === undefined
            ? previousWidth
            : isFirstSlotFill
              ? EMPTY_SLOT_WIDTH
              : previousWidth + EMPTY_SLOT_WIDTH + EMPTY_SLOT_GAP;
          const animation = resizerRef.current.animate(
            [
              { width: `${startWidth}px` },
              { width: `${morphTargetWidth}px` },
            ],
            {
              duration: SLOT_FILL_DURATION,
              easing: "cubic-bezier(0.22, 0.9, 0.25, 1)",
              fill: "both",
            },
          );
          resizeAnimationRef.current = animation;
          void animation.finished.then(() => {
            if (resizeAnimationRef.current !== animation) return;
            animation.cancel();
            resizeAnimationRef.current = undefined;
          }).catch(() => undefined);
        }
      }
    }

    if (isRemoving) {
      const exitDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? 0
        : isSecondSlotEmpty || isFirstSlotEmpty
          ? SLOT_EMPTY_HANDOFF_DELAY
          : CHIP_RESIZE_DURATION;
      exitTimerRef.current = setTimeout(() => {
        if (
          (isSecondSlotEmpty || isFirstSlotEmpty)
          && nextWidth !== undefined
          && resizerRef.current
        ) {
          const resizer = resizerRef.current;
          // The special shell includes the soon-to-be external dashed slot.
          // Collapse it to the surviving chip atomically with mounting that
          // slot, then restore ordinary width transitions on the next frame.
          resizer.style.transition = "none";
          resizer.style.width = `${nextWidth}px`;
          void resizer.offsetWidth;
          cancelAnimationFrame(transitionRestoreFrameRef.current ?? 0);
          transitionRestoreFrameRef.current = requestAnimationFrame(() => {
            resizer.style.removeProperty("transition");
            transitionRestoreFrameRef.current = undefined;
          });
        }
        rerenderAfterExit((version) => version + 1);
      }, exitDelay);
    }

    return () => clearTimeout(exitTimerRef.current);
  }, [
    fillingSlotIndex,
    isFirstSlotEmpty,
    isFirstSlotFill,
    isRemoving,
    isReversingGenericEntry,
    isReversingEntry,
    isSecondSlotEmpty,
    removalEdge,
    renderedSelectionKey,
    selectionKey,
    selectedCards.length,
  ]);

  useEffect(() => {
    previousCardsRef.current = selectedCards;
  }, [selectionKey, selectedCards]);

  useEffect(() => () => {
    cancelAnimationFrame(transitionRestoreFrameRef.current ?? 0);
  }, []);
  const emptySlotCount = Math.max(0, 2 - selectedCards.length);
  const visibleEmptySlotCount = isSecondSlotEmpty
    ? 0
    : isFirstSlotEmpty
      ? 1
      : isRemoving && selectedCards.length === 0
        ? 0
        : emptySlotCount;
  const emptySlotStartIndex = isFirstSlotEmpty ? 1 : selectedCards.length;

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
            "mx-1",
            isRemoving && "selection-chip-resizer-removing",
            isReversingGenericEntry && "selection-chip-resizer-reversing-entry",
            removalEdge
              && !isSecondSlotEmpty
              && `selection-chip-resizer-removing-${removalEdge}`,
          )}
          ref={resizerRef}
          style={{ width: chipWidthRef.current }}
        >
          <div
            className={cn(
              "selection-chip-shell",
              resizeDirection && `selection-chip-shell-${resizeDirection}`,
              isFirstSlotFill && "selection-chip-shell-first-fill",
              isSecondSlotFill && "selection-chip-shell-second-fill",
              isFirstSlotEmpty && "selection-chip-shell-first-empty",
              isSecondSlotEmpty && "selection-chip-shell-second-empty",
              isReversingGenericEntry && "selection-chip-shell-reversing-entry",
            )}
            key={renderedSelectionKey}
            aria-hidden={selectedCards.length === 0 || undefined}
          >
            {isSecondSlotFill && (
              <span className="selection-chip-absorbed-slot" aria-hidden="true" />
            )}
            {isSecondSlotEmpty && (
              <span className="selection-chip-released-slot" aria-hidden="true" />
            )}
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

      {Array.from({ length: visibleEmptySlotCount }, (_, i) => (
        <div
          key={`empty-${emptySlotStartIndex + i}`}
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
