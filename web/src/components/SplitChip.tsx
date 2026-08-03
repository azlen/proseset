import type { CSSProperties, ReactNode } from "react";
import { cn } from "@/lib/utils";

interface SplitChipProps {
  text: string;
  boundaries: ReadonlySet<number>;
  ariaLabel: string;
  className?: string;
  letterClassName?: (index: number) => string | undefined;
  renderLetterOverlay?: (character: string, index: number) => ReactNode;
}

/** Return the character positions where each word in a split ends. */
export function getWordBoundaries(words: string[]): Set<number> {
  const boundaries = new Set<number>();
  let position = 0;

  for (let i = 0; i < words.length - 1; i++) {
    position += words[i]!.length;
    boundaries.add(position);
  }

  return boundaries;
}

/**
 * A stable, letter-by-letter chip whose diagonal seams can move without
 * replacing the letters. Selection and reveal both use this same surface.
 */
export function SplitChip({
  text,
  boundaries,
  ariaLabel,
  className,
  letterClassName,
  renderLetterOverlay,
}: SplitChipProps) {
  const boundaryOrder = new Map(
    [...boundaries]
      .sort((left, right) => left - right)
      .map((position, index) => [position, index]),
  );

  return (
    <div className={cn("split-chip", className)} aria-label={ariaLabel}>
      {text.toUpperCase().split("").map((character, index) => (
        <span className="split-chip-unit" key={`${index}-${character}`} aria-hidden="true">
          {index > 0 && (
            <span
              className={cn(
                "split-chip-boundary",
                boundaries.has(index) && "split-chip-boundary-visible",
              )}
              style={{
                "--split-chip-boundary-order": boundaryOrder.get(index) ?? 0,
              } as CSSProperties}
            >
              <span className="split-chip-burst split-chip-burst-top" aria-hidden="true">
                <span className="split-chip-burst-line" />
                <span className="split-chip-burst-line" />
                <span className="split-chip-burst-line" />
              </span>
              <span className="split-chip-burst split-chip-burst-bottom" aria-hidden="true">
                <span className="split-chip-burst-line" />
                <span className="split-chip-burst-line" />
                <span className="split-chip-burst-line" />
              </span>
            </span>
          )}
          <span className={cn("split-chip-letter", letterClassName?.(index))}>
            {character}
            {renderLetterOverlay?.(character, index)}
          </span>
        </span>
      ))}
    </div>
  );
}
