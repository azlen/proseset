import type { ReactNode } from "react";
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
            />
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
