import type { CSSProperties } from "react";
import { cn } from "@/lib/utils";

interface SplitChipProps {
  text: string;
  boundaries: ReadonlySet<number>;
  ariaLabel: string;
  className?: string;
  letterClassName?: (index: number) => string | undefined;
  segments?: readonly string[];
  segmentClassName?: (segment: string, index: number) => string | undefined;
  segmentUnitClassName?: (segment: string, index: number) => string | undefined;
  segmentUnitStyle?: (segment: string, index: number) => CSSProperties | undefined;
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
 * Shared chip surface. Reveal keeps stable letter nodes while selection can
 * group letters into word segments so a newly selected word pops as one unit.
 */
export function SplitChip({
  text,
  boundaries,
  ariaLabel,
  className,
  letterClassName,
  segments,
  segmentClassName,
  segmentUnitClassName,
  segmentUnitStyle,
}: SplitChipProps) {
  const boundaryOrder = new Map(
    [...boundaries]
      .sort((left, right) => left - right)
      .map((position, index) => [position, index]),
  );

  return (
    <div className={cn("split-chip", className)} aria-label={ariaLabel}>
      {segments
        ? segments.map((segment, index) => {
            const boundaryPosition = segments
              .slice(0, index)
              .reduce((position, word) => position + word.length, 0);

            return (
              <span
                className={cn("split-chip-unit", segmentUnitClassName?.(segment, index))}
                key={segment}
                style={segmentUnitStyle?.(segment, index)}
                aria-hidden="true"
              >
                {index > 0 && (
                  <SplitChipBoundary
                    order={boundaryOrder.get(boundaryPosition) ?? index - 1}
                    visible={boundaries.has(boundaryPosition)}
                  />
                )}
                <span className={cn("split-chip-segment", segmentClassName?.(segment, index))}>
                  {segment.toUpperCase()}
                </span>
              </span>
            );
          })
        : text.toUpperCase().split("").map((character, index) => (
            <span className="split-chip-unit" key={`${index}-${character}`} aria-hidden="true">
              {index > 0 && (
                <SplitChipBoundary
                  order={boundaryOrder.get(index) ?? 0}
                  visible={boundaries.has(index)}
                />
              )}
              <span className={cn("split-chip-letter", letterClassName?.(index))}>
                {character}
              </span>
            </span>
          ))}
    </div>
  );
}

function SplitChipBoundary({ visible, order }: { visible: boolean; order: number }) {
  return (
    <span
      className={cn(
        "split-chip-boundary",
        visible && "split-chip-boundary-visible",
      )}
      style={{ "--split-chip-boundary-order": order } as CSSProperties}
    >
      {(["top", "bottom"] as const).map((edge) => (
        <span
          className={cn("split-chip-pop", `split-chip-pop-${edge}`)}
          key={edge}
          aria-hidden="true"
        >
          {Array.from({ length: 3 }, (_, lineIndex) => (
            <span className="split-chip-pop-line" key={lineIndex} />
          ))}
        </span>
      ))}
    </span>
  );
}
