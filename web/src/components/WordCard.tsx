import { useEffect, useId, useRef, useState, type CSSProperties } from "react";
import { cn } from "@/lib/utils";

interface CardHole {
  id: string;
  x: number;
  y: number;
  radius: number;
}

interface FallingChad {
  id: number;
  label: number | null;
  x: number;
  y: number;
}

const CARD_HOLE_RADIUS = 7;
const CARD_MARKER_RADIUS = 10;
// Tighter than the 14px opening diameter so adjacent punches merge into one hole.
const CARD_HOLE_SPACING = 11;
const CARD_HOLES_PER_ROW = 4;
const CHAD_FALL_DURATION = 1400;

function getCardHole(index: number): CardHole {
  return {
    id: `use-${index}`,
    x: 16 + (index % CARD_HOLES_PER_ROW) * CARD_HOLE_SPACING,
    y: 16 + Math.floor(index / CARD_HOLES_PER_ROW) * CARD_HOLE_SPACING,
    radius: CARD_HOLE_RADIUS,
  };
}

interface WordCardProps {
  word: string;
  selected: boolean;
  useCount: number;
  selectionIndex: number | null;
  onSelect: () => void;
  onDeselect: () => void;
}

interface PerforatedCardSurfaceProps {
  holes: CardHole[];
  selected: boolean;
}

/**
 * Paint the card as a compound shape instead of placing opaque badges on it.
 *
 * The first mask removes the union of every hole from the face and its outer
 * border. The second mask paints only the outside edge of that same union, so
 * overlapping holes continue to read as one clean, punched-out shape.
 */
function PerforatedCardSurface({ holes, selected }: PerforatedCardSurfaceProps) {
  const reactId = useId().replace(/:/g, "");
  const faceMaskId = `card-face-mask-${reactId}`;
  const holeEdgeMaskId = `card-hole-edge-mask-${reactId}`;

  return (
    <>
      <svg
        aria-hidden="true"
        className="word-card-surface pointer-events-none absolute inset-0 h-full w-full overflow-visible"
      >
        {holes.length > 0 && (
          <defs>
            <mask
              id={faceMaskId}
              x="-10"
              y="-10"
              width="600"
              height="100"
              maskUnits="userSpaceOnUse"
              style={{ maskType: "luminance" }}
            >
              <rect x="-10" y="-10" width="600" height="100" fill="white" />
              {holes.map((hole) => (
                <circle
                  key={hole.id}
                  cx={hole.x}
                  cy={hole.y}
                  r={hole.radius}
                  fill="black"
                />
              ))}
            </mask>

            <mask
              id={holeEdgeMaskId}
              x="-10"
              y="-10"
              width="600"
              height="100"
              maskUnits="userSpaceOnUse"
              style={{ maskType: "luminance" }}
            >
              <rect x="-10" y="-10" width="600" height="100" fill="black" />
              <g fill="white" stroke="white" strokeWidth="6">
                {holes.map((hole) => (
                  <circle
                    key={hole.id}
                    cx={hole.x}
                    cy={hole.y}
                    r={hole.radius}
                  />
                ))}
              </g>
              <g fill="black">
                {holes.map((hole) => (
                  <circle
                    key={hole.id}
                    cx={hole.x}
                    cy={hole.y}
                    r={hole.radius}
                  />
                ))}
              </g>
            </mask>
          </defs>
        )}

        <rect
          x="1.5"
          y="1.5"
          width="calc(100% - 3px)"
          height="77"
          rx="10.5"
          className={cn("word-card-face", selected && "word-card-face-selected")}
          mask={holes.length > 0 ? `url(#${faceMaskId})` : undefined}
        />
        {holes.length > 0 && (
          <rect
            width="100%"
            height="100%"
            className="word-card-hole-edge"
            mask={`url(#${holeEdgeMaskId})`}
          />
        )}
      </svg>

    </>
  );
}

export function WordCard({
  word,
  selected,
  useCount,
  selectionIndex,
  onSelect,
  onDeselect,
}: WordCardProps) {
  const displayWord = word;
  const holes = Array.from({ length: useCount }, (_, index) => getCardHole(index));
  const nextHole = getCardHole(useCount);
  const [fallingChads, setFallingChads] = useState<FallingChad[]>([]);
  const previousUseCountRef = useRef(useCount);
  const lastSelectionIndexRef = useRef<number | null>(selectionIndex);
  const nextChadIdRef = useRef(0);
  const chadTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  if (selectionIndex !== null) {
    lastSelectionIndexRef.current = selectionIndex;
  }

  useEffect(() => {
    const previousUseCount = previousUseCountRef.current;
    previousUseCountRef.current = useCount;
    if (useCount <= previousUseCount) return;

    const newChads = Array.from(
      { length: useCount - previousUseCount },
      (_, offset): FallingChad => {
        const hole = getCardHole(previousUseCount + offset);
        return {
          id: nextChadIdRef.current++,
          label: lastSelectionIndexRef.current === null
            ? null
            : lastSelectionIndexRef.current + 1,
          x: hole.x,
          y: hole.y,
        };
      },
    );
    const chadIds = new Set(newChads.map((chad) => chad.id));
    setFallingChads((current) => [...current, ...newChads]);

    const timer = setTimeout(() => {
      setFallingChads((current) => current.filter((candidate) => !chadIds.has(candidate.id)));
      chadTimersRef.current = chadTimersRef.current.filter((candidate) => candidate !== timer);
    }, CHAD_FALL_DURATION);
    chadTimersRef.current.push(timer);
  }, [useCount]);

  useEffect(() => () => {
    for (const timer of chadTimersRef.current) clearTimeout(timer);
    chadTimersRef.current = [];
  }, []);

  return (
    <div
      className={cn(
        "relative h-[80px] min-w-0 rounded-[12px]",
        (selected || fallingChads.length > 0) && "z-10",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "word-card-hatch-shadow pointer-events-none absolute inset-0 rounded-[12px] transition-opacity duration-150",
          selected ? "opacity-100" : "opacity-0",
        )}
      />
      <button
        onClick={selected ? onDeselect : onSelect}
        aria-pressed={selected}
        aria-label={selected && selectionIndex !== null
          ? `${word}, selection ${selectionIndex + 1}`
          : word}
        className={cn(
          "group relative z-[1] flex h-[80px] w-full min-w-0 items-center justify-center rounded-[12px] text-center text-base font-semibold uppercase tracking-wide text-card-foreground transition-transform duration-150 ease-out cursor-pointer",
          selected
            ? "-translate-x-1 -translate-y-1"
            : "translate-x-0 translate-y-0",
        )}
      >
        <PerforatedCardSurface holes={holes} selected={selected} />
        {selected && selectionIndex !== null && (
          <span
            aria-hidden="true"
            className="absolute z-[2] flex h-5 w-5 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground"
            style={{
              left: nextHole.x - CARD_MARKER_RADIUS,
              top: nextHole.y - CARD_MARKER_RADIUS,
            }}
          >
            {selectionIndex + 1}
          </span>
        )}
        <span className="relative z-[1]">{displayWord}</span>
      </button>

      {fallingChads.map((chad) => (
        <span
          key={chad.id}
          aria-hidden="true"
          className="word-card-chad pointer-events-none absolute z-20 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground"
          style={{
            left: chad.x - CARD_MARKER_RADIUS,
            top: chad.y - CARD_MARKER_RADIUS,
            "--word-card-chad-drift": chad.label !== null && chad.label % 2 === 0
              ? "36px"
              : "-34px",
            "--word-card-chad-spin": chad.label !== null && chad.label % 2 === 0
              ? "720deg"
              : "-680deg",
          } as CSSProperties}
        >
          {chad.label}
        </span>
      ))}
    </div>
  );
}
