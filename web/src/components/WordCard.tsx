import { useEffect, useId, useRef, useState, type CSSProperties } from "react";
import { cn } from "@/lib/utils";

interface CardHole {
  id: string;
  x: number;
  y: number;
  radius: number;
}

interface PunchBurst {
  id: number;
  x: number;
  y: number;
}

const CARD_HOLE_RADIUS = 7;
const CARD_MARKER_RADIUS = 10;
// Tighter than the 14px opening diameter so adjacent punches merge into one hole.
const CARD_HOLE_SPACING = 11;
const CARD_HOLES_PER_ROW = 4;
const PUNCH_INITIAL_DELAY = 90;
const PUNCH_STAGGER = 140;
const PUNCH_BURST_DURATION = 520;
const PUNCH_PARTICLES = 8;

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
          rx="14.5"
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
  const [visibleUseCount, setVisibleUseCount] = useState(useCount);
  const holes = Array.from({ length: visibleUseCount }, (_, index) => getCardHole(index));
  const nextHole = getCardHole(useCount);
  const [punchBursts, setPunchBursts] = useState<PunchBurst[]>([]);
  const previousUseCountRef = useRef(useCount);
  const lastSelectionIndexRef = useRef<number | null>(selectionIndex);
  const nextBurstIdRef = useRef(0);
  const punchTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  if (selectionIndex !== null) {
    lastSelectionIndexRef.current = selectionIndex;
  }

  useEffect(() => {
    const previousUseCount = previousUseCountRef.current;
    previousUseCountRef.current = useCount;
    if (useCount === previousUseCount) return;

    if (useCount < previousUseCount) {
      setVisibleUseCount(useCount);
      setPunchBursts([]);
      return;
    }

    const selectionIndex = lastSelectionIndexRef.current;
    if (selectionIndex === null) {
      setVisibleUseCount(useCount);
      return;
    }

    const delay = PUNCH_INITIAL_DELAY + selectionIndex * PUNCH_STAGGER;
    const newBursts = Array.from(
      { length: useCount - previousUseCount },
      (_, offset): PunchBurst => {
        const hole = getCardHole(previousUseCount + offset);
        return {
          id: nextBurstIdRef.current++,
          x: hole.x,
          y: hole.y,
        };
      },
    );
    const burstIds = new Set(newBursts.map((burst) => burst.id));

    const punchTimer = setTimeout(() => {
      setVisibleUseCount((current) => Math.max(current, useCount));
      setPunchBursts((current) => [...current, ...newBursts]);
      punchTimersRef.current = punchTimersRef.current.filter(
        (candidate) => candidate !== punchTimer,
      );

      const cleanupTimer = setTimeout(() => {
        setPunchBursts((current) => current.filter(
          (candidate) => !burstIds.has(candidate.id),
        ));
        punchTimersRef.current = punchTimersRef.current.filter(
          (candidate) => candidate !== cleanupTimer,
        );
      }, PUNCH_BURST_DURATION);
      punchTimersRef.current.push(cleanupTimer);
    }, delay);
    punchTimersRef.current.push(punchTimer);
  }, [useCount]);

  useEffect(() => () => {
    for (const timer of punchTimersRef.current) clearTimeout(timer);
    punchTimersRef.current = [];
  }, []);

  return (
    <div
      className={cn(
        "relative h-[80px] min-w-0 rounded-[16px]",
        punchBursts.length > 0 && "z-10",
      )}
    >
      <button
        onClick={selected ? onDeselect : onSelect}
        aria-pressed={selected}
        aria-label={selected && selectionIndex !== null
          ? `${word}, selection ${selectionIndex + 1}`
          : word}
        className="group relative z-[1] flex h-[80px] w-full min-w-0 cursor-pointer items-center justify-center rounded-[16px] text-center text-base font-semibold uppercase tracking-wide text-card-foreground"
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

      {punchBursts.map((burst) => (
        <span
          key={burst.id}
          aria-hidden="true"
          className="word-card-punch-burst pointer-events-none absolute z-20"
          style={{
            left: burst.x,
            top: burst.y,
          }}
        >
          {Array.from({ length: PUNCH_PARTICLES }, (_, index) => (
            <span
              key={index}
              className="word-card-punch-particle"
              style={{ "--word-card-particle-index": index } as CSSProperties}
            />
          ))}
        </span>
      ))}
    </div>
  );
}
