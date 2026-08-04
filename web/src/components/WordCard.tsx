import { useId, type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface CardHole {
  id: string;
  x: number;
  y: number;
  radius: number;
  label?: ReactNode;
}

interface WordCardProps {
  word: string;
  selected: boolean;
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

      {holes.map((hole) => hole.label !== undefined && (
        <span
          key={hole.id}
          aria-hidden="true"
          className="word-card-hole-label pointer-events-none absolute z-[2] flex items-center justify-center text-xs font-bold"
          style={{
            left: hole.x - hole.radius,
            top: hole.y - hole.radius,
            width: hole.radius * 2,
            height: hole.radius * 2,
          }}
        >
          {hole.label}
        </span>
      ))}
    </>
  );
}

export function WordCard({
  word,
  selected,
  selectionIndex,
  onSelect,
  onDeselect,
}: WordCardProps) {
  const displayWord = word;
  const holes: CardHole[] = selected && selectionIndex !== null
    ? [{ id: "selection", x: 16, y: 16, radius: 10, label: selectionIndex + 1 }]
    : [];

  return (
    <div className={cn("relative h-[80px] min-w-0 rounded-[12px]", selected && "z-10")}>
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
        <span className="relative z-[1]">{displayWord}</span>
      </button>
    </div>
  );
}
