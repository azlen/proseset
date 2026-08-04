import { cn } from "@/lib/utils";

interface WordCardProps {
  word: string;
  selected: boolean;
  selectionIndex: number | null;
  onSelect: () => void;
  onDeselect: () => void;
}

export function WordCard({
  word,
  selected,
  selectionIndex,
  onSelect,
  onDeselect,
}: WordCardProps) {
  const displayWord = word;

  return (
    <div className={cn("relative h-[80px] min-w-0", selected && "z-10")}>
      <span
        aria-hidden="true"
        className={cn(
          "word-card-hatch-shadow word-card-ticket-shape pointer-events-none absolute inset-0 transition-opacity duration-150",
          selected ? "opacity-100" : "opacity-0",
        )}
      />
      <button
        onClick={selected ? onDeselect : onSelect}
        aria-pressed={selected}
        className={cn(
          "word-card-ticket-shape relative z-[1] flex h-[80px] w-full min-w-0 items-center justify-center border-[3px] border-solid border-foreground text-center text-base font-semibold uppercase tracking-wide text-card-foreground transition-[background-color,translate] duration-150 ease-out cursor-pointer",
          selected
            ? "-translate-x-1 -translate-y-1 bg-[#ddd1b2]"
            : "translate-x-0 translate-y-0 bg-card hover:bg-[#e9dfc2]",
        )}
      >
        {selected && selectionIndex !== null && (
          <span className="absolute top-2 left-2 w-5 h-5 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center">
            {selectionIndex + 1}
          </span>
        )}
        {displayWord}
      </button>
    </div>
  );
}
