import { cn } from "@/lib/utils";

interface WordCardProps {
  word: string;
  selected: boolean;
  selectionIndex: number | null;
  used: boolean;
  onSelect: () => void;
  onDeselect: () => void;
}

export function WordCard({
  word,
  selected,
  selectionIndex,
  used,
  onSelect,
  onDeselect,
}: WordCardProps) {
  const displayWord = word;

  return (
    <button
      onClick={selected ? onDeselect : onSelect}
      className={cn(
        "relative min-w-0 rounded-lg border-[3px] border-solid border-foreground py-4 text-center text-base font-semibold uppercase tracking-wide text-card-foreground transition-colors duration-150 cursor-pointer",
        selected
          ? "bg-[#ddd1b2]"
          : used
            ? "bg-[#e7dec6] hover:bg-[#e1d5b6]"
            : "bg-card hover:bg-[#e9dfc2]",
      )}
    >
      {selected && selectionIndex !== null && (
        <span className="absolute top-1 right-1 w-5 h-5 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center">
          {selectionIndex + 1}
        </span>
      )}
      {displayWord}
    </button>
  );
}
