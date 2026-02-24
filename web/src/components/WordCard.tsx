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
      style={selected ? undefined : { borderColor: "#000" }}
      className={cn(
        "relative min-w-0 py-4 rounded-lg border-2 font-semibold text-base uppercase tracking-wide transition-all duration-150 cursor-pointer text-center",
        "hover:scale-105 active:scale-95",
        selected
          ? "border-primary bg-primary text-primary-foreground shadow-md"
          : used
            ? "bg-card text-card-foreground hover:border-primary/50"
            : "border-dotted bg-card text-card-foreground hover:border-primary/50"
      )}
    >
      {selected && selectionIndex !== null && (
        <span className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-primary-foreground text-primary text-xs font-bold flex items-center justify-center">
          {selectionIndex + 1}
        </span>
      )}
      {displayWord}
    </button>
  );
}
