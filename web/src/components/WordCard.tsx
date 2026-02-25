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
      style={selected ? { borderColor: "var(--primary)", backgroundColor: "#f0f0f0" } : { borderColor: "#000" }}
      className={cn(
        "relative min-w-0 py-4 rounded-lg border-2 font-semibold text-base uppercase tracking-wide transition-colors duration-150 cursor-pointer text-center text-card-foreground",
        !used && "border-dotted",
        !selected && "bg-card hover:border-primary/50"
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
