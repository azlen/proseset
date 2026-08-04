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
    <button
      onClick={selected ? onDeselect : onSelect}
      aria-pressed={selected}
      className={cn(
        "relative min-w-0 rounded-lg border-[3px] border-solid border-foreground py-4 text-center text-base font-semibold uppercase tracking-wide text-card-foreground transition-[background-color,box-shadow] duration-150 ease-out cursor-pointer",
        selected
          ? "z-10 bg-[#ddd1b2] shadow-[0_0_0_3px_var(--foreground)]"
          : "bg-card shadow-none hover:bg-[#e9dfc2]",
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
