import { cn } from "@/lib/utils";
import { getWordBoundaries, SplitChip } from "./SplitChip";

interface CardSlotsProps {
  selectedCards: string[];
  shake: boolean;
}

export function CardSlots({ selectedCards, shake }: CardSlotsProps) {
  const emptySlotCount = Math.max(0, 2 - selectedCards.length);

  return (
    <div
      className={cn(
        "flex justify-center items-center min-h-12 transition-all",
        shake && "animate-shake",
      )}
    >
      {selectedCards.length > 0 && (
        <SplitChip
          text={selectedCards.join("")}
          boundaries={getWordBoundaries(selectedCards)}
          ariaLabel={`Selected words: ${selectedCards.join(", ")}`}
          className={emptySlotCount > 0 ? "mx-1" : undefined}
        />
      )}
      {Array.from({ length: emptySlotCount }, (_, i) => (
        <div
          key={`empty-${i}`}
          aria-hidden="true"
          className="mx-1 px-6 py-2 rounded-lg border-2 border-dashed border-border/40 min-h-10 min-w-16"
        />
      ))}
    </div>
  );
}

interface ActionButtonsProps {
  selectedCards: string[];
  onClear: () => void;
  onShuffle: () => void;
  onSubmit: () => void;
  submitting?: boolean;
}

export function ActionButtons({ selectedCards, onClear, onShuffle, onSubmit, submitting }: ActionButtonsProps) {
  const canSubmit = selectedCards.length >= 2;

  return (
    <div className="flex gap-4 justify-center items-center">
      <button
        onClick={onClear}
        disabled={selectedCards.length === 0}
        className="px-6 py-2.5 rounded-full border-2 border-border text-sm font-medium disabled:opacity-30 hover:bg-muted transition-colors cursor-pointer disabled:cursor-default"
      >
        Clear
      </button>
      <button
        onClick={onShuffle}
        className="w-11 h-11 rounded-full border-2 border-border flex items-center justify-center hover:bg-muted transition-colors cursor-pointer"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21.5 2v6h-6" />
          <path d="M2.5 22v-6h6" />
          <path d="M2.5 11.5a10 10 0 0 1 18.8-4.3" />
          <path d="M21.5 12.5a10 10 0 0 1-18.8 4.2" />
        </svg>
      </button>
      <button
        onClick={onSubmit}
        disabled={!canSubmit || submitting}
        className="px-6 py-2.5 rounded-full border-2 border-border text-sm font-medium disabled:opacity-30 hover:bg-muted transition-colors cursor-pointer disabled:cursor-default"
      >
        {submitting ? "..." : "Enter"}
      </button>
    </div>
  );
}
