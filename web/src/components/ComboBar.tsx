import { cn } from "@/lib/utils";

interface ComboBarProps {
  selectedCards: string[];
  onClear: () => void;
  onSubmit: () => void;
  shake: boolean;
  submitting?: boolean;
}

export function ComboBar({ selectedCards, onClear, onSubmit, shake, submitting }: ComboBarProps) {
  const concat = selectedCards.join("");
  const canSubmit = selectedCards.length >= 2;

  return (
    <div className="w-full space-y-3">
      <div
        className={cn(
          "min-h-10 px-4 py-2 rounded-lg bg-muted text-center font-mono text-lg tracking-wide transition-all",
          shake && "animate-shake",
          selectedCards.length === 0 && "text-muted-foreground"
        )}
      >
        {selectedCards.length > 0 ? concat : "Select 2+ cards..."}
      </div>
      <div className="flex gap-3 justify-center">
        <button
          onClick={onClear}
          disabled={selectedCards.length === 0}
          className="px-5 py-2 rounded-lg border border-border text-sm font-medium disabled:opacity-30 hover:bg-muted transition-colors cursor-pointer disabled:cursor-default"
        >
          Clear
        </button>
        <button
          onClick={onSubmit}
          disabled={!canSubmit || submitting}
          className="px-5 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-30 hover:opacity-90 transition-colors cursor-pointer disabled:cursor-default"
        >
          {submitting ? "..." : "Submit"}
        </button>
      </div>
    </div>
  );
}
