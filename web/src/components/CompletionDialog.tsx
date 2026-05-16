interface CompletionDialogProps {
  score: number;
  combosFound: number;
  wordsFound: number;
  longestWord: string;
  onDismiss: () => void;
}

export function CompletionDialog({
  score,
  combosFound,
  wordsFound,
  longestWord,
  onDismiss,
}: CompletionDialogProps) {
  return (
    <div
      className="completion-backdrop fixed inset-0 z-50 flex items-center justify-center bg-black/35 px-4 py-6"
      onClick={onDismiss}
    >
      <div
        className="completion-dialog w-full max-w-sm rounded-lg bg-card p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-5 text-center">
          <div className="text-xs font-bold uppercase tracking-[0.18em] text-muted-foreground">
            Complete
          </div>
          <h2 className="mt-2 text-3xl font-black tracking-tight">Puzzle solved</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            You used every card at least once.
          </p>
        </div>

        <div className="mb-2 rounded-lg bg-foreground px-4 py-4 text-center text-background">
          <div className="text-4xl font-black tabular-nums">{score}</div>
          <div className="text-xs font-bold uppercase tracking-wide opacity-75">
            Points
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-center">
          <div className="rounded-lg bg-muted px-3 py-3">
            <div className="text-2xl font-black tabular-nums">{combosFound}</div>
            <div className="text-xs font-semibold uppercase text-muted-foreground">
              Combos
            </div>
          </div>
          <div className="rounded-lg bg-muted px-3 py-3">
            <div className="text-2xl font-black tabular-nums">{wordsFound}</div>
            <div className="text-xs font-semibold uppercase text-muted-foreground">
              Words
            </div>
          </div>
          <div className="col-span-2 rounded-lg bg-muted px-3 py-3">
            <div className="truncate text-2xl font-black uppercase">
              {longestWord || "-"}
            </div>
            <div className="text-xs font-semibold uppercase text-muted-foreground">
              Longest
            </div>
          </div>
        </div>

        <button
          onClick={onDismiss}
          className="mt-5 w-full rounded-full border-2 border-foreground bg-foreground px-5 py-3 text-sm font-bold text-background transition-opacity hover:opacity-90"
        >
          Keep playing
        </button>
      </div>
    </div>
  );
}
