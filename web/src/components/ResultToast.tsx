import type { ComboResult } from "@/lib/puzzle";

interface ResultToastProps {
  combo: ComboResult;
  cards: string[];
  isNew: boolean;
  onDismiss: () => void;
}

export function ResultToast({ combo, cards, isNew, onDismiss }: ResultToastProps) {
  return (
    <div
      onClick={onDismiss}
      className="fixed inset-x-0 bottom-8 flex justify-center z-50 cursor-pointer animate-in fade-in slide-in-from-bottom-4 duration-300"
    >
      <div className="bg-card border border-border rounded-xl shadow-lg px-5 py-3 max-w-sm w-full mx-4">
        <div className="text-xs text-muted-foreground mb-1">
          {isNew ? "New combo!" : "Already found"}
        </div>
        <div className="font-semibold text-sm mb-1">
          {cards.join(" + ")}
        </div>
        <div className="space-y-0.5">
          {combo.segmentations.map((seg, i) => (
            <div key={i} className="font-mono text-sm text-muted-foreground">
              {seg.join(" | ")}
            </div>
          ))}
        </div>
        {combo.madeWords.length > 0 && isNew && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {combo.madeWords
              .sort((a, b) => b.length - a.length)
              .map((w) => (
                <span key={w} className="px-1.5 py-0.5 rounded bg-primary/10 text-primary text-xs font-medium">
                  {w}
                </span>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
