import { useReducer, useEffect, useCallback } from "react";
import { fetchRandomPuzzle, validateCombo, loadDictionary } from "@/lib/puzzle";
import { gameReducer, initialState } from "@/lib/game-state";
import { saveProgress, loadProgress } from "@/lib/storage";
import { CardGrid } from "@/components/CardGrid";
import { ComboBar } from "@/components/ComboBar";
import { WordTicker } from "@/components/WordTicker";
import { ResultToast } from "@/components/ResultToast";
import "./index.css";

export function App() {
  const [state, dispatch] = useReducer(gameReducer, initialState);

  // Load dictionary + puzzle
  useEffect(() => {
    Promise.all([loadDictionary(), fetchRandomPuzzle()])
      .then(([, puzzle]) => {
        dispatch({ type: "LOAD_PUZZLE", puzzle });
      })
      .catch((err) => {
        console.error("Failed to load:", err);
      });
  }, []);

  // Save progress whenever foundCombos changes
  useEffect(() => {
    if (!state.puzzle) return;
    if (state.foundCombos.size === 0) return;
    saveProgress(state.puzzle.date, state.foundCombos);
  }, [state.foundCombos, state.puzzle]);

  // Auto-dismiss result toast
  useEffect(() => {
    if (!state.lastResult) return;
    const timer = setTimeout(() => {
      dispatch({ type: "DISMISS_RESULT" });
    }, 3000);
    return () => clearTimeout(timer);
  }, [state.lastResult]);

  const handleSelectCard = useCallback((card: string) => {
    dispatch({ type: "SELECT_CARD", card });
  }, []);

  const handleDeselectCard = useCallback((card: string) => {
    dispatch({ type: "DESELECT_CARD", card });
  }, []);

  const handleClear = useCallback(() => {
    dispatch({ type: "CLEAR_SELECTION" });
  }, []);

  const handleSubmit = useCallback(() => {
    if (state.selectedCards.length < 2) return;

    const key = state.selectedCards.join(",");
    if (state.foundCombos.has(key)) {
      const result = state.foundCombos.get(key)!;
      dispatch({ type: "SUBMIT_RESULT", cards: state.selectedCards, result });
      return;
    }

    const result = validateCombo(state.selectedCards);
    if (result.valid) {
      dispatch({ type: "SUBMIT_RESULT", cards: state.selectedCards, result });
    } else {
      dispatch({ type: "SUBMIT_INVALID" });
    }
  }, [state.selectedCards, state.foundCombos]);

  const handleDismissResult = useCallback(() => {
    dispatch({ type: "DISMISS_RESULT" });
  }, []);

  const handleRandomPuzzle = useCallback(async () => {
    const puzzle = await fetchRandomPuzzle();
    dispatch({ type: "LOAD_PUZZLE", puzzle });
  }, []);

  if (!state.puzzle) {
    return (
      <div className="max-w-lg w-full mx-auto p-8 text-center">
        <div className="text-muted-foreground">Loading puzzle...</div>
      </div>
    );
  }

  return (
    <div className="max-w-lg w-full mx-auto px-4 py-6 h-[100dvh] flex flex-col items-center overflow-hidden box-border">
      {/* Top: header + found words */}
      <div className="w-full flex justify-between items-baseline">
        <h1 className="text-xl font-bold tracking-tight">Proseset</h1>
        <button
          onClick={handleRandomPuzzle}
          className="text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
        >
          Random
        </button>
      </div>

      <div className="w-full mt-2">
        <WordTicker
          foundMadeWords={state.foundMadeWords}
          totalCards={state.puzzle.cards.length}
          usedCards={state.usedCards.size}
        />
      </div>

      {/* Middle: cards centered */}
      <div className="flex-1 flex items-center w-full">
        <CardGrid
          cards={state.puzzle.cards}
          selectedCards={state.selectedCards}
          usedCards={state.usedCards}
          onSelectCard={handleSelectCard}
          onDeselectCard={handleDeselectCard}
        />
      </div>

      {/* Bottom: submission area */}
      <div className="w-full pb-4">
        <ComboBar
          selectedCards={state.selectedCards}
          onClear={handleClear}
          onSubmit={handleSubmit}
          shake={state.shake}
          submitting={state.submitting}
        />
      </div>

      {state.lastResult && (
        <ResultToast
          combo={state.lastResult.combo}
          cards={state.lastResult.cards}
          isNew={state.lastResult.isNew}
          onDismiss={handleDismissResult}
        />
      )}
    </div>
  );
}

export default App;
