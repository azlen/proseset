import { useReducer, useEffect, useCallback, useState } from "react";
import { fetchRandomPuzzle, validateCombo, loadDictionary, type PuzzleData } from "../lib/puzzle";
import { gameReducer, initialState } from "../lib/game-state";
import { saveProgress } from "../lib/storage";
import { scoreWords } from "../lib/scoring";
import { CardGrid } from "./CardGrid";
import { CardSlots, ActionButtons } from "./ComboBar";
import { WordTicker } from "./WordTicker";
import { ComboReveal, getComboRevealDuration } from "./ComboReveal";
import { CompletionDialog } from "./CompletionDialog";

interface GameAppProps {
  initialPuzzle?: PuzzleData | null;
  persistProgress?: boolean;
  showRandom?: boolean;
  compact?: boolean;
}

interface TuningSliderProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}

function TuningSlider({ id, label, value, min, max, onChange }: TuningSliderProps) {
  return (
    <label htmlFor={id} className="min-w-0 text-xs font-medium">
      <span className="mb-1 flex items-center justify-between gap-1">
        <span>{label}</span>
        <output htmlFor={id} className="tabular-nums text-muted-foreground">{value}px</output>
      </span>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        className="block h-4 w-full cursor-pointer accent-primary"
      />
    </label>
  );
}

export function GameApp({
  initialPuzzle,
  persistProgress = true,
  showRandom = true,
  compact = false,
}: GameAppProps) {
  const [state, dispatch] = useReducer(gameReducer, initialState);
  const [cardHeight, setCardHeight] = useState(70);
  const [cardBorderWidth, setCardBorderWidth] = useState(3);
  const [cardBorderRadius, setCardBorderRadius] = useState(10);

  useEffect(() => {
    Promise.all([loadDictionary(), initialPuzzle ? Promise.resolve(initialPuzzle) : fetchRandomPuzzle()])
      .then(([, puzzle]) => {
        dispatch({ type: "LOAD_PUZZLE", puzzle });
      })
      .catch((err) => {
        console.error("Failed to load:", err);
      });
  }, [initialPuzzle]);

  useEffect(() => {
    if (!persistProgress) return;
    if (!state.puzzle) return;
    if (state.foundCombos.size === 0) return;
    saveProgress(state.puzzle.date, state.foundCombos);
  }, [persistProgress, state.foundCombos, state.puzzle]);

  useEffect(() => {
    if (!state.lastResult) return;
    const result = state.lastResult;
    const best = result.combo.bestSegmentations;
    const segmentations = best?.length ? best : result.combo.segmentations;
    const duration = getComboRevealDuration(result.cards, segmentations) + 1000;
    const timer = setTimeout(() => {
      for (const word of result.combo.madeWords) {
        dispatch({ type: "ADD_FOUND_WORD", word });
      }
      dispatch({ type: "DISMISS_RESULT" });
    }, duration);
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

  const handleDismissCompletion = useCallback(() => {
    dispatch({ type: "DISMISS_COMPLETION" });
  }, []);

  const handleWordRevealed = useCallback((word: string) => {
    dispatch({ type: "ADD_FOUND_WORD", word });
  }, []);

  const handleShuffle = useCallback(() => {
    dispatch({ type: "SHUFFLE_CARDS" });
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

  const foundScoringWords = state.foundMadeWords.filter((word) => word.length >= 4);
  const score = scoreWords(foundScoringWords);

  return (
    <div className={`${compact ? "max-w-md py-4" : "max-w-lg py-6"} w-full mx-auto px-4 h-[100dvh] flex flex-col items-center overflow-hidden box-border`}>
      <div className="w-full flex justify-between items-baseline">
        <h1 className="text-xl font-bold tracking-tight">Split</h1>
        {showRandom && (
          <button
            onClick={handleRandomPuzzle}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
          >
            Random
          </button>
        )}
      </div>

      <div className="w-full mt-2">
        <WordTicker
          foundMadeWords={state.foundMadeWords}
          totalCards={state.puzzle.cards.length}
          totalWords={state.puzzle.totalWords}
          wordLengths={state.puzzle.wordLengths}
          usedCards={state.usedCards.size}
        />
      </div>

      <section aria-label="Temporary card tuning controls" className="mt-2 w-full rounded-lg border bg-popover px-3 py-2">
        <div className="mb-1.5 text-xs font-bold uppercase tracking-wide text-muted-foreground">
          Card tuning · temporary
        </div>
        <div className="grid grid-cols-3 gap-3">
          <TuningSlider id="card-height" label="Height" value={cardHeight} min={48} max={100} onChange={setCardHeight} />
          <TuningSlider id="card-border" label="Border" value={cardBorderWidth} min={1} max={8} onChange={setCardBorderWidth} />
          <TuningSlider id="card-radius" label="Radius" value={cardBorderRadius} min={0} max={24} onChange={setCardBorderRadius} />
        </div>
      </section>

      <div className="flex-1 flex flex-col items-center w-full">
        <div className="flex-1 flex items-center w-full">
          <div className="w-full">
            {state.lastResult
              ? (
                  <ComboReveal
                    combo={state.lastResult.combo}
                    cards={state.lastResult.cards}
                    previouslyFoundWords={state.lastResult.previouslyFoundWords}
                    onDismiss={handleDismissResult}
                    onWordRevealed={handleWordRevealed}
                  />
                )
              : (
                  <CardSlots
                    key={state.invalidSubmitCount}
                    selectedCards={state.selectedCards}
                    shake={state.invalidSubmitCount > 0}
                  />
                )}
          </div>
        </div>
        <CardGrid
          cards={state.puzzle.cards}
          selectedCards={state.selectedCards}
          cardHeight={cardHeight}
          cardBorderWidth={cardBorderWidth}
          cardBorderRadius={cardBorderRadius}
          onSelectCard={handleSelectCard}
          onDeselectCard={handleDeselectCard}
        />
        <div className="flex-1 w-full" />
        <ActionButtons
          selectedCards={state.selectedCards}
          onClear={handleClear}
          onShuffle={handleShuffle}
          onSubmit={handleSubmit}
          submitting={state.submitting}
        />
        <div className="pb-4" />
      </div>

      {state.showCompletion && !state.lastResult && (
        <CompletionDialog
          score={score}
          combosFound={state.foundCombos.size}
          wordsFound={foundScoringWords.length}
          longestWord={state.longestFoundWord}
          onDismiss={handleDismissCompletion}
        />
      )}
    </div>
  );
}
