import { useReducer, useEffect, useCallback, useState, type CSSProperties } from "react";
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

const DEFAULT_HATCH_LINE_WIDTH = 1.5;
const DEFAULT_HATCH_GAP_WIDTH = 1.5;
const DEFAULT_HATCH_ANGLE = 45;
const DEFAULT_HATCH_LINE_COLOR = "#000000";

interface HatchControlsProps {
  lineWidth: number;
  gapWidth: number;
  angle: number;
  lineColor: string;
  onLineWidthChange: (value: number) => void;
  onGapWidthChange: (value: number) => void;
  onAngleChange: (value: number) => void;
  onLineColorChange: (value: string) => void;
  onReset: () => void;
}

function HatchControls({
  lineWidth,
  gapWidth,
  angle,
  lineColor,
  onLineWidthChange,
  onGapWidthChange,
  onAngleChange,
  onLineColorChange,
  onReset,
}: HatchControlsProps) {
  return (
    <details
      open
      className="fixed top-3 left-3 z-[100] w-56 rounded-lg border-2 border-foreground bg-[#faf8f2]/95 shadow-lg backdrop-blur-sm"
    >
      <summary className="cursor-pointer px-3 py-2 text-xs font-black uppercase tracking-wide">
        Hatch tuning
      </summary>
      <div className="border-t border-foreground/20 px-3 pt-2 pb-3">
        <label className="block text-xs font-semibold">
          <span className="flex justify-between gap-3">
            <span>Line thickness</span>
            <output className="font-mono tabular-nums">{lineWidth}px</output>
          </span>
          <input
            type="range"
            min="0.5"
            max="6"
            step="0.5"
            value={lineWidth}
            onChange={(event) => onLineWidthChange(Number(event.target.value))}
            className="mt-1 w-full cursor-pointer accent-black"
          />
        </label>

        <label className="mt-2 block text-xs font-semibold">
          <span className="flex justify-between gap-3">
            <span>Gap / frequency</span>
            <output className="font-mono tabular-nums">{gapWidth}px</output>
          </span>
          <input
            type="range"
            min="1"
            max="16"
            step="0.5"
            value={gapWidth}
            onChange={(event) => onGapWidthChange(Number(event.target.value))}
            className="mt-1 w-full cursor-pointer accent-black"
          />
        </label>

        <label className="mt-2 block text-xs font-semibold">
          <span className="flex justify-between gap-3">
            <span>Rotation</span>
            <output className="font-mono tabular-nums">{Math.round(angle)}°</output>
          </span>
          <input
            type="range"
            min="0"
            max="360"
            step="any"
            value={angle}
            onChange={(event) => onAngleChange(Number(event.target.value))}
            className="mt-1 w-full cursor-pointer accent-black"
          />
        </label>

        <label className="mt-2 flex items-center justify-between gap-3 text-xs font-semibold">
          <span>Line color</span>
          <span className="flex items-center gap-2 font-mono uppercase">
            {lineColor}
            <input
              type="color"
              value={lineColor}
              onChange={(event) => onLineColorChange(event.target.value)}
              className="h-7 w-9 cursor-pointer rounded border border-foreground bg-transparent p-0.5"
            />
          </span>
        </label>

        <button
          type="button"
          onClick={onReset}
          className="mt-3 w-full cursor-pointer rounded border border-foreground px-2 py-1 text-xs font-bold hover:bg-muted"
        >
          Reset
        </button>
      </div>
    </details>
  );
}

export function GameApp({
  initialPuzzle,
  persistProgress = true,
  showRandom = true,
  compact = false,
}: GameAppProps) {
  const [state, dispatch] = useReducer(gameReducer, initialState);
  const [hatchLineWidth, setHatchLineWidth] = useState(DEFAULT_HATCH_LINE_WIDTH);
  const [hatchGapWidth, setHatchGapWidth] = useState(DEFAULT_HATCH_GAP_WIDTH);
  const [hatchAngle, setHatchAngle] = useState(DEFAULT_HATCH_ANGLE);
  const [hatchLineColor, setHatchLineColor] = useState(DEFAULT_HATCH_LINE_COLOR);

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
  const hatchStyle = {
    "--hatch-line-width": `${hatchLineWidth}px`,
    "--hatch-gap-width": `${hatchGapWidth}px`,
    "--hatch-angle": `${hatchAngle}deg`,
    "--hatch-line-color": hatchLineColor,
  } as CSSProperties;

  const resetHatch = () => {
    setHatchLineWidth(DEFAULT_HATCH_LINE_WIDTH);
    setHatchGapWidth(DEFAULT_HATCH_GAP_WIDTH);
    setHatchAngle(DEFAULT_HATCH_ANGLE);
    setHatchLineColor(DEFAULT_HATCH_LINE_COLOR);
  };

  return (
    <div
      style={hatchStyle}
      className={`${compact ? "max-w-md py-4" : "max-w-lg py-6"} w-full mx-auto px-4 h-[100dvh] flex flex-col items-center overflow-hidden box-border`}
    >
      {showRandom && (
        <HatchControls
          lineWidth={hatchLineWidth}
          gapWidth={hatchGapWidth}
          angle={hatchAngle}
          lineColor={hatchLineColor}
          onLineWidthChange={setHatchLineWidth}
          onGapWidthChange={setHatchGapWidth}
          onAngleChange={setHatchAngle}
          onLineColorChange={setHatchLineColor}
          onReset={resetHatch}
        />
      )}
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
