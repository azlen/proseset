import type { PuzzleData, ComboResult } from "./puzzle";

export interface GameState {
  puzzle: PuzzleData | null;
  selectedCards: string[];
  foundCombos: Map<string, ComboResult>;
  usedCards: Set<string>;
  foundMadeWords: string[];
  longestFoundWord: string;
  completed: boolean;
  showCompletion: boolean;
  lastResult: {
    combo: ComboResult;
    cards: string[];
    isNew: boolean;
    /** Words that had already been found before this combo was submitted */
    previouslyFoundWords: Set<string>;
  } | null;
  invalidSubmitCount: number;
  submitting: boolean;
}

export type GameAction =
  | { type: "LOAD_PUZZLE"; puzzle: PuzzleData }
  | { type: "SELECT_CARD"; card: string }
  | { type: "DESELECT_CARD"; card: string }
  | { type: "CLEAR_SELECTION" }
  | { type: "SUBMIT_START" }
  | { type: "SUBMIT_RESULT"; cards: string[]; result: ComboResult }
  | { type: "SUBMIT_INVALID" }
  | { type: "DISMISS_RESULT" }
  | { type: "DISMISS_COMPLETION" }
  | { type: "SHUFFLE_CARDS" }
  | { type: "RESTORE_PROGRESS"; combos: Array<{ key: string; result: ComboResult }> }
  | { type: "ADD_FOUND_WORD"; word: string };

export const initialState: GameState = {
  puzzle: null,
  selectedCards: [],
  foundCombos: new Map(),
  usedCards: new Set(),
  foundMadeWords: [],
  longestFoundWord: "",
  completed: false,
  showCompletion: false,
  lastResult: null,
  invalidSubmitCount: 0,
  submitting: false,
};

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "LOAD_PUZZLE":
      return { ...initialState, puzzle: action.puzzle };

    case "SELECT_CARD": {
      if (state.selectedCards.includes(action.card)) return state;
      return { ...state, selectedCards: [...state.selectedCards, action.card], invalidSubmitCount: 0 };
    }

    case "DESELECT_CARD": {
      const idx = state.selectedCards.indexOf(action.card);
      if (idx === -1) return state;
      return {
        ...state,
        selectedCards: state.selectedCards.filter((c) => c !== action.card),
        invalidSubmitCount: 0,
      };
    }

    case "CLEAR_SELECTION":
      return { ...state, selectedCards: [], invalidSubmitCount: 0 };

    case "SUBMIT_START":
      return { ...state, submitting: true, invalidSubmitCount: 0 };

    case "SUBMIT_INVALID":
      return {
        ...state,
        submitting: false,
        invalidSubmitCount: state.invalidSubmitCount + 1,
      };

    case "SUBMIT_RESULT": {
      const { cards, result } = action;
      const key = cards.join(",");
      const isNew = !state.foundCombos.has(key);
      const previouslyFoundWords = new Set(state.foundMadeWords);

      if (isNew) {
        const newFoundCombos = new Map(state.foundCombos);
        newFoundCombos.set(key, result);

        const newUsedCards = new Set(state.usedCards);
        for (const card of cards) {
          newUsedCards.add(card);
        }
        const completedNow =
          !state.completed &&
          state.puzzle !== null &&
          newUsedCards.size === state.puzzle.cards.length;

        // Don't add words to foundMadeWords yet — they'll be added
        // one-by-one via ADD_FOUND_WORD as each split is revealed.
        let longestFoundWord = state.longestFoundWord;
        for (const word of result.madeWords) {
          if (word.length > longestFoundWord.length) {
            longestFoundWord = word;
          }
        }

        return {
          ...state,
          foundCombos: newFoundCombos,
          usedCards: newUsedCards,
          longestFoundWord,
          completed: state.completed || completedNow,
          showCompletion: state.showCompletion || completedNow,
          selectedCards: [],
          lastResult: { combo: result, cards, isNew: true, previouslyFoundWords },
          invalidSubmitCount: 0,
          submitting: false,
        };
      }

      return {
        ...state,
        selectedCards: [],
        lastResult: { combo: result, cards, isNew: false, previouslyFoundWords },
        invalidSubmitCount: 0,
        submitting: false,
      };
    }

    case "ADD_FOUND_WORD": {
      const word = action.word;
      if (state.foundMadeWords.includes(word)) return state;
      return {
        ...state,
        foundMadeWords: [word, ...state.foundMadeWords],
      };
    }

    case "DISMISS_RESULT":
      return { ...state, lastResult: null };

    case "DISMISS_COMPLETION":
      return { ...state, showCompletion: false };

    case "SHUFFLE_CARDS": {
      if (!state.puzzle) return state;
      const shuffled = [...state.puzzle.cards];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
      }
      return { ...state, puzzle: { ...state.puzzle, cards: shuffled } };
    }

    case "RESTORE_PROGRESS": {
      const newFoundCombos = new Map(state.foundCombos);
      const newUsedCards = new Set(state.usedCards);
      const seenWords = new Set(state.foundMadeWords);
      const newFoundMadeWords = [...state.foundMadeWords];
      let longestFoundWord = state.longestFoundWord;

      for (const { key, result } of action.combos) {
        newFoundCombos.set(key, result);
        for (const card of key.split(",")) {
          newUsedCards.add(card);
        }
        for (const word of result.madeWords) {
          if (!seenWords.has(word)) {
            seenWords.add(word);
            newFoundMadeWords.push(word);
          }
          if (word.length > longestFoundWord.length) {
            longestFoundWord = word;
          }
        }
      }

      return {
        ...state,
        foundCombos: newFoundCombos,
        usedCards: newUsedCards,
        foundMadeWords: newFoundMadeWords,
        longestFoundWord,
      };
    }

    default:
      return state;
  }
}
