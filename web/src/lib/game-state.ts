import type { PuzzleData, ComboResult } from "./puzzle";

export interface GameState {
  puzzle: PuzzleData | null;
  selectedCards: string[];
  foundCombos: Map<string, ComboResult>;
  usedCards: Set<string>;
  foundMadeWords: string[];
  longestFoundWord: string;
  lastResult: {
    combo: ComboResult;
    cards: string[];
    isNew: boolean;
  } | null;
  shake: boolean;
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
  | { type: "RESTORE_PROGRESS"; combos: Array<{ key: string; result: ComboResult }> };

export const initialState: GameState = {
  puzzle: null,
  selectedCards: [],
  foundCombos: new Map(),
  usedCards: new Set(),
  foundMadeWords: [],
  longestFoundWord: "",
  lastResult: null,
  shake: false,
  submitting: false,
};

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "LOAD_PUZZLE":
      return { ...initialState, puzzle: action.puzzle };

    case "SELECT_CARD": {
      if (state.selectedCards.includes(action.card)) return state;
      return { ...state, selectedCards: [...state.selectedCards, action.card], shake: false };
    }

    case "DESELECT_CARD": {
      const idx = state.selectedCards.indexOf(action.card);
      if (idx === -1) return state;
      return { ...state, selectedCards: state.selectedCards.slice(0, idx), shake: false };
    }

    case "CLEAR_SELECTION":
      return { ...state, selectedCards: [], shake: false };

    case "SUBMIT_START":
      return { ...state, submitting: true, shake: false };

    case "SUBMIT_INVALID":
      return { ...state, submitting: false, shake: true };

    case "SUBMIT_RESULT": {
      const { cards, result } = action;
      const key = cards.join(",");
      const isNew = !state.foundCombos.has(key);

      if (isNew) {
        const newFoundCombos = new Map(state.foundCombos);
        newFoundCombos.set(key, result);

        const newUsedCards = new Set(state.usedCards);
        for (const card of cards) {
          newUsedCards.add(card);
        }

        const existing = new Set(state.foundMadeWords);
        const newWords = result.madeWords.filter((w) => !existing.has(w));
        const newFoundMadeWords = [...newWords, ...state.foundMadeWords];

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
          foundMadeWords: newFoundMadeWords,
          longestFoundWord,
          selectedCards: [],
          lastResult: { combo: result, cards, isNew: true },
          shake: false,
          submitting: false,
        };
      }

      return {
        ...state,
        selectedCards: [],
        lastResult: { combo: result, cards, isNew: false },
        shake: false,
        submitting: false,
      };
    }

    case "DISMISS_RESULT":
      return { ...state, lastResult: null };

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
