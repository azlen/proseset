export interface PuzzleData {
  date: string;
  cards: string[];
  totalWords: number;
  /** Lengths of all 4+ letter words, sorted descending */
  wordLengths: number[];
}

export type { ComboResult } from "./segmentation";
export { validateCombo, loadDictionary } from "./segmentation";

export interface RawPuzzle {
  id: number;
  cards: string[];
  made_words: string[];
  num_made_words_4plus: number;
  anchor_word?: string;
  num_valid_combos?: number;
  longest_made_words?: string[];
  enabled_target_words?: string[];
}

let puzzlesCache: RawPuzzle[] | null = null;

export async function loadPuzzles() {
  if (puzzlesCache) return puzzlesCache;
  const res = await fetch("/puzzles/megapuzzle2-1000-20260430-154557.json");
  const data = await res.json() as { puzzles: RawPuzzle[] };
  puzzlesCache = data.puzzles;
  return puzzlesCache;
}

export function puzzleFromRaw(puzzle: RawPuzzle): PuzzleData {
  const words4plus = puzzle.made_words.filter((w) => w.length >= 4);
  const wordLengths = words4plus.map((w) => w.length).sort((a, b) => a - b);
  return {
    date: `puzzle-${puzzle.id}`,
    cards: puzzle.cards,
    totalWords: words4plus.length,
    wordLengths,
  };
}

export async function fetchRandomPuzzle(): Promise<PuzzleData> {
  const puzzles = await loadPuzzles();
  const puzzle = puzzles[Math.floor(Math.random() * puzzles.length)]!;
  return puzzleFromRaw(puzzle);
}
