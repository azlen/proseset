export interface PuzzleData {
  date: string;
  cards: string[];
  totalWords: number;
  /** Lengths of all 4+ letter words, sorted descending */
  wordLengths: number[];
}

export type { ComboResult } from "./segmentation";
export { validateCombo, loadDictionary } from "./segmentation";

interface RawPuzzle {
  id: number;
  cards: string[];
  made_words: string[];
  num_made_words_4plus: number;
}

let puzzlesCache: RawPuzzle[] | null = null;

async function loadPuzzles() {
  if (puzzlesCache) return puzzlesCache;
  const res = await fetch("/newpuzzle.json");
  const data = await res.json() as { puzzles: RawPuzzle[] };
  puzzlesCache = data.puzzles;
  return puzzlesCache;
}

export async function fetchRandomPuzzle(): Promise<PuzzleData> {
  const puzzles = await loadPuzzles();
  const puzzle = puzzles[Math.floor(Math.random() * puzzles.length)]!;
  const words4plus = puzzle.made_words.filter((w) => w.length >= 4);
  const wordLengths = words4plus.map((w) => w.length).sort((a, b) => a - b);
  return { date: `puzzle-${puzzle.id}`, cards: puzzle.cards, totalWords: words4plus.length, wordLengths };
}
