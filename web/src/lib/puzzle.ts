export interface PuzzleData {
  date: string;
  cards: string[];
}

export type { ComboResult } from "./segmentation";
export { validateCombo, loadDictionary } from "./segmentation";

let puzzlesCache: Array<{ id: number; cards: string[] }> | null = null;

async function loadPuzzles() {
  if (puzzlesCache) return puzzlesCache;
  const res = await fetch("/puzzles.json");
  const data = await res.json() as { puzzles: Array<{ id: number; cards: string[] }> };
  puzzlesCache = data.puzzles;
  return puzzlesCache;
}

export async function fetchRandomPuzzle(): Promise<PuzzleData> {
  const puzzles = await loadPuzzles();
  const puzzle = puzzles[Math.floor(Math.random() * puzzles.length)]!;
  return { date: `puzzle-${puzzle.id}`, cards: puzzle.cards };
}
