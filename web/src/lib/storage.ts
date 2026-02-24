import type { ComboResult } from "./puzzle";

const STORAGE_KEY = "proseset-progress";

interface SavedCombo {
  key: string;
  result: ComboResult;
}

interface SavedProgress {
  date: string;
  combos: SavedCombo[];
}

export function saveProgress(date: string, combos: Map<string, ComboResult>): void {
  const data: SavedProgress = {
    date,
    combos: Array.from(combos.entries()).map(([key, result]) => ({ key, result })),
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    // localStorage full or unavailable
  }
}

export function loadProgress(date: string): SavedCombo[] | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data: SavedProgress = JSON.parse(raw);
    if (data.date !== date) return null;
    return data.combos;
  } catch {
    return null;
  }
}
