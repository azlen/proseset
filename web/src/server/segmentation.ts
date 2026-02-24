/**
 * Server-side segmentation engine for validating Proseset combos.
 * Ports the core logic from puzzle1.py's segment_word + puzzle3.py's boundary check.
 */

let dictionary: Set<string> = new Set();

export async function loadDictionary(path: string): Promise<void> {
  const file = Bun.file(path);
  const text = await file.text();
  dictionary = new Set(text.split("\n").filter(Boolean));
  console.log(`Dictionary loaded: ${dictionary.size} words`);
}

/**
 * Find all ways to segment a string into valid dictionary words.
 * Uses dynamic programming with memoization.
 */
function findSegmentations(str: string): string[][] {
  const memo = new Map<number, string[][]>();

  function dp(start: number): string[][] {
    if (start === str.length) return [[]];

    const cached = memo.get(start);
    if (cached) return cached;

    const results: string[][] = [];
    for (let end = start + 1; end <= str.length; end++) {
      const prefix = str.slice(start, end);
      if (dictionary.has(prefix)) {
        for (const tail of dp(end)) {
          results.push([prefix, ...tail]);
        }
      }
    }

    memo.set(start, results);
    return results;
  }

  return dp(0);
}

/**
 * Compute internal boundary positions for a sequence of words.
 * E.g., ["ant", "art", "her"] -> boundaries at positions 3, 6
 */
function computeBoundaries(words: string[]): Set<number> {
  const boundaries = new Set<number>();
  let pos = 0;
  for (let i = 0; i < words.length - 1; i++) {
    pos += words[i]!.length;
    boundaries.add(pos);
  }
  return boundaries;
}

/**
 * Check if a segmentation shares NO boundary positions with the original.
 */
function isAlternateSegmentation(
  seg: string[],
  originalBoundaries: Set<number>
): boolean {
  const segBoundaries = computeBoundaries(seg);
  for (const b of segBoundaries) {
    if (originalBoundaries.has(b)) return false;
  }
  return true;
}

export interface ValidationResult {
  valid: boolean;
  concat: string;
  segmentations: string[][];
  madeWords: string[];
}

/**
 * Validate a combo of cards: concatenate, find alternate segmentations,
 * extract made words.
 */
export function validateCombo(cards: string[]): ValidationResult {
  const concat = cards.join("");
  const originalBoundaries = computeBoundaries(cards);

  const allSegs = findSegmentations(concat);

  // Filter to alternate segmentations (no shared boundaries with original)
  const altSegs = allSegs.filter((seg) =>
    isAlternateSegmentation(seg, originalBoundaries)
  );

  if (altSegs.length === 0) {
    return { valid: false, concat, segmentations: [], madeWords: [] };
  }

  // Collect made words (words not at the same span as original cards)
  const madeWordsSet = new Set<string>();
  for (const seg of altSegs) {
    let pos = 0;
    for (const word of seg) {
      const start = pos;
      const end = pos + word.length;

      // Check if this word occupies the exact same span as an original card
      let isOriginal = false;
      let cardPos = 0;
      for (const card of cards) {
        if (word === card && start === cardPos && end === cardPos + card.length) {
          isOriginal = true;
          break;
        }
        cardPos += card.length;
      }

      if (!isOriginal) {
        madeWordsSet.add(word);
      }
      pos = end;
    }
  }

  return {
    valid: true,
    concat,
    segmentations: altSegs,
    madeWords: Array.from(madeWordsSet).sort((a, b) => b.length - a.length),
  };
}
