let dictionary: Set<string> = new Set();
let loaded = false;

export async function loadDictionary(): Promise<void> {
  if (loaded) return;
  const res = await fetch("/dictionary.txt");
  const text = await res.text();
  dictionary = new Set(text.split("\n").filter(Boolean));
  loaded = true;
}

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

function computeBoundaries(words: string[]): Set<number> {
  const boundaries = new Set<number>();
  let pos = 0;
  for (let i = 0; i < words.length - 1; i++) {
    pos += words[i]!.length;
    boundaries.add(pos);
  }
  return boundaries;
}

function isAlternateSegmentation(seg: string[], originalBoundaries: Set<number>): boolean {
  const segBoundaries = computeBoundaries(seg);
  for (const b of segBoundaries) {
    if (originalBoundaries.has(b)) return false;
  }
  return true;
}

export interface ComboResult {
  valid: boolean;
  concat: string;
  segmentations: string[][];
  madeWords: string[];
}

export function validateCombo(cards: string[]): ComboResult {
  const concat = cards.join("");
  const originalBoundaries = computeBoundaries(cards);

  const allSegs = findSegmentations(concat);
  const altSegs = allSegs.filter((seg) => isAlternateSegmentation(seg, originalBoundaries));

  if (altSegs.length === 0) {
    return { valid: false, concat, segmentations: [], madeWords: [] };
  }

  const madeWordsSet = new Set<string>();
  for (const seg of altSegs) {
    let pos = 0;
    for (const word of seg) {
      const start = pos;
      const end = pos + word.length;

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
