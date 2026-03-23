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
  /** The "best" segmentations filtered to show the most interesting decompositions */
  bestSegmentations: string[][];
  madeWords: string[];
}

/** Get the set of "new" (non-original) words in a segmentation */
function getMadeWords(seg: string[], cards: string[]): Set<string> {
  const made = new Set<string>();
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
      made.add(word);
    }
    pos = end;
  }
  return made;
}

/**
 * Pick the most interesting segmentations:
 * - Score each by its longest unique 4+ letter word that no other segmentation has
 * - If tied, prefer fewer total words (less fragmented)
 * - Deduplicate: skip segmentations whose 4+ letter words are a subset of an already-picked one
 * - Always return at least one segmentation
 */
function pickBestSegmentations(altSegs: string[][], cards: string[]): string[][] {
  if (altSegs.length <= 1) return altSegs;

  // For each segmentation, find its made words (4+ letters)
  const segInfos = altSegs.map((seg) => {
    const madeWords = getMadeWords(seg, cards);
    const longWords = new Set([...madeWords].filter((w) => w.length >= 4));
    const longestWord = [...longWords].reduce((a, b) => (b.length > a.length ? b : a), "");
    return { seg, madeWords, longWords, longestWord };
  });

  // Count how many segmentations each 4+ word appears in
  const wordFreq = new Map<string, number>();
  for (const info of segInfos) {
    for (const w of info.longWords) {
      wordFreq.set(w, (wordFreq.get(w) ?? 0) + 1);
    }
  }

  // Score: prioritize segmentations that have a unique (freq=1) long word,
  // then by longest word length, then by fewest fragments
  function score(info: typeof segInfos[0]): number {
    let uniqueLongest = 0;
    for (const w of info.longWords) {
      if (wordFreq.get(w) === 1 && w.length > uniqueLongest) {
        uniqueLongest = w.length;
      }
    }
    // Primary: has unique 4+ letter word (bonus), length of that word
    // Secondary: longest word overall
    // Tertiary: fewer words = less fragmented
    return uniqueLongest * 1000 + info.longestWord.length * 10 + (20 - info.seg.length);
  }

  // Sort descending by score
  const sorted = [...segInfos].sort((a, b) => score(b) - score(a));

  // Greedily pick segmentations, skipping those whose long-word set is a subset of one already picked
  const picked: typeof segInfos = [];
  const coveredWords = new Set<string>();

  for (const info of sorted) {
    // Check if this segmentation adds any new 4+ letter word
    const newLongWords = [...info.longWords].filter((w) => !coveredWords.has(w));
    if (picked.length > 0 && newLongWords.length === 0) {
      continue; // skip, all its interesting words are already covered
    }
    picked.push(info);
    for (const w of info.longWords) {
      coveredWords.add(w);
    }
  }

  // Always return at least one
  if (picked.length === 0) {
    picked.push(sorted[0]!);
  }

  return picked.map((p) => p.seg);
}

export function validateCombo(cards: string[]): ComboResult {
  const concat = cards.join("");
  const originalBoundaries = computeBoundaries(cards);

  const allSegs = findSegmentations(concat);
  const altSegs = allSegs.filter((seg) => isAlternateSegmentation(seg, originalBoundaries));

  if (altSegs.length === 0) {
    return { valid: false, concat, segmentations: [], bestSegmentations: [], madeWords: [] };
  }

  const madeWordsSet = new Set<string>();
  for (const seg of altSegs) {
    const words = getMadeWords(seg, cards);
    for (const w of words) {
      madeWordsSet.add(w);
    }
  }

  const bestSegs = pickBestSegmentations(altSegs, cards);

  return {
    valid: true,
    concat,
    segmentations: altSegs,
    bestSegmentations: bestSegs,
    madeWords: Array.from(madeWordsSet).sort((a, b) => b.length - a.length),
  };
}
