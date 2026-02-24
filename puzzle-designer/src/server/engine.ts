/**
 * Proseset Puzzle Designer Engine
 *
 * Ports core logic from the Python puzzle scripts:
 * - Word segmentation (run15.py)
 * - Brute-force combo analysis (puzzle3.py)
 * - Target word tracking via decompositions (puzzle4.py / puzzle6.py)
 */

// ============================================================================
//                              TYPES
// ============================================================================

export interface Decomposition {
  leftFrag: string;
  interior: string[];
  rightFrag: string;
  score: number;
}

export interface GameData {
  deckWords: string[];
  segWords: string[];
  frequencies: Record<string, number>;
  decompositions: Record<string, Decomposition[]>;
  leftFragCards: Record<string, string[]>;
  rightFragCards: Record<string, string[]>;
}

export interface TargetStatus {
  word: string;
  decomp: Decomposition;
  missing: number;
  missingCards: string[]; // description of what's missing
  enabled: boolean;
}

export interface ComboResult {
  cards: string[];
  concat: string;
  segmentations: string[][];
}

export interface MadeWord {
  word: string;
  length: number;
  frequency: number;
  combos: { cards: string[]; segmentation: string[] }[];
}

export interface AnalysisResult {
  enabledTargets: TargetStatus[];
  nearTargets: TargetStatus[];   // 1-2 missing
  combos: ComboResult[];
  madeWords: MadeWord[];
}

export interface CandidateSuggestion {
  card: string;
  frequency: number;
  completions: string[];   // target words that would become fully enabled
  advances: number;        // target words moved closer
  score: number;
}

export interface BrowseEntry {
  word: string;
  length: number;
  frequency: number;
  decomps: {
    leftFrag: string;
    interior: string[];
    rightFrag: string;
    score: number;
    /** Resolved cards: the actual card words needed (fragments resolved to best card) */
    resolvedCards: string[];
    /** Which resolved cards are already in the deck */
    inDeck: boolean[];
    missing: number;
  }[];
  bestMissing: number; // fewest missing across all decomps
}

// ============================================================================
//                              ENGINE
// ============================================================================

export class PuzzleEngine {
  private segWordSet: Set<string>;
  private deckWordSet: Set<string>;
  private segCache: Map<string, string[][] | null> = new Map();
  private data: GameData;

  // Reverse index: card -> decomposition entries it participates in
  private reverseIndex: Map<string, { word: string; decompIdx: number }[]> = new Map();

  constructor(data: GameData) {
    this.data = data;
    this.segWordSet = new Set(data.segWords);
    this.deckWordSet = new Set(data.deckWords);

    // Build reverse index
    this.buildReverseIndex();
  }

  private buildReverseIndex() {
    for (const [word, decomps] of Object.entries(this.data.decompositions)) {
      for (let di = 0; di < decomps.length; di++) {
        const d = decomps[di];
        // Interior cards
        for (const card of d.interior) {
          if (!this.reverseIndex.has(card)) {
            this.reverseIndex.set(card, []);
          }
          this.reverseIndex.get(card)!.push({ word, decompIdx: di });
        }
        // Left fragment cards
        if (d.leftFrag) {
          const cards = this.data.leftFragCards[d.leftFrag] || [];
          for (const card of cards) {
            if (!this.reverseIndex.has(card)) {
              this.reverseIndex.set(card, []);
            }
            this.reverseIndex.get(card)!.push({ word, decompIdx: di });
          }
        }
        // Right fragment cards
        if (d.rightFrag) {
          const cards = this.data.rightFragCards[d.rightFrag] || [];
          for (const card of cards) {
            if (!this.reverseIndex.has(card)) {
              this.reverseIndex.set(card, []);
            }
            this.reverseIndex.get(card)!.push({ word, decompIdx: di });
          }
        }
      }
    }
  }

  // ==========================================================================
  //                          SEGMENTATION
  // ==========================================================================

  /**
   * Find all ways to segment a string into valid words.
   * Memoized for performance.
   */
  segmentWord(word: string): string[][] {
    if (this.segCache.has(word)) {
      return this.segCache.get(word) || [];
    }

    const results = this._segmentWordInner(word);
    this.segCache.set(word, results);
    return results;
  }

  private _segmentWordInner(word: string): string[][] {
    if (word === "") return [[]];

    const results: string[][] = [];
    for (let i = 1; i <= word.length; i++) {
      const prefix = word.slice(0, i);
      if (this.segWordSet.has(prefix)) {
        const suffix = word.slice(i);
        for (const tail of this.segmentWord(suffix)) {
          results.push([prefix, ...tail]);
        }
      }
    }
    return results;
  }

  // ==========================================================================
  //                    BRUTE-FORCE COMBO ANALYSIS
  // ==========================================================================

  /**
   * Compute boundary positions for a sequence of words.
   */
  private computeBoundaries(words: string[]): Set<number> {
    const boundaries = new Set<number>();
    let pos = 0;
    for (let i = 0; i < words.length - 1; i++) {
      pos += words[i].length;
      boundaries.add(pos);
    }
    return boundaries;
  }

  /**
   * Check if two sets have any common elements.
   */
  private setsIntersect(a: Set<number>, b: Set<number>): boolean {
    for (const v of a) {
      if (b.has(v)) return true;
    }
    return false;
  }

  /**
   * Enumerate all ordered 3-card permutations, find alternate segmentations.
   */
  bruteForceAnalysis(deck: string[]): { combos: ComboResult[]; madeWords: MadeWord[] } {
    const combos: ComboResult[] = [];
    const madeWordMap = new Map<string, { cards: string[]; segmentation: string[] }[]>();

    // All ordered 3-permutations
    for (let i = 0; i < deck.length; i++) {
      for (let j = 0; j < deck.length; j++) {
        if (j === i) continue;
        for (let k = 0; k < deck.length; k++) {
          if (k === i || k === j) continue;

          const cards = [deck[i], deck[j], deck[k]];
          const concat = cards.join("");
          const originalBoundaries = this.computeBoundaries(cards);

          const segs = this.segmentWord(concat);
          if (segs.length === 0) continue;

          // Filter to alternate segmentations (no shared boundaries)
          const altSegs = segs.filter(
            (seg) => !this.setsIntersect(this.computeBoundaries(seg), originalBoundaries)
          );
          if (altSegs.length === 0) continue;

          combos.push({ cards, concat, segmentations: altSegs });

          // Collect made words
          for (const seg of altSegs) {
            let pos = 0;
            for (const madeWord of seg) {
              const start = pos;
              const end = pos + madeWord.length;

              // Skip if same span as an original card
              let isOriginal = false;
              let cardPos = 0;
              for (const card of cards) {
                if (madeWord === card && start === cardPos && end === cardPos + card.length) {
                  isOriginal = true;
                  break;
                }
                cardPos += card.length;
              }

              if (!isOriginal) {
                if (!madeWordMap.has(madeWord)) {
                  madeWordMap.set(madeWord, []);
                }
                madeWordMap.get(madeWord)!.push({ cards, segmentation: seg });
              }

              pos = end;
            }
          }
        }
      }
    }

    // Convert made words map to sorted array
    const madeWords: MadeWord[] = [];
    for (const [word, wordCombos] of madeWordMap) {
      // Deduplicate by unique card combos
      const seen = new Set<string>();
      const uniqueCombos: { cards: string[]; segmentation: string[] }[] = [];
      for (const c of wordCombos) {
        const key = c.cards.join("+");
        if (!seen.has(key)) {
          seen.add(key);
          uniqueCombos.push(c);
        }
      }

      madeWords.push({
        word,
        length: word.length,
        frequency: this.data.frequencies[word] ?? 0,
        combos: uniqueCombos.slice(0, 3), // Keep top 3 examples
      });
    }

    // Sort by length descending, then frequency
    madeWords.sort((a, b) => b.length - a.length || b.frequency - a.frequency);

    return { combos, madeWords };
  }

  // ==========================================================================
  //                    TARGET WORD TRACKING
  // ==========================================================================

  /**
   * For each decomposition, compute how many cards are missing from the deck.
   */
  computeTargetStatuses(deck: string[]): TargetStatus[] {
    const deckSet = new Set(deck);
    const results: TargetStatus[] = [];
    const seen = new Set<string>(); // Only show best decomp per target word

    for (const [word, decomps] of Object.entries(this.data.decompositions)) {
      let bestStatus: TargetStatus | null = null;

      for (const decomp of decomps) {
        const missingCards: string[] = [];
        let missing = 0;

        // Check interior cards
        for (const card of decomp.interior) {
          if (!deckSet.has(card)) {
            missing++;
            missingCards.push(card);
          }
        }

        // Check left fragment
        if (decomp.leftFrag) {
          const fragCards = this.data.leftFragCards[decomp.leftFrag] || [];
          if (!fragCards.some((c) => deckSet.has(c))) {
            missing++;
            missingCards.push(`*${decomp.leftFrag} (need card ending with it)`);
          }
        }

        // Check right fragment
        if (decomp.rightFrag) {
          const fragCards = this.data.rightFragCards[decomp.rightFrag] || [];
          if (!fragCards.some((c) => deckSet.has(c))) {
            missing++;
            missingCards.push(`${decomp.rightFrag}* (need card starting with it)`);
          }
        }

        const status: TargetStatus = {
          word,
          decomp,
          missing,
          missingCards,
          enabled: missing === 0,
        };

        if (bestStatus === null || missing < bestStatus.missing) {
          bestStatus = status;
        }
      }

      if (bestStatus) {
        results.push(bestStatus);
      }
    }

    // Sort: enabled first, then by missing count, then by word length desc
    results.sort((a, b) => {
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1;
      if (a.missing !== b.missing) return a.missing - b.missing;
      return b.word.length - a.word.length;
    });

    return results;
  }

  // ==========================================================================
  //                    CANDIDATE SUGGESTIONS
  // ==========================================================================

  /**
   * Rank candidate cards by how many target words they'd complete or advance.
   * Inverted approach: find near-complete decomps, then map back to which cards help.
   */
  rankCandidates(deck: string[], limit: number = 20): CandidateSuggestion[] {
    const deckSet = new Set(deck);

    // card -> { word -> { contribution, isCompletion } }
    const cardScores = new Map<string, Map<string, { contribution: number; isCompletion: boolean }>>();

    // Iterate all decompositions, find near-complete ones (missing 1-3),
    // then find which cards could reduce their missing count
    for (const [word, decomps] of Object.entries(this.data.decompositions)) {
      for (const decomp of decomps) {
        // Compute missing and collect which cards are needed
        const neededCards: string[] = [];

        for (const c of decomp.interior) {
          if (!deckSet.has(c)) neededCards.push(c);
        }

        let leftFragSatisfied = true;
        let leftFragCandidates: string[] = [];
        if (decomp.leftFrag) {
          const fc = this.data.leftFragCards[decomp.leftFrag] || [];
          if (!fc.some((c) => deckSet.has(c))) {
            leftFragSatisfied = false;
            leftFragCandidates = fc;
          }
        }

        let rightFragSatisfied = true;
        let rightFragCandidates: string[] = [];
        if (decomp.rightFrag) {
          const fc = this.data.rightFragCards[decomp.rightFrag] || [];
          if (!fc.some((c) => deckSet.has(c))) {
            rightFragSatisfied = false;
            rightFragCandidates = fc;
          }
        }

        const totalMissing =
          neededCards.length +
          (leftFragSatisfied ? 0 : 1) +
          (rightFragSatisfied ? 0 : 1);

        if (totalMissing === 0 || totalMissing > 3) continue;

        const missingAfter = totalMissing - 1;
        let contribution: number;
        let isCompletion: boolean;
        if (missingAfter === 0) {
          contribution = decomp.score * 10;
          isCompletion = true;
        } else if (missingAfter === 1) {
          contribution = decomp.score * 2;
          isCompletion = false;
        } else {
          contribution = decomp.score * 0.3;
          isCompletion = false;
        }

        // Each needed interior card and each fragment candidate gets this contribution
        const helpfulCards = [
          ...neededCards,
          ...leftFragCandidates,
          ...rightFragCandidates,
        ];

        for (const card of helpfulCards) {
          if (deckSet.has(card)) continue;

          if (!cardScores.has(card)) {
            cardScores.set(card, new Map());
          }
          const wordMap = cardScores.get(card)!;
          const prev = wordMap.get(word);
          if (!prev || contribution > prev.contribution) {
            wordMap.set(word, { contribution, isCompletion });
          }
        }
      }
    }

    // Aggregate scores per card (only cards in deckWords)
    const candidates: CandidateSuggestion[] = [];
    for (const [card, wordMap] of cardScores) {
      if (!this.deckWordSet.has(card)) continue;
      let score = 0;
      const completions: string[] = [];
      let advances = 0;

      for (const [word, { contribution, isCompletion }] of wordMap) {
        score += contribution;
        if (isCompletion) completions.push(word);
        else advances++;
      }

      if (score <= 0) continue;

      const freq = this.data.frequencies[card] ?? 0;
      score *= Math.max(0.3, Math.min(1.0, freq / 5.0));

      candidates.push({ card, frequency: freq, completions, advances, score });
    }

    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, limit);
  }

  // ==========================================================================
  //                    SEARCH
  // ==========================================================================

  searchWords(query: string, limit: number = 20): string[] {
    if (!query) return [];
    const q = query.toLowerCase();
    const results: string[] = [];
    for (const word of this.data.deckWords) {
      if (word.startsWith(q)) {
        results.push(word);
        if (results.length >= limit) break;
      }
    }
    return results;
  }

  // ==========================================================================
  //                    FULL ANALYSIS
  // ==========================================================================

  analyze(deck: string[]): AnalysisResult {
    const targets = this.computeTargetStatuses(deck);
    const enabledTargets = targets.filter((t) => t.enabled);
    const nearTargets = targets.filter((t) => !t.enabled && t.missing <= 2);

    let combos: ComboResult[] = [];
    let madeWords: MadeWord[] = [];

    if (deck.length >= 3) {
      const bf = this.bruteForceAnalysis(deck);
      combos = bf.combos;
      madeWords = bf.madeWords;
    }

    return {
      enabledTargets,
      nearTargets: nearTargets.slice(0, 50),
      combos,
      madeWords,
    };
  }

  getFrequency(word: string): number {
    return this.data.frequencies[word] ?? 0;
  }

  /**
   * Resolve a fragment to the best actual card word.
   */
  private resolveFrag(frag: string, isLeft: boolean): string | null {
    const cards = isLeft
      ? this.data.leftFragCards[frag] || []
      : this.data.rightFragCards[frag] || [];
    if (cards.length === 0) return null;
    return cards.reduce((a, b) =>
      (this.data.frequencies[a] ?? 0) > (this.data.frequencies[b] ?? 0) ? a : b
    );
  }

  /**
   * Browse the decomposition catalog.
   * Returns target words with their decompositions, cards resolved,
   * and missing counts relative to the current deck.
   *
   * Sorted by: bestMissing ASC (closest first), then word length DESC.
   */
  browse(
    deck: string[],
    {
      minLength = 6,
      maxMissing = 12,
      limit = 200,
      search = "",
    }: { minLength?: number; maxMissing?: number; limit?: number; search?: string } = {}
  ): BrowseEntry[] {
    const deckSet = new Set(deck);
    const searchLower = search.toLowerCase();
    const results: BrowseEntry[] = [];

    for (const [word, decomps] of Object.entries(this.data.decompositions)) {
      if (word.length < minLength) continue;
      if (searchLower && !word.includes(searchLower)) continue;

      let bestMissing = Infinity;
      const resolvedDecomps: BrowseEntry["decomps"] = [];

      for (const decomp of decomps) {
        // Resolve fragment cards
        const resolvedCards: string[] = [];
        const inDeck: boolean[] = [];
        let missing = 0;

        if (decomp.leftFrag) {
          const card = this.resolveFrag(decomp.leftFrag, true);
          if (!card) continue; // No card available for this fragment
          resolvedCards.push(card);
          const has = deckSet.has(card);
          inDeck.push(has);
          if (!has) missing++;
        }

        for (const card of decomp.interior) {
          resolvedCards.push(card);
          const has = deckSet.has(card);
          inDeck.push(has);
          if (!has) missing++;
        }

        if (decomp.rightFrag) {
          const card = this.resolveFrag(decomp.rightFrag, false);
          if (!card) continue;
          resolvedCards.push(card);
          const has = deckSet.has(card);
          inDeck.push(has);
          if (!has) missing++;
        }

        // Skip if too many duplicates among resolved cards
        if (new Set(resolvedCards).size !== resolvedCards.length) continue;

        if (missing > maxMissing) continue;
        if (missing < bestMissing) bestMissing = missing;

        resolvedDecomps.push({
          leftFrag: decomp.leftFrag,
          interior: decomp.interior,
          rightFrag: decomp.rightFrag,
          score: decomp.score,
          resolvedCards,
          inDeck,
          missing,
        });
      }

      if (resolvedDecomps.length === 0) continue;

      // Sort decomps: fewest missing first, then highest score
      resolvedDecomps.sort((a, b) => a.missing - b.missing || b.score - a.score);

      results.push({
        word,
        length: word.length,
        frequency: this.data.frequencies[word] ?? 0,
        decomps: resolvedDecomps.slice(0, 3), // top 3 decomps per word
        bestMissing,
      });
    }

    // Sort: fewest missing first, then longest word, then highest frequency
    results.sort((a, b) =>
      a.bestMissing - b.bestMissing ||
      b.length - a.length ||
      b.frequency - a.frequency
    );

    return results.slice(0, limit);
  }

  /**
   * Get all fragment provider cards for a given fragment.
   */
  getFragProviders(frag: string, isLeft: boolean): string[] {
    return isLeft
      ? this.data.leftFragCards[frag] || []
      : this.data.rightFragCards[frag] || [];
  }
}
