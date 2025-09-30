# Brainstorming: Building an Optimal Proseset Deck

## Goals
- Generate a deck of word cards that maximizes the number and diversity of valid re-segmentation sets.
- Favor words that can appear flexibly at the start, middle, and end of a concatenated run.
- Avoid trivial pluralizations and other low-value overlaps while encouraging satisfying re-breaks and containments.

## Working Definition
A *set* is an ordered list of ≥3 cards whose concatenated letters can also be segmented into ≥3 valid words in a different boundary arrangement (letters must stay in order, no insertions/deletions). Single-letter words such as `a` and `I` are allowed.

## Data Considerations
- Start with a cleaned English word list (e.g., wordfreq, wordfreq zipf > 3) or enable dictionary toggles for niche words, names, acronyms.
- Precompute metadata: word length, prefixes/suffixes of length 1–5, internal substrings that are standalone words, whether word forms invite easy pluralization (`s`, `es`).

## Core Challenges
1. **Overlap discovery**: Quickly find words whose concatenations admit alternate segmentations.
2. **Scoring**: Rank words by their combinatorial value relative to other selected cards.
3. **Deck balance**: Ensure coverage across different lengths, vowel/consonant profiles, positions (start/middle/end roles).

## Data Structures & Indexes
- **Prefix trie (forward)**: For fast lookup of words by starting letters; each node tracks word endings reachable from that prefix.
- **Suffix trie (reverse)**: Mirror trie built from reversed words to match overlapping suffixes.
- **Prefix/Suffix tables**: Maps from `prefix -> word_ids` and `suffix -> word_ids`; keep lengths 1–5 to capture useful overlaps without blowing up memory.
- **Substring dictionary**: Map `substring -> word_ids` when substring is itself a valid word; enables containment cases (`fee leverage`).
- **Co-occurrence matrix**: Sparse structure counting how often two (or three) words appear together in at least one valid re-segmentation.

## Algorithmic Sketch (Python or Node)
1. **Dictionary ingestion**
   - Normalize case, strip punctuation, optionally remove plurals ending in `s` unless they have non-trivial interior splits.
   - Calculate `starts_with_s` flag so we can deprioritize or exclude.

2. **Build tries and lookup tables**
   - Construct a forward trie for prefixes and a reverse trie for suffixes.
   - For each word, store all prefixes/suffixes length 1–5 and any internal substrings that appear in dictionary.

3. **Enumerate overlap opportunities**
   - For each ordered pair `(A, B)`, find overlaps where suffix of `A` equals prefix of `B`; use trie traversal instead of brute-force.
   - For each overlap, record the concatenated string and the segmentation `A | B`. Keep metadata about remaining unmatched prefix/suffix letters.
   - Extend to triples `(A, B, C)` by chaining overlap results: suffix of `AB` vs prefix of `C`, or equivalently, iterate over valid breakpoints in the concatenated string and ensure each segment is in dictionary.
   - Containment: For each word `W`, scan the trie to find internal segmentations that produce multiple words; treat `W` as the middle word in sets.

4. **Segment validation**
   - Given a concatenated string `S` and original segmentation, run a DFS over indexes using the trie to find alternate segmentation paths. Memoize substring indices to avoid repeated searches.
   - Accept segmentations that differ from the original (order of card words must be identical, but breakpoints change).

5. **Scoring heuristics**
   - `flexibility_score`: number of unique positions the word appears (start/middle/end) across valid sets.
   - `partner_diversity`: count of distinct partners in valid triples/quads; weighted higher for cross-length or cross-letter overlaps.
   - `rebreak_richness`: number of distinct alternate segmentations the word participates in (including containment cases).
   - Penalize words that only function in one position or rely on trivial `s` prefixes/suffixes.
   - Consider iterative scoring: start with global metrics, then re-score relative to currently selected deck (e.g., greedy selection updating partner counts).

6. **Deck construction strategy**
   - **Greedy iterative**: pick the word with highest score, update co-occurrence scores for remaining words based on new overlaps, repeat until deck size reached.
   - **Community detection**: build a graph where nodes are words, weighted edges represent co-occurrence counts; use clustering (Louvain, spectral) to ensure selected cards span multiple communities for diversity.
   - **Role balancing**: maintain quotas for start/middle/end heavy words so gameplay always has viable sets on table.

7. **Analysis & Visualization**
   - Generate heatmaps or chord diagrams of word overlaps to intuitively inspect coverage.
   - Track dead cards (zero valid sets) and prune them early.

## Computational Notes
- Precomputing all valid triples is feasible with pruning: use tries to limit branching; memoize intermediate concatenations; skip paths exceeding a max length.
- Use multiprocessing or worker threads (Node worker pool) for large dictionaries.
- Serialize tries and lookup tables to disk (JSON, pickle) for quick reloads during tuning.

## TODO
- [ ] Choose and clean the base dictionary; define exclusion rules (proper nouns, plural-only forms).
- [ ] Implement trie builders (forward and reverse) plus prefix/suffix tables.
- [ ] Write a segmentation validator that enumerates alternate breaks using memoized DFS.
- [ ] Build pairwise overlap finder; extend to triple enumeration with pruning based on overlap length.
- [ ] Record co-occurrence statistics and role metadata for each word.
- [ ] Design initial scoring functions and run experiments to calibrate weights.
- [ ] Prototype greedy deck builder; compare with graph-clustering approach.
- [ ] Create scripts for visualizing overlaps and identifying dead/overly dominant words.
- [ ] Package results into reproducible reports (notebooks or markdown summaries).
