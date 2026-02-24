#!/usr/bin/env python3
"""
Proseset Puzzle Analyzer v3 - puzzle3.py

Runs puzzle2's anchor-based optimizer, then performs brute-force
enumeration of ALL valid triplets (and optionally quadruplets) over
the final 12-word deck. This gives ground-truth connectivity and
made-word data, unlike puzzle1's template system which misses
words spanning multiple card boundaries.

With 12 cards:
  - Ordered triplets:    12 × 11 × 10 = 1,320
  - Ordered quadruplets: 12 × 11 × 10 × 9 = 11,880
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
from itertools import permutations
from collections import defaultdict
import sys
import os
import random
import copy

# Import from puzzle1
from puzzle1 import (
    load_dictionary, set_segmentation_dict, segment_word,
    build_lookups, Lookups, DeckState,
    create_deck_state, get_zipf,
    BANNED_WORDS, WORD_BLACKLIST, _seg_words,
    ALLOWED_1_LETTER, ALLOWED_2_LETTER,
    get_cache_key as get_puzzle1_cache_key,
    load_cache, save_cache,
)
import third_party.twl as twl

# Import from puzzle2
from puzzle2 import (
    load_stub_cache, select_anchors, format_anchor,
    optimize_puzzle_with_anchors, Decomposition,
    GREEN, RED, YELLOW, RESET,
)


# ============================================================================ #
#                         BRUTE-FORCE ANALYSIS                                 #
# ============================================================================ #

def find_all_segmentations(concat: str) -> List[Tuple[str, ...]]:
    """Find all ways to segment a concatenation into valid words."""
    return segment_word(concat, require_start=True, require_end=True)


def compute_boundaries(words: Tuple[str, ...]) -> Set[int]:
    """Compute internal boundary positions for a sequence of words."""
    boundaries = set()
    pos = 0
    for w in words[:-1]:  # Skip last word (no boundary after it)
        pos += len(w)
        boundaries.add(pos)
    return boundaries


def is_alternate_segmentation(seg: Tuple[str, ...], original_boundaries: Set[int]) -> bool:
    """Check if a segmentation has NO boundaries matching the original card splits."""
    seg_boundaries = compute_boundaries(seg)
    return not (seg_boundaries & original_boundaries)


def brute_force_combos(
    deck: List[str],
    max_combo_size: int = 3,
) -> Dict[str, object]:
    """
    Enumerate all ordered combinations of deck words (size 3 and optionally 4),
    concatenate, segment, and collect made-words and connectivity data.

    Returns a dict with:
      - 'made_words': {word: list of (combo, segmentation)}
      - 'combos': list of (combo_tuple, list_of_valid_segmentations)
      - 'word_connections': {word: set of words it can be adjacent to}
      - 'word_left_neighbors': {word: set of words that can appear to its left}
      - 'word_right_neighbors': {word: set of words that can appear to its right}
      - 'word_combo_count': {word: number of combos it participates in}
    """
    made_words = defaultdict(list)       # made_word -> [(combo, seg), ...]
    valid_combos = []                     # (combo, [segs])
    word_left_neighbors = defaultdict(set)
    word_right_neighbors = defaultdict(set)
    word_combo_count = defaultdict(int)

    total_checked = 0
    total_valid = 0

    for size in range(3, max_combo_size + 1):
        for combo in permutations(deck, size):
            total_checked += 1
            concat = ''.join(combo)
            original_boundaries = compute_boundaries(combo)

            segs = find_all_segmentations(concat)
            if not segs:
                continue

            # Filter to alternate segmentations (no shared boundaries)
            alt_segs = [s for s in segs if is_alternate_segmentation(s, original_boundaries)]
            if not alt_segs:
                continue

            total_valid += 1
            valid_combos.append((combo, alt_segs))

            # Track connectivity — adjacent pairs
            for i in range(len(combo) - 1):
                word_right_neighbors[combo[i]].add(combo[i + 1])
                word_left_neighbors[combo[i + 1]].add(combo[i])

            # Track combo participation
            for w in combo:
                word_combo_count[w] += 1

            # Collect made words
            for seg in alt_segs:
                pos = 0
                for made_word in seg:
                    start = pos
                    end = pos + len(made_word)

                    # Skip if this word occupies the exact same span as an original card
                    is_original = False
                    card_pos = 0
                    for card in combo:
                        if made_word == card and start == card_pos and end == card_pos + len(card):
                            is_original = True
                            break
                        card_pos += len(card)

                    if not is_original:
                        made_words[made_word].append((combo, seg))

                    pos = end

    return {
        'made_words': dict(made_words),
        'valid_combos': valid_combos,
        'word_left_neighbors': dict(word_left_neighbors),
        'word_right_neighbors': dict(word_right_neighbors),
        'word_combo_count': dict(word_combo_count),
        'total_checked': total_checked,
        'total_valid': total_valid,
    }


# ============================================================================ #
#                         DISPLAY                                              #
# ============================================================================ #

def display_analysis(deck: List[str], results: dict, anchor_cards: Set[str] = None):
    if anchor_cards is None:
        anchor_cards = set()

    print(f"\n{'='*70}")
    print(f"BRUTE-FORCE ANALYSIS ({len(deck)} words)")
    print(f"{'='*70}")
    print(f"Words: {', '.join(sorted(deck))}")
    print(f"Combos checked: {results['total_checked']:,}")
    print(f"Valid combos (with alternate segmentations): {results['total_valid']:,}")

    # --- Connectivity ---
    print(f"\n{'='*70}")
    print("CONNECTIVITY")
    print(f"{'='*70}")

    deck_size = len(deck)
    connectivity_data = []
    for word in sorted(deck):
        left = results['word_left_neighbors'].get(word, set())
        right = results['word_right_neighbors'].get(word, set())
        total = left | right
        combos = results['word_combo_count'].get(word, 0)
        connectivity_data.append((word, left, right, total, combos))

    connectivity_data.sort(key=lambda x: -len(x[3]))

    for word, left, right, total, combos in connectivity_data:
        left_pct = len(left) / (deck_size - 1) * 100 if deck_size > 1 else 0
        right_pct = len(right) / (deck_size - 1) * 100 if deck_size > 1 else 0
        total_pct = len(total) / (deck_size - 1) * 100 if deck_size > 1 else 0
        lock = " [anchor]" if word in anchor_cards else ""
        print(f"  {word:<12} L:{len(left):2d} R:{len(right):2d} Total:{len(total):2d}/{deck_size-1} ({total_pct:4.0f}%)  combos:{combos:4d}{lock}")

    avg_total = sum(len(x[3]) for x in connectivity_data) / len(connectivity_data) if connectivity_data else 0
    avg_pct = avg_total / (deck_size - 1) * 100 if deck_size > 1 else 0
    print(f"\n  Average connectivity: {avg_pct:.1f}%")

    # --- Made Words ---
    made_words = results['made_words']
    print(f"\n{'='*70}")
    print(f"MADE WORDS ({len(made_words)} unique)")
    print(f"{'='*70}")

    if not made_words:
        print("  No made words found!")
        return

    # Deduplicate: count unique combos per made word (not per segmentation)
    made_word_stats = []
    for mw, occurrences in made_words.items():
        unique_combos = set(combo for combo, seg in occurrences)
        freq = get_zipf(mw)
        made_word_stats.append((mw, len(unique_combos), len(occurrences), freq))

    # --- By length (longest first) ---
    by_length = sorted(made_word_stats, key=lambda x: (-len(x[0]), -x[1]))
    print(f"\nLongest made words:")
    for i, (mw, n_combos, n_total, freq) in enumerate(by_length[:40]):
        # Show an example combo
        example_combo, example_seg = made_words[mw][0]
        combo_str = " + ".join(example_combo)
        seg_str = " | ".join(example_seg)
        print(f"  {mw:<20} (len={len(mw):2d}, freq={freq:.1f}) from {n_combos:3d} combo(s)")
        if i < 10:
            print(f"      e.g. {combo_str} -> {seg_str}")

    # --- By frequency (most recognizable) ---
    long_and_common = [x for x in made_word_stats if len(x[0]) >= 4]
    long_and_common.sort(key=lambda x: (-x[3], -len(x[0])))
    print(f"\nMost recognizable made words (len>=4, by frequency):")
    for i, (mw, n_combos, n_total, freq) in enumerate(long_and_common[:30]):
        example_combo, example_seg = made_words[mw][0]
        combo_str = " + ".join(example_combo)
        seg_str = " | ".join(example_seg)
        print(f"  {mw:<20} (len={len(mw):2d}, freq={freq:.1f}) from {n_combos:3d} combo(s)")
        if i < 5:
            print(f"      e.g. {combo_str} -> {seg_str}")

    # --- By versatility (most different combos) ---
    by_versatility = sorted(made_word_stats, key=lambda x: (-x[1], -len(x[0])))
    print(f"\nMost versatile made words (most unique combos):")
    for i, (mw, n_combos, n_total, freq) in enumerate(by_versatility[:20]):
        print(f"  {mw:<20} (len={len(mw):2d}) created by {n_combos:3d} different combos")

    # --- Sample valid combos ---
    print(f"\n{'='*70}")
    print("SAMPLE VALID COMBOS")
    print(f"{'='*70}")

    # Sort by number of alternate segmentations (most interesting first)
    interesting_combos = sorted(results['valid_combos'], key=lambda x: -len(x[1]))
    shown = 0
    for combo, alt_segs in interesting_combos[:20]:
        combo_str = " + ".join(combo)
        concat = ''.join(combo)
        print(f"\n  {combo_str} = \"{concat}\"")
        for seg in alt_segs[:3]:
            seg_str = " | ".join(seg)
            print(f"    -> {seg_str}")
        if len(alt_segs) > 3:
            print(f"    ... and {len(alt_segs) - 3} more segmentations")
        shown += 1


# ============================================================================ #
#                              MAIN                                            #
# ============================================================================ #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Puzzle optimizer with brute-force analysis")
    parser.add_argument('--anchors', type=int, default=1, help='Number of anchor decompositions (default: 1)')
    parser.add_argument('--min-length', type=int, default=10, help='Minimum target word length for anchors (default: 10)')
    parser.add_argument('--puzzle-size', type=int, default=12, help='Total puzzle size (default: 12)')
    parser.add_argument('--max-swaps', type=int, default=50, help='Max optimization swaps (default: 50)')
    parser.add_argument('--max-combo', type=int, default=3, help='Max combo size to check (3=triplets, 4=+quadruplets)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--words', type=str, default=None, help='Comma-separated deck words (skip optimization, just analyze)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(seed)
    print(f"Seed: {seed}")

    # --- Load puzzle1 infrastructure ---
    cache_path = get_puzzle1_cache_key()

    if os.path.exists(cache_path):
        try:
            deck_words, seg_words, lookups = load_cache(cache_path)
            seg_words = seg_words - BANNED_WORDS
            set_segmentation_dict(seg_words)
            print(f"Deck words: {len(deck_words)}, Segmentation words: {len(seg_words)}")
        except Exception as e:
            print(f"Cache load failed: {e}, rebuilding...")
            deck_words, seg_words, lookups = None, None, None
    else:
        deck_words, seg_words, lookups = None, None, None

    if deck_words is None or seg_words is None:
        print("Loading dictionary...")
        deck_words, seg_words = load_dictionary()
        print(f"Deck words: {len(deck_words)}, Segmentation words: {len(seg_words)}")
        set_segmentation_dict(seg_words)
        lookups = None

    if lookups is None:
        print("\nBuilding lookups...")
        lookups = build_lookups(deck_words, seg_words)
        save_cache(cache_path, deck_words, seg_words, lookups)

    anchor_cards = set()

    if args.words:
        # --- Direct analysis mode: skip optimization ---
        deck = [w.strip().lower() for w in args.words.split(',')]
        print(f"\nDirect analysis of {len(deck)} words: {', '.join(deck)}")
    else:
        # --- Load puzzle2-stub cache ---
        table, left_frag_cards, right_frag_cards = load_stub_cache()

        # --- Select anchors ---
        print(f"\n{'='*60}")
        print("SELECTING ANCHOR DECOMPOSITIONS")
        print(f"{'='*60}")

        anchors = select_anchors(
            table, left_frag_cards, right_frag_cards,
            deck_words=deck_words,
            num_anchors=args.anchors,
            min_target_length=args.min_length,
        )

        if not anchors:
            print("Failed to select anchors. Exiting.")
            return

        for i, (word, decomp, cards) in enumerate(anchors):
            fmt = format_anchor(word, decomp, left_frag_cards, right_frag_cards)
            print(f"\n  Anchor {i+1}: {word} ({len(word)} chars)")
            print(f"    Decomposition: {fmt}")
            print(f"    Cards needed: {', '.join(cards)}")
            anchor_cards.update(cards)

        print(f"\n  Total anchor cards: {len(anchor_cards)} ({', '.join(sorted(anchor_cards))})")

        # --- Optimize ---
        print(f"\n{'='*60}")
        print("OPTIMIZING PUZZLE")
        print(f"{'='*60}")

        deck, state = optimize_puzzle_with_anchors(
            anchor_cards=anchor_cards,
            deck_words=deck_words,
            lookups=lookups,
            puzzle_size=args.puzzle_size,
            max_swaps=args.max_swaps,
            min_triplets=2,
        )

        # --- Show anchor info ---
        print(f"\n{'='*60}")
        print("ANCHOR WORDS IN PUZZLE")
        print(f"{'='*60}")
        for word, decomp, cards in anchors:
            fmt = format_anchor(word, decomp, left_frag_cards, right_frag_cards)
            print(f"  Target: {word}")
            print(f"    {fmt}")
            print(f"    Cards: {' + '.join(cards)}")

    # --- Switch to broad dictionary for brute-force analysis ---
    # During optimization we use top-50k words for card selection,
    # but for finding made-words we want the full TWL dictionary
    # (a player can spot any valid Scrabble word in the letter sequence)
    broad_seg_words = (
        ALLOWED_1_LETTER | ALLOWED_2_LETTER |
        {w for w in twl.iterator() if w.isalpha() and len(w) > 2}
    ) - BANNED_WORDS - WORD_BLACKLIST
    set_segmentation_dict(broad_seg_words)
    print(f"\nSwitched to broad TWL dictionary for analysis: {len(broad_seg_words)} seg words")

    # --- Brute-force analysis ---
    print(f"\n{'='*60}")
    print(f"RUNNING BRUTE-FORCE ANALYSIS (max combo size: {args.max_combo})")
    print(f"{'='*60}")

    results = brute_force_combos(deck, max_combo_size=args.max_combo)

    display_analysis(deck, results, anchor_cards=anchor_cards)


if __name__ == "__main__":
    main()
