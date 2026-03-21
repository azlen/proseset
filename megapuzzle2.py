#!/usr/bin/env python3
"""
Proseset Batch Puzzle Generator v2 - megapuzzle2.py

Changes from megapuzzle1:
  1. Includes doublets (2-card combos) and optionally quads (4-card)
  2. Filters combos to require at least one 4+ letter made word
  3. Saves full list of all made words per puzzle in JSON output
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, FrozenSet
from collections import defaultdict, Counter
from itertools import permutations
import json
import math
import random
import time
import argparse
import sys

from puzzle1 import (
    load_dictionary, set_segmentation_dict, segment_word,
    get_zipf, BANNED_WORDS, WORD_BLACKLIST,
    ALLOWED_1_LETTER, ALLOWED_2_LETTER,
    has_adjacent_one_letter_words,
)
import third_party.twl as twl

from puzzle2 import (
    load_stub_cache, Decomposition,
    best_extension_card, format_anchor,
)

from puzzle3 import (
    find_all_segmentations, compute_boundaries, is_alternate_segmentation,
)

from puzzle4 import (
    TargetReq, score_target, build_target_reqs,
    build_reverse_index, resolve_cards,
)

# Re-import everything else from megapuzzle1 that doesn't change
from megapuzzle1 import (
    DiversityTracker, sample_top_k,
    select_anchor_diverse, greedy_build_diverse,
    _save_results, _print_summary,
)

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'


# ============================================================================ #
#                         BRUTE-FORCE WITH 4+ LETTER FILTER                    #
# ============================================================================ #

def brute_force_combos_v2(
    deck: List[str],
    min_combo_size: int = 2,
    max_combo_size: int = 4,
    min_made_word_len: int = 4,
) -> Dict[str, object]:
    """
    Like brute_force_combos but:
      - Starts from min_combo_size (default 2, i.e. doublets)
      - Filters: a combo is only valid if at least one alternate segmentation
        produces a non-original made word of length >= min_made_word_len
      - Returns all made words
    """
    made_words = defaultdict(list)       # made_word -> [(combo, seg), ...]
    valid_combos = []                     # (combo, [segs])
    word_left_neighbors = defaultdict(set)
    word_right_neighbors = defaultdict(set)
    word_combo_count = defaultdict(int)

    total_checked = 0
    total_valid = 0
    total_skipped_short = 0

    for size in range(min_combo_size, max_combo_size + 1):
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

            # Check if any segmentation has a 4+ letter non-original made word
            has_long_made_word = False
            for seg in alt_segs:
                pos = 0
                for word in seg:
                    start = pos
                    end = pos + len(word)

                    # Check if this is an original card at same span
                    is_original = False
                    card_pos = 0
                    for card in combo:
                        if word == card and start == card_pos and end == card_pos + len(card):
                            is_original = True
                            break
                        card_pos += len(card)

                    if not is_original and len(word) >= min_made_word_len:
                        has_long_made_word = True
                        break

                    pos = end
                if has_long_made_word:
                    break

            if not has_long_made_word:
                total_skipped_short += 1
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
        'total_skipped_short': total_skipped_short,
    }


# ============================================================================ #
#                         BATCH GENERATOR V2                                    #
# ============================================================================ #

def generate_batch_v2(
    num_puzzles: int,
    deck_words: Set[str],
    reqs: List[TargetReq],
    reverse_index: Dict[str, List[int]],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
    broad_seg_words: Set[str],
    seg_words: Set[str],
    puzzle_size: int = 12,
    top_k: int = 10,
    temperature: float = 1.0,
    min_anchor_length: int = 10,
    card_decay: float = 0.5,
    made_word_decay: float = 0.3,
    min_combo: int = 2,
    max_combo: int = 4,
    min_made_word_len: int = 4,
    verbose: bool = False,
    output_file: str = "puzzles.json",
) -> List[dict]:
    """Generate a batch of diverse puzzles with v2 combo analysis."""

    tracker = DiversityTracker(card_decay=card_decay, made_word_decay=made_word_decay)
    puzzles = []

    print(f"\n{'='*70}")
    print(f"GENERATING {num_puzzles} PUZZLES (v2)")
    print(f"{'='*70}")
    print(f"Puzzle size: {puzzle_size}, Top-K: {top_k}, Temp: {temperature}")
    print(f"Card decay: {card_decay}, Made-word decay: {made_word_decay}")
    print(f"Min anchor length: {min_anchor_length}")
    print(f"Combo sizes: {min_combo}-{max_combo}, Min made word length: {min_made_word_len}")

    t_total_start = time.time()

    for puzzle_num in range(1, num_puzzles + 1):
        t_start = time.time()

        # Use narrow dictionary for building
        set_segmentation_dict(seg_words)

        # Select anchor with diversity
        anchor_result = select_anchor_diverse(
            reqs, deck_words, left_frag_cards, right_frag_cards,
            tracker, min_length=min_anchor_length,
        )

        if anchor_result is None:
            print(f"\n{RED}Puzzle {puzzle_num}: Failed to select anchor!{RESET}")
            continue

        anchor_req, anchor_cards = anchor_result
        anchor_word = anchor_req.word

        # Wave-collapse: only seed interior cards
        interior_cards = list(anchor_req.decomp.interior)

        # Find anchor req indices for boosting
        anchor_req_indices = set()
        for i, req in enumerate(reqs):
            if req.word == anchor_word:
                anchor_req_indices.add(i)

        # Build puzzle
        deck = greedy_build_diverse(
            anchor_cards=interior_cards,
            deck_words=deck_words,
            reqs=reqs,
            reverse_index=reverse_index,
            left_frag_cards=left_frag_cards,
            right_frag_cards=right_frag_cards,
            tracker=tracker,
            anchor_req_indices=anchor_req_indices,
            target_size=puzzle_size,
            top_k=top_k,
            temperature=temperature,
            verbose=verbose,
        )

        if len(deck) < puzzle_size:
            print(f"\n{YELLOW}Puzzle {puzzle_num}: Only got {len(deck)} cards (target: {puzzle_size}){RESET}")

        # Switch to broad dictionary for brute-force analysis
        set_segmentation_dict(broad_seg_words)
        results = brute_force_combos_v2(
            deck,
            min_combo_size=min_combo,
            max_combo_size=max_combo,
            min_made_word_len=min_made_word_len,
        )

        # Collect made words
        made_words_set = set(results['made_words'].keys())

        # Count valid combos
        n_combos = results['total_valid']
        n_skipped = results['total_skipped_short']

        # Find enabled target words
        enabled_targets = []
        deck_set = set(deck)
        for req in reqs:
            if req.word == anchor_word:
                continue
            all_interior_present = req.interior <= deck_set
            left_ok = not req.left_frag or any(
                c in deck_set for c in left_frag_cards.get(req.left_frag, [])
            )
            right_ok = not req.right_frag or any(
                c in deck_set for c in right_frag_cards.get(req.right_frag, [])
            )
            if all_interior_present and left_ok and right_ok:
                enabled_targets.append(req.word)
        enabled_targets = sorted(set(enabled_targets), key=lambda w: -len(w))

        # Compute diversity metrics for this puzzle
        new_cards = sum(1 for c in deck if tracker.card_usage[c] == 0)
        new_made_words = sum(1 for mw in made_words_set if tracker.made_word_usage[mw] == 0)

        # Record puzzle in tracker
        tracker.record_puzzle(deck, made_words_set, anchor_word=anchor_word)

        t_elapsed = time.time() - t_start

        # Build puzzle record
        anchor_fmt = format_anchor(anchor_word, anchor_req.decomp, left_frag_cards, right_frag_cards)

        # Sort made words: longest first, then alphabetical
        all_made_sorted = sorted(made_words_set, key=lambda w: (-len(w), w))
        made_4plus = [w for w in all_made_sorted if len(w) >= 4]

        puzzle_record = {
            "id": puzzle_num,
            "cards": sorted(deck),
            "anchor_word": anchor_word,
            "anchor_decomposition": anchor_fmt,
            "anchor_interior_cards": list(anchor_req.decomp.interior),
            "num_valid_combos": n_combos,
            "num_skipped_short_words": n_skipped,
            "num_made_words": len(made_words_set),
            "num_made_words_4plus": len(made_4plus),
            "made_words": all_made_sorted,
            "longest_made_words": all_made_sorted[:10],
            "enabled_target_words": enabled_targets[:20],
            "new_cards": new_cards,
            "new_made_words": new_made_words,
        }
        puzzles.append(puzzle_record)

        # Print summary line
        print(
            f"  {CYAN}#{puzzle_num:<4}{RESET} "
            f"anchor={anchor_word:<18} "
            f"combos={n_combos:4d} (skip={n_skipped:3d}) "
            f"made={len(made_words_set):4d} (4+:{len(made_4plus):3d}) "
            f"new_cards={new_cards:2d}/{len(deck)} "
            f"({t_elapsed:.1f}s)"
        )

        if verbose and all_made_sorted:
            print(f"         longest: {', '.join(all_made_sorted[:5])}")

        # Periodic diversity stats
        if puzzle_num % 50 == 0:
            print(f"\n  --- After {puzzle_num} puzzles: {tracker.stats()} ---\n")

        # Periodic save
        if puzzle_num % 100 == 0:
            _save_results(puzzles, tracker, output_file)

    t_total = time.time() - t_total_start

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(puzzles)} puzzles in {t_total:.1f}s ({t_total/max(1,len(puzzles)):.2f}s/puzzle)")
    print(f"{'='*70}")
    print(f"Final diversity: {tracker.stats()}")

    # Final save
    _save_results(puzzles, tracker, output_file)

    # Summary statistics
    _print_summary(puzzles, tracker)

    return puzzles


# ============================================================================ #
#                              MAIN                                            #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Generate a batch of diverse Proseset puzzles (v2: doublets, 4+ letter filter)"
    )
    parser.add_argument('--num-puzzles', '-n', type=int, default=1000,
                        help='Number of puzzles to generate (default: 1000)')
    parser.add_argument('--puzzle-size', type=int, default=12,
                        help='Cards per puzzle (default: 12)')
    parser.add_argument('--min-length', type=int, default=10,
                        help='Min anchor word length (default: 10)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Top-K sampling breadth (default: 10)')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='Sampling temperature (default: 1.2)')
    parser.add_argument('--card-decay', type=float, default=0.5,
                        help='Card reuse penalty decay (default: 0.5)')
    parser.add_argument('--made-word-decay', type=float, default=0.3,
                        help='Made-word reuse penalty decay (default: 0.3)')
    parser.add_argument('--min-combo', type=int, default=2,
                        help='Min combo size (default: 2, i.e. doublets)')
    parser.add_argument('--max-combo', type=int, default=4,
                        help='Max combo size (default: 4, i.e. quads)')
    parser.add_argument('--min-made-word-len', type=int, default=4,
                        help='Min length of made word to count a combo (default: 4)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--output', '-o', type=str, default='puzzles.json',
                        help='Output JSON file (default: puzzles.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show per-card build details')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(seed)
    print(f"Seed: {seed}")

    # Load dictionary
    print("Loading dictionary...")
    deck_words, seg_words = load_dictionary()
    set_segmentation_dict(seg_words)
    print(f"Deck words: {len(deck_words)}, Seg words: {len(seg_words)}")

    # Load decomposition cache
    table, left_frag_cards, right_frag_cards = load_stub_cache()

    # Build target requirements + reverse index
    print("\nBuilding target requirements...")
    reqs = build_target_reqs(table, deck_words)
    print(f"Target requirements: {len(reqs)} (from {len(table)} decomposable words)")

    print("Building reverse index...")
    reverse_index = build_reverse_index(reqs, left_frag_cards, right_frag_cards)
    print(f"Cards in reverse index: {len(reverse_index)}")

    # Build broad dictionary for brute-force analysis
    broad_seg_words = (
        ALLOWED_1_LETTER | ALLOWED_2_LETTER |
        {w for w in twl.iterator() if w.isalpha() and len(w) > 2}
    ) - BANNED_WORDS - WORD_BLACKLIST
    print(f"Broad TWL dictionary: {len(broad_seg_words)} seg words")

    # Generate puzzles
    puzzles = generate_batch_v2(
        num_puzzles=args.num_puzzles,
        deck_words=deck_words,
        reqs=reqs,
        reverse_index=reverse_index,
        left_frag_cards=left_frag_cards,
        right_frag_cards=right_frag_cards,
        broad_seg_words=broad_seg_words,
        seg_words=seg_words,
        puzzle_size=args.puzzle_size,
        top_k=args.top_k,
        temperature=args.temperature,
        min_anchor_length=args.min_length,
        card_decay=args.card_decay,
        made_word_decay=args.made_word_decay,
        min_combo=args.min_combo,
        max_combo=args.max_combo,
        min_made_word_len=args.min_made_word_len,
        verbose=args.verbose,
        output_file=args.output,
    )

    print(f"\nDone! {len(puzzles)} puzzles saved to {args.output}")


if __name__ == "__main__":
    main()
