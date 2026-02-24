#!/usr/bin/env python3
"""
Proseset Puzzle Optimizer v2 - puzzle2.py

Builds puzzles around pre-computed word decompositions from puzzle2-stub.
Starts with 1-2 random "anchor" decompositions (long target words that
can be formed from 3+ cards) whose cards are protected from swapping,
then optimizes the remaining slots using puzzle1's swap-based optimizer.

Usage:
    python puzzle2.py                        # 1 anchor, default settings
    python puzzle2.py --anchors 2            # 2 anchors
    python puzzle2.py --min-length 12        # longer anchor words
    python puzzle2.py --seed 42              # reproducible
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from wordfreq import zipf_frequency
import pickle
import os
import random
import sys
import copy

# Import infrastructure from puzzle1
from puzzle1 import (
    load_dictionary, set_segmentation_dict, segment_word,
    build_lookups, Lookups, DeckState,
    create_deck_state, add_word_to_state, remove_word_from_state,
    compute_word_contribution, compute_marginal_value,
    score_actual_made_words, count_word_triplets,
    contains_blacklisted_word, has_adjacent_one_letter_words,
    analyze_deck, get_cache_key as get_puzzle1_cache_key,
    load_cache, save_cache, get_zipf,
    WORD_BLACKLIST, BANNED_WORDS,
)

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


# ============================================================================ #
#                              DECOMPOSITION CLASS                             #
# ============================================================================ #

# Must match puzzle2-stub's Decomposition for cache unpickling
@dataclass
class Decomposition:
    left_frag: str
    interior: Tuple[str, ...]
    right_frag: str

    @property
    def num_cards(self) -> int:
        return (1 if self.left_frag else 0) + len(self.interior) + (1 if self.right_frag else 0)

    @property
    def is_pure(self) -> bool:
        return not self.left_frag and not self.right_frag

    def all_pieces(self) -> Tuple[str, ...]:
        parts = []
        if self.left_frag:
            parts.append(self.left_frag)
        parts.extend(self.interior)
        if self.right_frag:
            parts.append(self.right_frag)
        return tuple(parts)


class _DecompUnpickler(pickle.Unpickler):
    """Custom unpickler that maps any 'Decomposition' class to ours."""
    def find_class(self, module, name):
        if name == 'Decomposition':
            return Decomposition
        return super().find_class(module, name)


# ============================================================================ #
#                              STUB CACHE LOADING                              #
# ============================================================================ #

def load_stub_cache() -> Tuple[Dict, Dict, Dict]:
    """Load the precomputed decomposition cache from puzzle2-stub."""
    import glob
    caches = glob.glob(".cache_puzzle2_*.pkl")
    if not caches:
        print("ERROR: No puzzle2-stub cache found. Run puzzle2-stub.py first.")
        sys.exit(1)

    # Sort by modification time, newest first
    caches.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for cache_path in caches:
        try:
            print(f"Trying decomposition cache {cache_path}...")
            with open(cache_path, 'rb') as f:
                data = _DecompUnpickler(f).load()
            if isinstance(data, tuple) and len(data) == 3:
                table, left_frag_cards, right_frag_cards = data
                print(f"Loaded {len(table)} decomposable words")
                return table, left_frag_cards, right_frag_cards
            else:
                print(f"  Skipping: unexpected format (not a 3-tuple)")
        except Exception as e:
            print(f"  Skipping: {e}")

    print("ERROR: No valid puzzle2-stub cache found. Run puzzle2-stub.py first.")
    sys.exit(1)


# ============================================================================ #
#                              ANCHOR SELECTION                                #
# ============================================================================ #

def best_extension_card(cards: List[str], frag: str, frag_is_prefix: bool) -> str:
    """Pick the most recognizable extension card for a fragment."""
    def card_score(card):
        if frag_is_prefix:
            leftover = card[len(frag):]
        else:
            leftover = card[:-len(frag)]
        return get_zipf(card) + get_zipf(leftover)
    return max(cards, key=card_score)


def decomposition_to_cards(
    decomp: Decomposition,
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
) -> List[str]:
    """Convert a decomposition into the actual card words needed."""
    cards = []

    if decomp.left_frag:
        frag_cards = left_frag_cards.get(decomp.left_frag, [])
        if not frag_cards:
            return []
        card = best_extension_card(frag_cards, decomp.left_frag, frag_is_prefix=False)
        cards.append(card)

    cards.extend(decomp.interior)

    if decomp.right_frag:
        frag_cards = right_frag_cards.get(decomp.right_frag, [])
        if not frag_cards:
            return []
        card = best_extension_card(frag_cards, decomp.right_frag, frag_is_prefix=True)
        cards.append(card)

    return cards


def score_decomposition(word: str, decomp: Decomposition) -> float:
    """Score a decomposition (same logic as puzzle2-stub)."""
    score = 1.0
    word_freq = get_zipf(word)
    score *= max(0.1, min(1.0, word_freq / 5.0))
    score *= len(word) ** 0.5

    pieces = decomp.all_pieces()
    for piece in pieces:
        freq = get_zipf(piece)
        if freq < 1.0:
            score *= 0.05
        elif freq < 2.0:
            score *= 0.15
        elif freq < 3.0:
            score *= 0.4
        elif freq < 4.0:
            score *= 0.7

    score *= 0.7 ** max(0, len(pieces) - 3)
    min_piece_len = min(len(p) for p in pieces)
    score *= min_piece_len ** 0.5

    if decomp.is_pure:
        score *= 1.5

    return score


def format_anchor(
    word: str,
    decomp: Decomposition,
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
) -> str:
    """Format an anchor decomposition for display."""
    parts = []

    if decomp.left_frag:
        cards = left_frag_cards.get(decomp.left_frag, [])
        if cards:
            best = best_extension_card(cards, decomp.left_frag, frag_is_prefix=False)
            leftover = best[:-len(decomp.left_frag)]
            parts.append(f"[{leftover}]{decomp.left_frag}")

    parts.extend(decomp.interior)

    if decomp.right_frag:
        cards = right_frag_cards.get(decomp.right_frag, [])
        if cards:
            best = best_extension_card(cards, decomp.right_frag, frag_is_prefix=True)
            leftover = best[len(decomp.right_frag):]
            parts.append(f"{decomp.right_frag}[{leftover}]")

    return " + ".join(parts)


def select_anchors(
    table: Dict[str, List[Decomposition]],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
    deck_words: Set[str],
    num_anchors: int = 1,
    min_target_length: int = 10,
) -> List[Tuple[str, Decomposition, List[str]]]:
    """
    Select random extended-only decompositions as puzzle anchors.

    Returns list of (target_word, decomposition, cards).
    """
    candidates = []
    for word, decomps in table.items():
        if len(word) < min_target_length:
            continue
        # Extended-only: no pure decompositions
        if any(d.is_pure for d in decomps):
            continue

        # Pick best decomposition for this word
        best_decomp = max(decomps, key=lambda d: score_decomposition(word, d))

        # Convert to cards and check they're all valid deck words
        cards = decomposition_to_cards(best_decomp, left_frag_cards, right_frag_cards)
        if not cards:
            continue

        if not all(c in deck_words for c in cards):
            continue

        # No duplicate cards
        if len(set(cards)) != len(cards):
            continue

        # No blacklisted cards
        if any(c in WORD_BLACKLIST for c in cards):
            continue

        candidates.append((word, best_decomp, cards))

    if not candidates:
        print("WARNING: No valid anchor candidates found!")
        return []

    print(f"Found {len(candidates)} valid anchor candidates (extended-only, len>={min_target_length})")

    # Select non-overlapping anchors randomly
    selected = []
    random.shuffle(candidates)
    used_cards = set()

    for word, decomp, cards in candidates:
        if len(selected) >= num_anchors:
            break
        if used_cards & set(cards):
            continue
        selected.append((word, decomp, cards))
        used_cards.update(cards)

    return selected


# ============================================================================ #
#                              PUZZLE OPTIMIZER                                #
# ============================================================================ #

def optimize_puzzle_with_anchors(
    anchor_cards: Set[str],
    deck_words: Set[str],
    lookups: Lookups,
    puzzle_size: int = 12,
    max_swaps: int = 50,
    min_triplets: int = 2,
) -> Tuple[List[str], DeckState]:
    """
    Optimize a puzzle with some cards protected (anchor cards).

    Like puzzle1's optimize_puzzle but:
    - Starts with anchor cards pre-seeded
    - Anchor cards cannot be swapped out
    - Remaining slots filled randomly then optimized
    """
    # Filter candidates (same as puzzle1)
    candidates = deck_words.copy()
    candidates = {w for w in candidates if not (w.startswith('s') or w.endswith('s'))}
    candidates = {w for w in candidates if not has_adjacent_one_letter_words(w)}
    candidates = {w for w in candidates if w in lookups.templates_as_middle}
    candidates = {w for w in candidates if w not in WORD_BLACKLIST}

    # Remove anchor cards from candidate pool
    candidates -= anchor_cards
    viable = list(candidates)

    # Build initial deck: anchor cards + random fill
    remaining_slots = puzzle_size - len(anchor_cards)
    if remaining_slots < 0:
        print(f"WARNING: Anchor cards ({len(anchor_cards)}) exceed puzzle size ({puzzle_size})")
        remaining_slots = 0

    fill_words = random.sample(viable, remaining_slots) if remaining_slots > 0 else []
    initial_words = list(anchor_cards) + fill_words

    state = create_deck_state(initial_words, lookups)
    deck_list = list(initial_words)

    print(f"\nStarting puzzle optimization with {len(anchor_cards)} anchor cards + {remaining_slots} random fills")
    print(f"Anchor cards (protected): {', '.join(sorted(anchor_cards))}")
    print(f"Fill cards: {', '.join(fill_words)}")
    print(f"Max swaps: {max_swaps}, Min triplets per word: {min_triplets}")

    # Check initial triplet counts
    print(f"\nInitial triplet counts:")
    for word in deck_list:
        triplet_count = count_word_triplets(word, state, lookups)
        lock = " [anchor]" if word in anchor_cards else ""
        print(f"  {word:<12} can form {triplet_count:3d} triplets{lock}")

    best_score = sum(compute_word_contribution(w, state, lookups)[0] for w in deck_list)
    print(f"\nInitial total score: {best_score:.1f}")

    # Track best state
    best_deck_score = best_score
    best_deck_list = deck_list.copy()
    best_deck_state = copy.deepcopy(state)

    consecutive_regressions = 0
    total_regressions = 0
    MAX_CONSECUTIVE_REGRESSIONS = 3
    MAX_TOTAL_REGRESSIONS = 8

    swap_count = 0
    candidates_pool = [w for w in viable if w not in set(deck_list)]
    recent_swaps = set()  # Track (removed, added) pairs to detect cycles

    while swap_count < max_swaps:
        swap_count += 1
        print(f"\n{'='*60}")
        print(f"SWAP {swap_count}/{max_swaps}")
        print(f"{'='*60}")

        # Find worst SWAPPABLE word (skip anchors)
        worst_word = None
        worst_score = float('inf')
        worst_triplets = 0

        for word in deck_list:
            if word in anchor_cards:
                continue  # Protected!

            triplet_count = count_word_triplets(word, state, lookups)
            score, _ = compute_word_contribution(word, state, lookups)

            if triplet_count < min_triplets:
                score = score * 0.1

            if score < worst_score:
                worst_score = score
                worst_word = word
                worst_triplets = triplet_count

        if worst_word is None:
            print("All words are anchors — nothing to swap!")
            break

        print(f"Worst swappable word: {worst_word} (score={worst_score:.1f}, triplets={worst_triplets})")

        # Temporarily remove worst word to evaluate replacements
        remove_word_from_state(worst_word, state, lookups)

        # Fast scoring for top candidates
        candidate_scores = []
        for candidate in candidates_pool:
            if candidate in state.deck:
                continue
            if contains_blacklisted_word(candidate, state, lookups):
                continue

            triplet_count = count_word_triplets(candidate, state, lookups)
            if triplet_count < min_triplets:
                continue

            base_score, combos, breakdown = compute_marginal_value(candidate, state, lookups)
            if base_score > 0:
                candidate_scores.append((candidate, base_score, combos, breakdown, triplet_count))

        if not candidate_scores:
            add_word_to_state(worst_word, state, lookups)
            print(f"No viable replacement found. Stopping.")
            break

        # Refine top candidates with made-word scoring
        # Skip overly connected words, take mid-range for better balance
        candidate_scores.sort(key=lambda x: -x[1])
        if len(candidate_scores) >= 50:
            top_candidates = candidate_scores[30:50]
        elif len(candidate_scores) > 30:
            top_candidates = candidate_scores[30:]
        else:
            top_candidates = candidate_scores[:min(20, len(candidate_scores))]

        best_replacement = None
        best_replacement_score = -float('inf')
        best_replacement_triplets = 0
        best_base = 0
        best_made = 0

        for candidate, base_score, combos, breakdown, triplet_count in top_candidates:
            made_word_bonus = score_actual_made_words(candidate, state, lookups)
            total_score = base_score + made_word_bonus

            if total_score > best_replacement_score:
                best_replacement_score = total_score
                best_replacement = candidate
                best_replacement_triplets = triplet_count
                best_base = base_score
                best_made = made_word_bonus

        # Re-add worst word before performing swap
        add_word_to_state(worst_word, state, lookups)

        if best_replacement is None:
            print(f"No viable replacement found. Stopping.")
            break

        print(f"Best replacement: {best_replacement} (base={best_base:.1f} +made={best_made:.1f} total={best_replacement_score:.1f})")

        # Cycle detection: have we seen this exact swap before?
        swap_pair = (worst_word, best_replacement)
        if swap_pair in recent_swaps:
            print(f"\n{YELLOW}Cycle detected: already swapped {worst_word} -> {best_replacement}. Stopping.{RESET}")
            break
        recent_swaps.add(swap_pair)

        # Perform swap
        print(f"{RED}-{worst_word:<12}{RESET} -> {GREEN}+{best_replacement:<12}{RESET}")
        remove_word_from_state(worst_word, state, lookups)
        deck_list.remove(worst_word)
        add_word_to_state(best_replacement, state, lookups)
        deck_list.append(best_replacement)

        new_score = sum(compute_word_contribution(w, state, lookups)[0] for w in deck_list)
        improvement = new_score - best_score
        print(f"Total score: {best_score:.1f} -> {new_score:.1f} (delta={improvement:+.1f})")

        candidates_pool.remove(best_replacement)
        candidates_pool.append(worst_word)
        best_score = new_score

        if improvement <= 0:
            consecutive_regressions += 1
            total_regressions += 1
            print(f"{RED}Regression accepted{RESET} (consecutive: {consecutive_regressions}/{MAX_CONSECUTIVE_REGRESSIONS}, total: {total_regressions}/{MAX_TOTAL_REGRESSIONS})")

            if consecutive_regressions >= MAX_CONSECUTIVE_REGRESSIONS:
                print(f"\n{RED}Maximum consecutive regressions reached. Stopping.{RESET}")
                break
            if total_regressions >= MAX_TOTAL_REGRESSIONS:
                print(f"\n{RED}Maximum total regressions reached. Stopping.{RESET}")
                break
        else:
            consecutive_regressions = 0
            print(f"{GREEN}Improvement! Consecutive regressions reset.{RESET}")

            if new_score > best_deck_score:
                best_deck_score = new_score
                best_deck_list = deck_list.copy()
                best_deck_state = copy.deepcopy(state)
                print(f"{GREEN}* New best score: {best_deck_score:.1f}{RESET}")

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE ({swap_count} swaps)")
    print(f"{'='*60}")

    # Restore best state if we ended on a regression
    if best_deck_state is not None and best_score < best_deck_score:
        print(f"\n{GREEN}Restoring best state (score: {best_deck_score:.1f} vs current: {best_score:.1f}){RESET}")
        deck_list = best_deck_list
        state = best_deck_state

    print(f"Final words: {', '.join(sorted(deck_list))}")

    # Final triplet check
    print(f"\nFinal triplet counts:")
    for word in sorted(deck_list):
        triplet_count = count_word_triplets(word, state, lookups)
        lock = " [anchor]" if word in anchor_cards else ""
        status = "ok" if triplet_count >= min_triplets else "LOW"
        print(f"  {status:3s} {word:<12} can form {triplet_count:3d} triplets{lock}")

    return deck_list, state


# ============================================================================ #
#                              MAIN                                            #
# ============================================================================ #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize puzzles around anchor decompositions")
    parser.add_argument('--anchors', type=int, default=1, help='Number of anchor decompositions (default: 1)')
    parser.add_argument('--min-length', type=int, default=10, help='Minimum target word length for anchors (default: 10)')
    parser.add_argument('--puzzle-size', type=int, default=12, help='Total puzzle size (default: 12)')
    parser.add_argument('--max-swaps', type=int, default=50, help='Max optimization swaps (default: 50)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
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

    all_anchor_cards = set()
    for i, (word, decomp, cards) in enumerate(anchors):
        fmt = format_anchor(word, decomp, left_frag_cards, right_frag_cards)
        print(f"\n  Anchor {i+1}: {word} ({len(word)} chars)")
        print(f"    Decomposition: {fmt}")
        print(f"    Cards needed: {', '.join(cards)}")
        all_anchor_cards.update(cards)

    print(f"\n  Total anchor cards: {len(all_anchor_cards)} ({', '.join(sorted(all_anchor_cards))})")

    # --- Optimize ---
    print(f"\n{'='*60}")
    print("OPTIMIZING PUZZLE")
    print(f"{'='*60}")

    deck, state = optimize_puzzle_with_anchors(
        anchor_cards=all_anchor_cards,
        deck_words=deck_words,
        lookups=lookups,
        puzzle_size=args.puzzle_size,
        max_swaps=args.max_swaps,
        min_triplets=2,
    )

    # --- Show anchor info in final output ---
    print(f"\n{'='*60}")
    print("ANCHOR WORDS IN PUZZLE")
    print(f"{'='*60}")
    for word, decomp, cards in anchors:
        fmt = format_anchor(word, decomp, left_frag_cards, right_frag_cards)
        print(f"  Target: {word}")
        print(f"    {fmt}")
        print(f"    Cards: {' + '.join(cards)}")

    analyze_deck(deck, lookups, state)


if __name__ == "__main__":
    main()
