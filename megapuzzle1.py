#!/usr/bin/env python3
"""
Proseset Batch Puzzle Generator - megapuzzle1.py

Generates 1000+ diverse puzzles by tracking card and made-word usage
across all previously generated puzzles. Words that have been used before
score less, pushing the generator toward fresh vocabulary each time.

Built on puzzle6's incremental scoring + top-K sampling foundation.

Diversity mechanisms:
  1. Card penalty: cards used in previous puzzles get a score discount
  2. Made-word penalty: made-words seen in previous puzzles are less valuable
  3. Anchor penalty: anchor words/cards used before are deprioritized
  4. Per-puzzle randomness via top-K sampling with temperature
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, FrozenSet
from collections import defaultdict, Counter
import json
import math
import random
import time
import argparse
import sys

from puzzle1 import (
    load_dictionary, set_segmentation_dict,
    get_zipf, BANNED_WORDS, WORD_BLACKLIST,
    ALLOWED_1_LETTER, ALLOWED_2_LETTER,
    has_adjacent_one_letter_words,
)
import third_party.twl as twl

from puzzle2 import (
    load_stub_cache, Decomposition,
    best_extension_card, format_anchor,
)

from puzzle3 import brute_force_combos

from puzzle4 import (
    TargetReq, score_target, build_target_reqs,
    build_reverse_index, resolve_cards,
)

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'


# ============================================================================ #
#                         DIVERSITY TRACKER                                     #
# ============================================================================ #

class DiversityTracker:
    """Tracks word usage across all generated puzzles for diversity scoring."""

    def __init__(self, card_decay: float = 0.5, made_word_decay: float = 0.3):
        # How many puzzles each card/made-word has appeared in
        self.card_usage: Counter = Counter()
        self.made_word_usage: Counter = Counter()
        self.anchor_word_usage: Counter = Counter()

        # Decay controls how aggressively we penalize re-use
        # discount = 1 / (1 + decay * usage_count)
        self.card_decay = card_decay
        self.made_word_decay = made_word_decay

        self.num_puzzles = 0

    def card_discount(self, card: str) -> float:
        """Discount factor for a card based on prior usage. Range (0, 1]."""
        usage = self.card_usage[card]
        if usage == 0:
            return 1.0
        return 1.0 / (1.0 + self.card_decay * usage)

    def made_word_discount(self, word: str) -> float:
        """Discount factor for a made-word based on prior usage."""
        usage = self.made_word_usage[word]
        if usage == 0:
            return 1.0
        return 1.0 / (1.0 + self.made_word_decay * usage)

    def anchor_discount(self, word: str) -> float:
        """Discount factor for an anchor word based on prior usage."""
        usage = self.anchor_word_usage[word]
        if usage == 0:
            return 1.0
        # Aggressive penalty for reusing anchors
        return 1.0 / (1.0 + 2.0 * usage)

    def record_puzzle(self, cards: List[str], made_words: Set[str], anchor_word: str = None):
        """Record a generated puzzle's cards and made-words."""
        self.num_puzzles += 1
        for card in cards:
            self.card_usage[card] += 1
        for mw in made_words:
            self.made_word_usage[mw] += 1
        if anchor_word:
            self.anchor_word_usage[anchor_word] += 1

    def stats(self) -> str:
        """Return a summary string of diversity stats."""
        unique_cards = len(self.card_usage)
        unique_made = len(self.made_word_usage)
        avg_card_use = sum(self.card_usage.values()) / max(1, unique_cards)
        most_used = self.card_usage.most_common(5)
        most_str = ", ".join(f"{w}({c})" for w, c in most_used)
        return (
            f"Unique cards: {unique_cards}, Unique made-words: {unique_made}, "
            f"Avg card reuse: {avg_card_use:.1f}, Most used: {most_str}"
        )


# ============================================================================ #
#                         TOP-K SAMPLING                                        #
# ============================================================================ #

def sample_top_k(
    candidates: List[Tuple[str, float, List[str], int]],
    k: int = 10,
    temperature: float = 1.0,
) -> Tuple[str, float, List[str], int]:
    """Sample from the top-K candidates weighted by score."""
    candidates.sort(key=lambda x: -x[1])
    top = candidates[:k]

    if temperature <= 0 or len(top) == 1:
        return top[0]

    scores = [max(x[1], 0.001) for x in top]
    max_score = max(scores)

    weights = []
    for s in scores:
        log_w = math.log(s / max_score) / temperature
        weights.append(math.exp(log_w))

    total = sum(weights)
    weights = [w / total for w in weights]

    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return top[i]

    return top[-1]


# ============================================================================ #
#                         ANCHOR SELECTION WITH DIVERSITY                       #
# ============================================================================ #

def select_anchor_diverse(
    reqs: List[TargetReq],
    deck_words: Set[str],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
    tracker: DiversityTracker,
    min_length: int = 10,
    extended_only: bool = True,
) -> Tuple[TargetReq, List[str]] | None:
    """Select a single anchor, penalizing previously used ones."""
    candidates = []

    for req in reqs:
        if len(req.word) < min_length:
            continue
        if extended_only and req.decomp.is_pure:
            continue

        cards = resolve_cards(req.decomp, left_frag_cards, right_frag_cards)
        if not cards:
            continue
        if not all(c in deck_words for c in cards):
            continue
        if len(set(cards)) != len(cards):
            continue
        if any(c in WORD_BLACKLIST for c in cards):
            continue

        # Apply diversity discounts
        anchor_disc = tracker.anchor_discount(req.word)
        card_disc = 1.0
        for c in cards:
            card_disc *= tracker.card_discount(c)

        adjusted_score = req.score * anchor_disc * card_disc
        candidates.append((req, cards, adjusted_score))

    if not candidates:
        return None

    # Sort by diversity-adjusted score, sample from top 30
    candidates.sort(key=lambda x: -x[2])
    top = candidates[:30]

    # Weighted sampling from top candidates
    if len(top) == 1:
        return (top[0][0], top[0][1])

    scores = [max(x[2], 0.001) for x in top]
    max_score = max(scores)
    weights = [math.exp(math.log(s / max_score) / 0.8) for s in scores]
    total = sum(weights)
    weights = [w / total for w in weights]

    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return (top[i][0], top[i][1])

    return (top[-1][0], top[-1][1])


# ============================================================================ #
#                         GREEDY BUILD WITH DIVERSITY                          #
# ============================================================================ #

def greedy_build_diverse(
    anchor_cards: List[str],
    deck_words: Set[str],
    reqs: List[TargetReq],
    reverse_index: Dict[str, List[int]],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
    tracker: DiversityTracker,
    anchor_req_indices: Set[int] = None,
    target_size: int = 12,
    top_k: int = 10,
    temperature: float = 1.0,
    verbose: bool = False,
) -> List[str]:
    """Build deck with incremental scoring + diversity penalties."""
    if anchor_req_indices is None:
        anchor_req_indices = set()
    deck = set(anchor_cards)
    deck_list = list(anchor_cards)

    # Filter candidates
    allowed = deck_words.copy()
    allowed = {w for w in allowed if not (w.startswith('s') or w.endswith('s'))}
    allowed = {w for w in allowed if not has_adjacent_one_letter_words(w)}
    allowed = {w for w in allowed if w not in WORD_BLACKLIST}
    viable = (set(reverse_index.keys()) & allowed) - deck

    # Precompute fragment card sets
    left_frag_card_sets = {frag: set(cards) for frag, cards in left_frag_cards.items()}
    right_frag_card_sets = {frag: set(cards) for frag, cards in right_frag_cards.items()}

    # Compute initial missing_count for every req
    req_missing = [0] * len(reqs)
    req_left_frag_satisfied = [False] * len(reqs)
    req_right_frag_satisfied = [False] * len(reqs)

    for i, req in enumerate(reqs):
        missing = len(req.interior - deck)
        if req.left_frag:
            if deck & left_frag_card_sets.get(req.left_frag, set()):
                req_left_frag_satisfied[i] = True
            else:
                missing += 1
        if req.right_frag:
            if deck & right_frag_card_sets.get(req.right_frag, set()):
                req_right_frag_satisfied[i] = True
            else:
                missing += 1
        req_missing[i] = missing

    # Precompute card -> req roles
    card_req_roles = defaultdict(list)
    for card in viable | deck:
        for req_idx in reverse_index.get(card, []):
            req = reqs[req_idx]
            if card in req.interior:
                card_req_roles[card].append((req_idx, 'interior'))
            if req.left_frag and card in left_frag_card_sets.get(req.left_frag, set()):
                card_req_roles[card].append((req_idx, 'left_frag'))
            if req.right_frag and card in right_frag_card_sets.get(req.right_frag, set()):
                card_req_roles[card].append((req_idx, 'right_frag'))

    # Track enabled target words
    enabled_words = set()
    for i, req in enumerate(reqs):
        if req_missing[i] == 0:
            enabled_words.add(req.word)

    # Build req_to_candidates for dirty tracking
    req_to_candidates = defaultdict(set)
    for card in viable:
        for req_idx in reverse_index.get(card, []):
            req_to_candidates[req_idx].add(card)

    # Score a candidate using cached req_missing values + diversity
    def score_candidate(candidate: str) -> Tuple[float, List[str], int]:
        word_best = {}

        for req_idx, role in card_req_roles.get(candidate, []):
            req = reqs[req_idx]
            if req.word in enabled_words:
                continue

            current_missing = req_missing[req_idx]

            reduces = False
            if role == 'interior' and candidate not in deck:
                reduces = True
            elif role == 'left_frag' and not req_left_frag_satisfied[req_idx]:
                reduces = True
            elif role == 'right_frag' and not req_right_frag_satisfied[req_idx]:
                reduces = True

            if not reduces:
                continue

            missing_after = current_missing - 1

            anchor_mult = 20.0 if req_idx in anchor_req_indices else 1.0

            if missing_after == 0:
                contribution = req.score * 10.0 * anchor_mult
                is_completion = True
            elif missing_after == 1:
                contribution = req.score * 2.0 * anchor_mult
                is_completion = False
            elif missing_after == 2:
                contribution = req.score * 0.3 * anchor_mult
                is_completion = False
            else:
                continue

            # Apply made-word diversity discount to the target word
            mw_disc = tracker.made_word_discount(req.word)
            contribution *= mw_disc

            prev = word_best.get(req.word)
            if prev is None or contribution > prev[0]:
                word_best[req.word] = (contribution, is_completion, missing_after)

        score = 0.0
        newly_enabled = []
        near_complete = 0
        for word, (contribution, is_completion, missing_after) in word_best.items():
            score += contribution
            if is_completion:
                newly_enabled.append(word)
            if missing_after == 1:
                near_complete += 1

        card_freq = get_zipf(candidate)
        score *= max(0.3, min(1.0, card_freq / 5.0))

        # Apply card diversity discount
        score *= tracker.card_discount(candidate)

        return score, newly_enabled, near_complete

    # Initial scoring
    cached_scores = {}
    for candidate in viable:
        cached_scores[candidate] = score_candidate(candidate)

    while len(deck_list) < target_size:
        scored_candidates = []
        for candidate in viable:
            sc, ne, nc = cached_scores[candidate]
            if sc > 0:
                scored_candidates.append((candidate, sc, ne, nc))

        if not scored_candidates:
            if verbose:
                print("  No viable candidates left!")
            break

        chosen_card, chosen_score, chosen_enabled, chosen_near = sample_top_k(
            scored_candidates, k=top_k, temperature=temperature,
        )

        deck.add(chosen_card)
        deck_list.append(chosen_card)
        viable.discard(chosen_card)
        del cached_scores[chosen_card]

        # Update req_missing
        affected_reqs = set()
        for req_idx, role in card_req_roles.get(chosen_card, []):
            reduced = False
            if role == 'interior':
                req_missing[req_idx] -= 1
                reduced = True
            elif role == 'left_frag' and not req_left_frag_satisfied[req_idx]:
                req_left_frag_satisfied[req_idx] = True
                req_missing[req_idx] -= 1
                reduced = True
            elif role == 'right_frag' and not req_right_frag_satisfied[req_idx]:
                req_right_frag_satisfied[req_idx] = True
                req_missing[req_idx] -= 1
                reduced = True

            if reduced:
                affected_reqs.add(req_idx)

        new_enabled = set()
        for req_idx in affected_reqs:
            if req_missing[req_idx] == 0:
                new_enabled.add(reqs[req_idx].word)
        enabled_words.update(new_enabled)

        for req_idx in reverse_index.get(chosen_card, []):
            req_to_candidates[req_idx].discard(chosen_card)

        dirty = set()
        for req_idx in affected_reqs:
            dirty.update(req_to_candidates[req_idx])
        dirty &= viable

        for candidate in dirty:
            cached_scores[candidate] = score_candidate(candidate)

        if verbose:
            unique_new = sorted(set(chosen_enabled))
            freq = get_zipf(chosen_card)
            disc = tracker.card_discount(chosen_card)
            completions = f"  enables: {', '.join(unique_new)}" if unique_new else ""
            print(f"  {len(deck_list):2d}. {GREEN}+{chosen_card:<12}{RESET} score={chosen_score:8.1f} freq={freq:.1f} disc={disc:.2f}{completions}")

    return deck_list


# ============================================================================ #
#                         BATCH GENERATOR                                       #
# ============================================================================ #

def generate_batch(
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
    max_combo: int = 3,
    verbose: bool = False,
    output_file: str = "puzzles.json",
) -> List[dict]:
    """Generate a batch of diverse puzzles."""

    tracker = DiversityTracker(card_decay=card_decay, made_word_decay=made_word_decay)
    puzzles = []

    print(f"\n{'='*70}")
    print(f"GENERATING {num_puzzles} PUZZLES")
    print(f"{'='*70}")
    print(f"Puzzle size: {puzzle_size}, Top-K: {top_k}, Temp: {temperature}")
    print(f"Card decay: {card_decay}, Made-word decay: {made_word_decay}")
    print(f"Min anchor length: {min_anchor_length}")

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
        results = brute_force_combos(deck, max_combo_size=max_combo)

        # Collect made words
        made_words_set = set(results['made_words'].keys())

        # Count valid combos
        n_combos = results['total_valid']

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

        # Get longest made words for summary
        longest_made = sorted(made_words_set, key=lambda w: (-len(w), w))[:10]

        puzzle_record = {
            "id": puzzle_num,
            "cards": sorted(deck),
            "anchor_word": anchor_word,
            "anchor_decomposition": anchor_fmt,
            "anchor_interior_cards": list(anchor_req.decomp.interior),
            "num_valid_combos": n_combos,
            "num_made_words": len(made_words_set),
            "longest_made_words": longest_made,
            "enabled_target_words": enabled_targets[:20],
            "new_cards": new_cards,
            "new_made_words": new_made_words,
        }
        puzzles.append(puzzle_record)

        # Print summary line
        disc_anchor = tracker.anchor_discount(anchor_word)
        print(
            f"  {CYAN}#{puzzle_num:<4}{RESET} "
            f"anchor={anchor_word:<18} "
            f"combos={n_combos:4d} "
            f"made={len(made_words_set):4d} "
            f"new_cards={new_cards:2d}/{len(deck)} "
            f"new_mw={new_made_words:4d} "
            f"({t_elapsed:.1f}s)"
        )

        if verbose and longest_made:
            print(f"         longest: {', '.join(longest_made[:5])}")

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


def _save_results(puzzles: List[dict], tracker: DiversityTracker, output_file: str):
    """Save puzzles and diversity stats to JSON."""
    output = {
        "num_puzzles": len(puzzles),
        "diversity_stats": {
            "unique_cards_used": len(tracker.card_usage),
            "unique_made_words_seen": len(tracker.made_word_usage),
            "unique_anchors_used": len(tracker.anchor_word_usage),
            "most_used_cards": tracker.card_usage.most_common(20),
            "most_used_anchors": tracker.anchor_word_usage.most_common(20),
        },
        "puzzles": puzzles,
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {len(puzzles)} puzzles to {output_file}")


def _print_summary(puzzles: List[dict], tracker: DiversityTracker):
    """Print a summary of the batch generation."""
    if not puzzles:
        return

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Card usage distribution
    usage_vals = list(tracker.card_usage.values())
    print(f"\nCard usage distribution:")
    print(f"  Used once:  {sum(1 for v in usage_vals if v == 1)}")
    print(f"  Used 2-5:   {sum(1 for v in usage_vals if 2 <= v <= 5)}")
    print(f"  Used 6-10:  {sum(1 for v in usage_vals if 6 <= v <= 10)}")
    print(f"  Used 11-20: {sum(1 for v in usage_vals if 11 <= v <= 20)}")
    print(f"  Used 21+:   {sum(1 for v in usage_vals if v > 20)}")

    # Most used cards
    print(f"\nTop 20 most used cards:")
    for card, count in tracker.card_usage.most_common(20):
        freq = get_zipf(card)
        print(f"  {card:<15} used {count:4d} times (freq={freq:.1f})")

    # Anchor diversity
    print(f"\nAnchor word diversity:")
    print(f"  Unique anchors: {len(tracker.anchor_word_usage)}")
    anchor_usage = list(tracker.anchor_word_usage.values())
    if anchor_usage:
        print(f"  Anchors used once:  {sum(1 for v in anchor_usage if v == 1)}")
        print(f"  Anchors used 2+:    {sum(1 for v in anchor_usage if v >= 2)}")

    # Combo stats
    combo_counts = [p['num_valid_combos'] for p in puzzles]
    made_counts = [p['num_made_words'] for p in puzzles]
    new_card_counts = [p['new_cards'] for p in puzzles]
    print(f"\nPer-puzzle stats:")
    print(f"  Valid combos: min={min(combo_counts)}, max={max(combo_counts)}, avg={sum(combo_counts)/len(combo_counts):.1f}")
    print(f"  Made words:   min={min(made_counts)}, max={max(made_counts)}, avg={sum(made_counts)/len(made_counts):.1f}")
    print(f"  New cards:    min={min(new_card_counts)}, max={max(new_card_counts)}, avg={sum(new_card_counts)/len(new_card_counts):.1f}")


# ============================================================================ #
#                              MAIN                                            #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Generate a batch of diverse Proseset puzzles"
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
    parser.add_argument('--max-combo', type=int, default=3,
                        help='Max combo size for analysis (default: 3)')
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
    puzzles = generate_batch(
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
        max_combo=args.max_combo,
        verbose=args.verbose,
        output_file=args.output,
    )

    print(f"\nDone! {len(puzzles)} puzzles saved to {args.output}")


if __name__ == "__main__":
    main()
