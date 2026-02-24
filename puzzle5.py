#!/usr/bin/env python3
"""
Proseset Puzzle Builder v5 - puzzle5.py

Same constructive greedy approach as puzzle4, but with fully incremental scoring.

Key optimization: instead of recomputing cards_missing() from scratch for each
candidate on each step, we maintain a cached missing_count per req. When a card
is added to the deck, we decrement missing_count for all reqs it participates in,
then only rescore the affected candidates by summing their cached req contributions.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, FrozenSet
from collections import defaultdict
import random
import time

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

from puzzle3 import brute_force_combos, display_analysis

from puzzle4 import (
    TargetReq, score_target, build_target_reqs,
    build_reverse_index, resolve_cards, select_anchors,
)

GREEN = '\033[92m'
RESET = '\033[0m'


# ============================================================================ #
#                         INCREMENTAL GREEDY CONSTRUCTION                       #
# ============================================================================ #

def greedy_build_incremental(
    anchor_cards: List[str],
    deck_words: Set[str],
    reqs: List[TargetReq],
    reverse_index: Dict[str, List[int]],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
    target_size: int = 12,
) -> List[str]:
    """Build deck greedily with fully incremental missing-count tracking."""
    deck = set(anchor_cards)
    deck_list = list(anchor_cards)

    # Filter candidates
    allowed = deck_words.copy()
    allowed = {w for w in allowed if not (w.startswith('s') or w.endswith('s'))}
    allowed = {w for w in allowed if not has_adjacent_one_letter_words(w)}
    allowed = {w for w in allowed if w not in WORD_BLACKLIST}
    viable = (set(reverse_index.keys()) & allowed) - deck

    # Precompute which cards satisfy which fragments
    # left_frag -> set of cards that provide it, right_frag -> set of cards that provide it
    left_frag_card_sets = {frag: set(cards) for frag, cards in left_frag_cards.items()}
    right_frag_card_sets = {frag: set(cards) for frag, cards in right_frag_cards.items()}

    # Compute initial missing_count for every req
    # Also build: for each card, which reqs does it reduce missing for when added?
    # A card reduces missing for a req if:
    #   - it's an interior card of that req, OR
    #   - it provides the left_frag, OR
    #   - it provides the right_frag
    # But for fragments: only the FIRST card providing that fragment reduces missing
    # (subsequent ones are redundant). We track this with frag_satisfied flags.

    req_missing = [0] * len(reqs)  # cached missing count per req
    req_left_frag_satisfied = [False] * len(reqs)  # is left frag covered by deck?
    req_right_frag_satisfied = [False] * len(reqs)  # is right frag covered by deck?

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

    # For each candidate, precompute which reqs it would reduce missing for
    # card_req_role[card] = list of (req_idx, role)
    # role: 'interior' | 'left_frag' | 'right_frag'
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

    print(f"\nInitial deck ({len(deck_list)}): {', '.join(sorted(deck_list))}")
    print(f"Initially enabled target words: {len(enabled_words)}")
    print(f"Viable candidates: {len(viable)}")

    # Score a candidate using cached req_missing values
    def score_candidate_fast(candidate: str) -> Tuple[float, List[str], int]:
        word_best = {}  # word -> (contribution, is_completion, missing_after)

        for req_idx, role in card_req_roles.get(candidate, []):
            req = reqs[req_idx]
            if req.word in enabled_words:
                continue

            current_missing = req_missing[req_idx]

            # Would this card reduce missing?
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

            if missing_after == 0:
                contribution = req.score * 10.0
                is_completion = True
            elif missing_after == 1:
                contribution = req.score * 2.0
                is_completion = False
            elif missing_after == 2:
                contribution = req.score * 0.3
                is_completion = False
            else:
                continue

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

        return score, newly_enabled, near_complete

    # Initial scoring
    t0 = time.time()
    cached_scores = {}
    for candidate in viable:
        cached_scores[candidate] = score_candidate_fast(candidate)
    t1 = time.time()
    print(f"Initial scoring: {t1-t0:.2f}s")

    while len(deck_list) < target_size:
        t_step = time.time()

        # Pick best candidate
        best_card = None
        best_score = -1
        best_newly_enabled = []
        best_near_complete = 0

        for candidate in viable:
            sc, ne, nc = cached_scores[candidate]
            if sc > best_score:
                best_score = sc
                best_card = candidate
                best_newly_enabled = ne
                best_near_complete = nc

        if best_card is None or best_score <= 0:
            print("  No viable candidates left!")
            break

        # Add best card to deck
        deck.add(best_card)
        deck_list.append(best_card)
        viable.discard(best_card)
        del cached_scores[best_card]

        # Update req_missing for all reqs this card participates in
        affected_reqs = set()
        for req_idx, role in card_req_roles.get(best_card, []):
            reduced = False
            if role == 'interior':
                # This card is an interior card of this req and just entered deck
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

        # Update enabled words
        new_enabled = set()
        for req_idx in affected_reqs:
            if req_missing[req_idx] == 0:
                new_enabled.add(reqs[req_idx].word)
        enabled_words.update(new_enabled)

        # Remove best_card from req_to_candidates
        for req_idx in reverse_index.get(best_card, []):
            req_to_candidates[req_idx].discard(best_card)

        # Find dirty candidates: those sharing any affected req
        dirty = set()
        for req_idx in affected_reqs:
            dirty.update(req_to_candidates[req_idx])
        dirty &= viable

        # Rescore only dirty candidates
        for candidate in dirty:
            cached_scores[candidate] = score_candidate_fast(candidate)

        t_end = time.time()

        # Print step info
        unique_new = sorted(set(best_newly_enabled))
        freq = get_zipf(best_card)
        completions = f"  enables: {', '.join(unique_new)}" if unique_new else ""
        near_str = f"  ({best_near_complete} near-complete)" if best_near_complete else ""
        dirty_str = f"  [rescored {len(dirty)}/{len(viable)} in {(t_end-t_step)*1000:.0f}ms]"
        print(f"  {len(deck_list):2d}. {GREEN}+{best_card:<12}{RESET} score={best_score:8.1f} freq={freq:.1f}{completions}{near_str}{dirty_str}")

    print(f"\n{'='*60}")
    print(f"DECK COMPLETE ({len(deck_list)} cards)")
    print(f"{'='*60}")
    print(f"Cards: {', '.join(sorted(deck_list))}")
    print(f"Total enabled target words: {len(enabled_words)}")

    if enabled_words:
        by_length = sorted(enabled_words, key=lambda w: (-len(w), w))
        print(f"\nEnabled target words ({len(enabled_words)}):")
        for w in by_length:
            best_req = None
            for i, req in enumerate(reqs):
                if req.word == w and req_missing[i] == 0:
                    if best_req is None or req.score > best_req.score:
                        best_req = req
            if best_req:
                fmt = format_anchor(w, best_req.decomp, left_frag_cards, right_frag_cards)
                print(f"  {w:<20} ({len(w):2d}) {fmt}")

    return deck_list


# ============================================================================ #
#                              MAIN                                            #
# ============================================================================ #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Constructive puzzle builder with incremental rescoring")
    parser.add_argument('--anchors', type=int, default=1, help='Number of anchor decompositions (default: 1)')
    parser.add_argument('--min-length', type=int, default=10, help='Min target word length for anchors (default: 10)')
    parser.add_argument('--puzzle-size', type=int, default=12, help='Total puzzle size (default: 12)')
    parser.add_argument('--max-combo', type=int, default=3, help='Max combo size for brute-force analysis (default: 3)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
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

    # Select anchor
    print(f"\n{'='*60}")
    print("SELECTING ANCHOR")
    print(f"{'='*60}")

    anchors = select_anchors(
        reqs, deck_words, left_frag_cards, right_frag_cards,
        num_anchors=args.anchors,
        min_length=args.min_length,
    )

    if not anchors:
        print("Failed to select anchor!")
        return

    all_anchor_cards = []
    anchor_card_set = set()
    for i, (req, cards) in enumerate(anchors):
        fmt = format_anchor(req.word, req.decomp, left_frag_cards, right_frag_cards)
        print(f"\n  Anchor {i+1}: {req.word} ({len(req.word)} chars, score={req.score:.2f})")
        print(f"    Decomposition: {fmt}")
        print(f"    Cards: {', '.join(cards)}")
        all_anchor_cards.extend(cards)
        anchor_card_set.update(cards)

    # Greedy build
    print(f"\n{'='*60}")
    print("GREEDY DECK CONSTRUCTION (incremental)")
    print(f"{'='*60}")

    deck = greedy_build_incremental(
        anchor_cards=all_anchor_cards,
        deck_words=deck_words,
        reqs=reqs,
        reverse_index=reverse_index,
        left_frag_cards=left_frag_cards,
        right_frag_cards=right_frag_cards,
        target_size=args.puzzle_size,
    )

    # Brute-force analysis with broad TWL dictionary
    broad_seg_words = (
        ALLOWED_1_LETTER | ALLOWED_2_LETTER |
        {w for w in twl.iterator() if w.isalpha() and len(w) > 2}
    ) - BANNED_WORDS - WORD_BLACKLIST
    set_segmentation_dict(broad_seg_words)
    print(f"\nSwitched to broad TWL dictionary: {len(broad_seg_words)} seg words")

    print(f"\n{'='*60}")
    print(f"BRUTE-FORCE ANALYSIS (max combo size: {args.max_combo})")
    print(f"{'='*60}")

    results = brute_force_combos(deck, max_combo_size=args.max_combo)
    display_analysis(deck, results, anchor_cards=anchor_card_set)


if __name__ == "__main__":
    main()
