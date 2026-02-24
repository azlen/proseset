#!/usr/bin/env python3
"""
Proseset Puzzle Builder v6 - puzzle6.py

Builds on puzzle5's incremental scoring, adding top-K sampling with temperature
to produce diverse puzzles instead of always converging to the same "universally
useful" cards like editing/there/men/develop.

Instead of argmax, each step samples from the top-K candidates weighted by score,
with a temperature parameter controlling diversity:
  temp=0  -> always pick the best (deterministic, like puzzle5)
  temp=1  -> sample proportional to score (moderate diversity)
  temp=2+ -> near-uniform from top-K (maximum diversity)
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, FrozenSet
from collections import defaultdict
import math
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
#                         TOP-K SAMPLING                                        #
# ============================================================================ #

def sample_top_k(
    candidates: List[Tuple[str, float, List[str], int]],
    k: int = 10,
    temperature: float = 1.0,
) -> Tuple[str, float, List[str], int]:
    """
    Sample from the top-K candidates weighted by score.

    candidates: list of (card, score, newly_enabled, near_complete)
    Returns the selected (card, score, newly_enabled, near_complete).
    """
    # Sort by score descending, take top K
    candidates.sort(key=lambda x: -x[1])
    top = candidates[:k]

    if temperature <= 0 or len(top) == 1:
        return top[0]

    # Compute sampling weights: score^(1/temperature)
    # Higher temperature -> flatter distribution
    scores = [max(x[1], 0.001) for x in top]
    max_score = max(scores)

    # Normalize scores to prevent overflow: divide by max before exponentiating
    weights = []
    for s in scores:
        log_w = math.log(s / max_score) / temperature
        weights.append(math.exp(log_w))

    total = sum(weights)
    weights = [w / total for w in weights]

    # Weighted random choice
    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return top[i]

    return top[-1]


# ============================================================================ #
#                         INCREMENTAL GREEDY WITH SAMPLING                      #
# ============================================================================ #

def greedy_build_sampled(
    anchor_cards: List[str],
    deck_words: Set[str],
    reqs: List[TargetReq],
    reverse_index: Dict[str, List[int]],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
    anchor_req_indices: Set[int] = None,
    target_size: int = 12,
    top_k: int = 10,
    temperature: float = 1.0,
) -> List[str]:
    """Build deck with incremental scoring + top-K sampling."""
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

    print(f"\nInitial deck ({len(deck_list)}): {', '.join(sorted(deck_list))}")
    print(f"Initially enabled target words: {len(enabled_words)}")
    print(f"Viable candidates: {len(viable)}")
    print(f"Sampling: top-{top_k}, temperature={temperature}")

    # Score a candidate using cached req_missing values
    def score_candidate_fast(candidate: str) -> Tuple[float, List[str], int]:
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

            # Anchor reqs get a big boost so their fragments get collapsed
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

        # Collect all candidates with positive scores
        scored_candidates = []
        for candidate in viable:
            sc, ne, nc = cached_scores[candidate]
            if sc > 0:
                scored_candidates.append((candidate, sc, ne, nc))

        if not scored_candidates:
            print("  No viable candidates left!")
            break

        # Sample from top-K
        chosen_card, chosen_score, chosen_enabled, chosen_near = sample_top_k(
            scored_candidates, k=top_k, temperature=temperature,
        )

        # Show what rank we picked (for debugging diversity)
        scored_candidates.sort(key=lambda x: -x[1])
        rank = next(i for i, (c, _, _, _) in enumerate(scored_candidates) if c == chosen_card) + 1

        # Add chosen card to deck
        deck.add(chosen_card)
        deck_list.append(chosen_card)
        viable.discard(chosen_card)
        del cached_scores[chosen_card]

        # Update req_missing for all reqs this card participates in
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

        # Update enabled words
        new_enabled = set()
        for req_idx in affected_reqs:
            if req_missing[req_idx] == 0:
                new_enabled.add(reqs[req_idx].word)
        enabled_words.update(new_enabled)

        # Remove chosen_card from req_to_candidates
        for req_idx in reverse_index.get(chosen_card, []):
            req_to_candidates[req_idx].discard(chosen_card)

        # Find dirty candidates and rescore
        dirty = set()
        for req_idx in affected_reqs:
            dirty.update(req_to_candidates[req_idx])
        dirty &= viable

        for candidate in dirty:
            cached_scores[candidate] = score_candidate_fast(candidate)

        t_end = time.time()

        # Print step info
        unique_new = sorted(set(chosen_enabled))
        freq = get_zipf(chosen_card)
        completions = f"  enables: {', '.join(unique_new)}" if unique_new else ""
        near_str = f"  ({chosen_near} near-complete)" if chosen_near else ""
        rank_str = f"  [rank {rank}/{len(scored_candidates)}]" if rank > 1 else ""
        print(f"  {len(deck_list):2d}. {GREEN}+{chosen_card:<12}{RESET} score={chosen_score:8.1f} freq={freq:.1f}{completions}{near_str}{rank_str}")

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
    parser = argparse.ArgumentParser(description="Constructive puzzle builder with top-K sampling")
    parser.add_argument('--anchors', type=int, default=1, help='Number of anchor decompositions (default: 1)')
    parser.add_argument('--min-length', type=int, default=10, help='Min target word length for anchors (default: 10)')
    parser.add_argument('--puzzle-size', type=int, default=12, help='Total puzzle size (default: 12)')
    parser.add_argument('--max-combo', type=int, default=3, help='Max combo size for brute-force analysis (default: 3)')
    parser.add_argument('--top-k', type=int, default=10, help='Top-K candidates to sample from (default: 10)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature: 0=greedy, 1=proportional, 2+=uniform (default: 1.0)')
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

        # Wave-collapse: only seed interior cards, leave fragments open
        interior = list(req.decomp.interior)
        open_frags = []
        if req.decomp.left_frag:
            n_providers = len(left_frag_cards.get(req.decomp.left_frag, []))
            open_frags.append(f"*{req.decomp.left_frag} ({n_providers} options)")
        if req.decomp.right_frag:
            n_providers = len(right_frag_cards.get(req.decomp.right_frag, []))
            open_frags.append(f"{req.decomp.right_frag}* ({n_providers} options)")

        print(f"    Interior cards: {', '.join(interior)}")
        if open_frags:
            print(f"    Open fragments: {', '.join(open_frags)}")

        all_anchor_cards.extend(interior)
        anchor_card_set.update(interior)

    # Find ALL req indices for anchor words (any decomposition), for score boosting
    anchor_words = {req.word for req, _ in anchors}
    anchor_req_indices = set()
    for i, req in enumerate(reqs):
        if req.word in anchor_words:
            anchor_req_indices.add(i)
    print(f"\n  Anchor words: {', '.join(sorted(anchor_words))}")
    print(f"  Anchor req indices: {len(anchor_req_indices)} (all decompositions)")

    # Greedy build with sampling
    print(f"\n{'='*60}")
    print("GREEDY DECK CONSTRUCTION (top-K sampling)")
    print(f"{'='*60}")

    deck = greedy_build_sampled(
        anchor_cards=all_anchor_cards,
        deck_words=deck_words,
        reqs=reqs,
        reverse_index=reverse_index,
        left_frag_cards=left_frag_cards,
        right_frag_cards=right_frag_cards,
        anchor_req_indices=anchor_req_indices,
        target_size=args.puzzle_size,
        top_k=args.top_k,
        temperature=args.temperature,
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
