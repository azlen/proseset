#!/usr/bin/env python3
"""
Proseset Puzzle Builder v4 - puzzle4.py

Constructive puzzle builder using decomposition reverse-indexes.
No swaps — builds the deck card by card, choosing each card to
maximize the number of interesting target words that become formable.

Algorithm:
1. Pick an anchor target word (long, interesting decomposition)
2. Seed deck with its required cards
3. For each remaining slot, greedily add the card that enables
   the most/best additional target words (given cards already in deck)
4. Run brute-force analysis on final deck

Key data structures:
- TargetReq: a specific decomposition of a target word, tracking
  which interior cards and edge fragments are needed
- reverse_index: card -> list of TargetReqs it participates in
  (as interior card or fragment provider)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from collections import defaultdict
from wordfreq import zipf_frequency
import random
import sys
import os

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

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


# ============================================================================ #
#                         TARGET TRACKING                                      #
# ============================================================================ #

@dataclass
class TargetReq:
    """One way a target word can be formed — tracks what cards are needed."""
    word: str
    decomp: Decomposition
    interior: FrozenSet[str]   # Interior cards needed
    left_frag: str             # "" if none
    right_frag: str            # "" if none
    num_cards: int             # Total cards needed
    score: float               # Quality score

    def cards_missing(self, deck: Set[str],
                      left_frag_cards: Dict[str, List[str]],
                      right_frag_cards: Dict[str, List[str]]) -> int:
        missing = len(self.interior - deck)
        if self.left_frag and not any(c in deck for c in left_frag_cards.get(self.left_frag, [])):
            missing += 1
        if self.right_frag and not any(c in deck for c in right_frag_cards.get(self.right_frag, [])):
            missing += 1
        return missing

    def is_enabled(self, deck: Set[str],
                   left_frag_cards: Dict[str, List[str]],
                   right_frag_cards: Dict[str, List[str]]) -> bool:
        return self.cards_missing(deck, left_frag_cards, right_frag_cards) == 0


def score_target(word: str, decomp: Decomposition) -> float:
    """Score how interesting/valuable a target word is."""
    freq = get_zipf(word)
    length_bonus = len(word) ** 1.75
    freq_bonus = max(0.1, min(1.0, freq / 5.0))
    piece_penalty = 0.8 ** max(0, len(decomp.all_pieces()) - 3)
    pure_bonus = 1.3 if decomp.is_pure else 1.0
    return length_bonus * freq_bonus * piece_penalty * pure_bonus


# ============================================================================ #
#                         INDEX BUILDING                                       #
# ============================================================================ #

def build_target_reqs(
    table: Dict[str, List[Decomposition]],
    deck_words: Set[str],
    max_decomps_per_word: int = 5,
    max_cards: int = 7,
) -> List[TargetReq]:
    """Build TargetReq objects for all viable decompositions."""
    reqs = []

    for word, decomps in table.items():
        scored = []
        for d in decomps:
            if not all(c in deck_words for c in d.interior):
                continue
            if len(set(d.interior)) != len(d.interior):
                continue
            num_cards = d.num_cards
            if num_cards > max_cards:
                continue
            sc = score_target(word, d)
            scored.append((d, sc, num_cards))

        scored.sort(key=lambda x: -x[1])
        for d, sc, num_cards in scored[:max_decomps_per_word]:
            reqs.append(TargetReq(
                word=word,
                decomp=d,
                interior=frozenset(d.interior),
                left_frag=d.left_frag,
                right_frag=d.right_frag,
                num_cards=num_cards,
                score=sc,
            ))

    return reqs


def build_reverse_index(
    reqs: List[TargetReq],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
) -> Dict[str, List[int]]:
    """Build card -> list of req indices (all reqs this card is relevant to)."""
    index = defaultdict(list)

    for i, req in enumerate(reqs):
        for card in req.interior:
            index[card].append(i)
        if req.left_frag:
            for card in left_frag_cards.get(req.left_frag, []):
                index[card].append(i)
        if req.right_frag:
            for card in right_frag_cards.get(req.right_frag, []):
                index[card].append(i)

    return dict(index)


# ============================================================================ #
#                         ANCHOR SELECTION                                     #
# ============================================================================ #

def resolve_cards(
    decomp: Decomposition,
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
) -> List[str]:
    """Resolve a decomposition to actual card words."""
    cards = []
    if decomp.left_frag:
        frag_cards = left_frag_cards.get(decomp.left_frag, [])
        if not frag_cards:
            return []
        cards.append(best_extension_card(frag_cards, decomp.left_frag, frag_is_prefix=False))
    cards.extend(decomp.interior)
    if decomp.right_frag:
        frag_cards = right_frag_cards.get(decomp.right_frag, [])
        if not frag_cards:
            return []
        cards.append(best_extension_card(frag_cards, decomp.right_frag, frag_is_prefix=True))
    return cards


def select_anchors(
    reqs: List[TargetReq],
    deck_words: Set[str],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
    num_anchors: int = 1,
    min_length: int = 10,
    extended_only: bool = True,
) -> List[Tuple[TargetReq, List[str]]]:
    """Select random anchor(s) from top-scoring candidates."""
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

        candidates.append((req, cards))

    if not candidates:
        return []

    # Sort by score, pick randomly from top 50
    candidates.sort(key=lambda x: -x[0].score)
    top = candidates[:50]

    selected = []
    used_cards = set()
    random.shuffle(top)

    for req, cards in top:
        if len(selected) >= num_anchors:
            break
        if used_cards & set(cards):
            continue
        selected.append((req, cards))
        used_cards.update(cards)

    return selected


# ============================================================================ #
#                         GREEDY CONSTRUCTION                                  #
# ============================================================================ #

def greedy_build(
    anchor_cards: List[str],
    deck_words: Set[str],
    reqs: List[TargetReq],
    reverse_index: Dict[str, List[int]],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
    target_size: int = 12,
) -> List[str]:
    """Build deck greedily, adding the card that enables the most target words."""
    deck = set(anchor_cards)
    deck_list = list(anchor_cards)

    # Filter candidates
    allowed = deck_words.copy()
    allowed = {w for w in allowed if not (w.startswith('s') or w.endswith('s'))}
    allowed = {w for w in allowed if not has_adjacent_one_letter_words(w)}
    allowed = {w for w in allowed if w not in WORD_BLACKLIST}

    # Only consider cards that appear in the reverse index (others have 0 value)
    viable = (set(reverse_index.keys()) & allowed) - deck

    # Track enabled target words
    enabled_words = set()
    for req in reqs:
        if req.is_enabled(deck, left_frag_cards, right_frag_cards):
            enabled_words.add(req.word)

    print(f"\nInitial deck ({len(deck_list)}): {', '.join(sorted(deck_list))}")
    print(f"Initially enabled target words: {len(enabled_words)}")
    print(f"Viable candidates: {len(viable)}")

    while len(deck_list) < target_size:
        best_card = None
        best_score = -1
        best_newly_enabled = []
        best_near_complete = 0

        for candidate in viable:
            if candidate in deck:
                continue

            new_deck = deck | {candidate}
            score = 0.0
            newly_enabled = []
            near_complete = 0

            # Deduplicate: only count the best decomposition per target word
            word_best_contribution = {}  # word -> best score contribution

            for req_idx in reverse_index.get(candidate, []):
                req = reqs[req_idx]

                if req.word in enabled_words:
                    continue

                missing_before = req.cards_missing(deck, left_frag_cards, right_frag_cards)
                missing_after = req.cards_missing(new_deck, left_frag_cards, right_frag_cards)

                # Only credit if this card actually reduces missing count
                if missing_after >= missing_before:
                    continue

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

                prev = word_best_contribution.get(req.word)
                if prev is None or contribution > prev[0]:
                    word_best_contribution[req.word] = (contribution, is_completion, missing_after)

            for word, (contribution, is_completion, missing_after) in word_best_contribution.items():
                score += contribution
                if is_completion:
                    newly_enabled.append(word)
                if missing_after == 1:
                    near_complete += 1

            # Card recognizability
            card_freq = get_zipf(candidate)
            score *= max(0.3, min(1.0, card_freq / 5.0))

            if score > best_score:
                best_score = score
                best_card = candidate
                best_newly_enabled = newly_enabled
                best_near_complete = near_complete

        if best_card is None or best_score <= 0:
            print("  No viable candidates left!")
            break

        deck.add(best_card)
        deck_list.append(best_card)
        viable.discard(best_card)

        # Update enabled words
        new_enabled = set()
        for req_idx in reverse_index.get(best_card, []):
            req = reqs[req_idx]
            if req.word not in enabled_words and req.is_enabled(deck, left_frag_cards, right_frag_cards):
                new_enabled.add(req.word)
        enabled_words.update(new_enabled)

        # Deduplicate newly_enabled (same word from multiple decomps)
        unique_new = sorted(set(best_newly_enabled))
        freq = get_zipf(best_card)
        completions = f"  enables: {', '.join(unique_new)}" if unique_new else ""
        near_str = f"  ({best_near_complete} near-complete)" if best_near_complete else ""
        print(f"  {len(deck_list):2d}. {GREEN}+{best_card:<12}{RESET} score={best_score:8.1f} freq={freq:.1f}{completions}{near_str}")

    print(f"\n{'='*60}")
    print(f"DECK COMPLETE ({len(deck_list)} cards)")
    print(f"{'='*60}")
    print(f"Cards: {', '.join(sorted(deck_list))}")
    print(f"Total enabled target words: {len(enabled_words)}")

    if enabled_words:
        by_length = sorted(enabled_words, key=lambda w: (-len(w), w))
        print(f"\nEnabled target words ({len(enabled_words)}):")
        for w in by_length:
            # Find the best decomposition that's actually enabled
            best_req = None
            for req in reqs:
                if req.word == w and req.is_enabled(deck, left_frag_cards, right_frag_cards):
                    if best_req is None or req.score > best_req.score:
                        best_req = req
            if best_req:
                cards = resolve_cards(best_req.decomp, left_frag_cards, right_frag_cards)
                # Show which cards from the deck are actually used
                deck_cards = [c for c in cards if c in deck]
                fmt = format_anchor(w, best_req.decomp, left_frag_cards, right_frag_cards)
                print(f"  {w:<20} ({len(w):2d}) {fmt}")

    return deck_list


# ============================================================================ #
#                              MAIN                                            #
# ============================================================================ #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Constructive puzzle builder using decomposition lookups")
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
    print("GREEDY DECK CONSTRUCTION")
    print(f"{'='*60}")

    deck = greedy_build(
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
