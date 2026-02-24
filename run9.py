#!/usr/bin/env python3
"""
Proseset Deck Generator - run9.py

Based on run8.py with smarter 's' handling:
- Words are only banned if 's' can appear as a single-letter segment at start or end
- e.g., "wants" -> want|s (banned), "slider" -> s|lid|er (banned)
- But "storm" is fine since 's' can't be isolated

TRUE O(1) scoring via incremental counters + precomputed participation lists.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Set, Tuple
from wordfreq import top_n_list, zipf_frequency
import third_party.twl as twl
from tqdm import tqdm

# ============================================================================ #
#                              CONFIGURATION                                   #
# ============================================================================ #

N_WORDS = 20000
MAX_WORD_LENGTH = 7
MIN_WORD_LENGTH = 3

ALLOWED_1_LETTER = {'a', 'i'}
ALLOWED_2_LETTER = {
    'ah', 'am', 'an', 'as', 'at', 'ax', 'be', 'by', 'do', 'eh', 'ex',
    'go', 'he', 'if', 'in', 'is', 'it', 'me', 'no', 'of', 'oh', 'on',
    'or', 'ow', 'ox', 'pi', 'up', 'us', 'we', 'ya', 'yo'
}

PENALIZE_S_PREFIX = False  # Replaced by smarter 's' segmentation check


# ============================================================================ #
#                              HELPERS                                         #
# ============================================================================ #

def progress(iterable, desc=""):
    return tqdm(iterable, desc=desc, ascii=" ▖▘▝▗▚▞█", bar_format='{desc}: |{bar:20}|')


@lru_cache(maxsize=None)
def get_zipf(word: str) -> float:
    return zipf_frequency(word, 'en')


def score_segment(seg: str) -> float:
    """Score segment by frequency. Multiplied together, obscure = low score."""
    freq = get_zipf(seg)
    if freq < 1:
        return 0.01
    elif freq < 2:
        return 0.05
    elif freq < 3:
        return 0.15
    elif freq < 4:
        return 0.4
    else:
        return min(1.0, freq / 6)


# ============================================================================ #
#                              DICTIONARY                                      #
# ============================================================================ #

def load_dictionary() -> Tuple[Set[str], Set[str]]:
    top_words = set(top_n_list('en', N_WORDS, wordlist='best'))
    # Only use curated 1-2 letter words, not all TWL 2-letter words (like 'aa', 'xu')
    seg_words = (
        ALLOWED_1_LETTER | ALLOWED_2_LETTER |
        {w for w in top_words if twl.check(w) and w.isalpha() and len(w) > 2}
    )
    deck_words = {w for w in seg_words if MIN_WORD_LENGTH <= len(w) <= MAX_WORD_LENGTH}
    return deck_words, seg_words


# ============================================================================ #
#                              SEGMENTATION                                    #
# ============================================================================ #

_seg_words: Set[str] = set()

def set_segmentation_dict(words: Set[str]):
    global _seg_words
    _seg_words = words
    _segment_word_cached.cache_clear()


@lru_cache(maxsize=None)
def _segment_word_cached(word: str, require_start: bool, require_end: bool) -> Tuple[Tuple[str, ...], ...]:
    if not word:
        return ((),)
    results = []
    for i in range(1, len(word) + 1):
        prefix = word[:i]
        suffix = word[i:]
        prefix_ok = (prefix in _seg_words) or (not require_start) or (i == len(word) and not require_end)
        if prefix_ok:
            for tail in _segment_word_cached(suffix, True, require_end):
                results.append((prefix,) + tail)
    return tuple(results)


def segment_word(word: str, *, require_start: bool = False, require_end: bool = False) -> List[Tuple[str, ...]]:
    return list(_segment_word_cached(word, require_start, require_end))


def has_isolated_s(word: str) -> bool:
    """
    Check if word can segment with 's' alone at start or end.
    e.g., "wants" -> want|s (True), "slider" -> s|lid|er (True), "storm" (False)
    """
    # Check for 's' at start: word starts with 's' and rest is valid
    if word.startswith('s') and word[1:] in _seg_words:
        return True

    # Check for 's' at end: word ends with 's' and prefix is valid
    if word.endswith('s') and word[:-1] in _seg_words:
        return True

    return False


# ============================================================================ #
#                              DATA STRUCTURES                                 #
# ============================================================================ #

@dataclass
class Template:
    """A segmentation template for combos."""
    middle_word: str
    segments: Tuple[str, ...]
    first_key: str   # Left attachment key
    last_key: str    # Right attachment key
    score: float     # Precomputed quality score


@dataclass
class Lookups:
    # word_left_keys[word] = set of keys this word can attach LEFT at
    word_left_keys: Dict[str, Set[str]] = field(default_factory=dict)
    # word_right_keys[word] = set of keys this word can attach RIGHT at
    word_right_keys: Dict[str, Set[str]] = field(default_factory=dict)

    # Templates where word is MIDDLE
    templates_as_middle: Dict[str, List[Template]] = field(default_factory=dict)
    # Templates where word can be LEFT (grouped by middle_word for deck checking)
    templates_as_left: Dict[str, List[Template]] = field(default_factory=dict)
    # Templates where word can be RIGHT
    templates_as_right: Dict[str, List[Template]] = field(default_factory=dict)

    # All keys that have any templates
    valid_first_keys: Set[str] = field(default_factory=set)
    valid_last_keys: Set[str] = field(default_factory=set)


@dataclass
class DeckState:
    """Incremental counters for O(1) scoring."""
    deck: Set[str] = field(default_factory=set)
    left_count: Dict[str, int] = field(default_factory=dict)   # key -> count of deck words that can attach left
    right_count: Dict[str, int] = field(default_factory=dict)  # key -> count of deck words that can attach right
    # For scoring as left/right: how many templates with this middle are "active" (middle in deck)
    middle_in_deck: Dict[str, bool] = field(default_factory=dict)


# ============================================================================ #
#                              BUILD LOOKUPS                                   #
# ============================================================================ #

def build_lookups(deck_words: Set[str], seg_words: Set[str]) -> Lookups:
    lookups = Lookups()

    # Build prefix/suffix indexes
    words_by_prefix: Dict[str, Set[str]] = {}
    words_by_suffix: Dict[str, Set[str]] = {}
    for word in seg_words:
        for i in range(1, len(word)):
            words_by_prefix.setdefault(word[:i], set()).add(word)
            words_by_suffix.setdefault(word[i:], set()).add(word)

    print(f"Built prefix index: {len(words_by_prefix)} entries")
    print(f"Built suffix index: {len(words_by_suffix)} entries")

    # Build left/right keys for each word
    for word in progress(deck_words, "Building attachment keys"):
        # Left attachment: LAST segment merges with middle's first
        # So last segment doesn't need to be valid (will merge), others must be valid
        for seg in segment_word(word, require_start=True, require_end=False):
            if not seg:
                continue
            last = seg[-1]
            for full_word in words_by_prefix.get(last, []):
                remainder = full_word[len(last):]
                if remainder:
                    lookups.word_left_keys.setdefault(word, set()).add(remainder)

        # Right attachment: FIRST segment merges with middle's last
        # So first segment doesn't need to be valid (will merge), others must be valid
        for seg in segment_word(word, require_start=False, require_end=True):
            if not seg:
                continue
            first = seg[0]
            for full_word in words_by_suffix.get(first, []):
                remainder = full_word[:-len(first)]
                if remainder:
                    lookups.word_right_keys.setdefault(word, set()).add(remainder)

    print(f"Words with left keys: {len(lookups.word_left_keys)}")
    print(f"Words with right keys: {len(lookups.word_right_keys)}")

    # Collect all valid keys (keys that at least one word can attach at)
    all_left_keys: Set[str] = set()
    all_right_keys: Set[str] = set()
    for keys in lookups.word_left_keys.values():
        all_left_keys.update(keys)
    for keys in lookups.word_right_keys.values():
        all_right_keys.update(keys)

    # Build templates
    all_templates: List[Template] = []

    for word in progress(deck_words, "Building templates"):
        # For middle words: first and last segments MERGE, so don't need to be valid
        # Only middle segments (between first and last) must be valid
        for seg in segment_word(word, require_start=False, require_end=False):
            if len(seg) < 2:
                continue

            # Check that middle segments (if any) are valid words
            middle_segments = seg[1:-1]
            if not all(s in _seg_words for s in middle_segments):
                continue

            first_key = seg[0]
            last_key = seg[-1]

            # Must have potential attachments
            if first_key not in all_left_keys or last_key not in all_right_keys:
                continue

            # Score: product of segment frequencies * word frequency
            score = get_zipf(word)
            for s in seg:
                score *= score_segment(s)

            template = Template(
                middle_word=word,
                segments=seg,
                first_key=first_key,
                last_key=last_key,
                score=score,
            )
            all_templates.append(template)

            lookups.templates_as_middle.setdefault(word, []).append(template)
            lookups.valid_first_keys.add(first_key)
            lookups.valid_last_keys.add(last_key)

    print(f"Built {len(all_templates)} templates for {len(lookups.templates_as_middle)} words")

    # Build templates_as_left and templates_as_right
    # For each word, find templates where it can participate
    for word in progress(deck_words, "Building participation lists"):
        left_keys = lookups.word_left_keys.get(word, set())
        right_keys = lookups.word_right_keys.get(word, set())

        # Templates where this word can be LEFT
        for template in all_templates:
            if template.middle_word == word:
                continue
            if template.first_key in left_keys:
                lookups.templates_as_left.setdefault(word, []).append(template)
            if template.last_key in right_keys:
                lookups.templates_as_right.setdefault(word, []).append(template)

    print(f"Words that can be left: {len(lookups.templates_as_left)}")
    print(f"Words that can be right: {len(lookups.templates_as_right)}")

    return lookups


# ============================================================================ #
#                              DECK STATE                                      #
# ============================================================================ #

def create_deck_state(seed_words: List[str], lookups: Lookups) -> DeckState:
    state = DeckState()
    for word in seed_words:
        add_word_to_state(word, state, lookups)
    return state


def add_word_to_state(word: str, state: DeckState, lookups: Lookups):
    if word in state.deck:
        return

    state.deck.add(word)
    state.middle_in_deck[word] = True

    for key in lookups.word_left_keys.get(word, set()):
        state.left_count[key] = state.left_count.get(key, 0) + 1

    for key in lookups.word_right_keys.get(word, set()):
        state.right_count[key] = state.right_count.get(key, 0) + 1


def remove_word_from_state(word: str, state: DeckState, lookups: Lookups):
    if word not in state.deck:
        return

    state.deck.remove(word)
    state.middle_in_deck[word] = False

    for key in lookups.word_left_keys.get(word, set()):
        state.left_count[key] = state.left_count.get(key, 0) - 1

    for key in lookups.word_right_keys.get(word, set()):
        state.right_count[key] = state.right_count.get(key, 0) - 1


def compute_word_contribution(word: str, state: DeckState, lookups: Lookups) -> Tuple[float, int]:
    """
    Compute how much a word currently IN the deck contributes.
    This is the value that would be LOST if we removed it.
    """
    total_score = 0.0
    total_combos = 0

    # --- As MIDDLE ---
    for t in lookups.templates_as_middle.get(word, []):
        left = state.left_count.get(t.first_key, 0)
        right = state.right_count.get(t.last_key, 0)
        if left > 0 and right > 0:
            combos = left * right
            total_combos += combos
            total_score += t.score * combos

    # --- As LEFT ---
    for t in lookups.templates_as_left.get(word, []):
        if t.middle_word not in state.deck:
            continue
        right = state.right_count.get(t.last_key, 0)
        if right > 0:
            total_combos += right
            total_score += t.score * right

    # --- As RIGHT ---
    for t in lookups.templates_as_right.get(word, []):
        if t.middle_word not in state.deck:
            continue
        left = state.left_count.get(t.first_key, 0)
        if left > 0:
            total_combos += left
            total_score += t.score * left

    # Apply same frequency weighting (cap at 1.0, penalize below zipf 5)
    word_freq = get_zipf(word)
    if word_freq >= 5.0:
        freq_weight = 1.0
    else:
        freq_weight = 10 ** (word_freq - 5.0)
    total_score *= freq_weight

    return total_score, total_combos


# ============================================================================ #
#                              O(1) SCORING                                    #
# ============================================================================ #

def compute_marginal_value(candidate: str, state: DeckState, lookups: Lookups) -> Tuple[float, int, Tuple[int, int, int]]:
    """
    Compute marginal value of adding candidate.

    As MIDDLE: sum over candidate's templates of score * left_count * right_count
    As LEFT: sum over templates_as_left where middle is in deck
    As RIGHT: sum over templates_as_right where middle is in deck

    Returns: (score, total_combos, (left_combos, middle_combos, right_combos))
    """
    total_score = 0.0
    left_combos = 0
    middle_combos = 0
    right_combos = 0

    # --- As MIDDLE ---
    for t in lookups.templates_as_middle.get(candidate, []):
        left = state.left_count.get(t.first_key, 0)
        right = state.right_count.get(t.last_key, 0)
        if left > 0 and right > 0:
            combos = left * right
            middle_combos += combos
            total_score += t.score * combos

    # --- As LEFT ---
    for t in lookups.templates_as_left.get(candidate, []):
        if t.middle_word not in state.deck:
            continue
        right = state.right_count.get(t.last_key, 0)
        if right > 0:
            left_combos += right
            total_score += t.score * right

    # --- As RIGHT ---
    for t in lookups.templates_as_right.get(candidate, []):
        if t.middle_word not in state.deck:
            continue
        left = state.left_count.get(t.first_key, 0)
        if left > 0:
            right_combos += left
            total_score += t.score * left

    total_combos = left_combos + middle_combos + right_combos

    # === WORD RECOGNIZABILITY ===
    # Penalize uncommon words, but don't boost super common ones
    # Cap at 1.0 so top ~5000 words (zipf >= 5) are treated equally
    word_freq = get_zipf(candidate)
    if word_freq >= 5.0:
        freq_weight = 1.0  # Common words: no adjustment
    else:
        freq_weight = 10 ** (word_freq - 5.0)  # zipf 4 → 0.1x, zipf 3 → 0.01x
    total_score *= freq_weight

    # === BALANCE BONUS/PENALTY ===
    # Heavily penalize words missing any position, reward balanced distribution
    positions = [left_combos, middle_combos, right_combos]
    zeros = positions.count(0)

    if zeros >= 2:
        total_score *= 0.01  # Almost useless - only works in one position
    elif zeros == 1:
        total_score *= 0.1   # Heavy penalty - missing a position
    else:
        # All positions have combos - reward balance
        min_pos = min(positions)
        max_pos = max(positions)
        balance_ratio = min_pos / max_pos  # 0 to 1, higher = more balanced
        total_score *= (1 + balance_ratio * 2)  # 1x to 3x multiplier for balance

    return total_score, total_combos, (left_combos, middle_combos, right_combos)


# ============================================================================ #
#                              DECK BUILDING                                   #
# ============================================================================ #

def build_deck_greedy(
    seed_words: List[str],
    target_size: int,
    deck_words: Set[str],
    lookups: Lookups,
    prune: bool = True,
) -> List[str]:
    state = create_deck_state(seed_words, lookups)
    deck_list = list(seed_words)
    protected = set()  # No special protection for seeds

    candidates = deck_words - state.deck

    # Filter out words that can segment with 's' alone at start or end
    candidates = {w for w in candidates if not has_isolated_s(w)}

    # Only viable if has templates
    viable = {w for w in candidates if w in lookups.templates_as_middle}
    dropped = set()  # Track dropped words so we don't re-add them immediately

    print(f"\nSeeds: {seed_words}")
    print(f"Viable candidates: {len(viable)}")

    additions_since_prune = 0

    while len(deck_list) < target_size:
        # Find best word to add
        best_word = None
        best_score = -1
        best_combos = 0
        best_breakdown = (0, 0, 0)

        for candidate in viable:
            if candidate in state.deck or candidate in dropped:
                continue
            score, combos, breakdown = compute_marginal_value(candidate, state, lookups)
            if score > best_score:
                best_score = score
                best_word = candidate
                best_combos = combos
                best_breakdown = breakdown

        if best_word is None or best_score <= 0:
            print(f"No viable candidates")
            break

        add_word_to_state(best_word, state, lookups)
        deck_list.append(best_word)
        viable.discard(best_word)
        additions_since_prune += 1

        l, m, r = best_breakdown
        freq = get_zipf(best_word)
        print(f"{len(deck_list):3d}. +{best_word:<12} ~{best_combos:4d} combos (L={l:3d} M={m:3d} R={r:3d}) freq={freq:.1f}")

        # Every 2 additions, drop the lowest contributor (if pruning enabled)
        if prune and additions_since_prune >= 2 and len(deck_list) > len(seed_words) + 2:
            # Find lowest contributor
            worst_word = None
            worst_score = float('inf')
            worst_combos = 0

            for word in deck_list:
                if word in protected:
                    continue
                score, combos = compute_word_contribution(word, state, lookups)
                if score < worst_score:
                    worst_score = score
                    worst_word = word
                    worst_combos = combos

            if worst_word:
                remove_word_from_state(worst_word, state, lookups)
                deck_list.remove(worst_word)
                dropped.add(worst_word)  # Don't re-add immediately
                print(f"     -{worst_word:<12} (contributed {worst_combos} combos, score={worst_score:.1f})")

            additions_since_prune = 0

            # Clear dropped set periodically to allow reconsidering old words
            if len(dropped) > 20:
                dropped.clear()

    return deck_list


# ============================================================================ #
#                              ANALYSIS                                        #
# ============================================================================ #

def analyze_deck(deck: List[str], lookups: Lookups):
    state = create_deck_state(deck, lookups)

    print(f"\n{'='*60}")
    print(f"DECK ({len(deck)} words)")
    print(f"{'='*60}")
    print(f"Words: {', '.join(deck)}")

    # Count combos per word (as middle, left, right)
    word_combos: Dict[str, Dict[str, int]] = {w: {"middle": 0, "left": 0, "right": 0} for w in deck}

    total_combos = 0
    for mid in deck:
        for t in lookups.templates_as_middle.get(mid, []):
            left_count = state.left_count.get(t.first_key, 0)
            right_count = state.right_count.get(t.last_key, 0)
            combos = left_count * right_count
            total_combos += combos
            word_combos[mid]["middle"] += combos

            # Attribute to left/right words too
            for w in deck:
                if w == mid:
                    continue
                if t.first_key in lookups.word_left_keys.get(w, set()):
                    word_combos[w]["left"] += right_count  # this word pairs with all rights
                if t.last_key in lookups.word_right_keys.get(w, set()):
                    word_combos[w]["right"] += left_count  # this word pairs with all lefts

    print(f"\nTotal combos: {total_combos}")

    # Show per-word breakdown sorted by total participation
    word_totals = [(w, c["left"] + c["middle"] + c["right"], c) for w, c in word_combos.items()]
    word_totals.sort(key=lambda x: -x[1])

    print(f"\nPer-word combo participation (L=left, M=middle, R=right):")
    for i, (w, total, c) in enumerate(word_totals):
        if i < 20 or total > 0:
            print(f"  {w:<12} total={total:5d}  (L={c['left']:4d} M={c['middle']:4d} R={c['right']:4d})")

    # To show actual combos, we need to find the segmentations
    print(f"\nSample combos:")
    shown = 0
    for mid in deck[:30]:
        for t in lookups.templates_as_middle.get(mid, [])[:5]:
            # Find a left word with matching segmentation
            left_word = None
            left_seg = None
            for w in deck:
                if w == mid:
                    continue
                # Check if w can be left for this template
                for seg in segment_word(w, require_start=False, require_end=True):
                    if seg and seg[-1] + t.first_key in _seg_words:
                        # This seg's last + middle's first = valid word
                        # But we need to check if the REMAINDER matches
                        # The key is: seg[-1] is a prefix of some word, and that word minus seg[-1] = t.first_key
                        pass
                # Simpler: just check if t.first_key is in word's left_keys
                if t.first_key in lookups.word_left_keys.get(w, set()):
                    left_word = w
                    # Left: last segment merges, others must be valid
                    for seg in segment_word(w, require_start=True, require_end=False):
                        if seg:
                            # Check: seg[-1] + t.first_key should form a valid word
                            merged_seg = seg[-1] + t.first_key
                            if merged_seg in _seg_words:
                                left_seg = seg
                                break
                    if left_seg:
                        break

            if not left_word:
                continue

            # Find a right word
            right_word = None
            right_seg = None
            for w in deck:
                if w == mid or w == left_word:
                    continue
                if t.last_key in lookups.word_right_keys.get(w, set()):
                    right_word = w
                    # Right: first segment merges, others must be valid
                    for seg in segment_word(w, require_start=False, require_end=True):
                        if seg:
                            merged_seg = t.last_key + seg[0]
                            if merged_seg in _seg_words:
                                right_seg = seg
                                break
                    if right_seg:
                        break

            if left_word and right_word and left_seg and right_seg:
                # Build the merged output
                mid_seg = t.segments
                merged = list(left_seg[:-1])
                merged.append(left_seg[-1] + mid_seg[0])  # merge point 1
                merged.extend(mid_seg[1:-1])
                merged.append(mid_seg[-1] + right_seg[0])  # merge point 2
                merged.extend(right_seg[1:])

                concat = left_word + mid + right_word
                print(f"  {left_word} + {mid} + {right_word} = {concat}")
                print(f"    -> {' | '.join(merged)}")
                shown += 1
                if shown >= 15:
                    break
        if shown >= 15:
            break


def rank_words(deck_words: Set[str], lookups: Lookups, top_n: int = 50):
    results = []
    for word in deck_words:
        if PENALIZE_S_PREFIX and word.startswith('s'):
            continue
        templates = lookups.templates_as_middle.get(word, [])
        if not templates:
            continue
        total_score = sum(t.score for t in templates)
        results.append((word, len(templates), total_score, get_zipf(word)))
    results.sort(key=lambda x: -x[2])
    return results[:top_n]


# ============================================================================ #
#                              MAIN                                            #
# ============================================================================ #

def main():
    print("Loading dictionary...")
    deck_words, seg_words = load_dictionary()
    print(f"Deck words: {len(deck_words)}, Segmentation words: {len(seg_words)}")

    set_segmentation_dict(seg_words)

    print("\nBuilding lookups...")
    lookups = build_lookups(deck_words, seg_words)

    print("\nTop 20 words by template score:")
    for word, n, score, freq in rank_words(deck_words, lookups, 20):
        print(f"  {word:<12} templates={n:<3} score={score:.4f} freq={freq:.2f}")

    seeds = ['area', 'able']

    print(f"\n{'='*60}")
    print("BUILDING DECK")
    print(f"{'='*60}")

    deck = build_deck_greedy(seeds, 150, deck_words, lookups)
    analyze_deck(deck, lookups)


if __name__ == "__main__":
    main()
