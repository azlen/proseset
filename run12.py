#!/usr/bin/env python3
"""
Proseset Deck Generator - run12.py

Based on run11.py with parallelization:
- Parallelizes candidate scoring during deck building
- Parallelizes triplet enumeration for made words analysis
- Uses multiprocessing to speed up expensive operations

TRUE O(1) scoring via incremental counters + precomputed participation lists.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Set, Tuple
from wordfreq import top_n_list, zipf_frequency
import third_party.twl as twl
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
import hashlib
import os

# ============================================================================ #
#                              CONFIGURATION                                   #
# ============================================================================ #

N_WORDS = 50000
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

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def _enumerate_triplets_worker(args: Tuple[List[str], List[str], Lookups]) -> Tuple[int, Dict[str, int]]:
    """Worker function to enumerate triplets for a chunk of left words."""
    left_words_chunk, deck, lookups = args
    made_word_counts = {}
    total_triplets = 0

    for left_word in left_words_chunk:
        for mid_word in deck:
            if left_word == mid_word:
                continue

            # Check if left can attach to middle
            for t in lookups.templates_as_middle.get(mid_word, []):
                if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                    continue

                # Left can attach - now check for right words
                for right_word in deck:
                    if right_word == mid_word or right_word == left_word:
                        continue

                    if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                        continue

                    # Valid triplet! Compute the concatenation and all segmentations
                    concat = left_word + mid_word + right_word
                    segmentations = segment_word(concat, require_start=True, require_end=True)

                    if segmentations:
                        total_triplets += 1
                        # Track each segment in each segmentation
                        for seg in segmentations:
                            for word in seg:
                                made_word_counts[word] = made_word_counts.get(word, 0) + 1

    return (total_triplets, made_word_counts)

def progress(iterable, desc=""):
    return tqdm(iterable, desc=desc, ascii=" ▖▘▝▗▚▞█", bar_format='{desc}: |{bar:20}|')


def get_cache_key() -> str:
    """Generate cache key based on configuration."""
    config_str = f"{N_WORDS}_{MAX_WORD_LENGTH}_{MIN_WORD_LENGTH}_{sorted(ALLOWED_1_LETTER)}_{sorted(ALLOWED_2_LETTER)}_{PENALIZE_S_PREFIX}"
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f".cache_lookups_{cache_hash}.pkl"


def save_cache(cache_path: str, deck_words: Set[str], seg_words: Set[str], lookups: Lookups):
    """Save precomputed lookups to cache."""
    print(f"Saving cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump((deck_words, seg_words, lookups), f)
    print(f"Cache saved!")


def load_cache(cache_path: str) -> Tuple[Set[str], Set[str], Lookups]:
    """Load precomputed lookups from cache."""
    print(f"Loading cache from {cache_path}...")
    with open(cache_path, 'rb') as f:
        deck_words, seg_words, lookups = pickle.load(f)
    print(f"Cache loaded!")
    return deck_words, seg_words, lookups


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

    # Prefix/suffix indexes for finding long words
    words_by_prefix: Dict[str, Set[str]] = field(default_factory=dict)
    words_by_suffix: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class DeckState:
    """Incremental counters for O(1) scoring."""
    deck: Set[str] = field(default_factory=set)
    left_count: Dict[str, int] = field(default_factory=dict)   # key -> count of deck words that can attach left
    right_count: Dict[str, int] = field(default_factory=dict)  # key -> count of deck words that can attach right
    # For scoring as left/right: how many templates with this middle are "active" (middle in deck)
    middle_in_deck: Dict[str, bool] = field(default_factory=dict)

    # Made-word tracking
    made_word_counts: Dict[str, int] = field(default_factory=dict)  # Global counts of all made words
    word_contributions: Dict[str, Dict[str, int]] = field(default_factory=dict)  # Per-word: what made-words it contributes to


# ============================================================================ #
#                              BUILD LOOKUPS                                   #
# ============================================================================ #

def build_lookups(deck_words: Set[str], seg_words: Set[str]) -> Lookups:
    lookups = Lookups()

    # Build prefix/suffix indexes
    for word in seg_words:
        for i in range(1, len(word)):
            lookups.words_by_prefix.setdefault(word[:i], set()).add(word)
            lookups.words_by_suffix.setdefault(word[i:], set()).add(word)

    print(f"Built prefix index: {len(lookups.words_by_prefix)} entries")
    print(f"Built suffix index: {len(lookups.words_by_suffix)} entries")

    # Build left/right keys for each word
    for word in progress(deck_words, "Building attachment keys"):
        # Left attachment: LAST segment merges with middle's first
        # So last segment doesn't need to be valid (will merge), others must be valid
        for seg in segment_word(word, require_start=True, require_end=False):
            if not seg:
                continue
            last = seg[-1]
            for full_word in lookups.words_by_prefix.get(last, []):
                remainder = full_word[len(last):]
                if remainder:
                    lookups.word_left_keys.setdefault(word, set()).add(remainder)

        # Right attachment: FIRST segment merges with middle's last
        # So first segment doesn't need to be valid (will merge), others must be valid
        for seg in segment_word(word, require_start=False, require_end=True):
            if not seg:
                continue
            first = seg[0]
            for full_word in lookups.words_by_suffix.get(first, []):
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


def update_made_words_for_triplet(left: str, mid: str, right: str, state: DeckState, delta: int):
    """
    Update made-word counts for a specific triplet.
    delta = +1 when adding, -1 when removing
    """
    concat = left + mid + right
    segmentations = segment_word(concat, require_start=True, require_end=True)

    if segmentations:
        for seg in segmentations:
            for made_word in seg:
                # Update each participating word's contribution
                for word in [left, mid, right]:
                    if word not in state.word_contributions:
                        state.word_contributions[word] = {}
                    state.word_contributions[word][made_word] = \
                        state.word_contributions[word].get(made_word, 0) + delta
                    # Clean up zeros
                    if state.word_contributions[word][made_word] <= 0:
                        del state.word_contributions[word][made_word]

                # Update global count (only once per made-word, not per participating word)
                state.made_word_counts[made_word] = state.made_word_counts.get(made_word, 0) + delta
                if state.made_word_counts[made_word] <= 0:
                    del state.made_word_counts[made_word]


def add_word_to_state(word: str, state: DeckState, lookups: Lookups):
    if word in state.deck:
        return

    state.deck.add(word)
    state.middle_in_deck[word] = True

    for key in lookups.word_left_keys.get(word, set()):
        state.left_count[key] = state.left_count.get(key, 0) + 1

    for key in lookups.word_right_keys.get(word, set()):
        state.right_count[key] = state.right_count.get(key, 0) + 1

    # Update made-word tracking by enumerating all triplets involving this word
    deck_list = list(state.deck)

    # Word as MIDDLE
    for t in lookups.templates_as_middle.get(word, []):
        for left_word in deck_list:
            if left_word == word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            for right_word in deck_list:
                if right_word == left_word or right_word == word:
                    continue
                if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                    continue

                update_made_words_for_triplet(left_word, word, right_word, state, +1)

    # Word as LEFT
    for t in lookups.templates_as_left.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for right_word in deck_list:
            if right_word == t.middle_word or right_word == word:
                continue
            if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                continue

            update_made_words_for_triplet(word, t.middle_word, right_word, state, +1)

    # Word as RIGHT
    for t in lookups.templates_as_right.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for left_word in deck_list:
            if left_word == t.middle_word or left_word == word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            update_made_words_for_triplet(left_word, t.middle_word, word, state, +1)


def remove_word_from_state(word: str, state: DeckState, lookups: Lookups):
    if word not in state.deck:
        return

    # Update made-word tracking BEFORE removing from deck
    # (need deck intact to enumerate triplets)
    deck_list = list(state.deck)

    # Word as MIDDLE
    for t in lookups.templates_as_middle.get(word, []):
        for left_word in deck_list:
            if left_word == word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            for right_word in deck_list:
                if right_word == left_word or right_word == word:
                    continue
                if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                    continue

                update_made_words_for_triplet(left_word, word, right_word, state, -1)

    # Word as LEFT
    for t in lookups.templates_as_left.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for right_word in deck_list:
            if right_word == t.middle_word or right_word == word:
                continue
            if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                continue

            update_made_words_for_triplet(word, t.middle_word, right_word, state, -1)

    # Word as RIGHT
    for t in lookups.templates_as_right.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for left_word in deck_list:
            if left_word == t.middle_word or left_word == word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            update_made_words_for_triplet(left_word, t.middle_word, word, state, -1)

    # Clean up the removed word's contributions
    if word in state.word_contributions:
        del state.word_contributions[word]

    # Now remove from deck
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
    Includes both base combo score AND made-word score.
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

    # --- MADE-WORD CONTRIBUTION ---
    # Score the value of made-words this word contributes to
    made_word_score = 0.0
    if word in state.word_contributions:
        for made_word, count in state.word_contributions[word].items():
            # Length bonus
            length_score = (len(made_word) ** 1.5) * count

            # Rarity bonus based on how rare this made-word is
            global_count = state.made_word_counts.get(made_word, 0)
            if global_count <= 1:
                # Only this word produces it - very valuable!
                rarity_bonus = 100.0 * count
            elif global_count < 5:
                rarity_bonus = 50.0 * count / global_count
            elif global_count < 20:
                rarity_bonus = 20.0 * count / global_count
            elif global_count < 100:
                rarity_bonus = 5.0 * count / global_count
            else:
                rarity_bonus = 1.0 * count / global_count

            made_word_score += length_score + rarity_bonus

    # Scale down to balance with base scores
    total_score += made_word_score / 100.0

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
    # Bonus for potential long words based on first_key/last_key
    middle_long_word_bonus = 0.0
    for t in lookups.templates_as_middle.get(candidate, []):
        left = state.left_count.get(t.first_key, 0)
        right = state.right_count.get(t.last_key, 0)
        if left > 0 and right > 0:
            combos = left * right
            middle_combos += combos
            total_score += t.score * combos

            # Bonus based on longest possible words that could be formed
            # (even if we don't know if deck can produce the exact prefix/suffix)
            left_words = lookups.words_by_suffix.get(t.first_key, [])
            right_words = lookups.words_by_prefix.get(t.last_key, [])

            if left_words:
                max_left_len = max(len(w) for w in left_words)
                middle_long_word_bonus += (max_left_len ** 1.2) * left * 0.1

            if right_words:
                max_right_len = max(len(w) for w in right_words)
                middle_long_word_bonus += (max_right_len ** 1.2) * right * 0.1

    total_score += middle_long_word_bonus

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
#                              MADE WORD TRACKING                              #
# ============================================================================ #

def enumerate_word_triplets(word: str, state: DeckState, lookups: Lookups) -> Dict[str, int]:
    """
    Enumerate all triplets involving this word and return made-word counts.
    Returns: dict of made_word -> count
    """
    made_words = {}
    deck_list = list(state.deck)

    # Word as MIDDLE
    for t in lookups.templates_as_middle.get(word, []):
        for left_word in deck_list:
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            for right_word in deck_list:
                if right_word == left_word:
                    continue
                if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                    continue

                concat = left_word + word + right_word
                segmentations = segment_word(concat, require_start=True, require_end=True)

                if segmentations:
                    for seg in segmentations:
                        for made_word in seg:
                            made_words[made_word] = made_words.get(made_word, 0) + 1

    # Word as LEFT
    for t in lookups.templates_as_left.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for right_word in deck_list:
            if right_word == t.middle_word:
                continue
            if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                continue

            concat = word + t.middle_word + right_word
            segmentations = segment_word(concat, require_start=True, require_end=True)

            if segmentations:
                for seg in segmentations:
                    for made_word in seg:
                        made_words[made_word] = made_words.get(made_word, 0) + 1

    # Word as RIGHT
    for t in lookups.templates_as_right.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for left_word in deck_list:
            if left_word == t.middle_word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            concat = left_word + t.middle_word + word
            segmentations = segment_word(concat, require_start=True, require_end=True)

            if segmentations:
                for seg in segmentations:
                    for made_word in seg:
                        made_words[made_word] = made_words.get(made_word, 0) + 1

    return made_words


# ============================================================================ #
#                              MADE WORD SCORING                               #
# ============================================================================ #

def score_actual_made_words(candidate: str, state: DeckState, lookups: Lookups) -> float:
    """
    Score a candidate based on the actual made-words it would produce.
    Rewards words that create rare/unique made-words (low count in current deck).
    """
    candidate_made_words = enumerate_word_triplets(candidate, state, lookups)

    if not candidate_made_words:
        return 0.0

    total_score = 0.0

    for made_word, count_added in candidate_made_words.items():
        # Current count of this made-word in deck (0 if new)
        current_count = state.made_word_counts.get(made_word, 0)

        # Length bonus: longer words are more valuable (exponential)
        length_score = (len(made_word) ** 1.5) * count_added

        # Rarity bonus: new or rare made-words are more valuable
        # Inverse relationship with current count
        if current_count == 0:
            # Brand new made-word! Big bonus
            rarity_bonus = 100.0 * count_added
        elif current_count < 5:
            # Rare - high bonus
            rarity_bonus = 50.0 * count_added / (current_count + 1)
        elif current_count < 20:
            # Uncommon - moderate bonus
            rarity_bonus = 20.0 * count_added / (current_count + 1)
        elif current_count < 100:
            # Common - small bonus
            rarity_bonus = 5.0 * count_added / (current_count + 1)
        else:
            # Very common - minimal bonus
            rarity_bonus = 1.0 * count_added / (current_count + 1)

        total_score += length_score + rarity_bonus

    # Diversity bonus: producing many different made-words
    diversity_bonus = len(candidate_made_words) * 10.0
    total_score += diversity_bonus

    # Scale down to balance with base scores
    return total_score / 100.0


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
    refinement_iterations = 0
    max_refinement = 100
    last_removed = None
    last_added = None
    iteration = 0
    word_added_at = {w: 0 for w in seed_words}  # Track when each word was added
    PROTECTION_WINDOW = 7  # Words can't be removed for this many iterations after being added

    while len(deck_list) < target_size or (len(deck_list) >= target_size and refinement_iterations < max_refinement):
        iteration += 1

        # Find best word to add
        # Step 1: Fast scoring to get top candidates
        candidate_scores = []
        for candidate in viable:
            if candidate in state.deck or candidate in dropped:
                continue
            score, combos, breakdown = compute_marginal_value(candidate, state, lookups)
            if score > 0:
                candidate_scores.append((candidate, score, combos, breakdown))

        if not candidate_scores:
            print(f"No viable candidates")
            break

        # Step 2: Take top 20 and refine with made-word scoring
        candidate_scores.sort(key=lambda x: -x[1])
        top_candidates = candidate_scores[:20]

        best_word = None
        best_total_score = -1
        best_combos = 0
        best_breakdown = (0, 0, 0)
        best_base_score = 0
        best_made_word_bonus = 0

        for candidate, base_score, combos, breakdown in top_candidates:
            # Compute actual made-word bonus
            made_word_bonus = score_actual_made_words(candidate, state, lookups)
            total_score = base_score + made_word_bonus

            if total_score > best_total_score:
                best_total_score = total_score
                best_word = candidate
                best_combos = combos
                best_breakdown = breakdown
                best_base_score = base_score
                best_made_word_bonus = made_word_bonus

        if best_word is None or best_total_score <= 0:
            print(f"No viable candidates")
            break

        # Detect cycling: trying to add back what we just removed
        if len(deck_list) >= target_size and best_word == last_removed:
            print(f"     Refinement complete: cycling detected (would re-add {best_word})")
            break

        add_word_to_state(best_word, state, lookups)
        deck_list.append(best_word)
        viable.discard(best_word)
        word_added_at[best_word] = iteration  # Track when this word was added
        last_added = best_word
        additions_since_prune += 1

        l, m, r = best_breakdown
        freq = get_zipf(best_word)

        if len(deck_list) < target_size:
            print(f"{len(deck_list):3d}. {GREEN}+{best_word:<12}{RESET} ~{best_combos:4d} combos (L={l:3d} M={m:3d} R={r:3d}) freq={freq:.1f} base={best_base_score:.0f} +made={best_made_word_bonus:.0f}")
        else:
            if refinement_iterations == 0:
                print(f"\n{'='*60}")
                print("REFINEMENT PHASE: swapping to improve deck quality")
                print(f"{'='*60}")
            refinement_iterations += 1
            print(f"  Swap {refinement_iterations:3d}: {GREEN}+{best_word:<12}{RESET} (L={l:3d} M={m:3d} R={r:3d}) freq={freq:.1f} base={best_base_score:.0f} +made={best_made_word_bonus:.0f}")

            # Stop if we've hit the max refinement iterations
            if refinement_iterations >= max_refinement:
                print(f"\n     Maximum refinement iterations ({max_refinement}) reached")
                break

        # Pruning logic: 2:1 ratio before target, 1:1 after target
        prune_threshold = 1 if len(deck_list) >= target_size else 2
        should_prune = prune and additions_since_prune >= prune_threshold and len(deck_list) > len(seed_words) + 2

        if should_prune:
            # Find lowest contributor
            worst_word = None
            worst_score = float('inf')
            worst_combos = 0

            for word in deck_list:
                if word in protected:
                    continue
                # Skip recently added words (within protection window) - only during building phase
                if len(deck_list) < target_size and iteration - word_added_at.get(word, 0) < PROTECTION_WINDOW:
                    continue
                score, combos = compute_word_contribution(word, state, lookups)
                if score < worst_score:
                    worst_score = score
                    worst_word = word
                    worst_combos = combos

            if worst_word:
                # Check for cycling in refinement phase - are we removing what we just added?
                if len(deck_list) >= target_size and worst_word == last_added:
                    print(f"     Refinement complete: cycling detected (would remove {worst_word} which was just added)")
                    break

                remove_word_from_state(worst_word, state, lookups)
                deck_list.remove(worst_word)
                dropped.add(worst_word)  # Don't re-add immediately
                last_removed = worst_word
                if worst_word in word_added_at:
                    del word_added_at[worst_word]

                if len(deck_list) < target_size:
                    print(f"     {RED}-{worst_word:<12}{RESET} (contributed {worst_combos} combos, total_score={worst_score:.0f})")
                else:
                    print(f"                  {RED}-{worst_word:<12}{RESET} (total_score={worst_score:.0f})")

            additions_since_prune = 0

    return deck_list, state


# ============================================================================ #
#                              ANALYSIS                                        #
# ============================================================================ #

def analyze_deck(deck: List[str], lookups: Lookups, state: DeckState = None):
    if state is None:
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

    # Use the made-word counts already tracked during deck building
    print(f"\n{'='*60}")
    print("MADE WORDS ANALYSIS (from incremental tracking)")
    print(f"{'='*60}")

    made_word_counts = state.made_word_counts
    print(f"Total unique made words: {len(made_word_counts)}")

    if made_word_counts:
        # Sort by length (longest)
        by_length = sorted(made_word_counts.items(), key=lambda x: (-len(x[0]), -x[1]))
        print(f"\nLongest made words:")
        for word, count in by_length[:30]:
            print(f"  {word:<20} (len={len(word):2d}) appears in {count:6d} triplets")

        # Sort by count (most common)
        by_count = sorted(made_word_counts.items(), key=lambda x: -x[1])
        print(f"\nMost common made words:")
        for word, count in by_count[:30]:
            print(f"  {word:<20} appears in {count:6d} triplets")

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
    cache_path = get_cache_key()

    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            deck_words, seg_words, lookups = load_cache(cache_path)
            set_segmentation_dict(seg_words)
            print(f"Deck words: {len(deck_words)}, Segmentation words: {len(seg_words)}")
        except Exception as e:
            print(f"Cache load failed: {e}, rebuilding...")
            deck_words, seg_words, lookups = None, None, None
    else:
        deck_words, seg_words, lookups = None, None, None

    # Build if not cached
    if lookups is None:
        print("Loading dictionary...")
        deck_words, seg_words = load_dictionary()
        print(f"Deck words: {len(deck_words)}, Segmentation words: {len(seg_words)}")

        set_segmentation_dict(seg_words)

        print("\nBuilding lookups...")
        lookups = build_lookups(deck_words, seg_words)

        # Save to cache
        save_cache(cache_path, deck_words, seg_words, lookups)

    print("\nTop 20 words by template score:")
    for word, n, score, freq in rank_words(deck_words, lookups, 20):
        print(f"  {word:<12} templates={n:<3} score={score:.4f} freq={freq:.2f}")

    seeds = ['area', 'able']

    print(f"\n{'='*60}")
    print("BUILDING DECK")
    print(f"{'='*60}")

    deck, state = build_deck_greedy(seeds, 200, deck_words, lookups)
    analyze_deck(deck, lookups, state)


if __name__ == "__main__":
    main()
