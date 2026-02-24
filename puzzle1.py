#!/usr/bin/env python3
"""
Proseset Puzzle Optimizer - puzzle1.py

Optimizes a fixed set of 12 words for gameplay:
- Starts with 12 random words
- Swaps words to optimize for long/interesting/unique made-words
- Ensures all words can exist in at least 3 triplets
- Max 50 swap iterations
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
MIN_WORD_LENGTH = 2

ALLOWED_1_LETTER = {'a', 'i'}
ALLOWED_2_LETTER = {
    'ah', 'am', 'an', 'as', 'at', 'ax', 'be', 'by', 'do', 'eh', 'ex',
    'go', 'he', 'if', 'in', 'is', 'it', 'me', 'no', 'of', 'oh', 'on',
    'or', 'ow', 'ox', 'pi', 'up', 'us', 'we', 'ya', 'yo'
}

# Banned words - Obscure Scrabble words that normal people wouldn't recognize
BANNED_WORDS = {
    # Obscure 2-letter words (technical terms, foreign words, letter spellings)
    'aa', 'ae', 'ag', 'ai', 'al', 'ar', 'aw', 'ay', 'ba', 'bi', 'bo',
    'de', 'ed', 'ef', 'el', 'em', 'en', 'er', 'es', 'et', 'ew', 'fa', 'fe', 'gi',
    'gu', 'hm', 'ho', 'id', 'jo', 'ka', 'ki', 'la', 'li', 'lo', 'mi', 'mm', 'mo',
    'mu', 'na', 'ne', 'nu', 'od', 'oe', 'oi', 'ok', 'om', 'op', 'os', 'pe', 'po', 'qi',
    're', 'sh', 'si', 'ta', 'te', 'ti', 'un', 'ut', 'wo', 'xi', 'xu', 'ya', 'ye', 'za',
    # Obscure 3-letter words (technical/archaic/very uncommon)
    'aas', 'aba', 'abo', 'abs', 'aby', 'adz', 'aga', 'ahi', 'ahs', 'ais', 'ala',
    'alb', 'alf', 'als', 'ama', 'ami', 'amu', 'ana', 'ane', 'ani', 'apo',
    'arf', 'ars', 'asp', 'att', 'auk', 'ava', 'avo', 'awa', 'awn',
    'ays', 'azo', 'baa', 'bal', 'bap', 'bas', 'bel', 'ben', 'bes', 'bey',
    'bis', 'bop', 'bot', 'brr', 'bub', 'bur', 'bys',
    'caw', 'cay', 'cee', 'cel', 'cep', 'cis', 'col', 'cor', 'cos', 'coz', 'cru',
    'cwm', 'dag', 'dah', 'dak', 'dal', 'dap', 'daw', 'deb', 'dee', 'dex', 'dey',
    'dib', 'dif', 'dis', 'dit', 'dol', 'dom', 'dor', 'dos', 'dow', 'dup',
    'edh', 'efs', 'ehs', 'eme', 'ems', 'eng', 'ens', 'erg', 'ern', 'ers', 'ess', 'eta', 'eth',
    'fah', 'fas', 'fay', 'fer', 'fes', 'fet', 'feu', 'fey', 'foh', 'fon', 'fou', 'foy', 'fub', 'fud', 'fug',
    'gad', 'gam', 'gan', 'gar', 'gat', 'ged', 'ghi', 'gib', 'gid', 'gie', 'gip', 'goa', 'gor', 'gox',
    'hae', 'haj', 'hao', 'hap', 'hep', 'hes', 'het', 'hic', 'hie', 'hin', 'hob', 'hod', 'hoy', 'hyp',
    'ich', 'iff', 'jee', 'jeu', 'jin', 'jow',
    'kab', 'kae', 'kaf', 'kas', 'kat', 'kea', 'kef', 'kep', 'kex', 'khi', 'kif', 'kip', 'kir', 'kis', 'koa', 'kob', 'kop', 'kor', 'kos', 'kue', 'kye',
    'lac', 'lam', 'lar', 'las', 'lat', 'lav', 'lea', 'lek', 'leu', 'lev', 'lex', 'lez', 'lin', 'lis', 'lux',
    'mae', 'mam', 'mas', 'meg', 'mel', 'mem', 'mew', 'mho', 'mib', 'mig', 'mil', 'mim', 'mir', 'mis', 'moa', 'moc', 'mon', 'mor', 'mos', 'mot', 'mun', 'mus', 'mut', 'myc',
    'nae', 'nam', 'naw', 'neb', 'neg', 'nim', 'nog', 'noh', 'noo', 'nos', 'nth', 'nus',
    'oba', 'obe', 'obi', 'oca', 'och', 'oda', 'ods', 'oes', 'oho', 'ohs', 'oik', 'oka', 'oke', 'olf', 'oms', 'ono', 'ons', 'oot', 'ope', 'ops', 'ora', 'ors', 'ort', 'ose', 'oud', 'owt', 'oxo', 'oxy', 'oye', 'oyo', 'oys',
    'pac', 'pah', 'pam', 'pap', 'pas', 'pax', 'pec', 'ped', 'peh', 'pes', 'pht', 'pia', 'pis', 'piu', 'pix', 'plu', 'poh', 'pom', 'pul', 'pya', 'pye', 'pyx',
    'qat', 'qis', 'qua',
    'rai', 'raj', 'ras', 'rax', 'reb', 'rec', 'ree', 'rei', 'ret', 'rex', 'ria', 'rif', 'rin', 'rom', 'rya',
    'sab', 'sac', 'sae', 'sal', 'sau', 'seg', 'sei', 'sel', 'sen', 'ser', 'sha', 'shh', 'sib', 'sim', 'soh', 'som', 'sot', 'sou', 'sri', 'suk', 'suq', 'syn',
    'tae', 'taj', 'tam', 'tao', 'tas', 'tat', 'tav', 'taw', 'tec', 'ted', 'teg', 'tel', 'tes', 'tet', 'tew', 'tho', 'tig', 'tis', 'tix', 'tod', 'tui', 'tum', 'tun', 'tup', 'twa', 'tye',
    'udo', 'ugs', 'ule', 'ulu', 'ums', 'uns', 'upo', 'urb', 'urd', 'urp', 'urs', 'uta', 'ute', 'uts',
    'vas', 'vau', 'vav', 'vaw', 'vee', 'vig', 'voe', 'vox', 'vug', 'vum',
    'wab', 'wae', 'wap', 'wat', 'waw', 'wha', 'wis', 'wog', 'wop', 'wos', 'wot', 'wud', 'wyn',
    'xis',
    'yag', 'yah', 'yar', 'yaw', 'yeh', 'yid', 'yob', 'yod', 'yok', 'yom', 'yon', 'yow',
    'zas', 'zax', 'zek', 'zep', 'zho', 'zin', 'zoa', 'zow', 'zuz',

    'martin', 'alan', 'tit', 'tor', 'ling', 'amin', 'cur'
}

# Word blacklist - Words that CANNOT appear in deck OR be formed in any triplet
# This is more extreme than BANNED_WORDS - we reject deck candidates if they could form these words
WORD_BLACKLIST = {
    # Add offensive, problematic, or unwanted words here
    # Example: 'nazi', 'rape', etc.
    "nazi", "rape", "raper", "rapist", "ass", "dick", "fuck", "molest", "anal", "assault", "sex"
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
    # Remove banned words (weird Scrabble words that aren't normal)
    seg_words = seg_words - BANNED_WORDS
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
    [UNUSED] Check if word can segment with 's' alone at start or end.
    e.g., "wants" -> want|s (True), "slider" -> s|lid|er (True), "storm" (False)
    Currently replaced with simple startswith('s') check.
    """
    # Check for 's' at start: word starts with 's' and rest is valid
    if word.startswith('s') and word[1:] in _seg_words:
        return True

    # Check for 's' at end: word ends with 's' and prefix is valid
    if word.endswith('s') and word[:-1] in _seg_words:
        return True

    return False


def has_adjacent_one_letter_words(word: str) -> bool:
    """
    Check if word contains 'ai' or 'ia' substrings.
    These can be isolated as two adjacent 1-letter words (a, i) which is not fun.
    e.g., "rain" -> r|a|i|n, "dial" -> d|i|a|l
    """
    return 'ai' in word or 'ia' in word


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

    # Segmentation cache: concat_string -> list of valid alternate segmentations
    # Pre-computed during build_lookups to avoid repeated segment_word calls
    concat_segmentations: Dict[str, List[Tuple[str, ...]]] = field(default_factory=dict)


@dataclass
class DeckState:
    """Incremental counters for O(1) scoring."""
    deck: Set[str] = field(default_factory=set)
    left_count: Dict[str, int] = field(default_factory=dict)   # key -> count of deck words that can attach left
    right_count: Dict[str, int] = field(default_factory=dict)  # key -> count of deck words that can attach right
    # For scoring as left/right: how many templates with this middle are "active" (middle in deck)
    middle_in_deck: Dict[str, bool] = field(default_factory=dict)

    # Made-word tracking - track unique triplet recipes, not occurrence counts
    made_word_recipes: Dict[str, Set[Tuple[str, str, str]]] = field(default_factory=dict)  # made_word -> set of (left, mid, right) triplets
    word_contributions: Dict[str, Dict[str, int]] = field(default_factory=dict)  # Per-word: what made-words it contributes to (still count for scoring)

    # Connectivity tracking - which words can each word connect with
    word_connections: Dict[str, Set[str]] = field(default_factory=dict)  # word -> set of other words it can form triplets with


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


def get_segmentations_cached(concat: str, lookups: Lookups) -> List[Tuple[str, ...]]:
    """
    Get segmentations for a concatenation, using cache when available.
    Cache is stored in lookups.concat_segmentations.
    """
    # Handle old cached lookups without concat_segmentations field
    if not hasattr(lookups, 'concat_segmentations'):
        lookups.concat_segmentations = {}

    if concat not in lookups.concat_segmentations:
        # Compute and cache
        lookups.concat_segmentations[concat] = segment_word(concat, require_start=True, require_end=True)
    return lookups.concat_segmentations[concat]


def find_contributing_words(made_word: str, concat: str, left: str, mid: str, right: str) -> Tuple[str, ...]:
    """
    Determine which of the three source words actually contribute letters to the made_word.
    Returns a tuple of contributing words (could be 1, 2, or 3 words).
    """
    # Find where made_word appears in concat
    start_pos = concat.find(made_word)
    if start_pos == -1:
        # Made word doesn't appear as substring, must be formed by merging
        # In this case, all three words could contribute, return all
        return (left, mid, right)

    end_pos = start_pos + len(made_word)

    # Determine which source words overlap with [start_pos, end_pos)
    left_end = len(left)
    mid_end = left_end + len(mid)
    right_end = mid_end + len(right)

    contributors = []
    if start_pos < left_end and end_pos > 0:
        contributors.append(left)
    if start_pos < mid_end and end_pos > left_end:
        contributors.append(mid)
    if start_pos < right_end and end_pos > mid_end:
        contributors.append(right)

    return tuple(contributors) if contributors else (left, mid, right)


def update_made_words_for_triplet(left: str, mid: str, right: str, state: DeckState, lookups: Lookups, delta: int):
    """
    Update made-word recipes and connections for a specific triplet.
    delta = +1 when adding, -1 when removing
    """
    concat = left + mid + right
    segmentations = get_segmentations_cached(concat, lookups)

    # Track connections between words (they can form a valid triplet together)
    if segmentations:  # Only count as connection if it produces valid segmentations
        for word in [left, mid, right]:
            if word not in state.word_connections:
                state.word_connections[word] = set()
            if delta > 0:
                # Adding connections
                state.word_connections[word].update([w for w in [left, mid, right] if w != word])
            # Note: we don't remove connections on delta < 0 because a word might still connect
            # via other triplets. We'd need to track all triplets per connection to do that properly.

        # Original triplet boundary positions (internal boundaries only, not start/end)
        original_boundaries = {len(left), len(left) + len(mid)}

        for seg in segmentations:
            # Calculate boundary positions for this segmentation
            seg_boundaries = set()
            pos = 0
            for word in seg[:-1]:  # All words except the last (last word has no boundary after it)
                pos += len(word)
                seg_boundaries.add(pos)

            # Skip if ANY boundary position matches the original
            if seg_boundaries & original_boundaries:  # Set intersection
                continue

            # This is a valid alternate segmentation!
            # Now process each made-word in it
            pos = 0
            for made_word in seg:
                start = pos
                end = pos + len(made_word)

                # Skip words that have the same character span as original words
                # (these aren't "made" - they're just the original deck words)
                if made_word == left and start == 0 and end == len(left):
                    pos = end
                    continue
                if made_word == mid and start == len(left) and end == len(left) + len(mid):
                    pos = end
                    continue
                if made_word == right and start == len(left) + len(mid) and end == len(concat):
                    pos = end
                    continue

                # Track the FULL triplet (left, mid, right) for this made-word
                # This ensures every made-word is tied to a valid 3-word combination
                if made_word not in state.made_word_recipes:
                    state.made_word_recipes[made_word] = set()

                if delta > 0:
                    state.made_word_recipes[made_word].add((left, mid, right))
                else:
                    state.made_word_recipes[made_word].discard((left, mid, right))
                    if not state.made_word_recipes[made_word]:
                        del state.made_word_recipes[made_word]

                # Update each participating word's contribution count (for scoring)
                for word in [left, mid, right]:
                    if word not in state.word_contributions:
                        state.word_contributions[word] = {}
                    state.word_contributions[word][made_word] = \
                        state.word_contributions[word].get(made_word, 0) + delta
                    # Clean up zeros
                    if state.word_contributions[word][made_word] <= 0:
                        del state.word_contributions[word][made_word]

                pos = end


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

                update_made_words_for_triplet(left_word, word, right_word, state, lookups, +1)

    # Word as LEFT
    for t in lookups.templates_as_left.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for right_word in deck_list:
            if right_word == t.middle_word or right_word == word:
                continue
            if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                continue

            update_made_words_for_triplet(word, t.middle_word, right_word, state, lookups, +1)

    # Word as RIGHT
    for t in lookups.templates_as_right.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for left_word in deck_list:
            if left_word == t.middle_word or left_word == word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            update_made_words_for_triplet(left_word, t.middle_word, word, state, lookups, +1)


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

                update_made_words_for_triplet(left_word, word, right_word, state, lookups, -1)

    # Word as LEFT
    for t in lookups.templates_as_left.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for right_word in deck_list:
            if right_word == t.middle_word or right_word == word:
                continue
            if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                continue

            update_made_words_for_triplet(word, t.middle_word, right_word, state, lookups, -1)

    # Word as RIGHT
    for t in lookups.templates_as_right.get(word, []):
        if t.middle_word not in state.deck:
            continue

        for left_word in deck_list:
            if left_word == t.middle_word or left_word == word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            update_made_words_for_triplet(left_word, t.middle_word, word, state, lookups, -1)

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
    Uses connectivity-based scoring + made-word score.
    """
    if len(state.deck) <= 1:
        return 0.0, 0

    # Use existing word_connections if available, otherwise compute
    if word in state.word_connections:
        # Use cached connectivity
        left_neighbors = set()
        right_neighbors = set()
        for neighbor in state.word_connections[word]:
            # We need to distinguish left vs right neighbors
            # For now, just use the set as-is (represents total connectivity)
            pass
        # Fall back to computing it directly

    deck_list = [w for w in state.deck if w != word]
    deck_size = len(deck_list)

    left_neighbors = set()
    right_neighbors = set()
    total_combos = 0

    # --- As MIDDLE ---
    for t in lookups.templates_as_middle.get(word, []):
        for left_word in deck_list:
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            for right_word in deck_list:
                if right_word == left_word:
                    continue
                if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                    continue

                left_neighbors.add(left_word)
                right_neighbors.add(right_word)
                total_combos += 1

    # --- As LEFT ---
    for t in lookups.templates_as_left.get(word, []):
        if t.middle_word not in state.deck or t.middle_word == word:
            continue

        for right_word in deck_list:
            if right_word == t.middle_word:
                continue
            if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                continue

            right_neighbors.add(t.middle_word)
            total_combos += 1

    # --- As RIGHT ---
    for t in lookups.templates_as_right.get(word, []):
        if t.middle_word not in state.deck or t.middle_word == word:
            continue

        for left_word in deck_list:
            if left_word == t.middle_word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            left_neighbors.add(t.middle_word)
            total_combos += 1

    # === CONNECTIVITY SCORE ===
    left_percent = (len(left_neighbors) / deck_size) * 100 if deck_size > 0 else 0
    right_percent = (len(right_neighbors) / deck_size) * 100 if deck_size > 0 else 0

    # Target ~30% connectivity - penalize being too far from target
    TARGET_CONNECTIVITY = 40.0
    avg_connectivity = (left_percent + right_percent) / 2.0

    # Distance from target (0 = perfect, higher = worse)
    connectivity_distance = abs(avg_connectivity - TARGET_CONNECTIVITY)

    # MUCH STRONGER penalty: scale of 10 instead of 20
    # 10% off = ~0.37x, 20% off = ~0.14x, 40% off = ~0.02x
    connectivity_penalty = 2.71828 ** (-connectivity_distance / 10.0)

    # DRASTICALLY reduced base connectivity (from 0.1 to 0.001) - connectivity is now very minor
    base_connectivity = left_percent * right_percent * 0.001

    # Apply penalty to encourage targeting 30%
    connectivity_score = base_connectivity * connectivity_penalty

    # === WORD RECOGNIZABILITY ===
    word_freq = get_zipf(word)
    if word_freq >= 5.0:
        freq_weight = 1.0
    else:
        freq_weight = 10 ** (word_freq - 5.0)

    total_score = connectivity_score * freq_weight

    # --- MADE-WORD CONTRIBUTION ---
    made_word_score = 0.0
    if word in state.word_contributions:
        for made_word, count in state.word_contributions[word].items():
            length_score = (len(made_word) ** 1.5) * count
            global_count = len(state.made_word_recipes.get(made_word, set()))
            if global_count <= 1:
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

    # MASSIVE increase: multiply by 100 instead of dividing - made-words are now PRIMARY driver
    total_score += made_word_score * 100.0

    return total_score, total_combos


# ============================================================================ #
#                              O(1) SCORING                                    #
# ============================================================================ #

def compute_marginal_value(candidate: str, state: DeckState, lookups: Lookups) -> Tuple[float, int, Tuple[int, int, int]]:
    """
    Compute marginal value of adding candidate using connectivity-based scoring.

    Connectivity score = left_percent * right_percent
    Where:
      - left_percent = (# unique deck words that can be left neighbors) / len(deck)
      - right_percent = (# unique deck words that can be right neighbors) / len(deck)

    This naturally optimizes for balanced words that can connect broadly.

    Returns: (score, total_combos, (left_combos, middle_combos, right_combos))
    Note: total_combos and breakdown are still tracked for display purposes
    """
    if len(state.deck) == 0:
        return 0.0, 0, (0, 0, 0)

    deck_list = list(state.deck)
    deck_size = len(deck_list)

    # Track unique neighbors for connectivity
    left_neighbors = set()
    right_neighbors = set()

    # Track combos for display purposes only
    left_combos = 0
    middle_combos = 0
    right_combos = 0

    # --- As MIDDLE ---
    for t in lookups.templates_as_middle.get(candidate, []):
        for left_word in deck_list:
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            for right_word in deck_list:
                if right_word == left_word:
                    continue
                if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                    continue

                # Found valid triplet: left_word + candidate + right_word
                left_neighbors.add(left_word)
                right_neighbors.add(right_word)
                middle_combos += 1

    # --- As LEFT ---
    for t in lookups.templates_as_left.get(candidate, []):
        if t.middle_word not in state.deck:
            continue

        for right_word in deck_list:
            if right_word == t.middle_word:
                continue
            if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                continue

            # Found valid triplet: candidate + middle + right_word
            right_neighbors.add(t.middle_word)
            left_combos += 1

    # --- As RIGHT ---
    for t in lookups.templates_as_right.get(candidate, []):
        if t.middle_word not in state.deck:
            continue

        for left_word in deck_list:
            if left_word == t.middle_word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            # Found valid triplet: left_word + middle + candidate
            left_neighbors.add(t.middle_word)
            right_combos += 1

    total_combos = left_combos + middle_combos + right_combos

    # === CONNECTIVITY SCORE ===
    # Percentage of deck words that can be neighbors
    left_percent = (len(left_neighbors) / deck_size) * 100  # 0-100
    right_percent = (len(right_neighbors) / deck_size) * 100  # 0-100

    # Target ~30% connectivity - penalize being too far from target
    TARGET_CONNECTIVITY = 30.0
    avg_connectivity = (left_percent + right_percent) / 2.0

    # Distance from target (0 = perfect, higher = worse)
    connectivity_distance = abs(avg_connectivity - TARGET_CONNECTIVITY)

    # MUCH STRONGER penalty: scale of 10 instead of 20
    # 10% off = ~0.37x, 20% off = ~0.14x, 40% off = ~0.02x
    connectivity_penalty = 2.71828 ** (-connectivity_distance / 10.0)

    # DRASTICALLY reduced base connectivity (from 0.1 to 0.001) - connectivity is now very minor
    base_connectivity = left_percent * right_percent * 0.001

    # Apply penalty to encourage targeting 30%
    connectivity_score = base_connectivity * connectivity_penalty

    # === WORD RECOGNIZABILITY ===
    # Penalize uncommon words
    word_freq = get_zipf(candidate)
    if word_freq >= 5.0:
        freq_weight = 1.0
    else:
        freq_weight = 10 ** (word_freq - 5.0)

    total_score = connectivity_score * freq_weight

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
                segmentations = get_segmentations_cached(concat, lookups)

                if segmentations:
                    # Original triplet boundary positions (internal boundaries only)
                    original_boundaries = {len(left_word), len(left_word) + len(word)}

                    for seg in segmentations:
                        # Calculate boundary positions for this segmentation
                        seg_boundaries = set()
                        pos = 0
                        for seg_word in seg[:-1]:
                            pos += len(seg_word)
                            seg_boundaries.add(pos)

                        # Skip if ANY boundary position matches the original
                        if seg_boundaries & original_boundaries:
                            continue

                        # Valid alternate segmentation - count each made-word
                        pos = 0
                        for made_word in seg:
                            start = pos
                            end = pos + len(made_word)

                            # Skip words that are the same as original deck words at same position
                            if (made_word == left_word and start == 0 and end == len(left_word)):
                                pos = end
                                continue
                            if (made_word == word and start == len(left_word) and end == len(left_word) + len(word)):
                                pos = end
                                continue
                            if (made_word == right_word and start == len(left_word) + len(word) and end == len(concat)):
                                pos = end
                                continue

                            made_words[made_word] = made_words.get(made_word, 0) + 1
                            pos = end

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
            segmentations = get_segmentations_cached(concat, lookups)

            if segmentations:
                # Original triplet boundary positions (internal boundaries only)
                original_boundaries = {len(word), len(word) + len(t.middle_word)}

                for seg in segmentations:
                    # Calculate boundary positions for this segmentation
                    seg_boundaries = set()
                    pos = 0
                    for seg_word in seg[:-1]:
                        pos += len(seg_word)
                        seg_boundaries.add(pos)

                    # Skip if ANY boundary position matches the original
                    if seg_boundaries & original_boundaries:
                        continue

                    # Valid alternate segmentation - count each made-word
                    pos = 0
                    for made_word in seg:
                        start = pos
                        end = pos + len(made_word)

                        # Skip words that are the same as original deck words at same position
                        if (made_word == word and start == 0 and end == len(word)):
                            pos = end
                            continue
                        if (made_word == t.middle_word and start == len(word) and end == len(word) + len(t.middle_word)):
                            pos = end
                            continue
                        if (made_word == right_word and start == len(word) + len(t.middle_word) and end == len(concat)):
                            pos = end
                            continue

                        made_words[made_word] = made_words.get(made_word, 0) + 1
                        pos = end

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
            segmentations = get_segmentations_cached(concat, lookups)

            if segmentations:
                # Original triplet boundary positions (internal boundaries only)
                original_boundaries = {len(left_word), len(left_word) + len(t.middle_word)}

                for seg in segmentations:
                    # Calculate boundary positions for this segmentation
                    seg_boundaries = set()
                    pos = 0
                    for seg_word in seg[:-1]:
                        pos += len(seg_word)
                        seg_boundaries.add(pos)

                    # Skip if ANY boundary position matches the original
                    if seg_boundaries & original_boundaries:
                        continue

                    # Valid alternate segmentation - count each made-word
                    pos = 0
                    for made_word in seg:
                        start = pos
                        end = pos + len(made_word)

                        # Skip words that are the same as original deck words at same position
                        if (made_word == left_word and start == 0 and end == len(left_word)):
                            pos = end
                            continue
                        if (made_word == t.middle_word and start == len(left_word) and end == len(left_word) + len(t.middle_word)):
                            pos = end
                            continue
                        if (made_word == word and start == len(left_word) + len(t.middle_word) and end == len(concat)):
                            pos = end
                            continue

                        made_words[made_word] = made_words.get(made_word, 0) + 1
                        pos = end

    return made_words


def contains_blacklisted_word(candidate: str, state: DeckState, lookups: Lookups) -> bool:
    """
    Check if adding this candidate would create any blacklisted words.
    Returns True if any blacklisted word would be formed.
    """
    if not WORD_BLACKLIST:
        return False

    # Check if the candidate itself is blacklisted
    if candidate in WORD_BLACKLIST:
        return True

    candidate_made_words = enumerate_word_triplets(candidate, state, lookups)

    # Check if any made words are blacklisted
    for made_word in candidate_made_words.keys():
        if made_word in WORD_BLACKLIST:
            return True

    return False


# ============================================================================ #
#                              MADE WORD SCORING                               #
# ============================================================================ #

def calculate_complexity_score(made_word: str, concat: str, left: str, mid: str, right: str) -> float:
    """
    Calculate a complexity score for a made-word based on how it's formed.

    Returns higher scores for:
    - Multi-word merges with substantial fragments from each (e.g., "however"+"and"+"another"="veranda")
    - Complex combinations that merge parts from multiple words (e.g., "over"+"used"="overuse")

    Returns lower scores for:
    - Simple substring extractions (e.g., "another"="other")
    - Single-letter + word combinations (e.g., "r"+"over"="rover")
    """
    # Find where made_word appears in concat
    start_pos = concat.find(made_word)
    if start_pos == -1:
        # Should not happen, but return neutral score
        return 1.0

    end_pos = start_pos + len(made_word)

    # Determine which source words overlap with [start_pos, end_pos)
    left_end = len(left)
    mid_end = left_end + len(mid)
    right_end = mid_end + len(right)

    # Calculate how many characters from each source word contribute
    left_contribution = 0
    mid_contribution = 0
    right_contribution = 0

    if start_pos < left_end and end_pos > 0:
        left_contribution = min(left_end, end_pos) - max(0, start_pos)
    if start_pos < mid_end and end_pos > left_end:
        mid_contribution = min(mid_end, end_pos) - max(left_end, start_pos)
    if start_pos < right_end and end_pos > mid_end:
        right_contribution = min(right_end, end_pos) - max(mid_end, start_pos)

    total_contribution = left_contribution + mid_contribution + right_contribution
    if total_contribution != len(made_word):
        # Sanity check failed
        return 1.0

    # Count contributing words (those with non-zero contribution)
    num_contributors = sum([
        1 if left_contribution > 0 else 0,
        1 if mid_contribution > 0 else 0,
        1 if right_contribution > 0 else 0,
    ])

    # PENALTY 1: Simple substring extraction (mostly from one word)
    # If one word contributes >70% of the made-word, it's a simple extraction
    max_contribution = max(left_contribution, mid_contribution, right_contribution)
    if num_contributors == 1:
        # Pure substring - heavy penalty
        return 0.2
    elif max_contribution / len(made_word) > 0.7:
        # Dominated by one word - moderate penalty
        return 0.5

    # PENALTY 2: Single-letter contributions
    # Penalize if any contributing word only provides 1 character
    contributions = [c for c in [left_contribution, mid_contribution, right_contribution] if c > 0]
    if min(contributions) == 1:
        # Has a single-letter contribution
        if num_contributors == 2:
            # Two words, one is single letter - moderate penalty
            return 0.6
        else:
            # Three words, one is single letter - lighter penalty
            return 0.8

    # BONUS: Multi-word merge with substantial fragments
    # All contributors provide at least 2 chars each
    if num_contributors >= 2 and min(contributions) >= 2:
        # Good! Reward based on number of contributors
        if num_contributors == 3:
            return 2.0  # Best case: all three words contribute meaningfully
        else:
            return 1.5  # Two words contribute meaningfully

    # Default: neutral
    return 1.0


def score_actual_made_words(candidate: str, state: DeckState, lookups: Lookups) -> float:
    """
    Score a candidate based on the actual made-words it would produce.
    Rewards words that create rare/unique made-words (low count in current deck).
    Now includes complexity scoring to reward interesting combinations.
    """
    # Enumerate all triplets and track made-words with their source triplets
    made_word_data = {}  # made_word -> list of (left, mid, right, concat) tuples
    deck_list = list(state.deck)

    # Word as MIDDLE
    for t in lookups.templates_as_middle.get(candidate, []):
        for left_word in deck_list:
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            for right_word in deck_list:
                if right_word == left_word:
                    continue
                if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                    continue

                concat = left_word + candidate + right_word
                segmentations = get_segmentations_cached(concat, lookups)

                if segmentations:
                    original_boundaries = {len(left_word), len(left_word) + len(candidate)}

                    for seg in segmentations:
                        seg_boundaries = set()
                        pos = 0
                        for seg_word in seg[:-1]:
                            pos += len(seg_word)
                            seg_boundaries.add(pos)

                        if seg_boundaries & original_boundaries:
                            continue

                        pos = 0
                        for made_word in seg:
                            start = pos
                            end = pos + len(made_word)

                            if (made_word == left_word and start == 0 and end == len(left_word)):
                                pos = end
                                continue
                            if (made_word == candidate and start == len(left_word) and end == len(left_word) + len(candidate)):
                                pos = end
                                continue
                            if (made_word == right_word and start == len(left_word) + len(candidate) and end == len(concat)):
                                pos = end
                                continue

                            if made_word not in made_word_data:
                                made_word_data[made_word] = []
                            made_word_data[made_word].append((left_word, candidate, right_word, concat))
                            pos = end

    # Word as LEFT
    for t in lookups.templates_as_left.get(candidate, []):
        if t.middle_word not in state.deck:
            continue

        for right_word in deck_list:
            if right_word == t.middle_word:
                continue
            if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                continue

            concat = candidate + t.middle_word + right_word
            segmentations = get_segmentations_cached(concat, lookups)

            if segmentations:
                original_boundaries = {len(candidate), len(candidate) + len(t.middle_word)}

                for seg in segmentations:
                    seg_boundaries = set()
                    pos = 0
                    for seg_word in seg[:-1]:
                        pos += len(seg_word)
                        seg_boundaries.add(pos)

                    if seg_boundaries & original_boundaries:
                        continue

                    pos = 0
                    for made_word in seg:
                        start = pos
                        end = pos + len(made_word)

                        if (made_word == candidate and start == 0 and end == len(candidate)):
                            pos = end
                            continue
                        if (made_word == t.middle_word and start == len(candidate) and end == len(candidate) + len(t.middle_word)):
                            pos = end
                            continue
                        if (made_word == right_word and start == len(candidate) + len(t.middle_word) and end == len(concat)):
                            pos = end
                            continue

                        if made_word not in made_word_data:
                            made_word_data[made_word] = []
                        made_word_data[made_word].append((candidate, t.middle_word, right_word, concat))
                        pos = end

    # Word as RIGHT
    for t in lookups.templates_as_right.get(candidate, []):
        if t.middle_word not in state.deck:
            continue

        for left_word in deck_list:
            if left_word == t.middle_word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            concat = left_word + t.middle_word + candidate
            segmentations = get_segmentations_cached(concat, lookups)

            if segmentations:
                original_boundaries = {len(left_word), len(left_word) + len(t.middle_word)}

                for seg in segmentations:
                    seg_boundaries = set()
                    pos = 0
                    for seg_word in seg[:-1]:
                        pos += len(seg_word)
                        seg_boundaries.add(pos)

                    if seg_boundaries & original_boundaries:
                        continue

                    pos = 0
                    for made_word in seg:
                        start = pos
                        end = pos + len(made_word)

                        if (made_word == left_word and start == 0 and end == len(left_word)):
                            pos = end
                            continue
                        if (made_word == t.middle_word and start == len(left_word) and end == len(left_word) + len(t.middle_word)):
                            pos = end
                            continue
                        if (made_word == candidate and start == len(left_word) + len(t.middle_word) and end == len(concat)):
                            pos = end
                            continue

                        if made_word not in made_word_data:
                            made_word_data[made_word] = []
                        made_word_data[made_word].append((left_word, t.middle_word, candidate, concat))
                        pos = end

    if not made_word_data:
        return 0.0

    total_score = 0.0

    for made_word, triplet_list in made_word_data.items():
        count_added = len(triplet_list)

        # Current number of unique recipes for this made-word in deck (0 if new)
        current_count = len(state.made_word_recipes.get(made_word, set()))

        # Calculate average complexity score across all ways this word is made
        complexity_scores = []
        for left, mid, right, concat in triplet_list:
            complexity = calculate_complexity_score(made_word, concat, left, mid, right)
            complexity_scores.append(complexity)
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 1.0

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

        # POWER-BASED COMPLEXITY SCORING (Option 3)
        # Raise the combined score to the power of complexity
        # This creates dramatic separation:
        #   Complexity 0.2 (bad): score^0.2 = tiny value
        #   Complexity 1.0 (neutral): score^1.0 = score
        #   Complexity 2.0 (great): score^2.0 = score squared (huge!)
        base_score = length_score + rarity_bonus
        complexity_adjusted_score = base_score ** avg_complexity

        total_score += complexity_adjusted_score

    # Diversity bonus: producing many different made-words
    diversity_bonus = len(made_word_data) * 10.0
    total_score += diversity_bonus

    # MASSIVE weight increase: multiply by 100 instead of dividing - made-words dominate scoring
    return total_score * 100.0  # Made-words are now the PRIMARY driver of optimization


# ============================================================================ #
#                              DECK BUILDING                                   #
# ============================================================================ #

def count_word_triplets(word: str, state: DeckState, lookups: Lookups) -> int:
    """
    Count how many valid triplets this word can participate in.
    Used to ensure words meet minimum triplet requirement.
    """
    count = 0
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

                # Valid triplet found
                count += 1

    # Word as LEFT
    for t in lookups.templates_as_left.get(word, []):
        if t.middle_word not in state.deck or t.middle_word == word:
            continue

        for right_word in deck_list:
            if right_word == t.middle_word or right_word == word:
                continue
            if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                continue

            count += 1

    # Word as RIGHT
    for t in lookups.templates_as_right.get(word, []):
        if t.middle_word not in state.deck or t.middle_word == word:
            continue

        for left_word in deck_list:
            if left_word == t.middle_word or left_word == word:
                continue
            if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                continue

            count += 1

    return count


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

    # Filter out words that start or end with 's' (avoids plural optimization zone)
    candidates = {w for w in candidates if not (w.startswith('s') or w.endswith('s'))}

    # Filter out words containing 'ai' or 'ia' (can create adjacent 1-letter words)
    candidates = {w for w in candidates if not has_adjacent_one_letter_words(w)}

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
            # Skip candidates that would create blacklisted words
            if contains_blacklisted_word(candidate, state, lookups):
                continue
            score, combos, breakdown = compute_marginal_value(candidate, state, lookups)
            if score > 0:
                candidate_scores.append((candidate, score, combos, breakdown))

        if not candidate_scores:
            print(f"No viable candidates")
            break

        # Step 2: Take top 10 and refine with made-word scoring
        candidate_scores.sort(key=lambda x: -x[1])
        top_candidates = candidate_scores[:10]

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

        # Pruning logic: 3:1 ratio before target, 1:1 after target
        prune_threshold = 1 if len(deck_list) >= target_size else 3
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


def optimize_puzzle(
    deck_words: Set[str],
    lookups: Lookups,
    puzzle_size: int = 12,
    max_swaps: int = 50,
    min_triplets: int = 3,
) -> Tuple[List[str], DeckState]:
    """
    Optimize a fixed-size puzzle by swapping words.
    Starts with random words and swaps to maximize made-word quality.
    """
    import random

    # Filter candidates
    candidates = deck_words.copy()
    candidates = {w for w in candidates if not (w.startswith('s') or w.endswith('s'))}
    candidates = {w for w in candidates if not has_adjacent_one_letter_words(w)}
    candidates = {w for w in candidates if w in lookups.templates_as_middle}
    # Filter out blacklisted words
    candidates = {w for w in candidates if w not in WORD_BLACKLIST}
    viable = list(candidates)

    # Start with random words
    initial_words = random.sample(viable, puzzle_size)
    state = create_deck_state(initial_words, lookups)
    deck_list = list(initial_words)

    print(f"\nStarting puzzle optimization with {puzzle_size} random words")
    print(f"Initial words: {', '.join(deck_list)}")
    print(f"Max swaps: {max_swaps}, Min triplets per word: {min_triplets}")

    # Check initial constraint
    print(f"\nChecking initial triplet counts...")
    for word in deck_list:
        triplet_count = count_word_triplets(word, state, lookups)
        print(f"  {word:<12} can form {triplet_count:3d} triplets")

    best_score = sum(compute_word_contribution(w, state, lookups)[0] for w in deck_list)
    print(f"\nInitial total score: {best_score:.1f}")

    # Track best state for controlled regressions
    best_deck_score = best_score
    best_deck_list = deck_list.copy()
    best_deck_state = None  # Will deep copy when we find a better score

    consecutive_regressions = 0
    total_regressions = 0
    MAX_CONSECUTIVE_REGRESSIONS = 3
    MAX_TOTAL_REGRESSIONS = 8

    swap_count = 0
    candidates_pool = [w for w in viable if w not in deck_list]

    while swap_count < max_swaps:
        swap_count += 1
        print(f"\n{'='*60}")
        print(f"SWAP {swap_count}/{max_swaps}")
        print(f"{'='*60}")

        # Find worst word in deck
        worst_word = None
        worst_score = float('inf')
        worst_triplets = 0

        for word in deck_list:
            triplet_count = count_word_triplets(word, state, lookups)
            score, _ = compute_word_contribution(word, state, lookups)

            # Penalize words below minimum triplet count
            if triplet_count < min_triplets:
                score = score * 0.1  # Heavy penalty

            if score < worst_score:
                worst_score = score
                worst_word = word
                worst_triplets = triplet_count

        print(f"Worst word: {worst_word} (score={worst_score:.1f}, triplets={worst_triplets})")

        # Find best replacement - use thorough scoring like run15
        # Temporarily remove worst word to evaluate replacements
        remove_word_from_state(worst_word, state, lookups)

        # Step 1: Fast scoring to get top candidates
        candidate_scores = []
        for candidate in candidates_pool:
            if candidate in state.deck:
                continue

            # Skip blacklisted words or words that would create blacklisted made-words
            if contains_blacklisted_word(candidate, state, lookups):
                continue

            # Quick check: can it form enough triplets?
            triplet_count = count_word_triplets(candidate, state, lookups)
            if triplet_count < min_triplets:
                continue

            # Fast scoring with connectivity
            base_score, combos, breakdown = compute_marginal_value(candidate, state, lookups)
            if base_score > 0:
                candidate_scores.append((candidate, base_score, combos, breakdown, triplet_count))

        if not candidate_scores:
            # Re-add worst word
            add_word_to_state(worst_word, state, lookups)
            print(f"No viable replacement found. Stopping.")
            break

        # Step 2: Skip top 30 (overly connected), take next 20 and refine with made-word scoring
        candidate_scores.sort(key=lambda x: -x[1])
        # Skip top 30 most connected, take next 20 (candidates 31-50)
        if len(candidate_scores) >= 50:
            top_candidates = candidate_scores[30:50]
        elif len(candidate_scores) > 30:
            # Have some candidates after skipping 30, take what's left
            top_candidates = candidate_scores[30:]
        else:
            # Not enough candidates, take what we have
            top_candidates = candidate_scores[:min(20, len(candidate_scores))]

        best_replacement = None
        best_replacement_score = -float('inf')
        best_replacement_triplets = 0
        best_base_score = 0
        best_made_word_bonus = 0

        for candidate, base_score, combos, breakdown, triplet_count in top_candidates:
            # Thorough scoring with actual made-word enumeration
            made_word_bonus = score_actual_made_words(candidate, state, lookups)
            total_score = base_score + made_word_bonus

            if total_score > best_replacement_score:
                best_replacement_score = total_score
                best_replacement = candidate
                best_replacement_triplets = triplet_count
                best_base_score = base_score
                best_made_word_bonus = made_word_bonus

        # Re-add worst word temporarily
        add_word_to_state(worst_word, state, lookups)

        if best_replacement is None:
            print(f"No viable replacement found. Stopping.")
            break

        print(f"Best replacement: {best_replacement} (base={best_base_score:.1f} +made={best_made_word_bonus:.1f} total={best_replacement_score:.1f}, triplets={best_replacement_triplets})")

        # Perform swap to test actual improvement
        print(f"{RED}-{worst_word:<12}{RESET} -> {GREEN}+{best_replacement:<12}{RESET}")
        remove_word_from_state(worst_word, state, lookups)
        deck_list.remove(worst_word)
        add_word_to_state(best_replacement, state, lookups)
        deck_list.append(best_replacement)

        # Check if swap actually improves overall score
        new_score = sum(compute_word_contribution(w, state, lookups)[0] for w in deck_list)
        improvement = new_score - best_score
        print(f"Total score: {best_score:.1f} -> {new_score:.1f} (Δ={improvement:+.1f})")

        # Keep the swap regardless of improvement (controlled regressions)
        candidates_pool.remove(best_replacement)
        candidates_pool.append(worst_word)
        best_score = new_score

        if improvement <= 0:
            # Track regression
            consecutive_regressions += 1
            total_regressions += 1
            print(f"{RED}Regression accepted{RESET} (consecutive: {consecutive_regressions}/{MAX_CONSECUTIVE_REGRESSIONS}, total: {total_regressions}/{MAX_TOTAL_REGRESSIONS})")

            # Check if we should stop due to too many regressions
            if consecutive_regressions >= MAX_CONSECUTIVE_REGRESSIONS:
                print(f"\n{RED}Maximum consecutive regressions ({MAX_CONSECUTIVE_REGRESSIONS}) reached. Stopping.{RESET}")
                break
            if total_regressions >= MAX_TOTAL_REGRESSIONS:
                print(f"\n{RED}Maximum total regressions ({MAX_TOTAL_REGRESSIONS}) reached. Stopping.{RESET}")
                break
        else:
            # Improvement - reset consecutive counter
            consecutive_regressions = 0
            print(f"{GREEN}Improvement! Consecutive regressions reset.{RESET}")

            # Update best state if this is a new high score
            if new_score > best_deck_score:
                import copy
                best_deck_score = new_score
                best_deck_list = deck_list.copy()
                best_deck_state = copy.deepcopy(state)
                print(f"{GREEN}★ New best score: {best_deck_score:.1f}{RESET}")

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE ({swap_count} swaps)")
    print(f"{'='*60}")

    # Restore best state if we ended on a regression
    if best_deck_state is not None and best_score < best_deck_score:
        print(f"\n{GREEN}Restoring best state (score: {best_deck_score:.1f} vs current: {best_score:.1f}){RESET}")
        deck_list = best_deck_list
        state = best_deck_state
    elif best_deck_state is None:
        # Never found an improvement, use initial state
        print(f"\n{YELLOW}No improvements found, returning initial state{RESET}")

    print(f"Final words: {', '.join(sorted(deck_list))}")

    # Final triplet check
    print(f"\nFinal triplet counts:")
    for word in sorted(deck_list):
        triplet_count = count_word_triplets(word, state, lookups)
        status = "✓" if triplet_count >= min_triplets else "✗"
        print(f"  {status} {word:<12} can form {triplet_count:3d} triplets")

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

    # Connectivity analysis - track directional adjacency
    print(f"\n{'='*60}")
    print("CONNECTIVITY ANALYSIS (directional adjacency)")
    print(f"{'='*60}")

    deck_size = len(deck)
    left_neighbors = {w: set() for w in deck}  # Words that can appear to the left of this word
    right_neighbors = {w: set() for w in deck}  # Words that can appear to the right of this word

    # Enumerate all valid triplets to find adjacent pairs
    for mid in deck:
        for t in lookups.templates_as_middle.get(mid, []):
            for left_word in deck:
                if left_word == mid:
                    continue
                if t.first_key not in lookups.word_left_keys.get(left_word, set()):
                    continue

                for right_word in deck:
                    if right_word == left_word or right_word == mid:
                        continue
                    if t.last_key not in lookups.word_right_keys.get(right_word, set()):
                        continue

                    # Valid triplet found - check if it produces segmentations
                    concat = left_word + mid + right_word
                    segs = segment_word(concat, require_start=True, require_end=True)
                    if segs:
                        # Track adjacent pairs only
                        # left_word <-> mid (adjacent)
                        right_neighbors[left_word].add(mid)
                        left_neighbors[mid].add(left_word)
                        # mid <-> right_word (adjacent)
                        right_neighbors[mid].add(right_word)
                        left_neighbors[right_word].add(mid)

    connectivity_scores = []
    for word in deck:
        left_count = len(left_neighbors[word])
        right_count = len(right_neighbors[word])
        total_neighbors = len(left_neighbors[word] | right_neighbors[word])
        left_pct = left_count / (deck_size - 1) if deck_size > 1 else 0
        right_pct = right_count / (deck_size - 1) if deck_size > 1 else 0
        total_pct = total_neighbors / (deck_size - 1) if deck_size > 1 else 0
        connectivity_scores.append((word, left_count, right_count, total_neighbors, left_pct, right_pct, total_pct))

    connectivity_scores.sort(key=lambda x: -x[6])
    print(f"\nWord connectivity (can be adjacent to % of other deck words):")
    for i, (word, left_cnt, right_cnt, total, left_pct, right_pct, total_pct) in enumerate(connectivity_scores):
        print(f"  {word:<12} L:{left_cnt:2d} R:{right_cnt:2d} Total:{total:2d}/{deck_size-1:2d} ({total_pct*100:.1f}%) [L:{left_pct*100:.0f}% R:{right_pct*100:.0f}%]")

    avg_left = sum(x[4] for x in connectivity_scores) / len(connectivity_scores) if connectivity_scores else 0
    avg_right = sum(x[5] for x in connectivity_scores) / len(connectivity_scores) if connectivity_scores else 0
    avg_total = sum(x[6] for x in connectivity_scores) / len(connectivity_scores) if connectivity_scores else 0
    print(f"\nAverage connectivity: {avg_total*100:.1f}% (Left: {avg_left*100:.1f}%, Right: {avg_right*100:.1f}%)")

    # Use the made-word recipes tracked during deck building
    print(f"\n{'='*60}")
    print("MADE WORDS ANALYSIS (unique creation patterns)")
    print(f"{'='*60}")

    made_word_recipes = state.made_word_recipes
    print(f"Total unique made words: {len(made_word_recipes)}")

    if made_word_recipes:
        # Sort by length (longest)
        by_length = sorted(made_word_recipes.items(), key=lambda x: (-len(x[0]), -len(x[1])))
        print(f"\nLongest made words:")
        for i, (made_word, recipes) in enumerate(by_length[:30]):
            # Calculate unique contributing word combinations
            contributing_recipes = set()
            for left, mid, right in recipes:
                concat = left + mid + right
                contributors = find_contributing_words(made_word, concat, left, mid, right)
                contributing_recipes.add(contributors)

            num_recipes = len(contributing_recipes)
            print(f"  {made_word:<20} (len={len(made_word):2d}) created by {num_recipes:3d} unique word combination{'s' if num_recipes != 1 else ''}")
            # Show example recipes for first few
            if i < 5:
                for j, recipe in enumerate(list(contributing_recipes)[:3]):
                    print(f"      e.g., {' + '.join(recipe)}")
                if num_recipes > 3:
                    print(f"      ... and {num_recipes - 3} more")

        # Sort by number of unique ways to create (diversity)
        # Need to calculate contributing recipes for each word first
        word_contributing_counts = []
        for made_word, recipes in made_word_recipes.items():
            contributing_recipes = set()
            for left, mid, right in recipes:
                concat = left + mid + right
                contributors = find_contributing_words(made_word, concat, left, mid, right)
                contributing_recipes.add(contributors)
            word_contributing_counts.append((made_word, contributing_recipes))

        by_diversity = sorted(word_contributing_counts, key=lambda x: -len(x[1]))
        print(f"\nMost versatile made words (most unique ways to create):")
        for i, (made_word, contributing_recipes) in enumerate(by_diversity[:30]):
            num_recipes = len(contributing_recipes)
            print(f"  {made_word:<20} created by {num_recipes:3d} unique word combination{'s' if num_recipes != 1 else ''}")
            # Show example recipes for first few
            if i < 5:
                for j, recipe in enumerate(list(contributing_recipes)[:3]):
                    print(f"      e.g., {' + '.join(recipe)}")
                if num_recipes > 3:
                    print(f"      ... and {num_recipes - 3} more")

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
            # Apply banned words filter (in case cache was from before banned words existed)
            seg_words_before = len(seg_words)
            seg_words = seg_words - BANNED_WORDS
            if len(seg_words) != seg_words_before:
                print(f"Removed {seg_words_before - len(seg_words)} banned words from cached seg_words")
                # Need to rebuild lookups since seg_words changed
                lookups = None
            set_segmentation_dict(seg_words)
            print(f"Deck words: {len(deck_words)}, Segmentation words: {len(seg_words)}")
        except Exception as e:
            print(f"Cache load failed: {e}, rebuilding...")
            deck_words, seg_words, lookups = None, None, None
    else:
        deck_words, seg_words, lookups = None, None, None

    # Build if not cached
    if deck_words is None or seg_words is None:
        print("Loading dictionary...")
        deck_words, seg_words = load_dictionary()
        print(f"Deck words: {len(deck_words)}, Segmentation words: {len(seg_words)}")
        set_segmentation_dict(seg_words)
        lookups = None  # Force rebuild of lookups

    if lookups is None:
        print("\nBuilding lookups...")
        lookups = build_lookups(deck_words, seg_words)

        # Save to cache
        save_cache(cache_path, deck_words, seg_words, lookups)

    print(f"\n{'='*60}")
    print("OPTIMIZING PUZZLE")
    print(f"{'='*60}")

    deck, state = optimize_puzzle(
        deck_words=deck_words,
        lookups=lookups,
        puzzle_size=12,
        max_swaps=50,
        min_triplets=2,  # Reduced from 3 to allow lower connectivity
    )
    analyze_deck(deck, lookups, state)


if __name__ == "__main__":
    main()
