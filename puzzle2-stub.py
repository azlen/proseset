#!/usr/bin/env python3
"""
Proseset Word Decomposition Precomputer - puzzle2-stub.py

Pre-computes which words (of ANY length) can be broken down into
sequences of valid card-length words, allowing edge fragments that
extend into adjacent cards.

In the game, a target word appears in the re-segmentation by spanning
across card boundaries. The edges of the target word may only partially
consume the first/last cards, with the leftover characters forming
other valid words.

Example: "constables" → [i]con + stab + les[son]
  Cards: icon, stab, lesson
  Re-segmentation: "i | constables | son"

Output: lookup tables of decomposable words with their decomposition
patterns and extension possibilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Set, Tuple
from wordfreq import top_n_list, zipf_frequency
import third_party.twl as twl
from tqdm import tqdm
import pickle
import hashlib
import os
import argparse

# ============================================================================ #
#                              CONFIGURATION                                   #
# ============================================================================ #

N_WORDS = 50000

# Length range for CARD words (interior pieces + edge card extensions)
CARD_MIN_LENGTH = 2
CARD_MAX_LENGTH = 7

# Minimum length of TARGET words (the long words we want to appear in re-segmentation)
TARGET_MIN_LENGTH = 4

ALLOWED_1_LETTER = {'a', 'i'}
ALLOWED_2_LETTER = {
    'ah', 'am', 'an', 'as', 'at', 'ax', 'be', 'by', 'do', 'eh', 'ex',
    'go', 'he', 'if', 'in', 'is', 'it', 'me', 'no', 'of', 'oh', 'on',
    'or', 'ow', 'ox', 'pi', 'up', 'us', 'we', 'ya', 'yo'
}

BANNED_WORDS = {
    'aa', 'ae', 'ag', 'ai', 'al', 'ar', 'aw', 'ay', 'ba', 'bi', 'bo',
    'de', 'ed', 'ef', 'el', 'em', 'en', 'er', 'es', 'et', 'ew', 'fa', 'fe', 'gi',
    'gu', 'hm', 'ho', 'id', 'jo', 'ka', 'ki', 'la', 'li', 'lo', 'mi', 'mm', 'mo',
    'mu', 'na', 'ne', 'nu', 'od', 'oe', 'oi', 'ok', 'om', 'op', 'os', 'pe', 'po', 'qi',
    're', 'sh', 'si', 'ta', 'te', 'ti', 'un', 'ut', 'wo', 'xi', 'xu', 'ya', 'ye', 'za',
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
    'martin', 'alan'
}

WORD_BLACKLIST = {
    "nazi", "rape", "raper", "rapist", "ass", "dick", "fuck", "molest", "anal", "assault", "sex"
}

MAX_DECOMPOSITIONS_PER_WORD = 50


# ============================================================================ #
#                              DICTIONARY                                      #
# ============================================================================ #

@lru_cache(maxsize=None)
def get_zipf(word: str) -> float:
    return zipf_frequency(word, 'en')


NO_SINGLE_LETTERS = False

def load_dictionary() -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns:
        seg_words: All valid words (any length) — for validating leftovers
        card_words: Words within card length range — valid cards
        target_words: Words >= TARGET_MIN_LENGTH — candidates to decompose
    """
    top_words = set(top_n_list('en', N_WORDS, wordlist='best'))

    seg_words = (
        ALLOWED_1_LETTER | ALLOWED_2_LETTER |
        {w for w in top_words if twl.check(w) and w.isalpha() and len(w) > 2}
    ) - BANNED_WORDS - WORD_BLACKLIST

    if NO_SINGLE_LETTERS:
        seg_words -= ALLOWED_1_LETTER

    card_words = {w for w in seg_words if CARD_MIN_LENGTH <= len(w) <= CARD_MAX_LENGTH}
    # Filter out s-starting words (avoids plural/suffix optimization traps)
    card_words = {w for w in card_words if not w.startswith('s')}
    target_words = {w for w in seg_words if len(w) >= TARGET_MIN_LENGTH}

    return seg_words, card_words, target_words


# ============================================================================ #
#                              SEGMENTATION HELPERS                            #
# ============================================================================ #

_seg_words: Set[str] = set()
_card_words: Set[str] = set()


def init_word_sets(seg_words: Set[str], card_words: Set[str]):
    global _seg_words, _card_words
    _seg_words = seg_words
    _card_words = card_words
    _can_segment.cache_clear()
    _segment_into_cards.cache_clear()


@lru_cache(maxsize=None)
def _can_segment(s: str) -> bool:
    """Check if string can be fully segmented into valid seg_words."""
    if not s:
        return True
    for i in range(1, len(s) + 1):
        if s[:i] in _seg_words and _can_segment(s[i:]):
            return True
    return False


@lru_cache(maxsize=None)
def _segment_into_cards(s: str) -> Tuple[Tuple[str, ...], ...]:
    """Find all ways to segment string into card_words."""
    if not s:
        return ((),)
    results = []
    for i in range(CARD_MIN_LENGTH, min(CARD_MAX_LENGTH, len(s)) + 1):
        prefix = s[:i]
        if prefix in _card_words:
            for tail in _segment_into_cards(s[i:]):
                results.append((prefix,) + tail)
    return tuple(results)


# ============================================================================ #
#                              FRAGMENT INDEXES                                #
# ============================================================================ #

def build_fragment_indexes(card_words: Set[str]) -> Tuple[Set[str], Set[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Precompute which fragments are valid as left/right extensions.

    A LEFT fragment f is valid if there exists a card word w ending with f
    where the leftover prefix w[:-len(f)] is segmentable into valid words.
    (The leftover becomes words in the re-segmentation.)

    A RIGHT fragment f is valid if there exists a card word w starting with f
    where the leftover suffix w[len(f):] is segmentable into valid words.

    Also builds indexes: fragment → list of (card, leftover) pairs.
    """
    valid_left_frags = set()
    valid_right_frags = set()
    # fragment → list of (card_word, leftover) for display/lookup
    left_frag_cards: Dict[str, List[str]] = {}   # frag → cards ending with frag
    right_frag_cards: Dict[str, List[str]] = {}  # frag → cards starting with frag

    for word in tqdm(sorted(card_words), desc="Building fragment indexes",
                     ascii=" ▖▘▝▗▚▞█", bar_format='{desc}: |{bar:30}| {n_fmt}/{total_fmt}'):
        for i in range(1, len(word)):
            # Left fragment: suffix of card = what the card donates to target word
            # Leftover: prefix of card = goes into re-segmentation
            frag = word[i:]     # donated to target word
            leftover = word[:i] # must be segmentable
            if _can_segment(leftover):
                valid_left_frags.add(frag)
                left_frag_cards.setdefault(frag, []).append(word)

            # Right fragment: prefix of card = what the card donates to target word
            # Leftover: suffix of card = goes into re-segmentation
            frag = word[:i]     # donated to target word
            leftover = word[i:] # must be segmentable
            if _can_segment(leftover):
                valid_right_frags.add(frag)
                right_frag_cards.setdefault(frag, []).append(word)

    return valid_left_frags, valid_right_frags, left_frag_cards, right_frag_cards


# ============================================================================ #
#                              DECOMPOSITION                                   #
# ============================================================================ #

@dataclass
class Decomposition:
    """One way a target word spans across card boundaries.

    left_frag:  Fragment at the left edge (suffix of a card). "" if first piece is a whole card.
    interior:   Complete card words fully consumed by the target word.
    right_frag: Fragment at the right edge (prefix of a card). "" if last piece is a whole card.

    The actual cards would be:
      [card ending with left_frag] + interior cards + [card starting with right_frag]
    And the leftover parts of the edge cards form other words in the re-segmentation.
    """
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


def decompose_word(
    word: str,
    valid_left_frags: Set[str],
    valid_right_frags: Set[str],
) -> List[Decomposition]:
    """
    Find all valid decompositions of a target word.

    A decomposition splits the target word as:
      [left_frag] + card_1 + card_2 + ... + card_n + [right_frag]
    where:
      - left_frag (optional): a suffix of some card, with segmentable leftover
      - card_1...card_n: complete card words
      - right_frag (optional): a prefix of some card, with segmentable leftover
      - total pieces >= 2
    """
    results = []
    max_frag_len = CARD_MAX_LENGTH - 1  # fragment < card length (need leftover)

    # Try all (left_end, right_start) pairs
    # left_end = length of left fragment (0 = no fragment)
    # right_start = index where right fragment begins (len(word) = no fragment)
    for left_end in range(0, min(len(word), max_frag_len) + 1):
        left_frag = word[:left_end] if left_end > 0 else ""

        if left_frag and left_frag not in valid_left_frags:
            continue

        for right_start in range(max(left_end, len(word) - max_frag_len), len(word) + 1):
            right_frag = word[right_start:] if right_start < len(word) else ""

            if right_frag and right_frag not in valid_right_frags:
                continue

            # Middle part: must segment entirely into card words
            middle = word[left_end:right_start]

            if not middle:
                # No interior — just two fragments = only 2 cards, not enough
                continue

            for interior in _segment_into_cards(middle):
                if not interior:
                    continue
                decomp = Decomposition(left_frag, interior, right_frag)
                if decomp.num_cards >= 3:
                    results.append(decomp)
                    if len(results) >= MAX_DECOMPOSITIONS_PER_WORD:
                        return results

    return results


# ============================================================================ #
#                              PRECOMPUTATION                                  #
# ============================================================================ #

def get_cache_key() -> str:
    config_str = (
        f"puzzle2v2_{N_WORDS}_{CARD_MIN_LENGTH}_{CARD_MAX_LENGTH}_"
        f"{TARGET_MIN_LENGTH}_{NO_SINGLE_LETTERS}_{sorted(ALLOWED_1_LETTER)}_{sorted(ALLOWED_2_LETTER)}"
    )
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f".cache_puzzle2_{cache_hash}.pkl"


def build_decomposition_table(
    target_words: Set[str],
    valid_left_frags: Set[str],
    valid_right_frags: Set[str],
) -> Dict[str, List[Decomposition]]:
    """
    For every target word, find all valid decompositions.
    Returns only words that have at least one valid decomposition.
    """
    table: Dict[str, List[Decomposition]] = {}

    for word in tqdm(sorted(target_words), desc="Decomposing words",
                     ascii=" ▖▘▝▗▚▞█", bar_format='{desc}: |{bar:30}| {n_fmt}/{total_fmt}'):
        decomps = decompose_word(word, valid_left_frags, valid_right_frags)
        if decomps:
            table[word] = decomps

    return table


# ============================================================================ #
#                              SCORING                                         #
# ============================================================================ #

def score_decomposition(word: str, decomp: Decomposition) -> float:
    """Score a single decomposition. Higher = more interesting for puzzles.

    Prefers: fewer pieces, longer pieces, recognizable words.
    """
    score = 1.0

    # Target word recognizability
    word_freq = get_zipf(word)
    score *= max(0.1, min(1.0, word_freq / 5.0))

    # Bonus for longer target word
    score *= len(word) ** 0.5

    pieces = decomp.all_pieces()

    # Piece quality: each piece should be long and recognizable
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

    # PENALTY for many pieces — fewer is cleaner and more interesting
    # Each piece beyond 3 costs 30%
    score *= 0.7 ** max(0, len(pieces) - 3)

    # Bonus for longer minimum piece (no tiny filler words)
    min_piece_len = min(len(p) for p in pieces)
    score *= min_piece_len ** 0.5

    # Pure decomposition bonus (cleaner, easier to use)
    if decomp.is_pure:
        score *= 1.5

    return score


def score_word(word: str, decomps: List[Decomposition]) -> float:
    if not decomps:
        return 0.0
    return max(score_decomposition(word, d) for d in decomps)


# ============================================================================ #
#                              DISPLAY                                         #
# ============================================================================ #

def best_extension_card(cards: List[str], frag: str, frag_is_prefix: bool) -> str:
    """Pick the most recognizable extension card for a fragment."""
    # Score by: card frequency + leftover frequency
    def card_score(card):
        if frag_is_prefix:
            leftover = card[len(frag):]
        else:
            leftover = card[:-len(frag)]
        return get_zipf(card) + get_zipf(leftover)
    return max(cards, key=card_score)


def format_decomposition(
    word: str,
    decomp: Decomposition,
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
) -> str:
    """Format a decomposition for display, showing extension cards."""
    parts = []

    if decomp.left_frag:
        cards = left_frag_cards.get(decomp.left_frag, [])
        if cards:
            best = best_extension_card(cards, decomp.left_frag, frag_is_prefix=False)
            leftover = best[:-len(decomp.left_frag)]
            parts.append(f"[{leftover}]{decomp.left_frag}")
        else:
            parts.append(f"?{decomp.left_frag}")

    parts.extend(decomp.interior)

    if decomp.right_frag:
        cards = right_frag_cards.get(decomp.right_frag, [])
        if cards:
            best = best_extension_card(cards, decomp.right_frag, frag_is_prefix=True)
            leftover = best[len(decomp.right_frag):]
            parts.append(f"{decomp.right_frag}[{leftover}]")
        else:
            parts.append(f"{decomp.right_frag}?")

    return " + ".join(parts)


def display_results(
    table: Dict[str, List[Decomposition]],
    left_frag_cards: Dict[str, List[str]],
    right_frag_cards: Dict[str, List[str]],
    top_n: int = 100,
):
    print(f"\n{'='*70}")
    print(f"DECOMPOSITION RESULTS")
    print(f"{'='*70}")
    print(f"Total decomposable words: {len(table)}")

    # Count pure vs extended
    pure_count = sum(1 for decomps in table.values() if any(d.is_pure for d in decomps))
    extended_only = len(table) - pure_count
    print(f"  Words with pure decompositions: {pure_count}")
    print(f"  Words with extended-only decompositions: {extended_only}")

    # Score and rank
    scored = [(word, decomps, score_word(word, decomps)) for word, decomps in table.items()]
    scored.sort(key=lambda x: -x[2])

    # --- Top by score ---
    print(f"\n--- Top {top_n} by score ---")
    for i, (word, decomps, sc) in enumerate(scored[:top_n]):
        best = max(decomps, key=lambda d: score_decomposition(word, d))
        fmt = format_decomposition(word, best, left_frag_cards, right_frag_cards)
        tag = "pure" if best.is_pure else "ext"
        print(f"  {i+1:3d}. {word:<20} ({len(word):2d}) -> {fmt:<45} [{tag}, {len(decomps)} way(s)] score={sc:.3f}")

    # --- Longest ---
    by_length = sorted(scored, key=lambda x: (-len(x[0]), -x[2]))
    print(f"\n--- Top {top_n} longest decomposable words ---")
    for i, (word, decomps, sc) in enumerate(by_length[:top_n]):
        best = max(decomps, key=lambda d: score_decomposition(word, d))
        fmt = format_decomposition(word, best, left_frag_cards, right_frag_cards)
        tag = "pure" if best.is_pure else "ext"
        print(f"  {i+1:3d}. {word:<20} ({len(word):2d}) -> {fmt:<45} [{tag}] score={sc:.3f}")

    # --- Extended-only (words that ONLY work with fragments) ---
    extended_only_words = [
        (word, decomps, sc) for word, decomps, sc in scored
        if not any(d.is_pure for d in decomps)
    ]
    extended_only_words.sort(key=lambda x: -x[2])
    print(f"\n--- Top {top_n} extended-only words (need fragments, no pure decomposition) ---")
    for i, (word, decomps, sc) in enumerate(extended_only_words[:top_n]):
        best = max(decomps, key=lambda d: score_decomposition(word, d))
        fmt = format_decomposition(word, best, left_frag_cards, right_frag_cards)
        print(f"  {i+1:3d}. {word:<20} ({len(word):2d}) -> {fmt:<45} [{len(decomps)} way(s)] score={sc:.3f}")

    # --- Stats ---
    from collections import Counter
    lengths = [len(w) for w in table]
    decomp_counts = [len(d) for d in table.values()]
    length_dist = Counter(lengths)
    print(f"\n--- Statistics ---")
    print(f"  Word lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    print(f"  Decompositions per word: min={min(decomp_counts)}, max={max(decomp_counts)}, avg={sum(decomp_counts)/len(decomp_counts):.1f}")
    print(f"\n  Length distribution:")
    for length in sorted(length_dist):
        print(f"    {length:2d} chars: {length_dist[length]:5d} words")


# ============================================================================ #
#                              MAIN                                            #
# ============================================================================ #

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute word decompositions for Proseset puzzles")
    parser.add_argument('--card-min', type=int, default=CARD_MIN_LENGTH,
                        help=f'Min length for card words (default: {CARD_MIN_LENGTH})')
    parser.add_argument('--card-max', type=int, default=CARD_MAX_LENGTH,
                        help=f'Max length for card words (default: {CARD_MAX_LENGTH})')
    parser.add_argument('--min-length', type=int, default=TARGET_MIN_LENGTH,
                        help=f'Min length for target words (default: {TARGET_MIN_LENGTH})')
    parser.add_argument('--top', type=int, default=100,
                        help='Number of results to show per category (default: 100)')
    parser.add_argument('--no-single-letters', action='store_true',
                        help='Ban single-letter words (a, i) from appearing as isolated leftovers')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force rebuild, ignoring cache')
    return parser.parse_args()


def main():
    args = parse_args()

    global CARD_MIN_LENGTH, CARD_MAX_LENGTH, TARGET_MIN_LENGTH, NO_SINGLE_LETTERS
    CARD_MIN_LENGTH = args.card_min
    CARD_MAX_LENGTH = args.card_max
    TARGET_MIN_LENGTH = args.min_length
    NO_SINGLE_LETTERS = args.no_single_letters

    cache_path = get_cache_key()

    table = None
    left_frag_cards = None
    right_frag_cards = None

    if not args.no_cache and os.path.exists(cache_path):
        try:
            print(f"Loading cache from {cache_path}...")
            with open(cache_path, 'rb') as f:
                table, left_frag_cards, right_frag_cards = pickle.load(f)
            print(f"Loaded {len(table)} decomposable words from cache")
        except Exception as e:
            print(f"Cache load failed: {e}, rebuilding...")
            table = None

    if table is None:
        print("Loading dictionary...")
        seg_words, card_words, target_words = load_dictionary()
        print(f"Seg words: {len(seg_words)}, Card words: {len(card_words)}, Target words: {len(target_words)}")

        init_word_sets(seg_words, card_words)

        print("\nBuilding fragment indexes...")
        valid_left_frags, valid_right_frags, left_frag_cards, right_frag_cards = build_fragment_indexes(card_words)
        print(f"Valid left fragments: {len(valid_left_frags)}")
        print(f"Valid right fragments: {len(valid_right_frags)}")

        print("\nBuilding decomposition table...")
        table = build_decomposition_table(target_words, valid_left_frags, valid_right_frags)

        print(f"\nSaving cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump((table, left_frag_cards, right_frag_cards), f)
        print("Cache saved!")

    display_results(table, left_frag_cards, right_frag_cards, top_n=args.top)


if __name__ == "__main__":
    main()
