#!/usr/bin/env python3
"""
Export Proseset game data as JSON for the web-based puzzle designer.

Reads:
  - puzzle2-stub pickle cache (decomposition table, fragment indexes)
  - Dictionary (TWL + wordfreq filtering)
  - Word frequencies

Writes:
  - puzzle-designer/public/data.json
"""

from __future__ import annotations
import json
import pickle
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from wordfreq import top_n_list, zipf_frequency
import third_party.twl as twl

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


# ============================================================================ #
#                              DECOMPOSITION                                   #
# ============================================================================ #

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
    def find_class(self, module, name):
        if name == 'Decomposition':
            return Decomposition
        return super().find_class(module, name)


def score_decomposition(word: str, decomp: Decomposition) -> float:
    score = 1.0
    word_freq = zipf_frequency(word, 'en')
    score *= max(0.1, min(1.0, word_freq / 5.0))
    score *= len(word) ** 0.5

    pieces = decomp.all_pieces()
    for piece in pieces:
        freq = zipf_frequency(piece, 'en')
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


# ============================================================================ #
#                              MAIN                                            #
# ============================================================================ #

def main():
    # Load dictionary
    print("Loading dictionary...")
    top_words = set(top_n_list('en', N_WORDS, wordlist='best'))
    seg_words = (
        ALLOWED_1_LETTER | ALLOWED_2_LETTER |
        {w for w in top_words if twl.check(w) and w.isalpha() and len(w) > 2}
    ) - BANNED_WORDS - WORD_BLACKLIST
    deck_words = {w for w in seg_words if MIN_WORD_LENGTH <= len(w) <= MAX_WORD_LENGTH}

    # Filter deck words same as puzzle builders
    deck_words_filtered = {w for w in deck_words if not (w.startswith('s') or w.endswith('s'))}
    deck_words_filtered = {w for w in deck_words_filtered if not ('ai' in w or 'ia' in w)}

    print(f"Seg words: {len(seg_words)}, Deck words: {len(deck_words_filtered)}")

    # Word frequencies
    print("Computing frequencies...")
    frequencies = {}
    all_words = seg_words | deck_words_filtered
    for w in all_words:
        frequencies[w] = round(zipf_frequency(w, 'en'), 2)

    # Load decomposition cache
    print("Loading decomposition cache...")
    caches = glob.glob(".cache_puzzle2_*.pkl")
    if not caches:
        print("ERROR: No puzzle2-stub cache found. Run puzzle2-stub.py first.")
        sys.exit(1)

    caches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    table = None
    left_frag_cards = None
    right_frag_cards = None

    for cache_path in caches:
        try:
            print(f"Trying {cache_path}...")
            with open(cache_path, 'rb') as f:
                data = _DecompUnpickler(f).load()
            if isinstance(data, tuple) and len(data) == 3:
                table, left_frag_cards, right_frag_cards = data
                print(f"Loaded {len(table)} decomposable words")
                break
        except Exception as e:
            print(f"  Skipping: {e}")

    if table is None:
        print("ERROR: No valid puzzle2-stub cache found.")
        sys.exit(1)

    # Convert decompositions to JSON-serializable format
    # Keep top 5 scored decompositions per word, only those using valid deck words
    print("Processing decompositions...")
    decompositions = {}
    for word, decomps in table.items():
        scored = []
        for d in decomps:
            # Check interior cards are valid deck words
            if not all(c in deck_words for c in d.interior):
                continue
            sc = score_decomposition(word, d)
            scored.append((d, sc))

        scored.sort(key=lambda x: -x[1])
        top_decomps = []
        for d, sc in scored[:5]:
            top_decomps.append({
                'leftFrag': d.left_frag,
                'interior': list(d.interior),
                'rightFrag': d.right_frag,
                'score': round(sc, 4),
            })

        if top_decomps:
            decompositions[word] = top_decomps

    print(f"Decompositions: {len(decompositions)} target words")

    # Convert fragment card indexes
    # Filter to only include cards that are in deck_words
    lfc = {}
    for frag, cards in left_frag_cards.items():
        valid = [c for c in cards if c in deck_words]
        if valid:
            lfc[frag] = valid

    rfc = {}
    for frag, cards in right_frag_cards.items():
        valid = [c for c in cards if c in deck_words]
        if valid:
            rfc[frag] = valid

    print(f"Left fragment cards: {len(lfc)} fragments")
    print(f"Right fragment cards: {len(rfc)} fragments")

    # Build output
    output = {
        'deckWords': sorted(deck_words_filtered),
        'segWords': sorted(seg_words),
        'frequencies': frequencies,
        'decompositions': decompositions,
        'leftFragCards': lfc,
        'rightFragCards': rfc,
    }

    # Write to file
    out_path = os.path.join('puzzle-designer', 'public', 'data.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Writing {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    file_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done! {file_size:.1f} MB")


if __name__ == '__main__':
    main()
