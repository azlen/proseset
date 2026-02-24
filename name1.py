#!/usr/bin/env python3
"""
name1.py - Find compound strings that can be split into 2 words multiple ways.

For naming the game: find strings where the same letters can be divided
into different pairs of valid words.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
from wordfreq import zipf_frequency
import third_party.twl as twl

from run15 import (
    load_dictionary, set_segmentation_dict, segment_word,
    get_zipf, ALLOWED_1_LETTER, ALLOWED_2_LETTER, BANNED_WORDS,
)

# Words/roots thematically related to the game concept
# (word-building, composition, language, puzzles, sets)
THEMATIC_ROOTS = {
    'word', 'spell', 'prose', 'letter', 'text', 'set', 'piece',
    'part', 'form', 'compose', 'build', 'craft', 'make', 'bond',
    'link', 'join', 'merge', 'blend', 'fuse', 'match', 'pair',
    'split', 'break', 'snap', 'lock', 'fit', 'tile', 'block',
    'stack', 'layer', 'cross', 'over', 'under', 'inter', 'com',
    'con', 'compact', 'compound', 'note', 'page', 'book', 'read',
    'speak', 'verse', 'script', 'write', 'pen', 'type', 'print',
    'press', 'mark', 'sign', 'code', 'play', 'game', 'puzzle',
    'card', 'deck', 'hand', 'deal', 'draw', 'pick', 'turn',
    'round', 'score', 'point', 'win', 'solve', 'find', 'seek',
    'hunt', 'quest', 'combo', 'chain', 'string', 'band', 'strip',
    'span', 'reach', 'bridge', 'gate', 'pass', 'path', 'way',
    'forge', 'cast', 'mold', 'shape', 'cut', 'slice', 'carve',
}


def is_trivial_s_split(splits: list[tuple[str, str]]) -> bool:
    """Check if the only difference between splits is an s-plural shift."""
    positions = sorted(len(left) for left, _ in splits)
    if len(positions) < 2:
        return True

    # Check each adjacent pair of split positions
    # If they differ by exactly 1 and it's just 's' moving between sides, it's trivial
    trivial_count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[j] - positions[i] == 1:
                # Find the splits at these positions
                left_i = [l for l, r in splits if len(l) == positions[i]][0]
                left_j = [l for l, r in splits if len(l) == positions[j]][0]
                # Check if it's just adding 's' to the left word
                if left_j == left_i + 's':
                    trivial_count += 1

    # If ALL pairs of adjacent splits are trivial s-shifts, reject
    non_trivial_pairs = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[j] - positions[i] != 1:
                non_trivial_pairs += 1
            else:
                left_i = [l for l, r in splits if len(l) == positions[i]][0]
                left_j = [l for l, r in splits if len(l) == positions[j]][0]
                if left_j != left_i + 's':
                    non_trivial_pairs += 1

    return non_trivial_pairs == 0


def has_thematic_relevance(compound: str, splits: list[tuple[str, str]]) -> float:
    """Score thematic relevance to the game concept."""
    bonus = 0.0
    compound_lower = compound.lower()

    # Check if compound contains thematic roots
    for root in THEMATIC_ROOTS:
        if root in compound_lower:
            bonus += 2.0
            break

    # Check if any split words are thematic
    for left, right in splits:
        if left in THEMATIC_ROOTS or right in THEMATIC_ROOTS:
            bonus += 1.0

    return bonus


def main():
    parser = argparse.ArgumentParser(description='Find compound word name candidates')
    parser.add_argument('--min-len', type=int, default=6, help='Min compound length')
    parser.add_argument('--max-len', type=int, default=12, help='Max compound length')
    parser.add_argument('--min-zipf', type=float, default=3.0, help='Min zipf freq for component words')
    parser.add_argument('--min-splits', type=int, default=2, help='Min number of distinct split points')
    parser.add_argument('--word-min', type=int, default=3, help='Min component word length')
    parser.add_argument('--word-max', type=int, default=7, help='Max component word length')
    parser.add_argument('--top', type=int, default=100, help='Show top N results')
    parser.add_argument('--real-only', action='store_true', help='Only show real compound words')
    parser.add_argument('--thematic', action='store_true', help='Boost thematic relevance')
    parser.add_argument('--no-s-splits', action='store_true', default=True,
                        help='Filter trivial s-plural splits (default: on)')
    parser.add_argument('--allow-s-splits', action='store_true', help='Allow s-plural splits')
    parser.add_argument('--ban', nargs='*', default=[], help='Ban specific words from results')
    parser.add_argument('--require-real', action='store_true', help='Require compound to be a real word')
    parser.add_argument('--save', type=str, default=None, help='Save results to file')
    parser.add_argument('--no-boring', action='store_true',
                        help='Filter splits where one side is a very common function word')
    args = parser.parse_args()

    filter_s = not args.allow_s_splits

    # Function words that make compounds feel like random glue
    BORING_WORDS = {
        'the', 'for', 'you', 'your', 'who', 'was', 'has', 'her', 'his', 'its',
        'our', 'are', 'were', 'had', 'not', 'but', 'may', 'can', 'will',
        'she', 'him', 'them', 'this', 'that', 'with', 'been', 'have',
        'and', 'they', 'very', 'every', 'any', 'all', 'hey', 'yes',
        'too', 'now', 'how', 'just', 'also', 'here', 'there',
        'what', 'when', 'why', 'did', 'does', 'done', 'got', 'get',
    }

    # Merge user bans
    banned_compounds = set(w.lower() for w in args.ban)

    print("Loading dictionary...")
    _, seg_words = load_dictionary()

    all_words = (
        ALLOWED_1_LETTER | ALLOWED_2_LETTER |
        {w for w in twl.iterator() if w.isalpha() and len(w) > 2}
    ) - BANNED_WORDS

    common = {w for w in all_words if get_zipf(w) >= args.min_zipf and len(w) >= 2}
    print(f"Common words (zipf>={args.min_zipf}): {len(common)}")

    set_segmentation_dict(all_words)

    short = sorted([w for w in common if args.word_min <= len(w) <= args.word_max])
    print(f"Component words (len {args.word_min}-{args.word_max}): {len(short)}")

    # Generate all compound pairs, group by compound string
    compound_splits = defaultdict(list)
    for left in short:
        for right in short:
            compound = left + right
            if len(compound) < args.min_len or len(compound) > args.max_len:
                continue
            compound_splits[compound].append((left, right))

    print(f"Unique compounds: {len(compound_splits)}")

    # Filter and score
    results = []
    for compound, splits in compound_splits.items():
        if compound in banned_compounds:
            continue

        # Deduplicate by split position (keep best freq pair at each position)
        split_positions = {}
        for left, right in splits:
            pos = len(left)
            if pos not in split_positions:
                split_positions[pos] = (left, right)
            else:
                # Keep the pair with higher min frequency
                old_min = min(get_zipf(split_positions[pos][0]), get_zipf(split_positions[pos][1]))
                new_min = min(get_zipf(left), get_zipf(right))
                if new_min > old_min:
                    split_positions[pos] = (left, right)

        if len(split_positions) < args.min_splits:
            continue

        all_splits = list(split_positions.values())

        # Filter trivial s-plural splits
        if filter_s and is_trivial_s_split(all_splits):
            continue

        # Filter boring function word glue
        if args.no_boring:
            has_boring = any(
                l in BORING_WORDS or r in BORING_WORDS
                for l, r in all_splits
            )
            if has_boring:
                continue

        # Check if compound itself is a real word
        is_word = compound in all_words

        if args.require_real and not is_word:
            continue

        # Find 3+ word segmentations
        multi_segs = segment_word(compound, require_start=True, require_end=True)
        good_multi = [
            seg for seg in multi_segs
            if len(seg) >= 3 and all(get_zipf(w) >= 2.5 for w in seg)
        ]

        avg_freq = sum(get_zipf(w) for s in all_splits for w in s) / (len(all_splits) * 2)
        min_freq = min(get_zipf(w) for s in all_splits for w in s)

        # Scoring
        score = 0.0
        score += len(all_splits) * 3.0          # More splits = better
        score += avg_freq                        # Higher freq words = better
        score += 3.0 if is_word else 0           # Real word bonus
        score += len(good_multi) * 0.5           # Multi-seg bonus
        if min_freq >= 3.5:
            score += 2.0                         # All words well-known

        # Length preference: shorter compounds are punchier for a game name
        if len(compound) <= 8:
            score += 1.5
        elif len(compound) <= 10:
            score += 0.5

        # Thematic relevance
        if args.thematic:
            score += has_thematic_relevance(compound, all_splits)

        results.append({
            'compound': compound,
            'splits': all_splits,
            'multi_segs': good_multi[:5],
            'is_word': is_word,
            'score': score,
            'min_freq': min_freq,
            'avg_freq': avg_freq,
        })

    results.sort(key=lambda x: -x['score'])

    print(f"\nFound {len(results)} compounds with {args.min_splits}+ non-trivial split points")

    def format_entry(i, r):
        compound = r['compound']
        real_tag = " *" if r['is_word'] else ""
        splits_str = " / ".join(f"{a}+{b}" for a, b in r['splits'])
        multi_str = ""
        if r['multi_segs']:
            multi_str = "  also: " + " ; ".join(" | ".join(seg) for seg in r['multi_segs'][:2])
        return f"{i+1:4d}. {compound.upper()}{real_tag:3s} ({len(r['splits'])}splits)  {splits_str}{multi_str}"

    if args.save:
        with open(args.save, 'w') as f:
            f.write(f"# Name candidates — {len(results)} results\n")
            f.write(f"# * = real dictionary word\n")
            f.write(f"# Filters: len {args.min_len}-{args.max_len}, "
                    f"zipf>={args.min_zipf}, "
                    f"{'no-boring ' if args.no_boring else ''}"
                    f"{'real-only ' if args.require_real else ''}\n")
            f.write(f"#\n")
            for i, r in enumerate(results[:args.top]):
                f.write(format_entry(i, r) + "\n")
        print(f"Saved {min(args.top, len(results))} results to {args.save}")
    else:
        print(f"\n{'='*70}")
        print("TOP RESULTS")
        print(f"{'='*70}")

        for i, r in enumerate(results[:args.top]):
            compound = r['compound']
            tags = []
            if r['is_word']:
                tags.append("WORD")
            print(f"\n{i+1:3d}. {compound.upper()}  {'[' + ', '.join(tags) + ']' if tags else ''}  "
                  f"({len(r['splits'])} splits, score={r['score']:.1f})")

            for a, b in r['splits']:
                za, zb = get_zipf(a), get_zipf(b)
                print(f"     {a:<12} + {b:<12}  (zipf: {za:.1f}, {zb:.1f})")

            for seg in r['multi_segs'][:2]:
                print(f"     {' | '.join(seg)}")


if __name__ == "__main__":
    main()
