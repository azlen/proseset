#!/usr/bin/env python3
"""Utility script for Proseset prototyping.

Features
- Loads a dictionary (default: macOS ``/usr/share/dict/words``).
- Builds forward and reverse tries and serializes them to ``artifacts/``.
- Provides an interactive prompt to explore alternate segmentations of words,
  where interior segments must be valid dictionary words while edge fragments
  may be arbitrary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

DEFAULT_WORDLIST = Path("/usr/share/dict/words")
SEGMENTATION_LIMIT = 200  # Prevent runaway output for highly segmentable words.


class TrieNode:
    """Basic trie node that stores children and word termination flag."""

    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.terminal: bool = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.terminal = True

    def to_dict(self) -> Dict[str, object]:
        """Convert trie to a JSON-serializable nested dict."""
        node_dict: Dict[str, object] = {"#": int(self.terminal)}
        for ch in sorted(self.children):
            node_dict[ch] = self.children[ch].to_dict()
        return node_dict

    def count_nodes(self) -> int:
        return 1 + sum(child.count_nodes() for child in self.children.values())


def load_words(
    path: Path,
    *,
    lowercase: bool = True,
    min_length: int = 1,
    allow_single_letters: bool = True,
    alphabetic_only: bool = True,
    allowed_single_letters: Iterable[str] | None = None,
) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {path}")

    words = set()
    allowed_single_letters_set: Set[str] | None = None
    if allow_single_letters:
        if allowed_single_letters is None:
            allowed_single_letters = {"a", "i"}
        allowed_single_letters_set = {
            letter.lower() for letter in allowed_single_letters
        }
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            word = raw.strip()
            if not word:
                continue
            if lowercase:
                word = word.lower()
            if alphabetic_only and not word.isalpha():
                continue
            if len(word) < min_length:
                continue
            if len(word) == 1:
                if not allow_single_letters:
                    continue
                assert allowed_single_letters_set is not None
                if word.lower() not in allowed_single_letters_set:
                    continue
            words.add(word)
    return sorted(words)


def build_tries(words: Iterable[str]) -> tuple[TrieNode, TrieNode]:
    forward = TrieNode()
    reverse = TrieNode()
    for word in words:
        forward.insert(word)
        reverse.insert(word[::-1])
    return forward, reverse


def serialize_trie(trie: TrieNode, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(trie.to_dict(), handle, ensure_ascii=False)


def collect_dictionary_sequences(
    word: str,
    start: int,
    word_set: Set[str],
    memo: Dict[int, List[Tuple[int, List[str]]]],
) -> List[Tuple[int, List[str]]]:
    """Return all (end_index, segments) covering word[start:end_index] with dictionary words."""
    if start in memo:
        return memo[start]

    results: List[Tuple[int, List[str]]] = []
    length = len(word)
    for end in range(start + 1, length + 1):
        segment = word[start:end]
        if segment in word_set:
            results.append((end, [segment]))
            for further_end, tail in collect_dictionary_sequences(word, end, word_set, memo):
                results.append((further_end, [segment] + tail))
    memo[start] = results
    return results


def segment_word(
    word: str,
    word_set: Set[str],
    *,
    require_valid_start: bool = False,
    require_valid_end: bool = False,
) -> List[List[str]]:
    """Return all segmentations where interior parts are dictionary words.

    ``require_valid_start`` enforces that the first segment in the result (if any
    prefix fragment exists) must be a dictionary word. ``require_valid_end``
    mirrors this constraint for any trailing suffix fragment.
    """
    if not word:
        return []

    memo: Dict[int, List[Tuple[int, List[str]]]] = {}
    seen: Set[Tuple[str, ...]] = set()
    all_segmentations: List[List[str]] = []
    length = len(word)

    for start in range(length + 1):
        prefix = word[:start]
        for end, segments in collect_dictionary_sequences(word, start, word_set, memo):
            if not segments:
                continue
            suffix = word[end:]
            if require_valid_start and prefix and prefix not in word_set:
                continue
            if require_valid_end and suffix and suffix not in word_set:
                continue
            assembled: List[str] = []
            if prefix:
                assembled.append(prefix)
            assembled.extend(segments)
            if suffix:
                assembled.append(suffix)
            key = tuple(assembled)
            if key not in seen:
                seen.add(key)
                all_segmentations.append(assembled)
            if len(all_segmentations) >= SEGMENTATION_LIMIT:
                return all_segmentations
    return all_segmentations


def interactive_loop(
    word_set: Set[str],
    *,
    require_valid_start: bool = False,
    require_valid_end: bool = False,
) -> None:
    print("Enter a word to view segmentations (blank line to quit):")
    while True:
        try:
            raw = input("word> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        word = raw.strip()
        if not word:
            break
        lookup = word.lower()
        segmentations = segment_word(
            lookup,
            word_set,
            require_valid_start=require_valid_start,
            require_valid_end=require_valid_end,
        )
        if not segmentations:
            print("  (no valid segmentations)")
            continue
        for idx, segmentation in enumerate(segmentations, start=1):
            print(f"  {idx:>3}: {' | '.join(segmentation)}")
        if len(segmentations) >= SEGMENTATION_LIMIT:
            print(f"  ...truncated after {SEGMENTATION_LIMIT} segmentations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tries and explore segmentations")
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=DEFAULT_WORDLIST,
        help=f"Path to input word list (default: {DEFAULT_WORDLIST})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write the serialized tries",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Exclude words shorter than this length",
    )
    parser.add_argument(
        "--disallow-single-letters",
        action="store_true",
        help="Drop single-letter words from the dictionary",
    )
    parser.add_argument(
        "--keep-non-alpha",
        action="store_true",
        help="Retain words containing non-alphabetic characters",
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Preserve original casing from the word list",
    )
    parser.add_argument(
        "--skip-serialization",
        action="store_true",
        help="Avoid writing trie JSON artifacts (still builds tries in memory)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the interactive segmentation prompt",
    )
    parser.add_argument(
        "--require-valid-start",
        action="store_true",
        help="Force the first segment in each result to be a dictionary word",
    )
    parser.add_argument(
        "--require-valid-end",
        action="store_true",
        help="Force the last segment in each result to be a dictionary word",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    words = load_words(
        args.dictionary,
        lowercase=not args.no_lowercase,
        min_length=args.min_length,
        allow_single_letters=not args.disallow_single_letters,
        alphabetic_only=not args.keep_non_alpha,
    )
    print(f"Loaded {len(words):,} unique words from {args.dictionary}")

    forward, reverse = build_tries(words)
    print(
        "Forward trie nodes:",
        f"{forward.count_nodes():,}",
        "| Reverse trie nodes:",
        f"{reverse.count_nodes():,}",
    )

    if not args.skip_serialization:
        output_dir: Path = args.output_dir
        serialize_trie(forward, output_dir / "forward_trie.json")
        serialize_trie(reverse, output_dir / "reverse_trie.json")
        print(f"Serialized tries to {output_dir.resolve()}")

    if not args.no_interactive:
        interactive_loop(
            set(words),
            require_valid_start=args.require_valid_start,
            require_valid_end=args.require_valid_end,
        )


if __name__ == "__main__":
    main()
