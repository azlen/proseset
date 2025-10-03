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
import importlib.util
import json
from pathlib import Path
from random import choice
from typing import Dict, Iterable, Iterator, List, Set, Tuple

DEFAULT_WORDLIST = Path("/usr/share/dict/words")
DEFAULT_SINGLE_LETTERS = {"a", "i"}
DEFAULT_MAX_LENGTH = 7
CANDIDATE_DISPLAY_LIMIT = 20
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


def filter_words(
    words: Iterable[str],
    *,
    lowercase: bool = True,
    min_length: int = 1,
    max_length: int | None = None,
    allow_single_letters: bool = True,
    alphabetic_only: bool = True,
    allowed_single_letters: Iterable[str] | None = None,
    sort_output: bool = True,
) -> List[str]:
    allowed_single_letters_set: Set[str] | None = None
    if allow_single_letters:
        if allowed_single_letters is None:
            allowed_single_letters = DEFAULT_SINGLE_LETTERS
        allowed_single_letters_set = {
            letter.lower() for letter in allowed_single_letters
        }

    seen: Set[str] = set()
    output: List[str] = []

    for raw in words:
        word = raw.strip()
        if not word:
            continue
        if lowercase:
            word = word.lower()
        if alphabetic_only and not word.isalpha():
            continue
        if len(word) < min_length:
            continue
        if max_length is not None and len(word) > max_length:
            continue
        if len(word) == 1:
            if not allow_single_letters:
                continue
            assert allowed_single_letters_set is not None
            if word.lower() not in allowed_single_letters_set:
                continue
        if word in seen:
            continue
        seen.add(word)
        output.append(word)

    if sort_output:
        output.sort()
    return output


def load_words(
    path: Path,
    *,
    lowercase: bool = True,
    min_length: int = 1,
    max_length: int | None = None,
    allow_single_letters: bool = True,
    alphabetic_only: bool = True,
    allowed_single_letters: Iterable[str] | None = None,
) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return filter_words(
            handle,
            lowercase=lowercase,
            min_length=min_length,
            max_length=max_length,
            allow_single_letters=allow_single_letters,
            alphabetic_only=alphabetic_only,
            allowed_single_letters=allowed_single_letters,
            sort_output=True,
        )


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


def load_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


_TWL_WORD_CACHE: Dict[Path, List[str]] = {}


def load_twl_words(twl_path: Path) -> List[str]:
    resolved = twl_path.resolve()
    if resolved in _TWL_WORD_CACHE:
        return _TWL_WORD_CACHE[resolved]

    if not twl_path.exists():
        raise FileNotFoundError(
            f"TWL source not found at {twl_path}. Provide --twl-path pointing to twl.py or a word list."
        )

    words: List[str] = []
    if twl_path.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("twl_module", resolved)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"Unable to import TWL module from {twl_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        if hasattr(module, "iterator"):
            iterable = module.iterator()
        elif hasattr(module, "WORDS"):
            iterable = getattr(module, "WORDS")  # type: ignore[assignment]
        else:  # pragma: no cover - defensive
            raise AttributeError("TWL module missing 'iterator' or 'WORDS'")
        for entry in iterable:
            if not isinstance(entry, str):
                continue
            clean = entry.strip().lower()
            if clean:
                words.append(clean)
    else:
        for entry in load_lines(twl_path):
            clean = entry.strip().lower()
            if clean:
                words.append(clean)

    _TWL_WORD_CACHE[resolved] = words
    return words


def build_wordfreq_twl_dictionary(
    *,
    limit: int,
    twl_path: Path,
    lowercase: bool = True,
    min_length: int = 1,
    max_length: int | None = None,
    allow_single_letters: bool = True,
    alphabetic_only: bool = True,
    allowed_single_letters: Iterable[str] | None = None,
) -> List[str]:
    try:
        from wordfreq import top_n_list
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "wordfreq package is required for --dictionary-source=wordfreq_twl"
        ) from exc

    twl_words = {word for word in load_twl_words(twl_path) if word.isalpha()}

    top_words = top_n_list("en", limit, wordlist="best")
    filtered = [word for word in top_words if word in twl_words]

    processed = filter_words(
        filtered,
        lowercase=lowercase,
        min_length=min_length,
        max_length=max_length,
        allow_single_letters=allow_single_letters,
        alphabetic_only=alphabetic_only,
        allowed_single_letters=allowed_single_letters,
        sort_output=False,
    )

    if allow_single_letters:
        single_letters = allowed_single_letters or DEFAULT_SINGLE_LETTERS
        for letter in single_letters:
            if len(letter) == 1 and letter not in processed:
                processed.append(letter)

    processed.sort()

    return processed


def get_zipf_frequency(word: str, wordlist: str = "best") -> float:
    """Return the wordfreq Zipf frequency for the given word."""

    if not hasattr(get_zipf_frequency, "_cache"):
        get_zipf_frequency._cache = {}  # type: ignore[attr-defined]

    cache: Dict[Tuple[str, str], float] = get_zipf_frequency._cache  # type: ignore[attr-defined]
    key = (word, wordlist)
    if key in cache:
        return cache[key]

    try:
        from wordfreq import zipf_frequency
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("wordfreq package is required to fetch word frequencies") from exc

    value = zipf_frequency(word, "en", wordlist=wordlist)
    cache[key] = value
    return value

def get_word_frequency(word: str, wordlist: str = "best") -> float:
    """Return the wordfreq Zipf frequency for the given word."""

    if not hasattr(get_word_frequency, "_cache"):
        get_word_frequency._cache = {}  # type: ignore[attr-defined]

    cache: Dict[Tuple[str, str], float] = get_word_frequency._cache  # type: ignore[attr-defined]
    key = (word, wordlist)
    if key in cache:
        return cache[key]

    try:
        from wordfreq import word_frequency
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("wordfreq package is required to fetch word frequencies") from exc

    value = word_frequency(word, "en", wordlist=wordlist)
    cache[key] = value
    return value

def find_trie_node(trie: TrieNode, prefix: str) -> TrieNode | None:
    node = trie
    for ch in prefix:
        node = node.children.get(ch)
        if node is None:
            return None
    return node


def trie_has_prefix(trie: TrieNode, prefix: str) -> bool:
    return find_trie_node(trie, prefix) is not None


def iter_words_with_prefix(trie: TrieNode, prefix: str) -> Iterator[str]:
    node = find_trie_node(trie, prefix)
    if node is None:
        return iter(())

    def dfs(current: TrieNode, suffix: str) -> Iterator[str]:
        if current.terminal:
            yield prefix + suffix
        for ch, child in current.children.items():
            yield from dfs(child, suffix + ch)

    return dfs(node, "")


def trie_has_suffix(reverse_trie: TrieNode, suffix: str) -> bool:
    return trie_has_prefix(reverse_trie, suffix[::-1])


def iter_words_with_suffix(reverse_trie: TrieNode, suffix: str) -> Iterator[str]:
    reversed_prefix = suffix[::-1]
    node = find_trie_node(reverse_trie, reversed_prefix)
    if node is None:
        return iter(())

    def dfs(current: TrieNode, tail: str) -> Iterator[str]:
        if current.terminal:
            yield (reversed_prefix + tail)[::-1]
        for ch, child in current.children.items():
            yield from dfs(child, tail + ch)

    return dfs(node, "")


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


def debug_extension_process(word: str, word_set: Set[str], trie: TrieNode) -> None:
    print(f"    [debug] analyzing '{word}' with require_valid_start=True")
    segmentations = segment_word(word, word_set, require_valid_start=True)
    if not segmentations:
        print("      no valid segmentations when enforcing start word")
        return

    added_prefixes: Set[str] = set()
    for seg_idx, segmentation in enumerate(segmentations, start=1):
        print(f"      segmentation {seg_idx}: {' | '.join(segmentation)}")
        if not segmentation:
            continue
        last_segment = segmentation[-1]
        print(f"        last segment: '{last_segment}'")
        candidates = list(iter_words_with_prefix(trie, last_segment))
        if not candidates:
            print(f"        no dictionary continuations starting with '{last_segment}'")
            continue
        print(
            f"        candidate continuations (showing up to {CANDIDATE_DISPLAY_LIMIT}):"
        )
        for candidate_idx, candidate in enumerate(candidates[:CANDIDATE_DISPLAY_LIMIT], start=1):
            remainder = candidate[len(last_segment) :]
            remainder_status = "ok" if remainder and trie_has_prefix(trie, remainder) else "skip"
            summary = remainder or "<empty>"
            print(
                f"          {candidate_idx:>3}. {candidate} | remainder='{summary}' -> {remainder_status}"
            )
            if remainder and trie_has_prefix(trie, remainder):
                added_prefixes.add(remainder)
        if len(candidates) > CANDIDATE_DISPLAY_LIMIT:
            print("          …")
    if added_prefixes:
        print(
            "      prefixes that would receive this word:",
            ", ".join(sorted(added_prefixes)),
        )
    else:
        print("      no prefixes would be updated for this word")


def interactive_loop(
    word_set: Set[str],
    trie: TrieNode,
    reverse_trie: TrieNode,
    forward_index_sets: Dict[str, Set[str]],
    backward_index_sets: Dict[str, Set[str]],
    *,
    require_valid_start: bool = False,
    require_valid_end: bool = False,
    extension_debug: bool = False,
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
            if extension_debug:
                debug_extension_process(lookup, word_set, trie)
            continue
        total_combos = 0
        total_left = 0
        total_right = 0
        combo_samples: List[Tuple[str, str, str]] = []
        for idx, segmentation in enumerate(segmentations, start=1):
            print(f"  {idx:>3}: {' | '.join(segmentation)}")
            if len(segmentation) < 2:
                continue
            first_segment = segmentation[0]
            last_segment = segmentation[-1]
            left_candidate_set: Set[str] = set()
            
            left_candidate_set = forward_index_sets.get(first_segment)

            right_candidate_set = backward_index_sets.get(last_segment)

            if not left_candidate_set or not right_candidate_set:
                continue
            left_candidates = list(left_candidate_set)
            right_candidates = list(right_candidate_set)
            total_combos += len(left_candidates) * len(right_candidates)
            total_left += len(left_candidates)
            total_right += len(right_candidates)

            if len(combo_samples) < 3:
                combo_samples.append(
                    (
                        choice(left_candidates),
                        lookup,
                        choice(right_candidates),
                    )
                )
        if len(segmentations) >= SEGMENTATION_LIMIT:
            print(f"  ...truncated after {SEGMENTATION_LIMIT} segmentations")
        if total_combos:
            print(f"  left: {total_left}")
            print(f"  right: {total_right}")
            print(f"  combinations: {total_combos}")
            for left, middle, right in combo_samples:
                print(f"    sample: {left} {middle} {right}")
        if extension_debug:
            debug_extension_process(lookup, word_set, trie)


def calculate_combinations_for_word(
    word: str,
    word_set: Set[str],
    forward_index_sets: Dict[str, Set[str]],
    backward_index_sets: Dict[str, Set[str]],
    *,
    require_valid_start: bool = False,
    require_valid_end: bool = False,
) -> Tuple[int, int, int, int, float, Dict[str, float]]:
    """Calculate combination metrics and edge weights for a single word.

    Returns:
        total_combos: Count of left-right combinations across segmentations.
        total_left: Total number of left candidates considered.
        total_right: Total number of right candidates considered.
        contributing_segmentations: Segmentations contributing combos.
        score: Heuristic score for the word (higher is better).
        edges: Mapping of neighbor word -> accumulated edge weight.
    """

    segmentations = segment_word(
        word,
        word_set,
        require_valid_start=require_valid_start,
        require_valid_end=require_valid_end,
    )

    total_combos = 0
    total_left = 0
    total_right = 0
    contributing_segmentations = 0

    score = 0.0
    word_length = len(word)
    edges: Dict[str, float] = {}

    for segmentation in segmentations:
        if len(segmentation) < 2:
            continue
        first_segment = segmentation[0]
        middle_segments = segmentation[1:-1]
        last_segment = segmentation[-1]
        left_candidates = forward_index_sets.get(first_segment)
        right_candidates = backward_index_sets.get(last_segment)
        if not left_candidates or not right_candidates:
            continue
        left_score = sum(get_zipf_frequency(candidate) * len(candidate) / (len(candidate) + word_length) for candidate in left_candidates)
        right_score = sum(get_zipf_frequency(candidate) * len(candidate) / (len(candidate) + word_length) for candidate in right_candidates)
        middle_score = sum(get_zipf_frequency(segment) * len(segment) / (len(segment) + word_length) for segment in middle_segments) if len(middle_segments) > 0 else 10

        for candidate in left_candidates:
            edges.setdefault(candidate, 0)
            edges[candidate] += get_zipf_frequency(candidate) * len(candidate) / (len(candidate) + word_length) * right_score
            if candidate[0] == 's':
                edges[candidate] = 0
        for candidate in right_candidates:
            edges.setdefault(candidate, 0)
            edges[candidate] += get_zipf_frequency(candidate) * len(candidate) / (len(candidate) + word_length) * left_score
            if candidate[0] == 's':
                edges[candidate] = 0

        score += left_score * middle_score * right_score
        left_count = len(left_candidates)
        right_count = len(right_candidates)
        if not left_count or not right_count:
            continue
        contributing_segmentations += 1
        total_left += left_count
        total_right += right_count
        total_combos += left_count * right_count

    score *= pow(word_length, 3) * get_zipf_frequency(word)

    if word[0] == 's':
        score = 0

    return total_combos, total_left, total_right, contributing_segmentations, score, edges


def walk_graph(
    start_word: str,
    word_set: Set[str],
    forward_index_sets: Dict[str, Set[str]],
    backward_index_sets: Dict[str, Set[str]],
    *,
    steps: int,
    require_valid_start: bool = False,
    require_valid_end: bool = False,
    combo_results: Dict[str, Tuple[float, int, int, int, int, int]] = {},
) -> Dict[str, object]:
    """Greedy walk across the graph using combination edges."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if start_word not in word_set:
        raise ValueError(f"start word '{start_word}' is not in the current dictionary")
    if not forward_index_sets or not backward_index_sets:
        raise ValueError("walk_graph requires populated extension indexes")

    visited: List[Dict[str, object]] = []
    visited_words: Set[str] = set()
    edge_totals: Dict[str, float] = {}

    current = start_word

    for _ in range(steps):
        combos, total_left, total_right, contributing, score, edges = calculate_combinations_for_word(
            current,
            word_set,
            forward_index_sets,
            backward_index_sets,
            require_valid_start=require_valid_start,
            require_valid_end=require_valid_end,
        )

        visited.append(
            {
                "word": current,
                "combinations": combos,
                "left_total": total_left,
                "right_total": total_right,
                "segmentations": contributing,
                "score": score,
            }
        )
        visited_words.add(current)

        for neighbor, weight in edges.items():
            if neighbor in visited_words:
                continue
            edge_totals[neighbor] = edge_totals.get(neighbor, 0.0) + weight

        edge_totals.pop(current, None)

        edge_totals_score_augmented = {
            neighbor: weight + combo_results.get(neighbor, (0.0, 0, 0, 0, 0, 0))[0]
            for neighbor, weight in edge_totals.items()
        }

        if len(visited) >= steps or not edge_totals:
            break

        import random

        next_word, _ = max(edge_totals.items(), key=lambda item: item[1])
        # sorted_edge_totals = sorted(edge_totals.items(), key=lambda item: item[1], reverse=True)
        # next_word = random.choice(sorted_edge_totals[:10])[0]
        edge_totals.pop(next_word, None)
        current = next_word

    for entry in visited_words:
        edge_totals.pop(entry, None)

    return {
        "path": visited,
        "edges": edge_totals,
    }

def build_extension_index(
    words: List[str],
    forward_trie: TrieNode,
    reverse_trie: TrieNode,
    *,
    limit: int = 1000,
) -> Dict[str, Dict[str, List[str]]]:
    """Produce remainder-keyed lookup tables describing compatible middle words."""

    subset = words[:limit]
    word_set = set(words)

    forward_index: Dict[str, Set[str]] = {}
    backward_index: Dict[str, Set[str]] = {}

    prefix_cache: Dict[str, List[str]] = {}
    suffix_cache: Dict[str, List[str]] = {}

    for word in subset:
        right_segmentations = segment_word(
            word,
            word_set,
            require_valid_start=True,
        )
        for segmentation in right_segmentations:
            if not segmentation:
                continue
            last_segment = segmentation[-1]
            candidates = prefix_cache.get(last_segment)
            if candidates is None:
                candidates = list(iter_words_with_prefix(forward_trie, last_segment))
                prefix_cache[last_segment] = candidates
            if not candidates:
                continue
            for candidate in candidates:
                remainder = candidate[len(last_segment) :]
                if not remainder:
                    continue
                if not trie_has_prefix(forward_trie, remainder):
                    continue
                forward_index.setdefault(remainder, set()).add(word)

        left_segmentations = segment_word(
            word,
            word_set,
            require_valid_end=True,
        )
        for segmentation in left_segmentations:
            if not segmentation:
                continue
            first_segment = segmentation[0]
            candidates = suffix_cache.get(first_segment)
            if candidates is None:
                candidates = list(iter_words_with_suffix(reverse_trie, first_segment))
                suffix_cache[first_segment] = candidates
            if not candidates:
                continue
            for candidate in candidates:
                remainder = candidate[: -len(first_segment)]
                if not remainder:
                    continue
                if not trie_has_suffix(reverse_trie, remainder):
                    continue
                backward_index.setdefault(remainder, set()).add(word)

    return {
        "forward": {rem: sorted(words) for rem, words in forward_index.items()},
        "backward": {rem: sorted(words) for rem, words in backward_index.items()},
    }


def serialize_extension_index(index: Dict[str, List[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tries and explore segmentations")
    parser.add_argument(
        "--dictionary-source",
        choices=["mac", "file", "wordfreq_twl"],
        default="wordfreq_twl",
        help="Which dictionary to load: Wordfreq+TWL blend (default), macOS list, or custom file",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=DEFAULT_WORDLIST,
        help=f"Path to input word list when using mac/file sources (default: {DEFAULT_WORDLIST})",
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
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Exclude words longer than this length",
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
    parser.add_argument(
        "--skip-extension-index",
        action="store_true",
        help="Skip building the leftover-prefix extension index (built by default)",
    )
    parser.add_argument(
        "--extension-limit",
        type=int,
        default=1000,
        help="Number of dictionary words to inspect when building extension index",
    )
    parser.add_argument(
        "--extension-output",
        type=Path,
        default=Path("artifacts/extension_index.json"),
        help="Where to write the extension index JSON",
    )
    parser.add_argument(
        "--interactive-extension-debug",
        action="store_true",
        help="During the interactive loop, print step-by-step extension analysis",
    )
    parser.add_argument(
        "--wordfreq-limit",
        type=int,
        default=60000,
        help="Top-N threshold when sampling from Wordfreq (used with --dictionary-source=wordfreq_twl)",
    )
    parser.add_argument(
        "--twl-path",
        type=Path,
        default=Path("third_party/twl.py"),
        help="Path to a TWL word list (twl.py or text) for Scrabble validation",
    )
    parser.add_argument(
        "--rank-combinations",
        action="store_true",
        help="Compute combination totals for every word and print the top scorers",
    )
    parser.add_argument(
        "--combination-top",
        type=int,
        default=500,
        help="How many of the highest-scoring words to display when ranking combinations",
    )
    parser.add_argument(
        "--walk-start",
        type=str,
        help="Starting word for greedy graph walk",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=5,
        help="Number of words to visit during the graph walk (including the start)",
    )
    parser.add_argument(
        "--walk-top-edges",
        type=int,
        default=10,
        help="How many of the remaining edges to display after the walk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lowercase = not args.no_lowercase
    allow_single_letters = not args.disallow_single_letters
    alphabetic_only = not args.keep_non_alpha
    max_length = args.max_length if args.max_length > 0 else None

    if args.dictionary_source in {"mac", "file"}:
        dictionary_path = args.dictionary
        words = load_words(
            dictionary_path,
            lowercase=lowercase,
            min_length=args.min_length,
            max_length=max_length,
            allow_single_letters=allow_single_letters,
            alphabetic_only=alphabetic_only,
            allowed_single_letters=DEFAULT_SINGLE_LETTERS,
        )
        dictionary_label = str(dictionary_path)
    elif args.dictionary_source == "wordfreq_twl":
        words = build_wordfreq_twl_dictionary(
            limit=args.wordfreq_limit,
            twl_path=args.twl_path,
            lowercase=lowercase,
            min_length=args.min_length,
            max_length=max_length,
            allow_single_letters=allow_single_letters,
            alphabetic_only=alphabetic_only,
            allowed_single_letters=DEFAULT_SINGLE_LETTERS,
        )
        dictionary_label = (
            f"Wordfreq top {args.wordfreq_limit} ∩ {args.twl_path}"
            if args.twl_path.exists()
            else f"Wordfreq top {args.wordfreq_limit}"
        )
    else:
        raise ValueError(f"Unknown dictionary source: {args.dictionary_source}")

    print(f"Loaded {len(words):,} unique words from {dictionary_label}")

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

    word_set: Set[str] = set(words)

    forward_extension_lists: Dict[str, List[str]] = {}
    backward_extension_lists: Dict[str, List[str]] = {}
    forward_extension_sets: Dict[str, Set[str]] = {}
    backward_extension_sets: Dict[str, Set[str]] = {}

    need_extension_index = (
        args.rank_combinations
        or args.walk_start is not None
        or not args.skip_extension_index
        or not args.no_interactive
    )

    if need_extension_index:
        extension_limit = args.extension_limit
        if args.rank_combinations and extension_limit < len(words):
            extension_limit = len(words)
            print(
                "Expanding extension index limit to cover entire dictionary:",
                f"{extension_limit:,}",
            )

        extension_index = build_extension_index(
            words,
            forward,
            reverse,
            limit=extension_limit,
        )

        forward_extension_lists = extension_index["forward"]
        backward_extension_lists = extension_index["backward"]
        forward_extension_sets = {
            remainder: set(word_list)
            for remainder, word_list in forward_extension_lists.items()
        }
        backward_extension_sets = {
            remainder: set(word_list)
            for remainder, word_list in backward_extension_lists.items()
        }

        print(
            "Extension index entries:",
            f"forward={len(forward_extension_lists):,} | backward={len(backward_extension_lists):,}",
        )

        if not args.skip_extension_index:
            serialize_extension_index(extension_index, args.extension_output)
            print("Saved extension index to", args.extension_output.resolve())
    
    combo_results: List[Tuple[float, int, int, int, int, int, str]] = []
    combo_results_dict: Dict[str, Tuple[float, int, int, int, int, int]] = {}
    # if args.rank_combinations:
    if not forward_extension_sets or not backward_extension_sets:
        print("Combination ranking requires the extension index; none available.")
    else:
        print("Calculating combination totals for all words…")
        
        i = 0
        for word in words:
            i += 1
            if i % 100 == 0:
                print(f"Processed {i:,} words")
            combos, total_left, total_right, contributing, score, _edges = calculate_combinations_for_word(
                word,
                word_set,
                forward_extension_sets,
                backward_extension_sets,
                require_valid_start=args.require_valid_start,
                require_valid_end=args.require_valid_end,
            )
            if combos:
                combo_results.append(
                    (score, combos, total_left, total_right, contributing, word)
                )
                combo_results_dict[word] = (score, combos, total_left, total_right, contributing)

        if combo_results:
            combo_results.sort(
                key=lambda item: (
                    -item[0],
                    -item[1],
                    -item[2],
                    -item[3],
                    -item[4],
                    item[5],
                )
            )
            top_n = min(args.combination_top, len(combo_results))
            print(
                f"Top {top_n} words by combination count (out of {len(combo_results):,} scoring words):"
            )
            for rank, (score, combos, total_left, total_right, contributing, word) in enumerate(
                combo_results[:top_n],
                start=1,
            ):
                print(
                    f"{rank:4d}. {word:<15} combinations={combos:<8} left_total={total_left:<6} "
                    f"right_total={total_right:<6} segs={contributing} score={score:<8}"
                )
            if len(combo_results) > top_n:
                print(
                    f"… {len(combo_results) - top_n:,} additional words have non-zero combinations."
                )
        else:
            print("No words produced any combinations with the current settings.")

    if args.walk_start:
        if args.walk_start not in word_set:
            print(f"Walk start word '{args.walk_start}' is not in the dictionary; skipping walk.")
        elif not forward_extension_sets or not backward_extension_sets:
            print("Graph walk requires the extension index; none available.")
        else:
            walk_steps = max(1, args.walk_length)
            try:
                walk_result = walk_graph(
                    args.walk_start,
                    word_set,
                    forward_extension_sets,
                    backward_extension_sets,
                    steps=walk_steps,
                    require_valid_start=args.require_valid_start,
                    require_valid_end=args.require_valid_end,
                    combo_results=combo_results_dict,
                )
            except ValueError as exc:
                print(f"Graph walk failed: {exc}")
            else:
                print(
                    f"Graph walk starting at '{args.walk_start}' for up to {walk_steps} words:" 
                )
                for idx, entry in enumerate(walk_result["path"], start=1):
                    print(
                        f"  {idx:2d}. {entry['word']:<15} combos={entry['combinations']:<6} "
                        f"left={entry['left_total']:<5} right={entry['right_total']:<5} "
                        f"segs={entry['segmentations']:<4} score={entry['score']:.3f}"
                    )
                remaining_edges = walk_result["edges"]
                if remaining_edges:
                    top_edges = sorted(
                        remaining_edges.items(), key=lambda item: item[1], reverse=True
                    )[: args.walk_top_edges]
                    print(
                        f"  Top {len(top_edges)} remaining edges after walk (word, weight):"
                    )
                    for neighbor, weight in top_edges:
                        print(f"    {neighbor:<15} {weight:.3f}")
                    leftover = len(remaining_edges) - len(top_edges)
                    if leftover > 0:
                        print(f"    … {leftover} more edges not shown")
                else:
                    print("  No remaining edges after walk.")

    if not args.no_interactive:
        interactive_loop(
            word_set,
            forward,
            reverse,
            forward_extension_sets,
            backward_extension_sets,
            require_valid_start=args.require_valid_start,
            require_valid_end=args.require_valid_end,
            extension_debug=args.interactive_extension_debug,
        )


if __name__ == "__main__":
    main()
