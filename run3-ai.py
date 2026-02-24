"""

Steps:
- Load and filter dictionary
- Construct prefix and suffix lookups
- Build extension index
- Graph pinching walk
- Output the results

"""

from functools import lru_cache
from wordfreq import top_n_list, zipf_frequency
import third_party.twl as twl
from tqdm import tqdm

def progress(iterable, desc=""):
    return tqdm(iterable, desc=desc, ascii=" ▖▘▝▗▚▞█", bar_format='{desc}: |{bar:20}|')

top_words = set(top_n_list('en', 20000, wordlist='best'))
valid_words_all = set(['a', 'i'] + [word for word in top_words if twl.check(word) and word.isalpha() and len(word) > 1])
valid_words = set([word for word in valid_words_all if len(word) <= 7])

@lru_cache(maxsize=None)
def _segment_word(word: str, start: bool, end: bool):
    if not word:
        return ((),)
    results = []
    for i in range(1, len(word) + 1):
        prefix = word[:i]
        if (prefix in valid_words) or (not start) or (i == len(word) and not end):
            tails = _segment_word(word[i:], True, end)
            for tail in tails:
                results.append((prefix,) + tail)
    return tuple(results)


def segment_word(word: str, *, start: bool = False, end: bool = False):
    return [list(seg) for seg in _segment_word(word, start, end)]


@lru_cache(maxsize=None)
def get_zipf_frequency(word: str) -> float:
    return zipf_frequency(word, "en")


def merge_segments(left_segments, middle_segments, right_segments):
    left_segments = list(left_segments)
    middle_segments = list(middle_segments)
    right_segments = list(right_segments)

    merged = []

    if left_segments:
        merged.extend(left_segments[:-1])
        left_tail = left_segments[-1]
    else:
        left_tail = ""

    if middle_segments:
        middle_head = middle_segments[0]
        middle_mid = middle_segments[1:-1]
        middle_tail = middle_segments[-1]
    else:
        middle_head = ""
        middle_mid = []
        middle_tail = ""

    if right_segments:
        right_head = right_segments[0]
        right_rest = right_segments[1:]
    else:
        right_head = ""
        right_rest = []

    if middle_segments:
        combined_first = (left_tail + middle_head) if left_tail or middle_head else ""
        if combined_first:
            merged.append(combined_first)
        elif left_tail:
            merged.append(left_tail)
        elif middle_head:
            merged.append(middle_head)
    elif left_tail:
        merged.append(left_tail)

    merged.extend(middle_mid)

    if middle_segments and right_segments:
        merged.append(middle_tail + right_head)
    elif middle_segments:
        if len(middle_segments) > 1:
            merged.append(middle_tail)
    elif right_segments:
        merged.append((left_tail + right_head) if left_tail else right_head)

    merged.extend(right_rest)

    return [segment for segment in merged if segment]

segmentations_cache = {}
segment_start_index = {}
segment_end_index = {}
word_combo_data = {}

def construct_lookups():
    print(f"Loaded {len(valid_words)} valid words")

    for word in progress(valid_words, "Segmenting words"):
        segments = segment_word(word, start=True, end=True)
        filtered_segments = [seg for seg in segments if len(seg) >= 1]
        segmentations_cache[word] = filtered_segments
        for seg in filtered_segments:
            first = seg[0]
            last = seg[-1]
            segment_start_index.setdefault(first, []).append({"word": word, "segments": seg})
            segment_end_index.setdefault(last, []).append({"word": word, "segments": seg})

    print(f"Indexed {len(segment_start_index)} starting segments and {len(segment_end_index)} ending segments")

    for word in progress(valid_words, "Preparing combo data"):
        combos = []
        for segments in segmentations_cache[word]:
            if len(segments) < 2:
                continue
            first = segments[0]
            last = segments[-1]
            left_entries = [entry for entry in segment_end_index.get(first, []) if entry["word"] != word]
            right_entries = [entry for entry in segment_start_index.get(last, []) if entry["word"] != word]
            if not left_entries or not right_entries:
                continue
            combos.append({
                "segments": segments,
                "left": left_entries,
                "right": right_entries,
            })
        word_combo_data[word] = combos

    print("Prepared combination data for", sum(1 for combos in word_combo_data.values() if combos), "words")

# Experiment to find segmentable words to try to come up with a name for this game
def find_segmentable_words():
    fully_segmentable_words = []
    for word in valid_words_all:
        if word.endswith('ers') or word.endswith('ors'):
            continue
        segments = segment_word(word, start=True, end=True)
        segments = list(filter(lambda x: len(x) == 2 and 1 not in [len(seg) for seg in x] and 2 not in [len(seg) for seg in x], segments))
        fully_segmentable_words.append((word, segments))

    fully_segmentable_words.sort(key=lambda x: (-len(x[1]), x[0]))

    print(f"Found {len(fully_segmentable_words)} fully segmentable words")
    print("Top 2000 most segmentable words:")
    for i, (word, segments) in enumerate(fully_segmentable_words[:2000], 1):
        segments_str = ', '.join(['+'.join(seg) for seg in segments])
        print(f"{i:3d}. {word:<15} {segments_str}")

def rank_words(top_n=500, *, used_segments=None):
    used_segments = used_segments or set()
    results = []
    for word in progress(valid_words, "Ranking words"):
        data = calculate_combinations_for_word(word, used_segments=used_segments)
        if data["score"] > 0:
            results.append(data)
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_n]

def calculate_combinations_for_word(word, *, used_segments=None):
    combos = []
    edges = {}
    total_score = 0.0
    total_combos = 0
    unique_segments = set()

    used_segments = used_segments or set()

    for info in word_combo_data.get(word, []):
        segments = info["segments"]
        left_entries = info["left"]
        right_entries = info["right"]
        for left_entry in left_entries:
            left_word = left_entry["word"]
            left_segments = left_entry["segments"]
            for right_entry in right_entries:
                right_word = right_entry["word"]
                if right_word == left_word:
                    continue
                right_segments = right_entry["segments"]
                merged = merge_segments(left_segments, segments, right_segments)
                new_segments = [seg for seg in merged if seg not in used_segments]
                weight = 0.0
                for seg in new_segments:
                    length = len(seg)
                    freq = get_zipf_frequency(seg)
                    if length == 1:
                        freq *= 0.02
                    elif length == 2:
                        freq *= -0.75
                    weight += freq
                combo = {
                    "left_word": left_word,
                    "left_segments": left_segments,
                    "middle_segments": segments,
                    "right_word": right_word,
                    "right_segments": right_segments,
                    "merged_segments": merged,
                    "weight": weight,
                }
                combos.append(combo)
                total_combos += 1
                unique_segments.update(merged)
                total_score += weight
                for neighbor in (left_word, right_word):
                    entry = edges.setdefault(neighbor, {"weight": 0.0, "segments": set(), "combos": []})
                    entry["weight"] += weight
                    entry["segments"].update(merged)
                    entry["combos"].append(combo)

    return {
        "word": word,
        "total_combos": total_combos,
        "score": total_score,
        "unique_segments": unique_segments,
        "edges": edges,
        "combos": combos,
    }


def walk_graph(start_word: str, steps: int, *, used_segments=None):
    if start_word not in valid_words:
        raise ValueError(f"Start word {start_word} is not in the dictionary")

    used_segments = used_segments or set()
    chosen = []
    chosen_details = {}
    aggregated_edges = {}
    current = start_word

    for _ in progress(range(steps), f"Walking graph from {start_word}"):
        data = calculate_combinations_for_word(current, used_segments=used_segments)
        chosen.append(current)
        chosen_details[current] = data

        for neighbor, info in data["edges"].items():
            if neighbor in chosen:
                continue
            entry = aggregated_edges.setdefault(neighbor, {"weight": 0.0, "segments": set(), "sources": set(), "combos": []})
            entry["weight"] += info["weight"]
            entry["segments"].update(info["segments"])
            entry["sources"].add(current)
            entry["combos"].extend(info["combos"])

        used_segments.update(data["unique_segments"])

        for word in chosen:
            aggregated_edges.pop(word, None)

        aggregated_edges = {k: v for k, v in aggregated_edges.items() if v["weight"] > 0}

        if len(chosen) >= steps or not aggregated_edges:
            break

        next_word = max(aggregated_edges.items(), key=lambda item: item[1]["weight"])[0]
        aggregated_edges.pop(next_word, None)
        current = next_word

    return {
        "path": chosen,
        "details": chosen_details,
        "edges": aggregated_edges,
        "used_segments": used_segments,
    }

if __name__ == "__main__":
    construct_lookups()
    ranked = rank_words(top_n=50)
    print("Top 50 words by score:")
    for idx, data in enumerate(ranked, start=1):
        print(
            f"{idx:3d}. {data['word']:<15} score={data['score']:.3f} "
            f"combos={data['total_combos']}"
        )

    result = walk_graph('area', 10)
    print("Chosen path:", ' -> '.join(result['path']))
    if result['edges']:
        top_edges = sorted(result['edges'].items(), key=lambda item: item[1]['weight'], reverse=True)[:10]
        print("Top remaining edges:")
        for neighbor, info in top_edges:
            print(f"  {neighbor:<15} weight={info['weight']:.3f} sources={', '.join(sorted(info['sources']))}")
    # find_segmentable_words()
