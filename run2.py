"""

Steps:
- Load and filter dictionary
- Construct prefix and suffix lookups
- Build extension index
- Graph pinching walk
- Output the results

"""

from wordfreq import top_n_list
import third_party.twl as twl
from tqdm import tqdm

def progress(iterable, desc=""):
    return tqdm(iterable, desc=desc, ascii=" ▖▘▝▗▚▞█", bar_format='{desc}: |{bar:20}|')

top_words = set(top_n_list('en', 20000, wordlist='best'))
valid_words_all = set(['a', 'i'] + [word for word in top_words if twl.check(word) and word.isalpha() and len(word) > 1])
valid_words = set([word for word in valid_words_all if len(word) <= 7])

def segment_word(word: str, *, start: bool = False, end: bool = False):
    if not word:
        return [[]]
    results = []
    for i in range(1, len(word) + 1):
        if (word[:i] in valid_words) or (not start) or (i == len(word) and not end):
            for tail in segment_word(word[i:], start=True, end=end):
                results.append([word[:i]] + tail)
    return results

affix_lookup = {
    "forward": {}, # words that start with prefix
    "backward": {}, # words that end with suffix
}

forward_lookup = {}
backward_lookup = {}

def construct_lookups():
    print(f"Loaded {len(valid_words)} valid words")

    for word in valid_words:
        for i in range(1, len(word)):
            affix_lookup['forward'].setdefault(word[:i], set()).add(word)
            affix_lookup['backward'].setdefault(word[i:], set()).add(word)

    print(f"Built forward affix lookup containing {len(affix_lookup['forward'])} entries")
    print(f"Built backward affix lookup containing {len(affix_lookup['backward'])} entries")

    for word in progress(valid_words, "Building forward lookups"):
        for segmentation in segment_word(word, start=False, end=True): # forward
            if segmentation:
                first_segment = segmentation[0]
                candidates = affix_lookup['backward'].get(first_segment, set())
                for candidate in candidates:
                    remainder = candidate[:-len(first_segment)]
                    forward_lookup.setdefault(remainder, set()).add(word)

    print(f"Built forward lookup containing {len(forward_lookup)} entries")

    for word in progress(valid_words, "Building backward lookups"):
        for segmentation in segment_word(word, start=True, end=False): # backward
            if segmentation:
                last_segment = segmentation[-1]
                candidates = affix_lookup['forward'].get(last_segment, set())
                for candidate in candidates:
                    remainder = candidate[len(last_segment):]
                    backward_lookup.setdefault(remainder, set()).add(word)

    print(f"Built backward lookup containing {len(backward_lookup)} entries")

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

def walk_graph(start_word: str, steps: int, score_func):
    if start_word not in valid_words:
        raise ValueError(f"Start word {start_word} is not in the dictionary")
    

    data = {}
    chosen = [start_word]
    current = start_word

    for word in valid_words:
        segments = [{
            "segment": segment,
            "left": backward_lookup.get(segment[0], set()),
            "right": forward_lookup.get(segment[-1], set()),
        } for segment in segment_word(word)]

        data[word] = segments
    
    for i in progress(range(steps), f"Walking graph from {start_word}"):
        # for every word that is possible we want to set score to be filtered by chosen words
        # OR filtering by chosen words once we set the score of all possible words

        scores = {}

        i = 0

        for word in chosen:
            for segment in data[word]:
                left_filtered = (set(segment['left']) & set(chosen)) - {word}
                right_filtered = (set(segment['right']) & set(chosen)) - {word}

                for left_word in left_filtered:
                    for right_word in set(segment['right']) - {left_word}:
                        if right_word not in chosen:
                            i+=1
                            if i < 20:
                                print(f"{segment['segment']} {left_word} {word} {right_word}")
                            scores.setdefault(right_word, 0)
                            scores[right_word] += 1
                    # scores.setdefault(left_word, 0)
                    # scores[left_word] += (len(left_filtered) + 1) * len(right_filtered)
                for right_word in right_filtered:
                    for left_word in set(segment['left']) - {right_word}:
                        if left_word not in chosen:
                            scores.setdefault(left_word, 0)
                            scores[left_word] += 1
                    # scores.setdefault(right_word, 0)
                    # scores[right_word] += len(left_filtered) * (len(right_filtered) + 1)

        for word in valid_words:
            if word in chosen or word.startswith('s'):
                continue
            for segment in data[word]:
                left_filtered = set(segment['left']) & set(chosen + [word])
                right_filtered = set(segment['right']) & set(chosen + [word])
                
                scores.setdefault(word, 0)
                scores[word] += len(left_filtered) * len(right_filtered)

        highest = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        print(highest)

        # Find the highest scoring word as the next current word
        next_word = max(scores.items(), key=lambda x: x[1])[0]
        chosen.append(next_word)
        current = next_word
    
    print(chosen)
        # segments = [{
        #     "segment": segment,
        #     "left": backward_lookup.get(segment[0], set()),
        #     "right": forward_lookup.get(segment[-1], set()),
        # } for segment in segment_word(word)]


    # for _ in progress(range(steps), f"Walking graph from {start_word}"):
    #     segments = segment_word(start_word)
    #     for segment in segments:
    #         left = backward_lookup.get(segment[0], set())
    #         right = forward_lookup.get(segment[-1], set())
           

if __name__ == "__main__":
    construct_lookups()
    walk_graph('area', 10, lambda x: 1)
    # find_segmentable_words()