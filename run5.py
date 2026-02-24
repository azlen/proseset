from wordfreq import top_n_list
import third_party.twl as twl
from tqdm import tqdm
import math

N = 20000 # number of words to load from wordfreq
MAX_LENGTH = 7 # maximum length for words

WHITELIST_1_LETTER = ['a', 'i']
WHITELIST_2_LETTER = ['ah', 'am', 'an', 'as', 'at', 'ax', 'be', 'by', 'do', 'eh', 'ex', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'no', 'of', 'oh', 'on', 'or', 'ow', 'ox', 'pi', 'up', 'us', 'we', 'ya', 'yo']

top_words = set(top_n_list('en', N, wordlist='best'))
valid_words_all = set(WHITELIST_1_LETTER + WHITELIST_2_LETTER + [word for word in top_words if twl.check(word) and word.isalpha() and len(word) > 2])
valid_words = set([word for word in valid_words_all if len(word) <= MAX_LENGTH]) # Limit to 7 letter words

print(f"Loaded {len(valid_words)} valid words")

# ================================ #
# =       HELPER FUNCTIONS       = #
# ================================ #

def progress(iterable, desc=""):
    """Progress bar with a custom format."""
    return tqdm(iterable, desc=desc, ascii=" ▖▘▝▗▚▞█", bar_format='{desc}: |{bar:20}|')

def merge_middle(*arrays):
    """
    Merge any number of arrays by concatenating adjacent boundaries.
    e.g., ['a', 'b'], ['c', 'd'], ['e', 'f'] -> ['a', 'bc', 'de', 'f']
    """
    # Filter out empty arrays
    arrays = [arr for arr in arrays if arr]
    
    if len(arrays) == 0:
        return []
    if len(arrays) == 1:
        return arrays[0]
    
    # Start with the first array
    result = arrays[0]
    
    # Iteratively merge with each subsequent array
    for arr in arrays[1:]:
        result = result[:-1] + [result[-1] + arr[0]] + arr[1:]
    
    return result

def segment_word(word: str, *, start: bool = False, end: bool = False):
    """
    Split word into possible segments, with flags to require valid start and end segments.
    e.g., 'area' -> [['a', 'rea'], ['ar', 'ea'], ['are', 'a']]
    """
    if not word:
        return [[]]
    results = []
    for i in range(1, len(word) + 1):
        if (word[:i] in valid_words) or (not start) or (i == len(word) and not end):
            for tail in segment_word(word[i:], start=True, end=end):
                results.append([word[:i]] + tail)
    return results

# ================================ #
# =      CONSTRUCT LOOKUPS       = #
# ================================ #

affix_lookup = {
    "forward": {}, # words that start with prefix
    "backward": {}, # words that end with suffix
}

forward_lookup = {}
backward_lookup = {}

def construct_lookups():
    # Build affix lookup. so we can look up words that start or end with given string
    for word in valid_words:
        for i in range(1, len(word)):
            affix_lookup['forward'].setdefault(word[:i], set()).add(word)
            affix_lookup['backward'].setdefault(word[i:], set()).add(word)

    print(f"Built forward affix lookup containing {len(affix_lookup['forward'])} entries")
    print(f"Built backward affix lookup containing {len(affix_lookup['backward'])} entries")

    # Forward lookup for possible words/segments that may be attached after the given word
    for word in progress(valid_words, "Building forward lookups"):
        for segmentation in segment_word(word, start=False, end=True): # forward
            if segmentation:
                first_segment = segmentation[0]
                candidates = affix_lookup['backward'].get(first_segment, set())
                for candidate in candidates:
                    remainder = candidate[:-len(first_segment)]
                    forward_lookup.setdefault(remainder, dict()).setdefault(word, []).append(segmentation)

    print(f"Built forward lookup containing {len(forward_lookup)} entries")

    # Backward lookup for possible words/segments that may be attached before the given word
    for word in progress(valid_words, "Building backward lookups"):
        for segmentation in segment_word(word, start=True, end=False): # backward
            if segmentation:
                last_segment = segmentation[-1]
                candidates = affix_lookup['forward'].get(last_segment, set())
                for candidate in candidates:
                    remainder = candidate[len(last_segment):]
                    backward_lookup.setdefault(remainder, dict()).setdefault(word, []).append(segmentation)

    print(f"Built backward lookup containing {len(backward_lookup)} entries")

# ================================ #
# =          WALK GRAPH          = #
# ================================ #

def walk_graph(start_words: list[str], steps: int):
    # keep track of chosen words
    
    # In loop
        # Go through each word and see what new words are possible to be made
            # Rank new words by how new they are, add together scores
            # Treat this as the score of the edge
            # Highest adge word gets added to chosen words

    chosen = start_words
    possible_words = dict()

    def filter_by(o, filter_obj):
        return dict(filter(lambda x: x[0] in filter_obj, o.items()))

    def filter_chosen(o):
        return filter_by(o, chosen)

    for i in progress(range(steps), f"Walking graph from {start_words}"):
        data = {}
        for word in valid_words:
            if word in chosen or word.startswith('s') or len(word) < 3: # s words are too easy to combine
                continue
        
            segments = segment_word(word)
            for segment in segments:
                # words that can be attached to the left or right of middle (m) segment
                m_left = filter_chosen(backward_lookup.get(segment[0], dict()))
                m_right = filter_chosen(forward_lookup.get(segment[-1], dict()))

                possible_new_words_m = dict()
                possible_new_words_l = dict()
                possible_new_words_r = dict()

                if len(m_left) > 0 and len(m_right) > 0: # there are valid sets with current chosen words
                    for left_word in m_left:
                        for left_segment in m_left[left_word]:
                            for right_word in m_right:
                                for right_segment in m_right[right_word]:
                                    made_words = merge_middle(left_segment, segment, right_segment)
                                    for made_word in made_words:
                                        possible_new_words_m.setdefault(made_word, 0)
                                        possible_new_words_m[made_word] += 1
                
                for m_word in chosen:
                    m_segments = segment_word(m_word)
                    for m_segment in m_segments:
                        m_left_a = filter_by(backward_lookup.get(m_segment[0], dict()), [word])
                        m_right_a = filter_by(forward_lookup.get(m_segment[-1], dict()), list(set(chosen) - {m_word}))

                        if len(m_left_a) > 0 and len(m_right_a) > 0:
                            for left_word in m_left_a:
                                for left_segment in m_left_a[left_word]:
                                    for right_word in m_right_a:
                                        for right_segment in m_right_a[right_word]:
                                            made_words = merge_middle(left_segment, m_segment, right_segment)
                                            for made_word in made_words:
                                                possible_new_words_l.setdefault(made_word, 0)
                                                possible_new_words_l[made_word] += 1

                        m_left_b = filter_by(backward_lookup.get(m_segment[0], dict()), list(set(chosen) - {m_word}))
                        m_right_b = filter_by(forward_lookup.get(m_segment[-1], dict()), [word])

                        if len(m_left_b) > 0 and len(m_right_b) > 0:
                            for left_word in m_left_b:
                                for left_segment in m_left_b[left_word]:
                                    for right_word in m_right_b:
                                        for right_segment in m_right_b[right_word]:
                                            made_words = merge_middle(left_segment, m_segment, right_segment)
                                            for made_word in made_words:
                                                possible_new_words_r.setdefault(made_word, 0)
                                                possible_new_words_r[made_word] += 1

                score_m = 0
                for made_word, count in possible_new_words_m.items():
                    score_m += 1 / ((possible_words.get(made_word, 1) + count) / 10) * math.sqrt(len(made_word))

                score_l = 0
                for made_word, count in possible_new_words_l.items():
                    score_l += 1 / ((possible_words.get(made_word, 1) + count) / 10) * math.sqrt(len(made_word))

                score_r = 0
                for made_word, count in possible_new_words_r.items():
                    score_r += 1 / ((possible_words.get(made_word, 1) + count) / 10) * math.sqrt(len(made_word))

                possible_new_words = {}
                for made_word, count in possible_new_words_m.items():
                    possible_new_words.setdefault(made_word, 0)
                    possible_new_words[made_word] += count
                for made_word, count in possible_new_words_l.items():
                    possible_new_words.setdefault(made_word, 0)
                    possible_new_words[made_word] += count
                for made_word, count in possible_new_words_r.items():
                    possible_new_words.setdefault(made_word, 0)
                    possible_new_words[made_word] += count

                data[word] = {
                    "scores": [score_l, score_m, score_r],
                    "score": score_l * score_m * score_r + score_m + score_l + score_r,
                    "possible_new_words": possible_new_words,
                }

        next_word = max(data.items(), key=lambda x: x[1]['score'])[0]
        print(next_word, data[next_word]['scores'])
        chosen.append(next_word)
        for word, score in data[next_word]['possible_new_words'].items():
            possible_words.setdefault(word, 0)
            possible_words[word] += score

    return chosen

if __name__ == "__main__":
    construct_lookups()
    result = walk_graph(['dear', 'area'], 100)
    print("Chosen words:", result)