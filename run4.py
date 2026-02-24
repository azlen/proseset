from wordfreq import top_n_list
import third_party.twl as twl
from tqdm import tqdm

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

def progress(iterable, desc=""):
    """Progress bar with a custom format."""
    return tqdm(iterable, desc=desc, ascii=" ▖▘▝▗▚▞█", bar_format='{desc}: |{bar:20}|')



# Load and filter dictionary
top_words = set(top_n_list('en', 20000, wordlist='best'))
valid_words_all = set(allowed_1_letter_words + allowed_2_letter_words + [word for word in top_words if twl.check(word) and word.isalpha() and len(word) > 2])
valid_words = set([word for word in valid_words_all if len(word) <= 7]) # Limit to 7 letter words

def segment_word(word: str, *, start: bool = False, end: bool = False):
    """Split word into possible segments, with flags to require valid start and end segments."""
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

forward_middle_lookup = {}
backward_middle_lookup = {}

def construct_lookups():
    print(f"Loaded {len(valid_words)} valid words")

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
        for segmentation in segment_word(word, start=True, end=False): # backward lookup
            if segmentation:
                last_segment = segmentation[-1]
                candidates = affix_lookup['forward'].get(last_segment, set())
                for candidate in candidates:
                    remainder = candidate[len(last_segment):]
                    backward_lookup.setdefault(remainder, dict()).setdefault(word, []).append(segmentation)

    print(f"Built backward lookup containing {len(backward_lookup)} entries")

    # Build middle lookups
    # for word in progress(valid_words, "Building middle lookups"):
    #     for segmentation in segment_word(word, start=False, end=False):
    #         if segmentation:
    #             last_segment = segmentation[-1]
    #             candidates = affix_lookup['forward'].get(last_segment, set())
    #             for candidate in candidates:
    #                 remainder = candidate[len(last_segment):]
    #                 backward_middle_lookup.setdefault(remainder, dict()).setdefault(word, []).append(segmentation)
    #             first_segment = segmentation[0]
    #             candidates = affix_lookup['backward'].get(first_segment, set())
    #             for candidate in candidates:
    #                 remainder = candidate[:-len(first_segment)]
    #                 forward_middle_lookup.setdefault(remainder, dict()).setdefault(word, []).append(segmentation)


def walk_graph(start_words: list[str], steps: int):
    for word in start_words:
        if word not in valid_words:
            raise ValueError(f"Start word {start_word} is not in the dictionary")
    
    data = {}
    chosen = start_words
    # current = start_word

    possible_words = dict()

    # filter chosen
    def filter_chosen(o):
        # return o
        return dict(filter(lambda x: x[0] in chosen, o.items()))

    for i in progress(range(steps), f"Walking graph from {start_words}"):
        scores = {}
        for word in valid_words:
            if word in chosen or word.startswith('s'):
                continue
            
            segments = segment_word(word)
            for segment in segments:
                m_left = filter_chosen(backward_lookup.get(segment[0], dict()))
                m_right = filter_chosen(forward_lookup.get(segment[-1], dict()))

                # really need a backward_middle lookup right????
                # r_left = filter_chosen(backward_middle_lookup.get(segment[0], dict()))
                # r_left_left = [filter_chosen(backward_lookup.get(left_word, dict())) for left_word in r_left]

                # l_right = filter_chosen(forward_middle_lookup.get(segment[-1], dict()))
                # l_right_right = [filter_chosen(forward_lookup.get(right_word, dict())) for right_word in l_right]
                # print (segment, left, right)

                # if len(m_left) == 0 or len(m_right) == 0 or len(r_left) == 0 or len(l_right) == 0 or len(r_left_left) == 0 or len(l_right_right) == 0:
                #     continue

                # print(m_left[0], word, m_right[0])
                # print(r_left_left[0][0], r_left[0], word)
                # print(word, l_right[0], l_right_right[0][0])
                
                if len(m_left) > 0 and len(m_right) > 0:
                    for left_word in m_left:
                        for left_segment in m_left[left_word]:
                            for right_word in m_right:
                                for right_segment in m_right[right_word]:
                                    # print(left_segment, segment, right_segment)
                                    made_words = merge_middle(left_segment, segment, right_segment)

                                    for made_word in made_words:
                                        possible_words.setdefault(made_word, 0)
                                        possible_words[made_word] += 1
    
    print(possible_words)

if __name__ == "__main__":
    construct_lookups()
    walk_graph(['dear', 'area'], 10)

    # print(affix_lookup['forward'].get('de', set()))
    # print(forward_lookup.get('d', dict()))

    # print(backward_middle_lookup.get('w', dict()))
    # print(forward_middle_lookup.get('d', dict()))