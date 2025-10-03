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

print(f"Loaded {len(valid_words)} valid words")

affix_lookup = {
    "forward": {}, # words that start with prefix
    "backward": {}, # words that end with suffix
}

for word in valid_words:
    for i in range(1, len(word)):
        affix_lookup['forward'].setdefault(word[:i], set()).add(word)
        affix_lookup['backward'].setdefault(word[i:], set()).add(word)

print(f"Built forward affix lookup containing {len(affix_lookup['forward'])} entries")
print(f"Built backward affix lookup containing {len(affix_lookup['backward'])} entries")

def segment_word(word: str, *, start: bool = False, end: bool = False):
    if not word:
        return [[]]
    results = []
    for i in range(1, len(word) + 1):
        if (word[:i] in valid_words) or (not start) or (i == len(word) and not end):
            for tail in segment_word(word[i:], start=True, end=end):
                results.append([word[:i]] + tail)
    return results

forward_lookup = {}
backward_lookup = {}

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

print(forward_lookup['sp'])