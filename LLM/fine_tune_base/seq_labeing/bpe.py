import re
from collections import Counter, defaultdict


def build_vocab(corpus: str) -> dict:
    """Step 1. Build vocab from text corpus"""

    # Separate each char in word by space and add mark end of token
    tokens = [" ".join(word) + " </w>" for word in corpus.split()]

    # Count frequency of tokens in corpus
    vocab = Counter(tokens)

    return vocab


def get_stats(vocab: dict) -> dict:
    """Step 2. Get counts of pairs of consecutive symbols"""

    pairs = defaultdict(int)
    for word, frequency in vocab.items():
        symbols = word.split()

        # Counting up occurrences of pairs
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += frequency

    return pairs


def merge_vocab(pair: tuple, v_in: dict) -> dict:
    """Step 3. Merge all occurrences of the most frequent pair"""

    v_out = {}
    bigram = re.escape(' '.join(pair))
    print(bigram)
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    print("p:", p)

    for word in v_in:
        # replace most frequent pair in all vocabulary
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out

corpus = "Hello, World!"
vocab = build_vocab(corpus)  # Step 1

num_merges = 50  # Hyperparameter
for i in range(num_merges):

    pairs = get_stats(vocab)  # Step 2

    if not pairs:
        break

    # step 3
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)