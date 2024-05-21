import regex as re
import math
from collections import defaultdict
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from utils import logger, parallel_concat

Pair = Tuple[int, int]
NUM_THREADS = 1  # You can adjust this to the number of threads you want


class Word:
    def __init__(self, symbols: List[int], word_count: int):
        self.symbols = symbols
        self.word_count = word_count

    def __repr__(self) -> str:
        return f"({self.symbols}, {self.word_count})"

    def clone(self):
        return Word(self.symbols[:], self.word_count)


def count_words(words: List[str]) -> List[Word]:
    num_words = len(words)
    chunk_size = math.ceil(num_words / NUM_THREADS)

    def chunk_word_count(words_chunk: List[str]) -> Dict[str, int]:
        word_counts = defaultdict(int)
        for word in words_chunk:
            word_counts[word] += 1
        return word_counts

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        chunks = [words[i:i + chunk_size]
                  for i in range(0, num_words, chunk_size)]
        thread_results = list(executor.map(chunk_word_count, chunks))

    all_counts = defaultdict(int)
    for thread_word_counts in thread_results:
        for word, count in thread_word_counts.items():
            all_counts[word] += count

    return [
        Word(symbols=[ord(c) for c in word], word_count=count)
        for word, count in all_counts.items()
    ]


def count_pairs(words: List[Word]) -> Dict[Pair, int]:
    symbol_counts = defaultdict(int)
    for word in words:
        for i in range(len(word.symbols) - 1):
            pair = (word.symbols[i], word.symbols[i + 1])
            symbol_counts[pair] += word.word_count
    return dict(symbol_counts)


def update_word(w: Word, pair: Pair, new_symbol: int) -> List[Tuple[Pair, int]]:
    i = 0
    count_changes = []
    while i < len(w.symbols) - 1:
        if w.symbols[i] == pair[0] and w.symbols[i + 1] == pair[1]:
            if i >= 1:
                count_changes.append(
                    ((w.symbols[i - 1], pair[0]), -w.word_count))
                count_changes.append(
                    ((w.symbols[i - 1], new_symbol), w.word_count))
            if len(w.symbols) >= 3 and i <= len(w.symbols) - 3:
                count_changes.append(
                    ((pair[1], w.symbols[i + 2]), -w.word_count))
                count_changes.append(
                    ((new_symbol, w.symbols[i + 2]), w.word_count))
            w.symbols[i] = new_symbol
            del w.symbols[i + 1]
        else:
            i += 1
    return count_changes


def update_words(words: List[Word], pair: Pair, new_symbol: int) -> Dict[Pair, int]:
    count_changes = defaultdict(int)

    chunk_size = math.ceil(len(words) / NUM_THREADS)

    def process_chunk(chunk):
        local_changes = defaultdict(int)
        for word in chunk:
            count_changes_word = update_word(word, pair, new_symbol)
            for p, change in count_changes_word:
                local_changes[p] += change
        return local_changes

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        chunks = [words[i:i + chunk_size]
                  for i in range(0, len(words), chunk_size)]
        thread_results = list(executor.map(process_chunk, chunks))

    for local_changes in thread_results:
        for p, change in local_changes.items():
            count_changes[p] += change

    return dict(count_changes)


def assemble_token(token: int, symbols: List[List[int]]) -> str:
    return ''.join(chr(x) for x in symbols[token])


def find_words_in_chunk(chunk: str, re_pattern: re.Pattern) -> List[str]:
    return re.findall(re_pattern, chunk)


def train_bpe(in_string: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, List[int]], List[Tuple[List[int], List[int]]]]:
    chunk_size = (len(in_string) + NUM_THREADS - 1) // NUM_THREADS

    boundaries = [0]
    for i in range(1, NUM_THREADS):
        loc = i * chunk_size
        while loc < len(in_string) - 1:
            if in_string[loc] == '.' and in_string[loc + 1] == '\n':
                loc += 1
                break
            loc += 1
        boundaries.append(loc)
    boundaries.append(len(in_string))
    boundaries = sorted(set(boundaries))
    logger(boundaries)

    chunk_ranges = [(boundaries[i], boundaries[i+1])
                    for i in range(len(boundaries)-1)]    
    logger(chunk_ranges)

    regex = re.compile(
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    def process_chunk(start, end):
        chunk = in_string[start:end]
        return [m.group(0) for m in regex.finditer(chunk)]

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        words_list = list(executor.map(
            lambda r: process_chunk(*r), chunk_ranges))
    
    words = parallel_concat(words_list)
    words = count_words(words)
    max_symbols = vocab_size

    symbol_counts = count_pairs(words)

    symbols = [[i] for i in range(256)]
    symbols.extend([[ord(char) for char in token] for token in special_tokens])

    pq = defaultdict(int, symbol_counts)
    merges = []

    while pq and len(symbols) < max_symbols:
        pair = max(pq, key=pq.get)
        pq.pop(pair)

        new_symbol = symbols[pair[0]] + symbols[pair[1]]
        merges.append((symbols[pair[0]], symbols[pair[1]]))
        symbols.append(new_symbol)

        count_changes = update_words(words, pair, len(symbols) - 1)

        for p, change in count_changes.items():
            pq[p] += change

    vocab = {i: v for i, v in enumerate(symbols)}

    return vocab, merges


def main():
    logger("New Test ------------------------------")
    in_string = "the cat ate the rat"
    vocab_size = 300
    special_tokens = ["<s>", "<e>", "<unk>"]
    vocab, merges = train_bpe(in_string, vocab_size, special_tokens)
    for key, item in vocab.items():
        logger(key, bytearray(item))


if __name__ == "__main__":
    main()
