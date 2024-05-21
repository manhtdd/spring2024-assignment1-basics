import numpy as np
import regex as re
import math
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import List, Dict, Tuple
from utils import parallel_concat
from bpe_encode_decode import encode as bpe_encode
from bpe_encode_decode import decode as bpe_decode


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

    chunk_ranges = [(boundaries[i], boundaries[i+1])
                    for i in range(len(boundaries)-1)]

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


class Tokenizer:
    def __init__(self, vocab: Dict[int, List[int]], merges: List[Tuple[List[int], List[int]]], special_tokens: List[str]):
        special_tokens.sort(key=lambda x: -len(x))
        self.special_regex = re.compile('|'.join(
            re.escape(token) for token in special_tokens)) if special_tokens else None

        self.vocab = vocab
        self.vocab_inv_bytes = [(v[0], k)
                                for k, v in self.vocab.items() if len(v) == 1]

        merged_vocab = {}
        for e1, e2 in merges:
            merged = e1 + e2
            e1_token = next(k for k, v in vocab.items() if v == e1)
            e2_token = next(k for k, v in vocab.items() if v == e2)
            merged_token = next(k for k, v in vocab.items() if v == merged)
            merged_vocab[(e1_token, e2_token)] = merged_token

        self.merges = merged_vocab

        vocab_inv = {tuple(v): k for k, v in vocab.items()}
        self.special_tokens_inv = {
            tuple(v.encode()): vocab_inv[tuple(v.encode())] for v in special_tokens}

        self.re = re.compile(
            r"'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+")

    def encode(self, text: str) -> np.ndarray:
        if not text:
            return np.array([], dtype=np.uint16)

        num_threads = 4 if len(text) > 100000 else 1
        chunk_size = (len(text) + num_threads - 1) // num_threads

        boundaries = [0]
        for i in range(1, num_threads):
            loc = i * chunk_size
            while loc < len(text) - 1:
                if text[loc] == '.' and text[loc + 1] == '\n':
                    loc += 1
                    break
                loc += 1
            boundaries.append(loc)
        boundaries.append(len(text))

        boundaries = sorted(set(boundaries))
        chunk_ranges = [range(boundaries[i], boundaries[i + 1])
                        for i in range(len(boundaries) - 1)]

        def process_chunk(chunk_range):
            chunk = text[chunk_range.start:chunk_range.stop]
            tokens = []
            offset = 0
            if self.special_regex:
                for snip in self.special_regex.finditer(chunk):
                    chunk_text = chunk[offset:snip.start()]
                    tokens.extend(bpe_encode(
                        self.re, self.vocab_inv_bytes, self.merges, chunk_text))
                    special_token = self.special_tokens_inv[tuple(
                        snip.group().encode())]
                    tokens.append(special_token)
                    offset = snip.end()
            if offset < len(chunk):
                tokens.extend(bpe_encode(
                    self.re, self.vocab_inv_bytes, self.merges, chunk[offset:]))
            return tokens

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            words_chunks = list(executor.map(process_chunk, chunk_ranges))

        words = parallel_concat(words_chunks)
        return words

    def decode(self, tokens: List[int]) -> str:
        decoded = bpe_decode(tokens, self.vocab)
        return ''.join(map(chr, decoded))


# Example usage:
if __name__ == "__main__":
    in_string = "<s>the cat ate <unk> the rat<e>"
    vocab_size = 300
    special_tokens = ["<s>", "<e>", "<unk>"]
    vocab, merges = train_bpe(in_string, vocab_size, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    encoded = tokenizer.encode(in_string)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
