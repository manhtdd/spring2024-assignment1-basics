import regex as re
from itertools import tee, islice
from typing import List, Tuple, Dict


def encode(
    re_pattern: str,  # Regex pattern to pre-tokenize
    # Mapping from byte to token for initial vocab
    vocab_inv_bytes: List[Tuple[bytes, int]],
    merges: Dict[Tuple[int, int], int],  # Tuple of tokens to merged token
    text: str
) -> List[int]:
    def tuple_windows(iterable, n=2):
        iters = tee(iterable, n)
        for i, it in enumerate(iters):
            next(islice(it, i, i), None)
        return zip(*iters)

    vocab_inv = {byte: token for byte, token in vocab_inv_bytes if byte}
    words = re.findall(re_pattern, text)

    encoded = []
    for word in words:
        word_bytes = word.encode()

        symbols = [vocab_inv[byte] for byte in word_bytes]

        while True:
            candidate_merges = []
            for i, (a, b) in enumerate(tuple_windows(symbols)):

                if (a, b) in merges:

                    candidate_merges.append((i, merges[(a, b)]))
            if not candidate_merges:
                break

            merge_index, merge_token = min(
                candidate_merges, key=lambda x: x[1])

            symbols[merge_index] = merge_token
            symbols.pop(merge_index + 1)

        encoded.extend(symbols)

    return encoded


def decode(v: List[int], vocab: Dict[int, bytes]) -> bytes:
    return b''.join(bytearray(vocab[token]) for token in v)


if __name__ == "__main__":
    import unittest

    class TestEncoding(unittest.TestCase):
        def test_encode(self):

            re_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t",
                     6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}

            vocab_inv_bytes = [(v[0], k)
                               for k, v in vocab.items() if len(v) == 1]

            merges = {(5, 4): 6, (0, 2): 7, (0, 1): 8, (6, 3): 9, (8, 5): 10}

            text = "the cat ate"
            encoded = encode(re_pattern, vocab_inv_bytes, merges, text)

            decoded = decode(encoded, vocab)

            self.assertEqual(encoded, [9, 7, 1, 5, 10, 3])
            self.assertEqual(decoded, b"the cat ate")

    unittest.main()
