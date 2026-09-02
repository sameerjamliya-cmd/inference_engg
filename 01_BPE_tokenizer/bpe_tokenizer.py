class BPETokenizer:
    def __init__(self):
        self.merges = {}                                  # (a, b) -> new_id, in learn order
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes, base 256 + learned merges

    @staticmethod
    def _get_pair_counts(token_ids):
        """Count every adjacent pair (t_i, t_{i+1}) in token_ids."""
        counts = {}
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(token_ids, pair, new_id):
        """Replace every occurrence of `pair` in token_ids with `new_id`."""
        merged = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(token_ids[i])
                i += 1
        return merged

    def train(self, text, num_merges):
        """Learn `num_merges` merge rules from `text`. Returns the final training token ids."""
        token_ids = list(text.encode("utf-8"))
        next_id = 256
        for _ in range(num_merges):
            counts = self._get_pair_counts(token_ids)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            if counts[best_pair] < 2:
                break   # merging a pair that occurs once buys nothing
            token_ids = self._merge(token_ids, best_pair, next_id)
            self.merges[best_pair] = next_id
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            next_id += 1
        return token_ids

    def encode(self, text):
        """Encode new text using the learned merge rules, in learned priority order."""
        token_ids = list(text.encode("utf-8"))
        while True:
            counts = self._get_pair_counts(token_ids)
            # keep only pairs that are actually learned merge rules
            candidates = {pair: self.merges[pair] for pair in counts if pair in self.merges}
            if not candidates:
                break
            # smallest new_id == earliest-learned rule == highest priority
            best_pair = min(candidates, key=candidates.get)
            token_ids = self._merge(token_ids, best_pair, self.merges[best_pair])
        return token_ids

    def decode(self, token_ids):
        """Decode a list of token ids back into the original string."""
        byte_seq = b"".join(self.vocab[t] for t in token_ids)
        return byte_seq.decode("utf-8", errors="strict")
