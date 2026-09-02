def get_pair_counts(token_ids):
    """Count occurrences of every adjacent pair (t_i, t_{i+1}) in token_ids."""
    counts = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# tiny hand-checkable example
tokens = [97, 98, 99, 97, 98]   # a b c a b
counts = get_pair_counts(tokens)
print("token_ids:", tokens)
print("pair counts:", counts)

# expected: (97,98) -> 2   [positions 0-1 and 3-4]
#           (98,99) -> 1   [position 1-2]
#           (99,97) -> 1   [position 2-3]
