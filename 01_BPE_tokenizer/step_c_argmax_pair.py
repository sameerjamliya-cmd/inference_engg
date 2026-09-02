def get_pair_counts(token_ids):
    counts = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def get_max_pair(counts):
    """Return the pair with the highest count."""
    best_pair = max(counts, key=counts.get)
    return best_pair


tokens = [97, 98, 99, 97, 98]   # a b c a b
counts = get_pair_counts(tokens)
best_pair = get_max_pair(counts)
print("pair counts:", counts)
print("best pair:  ", best_pair)

# expected: (97, 98) -- it occurs twice, more than any other pair
