def merge(token_ids, pair, new_id):
    """Replace every occurrence of `pair` in token_ids with `new_id`."""
    merged = []
    i = 0
    while i < len(token_ids):
        # if we're not at the last position and the pair matches here, merge it
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2   # skip both tokens we just merged
        else:
            merged.append(token_ids[i])
            i += 1
    return merged


tokens = [97, 98, 99, 97, 98]   # a b c a b
new_tokens = merge(tokens, (97, 98), 256)
print("before:", tokens)
print("after: ", new_tokens)

# expected: [256, 99, 256]   -- both (97,98) occurrences replaced with 256
