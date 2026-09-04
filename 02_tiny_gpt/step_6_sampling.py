import torch
import torch.nn.functional as F

torch.manual_seed(0)   # so sampling results are reproducible when you run this


def greedy_sample(logits):
    """Always pick the single highest-scoring token."""
    return torch.argmax(logits).item()


def temperature_sample(logits, temperature=1.0):
    """Scale logits by 1/temperature, softmax, then sample from that distribution."""
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def top_k_sample(logits, k=5, temperature=1.0):
    """Zero out everything except the top-k logits, then temperature-sample among those."""
    top_k_values, top_k_indices = torch.topk(logits, k)
    scaled = top_k_values / temperature
    probs = F.softmax(scaled, dim=-1)
    sampled_index_within_top_k = torch.multinomial(probs, num_samples=1).item()
    return top_k_indices[sampled_index_within_top_k].item()


# a toy logits vector over a tiny 8-token vocab, deliberately shaped so we can
# reason about it by hand: token 3 is clearly the best, tokens 4/5 are close
# runners-up, and the rest trail off.
toy_logits = torch.tensor([0.5, -1.0, 0.2, 4.0, 3.0, 2.5, -2.0, 0.1])
print("toy logits:", toy_logits.tolist())
print("softmax probs:", [round(p, 3) for p in F.softmax(toy_logits, dim=-1).tolist()])
print()

print("--- greedy ---")
print("always picks:", greedy_sample(toy_logits), "(should always be token 3, the max)")
print()

print("--- temperature sampling ---")
for T in [0.5, 1.0, 2.0]:
    samples = [temperature_sample(toy_logits, temperature=T) for _ in range(10)]
    print(f"T={T}: 10 samples -> {samples}")
print("(lower T should look more repetitive/concentrated on token 3;")
print(" higher T should show more variety across tokens)")
print()

print("--- top-k sampling (k=3) ---")
samples = [top_k_sample(toy_logits, k=3, temperature=1.0) for _ in range(10)]
print(f"10 samples -> {samples}")
print("(should ONLY ever contain tokens 3, 4, or 5 -- the top 3 by score --")
print(" never token 1 or 6, the worst-scoring ones)")
