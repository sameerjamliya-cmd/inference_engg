import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
sys.path.append("../01_BPE_tokenizer")
from bpe_tokenizer import BPETokenizer


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_K = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_V = torch.nn.Linear(d_model, d_model, bias=False)
        self.d_k = d_model

    def forward(self, x):
        seq_len = x.shape[0]
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        scores = (Q @ K.T) / (self.d_k ** 0.5)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights @ V


class MLP(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(torch.nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, max_seq_len):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_seq_len, d_model)
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model, d_ff) for _ in range(n_layers)]
        )
        self.ln_final = torch.nn.LayerNorm(d_model)
        self.output_projection = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    def forward(self, token_ids):
        seq_len = token_ids.shape[0]
        positions = torch.arange(seq_len)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.output_projection(x)


def top_k_sample(logits, k=5, temperature=1.0):
    top_k_values, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_values / temperature, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).item()
    return top_k_indices[idx].item()


@torch.no_grad()
def generate_naive_timed(model, tokenizer, prompt, num_new_tokens, k=5, temperature=1.0):
    """Same naive generation loop as step 7, but records wall-clock time per step."""
    token_ids = tokenizer.encode(prompt)
    per_step_seconds = []

    for _ in range(num_new_tokens):
        tokens_tensor = torch.tensor(token_ids)

        t0 = time.perf_counter()
        logits = model(tokens_tensor)          # the full, wasteful forward pass
        next_token_logits = logits[-1]
        next_token = top_k_sample(next_token_logits, k=k, temperature=temperature)
        t1 = time.perf_counter()

        per_step_seconds.append(t1 - t0)
        token_ids.append(next_token)

    return token_ids, per_step_seconds


# --- set up ---
tokenizer = BPETokenizer()
corpus = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks at the fox. the fox runs away. "
) * 20
tokenizer.train(corpus, num_merges=50)
vocab_size = len(tokenizer.vocab)

torch.manual_seed(0)
# a slightly bigger model than earlier steps, so the per-token cost is
# measurable above Python/PyTorch call overhead noise
model = TinyGPT(vocab_size=vocab_size, d_model=128, d_ff=512, n_layers=6, max_seq_len=600)

prompt = "the fox"
num_new_tokens = 400

print(f"generating {num_new_tokens} tokens from prompt {prompt!r}...")
_, per_step_seconds = generate_naive_timed(model, tokenizer, prompt, num_new_tokens, k=5)

# --- report ---
seq_lengths = list(range(len(tokenizer.encode(prompt)) + 1,
                          len(tokenizer.encode(prompt)) + 1 + num_new_tokens))

print()
print(f"{'step':>5} | {'seq_len':>8} | {'time (ms)':>10}")
print("-" * 30)
for i in [0, 49, 99, 199, 299, 399]:
    print(f"{i+1:>5} | {seq_lengths[i]:>8} | {per_step_seconds[i]*1000:>9.2f}")

first_10_avg = sum(per_step_seconds[:10]) / 10
last_10_avg = sum(per_step_seconds[-10:]) / 10
print()
print(f"avg time/token, first 10 steps (seq_len ~{seq_lengths[9]}):  {first_10_avg*1000:.2f} ms")
print(f"avg time/token, last 10 steps  (seq_len ~{seq_lengths[-1]}): {last_10_avg*1000:.2f} ms")
print(f"slowdown factor: {last_10_avg / first_10_avg:.2f}x")

# --- plot ---
plt.figure(figsize=(8, 5))
plt.plot(seq_lengths, [t * 1000 for t in per_step_seconds])
plt.xlabel("sequence length at this generation step")
plt.ylabel("time for this step (ms)")
plt.title("Naive autoregressive generation: per-token latency vs sequence length\n"
          "(full forward pass re-run every step, no KV cache)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("step_8_latency_plot.png")
print()
print("plot saved to step_8_latency_plot.png")
