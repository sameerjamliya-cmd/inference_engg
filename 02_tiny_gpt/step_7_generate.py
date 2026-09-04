import torch
import torch.nn.functional as F

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
        return self.output_projection(x)   # [seq_len, vocab_size]


def top_k_sample(logits, k=5, temperature=1.0):
    top_k_values, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_values / temperature, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).item()
    return top_k_indices[idx].item()


def generate_naive(model, tokenizer, prompt, num_new_tokens, k=5, temperature=1.0):
    """Naive autoregressive generation: re-run the FULL forward pass every step."""
    token_ids = tokenizer.encode(prompt)   # plain python list

    for _ in range(num_new_tokens):
        tokens_tensor = torch.tensor(token_ids)
        logits = model(tokens_tensor)              # recomputes EVERYTHING, every step
        next_token_logits = logits[-1]
        next_token = top_k_sample(next_token_logits, k=k, temperature=temperature)
        token_ids.append(next_token)

    return token_ids


# --- set up tokenizer + model ---
tokenizer = BPETokenizer()
corpus = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks at the fox. the fox runs away. "
) * 20
tokenizer.train(corpus, num_merges=50)
vocab_size = len(tokenizer.vocab)

torch.manual_seed(0)
model = TinyGPT(vocab_size=vocab_size, d_model=16, d_ff=64, n_layers=4, max_seq_len=64)

prompt = "the fox"
print(f"prompt: {prompt!r}")

generated_ids = generate_naive(model, tokenizer, prompt, num_new_tokens=15, k=5, temperature=1.0)

# an untrained model has no notion of "valid UTF-8" -- it can sample a byte
# that starts a multi-byte character without ever sampling the matching
# continuation byte(s). decode leniently here (errors="replace") just to be
# able to print something; this is NOT how Project 01's tokenizer is tested
# for correctness -- those roundtrip tests stay strict, on real trained text.
byte_seq = b"".join(tokenizer.vocab[t] for t in generated_ids)
generated_text = byte_seq.decode("utf-8", errors="replace")

print(f"generated token ids: {generated_ids}")
print(f"decoded text: {generated_text!r}")
print()
print("(the text will look like gibberish -- the model is UNTRAINED, random")
print(" weights. this step is only testing the generation MECHANICS: does the")
print(" loop correctly grow the sequence, sample validly, and decode back to")
print(" text without crashing.)")
