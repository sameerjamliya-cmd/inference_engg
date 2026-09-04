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
        # tie output projection to the token embedding table (standard GPT trick):
        # reuse the same [vocab_size, d_model] matrix for both "encode a token"
        # and "decode a score for every token" -- saves vocab_size*d_model params.
        self.output_projection = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    def forward(self, token_ids):
        seq_len = token_ids.shape[0]
        positions = torch.arange(seq_len)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.output_projection(x)   # [seq_len, vocab_size]
        return logits


# --- run a forward pass on a real tokenized sentence ---
tokenizer = BPETokenizer()
corpus = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks at the fox. the fox runs away. "
) * 20
tokenizer.train(corpus, num_merges=50)
vocab_size = len(tokenizer.vocab)

model = TinyGPT(vocab_size=vocab_size, d_model=16, d_ff=64, n_layers=4, max_seq_len=32)

text = "the fox runs"
token_ids = torch.tensor(tokenizer.encode(text))
print(f"text: {text!r}")
print(f"token_ids: {token_ids.tolist()}  (seq_len={len(token_ids)})")
print()

logits = model(token_ids)
print(f"logits shape: {logits.shape}   (expected: [seq_len={len(token_ids)}, vocab_size={vocab_size}])")
print()

last_logits = logits[-1]   # the only row we actually use for next-token prediction
print(f"last position's logits shape: {last_logits.shape}")
print(f"top-5 predicted token ids (untrained, so meaningless): "
      f"{torch.topk(last_logits, 5).indices.tolist()}")

probs = F.softmax(last_logits, dim=-1)
print(f"do the predicted probabilities sum to 1? {torch.isclose(probs.sum(), torch.tensor(1.0))}")
