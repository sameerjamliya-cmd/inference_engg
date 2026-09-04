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
        x = x + self.attn(self.ln1(x))    # residual around attention
        x = x + self.mlp(self.ln2(x))     # residual around MLP
        return x


# --- run it on a real tokenized sentence ---
tokenizer = BPETokenizer()
corpus = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks at the fox. the fox runs away. "
) * 20
tokenizer.train(corpus, num_merges=50)
vocab_size = len(tokenizer.vocab)

d_model = 16
d_ff = 4 * d_model
max_seq_len = 32

token_embedding = torch.nn.Embedding(vocab_size, d_model)
position_embedding = torch.nn.Embedding(max_seq_len, d_model)

text = "the fox"
token_ids = torch.tensor(tokenizer.encode(text))
positions = torch.arange(len(token_ids))
x = token_embedding(token_ids) + position_embedding(positions)

print(f"input to transformer block, shape: {x.shape}")

block = TransformerBlock(d_model, d_ff)
output = block(x)

print(f"output of transformer block, shape: {output.shape}")
print(f"same shape as input? {output.shape == x.shape}")
print()
print("output values:")
print(output)
