import torch
import torch.nn.functional as F

import sys
sys.path.append("../01_BPE_tokenizer")
from bpe_tokenizer import BPETokenizer

# --- reuse step 1's setup: a trained tokenizer + embeddings ---
tokenizer = BPETokenizer()
corpus = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks at the fox. the fox runs away. "
) * 20
tokenizer.train(corpus, num_merges=50)
vocab_size = len(tokenizer.vocab)

d_model = 16
max_seq_len = 32

token_embedding = torch.nn.Embedding(vocab_size, d_model)
position_embedding = torch.nn.Embedding(max_seq_len, d_model)

text = "the fox"
token_ids = torch.tensor(tokenizer.encode(text))
positions = torch.arange(len(token_ids))
x = token_embedding(token_ids) + position_embedding(positions)   # [seq_len, d_model]
seq_len = x.shape[0]

print(f"input x shape: {x.shape}")
print()

# --- self-attention, single head, minimal version ---

d_k = d_model   # single head uses the full d_model as the key/query dimension

W_Q = torch.nn.Linear(d_model, d_k, bias=False)
W_K = torch.nn.Linear(d_model, d_k, bias=False)
W_V = torch.nn.Linear(d_model, d_k, bias=False)

Q = W_Q(x)   # [seq_len, d_k]  -- "what each position is looking for"
K = W_K(x)   # [seq_len, d_k]  -- "what each position offers to be matched against"
V = W_V(x)   # [seq_len, d_k]  -- "the actual content each position hands over"

print(f"Q shape: {Q.shape}   (one query vector per position)")
print(f"K shape: {K.shape}   (one key vector per position)")
print(f"V shape: {V.shape}   (one value vector per position)")
print()

# raw similarity scores: score[i, j] = q_i . k_j
scores = Q @ K.T                     # [seq_len, seq_len]
scaled_scores = scores / (d_k ** 0.5)   # scale by sqrt(d_k) for numerical stability

print("raw scores (Q @ K^T), shape", scores.shape, ":")
print(scores)
print()

# causal mask: position i must not see position j > i (future tokens)
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print("causal mask (True = blocked / future position):")
print(causal_mask)
print()

masked_scores = scaled_scores.masked_fill(causal_mask, float("-inf"))
print("scaled + masked scores (before softmax):")
print(masked_scores)
print()

attn_weights = F.softmax(masked_scores, dim=-1)   # each row sums to 1
print("attention weights (after softmax), each row sums to 1:")
print(attn_weights)
print("row sums:", attn_weights.sum(dim=-1))
print()

output = attn_weights @ V   # [seq_len, d_k] -- weighted blend of value vectors
print(f"attention output shape: {output.shape}  (same shape as input x)")
print(output)
