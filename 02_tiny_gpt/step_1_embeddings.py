import torch

import sys
sys.path.append("../01_BPE_tokenizer")
from bpe_tokenizer import BPETokenizer

# reuse a tokenizer trained the same way as Project 01, so vocab_size is real
tokenizer = BPETokenizer()
corpus = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks at the fox. the fox runs away. "
) * 20
tokenizer.train(corpus, num_merges=50)
vocab_size = len(tokenizer.vocab)   # 256 base bytes + however many merges actually got learned
print(f"vocab_size: {vocab_size}")

d_model = 16          # tiny embedding dimension, just to see shapes clearly
max_seq_len = 32      # max positions we'll ever support

token_embedding = torch.nn.Embedding(vocab_size, d_model)
position_embedding = torch.nn.Embedding(max_seq_len, d_model)

text = "the fox"
token_ids = torch.tensor(tokenizer.encode(text))       # shape: [seq_len]
positions = torch.arange(len(token_ids))               # shape: [seq_len] -> [0, 1, 2, ...]

tok_emb = token_embedding(token_ids)        # shape: [seq_len, d_model]
pos_emb = position_embedding(positions)     # shape: [seq_len, d_model]
x = tok_emb + pos_emb                       # shape: [seq_len, d_model]

print(f"text: {text!r}")
print(f"token_ids: {token_ids.tolist()}")
print(f"tok_emb shape: {tok_emb.shape}")
print(f"pos_emb shape: {pos_emb.shape}")
print(f"x (tok_emb + pos_emb) shape: {x.shape}")
print()
print("first token's embedding vector (tok_emb[0]):")
print(tok_emb[0])
print()
print("first position's positional vector (pos_emb[0]):")
print(pos_emb[0])
print()
print("their sum (x[0]) -- should equal tok_emb[0] + pos_emb[0]:")
print(x[0])
