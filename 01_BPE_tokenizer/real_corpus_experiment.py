import time
from bpe_tokenizer import BPETokenizer

# Real English text: macOS's built-in dictionary word list (Webster's Second),
# one word per line, alphabetically sorted. Much bigger and more realistic
# than our earlier hand-written paragraphs -- ~2.4MB total. We take a 300KB
# slice (alphabetically early words, so anything starting with, say, 'z'
# below is guaranteed to be held-out / never seen during training).
with open("/usr/share/dict/words", encoding="utf-8") as f:
    full_text = f.read()

# set to len(full_text) to train on the entire ~2.4MB word list (takes ~1 minute)
CORPUS_SIZE = len(full_text)
corpus = full_text[:CORPUS_SIZE]
n_bytes = len(corpus.encode("utf-8"))
print(f"corpus: {n_bytes} bytes (~{n_bytes // 1024} KB) of real English words")
print(f"sample: {corpus[:80]!r}...")
print()

tokenizer = BPETokenizer()
t0 = time.time()
final_ids = tokenizer.train(corpus, num_merges=200)
elapsed = time.time() - t0

print(f"trained {len(tokenizer.merges)} merges in {elapsed:.1f}s")
print(f"compression: {n_bytes} bytes -> {len(final_ids)} tokens ({n_bytes/len(final_ids):.2f}x)")
print()

# peek at what some of the learned merges actually spell out -- do they look
# like real English morphemes (common prefixes/suffixes/letter pairs)?
print("sample of learned merge tokens (decoded back to text):")
sample_ids = [256, 260, 270, 290, 320, 350, 400, 450]
for tid in sample_ids:
    if tid in tokenizer.vocab:
        piece = tokenizer.vocab[tid].decode("utf-8", errors="replace")
        print(f"  id {tid}: {piece!r}")
print()

# held-out test: words from the END of the dictionary (never in our 300KB slice)
held_out_words = full_text[-2000:].split("\n")
held_out_words = [w for w in held_out_words if w][:5]

print("roundtrip test on held-out words (from the far end of the word list):")
for word in held_out_words:
    ids = tokenizer.encode(word)
    decoded = tokenizer.decode(ids)
    status = "OK" if decoded == word else "FAIL"
    print(f"  [{status}] {word!r} -> {len(word.encode('utf-8'))} bytes -> {len(ids)} tokens")
    assert decoded == word

print()
print("all roundtrip tests passed")
