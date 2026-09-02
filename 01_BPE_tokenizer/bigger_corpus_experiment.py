from bpe_tokenizer import BPETokenizer

# a few hundred words of varied prose -- no repeated identical sentences this time,
# so merges have to earn their place on genuinely recurring substrings (common
# words, common word-fragments), not just one repeated phrase.
corpus = """
Inference engineering is the discipline of making trained neural networks run
fast and cheap in production. A model that trains in a research lab still has
to serve millions of requests per day once it ships, and the bottlenecks that
matter change completely between training and inference. During training you
care about throughput across huge batches. During inference you often care
about latency for a single request, and about how many concurrent requests a
single GPU can serve before memory runs out.

The key data structure in autoregressive text generation is the KV cache. Each
attention layer caches the key and value vectors for every token that has
already been generated, so that generating the next token does not require
recomputing attention over the entire prefix from scratch. As sequences get
longer, the KV cache grows linearly with sequence length and with the number
of layers and heads in the model, which is why memory management for the KV
cache became its own subfield, with techniques like paged attention and
prefix caching designed specifically to reduce wasted memory.

Byte pair encoding is the tokenization algorithm most modern large language
models use to convert raw text into a sequence of integers before it ever
reaches the model. The algorithm starts from the raw bytes of the input text,
counts how often every pair of adjacent tokens occurs, merges the single most
frequent pair into a new token, and repeats this process a fixed number of
times. Because it always starts from raw bytes rather than characters or
words, byte pair encoding never encounters an unknown token, no matter what
language, script, or emoji appears in the input text.

Quantization reduces the numerical precision used to store model weights,
trading a small amount of accuracy for a large reduction in memory footprint
and often a meaningful increase in inference speed. Techniques like GPTQ use
a Hessian-based weighting scheme to decide which weights can tolerate more
aggressive rounding without harming model quality, while simpler schemes like
plain int8 quantization apply a uniform scale across each tensor. Bit-width
sweeps let engineers empirically measure the tradeoff between compression
ratio and perplexity for a specific model and dataset.

Speculative decoding speeds up autoregressive generation by using a small,
fast draft model to propose several tokens ahead, and then verifying all of
those proposed tokens with the larger target model in a single forward pass.
When the draft model's guesses match what the target model would have chosen
anyway, several tokens get accepted per target-model forward pass instead of
just one, which can meaningfully reduce end to end latency without changing
the output distribution at all.
""".strip()

print(f"corpus length: {len(corpus)} characters, {len(corpus.encode('utf-8'))} bytes")
print()

print(f"{'num_merges':>10} | {'tokens':>8} | {'compression':>12}")
print("-" * 38)
for num_merges in [0, 25, 50, 100, 200, 300, 500]:
    tok = BPETokenizer()
    final_ids = tok.train(corpus, num_merges=num_merges)
    n_bytes = len(corpus.encode("utf-8"))
    ratio = n_bytes / len(final_ids)
    print(f"{num_merges:>10} | {len(final_ids):>8} | {ratio:>11.2f}x")

print()

# now train once with a healthy number of merges and check roundtrip
tokenizer = BPETokenizer()
tokenizer.train(corpus, num_merges=300)
print(f"learned {len(tokenizer.merges)} actual merges (may stop early if pairs run out)")
print()

held_out_sentences = [
    "The KV cache is the single most important data structure in modern inference engines.",
    "Quantization and speculative decoding are complementary optimizations.",
    "unicode check: 日本語のテスト, emoji 🚀🔥👋, and accented café naïve",
]

for text in held_out_sentences:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    status = "OK" if decoded == text else "FAIL"
    n_bytes = len(text.encode("utf-8"))
    print(f"[{status}] {n_bytes} bytes -> {len(ids)} tokens | {text!r}")
    assert decoded == text
