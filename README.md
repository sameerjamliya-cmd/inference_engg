# Inference Engineering

A self-directed, hands-on curriculum for understanding LLM inference from
first principles up to production inference engines (vLLM, SGLang,
TensorRT-LLM).

## The approach

This is not a "call the library" exercise. Every early project is built
from scratch — no HF tokenizers, no tiktoken, no
`torch.nn.functional.scaled_dot_product_attention` — until we've built our
own version and compared it against the real thing. The goal is to be able
to look at production inference systems later and understand *why* they're
built the way they are, instead of treating them as black boxes.

Each project follows the same loop:

1. **Concept and math before code.** What problem exists, why it exists,
   how the naive approach works, why it's inefficient, what fixes it.
2. **Math → code mapping.** Every line of code traces back to a specific
   formula or algorithmic step — nothing is "magic."
3. **Simplest working version first**, built incrementally piece by piece
   and run by hand at each step, before any optimization.
4. **Test everything** — correctness checks (roundtrip tests, comparisons
   against reference implementations) before trusting any benchmark.
5. **Benchmark, then explain** the numbers, not just report them.

The full roadmap, working conventions, and progress log live in
[claude.md](claude.md).

## Project 01 — BPE Tokenizer, from scratch

[01_BPE_tokenizer/](01_BPE_tokenizer/)

Byte-level Byte Pair Encoding, the tokenization scheme behind GPT-style
models, built one piece at a time:

- **Byte encoding** — any text becomes a sequence of integers 0–255 (UTF-8
  bytes), giving a closed base vocabulary that can represent any language,
  script, or emoji with no "unknown token."
- **Pair-frequency counting** — count every adjacent token pair in the
  current sequence.
- **Merge selection** — pick the single most frequent pair.
- **Merge/rewrite** — replace every occurrence of that pair with a new
  token id, then repeat the count → pick → merge cycle for `k` rounds,
  recounting from scratch each round.
- **Encode** — apply the *learned* merge rules to new text, in the exact
  priority order they were discovered during training (not by
  recomputing frequencies).
- **Decode** — reconstruct the original bytes from token ids and decode
  back to a string.

Verified with roundtrip tests (encode → decode → identical to the
original) across a small hand-checked example, synthetic prose, unicode
and emoji text never seen during training, and a real ~2.4MB English word
list — correctness held at every scale, with compression improving as the
corpus and merge budget grew.
