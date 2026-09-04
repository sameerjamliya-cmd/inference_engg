# Inference Engineering — Learning Roadmap

## What this repo is

A progressive, hands-on curriculum to understand **LLM inference from first
principles up to production inference engines** (vLLM, SGLang, TensorRT-LLM).
This is NOT a "use the library" exercise — every early project is built from
scratch (no HF tokenizers, no tiktoken, no `torch.nn.functional.scaled_dot_product_attention`
until we've built our own and compared). The goal is to be able to look at
real systems later and understand *why* they're built the way they are,
not treat them as black boxes.

## How I (the user) want to work

1. **Concept and math BEFORE code, always.** For every new piece of work,
   explain: what problem exists → why it exists → how the naive approach
   works → why it's inefficient → what fixes it → how much improvement we
   actually measure. Only after that discussion do we write code.
2. **Math → code mapping.** Once we do write code, connect each formula or
   algorithmic step explicitly to the line(s) of code that implement it.
   Nothing should feel like "magic" — if a line exists, I should be able to
   point to the math concept it encodes.
3. **Simplest working version first.** Get a correct, readable, from-scratch
   implementation working before any optimization. Then benchmark it. Then
   optimize, and benchmark again, so the improvement is measured, not assumed.
4. **Test everything.** Every project needs a test/verification script —
   correctness checks (e.g. roundtrip tests, comparison against a reference
   like PyTorch's SDPA) before we trust benchmark numbers.
5. **Explain results, don't just print them.** After benchmarks/profiling,
   give a short explanation of *why* the numbers look the way they do.
6. Don't jump ahead to later projects unless asked — we go in order, one
   project at a time, and I confirm before moving to the next.
7. **Don't just hand me finished files.** For each project, build it WITH
   me incrementally: explain the next small piece (e.g. "counting pair
   frequencies"), show me a short snippet or ask me to write it, let me
   run it and see the output, THEN move to the next piece (e.g. "picking
   the max pair", then "merging", then "the training loop", then
   "encode/decode"). I want to type/run/see-output at each step, not
   receive a complete file to read passively. Only assemble the final
   clean file once I've built and understood each piece separately.

## Environment

- Language: Python + PyTorch.
- Primary dev machine: VS Code on Apple Silicon (MPS backend). Use
  `torch.backends.mps.is_available()` for device selection; fall back to CPU.
- CUDA-specific work (Triton kernels, custom CUDA ops, torch.profiler CUDA
  traces, vLLM/SGLang/TensorRT-LLM deployment) happens on a separate
  NVIDIA/CUDA machine — flag clearly in code/comments when something is
  CUDA-only and won't run on MPS/CPU.
- Package installs: prefer stdlib + torch + numpy + matplotlib. Only add a
  new dependency when the project genuinely needs it (e.g. `triton` in
  Project on custom kernels, `vllm`/`sglang` in the final projects).

## Repo structure

```
inference-engine/
├── CLAUDE.md                     <- this file
├── 01_bpe_tokenizer/             <- DONE
├── 02_tiny_gpt/                  <- IN PROGRESS, build this with me now
├── 03_sdpa/
├── 04_flash_attention/
├── 05_attention_profiling/
├── 06_int8_quant/
├── 07_gptq_quant/
├── 08_quant_bitwidth_sweep/
├── 09_speculative_decoding/
├── 10_kv_cache_manager/
├── 11_prefix_caching/
├── 12_tensor_parallel_sim/
├── 13_ops_byte_benchmark/
├── 14_torch_profiler/
├── 15_triton_kernel/             <- CUDA/Triton required
├── 16_custom_cuda_op/            <- CUDA required
├── 17_vllm_deploy/               <- real GPU + vLLM required
├── 18_sglang_deploy/             <- real GPU + SGLang required
├── 19_tensorrt_llm/              <- real GPU + TensorRT-LLM required
├── 20_nvidia_dynamo/             <- real GPU + Dynamo required
├── 21_continuous_batching_sim/
├── 22_paged_attention_viz/
└── 23_ttft_throughput_benchmark/
```

## Progress log

- [x] **01 — BPE Tokenizer from scratch. DONE.**
  Built and run incrementally, piece by piece: byte encoding, pair-frequency
  counting, argmax pair selection, merge/rewrite, the training loop
  (recounting from scratch each round), encode() respecting learned merge
  priority order, and decode(). Verified with roundtrip tests on a tiny
  hand-checked example, synthetic prose, out-of-distribution unicode/emoji
  text, and a real ~2.4MB English word-list corpus (correctness held at
  every scale; compression improved with corpus/merge-budget size).
- [ ] **02 — Tiny GPT + autoregressive decoder loop. (IN PROGRESS — current focus)**
  Understand the math conceptually (embedding + positional embedding ->
  transformer block stack -> last-position logits -> softmax -> sample ->
  append -> repeat), and why naive generation wastefully recomputes
  attention for every earlier token every step (motivates Project 10's KV
  cache). Building incrementally: embeddings, then attention (only after a
  dedicated Q/K/V discussion), MLP, one full transformer block, stacking
  into a model class, sampling functions, the naive generation loop, and a
  latency-vs-length benchmark. Weights are randomly initialized —
  mechanics of the forward pass and generation loop are the point, not
  coherent text.
- [ ] 03 — Scaled dot-product attention from scratch
- [ ] 04 — Simplified Flash Attention
- [ ] 05 — Attention memory/perf profiling
- [ ] 06 — INT8 quantization pipeline
- [ ] 07 — GPTQ-style round-to-nearest with Hessian weighting
- [ ] 08 — Quantization bit-width sweep vs perplexity
- [ ] 09 — Speculative decoding (draft + target)
- [ ] 10 — KV cache manager (block allocator, eviction)
- [ ] 11 — Prefix caching (hash-based dedup)
- [ ] 12 — Tensor parallelism simulation
- [ ] 13 — Ops:byte ratio benchmark
- [ ] 14 — torch.profiler CUDA profiling
- [ ] 15 — Custom Triton elementwise kernel
- [ ] 16 — PyTorch custom op with CUDA backend
- [ ] 17 — vLLM deploy + TTFT/throughput benchmark
- [ ] 18 — SGLang deploy + structured output latency
- [ ] 19 — TensorRT-LLM compile vs eager comparison
- [ ] 20 — NVIDIA Dynamo disaggregated prefill
- [ ] 21 — Continuous batching simulation
- [ ] 22 — PagedAttention block layout visualization
- [ ] 23 — TTFT vs throughput tradeoff benchmark

## Conventions for code in this repo

- Every project folder gets: the from-scratch implementation, a
  `test_*.py` with correctness checks, and (where applicable) a
  benchmark/profiling script with printed or plotted results.
- Prefer explicit, heavily-commented code over clever/compressed code —
  readability for learning purposes beats performance in the "naive"
  implementations. Optimized versions come later, explicitly labeled.
- When comparing our implementation against a PyTorch built-in (e.g. our
  SDPA vs `F.scaled_dot_product_attention`, our attention vs Flash
  Attention), always include a numerical correctness check (max abs diff /
  cosine similarity) alongside the speed/memory comparison.