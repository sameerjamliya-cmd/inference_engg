# Chapter 01 — TTFT & TPS: improving latency and throughput

A small TypeScript benchmark harness for the two numbers that dominate LLM inference
latency/throughput engineering:

| Metric | Name | Measures |
|--------|------|----------|
| **TTFT** | Time To First Token | Latency from request start to the **first** output token. Dominated by **prefill** (processing the prompt). |
| **TPS** | Tokens Per Second | **Decode** speed: how many tokens the model emits per second once generation starts. |

Total latency ≈ `TTFT + output_tokens / TPS`.

## The two phases of inference

- **Prefill** processes the whole prompt in parallel (one forward pass per layer). Cost scales with **input length** → TTFT grows with prompt size.
- **Decode** generates tokens one at a time (each token needs a full forward pass; the KV cache is what makes this cheaper than re-prefilling). Cost is **per token** → TPS is a property of the model + hardware, roughly flat regardless of `max_tokens`.

## The four levers (experiments)

Run each scenario one at a time — each writes its own report.

### 1. Streaming vs non-streaming
Streaming returns the first token as soon as it is produced; non-streaming waits for the
complete answer. TTFT collapses with streaming; TPS stays roughly equal.

```bash
node src/cli.ts --backend mock --scenario 1          # offline demo
node src/cli.ts --backend ollama --model llama3.2:1b --scenario 1
```

### 2. Concurrency & batching
Fire N identical requests in parallel. The server batches them: aggregate throughput
(tokens/s across all requests) rises, while per-request latency degrades — the
latency/throughput tradeoff curve.

```bash
node src/cli.ts --backend ollama --model llama3.2:1b --scenario 2 --concurrency 1,2,4,8,16
```

### 3. Model selection
Same prompt, several model sizes/configs. Small or quantized models win on TTFT and TPS;
larger models win on quality. Selection = find the smallest model that meets your quality bar.

```bash
node src/cli.ts --backend ollama --models llama3.2:1b,llama3.2:3b,llama3:8b --scenario 3
node src/cli.ts --backend mock --scenario 3          # mock-fast/mid/slow
```

### 4. Prompt & output size
Sweep input length → prefill grows → **TTFT rises**. Sweep `max_tokens` → decode time grows
linearly while **TPS stays flat**.

```bash
node src/cli.ts --backend ollama --model llama3.2:1b --scenario 4
```

## Setup

```bash
cd chapter_01
npm install

# optional: local models (no API key needed)
brew install ollama          # or: https://ollama.com/download
ollama pull llama3.2:1b
ollama pull llama3.2:3b

# or use a cloud API (needs key)
export OPENAI_API_KEY=sk-...
```

## Usage

```
node src/cli.ts --backend mock --scenario 1        # instant demo, no model needed
node src/cli.ts --backend ollama --model llama3.2:1b --scenario 2
OPENAI_API_KEY=... node src/cli.ts --backend openai --model gpt-4o-mini --scenario 1
```

Run `node src/cli.ts --help` for all options (repeats, prompt, sweeps, output dir...).

## Output (in `out/`)

- `scenario-N-<name>.json` — full result (metadata + metrics rows)
- `scenario-N-<name>.csv` — flat table
- `scenario-N-<name>.html` — self-contained interactive report (Chart.js inlined, just open it in a browser)

## Metrics conventions

- TTFT and TPS are measured from the streamed response: TTFT = time to first delta,
  TPS = output tokens / decode time (total minus TTFT).
- Non-streaming has no first-token signal, so TTFT = total latency there.
- Tokens are estimated as whitespace-separated words (exact counts depend on the tokenizer).
- Streaming mode repeats each measurement and reports the median run to dampen noise.