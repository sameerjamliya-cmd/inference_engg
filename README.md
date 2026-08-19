# inference_engg

Applied LLM inference engineering, chapter by chapter.

## [Chapter 01 — TTFT & TPS](chapter_01/README.md)

TypeScript benchmark harness measuring time-to-first-token (TTFT) and tokens-per-second
(TPS) across four scenarios: streaming, concurrency/batching, model selection, and
prompt/output size. Backend-agnostic (Ollama, OpenAI-compatible cloud APIs, or a built-in
mock). Each run produces JSON + CSV + a self-contained HTML report with Chart.js charts.

```bash
cd chapter_01 && npm install
node src/cli.ts --backend mock --scenario 1
```