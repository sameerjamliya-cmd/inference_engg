import type { ChatMessage, Client } from './client.ts'
import { mean, measureNonStreaming, measureStreaming, percentile } from './metrics.ts'
import type { ChartSpec, ScenarioResult } from './report.ts'
import type { StreamMetrics } from './metrics.ts'

export const DEFAULT_PROMPT =
  'Explain the difference between time-to-first-token (TTFT) and tokens-per-second (TPS) in LLM inference.'

const FILLER = 'the quick brown fox jumps over the lazy dog '

function padToChars(prompt: string, targetChars: number): string {
  if (prompt.length >= targetChars) return prompt
  let filler = ''
  while (filler.length < targetChars - prompt.length) {
    filler += FILLER
  }
  return filler.slice(0, targetChars - prompt.length) + prompt
}

function round(n: number, digits = 1): number {
  return Math.round(n * 10 ** digits) / 10 ** digits
}

export interface StreamingScenarioOptions {
  prompt?: string
  repeats?: number
  maxTokens?: number
}

export async function runStreamingScenario(client: Client, opts: StreamingScenarioOptions = {}): Promise<ScenarioResult> {
  const repeats = opts.repeats ?? 3
  const maxTokens = opts.maxTokens ?? 256
  const messages: ChatMessage[] = [{ role: 'user', content: opts.prompt ?? DEFAULT_PROMPT }]

  const rows: Record<string, string | number>[] = []
  const datasets: ChartSpec['datasets'] = [
    { label: 'TTFT (ms)', data: [] },
    { label: 'TPS (tok/s)', data: [] },
  ]

  for (const mode of ['streaming', 'non-streaming'] as const) {
    const runs: StreamMetrics[] = []
    for (let i = 0; i < repeats; i++) {
      const m = mode === 'streaming' ? await measureStreaming(client, messages, maxTokens) : await measureNonStreaming(client, messages, maxTokens)
      runs.push(m)
    }
    runs.sort((a, b) => a.totalMs - b.totalMs)
    const chosen = runs[Math.floor(runs.length / 2)]
    rows.push({
      mode,
      ttft_ms: round(chosen.ttftMs),
      total_ms: round(chosen.totalMs),
      tps: round(chosen.tps),
      tokens: chosen.outputTokens,
    })
    datasets[0].data.push(round(chosen.ttftMs))
    datasets[1].data.push(round(chosen.tps))
  }

  return {
    name: 'Streaming vs non-streaming',
    description: 'Same prompt and model, repeated ' + repeats + 'x per mode (median shown). Streaming exposes the first token immediately (TTFT ≈ 0), non-streaming returns everything at once (TTFT = total latency).',
    backend: client.model,
    model: client.model,
    timestamp: new Date().toISOString(),
    charts: [{ kind: 'bar', labels: ['streaming', 'non-streaming'], datasets, yLabel: 'ms / tok/s' }],
    rows,
  }
}

export interface ConcurrencyScenarioOptions {
  prompt?: string
  levels?: number[]
  maxTokens?: number
}

export async function runConcurrencyScenario(client: Client, opts: ConcurrencyScenarioOptions = {}): Promise<ScenarioResult> {
  const levels = opts.levels ?? [1, 2, 4, 8, 16]
  const maxTokens = opts.maxTokens ?? 200
  const messages: ChatMessage[] = [{ role: 'user', content: opts.prompt ?? DEFAULT_PROMPT }]

  const rows: Record<string, string | number>[] = []
  const tpsData: number[] = []
  const latencyData: number[] = []

  for (const level of levels) {
    const start = performance.now()
    const results = await Promise.all(Array.from({ length: level }, () => measureStreaming(client, messages, maxTokens)))
    const wallMs = performance.now() - start
    const aggregateTokens = results.reduce((acc, r) => acc + r.outputTokens, 0)
    const latencies = results.map((r) => r.totalMs).sort((a, b) => a - b)
    const aggregateTps = (aggregateTokens / wallMs) * 1000
    const avgLatency = mean(latencies)
    rows.push({
      concurrency: level,
      wall_ms: round(wallMs),
      aggregate_tps: round(aggregateTps),
      avg_latency_ms: round(avgLatency),
      p95_latency_ms: round(percentile(latencies, 0.95)),
      total_tokens: aggregateTokens,
    })
    tpsData.push(round(aggregateTps))
    latencyData.push(round(avgLatency))
  }

  return {
    name: 'Concurrency & batching',
    description: 'N identical streaming requests fired in parallel. Aggregate throughput (tokens/s across all requests) grows with concurrency while per-request latency degrades - the classic latency/throughput tradeoff.',
    backend: client.model,
    model: client.model,
    timestamp: new Date().toISOString(),
    charts: [
      {
        kind: 'line',
        labels: levels.map(String),
        datasets: [
          { label: 'Aggregate throughput (tok/s)', data: tpsData, yAxis: 'left' },
          { label: 'Avg latency (ms)', data: latencyData, yAxis: 'right' },
        ],
        xLabel: 'Concurrent requests',
      },
    ],
    rows,
  }
}

export interface ModelScenarioOptions {
  prompt?: string
  repeats?: number
  maxTokens?: number
}

export async function runModelScenario(clients: Client[], opts: ModelScenarioOptions = {}): Promise<ScenarioResult> {
  const repeats = opts.repeats ?? 2
  const maxTokens = opts.maxTokens ?? 128
  const messages: ChatMessage[] = [{ role: 'user', content: opts.prompt ?? DEFAULT_PROMPT }]

  const rows: Record<string, string | number>[] = []
  const ttftData: number[] = []
  const tpsData: number[] = []

  for (const client of clients) {
    const runs: StreamMetrics[] = []
    for (let i = 0; i < repeats; i++) {
      runs.push(await measureStreaming(client, messages, maxTokens))
    }
    runs.sort((a, b) => a.totalMs - b.totalMs)
    const chosen = runs[Math.floor(runs.length / 2)]
    rows.push({
      model: client.model,
      ttft_ms: round(chosen.ttftMs),
      total_ms: round(chosen.totalMs),
      tps: round(chosen.tps),
      tokens: chosen.outputTokens,
    })
    ttftData.push(round(chosen.ttftMs))
    tpsData.push(round(chosen.tps))
  }

  return {
    name: 'Model selection',
    description: 'Same prompt, different model sizes/configurations (quantized vs full precision). Small models win on TTFT and TPS; bigger models usually win on quality.',
    backend: clients[0]?.model ?? 'unknown',
    model: clients.map((c) => c.model).join(', '),
    timestamp: new Date().toISOString(),
    charts: [
      {
        kind: 'bar',
        labels: clients.map((c) => c.model),
        datasets: [
          { label: 'TTFT (ms)', data: ttftData },
          { label: 'TPS (tok/s)', data: tpsData },
        ],
        yLabel: 'ms / tok/s',
      },
    ],
    rows,
  }
}

export interface PromptOutputScenarioOptions {
  prompt?: string
  promptSizes?: number[]
  maxTokensSweep?: number[]
}

export async function runPromptOutputScenario(client: Client, opts: PromptOutputScenarioOptions = {}): Promise<ScenarioResult> {
  const promptSizes = opts.promptSizes ?? [200, 500, 1000, 2000, 4000]
  const maxTokensSweep = opts.maxTokensSweep ?? [16, 64, 128, 256, 512]
  const basePrompt = opts.prompt ?? DEFAULT_PROMPT

  const promptRows: Record<string, string | number>[] = []
  const ttftBySize: number[] = []
  for (const size of promptSizes) {
    const m = await measureStreaming(client, [{ role: 'user', content: padToChars(basePrompt, size) }], 64)
    promptRows.push({
      phase: 'prompt',
      prompt_chars: size,
      max_tokens: '',
      ttft_ms: round(m.ttftMs),
      decode_ms: '',
      total_ms: round(m.totalMs),
      tps: round(m.tps),
      tokens: '',
    })
    ttftBySize.push(round(m.ttftMs))
  }

  const outputRows: Record<string, string | number>[] = []
  const decodeByLimit: number[] = []
  const tpsByLimit: number[] = []
  for (const mt of maxTokensSweep) {
    const m = await measureStreaming(client, [{ role: 'user', content: basePrompt }], mt)
    outputRows.push({
      phase: 'output',
      prompt_chars: '',
      max_tokens: mt,
      ttft_ms: '',
      decode_ms: round(m.decodeMs),
      total_ms: round(m.totalMs),
      tps: round(m.tps),
      tokens: m.outputTokens,
    })
    decodeByLimit.push(round(m.decodeMs))
    tpsByLimit.push(round(m.tps))
  }

  return {
    name: 'Prompt & output size',
    description: 'Prefill scales with input length (prompt_chars -> TTFT). Decode is a per-token cost: capping max_tokens raises wall time linearly while TPS stays flat (a property of model + hardware).',
    backend: client.model,
    model: client.model,
    timestamp: new Date().toISOString(),
    charts: [
      {
        kind: 'line',
        labels: promptSizes.map(String),
        datasets: [{ label: 'TTFT (ms)', data: ttftBySize }],
        xLabel: 'Prompt length (chars)',
        yLabel: 'ms',
      },
      {
        kind: 'line',
        labels: maxTokensSweep.map(String),
        datasets: [
          { label: 'Decode time (ms)', data: decodeByLimit, yAxis: 'left' },
          { label: 'TPS (tok/s)', data: tpsByLimit, yAxis: 'right' },
        ],
        xLabel: 'max_tokens limit',
      },
    ],
    rows: [...promptRows, ...outputRows],
  }
}