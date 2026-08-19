import { parseArgs } from 'node:util'
import { MockClient, OpenAIClient } from './client.ts'
import type { Client } from './client.ts'
import {
  runConcurrencyScenario,
  runModelScenario,
  runPromptOutputScenario,
  runStreamingScenario,
} from './scenarios.ts'
import { writeScenarioOutput } from './report.ts'
import type { ScenarioResult } from './report.ts'

const SCENARIOS = {
  1: { slug: 'scenario-1-streaming', run: runStreamingScenario },
  2: { slug: 'scenario-2-concurrency', run: runConcurrencyScenario },
  3: { slug: 'scenario-3-models', run: runModelScenario },
  4: { slug: 'scenario-4-prompt-output', run: runPromptOutputScenario },
} as const

const BACKEND_DEFAULTS = {
  mock: { model: 'mock-mid', models: ['mock-fast', 'mock-mid', 'mock-slow'], baseUrl: undefined, apiKey: undefined },
  ollama: { model: 'llama3.2:1b', models: ['llama3.2:1b', 'llama3.2:3b', 'llama3:8b'], baseUrl: 'http://localhost:11434/v1', apiKey: 'ollama' },
  openai: { model: 'gpt-4o-mini', models: ['gpt-4o-mini', 'gpt-4o'], baseUrl: undefined, apiKey: undefined },
} as const

function splitList(value: string | undefined): number[] | undefined {
  if (!value) return undefined
  return value.split(',').map((s) => s.trim()).filter(Boolean).map(Number)
}

function printTable(rows: Record<string, string | number>[]): void {
  if (rows.length === 0) return
  const headers = Object.keys(rows[0])
  const widths = headers.map((h) => Math.max(h.length, ...rows.map((r) => String(r[h] ?? '').length)))
  const fmt = (cells: string[]) => cells.map((c, i) => c.padEnd(widths[i])).join('  ')
  console.log(fmt(headers))
  console.log(fmt(widths.map((w) => '-'.repeat(w))))
  for (const row of rows) {
    console.log(fmt(headers.map((h) => String(row[h] ?? ''))))
  }
}

function printHelp(): void {
  console.log(`TTFT & TPS benchmark - inference_engg chapter_01

Usage: node src/cli.ts [options]

Options:
  --backend <mock|ollama|openai>   Backend to run against (default: mock)
  --scenario <1|2|3|4>             1 streaming vs non-streaming (default)
                                  2 concurrency & batching
                                  3 model selection
                                  4 prompt & output size
  --model <name>                   Single model (scenarios 1, 2, 4)
  --models <a,b,c>                 Comma-separated models (scenario 3)
  --base-url <url>                 OpenAI-compatible base URL
  --api-key <key>                  API key (or env OPENAI_API_KEY)
  --out <dir>                      Output directory (default: out)
  --repeats <n>                    Repeats per measurement (scenarios 1, 3)
  --concurrency <1,2,4,8,16>       Concurrency levels (scenario 2)
  --prompt-sizes <200,500,...>     Prompt length sweep in chars (scenario 4)
  --max-tokens-sweep <16,64,...>   max_tokens sweep (scenario 4)
  --max-tokens <n>                 max_tokens for other scenarios (default: 256)
  --prompt <text>                  Prompt used in the benchmark
  --help                           Show this help

Examples:
  node src/cli.ts --backend mock --scenario 1
  node src/cli.ts --backend ollama --model llama3.2:1b --scenario 2
  OPENAI_API_KEY=sk-... node src/cli.ts --backend openai --models gpt-4o-mini,gpt-4o --scenario 3
`)
}

function parseBackend(backend: string): keyof typeof BACKEND_DEFAULTS {
  if (backend in BACKEND_DEFAULTS) return backend as keyof typeof BACKEND_DEFAULTS
  throw new Error(`Unknown backend "${backend}". Choose from: ${Object.keys(BACKEND_DEFAULTS).join(', ')}`)
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      backend: { type: 'string', default: 'mock' },
      scenario: { type: 'string', default: '1' },
      model: { type: 'string' },
      models: { type: 'string' },
      'base-url': { type: 'string' },
      'api-key': { type: 'string' },
      out: { type: 'string', default: 'out' },
      repeats: { type: 'string' },
      concurrency: { type: 'string' },
      'prompt-sizes': { type: 'string' },
      'max-tokens-sweep': { type: 'string' },
      'max-tokens': { type: 'string' },
      prompt: { type: 'string' },
      help: { type: 'boolean', default: false },
    },
  })

  if (values.help) {
    printHelp()
    return
  }

  const backend = parseBackend(values.backend)
  const scenarioNum = Number(values.scenario)
  if (!(scenarioNum in SCENARIOS)) throw new Error(`Scenario must be 1-4, got "${values.scenario}"`)
  const scenario = SCENARIOS[scenarioNum as keyof typeof SCENARIOS]
  const defaults = BACKEND_DEFAULTS[backend]

  const baseUrl = values['base-url'] ?? defaults.baseUrl
  const apiKey = values['api-key'] ?? defaults.apiKey
  const maxTokens = Number(values['max-tokens'] ?? 256)

  let clients: Client[]
  if (backend === 'mock') {
    const models = values.models?.split(',').map((s) => s.trim()) ??
      (scenarioNum === 3 ? defaults.models : [values.model ?? defaults.model])
    clients = models.map((m) => new MockClient(m))
  } else {
    const models = (values.models ?? defaults.models.join(',')).split(',').map((s) => s.trim())
    const single = values.model ?? defaults.model
    clients = (scenarioNum === 3 ? models : [single]).map(
      (m) => new OpenAIClient({ model: m, baseUrl, apiKey }),
    )
  }

  console.log(`Backend: ${backend} | Scenario: ${scenarioNum} - ${scenario.slug}`)
  console.log(`Models:  ${clients.map((c) => c.model).join(', ')}\n`)

  for (const client of clients) {
    console.log(`  ${await client.ping()}`)
  }
  console.log('')

  let result: ScenarioResult
  switch (scenarioNum) {
    case 1:
      result = await runStreamingScenario(clients[0], {
        prompt: values.prompt,
        repeats: Number(values.repeats ?? 3),
        maxTokens,
      })
      break
    case 2:
      result = await runConcurrencyScenario(clients[0], {
        prompt: values.prompt,
        levels: splitList(values.concurrency) ?? [1, 2, 4, 8, 16],
        maxTokens,
      })
      break
    case 3:
      result = await runModelScenario(clients, {
        prompt: values.prompt,
        repeats: Number(values.repeats ?? 2),
        maxTokens,
      })
      break
    default:
      result = await runPromptOutputScenario(clients[0], {
        prompt: values.prompt,
        promptSizes: splitList(values['prompt-sizes']) ?? [200, 500, 1000, 2000, 4000],
        maxTokensSweep: splitList(values['max-tokens-sweep']) ?? [16, 64, 128, 256, 512],
      })
  }

  console.log(result.name)
  printTable(result.rows)

  const files = await writeScenarioOutput(values.out, scenario.slug, result)
  console.log(`\nWrote:`)
  for (const f of files) console.log(`  ${f}`)
}

main().catch((err: unknown) => {
  console.error(`\nError: ${err instanceof Error ? err.message : String(err)}`)
  process.exit(1)
})