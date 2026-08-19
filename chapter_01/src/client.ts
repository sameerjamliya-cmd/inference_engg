import OpenAI from 'openai'

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface Client {
  readonly model: string
  streamText(messages: ChatMessage[], maxTokens: number): AsyncIterable<string>
  completeText(messages: ChatMessage[], maxTokens: number): Promise<string>
  ping(): Promise<string>
}

export interface OpenAIClientOptions {
  model: string
  baseUrl?: string
  apiKey?: string
}

export class OpenAIClient implements Client {
  readonly model: string
  private readonly client: OpenAI

  constructor(opts: OpenAIClientOptions) {
    this.model = opts.model
    this.client = new OpenAI({
      baseURL: opts.baseUrl,
      apiKey: opts.apiKey ?? process.env.OPENAI_API_KEY,
      timeout: 300_000,
    })
  }

  async *streamText(messages: ChatMessage[], maxTokens: number): AsyncIterable<string> {
    const stream = await this.client.chat.completions.create({
      model: this.model,
      messages,
      max_tokens: maxTokens,
      stream: true,
    })
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content
      if (delta) yield delta
    }
  }

  async completeText(messages: ChatMessage[], maxTokens: number): Promise<string> {
    const res = await this.client.chat.completions.create({
      model: this.model,
      messages,
      max_tokens: maxTokens,
      stream: false,
    })
    return res.choices[0]?.message?.content ?? ''
  }

  async ping(): Promise<string> {
    await this.completeText([{ role: 'user', content: 'ping' }], 4)
    return `OpenAI-compatible endpoint reachable (model: ${this.model})`
  }
}

export interface MockSpec {
  ttftBaseMs: number
  ttftPerCharMs: number
  msPerTokenMin: number
  msPerTokenMax: number
  outputTokensBase: number
}

export const MOCK_MODELS: Record<string, MockSpec> = {
  'mock-fast': { ttftBaseMs: 20, ttftPerCharMs: 0.03, msPerTokenMin: 8, msPerTokenMax: 18, outputTokensBase: 40 },
  'mock-mid': { ttftBaseMs: 60, ttftPerCharMs: 0.08, msPerTokenMin: 25, msPerTokenMax: 55, outputTokensBase: 60 },
  'mock-slow': { ttftBaseMs: 150, ttftPerCharMs: 0.2, msPerTokenMin: 60, msPerTokenMax: 120, outputTokensBase: 80 },
}

const MOCK_VOCAB = [
  'inference', 'latency', 'throughput', 'prefill', 'decode', 'kv', 'cache',
  'attention', 'token', 'batching', 'streaming', 'quantization', 'model', 'server',
]

function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export class MockClient implements Client {
  readonly model: string
  private readonly spec: MockSpec
  private readonly rng: () => number

  constructor(model: string) {
    const spec = MOCK_MODELS[model]
    if (!spec) {
      throw new Error(`Unknown mock model "${model}". Available: ${Object.keys(MOCK_MODELS).join(', ')}`)
    }
    this.model = model
    this.spec = spec
    this.rng = mulberry32(42)
  }

  async *streamText(messages: ChatMessage[], maxTokens: number): AsyncIterable<string> {
    const inputChars = messages.reduce((acc, m) => acc + m.content.length, 0)
    await sleep(this.spec.ttftBaseMs + this.spec.ttftPerCharMs * inputChars)
    const tokens = Math.min(maxTokens, this.spec.outputTokensBase + Math.floor(this.rng() * 40))
    for (let i = 0; i < tokens; i++) {
      await sleep(this.spec.msPerTokenMin + this.rng() * (this.spec.msPerTokenMax - this.spec.msPerTokenMin))
      yield MOCK_VOCAB[Math.floor(this.rng() * MOCK_VOCAB.length)] + ' '
    }
  }

  async completeText(messages: ChatMessage[], maxTokens: number): Promise<string> {
    let text = ''
    for await (const delta of this.streamText(messages, maxTokens)) {
      text += delta
    }
    return text
  }

  async ping(): Promise<string> {
    return `Mock backend ok (model: ${this.model}, deterministic seed)`
  }
}