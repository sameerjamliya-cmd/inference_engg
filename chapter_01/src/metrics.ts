import type { ChatMessage, Client } from './client.ts'

export interface StreamMetrics {
  ttftMs: number
  decodeMs: number
  totalMs: number
  outputTokens: number
  tps: number
  text: string
}

export function estimateTokens(text: string): number {
  const trimmed = text.trim()
  if (!trimmed) return 0
  return trimmed.split(/\s+/).length
}

export async function measureStreaming(
  client: Client,
  messages: ChatMessage[],
  maxTokens: number,
): Promise<StreamMetrics> {
  const start = performance.now()
  let ttftMs = 0
  let text = ''
  for await (const delta of client.streamText(messages, maxTokens)) {
    if (ttftMs === 0) ttftMs = performance.now() - start
    text += delta
  }
  const totalMs = performance.now() - start
  const outputTokens = estimateTokens(text)
  const decodeMs = totalMs - ttftMs
  const tps = decodeMs > 0 ? (outputTokens / decodeMs) * 1000 : 0
  return { ttftMs, decodeMs, totalMs, outputTokens, tps, text }
}

export async function measureNonStreaming(
  client: Client,
  messages: ChatMessage[],
  maxTokens: number,
): Promise<StreamMetrics> {
  const start = performance.now()
  const text = await client.completeText(messages, maxTokens)
  const totalMs = performance.now() - start
  const outputTokens = estimateTokens(text)
  return { ttftMs: totalMs, decodeMs: totalMs, totalMs, outputTokens, tps: (outputTokens / totalMs) * 1000, text }
}

export function median(sorted: number[]): number {
  if (sorted.length === 0) return 0
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

export function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length))
  return sorted[idx]
}

export function mean(xs: number[]): number {
  if (xs.length === 0) return 0
  return xs.reduce((a, b) => a + b, 0) / xs.length
}