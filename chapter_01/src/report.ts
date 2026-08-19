import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

export interface ChartDataset {
  label: string
  data: number[]
  yAxis?: 'left' | 'right'
}

export interface ChartSpec {
  kind: 'bar' | 'line'
  labels: string[]
  datasets: ChartDataset[]
  xLabel?: string
  yLabel?: string
}

export interface ScenarioResult {
  name: string
  description: string
  backend: string
  model: string
  timestamp: string
  charts: ChartSpec[]
  rows: Record<string, string | number>[]
}

function toCsv(rows: Record<string, string | number>[]): string {
  if (rows.length === 0) return ''
  const headers = Object.keys(rows[0])
  const lines = [headers.join(',')]
  for (const row of rows) {
    lines.push(headers.map((h) => String(row[h] ?? '')).join(','))
  }
  return lines.join('\n')
}

function renderTable(rows: Record<string, string | number>[]): string {
  if (rows.length === 0) return '<p>No data.</p>'
  const headers = Object.keys(rows[0])
  const head = headers.map((h) => `<th>${h}</th>`).join('')
  const body = rows
    .map((r) => `<tr>${headers.map((h) => `<td>${r[h] ?? ''}</td>`).join('')}</tr>`)
    .join('')
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`
}

export async function writeScenarioOutput(outDir: string, slug: string, result: ScenarioResult): Promise<string[]> {
  await mkdir(outDir, { recursive: true })

  const jsonPath = join(outDir, `${slug}.json`)
  const csvPath = join(outDir, `${slug}.csv`)
  await writeFile(jsonPath, JSON.stringify(result, null, 2) + '\n')
  await writeFile(csvPath, toCsv(result.rows) + '\n')

  const chartJsPath = join(__dirname, '..', 'node_modules', 'chart.js', 'dist', 'chart.umd.js')
  const chartJs = await readFile(chartJsPath, 'utf8')
  const data = JSON.stringify({ result })

  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${result.name} - inference_engg chapter_01</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #0f1115; color: #e6e6e6; }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.1rem; margin-top: 2.5rem; color: #8ab4f8; }
  .meta { color: #9aa0a6; font-size: 0.8rem; }
  .desc { color: #bdc1c6; }
  .chart-wrap { background: #16181d; border: 1px solid #2a2d33; border-radius: 8px; padding: 1rem; margin-top: 1rem; }
  canvas { max-height: 360px; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 1rem; }
  th, td { border: 1px solid #2a2d33; padding: 6px 10px; text-align: right; }
  th { background: #1b1e24; }
  td:first-child, th:first-child { text-align: left; }
</style>
</head>
<body>
<h1>${result.name}</h1>
<p class="meta">backend: ${result.backend} &middot; model(s): ${result.model} &middot; ${result.timestamp}</p>
<p class="desc">${result.description}</p>
${result.charts.map((_, i) => `<div class="chart-wrap"><canvas id="chart-${i}"></canvas></div>`).join('\n')}
<h2>Results</h2>
${renderTable(result.rows)}
<script>
${chartJs}
const DATA = ${data};
const palette = ['#4f9cf9', '#f9a84f', '#4fd0a0', '#e07be0'];
function buildChart(idx) {
  const spec = DATA.result.charts[idx];
  const hasRight = spec.datasets.some(d => d.yAxis === 'right');
  const datasets = spec.datasets.map((d, i) => ({
    label: d.label,
    data: d.data,
    backgroundColor: palette[i % palette.length],
    borderColor: palette[i % palette.length],
    yAxisID: d.yAxis === 'right' ? 'y1' : 'y',
    tension: 0.3,
    fill: false,
  }));
  const scales = {
    y: { title: { display: !!spec.yLabel, text: spec.yLabel || '' }, beginAtZero: true },
  };
  if (hasRight) {
    scales.y1 = { position: 'right', beginAtZero: true, grid: { drawOnChartArea: false } };
  }
  new Chart(document.getElementById('chart-' + idx), {
    type: spec.kind,
    data: { labels: spec.labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#e6e6e6' } } },
      scales,
    },
  });
}
DATA.result.charts.forEach((_, i) => buildChart(i));
</script>
</body>
</html>
`

  const htmlPath = join(outDir, `${slug}.html`)
  await writeFile(htmlPath, html)
  return [jsonPath, csvPath, htmlPath]
}