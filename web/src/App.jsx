import { useEffect, useMemo, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const initialTensor = {
  rows: 64,
  columns: 128,
  scheme: 'symmetric',
  granularity: 'per-row',
  seed: 7,
  standard_deviation: 0.5,
}

const initialModel = {
  model: 'gpt2',
  backend: 'fp32',
  device: 'auto',
  prompt: 'Quantization helps language models by',
  max_new_tokens: 48,
}

async function api(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`)
  return payload
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined) return '—'
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '—'
  return new Intl.NumberFormat('en-GB', { maximumFractionDigits: digits }).format(value)
}

function formatBytes(value) {
  if (!value) return '—'
  const units = ['B', 'KiB', 'MiB', 'GiB']
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1)
  return `${formatNumber(value / 1024 ** index, 2)} ${units[index]}`
}

function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  )
}

function Metric({ label, value, accent }) {
  return (
    <div className={`metric ${accent ? 'metric-accent' : ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function Scatter({ sample }) {
  const points = useMemo(() => {
    if (!sample) return []
    const all = [...sample.original, ...sample.quantized]
    const min = Math.min(...all)
    const max = Math.max(...all)
    const range = max - min || 1
    return sample.original.map((value, index) => ({
      x: 18 + (index / Math.max(sample.original.length - 1, 1)) * 564,
      yOriginal: 154 - ((value - min) / range) * 132,
      yQuantized: 154 - ((sample.quantized[index] - min) / range) * 132,
    }))
  }, [sample])

  if (!sample) return <div className="chart-empty">Run an experiment to plot reconstruction error.</div>
  return (
    <svg className="chart" viewBox="0 0 600 172" role="img" aria-label="Original and quantized samples">
      <line x1="18" x2="582" y1="154" y2="154" className="axis" />
      {points.map((point, index) => (
        <g key={index}>
          <circle cx={point.x} cy={point.yOriginal} r="2.5" className="point-original" />
          <circle cx={point.x} cy={point.yQuantized} r="1.7" className="point-quantized" />
        </g>
      ))}
      <text x="20" y="16">sampled weights</text>
    </svg>
  )
}

function TensorLab() {
  const [form, setForm] = useState(initialTensor)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const update = (event) => {
    const { name, value, type } = event.target
    setForm((current) => ({ ...current, [name]: type === 'number' ? Number(value) : value }))
  }

  const run = async (event) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      setResult(await api('/api/tensor', { method: 'POST', body: JSON.stringify(form) }))
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="workspace">
      <form className="control-panel" onSubmit={run}>
        <div className="section-heading">
          <span className="eyebrow">01 · NUMERICAL LAB</span>
          <h2>Tensor quantization</h2>
          <p>Compare real signed INT8 storage with reconstructed FP32 values.</p>
        </div>
        <div className="field-grid">
          <Field label="Rows"><input name="rows" type="number" min="1" max="2048" value={form.rows} onChange={update} /></Field>
          <Field label="Columns"><input name="columns" type="number" min="1" max="2048" value={form.columns} onChange={update} /></Field>
        </div>
        <Field label="Quantization scheme">
          <select name="scheme" value={form.scheme} onChange={update}>
            <option value="symmetric">Symmetric INT8</option>
            <option value="affine">Affine INT8</option>
          </select>
        </Field>
        <Field label="Granularity">
          <select name="granularity" value={form.granularity} onChange={update}>
            <option value="per-row">Per row</option>
            <option value="per-tensor">Per tensor</option>
          </select>
        </Field>
        <div className="field-grid">
          <Field label="Seed"><input name="seed" type="number" value={form.seed} onChange={update} /></Field>
          <Field label="Standard deviation"><input name="standard_deviation" type="number" min="0.01" max="100" step="0.1" value={form.standard_deviation} onChange={update} /></Field>
        </div>
        <button className="primary" disabled={busy}>{busy ? 'Quantizing…' : 'Run experiment'}</button>
        {error && <p className="error">{error}</p>}
      </form>

      <div className="results-panel">
        <div className="metric-grid">
          <Metric label="Compression" value={result ? `${formatNumber(result.compression_ratio, 2)}×` : '—'} accent />
          <Metric label="SQNR" value={result ? `${formatNumber(result.metrics.sqnr_db, 2)} dB` : '—'} />
          <Metric label="Cosine similarity" value={result ? formatNumber(result.metrics.cosine_similarity, 6) : '—'} />
          <Metric label="Mean squared error" value={result ? formatNumber(result.metrics.mse, 8) : '—'} />
        </div>
        <div className="chart-card">
          <div className="card-title"><span>Reconstruction trace</span><small>FP32 ● / INT8 ●</small></div>
          <Scatter sample={result?.sample} />
        </div>
        <div className="storage-row">
          <div><span>FP32 storage</span><strong>{formatBytes(result?.fp32_bytes)}</strong></div>
          <div className="arrow">→</div>
          <div><span>INT8 + metadata</span><strong>{formatBytes(result?.int8_bytes)}</strong></div>
        </div>
      </div>
    </section>
  )
}

function ModelLab() {
  const [form, setForm] = useState(initialModel)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const update = (event) => setForm((current) => ({ ...current, [event.target.name]: event.target.type === 'number' ? Number(event.target.value) : event.target.value }))

  const run = async (event) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      setResult(await api('/api/generate', { method: 'POST', body: JSON.stringify(form) }))
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="workspace">
      <form className="control-panel" onSubmit={run}>
        <div className="section-heading">
          <span className="eyebrow">02 · MODEL LAB</span>
          <h2>Local generation</h2>
          <p>The first run may download model weights. Later runs reuse the in-memory model.</p>
        </div>
        <Field label="Hugging Face model"><input name="model" value={form.model} onChange={update} /></Field>
        <Field label="Backend">
          <select name="backend" value={form.backend} onChange={update}>
            <option value="fp32">FP32</option><option value="fp16">FP16</option>
            <option value="bnb-int8">bitsandbytes INT8</option><option value="openvino-int8">OpenVINO INT8</option>
          </select>
        </Field>
        <Field label="Device" hint="Use auto unless targeting a specific accelerator."><input name="device" value={form.device} onChange={update} /></Field>
        <Field label="Prompt"><textarea name="prompt" rows="6" value={form.prompt} onChange={update} /></Field>
        <Field label="Maximum new tokens"><input name="max_new_tokens" type="number" min="1" max="512" value={form.max_new_tokens} onChange={update} /></Field>
        <button className="primary" disabled={busy}>{busy ? 'Loading and generating…' : 'Generate locally'}</button>
        {error && <p className="error">{error}</p>}
      </form>
      <div className="results-panel model-output">
        <div className="output-header">
          <div><span className="eyebrow">MODEL OUTPUT</span><h3>{result?.model || form.model}</h3></div>
          <span className="backend-pill">{result?.backend || form.backend}</span>
        </div>
        <div className="generated-text">{result?.text || 'Generated text will appear here.'}</div>
        <div className="output-meta">
          <span>Tokens <strong>{result?.generated_tokens ?? '—'}</strong></span>
          <span>Device <strong>{result?.device ?? '—'}</strong></span>
          <span>Privacy <strong>Local</strong></span>
        </div>
      </div>
    </section>
  )
}

function BenchmarkLab() {
  const [form, setForm] = useState({ model: 'gpt2', backend: 'fp32', device: 'auto', corpus: 'Quantization compresses neural networks.\nA shared corpus makes comparisons fair.', prompts: 'Large language models can\nEight bit inference is', max_length: 1024, stride: 512, max_new_tokens: 32, repeats: 1 })
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const update = (event) => setForm((current) => ({ ...current, [event.target.name]: event.target.type === 'number' ? Number(event.target.value) : event.target.value }))
  const run = async (event) => {
    event.preventDefault(); setBusy(true); setError('')
    const payload = { ...form, corpus: form.corpus.split('\n').filter(Boolean), prompts: form.prompts.split('\n').filter(Boolean) }
    try { setResult(await api('/api/benchmark', { method: 'POST', body: JSON.stringify(payload) })) }
    catch (requestError) { setError(requestError.message) }
    finally { setBusy(false) }
  }
  const download = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const link = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: `${result.model}-${result.backend}.json` })
    link.click(); URL.revokeObjectURL(link.href)
  }
  return (
    <section className="workspace">
      <form className="control-panel" onSubmit={run}>
        <div className="section-heading"><span className="eyebrow">03 · BENCHMARK</span><h2>Quality & throughput</h2><p>Evaluate every backend against identical text and decoding settings.</p></div>
        <Field label="Model"><input name="model" value={form.model} onChange={update} /></Field>
        <div className="field-grid"><Field label="Backend"><select name="backend" value={form.backend} onChange={update}><option value="fp32">FP32</option><option value="fp16">FP16</option><option value="bnb-int8">bitsandbytes INT8</option><option value="openvino-int8">OpenVINO INT8</option></select></Field><Field label="Device"><input name="device" value={form.device} onChange={update} /></Field></div>
        <Field label="Corpus · one document per line"><textarea name="corpus" rows="4" value={form.corpus} onChange={update} /></Field>
        <Field label="Prompts · one per line"><textarea name="prompts" rows="3" value={form.prompts} onChange={update} /></Field>
        <div className="field-grid"><Field label="Max length"><input name="max_length" type="number" value={form.max_length} onChange={update} /></Field><Field label="Stride"><input name="stride" type="number" value={form.stride} onChange={update} /></Field><Field label="New tokens"><input name="max_new_tokens" type="number" value={form.max_new_tokens} onChange={update} /></Field><Field label="Repeats"><input name="repeats" type="number" min="1" value={form.repeats} onChange={update} /></Field></div>
        <button className="primary" disabled={busy}>{busy ? 'Benchmarking…' : 'Run benchmark'}</button>{error && <p className="error">{error}</p>}
      </form>
      <div className="results-panel">
        <div className="metric-grid"><Metric label="Perplexity" value={formatNumber(result?.perplexity, 4)} accent /><Metric label="Tokens / second" value={formatNumber(result?.generation?.tokens_per_second, 2)} /><Metric label="Mean latency" value={result ? `${formatNumber(result.generation.mean_latency_seconds, 3)} s` : '—'} /><Metric label="P95 latency" value={result ? `${formatNumber(result.generation.p95_latency_seconds, 3)} s` : '—'} /></div>
        <div className="benchmark-summary"><span className="eyebrow">MEMORY REPORT</span><div><span>Parameter storage</span><strong>{formatBytes(result?.parameter_bytes)}</strong></div><div><span>Model footprint</span><strong>{formatBytes(result?.model_footprint_bytes)}</strong></div><div><span>Backend</span><strong>{result?.backend || '—'}</strong></div></div>
        <button className="secondary" disabled={!result} onClick={download}>Download JSON report</button>
      </div>
    </section>
  )
}

export default function App() {
  const [tab, setTab] = useState('tensor')
  const [online, setOnline] = useState(false)
  useEffect(() => {
    api('/api/health').then(() => setOnline(true)).catch(() => setOnline(false))
  }, [])
  return (
    <main>
      <header className="topbar"><a className="brand" href="/"><span className="brand-mark">Q</span><span>QuantLab<small>INT8 laboratory</small></span></a><div className={`status ${online ? 'online' : ''}`}><i />{online ? 'Python engine online' : 'Python engine offline'}</div></header>
      <div className="hero"><div><span className="eyebrow">LOCAL · REPRODUCIBLE · HARDWARE-AWARE</span><h1>See what quantization<br /><em>actually changes.</em></h1></div><p>Explore numerical error, compressed inference and model quality without sending prompts or weights to a hosted service.</p></div>
      <nav className="tabs" aria-label="Laboratories">{[['tensor', 'Tensor lab'], ['model', 'Model inference'], ['benchmark', 'Benchmark']].map(([id, label]) => <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>)}</nav>
      {tab === 'tensor' && <TensorLab />}{tab === 'model' && <ModelLab />}{tab === 'benchmark' && <BenchmarkLab />}
      <footer><span>QuantLab 0.2</span><span>Runs on your machine · No telemetry</span></footer>
    </main>
  )
}
