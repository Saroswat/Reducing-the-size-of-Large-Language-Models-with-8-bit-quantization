# QuantLab: reproducible LLM quantization benchmarks

QuantLab is a hardware-aware benchmark for studying how post-training quantization changes language-model memory, quality, and inference performance. It combines transparent NumPy implementations of INT8 quantization with deployable backends for Transformers, bitsandbytes, and OpenVINO.

The project deliberately separates two ideas that are often confused:

- **simulated quantization** quantizes and then dequantizes tensors to measure numerical error;
- **compressed inference** keeps weights in a low-bit representation and uses a runtime that can execute it efficiently.

The original GPT-2 notebook remains in the repository as historical learning material. The package and command-line tools provide the reproducible experiment layer.

## What this project answers

- How much memory do FP32, FP16, bitsandbytes INT8, and OpenVINO INT8 use?
- What is the perplexity change when every backend sees the same evaluation tokens?
- How do per-tensor and per-channel quantization affect reconstruction error?
- What are the latency and throughput trade-offs on the target machine?
- Which layers are most sensitive to quantization?

## Experiment matrix

| Backend | Representation | Primary use |
| --- | --- | --- |
| `fp32` | full precision | quality reference |
| `fp16` | half precision | reduced-precision GPU reference |
| `bnb-int8` | outlier-aware `LLM.int8()` | GPU compressed inference |
| `openvino-int8` | OpenVINO/NNCF INT8 | CPU/GPU deployment |

The core module additionally implements symmetric and affine signed INT8 quantization, both per-tensor and per-channel. Those implementations retain `int8` values and explicit scale/zero-point metadata; dequantization is a separate operation.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -e ".[dev]"             # core algorithms and tests
pip install -e ".[transformers]"    # FP32 and FP16 benchmarks
pip install -e ".[bitsandbytes]"    # NVIDIA LLM.int8()
pip install -e ".[openvino]"        # OpenVINO export/inference
```

Extras are separate because bitsandbytes and OpenVINO target different hardware environments.

## Quick start: inspect quantization error

```bash
quantlab tensor-demo --shape 256 768 --axis 0 --seed 7
```

Example output fields:

```json
{
  "scheme": "symmetric",
  "granularity": "per-channel(axis=0)",
  "storage_bytes": 198144,
  "compression_ratio_vs_fp32": 3.97,
  "mse": 0.000061,
  "cosine_similarity": 0.99997,
  "sqnr_db": 42.1
}
```

The exact values depend on the generated tensor and are intentionally not hard-coded in this README.

## Local browser laboratory

QuantLab includes a React dashboard backed by a local FastAPI service. It provides three interactive workspaces:

- **Tensor lab** visualizes FP32-to-INT8 reconstruction and compression metrics.
- **Model inference** loads a selected backend and generates text on the local machine.
- **Benchmark** measures shared-corpus perplexity, latency, throughput, and memory, then exports JSON.

### One-command Windows setup

From the repository root, copy and paste:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_and_run.ps1
```

The script safely creates or repairs `.venv`, installs the Transformers backend, installs the locked Node dependencies, validates runtime imports, starts both services, and opens the dashboard. Paths containing spaces are supported.

Choose OpenVINO or install every backend with:

```powershell
.\scripts\setup_and_run.ps1 -Backend openvino
.\scripts\setup_and_run.ps1 -Backend all
```

Useful options:

```powershell
.\scripts\setup_and_run.ps1 -CheckOnly       # install and run validation without starting servers
.\scripts\setup_and_run.ps1 -SkipInstall     # start using existing dependencies
.\scripts\setup_and_run.ps1 -NoBrowser       # start without opening a browser tab
```

The manual setup remains available below for users who prefer individual commands.

Install the Python service and at least one model backend from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[web,transformers]"
```

Install the browser dependencies and start both processes:

```powershell
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The React development server runs on port 5173 and the FastAPI service runs on port 8000. Press `Ctrl+C` in the terminal to stop both.

For Intel OpenVINO INT8 instead, install the OpenVINO extra before starting:

```powershell
pip install -e ".[web,openvino]"
```

The dashboard does not load remote fonts, analytics, or telemetry. Model repositories may be downloaded by their backend the first time a model is selected; subsequent inference uses the locally cached weights.

API documentation is available while the service runs at `http://127.0.0.1:8000/docs`.

## Fair model benchmark

Create a small UTF-8 corpus with one document per line, then run each backend against the same file and settings:

```bash
quantlab benchmark \
  --model gpt2 \
  --backend fp32 \
  --corpus examples/eval_corpus.txt \
  --prompts examples/prompts.txt \
  --output results/gpt2-fp32.json

quantlab benchmark --model gpt2 --backend bnb-int8 \
  --corpus examples/eval_corpus.txt --prompts examples/prompts.txt \
  --output results/gpt2-bnb-int8.json
```

Each report records:

- backend, model, device, package versions, and seed;
- parameter bytes and reported model memory footprint;
- perplexity on the exact same tokenized corpus;
- mean and p95 generation latency;
- generated tokens per second;
- prompt and generation settings.

Compare reports:

```bash
quantlab compare results/gpt2-*.json
```

## Methodology

### Quality

Perplexity is computed with a sliding window over one fixed token sequence. Labels that fall outside the newly evaluated stride are masked, preventing overlap from being counted twice. Comparing perplexity on separately generated text is invalid because the models are not evaluated on the same prediction task.

### Performance

The benchmark warms up the backend, uses deterministic greedy decoding, synchronizes CUDA when applicable, and reports end-to-end generation latency plus output-token throughput. Run measurements on the deployment hardware; cloud notebook numbers are not portable.

### Memory

`model.get_memory_footprint()` is recorded when supported. Parameter storage is calculated independently from tensor element sizes. Disk artifacts should be measured after exporting the model because simulated quantization does not reduce checkpoint size.

## Reproducible result table

Do not copy unverified numbers into documentation. Generate the reports on your hardware and summarize them in this form:

| Backend | Parameter/storage memory | Perplexity | Perplexity delta | Mean latency | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | generated | generated | baseline | generated | generated |
| FP16 | generated | generated | generated | generated | generated |
| bitsandbytes INT8 | generated | generated | generated | generated | generated |
| OpenVINO INT8 | generated | generated | generated | generated | generated |

## Advanced investigations

QuantLab is structured to support:

- layer-wise sensitivity analysis using reconstruction error or logit divergence;
- group-wise INT4/INT8 experiments;
- activation calibration and static quantization;
- outlier thresholds for `LLM.int8()`;
- AWQ/GPTQ backends;
- KL divergence between baseline and quantized next-token distributions;
- task-level evaluation through `lm-evaluation-harness`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests scripts
```

Core CI does not download model weights. Backend integration runs are intentionally hardware-specific and should be executed through the benchmark CLI.

## Repository layout

```text
src/quantlab/              quantization, metrics, evaluation, backends, CLI
web/                       Node, Vite, and React local dashboard
tests/                     deterministic model-free unit tests
examples/                  shared prompts and evaluation corpus
configs/                   reproducible experiment configuration
results/                   generated report guidance
scripts/                   experiment orchestrator
.github/workflows/ci.yml   lint and test checks
```

## Limitations

- GPT-2 is useful for a fast demonstration but is not representative of every modern architecture.
- Backend support depends on operating system and hardware.
- Perplexity captures language-modelling quality, not instruction following or factuality.
- Quantization results should include multiple models, datasets, and seeds before drawing broad conclusions.

## License

MIT. Model weights and datasets remain subject to their respective licences and terms.
