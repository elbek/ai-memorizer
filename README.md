# AI Memorizer

Quran ASR benchmarking tool for evaluating speech recognition models on Quran recitation.

## Models Evaluated

| Model | Params | Arabic Diacritics | Source |
|-------|--------|-------------------|--------|
| NVIDIA FastConformer `stt_ar_fastconformer_hybrid_large_pcd_v1.0` | 115M | Yes (tashkeel output) | NeMo |
| FunASR MLT-Nano `Fun-ASR-MLT-Nano-2512` | 800M | No | FunASR |

## Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| [tarteel-ai/everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah) | ~19K test | Multi-reciter Quran, fully diacritized |
| [Buraaq/quran-md-ayahs](https://huggingface.co/datasets/Buraaq/quran-md-ayahs) | 187K | 30 reciters, verse-level alignment |
| [RetaSy/quranic_audio_dataset](https://huggingface.co/datasets/RetaSy/quranic_audio_dataset) | ~6.8K | Learner recitations (correct-only filtered) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmark/requirements.txt
```

## CLI Commands

### `prepare` — Download and prepare datasets

```bash
python -m benchmark prepare --max-samples 10000 --output-dir data
```

Downloads audio from HuggingFace, converts to 16kHz mono WAV, and writes a unified manifest at `data/manifest.jsonl`. Samples are split 40% tarteel / 40% buraaq / 20% retasy.

| Flag | Default | Description |
|------|---------|-------------|
| `--max-samples` | 10000 | Total samples across all sources |
| `--output-dir` | `data` | Where to save audio and manifest |

### `evaluate` — Run model inference

```bash
python -m benchmark evaluate --model nemo --batch-size 4
python -m benchmark evaluate --model funasr --batch-size 4
```

Runs the specified model on the prepared dataset and saves predictions to `results/<model>_predictions.jsonl`.

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | `nemo` or `funasr` |
| `--data-dir` | `data` | Directory with manifest.jsonl |
| `--output-dir` | `results` | Where to save predictions |
| `--batch-size` | 8 | Inference batch size (lower if OOM) |

### `report` — Generate comparison report

```bash
python -m benchmark report
```

Loads all `*_predictions.jsonl` from the results directory, computes metrics, and generates `results/report.md` and `results/metrics.json`.

| Flag | Default | Description |
|------|---------|-------------|
| `--results-dir` | `results` | Directory with prediction files |
| `--data-dir` | `data` | Directory with ground truth |

### `run-all` — Full pipeline

```bash
python -m benchmark run-all --max-samples 10000 --batch-size 4
python -m benchmark run-all --models nemo          # single model only
python -m benchmark run-all --models nemo funasr   # both (default)
```

Runs prepare → evaluate (each model) → report in sequence.

| Flag | Default | Description |
|------|---------|-------------|
| `--max-samples` | 10000 | Samples for preparation |
| `--batch-size` | 8 | Inference batch size |
| `--models` | `nemo funasr` | Which models to evaluate |
| `--data-dir` | `data` | Data directory |
| `--results-dir` | `results` | Results directory |

## Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate on full diacritized text |
| **CER** | Character Error Rate on full diacritized text |
| **D-WER** | WER after stripping tashkeel (consonant recognition only) |
| **D-CER** | CER after stripping tashkeel |
| **Harakat Accuracy** | Position-by-position diacritical mark accuracy |

## Running Tests

```bash
python -m pytest tests/ -v
```
