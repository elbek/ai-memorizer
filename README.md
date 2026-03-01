# Quran ASR Benchmark

Benchmarking tool for evaluating speech recognition models on Quran recitation. Measures word/character error rates and diacritization (tashkeel) accuracy across professional and user-recorded audio.

## Why This Exists

Existing Arabic ASR models are trained on modern standard Arabic — news, conversations, etc. Quran recitation is fundamentally different: it requires preserving **full diacritization** (fatha, kasra, damma, sukun, shadda, tanween) because a single wrong harakat changes the meaning of a verse. This benchmark measures how well off-the-shelf ASR models handle Quran audio, especially from real users (not just professional reciters).

## Models

| Model | Params | Diacritics Output | Status |
|-------|--------|-------------------|--------|
| [NVIDIA FastConformer](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) | 115M | Yes (tashkeel) | Evaluated |
| [Tarteel Whisper Base](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) | 74M | Yes | Ready (not yet run) |
| [FunASR MLT-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | ~800M | No | Blocked (model class not registered in funasr 1.3.1) |

## Datasets

| Dataset | Type | Samples Used | Description |
|---------|------|-------------|-------------|
| [tarteel-ai/everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah) | Professional | 4,000 | 36 reciters, fully diacritized transcriptions, ~19K total in test split |
| [Buraaq/quran-md-ayahs](https://huggingface.co/datasets/Buraaq/quran-md-ayahs) | Professional | 3,998 | 30 reciters, ayah+word-level audio, 187K total |
| tusers (local) | User recordings | 12,909 eval / 2,754 val / 2,756 test | 18,419 recordings from 18,131 users, split by user ID to prevent leakage |
| [Quran Speech Dataset](https://archive.org/details/quran-speech-dataset) | Mixed | Not yet used | 24GB: 7 imams (43K WAVs) + 25K Tarteel.io user recordings, CC-BY-4.0 |

**Tusers split strategy:** 70% eval / 15% val / 15% test, split by user ID (not by sample) so the same user never appears in multiple splits.

**Total manifest:** 20,907 samples (4,000 tarteel + 3,998 buraaq + 12,909 tusers eval).

## Metrics

| Metric | What It Measures |
|--------|-----------------|
| **WER** | Word Error Rate on full diacritized text |
| **CER** | Character Error Rate on full diacritized text |
| **D-WER** | WER after stripping tashkeel — isolates consonant/word recognition |
| **D-CER** | CER after stripping tashkeel |
| **Harakat Accuracy** | Position-by-position diacritical mark comparison |

**D-WER** is the most important metric for "did the model hear the right words." **Harakat Accuracy** is the most important for "did it get the tashkeel right" — critical for detecting recitation errors.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmark/requirements.txt
```

For Whisper evaluation, also install:
```bash
pip install transformers torch
```

## CLI Usage

### Prepare datasets

```bash
# HuggingFace datasets only
python -m benchmark prepare --max-samples 10000

# With local tusers dataset
python -m benchmark prepare --max-samples 20000 --tusers-dir data/tusers_all_filtered
```

Downloads audio, converts to 16kHz mono WAV, writes unified manifest to `data/manifest.jsonl`. With `--tusers-dir`, also writes `manifest_val.jsonl` and `manifest_test.jsonl`.

### Evaluate a model

```bash
python -m benchmark evaluate --model nemo --batch-size 4
python -m benchmark evaluate --model whisper --batch-size 8
python -m benchmark evaluate --model nemo --manifest data/manifest_100.jsonl  # custom manifest
```

### Generate report

```bash
python -m benchmark report
```

Loads all `*_predictions.jsonl` from results/, computes metrics, generates `results/report.md` and `results/metrics.json`.

### Run full pipeline

```bash
python -m benchmark run-all --max-samples 10000 --batch-size 4 --models nemo whisper
```

## Project Structure

```
benchmark/
  __main__.py        CLI entry point (argparse subcommands)
  prepare.py         Dataset download, normalization, manifest generation
  evaluate.py        Model inference runners (NeMo, Whisper, FunASR)
  metrics.py         WER, CER, D-WER, D-CER, Harakat Accuracy
  report.py          Comparison table and metrics JSON generation
  arabic_utils.py    Tashkeel stripping, Quran text normalization
  requirements.txt   Python dependencies
tests/
  test_arabic_utils.py
  test_metrics.py
  test_report.py
data/                Downloaded/prepared data (gitignored)
results/             Evaluation outputs (gitignored)
docs/                Research notes and design docs
```

## Running Tests

```bash
python -m pytest tests/ -v
# 66 tests covering arabic_utils, metrics engine, and report generation
```

## Hardware

Developed and tested on Apple M4 Max (CPU inference). NeMo uses CPU-only RNNT decoder (numba JIT). Whisper can use MPS (Metal) acceleration on Apple Silicon. NVIDIA GPU with CUDA would give 10-30x speedup.
