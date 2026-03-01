# Quran ASR Benchmarking Tool — Design

## Goal

Assess out-of-the-box quality of three ASR models on Quran recitation:

1. **NVIDIA FastConformer** — `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` (115M params, Arabic + tashkeel)
2. **FunASR SenseVoice-Small** — Alibaba's lightweight multilingual model (~234M params)
3. **FunASR Paraformer-Large** — Non-autoregressive, multilingual variant

Target hardware: Apple M4 Max (CPU inference for all models).

## Datasets

Combine 5–10K samples from three HuggingFace sources:

| Source | Filter | What it provides |
|--------|--------|-----------------|
| `tarteel-ai/everyayah` | Sample across reciters/surahs | 829h Quran, 36 reciters, fully diacritized transcriptions |
| `Buraaq/quran-audio-text-dataset` | Sample across reciters | 187K ayahs, 30 reciters, word-level alignment |
| `RetaSy/quranic_audio_dataset` | `final_label == "correct"` only | Learner recitations labeled correct/incorrect |

All audio resampled to 16kHz mono WAV. Unified manifest format:

```json
{"audio_path": "...", "text": "diacritized", "text_no_diacritics": "stripped", "source": "...", "surah": 1, "ayah": 1}
```

## Metrics

Full Quran-specific suite:

- **WER** — Word Error Rate on full diacritized text
- **CER** — Character Error Rate on full diacritized text
- **D-WER** — WER after stripping tashkeel (isolates consonant recognition)
- **D-CER** — CER after stripping tashkeel
- **Harakat Accuracy** — Position-by-position diacritical mark comparison on aligned text
- **Per-surah breakdown** — All metrics grouped by surah number

## Architecture

Single unified Python CLI tool:

```
python -m benchmark prepare    → download & normalize datasets
python -m benchmark evaluate   → run inference per model
python -m benchmark report     → compute metrics & generate comparison
```

### Data Flow

```
HuggingFace datasets → prepare → unified JSONL manifest + audio/
                                         ↓
                              evaluate (per model) → predictions JSONL
                                         ↓
                              report → metrics JSON + markdown table
```

### Directory Structure

```
benchmark/
├── __main__.py          # CLI entry point (argparse subcommands)
├── prepare.py           # Dataset download & normalization
├── evaluate.py          # Model inference runners
├── metrics.py           # WER, CER, D-WER, D-CER, Harakat Accuracy
├── report.py            # Generate comparison tables
├── arabic_utils.py      # Tashkeel stripping, text normalization
└── requirements.txt
data/                    # Downloaded/prepared data (gitignored)
results/                 # Evaluation outputs (gitignored)
```

## Model Runners

- **NeMo FastConformer**: Load `.nemo` via `nemo.collections.asr`, call `transcribe()`
- **FunASR SenseVoice**: Load via `funasr.AutoModel(model="iic/SenseVoiceSmall")`, call `generate()`
- **FunASR Paraformer**: Load via `funasr.AutoModel(model="paraformer-large")`, call `generate()`

All run on CPU (MPS not supported by NeMo; FunASR MPS support is limited).
Predictions saved to JSONL alongside ground truth. Results cached to avoid re-inference when iterating on metrics.

## M4 Max Notes

- CPU inference only for all three models
- Batch processing with progress bars (tqdm)
- Audio files cached locally after first download
- Expect ~2-6 hours total eval time depending on final sample count
