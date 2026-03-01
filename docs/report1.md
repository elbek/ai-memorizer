# Report 1: Initial ASR Model Evaluation on Quran Recitation

**Date:** 2026-03-01
**Hardware:** Apple M4 Max, CPU inference
**Sample size:** 100 (33 tarteel + 33 buraaq + 34 tusers)

## Goal

Evaluate off-the-shelf ASR models on Quran recitation audio to establish a baseline before fine-tuning. The key question: how well do pretrained models handle real user recordings vs professional recitations?

## Models Attempted

### 1. NVIDIA FastConformer (NeMo)

- **Model:** `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
- **Params:** 115M, hybrid CTC/Transducer architecture
- **Output:** Diacritized Arabic (tashkeel included, but inconsistently)
- **Status:** Evaluated successfully

### 2. FunASR MLT-Nano

- **Model:** `FunAudioLLM/Fun-ASR-MLT-Nano-2512`
- **Status:** Failed — `FunASRNano` model class is not registered in funasr 1.3.1 (latest release). The model was published on HuggingFace after the last funasr package update. Cannot evaluate until funasr releases a new version.

### 3. Tarteel Whisper Base

- **Model:** `tarteel-ai/whisper-base-ar-quran`
- **Params:** 74M, fine-tuned on Quran recitations
- **Status:** Code ready, not yet evaluated

## Datasets

| Source | Type | Samples | Description |
|--------|------|---------|-------------|
| **tarteel** | Professional reciters (HuggingFace) | 33 | From tarteel-ai/everyayah, 36 professional reciters, clean studio audio, fully diacritized |
| **buraaq** | Professional reciters (HuggingFace) | 33 | From Buraaq/quran-md-ayahs, 30 reciters, verse-level alignment |
| **tusers** | Real user recordings (local) | 34 | 18K+ recordings from 18K+ unique users, varied quality, mobile microphones |

Tusers is the most important dataset because it represents real-world usage — actual users reciting Quran on their phones.

## Results: NeMo FastConformer on 100 Samples

### Overall

| Metric | Value |
|--------|-------|
| WER | 53.14% |
| CER | 18.06% |
| D-WER | 11.46% |
| D-CER | 6.29% |
| Harakat Accuracy | 59.44% |

### Per-Source Breakdown

| Source | WER | CER | D-WER | D-CER | Harakat Acc. |
|--------|-----|-----|-------|-------|-------------|
| **tarteel** | 26.15% | 6.10% | **0.57%** | 0.11% | **90.03%** |
| **buraaq** | 55.66% | 19.06% | **5.88%** | 4.79% | 53.04% |
| **tusers** | 82.99% | 31.37% | **30.56%** | 15.33% | 29.48% |

## Analysis

### Tarteel (professional reciters): Excellent

NeMo achieves near-perfect consonant recognition on professional recitations: **0.57% D-WER** and **90% Harakat Accuracy**. The model was likely trained on similar high-quality recitation data. The 26% raw WER comes from minor tashkeel differences — the right words are being recognized.

### Buraaq (professional reciters): Good consonants, poor diacritics

D-WER of **5.88%** shows strong word recognition, but raw WER of 55.66% and Harakat Accuracy of 53% reveal a diacritization problem. NeMo **strips diacritics entirely** on many buraaq samples — outputting bare consonantal text where the reference has full tashkeel. This is a formatting inconsistency in NeMo, not a recognition error.

### Tusers (real users): Poor

**30.56% D-WER** means NeMo misrecognizes nearly 1 in 3 words from user recordings. Harakat Accuracy of **29.48%** is barely above random. Two issues compound:

1. **Audio quality:** User recordings have background noise, varied microphones, non-professional pronunciation, different speaking distances. NeMo was trained on clean professional audio.

2. **Inconsistent diacritization:** NeMo outputs bare text (no tashkeel) on 41% of tusers samples. When the model doesn't output diacritics at all, every harakat comparison fails.

3. **Truncation:** Some hypotheses are shorter than references — the model cuts off or doesn't recognize portions of the recitation.

## Issues Discovered and Fixed

### Quran Unicode normalization

Tusers transcripts use **Uthmani Quran script** with special Unicode characters that no ASR model outputs:

| Character | Codepoint | Name | NeMo equivalent |
|-----------|-----------|------|-----------------|
| ٱ | U+0671 | ALEF WASLA | ا (plain alef) |
| ٰ | U+0670 | SUPERSCRIPT ALEF | (removed) |
| ۥ | U+06E5 | SMALL WAW | (nothing) |
| ۦ | U+06E6 | SMALL YEH | (nothing) |
| ۟ | U+06DF | SMALL HIGH ROUNDED ZERO | (nothing) |
| ۢ | U+06E2 | SMALL HIGH MEEM | (nothing) |

These are Quranic typographic marks with no pronunciation effect. Before fixing this, tusers D-WER was 32.99% — after normalizing these characters, it dropped to **30.56%**. The fix was added to `normalize_quran_text()` in `arabic_utils.py` and applied to all metric computations.

### FunASR model registration

`FunAudioLLM/Fun-ASR-MLT-Nano-2512` requires a `FunASRNano` model class that doesn't exist in funasr 1.3.1. The model repo on HuggingFace was published after the last funasr release. No workaround available.

### NeMo + MPS

Attempted running NeMo on Apple Metal (MPS) for faster inference. Result: **slower** (5.3s for 2 files on MPS vs ~2s/batch on CPU). The RNNT decoder uses `warprnnt_numba` — a CPU-only JIT-compiled kernel. Reverted to CPU.

### Retasy dataset removed

The RetaSy/quranic_audio_dataset was initially included but **99% of samples had corrupt audio** on Python 3.14 (only 6 out of 409 survived). Removed from the pipeline entirely.

## Key Takeaways

1. **NeMo is excellent on professional recitations** (D-WER < 1%) but **struggles on real user recordings** (D-WER 30%). This is the gap we need to close.

2. **The diacritization problem is twofold:** NeMo sometimes outputs fully diacritized text, sometimes bare text. There's no consistency. This makes both WER and Harakat Accuracy unreliable measures of the model's actual diacritization capability.

3. **Quran-specific Unicode normalization is essential.** Without it, even perfect recognition gets penalized for character-level mismatches (ٱ vs ا, etc.).

4. **Fine-tuning is the path forward.** The tusers val/test splits (2,754 / 2,756 samples) are ready for training. The eval split (12,909 samples) provides the benchmark. Fine-tuning NeMo or Whisper on diverse user recordings should dramatically improve the 30% D-WER.

## Next Steps

- [ ] Run Whisper evaluation on same 100 samples to compare against NeMo
- [ ] Run NeMo on full 20,907-sample manifest (needs ~2-3 hours on M4 Max)
- [ ] Fine-tune a model on tusers training data using val split for validation
- [ ] Evaluate on NVIDIA 3060 GPU for faster iteration
- [ ] Investigate NeMo's inconsistent diacritization behavior (CTC vs Transducer decoder mode)
