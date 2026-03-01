# WeNet Quran ASR — Datasets, Pretrained Models & Apple Silicon Guide

Addendum to the WeNet Practical Guide, covering: usable datasets with download links, pretrained model fine-tuning strategies, and M4 Max development workflows.

---

## 1. Usable Datasets for Quran ASR Training

### 1.1 Primary Quran-Specific Datasets

#### Salama1429/tarteel-ai-everyayah-Quran ⭐ TOP PICK
- **Source:** `huggingface.co/datasets/Salama1429/tarteel-ai-everyayah-Quran`
- **Size:** ~829 hours training + ~103 hours validation
- **Reciters:** 36 professional Qaris (Husary, Minshawi, Al-Afasy, Sudais, Ghamadi, etc.)
- **Format:** Audio arrays at 16kHz + fully diacritized Arabic text (tashkeel)
- **Why best:** Largest open Quran dataset. Ayah-level segmentation with full harakat. Professional reciters = clean audio. Directly loadable via HuggingFace `datasets`.
- **License:** Research use
- **Load:**
```python
from datasets import load_dataset
ds = load_dataset("Salama1429/tarteel-ai-everyayah-Quran",
                  verification_mode="no_checks")
# ds['train'] → ~829 hours, ds['validation'] → ~103 hours
# Each sample: {audio: {array, sampling_rate}, text: "بِسْمِ اللَّهِ...", reciter: "husary", duration: 6.47}
```

#### Buraaq/quran-audio-text-dataset ⭐ RUNNER-UP
- **Source:** `huggingface.co/datasets/Buraaq/quran-audio-text-dataset`
- **Size:** ~187,080 ayah-level audio files + 77,429 word-level audio files
- **Reciters:** 30 distinct Qaris
- **Format:** MP3 files with Arabic text, English translation, phonetic transliteration
- **Why valuable:** Word-level audio is unique — enables word-boundary training for real-time highlighting. Accepted at NeurIPS 2025 Muslims in ML Workshop.
- **Two sub-datasets:**
  - `Buraaq/quran-md-ayahs` — Full ayah recitations (187K samples)
  - `Buraaq/quran-md-words` — Individual word pronunciations (77K samples)
- **License:** Research use

#### MohamedRashad/Quran-Recitations
- **Source:** `huggingface.co/datasets/MohamedRashad/Quran-Recitations`
- **Size:** Multiple renowned reciters, full Quran coverage
- **Format:** Audio + fully diacritized text with tashkeel
- **Why useful:** Collected via AlQuran Cloud API, clean and authentic
- **License:** Research use — treat with utmost respect per dataset guidelines

#### RetaSy/quranic_audio_dataset (Non-Native Speakers)
- **Source:** `huggingface.co/datasets/RetaSy/quranic_audio_dataset`
- **Size:** ~7,000 recitations from 1,287 non-Arabic speakers across 11+ countries
- **Labels:** correct / incorrect / not_related_quran / not_match_aya
- **Why critical:** This is the ONLY labeled dataset with error annotations from learners. Essential for training the error detection component. Contains crowd-sourced correctness labels with 0.77 accuracy, 0.89 algorithm-expert agreement.
- **Use case:** Train a binary classifier or use as negative examples for the ASR model

#### OpenSLR Quran Datasets
- **OpenSLR-132:** Complete verse coverage, multiple reciters, diacritized
- **Source:** `openslr.org/resources`
- **Format:** WAV audio + text transcripts
- **Note:** Requires manual download and processing

#### CQDV1 (Comprehensive Quranic Dataset Version 1)
- **Source:** `ieee-dataport.org/documents/comprehensive-quranic-dataset-version-1-cqdv1`
- **Size:** ~218,000 audio files, 114 surahs, 6,236 ayahs × 35 reciters
- **Format:** MP3, Hafs from A'asim narration
- **Access:** Requires IEEE DataPort subscription
- **License:** Research use

#### Kaggle: Quran Ayat Speech-to-Text
- **Source:** `kaggle.com/datasets/bigguyubuntu/quran-ayat-speech-to-text`
- **Size:** Multiple reciters, ayah-level
- **Format:** Audio + text pairs
- **Access:** Free download with Kaggle account

### 1.2 General Arabic ASR Datasets (For Pretraining / Augmentation)

These datasets help the model learn Arabic phonetics and language patterns before Quran-specific fine-tuning.

#### Mozilla Common Voice — Arabic
- **Source:** `commonvoice.mozilla.org/en/datasets` (select Arabic)
- **Size:** ~157 hours (92 hours validated), 1,632 speakers (as of v22)
- **Format:** MP3 + text (no diacritization — undiacritized MSA)
- **Why useful:** Diverse speakers, varied recording conditions. Good for making the model robust to different microphones and environments (learner recordings will be noisy).
- **License:** CC-0 (public domain)
- **Load:**
```python
from datasets import load_dataset
cv = load_dataset("mozilla-foundation/common_voice_17_0", "ar",
                  split="train", trust_remote_code=True)
```

#### MASC (Massive Arabic Speech Corpus)
- **Source:** `ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus`
- **Size:** 1,000 hours from 700+ YouTube channels
- **Dialects:** MSA + 20+ dialects (Syrian, Egyptian, Saudi, Jordanian, etc.)
- **Format:** Audio segments + text
- **Why useful:** Largest open Arabic ASR dataset. Multi-dialect coverage improves robustness.
- **License:** Open access for research
- **Note:** Recent papers show that combining MASC + EveryAyah produces strong Arabic+Quran models

#### SADA (Saudi Audio Dataset for Arabic)
- **Source:** Referenced in ICASSP 2024, contact authors
- **Size:** 668 hours from 57 Saudi TV shows
- **Dialects:** Primarily Najdi, Hijazi, Khaleeji + some Yemeni
- **Why useful:** High-quality broadcast audio, good for clean Arabic pretraining
- **Access:** Request from authors

#### MGB-2 (Multi-Genre Broadcast)
- **Source:** `arabicspeech.org`
- **Size:** ~1,200 hours of Arabic broadcast TV (news, talk shows, drama)
- **Dialects:** 5 Arabic dialects + MSA
- **Why useful:** Largest broadcast Arabic corpus, well-established benchmark
- **Access:** Request via Arabic Speech community

#### FLEURS — Arabic
- **Source:** `huggingface.co/datasets/google/fleurs` (select "ar_eg")
- **Size:** ~12 hours
- **Format:** Audio + text
- **Why useful:** High-quality, curated, good for evaluation
- **License:** CC BY 4.0

#### Casablanca
- **Source:** `arxiv.org/abs/2410.04527`
- **Size:** 8 Arabic dialects across Africa and Asia, multi-layer annotations
- **Why useful:** Most dialectally diverse recent dataset
- **Access:** URLs + timestamps provided (copyright restricted)

### 1.3 Recommended Dataset Strategy

```
Phase 1: Pretrain on general Arabic (~1,200 hours)
├── MASC clean subset (filtered) ........... ~800h
├── Common Voice Arabic (validated) ........ ~92h
├── FLEURS Arabic .......................... ~12h
└── MGB-2 (if accessible) ................. ~300h

Phase 2: Fine-tune on Quran (~930 hours)
├── tarteel-ai-everyayah-Quran (train) .... ~829h
├── Buraaq/quran-md-ayahs ................. ~50h (estimated)
└── MohamedRashad/Quran-Recitations ....... ~50h (estimated)

Phase 3: Error Detection Training
├── RetaSy/quranic_audio_dataset .......... ~7,000 labeled samples
│   (correct vs incorrect recitations from non-native speakers)
└── Buraaq/quran-md-words ................. ~77K word samples
    (word-level alignment for precise error localization)
```

This gives you ~2,100 hours total, which is substantial for a domain-specific Conformer.

---

## 2. Pretrained Models — Can We Fine-Tune?

### 2.1 The Reality: No Arabic WeNet Pretrained Model Exists

WeNet's official pretrained models are primarily Chinese and English:
- AISHELL-1 / AISHELL-2 (Mandarin)
- LibriSpeech (English)
- WenetSpeech (Mandarin, 10K hours)
- GigaSpeech (English)

There is **no official Arabic WeNet checkpoint**. However, there are several viable paths:

### 2.2 Strategy A: Train WeNet from Scratch on Arabic (Recommended)

This is actually the best approach for Quran because:

1. **Character vocabulary mismatch.** English/Chinese pretrained models use completely different character sets. You'd have to replace the entire CTC head and attention decoder vocabulary anyway.

2. **Acoustic features transfer poorly.** Arabic phonetics (pharyngeal consonants like ع/ح, emphatic consonants like ص/ض/ط/ظ) have no equivalent in Chinese or English. The low-level acoustic patterns learned from Mandarin don't help with Arabic.

3. **~930 hours of Quran data is sufficient.** Major ASR systems achieve excellent results with this data volume. Tarteel AI trained their production system on similar data.

```bash
# Train from scratch with the config from the previous guide
torchrun --nproc_per_node=1 wenet/bin/train.py \
    --config conf/train_u2pp_conformer_quran.yaml \
    --data_type "raw" \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir exp/quran_conformer_u2pp
```

On your M4 Max, this is feasible for a medium model (see Section 3).

### 2.3 Strategy B: Use Whisper-Large-v3 as Feature Extractor + WeNet Decoder

WeNet now supports loading Whisper architecture. Whisper was trained on 680,000 hours including Arabic. You can leverage Whisper's encoder (which already knows Arabic acoustics) and attach WeNet's U2++ decoder for streaming:

```python
# WeNet supports whisper architecture since v2.x
# In your config:
# encoder: whisper_encoder
# Then fine-tune with frozen encoder first, unfreeze later

# Step 1: Export Whisper encoder weights
import whisper
import torch

model = whisper.load_model("large-v3")
encoder_state = {k: v for k, v in model.state_dict().items()
                 if k.startswith("encoder.")}
torch.save(encoder_state, "whisper_encoder_weights.pt")

# Step 2: Initialize WeNet with Whisper encoder
# Modify the WeNet config to match Whisper's encoder architecture
# Then load the encoder weights and freeze them during initial fine-tuning
```

**Pros:** Leverages 680K hours of pretraining including Arabic.
**Cons:** Complex weight mapping, Whisper encoder isn't designed for streaming (uses full attention), and you lose the U2 streaming advantage.

### 2.4 Strategy C: Two-Phase Training with General Arabic First

Train the WeNet Conformer first on general Arabic data, then fine-tune on Quran:

```bash
# Phase 1: General Arabic (MASC + CommonVoice)
# Use a SIMPLER vocabulary first (no harakat, just base Arabic chars ~40 tokens)
torchrun --nproc_per_node=1 wenet/bin/train.py \
    --config conf/train_u2pp_conformer_arabic_general.yaml \
    --train_data data/arabic_general/train/data.list \
    --cv_data data/arabic_general/dev/data.list \
    --model_dir exp/arabic_general

# Phase 2: Fine-tune on Quran (switch to harakat vocabulary ~55 tokens)
# The encoder weights transfer; replace CTC head + decoder for new vocab
python wenet/bin/train.py \
    --config conf/train_u2pp_conformer_quran.yaml \
    --train_data data/quran/train/data.list \
    --cv_data data/quran/dev/data.list \
    --model_dir exp/quran_finetuned \
    --checkpoint exp/arabic_general/avg_10.pt \
    --override_config "train_conf.optim_conf.lr=0.0002,train_conf.max_epoch=50"
```

**Important:** When switching vocabularies between Phase 1 and Phase 2, you need to handle the CTC linear layer and decoder embedding layer size change. Practically this means:

```python
import torch

# Load general Arabic checkpoint
ckpt = torch.load('exp/arabic_general/avg_10.pt', map_location='cpu')

# Remove CTC head and decoder embedding weights (they'll be re-initialized)
keys_to_remove = [k for k in ckpt.keys()
                  if 'ctc.ctc_lo' in k or 'decoder.embed' in k
                  or 'decoder.output_layer' in k]
for k in keys_to_remove:
    del ckpt[k]

# Save encoder-only checkpoint
torch.save(ckpt, 'exp/arabic_general/encoder_only.pt')

# Now train.py will initialize missing weights randomly
# and load encoder weights from the checkpoint
```

### 2.5 Strategy D: Use a Non-WeNet Arabic Pretrained Model's Knowledge

Several strong Arabic ASR models exist that you can use as teachers or references:

| Model | Framework | Arabic Data | Best WER | Notes |
|-------|-----------|------------|----------|-------|
| nvidia/stt_ar_conformer_ctc_large | NeMo | Proprietary | ~25% (avg) | Ranks #1 on Arabic leaderboard |
| openai/whisper-large-v3 | HuggingFace | Part of 680Kh | ~28% (avg) | Multilingual, not streaming |
| facebook/mms-1b-all | HuggingFace | 95h Arabic in pretraining | ~35% | 1100+ languages |
| elgeish/wav2vec2-large-xlsr-53-arabic | HuggingFace | 78h fine-tuned | ~40% | Good baseline |

**Knowledge distillation approach:** Use Whisper-large-v3 as a teacher model to generate pseudo-labels for unlabeled Arabic audio, then train WeNet on those labels. This effectively gives you unlimited training data:

```python
# Generate pseudo-labels with Whisper
import whisper
model = whisper.load_model("large-v3")

# Transcribe unlabeled Arabic audio
result = model.transcribe("unlabeled_arabic.wav", language="ar")
# result["text"] → pseudo-label for WeNet training
```

### 2.6 Recommended Path

For your situation (Quran ASR with streaming requirement):

**Go with Strategy C (Two-Phase):**

1. Train WeNet Conformer on MASC + CommonVoice Arabic (~1,000 hours) — this teaches the model Arabic phonetics and general language
2. Fine-tune on tarteel-ai-everyayah-Quran (~830 hours) with harakat vocabulary — this specializes it for Quran with full diacritization
3. Optionally use Whisper as a teacher to generate pseudo-labels for data augmentation

---

## 3. Apple Silicon (M4 Max) Development Guide

### 3.1 What Works and What Doesn't

Your M4 Max has:
- 16-core CPU (12 performance + 4 efficiency)
- 40-core GPU (Metal / MPS)
- 128GB unified memory (shared between CPU and GPU)
- ~273 GB/s memory bandwidth

Here's the compatibility matrix:

| Component | M4 Max Support | Notes |
|-----------|---------------|-------|
| **WeNet Training (PyTorch MPS)** | ✅ Works | Single-GPU only, no DDP. ~3-5x slower than A100 but functional |
| **WeNet Inference (PyTorch)** | ✅ Works | Full speed on CPU, MPS gives ~2-3x speedup |
| **Sherpa-ONNX (macOS ARM64)** | ✅ Native | Pre-built wheels on PyPI, CoreML provider available |
| **ONNX Runtime (CPU)** | ✅ Native | ARM64 wheels included in standard distribution |
| **ONNX Runtime (CoreML)** | ✅ Available | `onnxruntime-coreml` package for Neural Engine acceleration |
| **torchaudio** | ✅ Works | Feature extraction (fbank) works on MPS |
| **Multi-GPU (DDP/NCCL)** | ❌ No | MPS backend does not support distributed training |
| **fp16 / AMP** | ⚠️ Partial | MPS doesn't support fp16 training; use fp32 or bf16 (newer PyTorch) |
| **torch.compile** | ⚠️ Limited | Use sparingly on Mac, stick to eager mode |
| **float64** | ❌ No | MPS doesn't support double precision |

### 3.2 Environment Setup for M4 Max

```bash
# === Step 1: Install Homebrew (if needed) ===
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install cmake sox ffmpeg git-lfs

# === Step 2: Install Python 3.12 via pyenv ===
brew install pyenv
pyenv install 3.12.8
pyenv local 3.12.8

# === Step 3: Create virtual environment ===
python -m venv ~/wenet-env
source ~/wenet-env/bin/activate

# === Step 4: Install PyTorch with MPS support ===
# Standard pip install includes MPS on macOS ARM64
pip install torch torchvision torchaudio

# === Step 5: Verify MPS ===
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
x = torch.randn(1000, 1000, device='mps')
print(f'Tensor on MPS: {x.device}, shape: {x.shape}')
# Test Conformer-like operation
conv = torch.nn.Conv1d(80, 256, 3, padding=1).to('mps')
inp = torch.randn(4, 80, 100, device='mps')
out = conv(inp)
print(f'Conv output: {out.shape} ✓')
"

# === Step 6: Install WeNet ===
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
pip install -e .

# === Step 7: Install Sherpa-ONNX (for inference) ===
pip install sherpa-onnx  # ARM64 macOS wheel automatically selected

# === Step 8: Install additional tools ===
pip install datasets soundfile librosa wandb tensorboard
```

### 3.3 Training on M4 Max

WeNet training works on MPS but requires a few adjustments:

```bash
# Set fallback for unsupported MPS operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Train with MPS acceleration (single GPU)
python wenet/bin/train.py \
    --config conf/train_u2pp_conformer_quran.yaml \
    --data_type "raw" \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir exp/quran_conformer_u2pp \
    --num_workers 8 \
    --pin_memory
```

**Key adjustments for M4 Max:**

```yaml
# conf/train_u2pp_conformer_quran_m4max.yaml modifications:

encoder_conf:
    output_size: 256        # Use 256 not 512 (fits better in unified memory)
    num_blocks: 12          # 12 blocks (not 18) for reasonable training time

dataset_conf:
    batch_conf:
        batch_type: dynamic
        max_frames_in_batch: 16000  # Can be larger thanks to 128GB unified memory

train_conf:
    accum_grad: 8           # Compensate for single-GPU with gradient accumulation
    max_epoch: 80
    log_interval: 50
```

**Expected training performance on M4 Max:**

| Model Size | Params | Batch (frames) | Steps/sec (MPS) | Steps/sec (CPU) | Time for 100 epochs (830h data) |
|-----------|--------|----------------|-----------------|-----------------|-------------------------------|
| Small (256d, 12 blocks) | ~30M | 12000 | ~2.5 | ~0.8 | ~5-7 days |
| Medium (512d, 12 blocks) | ~80M | 8000 | ~1.2 | ~0.4 | ~12-15 days |
| Large (512d, 18 blocks) | ~120M | 6000 | ~0.6 | ~0.2 | ~25-30 days |

The 128GB unified memory is a significant advantage — you can load the entire small/medium model + large batches without OOM. On a typical 24GB GPU, you'd be limited to smaller batches.

### 3.4 Practical M4 Max Training Script

```bash
#!/bin/bash
# train_m4max.sh — Optimized for Apple M4 Max

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Let MPS use all available memory

# Use Activity Monitor to watch GPU usage
# Open a second terminal: sudo powermetrics --samplers gpu_power

python wenet/bin/train.py \
    --config conf/train_u2pp_conformer_quran_m4max.yaml \
    --data_type "raw" \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir exp/quran_m4max \
    --num_workers 10 \
    --pin_memory \
    2>&1 | tee exp/quran_m4max/train.log
```

### 3.5 Hybrid Workflow: Develop on M4 Max, Train on Cloud

The most practical approach for production models:

```
┌─────────────────────────────────────────────────────┐
│                    M4 Max (Local)                     │
│                                                       │
│  ✅ Data preparation & validation                     │
│  ✅ Quick experiments with small configs              │
│  ✅ Inference testing & debugging                     │
│  ✅ Sherpa-ONNX model testing (native ARM64)          │
│  ✅ Flutter app development & testing                 │
│  ✅ Full training on small model (256d, 12 blocks)    │
│  ⚠️ Medium model training (slower but works)          │
│  ❌ Large model training (too slow)                   │
│  ❌ Multi-GPU experiments                             │
└─────────────┬───────────────────────────────────────┘
              │ rsync / git
              ▼
┌─────────────────────────────────────────────────────┐
│               Cloud GPU (When Needed)                 │
│                                                       │
│  Options (by cost):                                   │
│  • RunPod: A100 80GB @ $1.64/hr                      │
│  • Lambda: A100 80GB @ $1.29/hr                      │
│  • Vast.ai: A100 @ $0.80-1.50/hr                    │
│  • GCP: L4 GPU @ $0.35/hr (for smaller jobs)        │
│                                                       │
│  Use for:                                             │
│  ✅ Large model training (512d, 18 blocks)            │
│  ✅ Multi-GPU training (4-8x A100)                    │
│  ✅ Hyperparameter sweeps                             │
│  ✅ Final production model training                   │
└─────────────────────────────────────────────────────┘
```

**Cost estimate for full training:**
- Small model (256d) on M4 Max: Free (5-7 days)
- Large model (512d, 18 blocks) on 4× A100: ~$200-400 (2-3 days at ~$6.50/hr)

### 3.6 Inference on M4 Max (Fast Path)

For inference and testing, M4 Max is excellent:

```bash
# === Sherpa-ONNX inference (fastest — native ARM64) ===
pip install sherpa-onnx

python -c "
import sherpa_onnx
import soundfile as sf

# Create recognizer
recognizer = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
    model='exp/quran_conformer_u2pp/sherpa/model.int8.onnx',
    tokens='exp/quran_conformer_u2pp/sherpa/tokens.txt',
    num_threads=12,  # Use M4 Max performance cores
)

# Recognize
audio, sr = sf.read('test_recitation.wav')
stream = recognizer.create_stream()
stream.accept_waveform(sr, audio)
recognizer.decode(stream)
print(stream.result.text)
"
```

```bash
# === Streaming from microphone on Mac ===
# Build sherpa-onnx from source for microphone support
git clone https://github.com/k2-fsa/sherpa-onnx.git
cd sherpa-onnx
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DSHERPA_ONNX_ENABLE_PORTAUDIO=ON \
      ..
make -j16  # Use all M4 Max cores

# Real-time microphone recognition
./bin/sherpa-onnx-microphone \
    --wenet-ctc-model=model-streaming.int8.onnx \
    --tokens=tokens.txt
```

### 3.7 CoreML Acceleration (Optional Advanced)

For maximum inference speed on Apple Silicon, convert to CoreML:

```python
# Convert ONNX → CoreML for Neural Engine acceleration
import coremltools as ct

# Note: sherpa-onnx already supports CoreML execution provider
# Just build with: cmake -DSHERPA_ONNX_ENABLE_COREML=ON ..

# Or manually convert
model = ct.converters.convert(
    "model.onnx",
    convert_to="mlpackage",
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + Neural Engine
)
model.save("QuranASR.mlpackage")
```

### 3.8 Flutter Development on M4 Max

Since you're building Flutter apps, sherpa-onnx has Flutter/Dart bindings:

```yaml
# pubspec.yaml
dependencies:
  sherpa_onnx_flutter: ^1.10.0
```

```dart
// lib/quran_asr.dart
import 'package:sherpa_onnx_flutter/sherpa_onnx_flutter.dart';

class QuranASR {
  late OnlineRecognizer recognizer;

  Future<void> init() async {
    final config = OnlineRecognizerConfig(
      model: OnlineModelConfig(
        wenetCtc: OnlineWeNetCtcModelConfig(
          model: 'assets/model-streaming.int8.onnx',
        ),
        tokens: 'assets/tokens.txt',
        numThreads: 4,
      ),
    );
    recognizer = OnlineRecognizer(config);
  }

  String processChunk(List<double> audioData, int sampleRate) {
    final stream = recognizer.createStream();
    stream.acceptWaveform(sampleRate: sampleRate, samples: audioData);
    recognizer.decode(stream);
    return recognizer.getResult(stream).text;
  }
}
```

Test the Flutter app on your M4 Max with iOS Simulator or directly on a connected iPhone, all locally without cloud dependencies.

---

## 4. Quick Start — What To Do First

### Day 1: Setup + Data

```bash
# 1. Set up environment on M4 Max
source ~/wenet-env/bin/activate
pip install datasets soundfile

# 2. Download the primary Quran dataset
python -c "
from datasets import load_dataset
ds = load_dataset('Salama1429/tarteel-ai-everyayah-Quran',
                  verification_mode='no_checks')
print(f'Train: {len(ds[\"train\"])} samples')
print(f'Val: {len(ds[\"validation\"])} samples')
print(f'Sample: {ds[\"train\"][0][\"text\"][:50]}...')
print(f'Reciters: {set(s[\"reciter\"] for s in ds[\"train\"].select(range(1000)))}')
"

# 3. Run data preparation
python prepare_quran_data.py \
    --source huggingface \
    --dataset_name "Salama1429/tarteel-ai-everyayah-Quran" \
    --output_dir data/quran
```

### Day 2: Quick Experiment

```bash
# Train a small model to verify everything works
# Modify config: output_size=256, num_blocks=6, max_epoch=5

python wenet/bin/train.py \
    --config conf/train_u2pp_conformer_quran_m4max.yaml \
    --data_type "raw" \
    --train_data data/quran/train/data.list \
    --cv_data data/quran/dev/data.list \
    --model_dir exp/quran_quick_test \
    --override_config "encoder_conf.num_blocks=6,train_conf.max_epoch=5"

# If loss decreases → your pipeline works end-to-end
```

### Week 1: Train Small Model

```bash
# Full training of small model (256d, 12 blocks)
# ~5-7 days on M4 Max
./train_m4max.sh
```

### Week 2+: Iterate

```bash
# Average checkpoints
python wenet/bin/average_model.py \
    --dst_model exp/quran_m4max/avg_10.pt \
    --src_path exp/quran_m4max \
    --num 10 --val_best

# Evaluate
python wenet/bin/recognize.py --mode attention_rescoring ...

# Export to sherpa-onnx
python sherpa-onnx/scripts/wenet/export-onnx-streaming.py ...

# Test real-time from microphone
./sherpa-onnx/build/bin/sherpa-onnx-microphone ...
```

---

## 5. Summary Table

| Question | Answer |
|----------|--------|
| **Best dataset?** | Salama1429/tarteel-ai-everyayah-Quran (829h, 36 reciters, diacritized) |
| **Error detection data?** | RetaSy/quranic_audio_dataset (7K labeled correct/incorrect from learners) |
| **Word-level data?** | Buraaq/quran-md-words (77K word pronunciations) |
| **Arabic pretrained WeNet?** | Does not exist. Train from scratch or two-phase (general Arabic → Quran) |
| **Best pretrained for fine-tune?** | Whisper-large-v3 (knows Arabic) as teacher; or train WeNet from scratch |
| **Can M4 Max train?** | Yes. Small model in ~5-7 days. Medium in ~12-15 days. MPS acceleration works. |
| **M4 Max inference?** | Excellent. Sherpa-ONNX has native ARM64 wheels. CoreML optional. |
| **Flutter support?** | Yes. sherpa-onnx has Flutter/Dart bindings for iOS/Android. |
| **Production deployment?** | Train final large model on cloud GPU, deploy via sherpa-onnx on K8s. |
