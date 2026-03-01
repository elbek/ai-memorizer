# FunASR + NVIDIA NeMo FastConformer — Quran Recitation ASR Assessment

A deep assessment of Alibaba's FunASR toolkit and NVIDIA's NeMo FastConformer for Quran recitation recognition, covering architecture, streaming, fine-tuning feasibility, and comparison with WeNet.

---

## 1. What Is FunASR?

FunASR is Alibaba DAMO Academy's (now Tongyi Lab's) open-source speech recognition toolkit. It sits at the intersection of research and production — built from academic papers but backed by Alibaba's industrial-scale deployment experience. 14K GitHub stars, MIT licensed, 151 contributors.

The ecosystem has three tiers:

**FunASR Toolkit** (github.com/modelscope/FunASR) — The framework itself. Training, fine-tuning, inference, deployment for multiple model architectures. This is what you'd use.

**SenseVoice** (2024) — A speech foundation model within FunASR. Non-autoregressive, 50+ language support, includes ASR + emotion recognition + audio event detection. Trained on 400K+ hours.

**Fun-ASR-Nano** (Dec 2025) — The latest flagship. 800M params, trained on tens of millions of hours, supports 31 languages including Arabic. Real-time streaming.

---

## 2. Architecture Deep Dive

### 2.1 Paraformer — The Core Innovation

Paraformer is FunASR's signature model architecture, published at Interspeech 2022. It's a **non-autoregressive** (NAR) transformer, meaning it generates all output tokens in a single forward pass — unlike autoregressive models (Whisper, WeNet's attention decoder) that generate tokens one by one.

**Key papers:**

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Paraformer (arxiv:2206.08317) | 2022 | CIF predictor + GLM sampler for single-step NAR ASR. 10x speedup over AR with <2% accuracy loss |
| FunASR Toolkit (Interspeech 2023) | 2023 | Open-source toolkit paper, industrial deployment |
| Paraformer-v2 (arxiv:2409.17746) | 2024 | Replaced CIF with CTC for token extraction. Better multilingual + noise robustness |
| FunAudioLLM (arxiv:2407.04051) | 2024 | SenseVoice + CosyVoice foundation model paper |
| Fun-ASR Technical Report (arxiv:2509.12508) | 2025 | Fun-ASR-Nano, tens of millions of hours, 31 languages |

**How Paraformer works:**

```
Audio → Encoder (SAN-M/Conformer) → CIF/CTC Predictor → Token Embeddings
                                                              ↓
                                          GLM Sampler → Semantic Embeddings
                                                              ↓
                                          Bidirectional NAR Decoder → All tokens at once
```

The critical component is the **Continuous Integrate-and-Fire (CIF)** predictor. It accumulates encoder hidden states and "fires" when accumulated weight crosses a threshold, producing exactly one token embedding per output token. This replaces the attention-based alignment of AR models with a monotonic, soft alignment.

**Paraformer-v2** replaced CIF with CTC-based token extraction, which solved two problems:
- CIF struggled with BPE-tokenized languages (English, Arabic) where token boundaries don't align cleanly with acoustic boundaries
- CIF was noise-sensitive, degrading in real-world environments

This is directly relevant to Arabic/Quran — Arabic with harakat uses character tokenization where boundaries are complex.

### 2.2 SenseVoice Architecture

SenseVoice uses a modified architecture called **SAN-M** (Self-Attention with Memory) encoder — essentially a deep FSMN (Feedforward Sequential Memory Network) combined with self-attention. It adds task-specific embeddings as prompts (language ID, emotion, event detection) to a single unified model.

Key specs:
- SenseVoice-Small: 234M params, encoder-only NAR, 5 languages (zh, en, yue, ja, ko)
- SenseVoice-Large: encoder-decoder, 50+ languages (includes Arabic)
- 15x faster than Whisper-Large inference (70ms for 10 seconds of audio)

### 2.3 Fun-ASR-Nano (Dec 2025)

The newest model. 800M params, trained on tens of millions of hours. Supports 31 languages — **Arabic is explicitly listed**. Supports streaming/real-time transcription.

---

## 3. Streaming Support

FunASR has **excellent streaming support** — this is one of its strongest advantages over other toolkits.

### 3.1 Streaming Architecture

FunASR uses a **chunk-based streaming** approach with their Paraformer-streaming model:

```python
from funasr import AutoModel

chunk_size = [0, 10, 5]  # [history, chunk, lookahead] in 60ms frames
# [0, 10, 5] → 600ms chunks, 300ms lookahead
# [0, 8, 4]  → 480ms chunks, 240ms lookahead

encoder_chunk_look_back = 4  # chunks for encoder self-attention
decoder_chunk_look_back = 1  # encoder chunks for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming")

# Process audio chunk by chunk
cache = {}
for i, chunk in enumerate(audio_chunks):
    is_final = (i == total_chunks - 1)
    result = model.generate(
        input=chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back
    )
```

### 3.2 Two-Pass (2pass) Mode

FunASR's production streaming service uses a **2pass** approach:
1. **First pass (streaming):** Low-latency partial results using paraformer-zh-streaming
2. **Second pass (offline):** Full-quality re-scoring using paraformer-zh (non-streaming) once the segment ends

This gives you real-time display with final accuracy matching offline models.

### 3.3 Production Streaming Server

FunASR includes a full WebSocket-based streaming server with Docker deployment:

```bash
# Pull and run the streaming server
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
docker run -p 10095:10095 -it funasr:funasr-runtime-sdk-online-cpu-0.1.12

# Includes:
# - VAD (fsmn-vad) for voice activity detection
# - Streaming ASR (paraformer-zh-streaming)
# - Offline ASR (paraformer-zh for 2nd pass)
# - Punctuation restoration (ct-punc)
# - WebSocket server with multi-client support
```

Client libraries available in Python, C++, Java, C#, HTML/JavaScript.

### 3.4 Streaming Comparison: FunASR vs WeNet

| Feature | FunASR | WeNet |
|---------|--------|-------|
| Streaming architecture | Chunk-based Paraformer (NAR) | U2/U2++ (dynamic chunk CTC+Attention) |
| Latency | 600ms chunks typical | Configurable chunk size (e.g., 640ms) |
| 2-pass refinement | Built-in 2pass mode | Attention rescoring on CTC |
| Inference speed | 10x faster (NAR single-step) | Standard AR/CTC speed |
| Production server | Docker + WebSocket included | Sherpa-ONNX WebSocket server |
| VAD integration | Built-in fsmn-vad pipeline | External |
| Punctuation | Built-in ct-punc model | External |

FunASR's NAR approach is fundamentally faster at inference — the decoder runs in a single step rather than auto-regressively. For real-time Quran recitation feedback, this is a meaningful advantage.

---

## 4. Fine-Tuning for Quran ASR

### 4.1 What Can Be Fine-Tuned

FunASR explicitly supports fine-tuning. The SenseVoice repo states it provides "convenient finetuning scripts and strategies."

**Fine-tunable models in FunASR:**

| Model | Fine-Tune Support | Arabic Support | Streaming |
|-------|-------------------|---------------|-----------|
| paraformer-zh | ✅ Full | ❌ Chinese only | Non-streaming |
| paraformer-zh-streaming | ✅ Full | ❌ Chinese only | ✅ Streaming |
| paraformer-en | ✅ Full | ❌ English only | Non-streaming |
| SenseVoice-Small | ✅ Documented | ⚠️ 5 langs, no Arabic | Non-streaming |
| SenseVoice-Large | ✅ Expected | ✅ 50+ langs incl. Arabic | Non-streaming |
| Fun-ASR-Nano | ⚠️ Not yet documented | ✅ 31 langs incl. Arabic | ✅ Streaming |
| Whisper-large-v3 (via FunASR) | ✅ Via FunASR wrapper | ✅ Multilingual | Non-streaming |

### 4.2 Fine-Tuning Approach

FunASR uses a standard PyTorch training loop with DeepSpeed support for distributed training:

```bash
# Fine-tune SenseVoice on custom data
# 1. Prepare data in jsonl format:
# {"key": "sample_001", "source": "/path/to/audio.wav", "target": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"}

# 2. Modify finetune.sh
# Set train_tool to absolute path of funasr/bin/train_ds.py
# Set init_param to pretrained model path
# Set train_data / val_data to your jsonl files

# 3. Run
bash finetune.sh
```

The fine-tuning script (`train_ds.py`) supports:
- DeepSpeed for multi-GPU training
- Gradient accumulation
- Learning rate scheduling
- Checkpoint resuming
- Custom vocabulary (critical for Arabic harakat)

### 4.3 The Arabic Fine-Tuning Challenge

Here's the honest assessment of fine-tuning FunASR for Quran Arabic:

**Option A: Fine-tune SenseVoice-Large (Most Promising)**

SenseVoice-Large already knows Arabic (50+ languages, trained on 400K+ hours). Fine-tuning it on Quran data would be transfer learning from a model that already understands Arabic phonetics.

Challenges:
- SenseVoice-Large is not open-sourced (only Small is). You'd need to check ModelScope for availability.
- SenseVoice-Small only supports 5 languages (zh, en, yue, ja, ko) — no Arabic.
- No streaming mode for SenseVoice (encoder-only architecture).

**Option B: Fine-tune Fun-ASR-Nano**

Fun-ASR-Nano explicitly lists Arabic among its 31 languages and supports streaming. This is theoretically the ideal base model. However:
- Released December 2025, very new
- Fine-tuning documentation may be limited
- 800M params — heavier to fine-tune on M4 Max
- Need to verify if fine-tuning scripts are available yet

**Option C: Train Paraformer Architecture from Scratch on Arabic**

FunASR supports training Paraformer/Conformer from scratch. You could:
1. Use the Paraformer-v2 architecture (CTC-based, better for non-Chinese)
2. Define your Arabic+harakat character vocabulary
3. Train on the ~2,100 hours of combined data

This avoids the vocabulary mismatch problem entirely but requires more data and compute.

**Option D: Use Whisper-large-v3 through FunASR**

FunASR wraps Whisper and provides its training/fine-tuning infrastructure. You could fine-tune Whisper (which already knows Arabic) through FunASR's convenient scripts, then deploy via FunASR's runtime.

### 4.4 Recommended Fine-Tuning Path

```
Priority 1: Fun-ASR-Nano (if fine-tuning available)
├── Already knows Arabic
├── Streaming support
├── Massive pretraining (tens of millions of hours)
├── Use as base, fine-tune on tarteel-ai-everyayah-Quran
└── Challenge: New model, fine-tuning scripts may be limited

Priority 2: Train Paraformer-v2 from scratch
├── CTC-based token extraction (better for Arabic)
├── Non-autoregressive (fast inference)
├── Full control over vocabulary (harakat as separate tokens)
├── ~2,100 hours is sufficient for Paraformer
└── Challenge: No Arabic pretraining to leverage

Priority 3: Whisper-large-v3 via FunASR
├── Strong Arabic baseline from 680K hours
├── FunASR provides fine-tuning infrastructure
├── Well-documented fine-tuning process
└── Challenge: Not streaming, autoregressive (slower)
```

---

## 5. FunASR vs WeNet — Head-to-Head for Quran ASR

| Criterion | FunASR | WeNet | Winner |
|-----------|--------|-------|--------|
| **Arabic pretrained model** | Fun-ASR-Nano (31 langs incl. Arabic) | None | FunASR |
| **Streaming** | ✅ Chunk Paraformer + 2pass | ✅ U2++ dynamic chunk | Tie |
| **Inference speed** | NAR single-step (10x faster) | AR attention decoder | FunASR |
| **Production deployment** | Docker + WebSocket + ONNX | Sherpa-ONNX + Go server | Tie |
| **Fine-tuning docs** | Good, scripts provided | Good, YAML configs | Tie |
| **Community/ecosystem** | 14K stars, Alibaba-backed | 4K stars, community-driven | FunASR |
| **Sherpa-ONNX export** | ✅ SenseVoice supported | ✅ Native support | Tie |
| **Apple Silicon (MPS)** | ⚠️ Possible but untested | ⚠️ Works with MPS fallback | Tie |
| **Flutter/mobile** | Via sherpa-onnx (SenseVoice) | Via sherpa-onnx (native) | Tie |
| **VAD built-in** | ✅ fsmn-vad pipeline | ❌ External | FunASR |
| **Punctuation built-in** | ✅ ct-punc model | ❌ External | FunASR |
| **Custom vocabulary control** | Full (train from scratch) | Full (char-level config) | Tie |
| **Harakat tokenization** | Need custom vocab | Native char-level support | WeNet (slight) |
| **Vendor independence** | Alibaba/ModelScope ecosystem | Fully independent | WeNet |
| **License** | MIT (code), custom (models) | Apache 2.0 | WeNet |
| **K8s/Go stack fit** | Good (Docker/WebSocket) | Excellent (Go native bindings) | WeNet |

### The Verdict

**FunASR is a strong contender, especially because of Fun-ASR-Nano's Arabic support.** The fact that it already has a model trained on Arabic with streaming capability is a significant advantage over WeNet's "train from scratch" approach.

However, there are important caveats:

1. **Model license**: FunASR code is MIT, but pretrained models have a separate MODEL_LICENSE that may restrict commercial use. Check the terms for Fun-ASR-Nano specifically.

2. **Ecosystem lock-in**: FunASR is tightly coupled with ModelScope (Alibaba's model hub). While HuggingFace is also supported, the primary ecosystem is Chinese. Documentation, issues, and community discussions are predominantly in Chinese.

3. **Fine-tuning maturity for Arabic**: The fine-tuning scripts are proven for Chinese and English. Arabic fine-tuning (especially with harakat) is uncharted territory in FunASR.

4. **Your Go/K8s stack**: WeNet's Sherpa-ONNX has native Go bindings that fit your microservices architecture naturally. FunASR's deployment is more Python/Docker-centric.

---

## 6. Practical Strategy — Using Both

The smartest approach may be to use both frameworks at different stages:

```
┌─────────────────────────────────────────────────────────┐
│              PHASE 1: Quick Baseline (Week 1)            │
│                                                           │
│  Use Fun-ASR-Nano out-of-the-box for Arabic Quran        │
│  → Zero training, just test accuracy on your ayah data   │
│  → Gives you a baseline WER/CER to beat                  │
│                                                           │
│  from funasr import AutoModel                             │
│  model = AutoModel(                                       │
│      model="FunAudioLLM/Fun-ASR-Nano-2512",              │
│      vad_model="fsmn-vad",                                │
│      device="mps"  # or "cpu" for M4 Max                 │
│  )                                                        │
│  res = model.generate(input="quran_recitation.wav")       │
│  # Check: Does it output Arabic? With harakat?            │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│         PHASE 2: Evaluate Fine-Tuning (Weeks 2-3)        │
│                                                           │
│  IF Fun-ASR-Nano produces Arabic but no harakat:          │
│  → Fine-tune on tarteel-ai-everyayah-Quran (829h)        │
│  → Use FunASR's finetune.sh with DeepSpeed               │
│                                                           │
│  IF Fun-ASR-Nano struggles or fine-tuning is blocked:     │
│  → Fall back to WeNet Conformer U2++ trained from scratch │
│  → Use the full pipeline from the practical guide         │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│         PHASE 3: Production Deployment                    │
│                                                           │
│  Export best model → Sherpa-ONNX (works for both)         │
│  Deploy via Go WebSocket server on K8s                    │
│  Flutter app uses sherpa-onnx Dart bindings               │
│                                                           │
│  Both FunASR and WeNet models export to Sherpa-ONNX,      │
│  so the deployment path is identical regardless of        │
│  which framework produced the model.                      │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Dataset Compatibility

FunASR uses a simple **jsonl** format for training data:

```json
{"key": "uttid_001", "source": "/data/audio/001.wav", "target": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"}
{"key": "uttid_002", "source": "/data/audio/002.wav", "target": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"}
```

Converting the datasets from the previous research:

```python
from datasets import load_dataset
import soundfile as sf
import json
import os

# Load tarteel-ai-everyayah-Quran
ds = load_dataset("Salama1429/tarteel-ai-everyayah-Quran",
                  verification_mode="no_checks")

output_dir = "data/quran_funasr"
audio_dir = f"{output_dir}/audio"
os.makedirs(audio_dir, exist_ok=True)

with open(f"{output_dir}/train.jsonl", "w", encoding="utf-8") as f:
    for i, sample in enumerate(ds["train"]):
        audio_path = f"{audio_dir}/train_{i:06d}.wav"
        sf.write(audio_path, sample["audio"]["array"],
                 sample["audio"]["sampling_rate"])
        entry = {
            "key": f"train_{i:06d}",
            "source": os.path.abspath(audio_path),
            "target": sample["text"]  # Already has full tashkeel
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

All the datasets from the previous document (tarteel-ai, Buraaq/quran-md, MohamedRashad, RetaSy, CommonVoice, MASC) can be converted to this format with minimal effort.

---

## 8. Research Papers to Read

### Core FunASR/Paraformer Papers

1. **Paraformer** (2022) — arxiv.org/abs/2206.08317
   - The foundational NAR ASR paper. CIF predictor, GLM sampler, MWER training.
   - Key insight: Non-autoregressive can match autoregressive with 10x speedup.

2. **Paraformer-v2** (2024) — arxiv.org/abs/2409.17746
   - CTC replaces CIF for token extraction. Better multilingual and noise robustness.
   - Directly relevant: Proves Paraformer works beyond Chinese.

3. **FunASR Toolkit** (Interspeech 2023) — cited as `gao2023funasr`
   - The toolkit paper. Architecture overview, training pipeline, deployment.

4. **FunAudioLLM** (2024) — arxiv.org/abs/2407.04051
   - SenseVoice + CosyVoice paper. Speech understanding foundation models.

5. **Fun-ASR Technical Report** (2025) — arxiv.org/abs/2509.12508
   - Fun-ASR-Nano. Tens of millions of hours, 31 languages, streaming.

### Related Arabic ASR Papers

6. **Open Universal Arabic ASR Leaderboard** (2024) — arxiv.org/abs/2412.13788
   - Benchmarks all major ASR models on Arabic. NeMo Conformer-CTC ranks #1.

7. **Quran-MD** (NeurIPS 2025 Workshop) — arxiv.org/abs/2601.17880
   - The Buraaq dataset paper. Multimodal Quran dataset with word-level audio.

8. **Quranic Audio Dataset** (2024) — arxiv.org/abs/2405.02675
   - RetaSy dataset paper. Crowdsourced correctness labels for error detection.

9. **Open ASR Models for Arabic** (2025) — arxiv.org/abs/2507.13977
   - Training Conformer-CTC on MASC + CommonVoice + EveryAyah for Arabic.

---

## 9. Apple Silicon Notes for FunASR

FunASR is PyTorch-based, so M4 Max compatibility follows the same rules as WeNet:

```bash
# Install FunASR on M4 Max
pip install torch torchvision torchaudio  # MPS included automatically
pip install funasr modelscope

# Set MPS fallback for unsupported ops
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Test inference (no GPU needed for inference)
python -c "
from funasr import AutoModel
model = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512',
                  device='cpu')  # Start with CPU, test MPS later
res = model.generate(input='test.wav')
print(res)
"
```

For fine-tuning on M4 Max:
- Single-GPU MPS training works (same as WeNet)
- DeepSpeed distributed training does NOT work (no NCCL on Mac)
- Use gradient accumulation to compensate for smaller effective batch size
- 128GB unified memory is a strong advantage for large models like Fun-ASR-Nano (800M params)

---

## 10. NVIDIA NeMo FastConformer — The Game-Changer

### 10.1 Critical Discovery: Arabic FastConformer WITH Diacritical Marks Already Exists

This is the single most important finding across all our ASR research. NVIDIA has released **two** Arabic FastConformer models:

| Model | HuggingFace ID | Arabic Harakat | Training Data | License |
|-------|---------------|----------------|---------------|---------|
| Arabic (no diacritics) | `nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0` | ❌ Without harakat | ~760h Arabic | CC-BY-4.0 |
| **Arabic (with diacritics)** | **`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`** | **✅ With harakat** | **~1100h Arabic** | **CC-BY-4.0** |

The `pcd` (punctuation + capitalization + diacritical) variant already outputs Arabic text **with tashkeel/harakat**. It was trained on ~1,100 hours of Arabic speech. This is exactly what Quran ASR needs — diacritical marks are not optional in Quranic text, they change meaning entirely.

Both are Hybrid Transducer-CTC models (115M params), trained with NeMo, and licensed CC-BY-4.0 (commercial use OK).

### 10.2 The English FastConformer CTC Large You Asked About

The model at `nvidia/stt_en_fastconformer_ctc_large`:

| Property | Value |
|----------|-------|
| Architecture | FastConformer encoder + CTC decoder |
| Parameters | ~115M |
| Input | 16kHz mono WAV |
| Output | Lowercase English text |
| Tokenizer | SentencePiece Unigram, vocab=1024 |
| Training data | NeMo ASRSet 3.0 (LibriSpeech, Fisher, Switchboard, WSJ, VCTK, VoxPopuli, MLS, CommonVoice, People's Speech — several thousand hours) |
| WER | 2.1% (LS clean), 4.2% (LS other) |
| License | CC-BY-4.0 |
| Paper | arxiv:2305.05084 — "Fast Conformer with Linearly Scalable Attention" |

**FastConformer** is an optimized Conformer with **8x depthwise-separable convolutional downsampling** instead of the standard 4x. This makes it 2x+ faster in both training and inference compared to regular Conformer, with no accuracy loss.

Key architectural differences from regular Conformer (used in WeNet):

```
Standard Conformer (WeNet):
  Audio → 4x Conv Subsampling → 12-18 Conformer Blocks → CTC/Attention

FastConformer (NeMo):
  Audio → 8x Depthwise-Sep Conv Subsampling → 17 Conformer Blocks → CTC/RNNT/TDT
                                                                     (multi-decoder)
```

The 8x subsampling reduces the sequence length fed into the self-attention layers, making attention compute grow linearly rather than quadratically. This is why it's called "Linearly Scalable Attention."

### 10.3 Why the English CTC Model Isn't the Right Starting Point

While `stt_en_fastconformer_ctc_large` is an excellent model, for Quran ASR you should NOT start from the English checkpoint. Here's why:

1. **Vocabulary mismatch**: English BPE tokenizer with 1024 English subwords. Arabic requires a completely different tokenizer.
2. **The Arabic model already exists**: `stt_ar_fastconformer_hybrid_large_pcd_v1.0` is literally an Arabic FastConformer with diacritical marks. Use THAT as your starting point.
3. **NVIDIA's own pattern**: Their Persian model (`stt_fa_fastconformer_hybrid_large`) was fine-tuned FROM the English checkpoint. But since the Arabic checkpoint already exists, skip that step.

### 10.4 The Right Starting Point: Arabic PCD Model

```python
import nemo.collections.asr as nemo_asr

# This model already outputs Arabic WITH harakat
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)

# Quick test — does it handle Quran recitation?
output = asr_model.transcribe(['quran_recitation.wav'])
print(output[0].text)
# Expected: Arabic text with tashkeel like بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ

# Switch to CTC decoder (faster, simpler)
asr_model.change_decoding_strategy(decoder_type='ctc')
output = asr_model.transcribe(['quran_recitation.wav'])
print(output[0].text)
```

### 10.5 How to Fine-Tune FastConformer on Quran Data

NVIDIA provides an official tutorial notebook: `NeMo/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb`

#### Step 1: Install NeMo

```bash
# On M4 Max (CPU/MPS) or Linux GPU
pip install nemo_toolkit['all']
# or from source for latest
git clone https://github.com/NVIDIA-NeMo/NeMo.git
cd NeMo && pip install -e '.[all]'
```

#### Step 2: Prepare Data in NeMo Manifest Format

NeMo uses JSON manifest files (one JSON object per line):

```python
import json
import soundfile as sf
from datasets import load_dataset

# Load Quran dataset
ds = load_dataset("Salama1429/tarteel-ai-everyayah-Quran",
                  verification_mode="no_checks")

def create_manifest(split, output_path, audio_dir):
    """Convert HuggingFace dataset to NeMo manifest format."""
    import os
    os.makedirs(audio_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(split):
            audio_path = f"{audio_dir}/{i:06d}.wav"

            # Save audio as 16kHz WAV
            sf.write(audio_path, sample["audio"]["array"],
                     sample["audio"]["sampling_rate"])

            # NeMo manifest: one JSON per line
            duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            entry = {
                "audio_filepath": os.path.abspath(audio_path),
                "text": sample["text"],        # Full tashkeel text
                "duration": round(duration, 2)
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

create_manifest(ds["train"], "data/quran_train_manifest.json", "data/audio/train")
create_manifest(ds["validation"], "data/quran_val_manifest.json", "data/audio/val")
```

#### Step 3: Build a New Tokenizer for Quran Arabic

The Arabic PCD model already has an Arabic tokenizer, but for Quran with full tashkeel, you may want a custom one:

```bash
# Option A: Use the existing tokenizer from the Arabic PCD model (recommended first)
# The PCD model already supports harakat — try fine-tuning without changing tokenizer first

# Option B: Build custom tokenizer if the existing one doesn't cover all harakat
python NeMo/scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest=data/quran_train_manifest.json \
    --data_root=data/tokenizer \
    --vocab_size=256 \
    --tokenizer="spe" \
    --spe_type="unigram" \
    --spe_character_coverage=1.0 \
    --no_lower_case
```

#### Step 4: Fine-Tune with the Official Script

**Option A: Keep existing tokenizer (easiest — try this first)**

```bash
python NeMo/examples/asr/speech_to_text_finetune.py \
    --config-path="../asr/conf/fastconformer/hybrid_transducer_ctc/" \
    --config-name="fastconformer_hybrid_transducer_ctc_bpe" \
    \
    +init_from_pretrained_model="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0" \
    \
    ++model.train_ds.manifest_filepath="data/quran_train_manifest.json" \
    ++model.validation_ds.manifest_filepath="data/quran_val_manifest.json" \
    ++model.train_ds.batch_size=16 \
    ++model.validation_ds.batch_size=16 \
    \
    ++trainer.devices=1 \
    ++trainer.max_epochs=50 \
    ++trainer.precision="bf16-mixed" \
    \
    ++model.optim.name="adamw" \
    ++model.optim.lr=1e-4 \
    ++model.optim.weight_decay=0.001 \
    ++model.optim.sched.warmup_steps=500 \
    ++model.optim.sched.d_model=512 \
    \
    ++exp_manager.exp_dir="experiments/quran_fastconformer" \
    ++exp_manager.create_wandb_logger=false
```

**Option B: Replace tokenizer for full harakat coverage**

If the existing tokenizer misses some Quran-specific characters, use the language fine-tuning approach:

```python
import nemo.collections.asr as nemo_asr
import copy

# Load pretrained Arabic model
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)

# Change tokenizer to the custom Quran one
asr_model.change_vocabulary(
    new_tokenizer_dir="data/tokenizer/tokenizer_spe_unigram_v256",
    new_tokenizer_type="bpe"
)

# Update data configs
asr_model.cfg.train_ds.manifest_filepath = "data/quran_train_manifest.json"
asr_model.cfg.validation_ds.manifest_filepath = "data/quran_val_manifest.json"
asr_model.cfg.train_ds.batch_size = 16
asr_model.cfg.validation_ds.batch_size = 16

# Setup data loaders
asr_model.setup_training_data(asr_model.cfg.train_ds)
asr_model.setup_validation_data(asr_model.cfg.validation_ds)

# Configure optimizer (lower LR for fine-tuning)
asr_model.cfg.optim.lr = 1e-4
asr_model.cfg.optim.weight_decay = 1e-3

# Train with PyTorch Lightning
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager

trainer = pl.Trainer(
    devices=1,
    accelerator="auto",     # "mps" on M4 Max, "gpu" on CUDA
    max_epochs=50,
    precision="bf16-mixed",  # Use "32" on M4 Max if bf16 has issues
    accumulate_grad_batches=4,
    enable_checkpointing=True,
    logger=True,
)

# Setup experiment manager for checkpoints
exp_manager(trainer, {
    "exp_dir": "experiments/quran_fastconformer",
    "name": "quran_ar_fastconformer_pcd",
    "checkpoint_callback_params": {
        "save_top_k": 5,
        "monitor": "val_wer",
        "mode": "min",
    },
})

trainer.fit(asr_model)
```

#### Step 5: Evaluate

```bash
python NeMo/examples/asr/speech_to_text_eval.py \
    model_path="experiments/quran_fastconformer/quran_ar_fastconformer_pcd/checkpoints/best.nemo" \
    dataset_manifest="data/quran_val_manifest.json" \
    output_filename="predictions.json" \
    batch_size=32 \
    amp=true
```

#### Step 6: Export to ONNX for Sherpa-ONNX Deployment

```python
# Export from .nemo to ONNX
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    "experiments/quran_fastconformer/best.nemo"
)

# Export CTC variant (simpler for deployment)
asr_model.change_decoding_strategy(decoder_type='ctc')
asr_model.export("quran_fastconformer_ctc.onnx")

# Then use sherpa-onnx for deployment (same pipeline as WeNet/FunASR)
```

### 10.6 Fine-Tuning Strategy for Quran

```
┌──────────────────────────────────────────────────────────┐
│  BASE MODEL: nvidia/stt_ar_fastconformer_hybrid_large_   │
│              pcd_v1.0                                     │
│  • Already knows Arabic with harakat                      │
│  • 115M params, ~1100h Arabic training                    │
│  • CC-BY-4.0 (commercial OK)                              │
│  • Hybrid: use CTC decoder for simplicity                 │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 1: Test Zero-Shot (Day 1)                           │
│  • Run inference on Quran test ayahs                      │
│  • Measure baseline CER with harakat                      │
│  • Check: does it output correct tashkeel?                │
│  • Expected: decent Arabic, imperfect tashkeel            │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 2: Fine-Tune on Quran (Weeks 1-2)                   │
│                                                            │
│  Dataset: tarteel-ai-everyayah-Quran (829h)               │
│  Keep existing tokenizer first → test CER                  │
│  If tokenizer insufficient → rebuild with Quran text       │
│                                                            │
│  Training config:                                          │
│  • LR: 1e-4 (10x lower than from-scratch)                 │
│  • Epochs: 30-50                                           │
│  • Freeze encoder first 10 epochs (optional)               │
│  • Accumulate gradients: 4-8 steps                         │
│  • M4 Max: ~3-5 days for 50 epochs                         │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 3: Add Error Detection Data (Week 3)                │
│                                                            │
│  Fine-tune further on RetaSy/quranic_audio_dataset         │
│  • 7K recitations with correct/incorrect labels            │
│  • Teach model to recognize non-native speaker errors      │
│  • Multi-task: ASR output + correctness classification     │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 4: Export & Deploy                                   │
│                                                            │
│  Export: .nemo → ONNX → Sherpa-ONNX                       │
│  Deploy: Go WebSocket server on K8s                        │
│  Client: Flutter app with sherpa-onnx Dart bindings        │
└──────────────────────────────────────────────────────────┘
```

### 10.7 Apple Silicon Notes for NeMo

NeMo is PyTorch Lightning-based. On M4 Max:

```bash
pip install nemo_toolkit['all']
# or minimal ASR-only install:
pip install nemo_toolkit['asr']

export PYTORCH_ENABLE_MPS_FALLBACK=1
```

- **Inference**: Works on CPU (fast enough with 16 cores). MPS may work but untested with NeMo.
- **Fine-tuning**: Use `accelerator="cpu"` or `accelerator="mps"` (experimental). bf16 may need to fall back to fp32 on MPS.
- **115M params**: Fits easily in 128GB unified memory. Much lighter than Fun-ASR-Nano (800M).
- **No multi-GPU**: Same limitation as WeNet/FunASR — no DDP on Mac.

For serious training, a single cloud A100 at ~$1.50/hr would finish 50 epochs on 829h in ~2-3 days (~$100).

---

## 11. Updated Head-to-Head: All Three Frameworks

| Criterion | NeMo FastConformer | FunASR | WeNet | Winner |
|-----------|-------------------|--------|-------|--------|
| **Arabic pretrained WITH harakat** | ✅ `stt_ar_*_pcd_v1.0` (1100h) | ✅ Fun-ASR-Nano (31 langs) | ❌ None | **NeMo** |
| **Diacritical marks support** | ✅ Dedicated PCD model | ⚠️ Untested on harakat | ✅ Custom char vocab | **NeMo** |
| **Model size** | 115M (efficient) | 800M (Fun-ASR-Nano) | 30-120M (configurable) | NeMo/WeNet |
| **Streaming** | ✅ Cache-aware streaming variant | ✅ Chunk Paraformer + 2pass | ✅ U2++ dynamic chunk | Tie |
| **Inference speed** | Fast (8x subsampling) | 10x NAR single-step | Standard AR/CTC | FunASR |
| **Fine-tuning docs** | ✅ Official notebook + scripts | ✅ finetune.sh scripts | ✅ YAML configs | Tie |
| **Fine-tune from Arabic base** | ✅ Proven (Persian was done same way) | ⚠️ Nano fine-tune undocumented | ❌ Train from scratch | **NeMo** |
| **Production deployment** | ONNX export + Riva | Docker + WebSocket + ONNX | Sherpa-ONNX native | Tie |
| **License (models)** | CC-BY-4.0 ✅ | Custom MODEL_LICENSE ⚠️ | Apache 2.0 ✅ | NeMo/WeNet |
| **Community** | NVIDIA-backed, 13K stars | Alibaba-backed, 14K stars | Community, 4K stars | NeMo/FunASR |
| **Go/K8s stack fit** | ONNX → Sherpa → Go | Docker/WebSocket | Go native bindings | WeNet |
| **Apple Silicon** | ⚠️ CPU works, MPS experimental | ⚠️ CPU works, MPS experimental | ⚠️ MPS with fallback | Tie |
| **Paper** | arxiv:2305.05084 | arxiv:2206.08317 | arxiv:2102.01547 | — |

### 11.1 The New Recommended Path

The discovery of `stt_ar_fastconformer_hybrid_large_pcd_v1.0` changes the entire strategy:

**Before**: Train from scratch (WeNet) or hope Fun-ASR-Nano handles Quran Arabic
**Now**: Fine-tune a proven Arabic+harakat model on 829h of Quran recitation data

This is the shortest path to production-quality Quran ASR:

```
Day 1:  pip install nemo_toolkit['asr']
        Test stt_ar_fastconformer_hybrid_large_pcd_v1.0 on Quran audio
        Measure baseline CER

Day 2:  Prepare tarteel-ai-everyayah-Quran in NeMo manifest format
        Test tokenizer coverage on Quran tashkeel

Week 1: Fine-tune with speech_to_text_finetune.py
        Monitor val_wer / val_cer

Week 2: Export to ONNX → Sherpa-ONNX
        Test in Flutter app

Week 3: Add error detection capability
        Deploy to K8s
```

---

## 12. All Research Papers

### NeMo / FastConformer
1. **Fast Conformer** (2023) — arxiv.org/abs/2305.05084
   - 8x depthwise-sep conv downsampling, linearly scalable attention. 2x faster than Conformer.

2. **Conformer** (2020) — arxiv.org/abs/2005.08100
   - Original Conformer architecture. Self-attention + convolution hybrid.

### FunASR / Paraformer
3. **Paraformer** (2022) — arxiv.org/abs/2206.08317
   - CIF predictor, GLM sampler, MWER training. NAR with 10x speedup.

4. **Paraformer-v2** (2024) — arxiv.org/abs/2409.17746
   - CTC replaces CIF. Better multilingual + noise robustness.

5. **FunAudioLLM** (2024) — arxiv.org/abs/2407.04051
   - SenseVoice + CosyVoice foundation models.

6. **Fun-ASR Technical Report** (2025) — arxiv.org/abs/2509.12508
   - Fun-ASR-Nano. Tens of millions of hours, 31 languages.

### WeNet
7. **WeNet (U2)** (2021) — arxiv.org/abs/2102.01547
   - Unified streaming/non-streaming with dynamic chunk CTC+Attention.

### Arabic ASR / Quran Datasets
8. **Open Arabic ASR Leaderboard** (2024) — arxiv.org/abs/2412.13788
9. **Quran-MD** (NeurIPS 2025) — arxiv.org/abs/2601.17880
10. **Quranic Audio Dataset** (2024) — arxiv.org/abs/2405.02675

---

## 13. Final Summary

**The strongest path for Quran ASR is now clear:**

Start with NVIDIA's `stt_ar_fastconformer_hybrid_large_pcd_v1.0` — a 115M parameter Arabic model that already outputs text with diacritical marks (harakat), trained on ~1,100 hours of Arabic speech, licensed CC-BY-4.0 for commercial use. Fine-tune it on the tarteel-ai-everyayah-Quran dataset (829h, 36 reciters, fully diacritized). This gives you transfer learning from a proven Arabic acoustic model rather than training from scratch.

**Use FunASR's Fun-ASR-Nano as a comparison baseline** — it already supports Arabic with streaming and costs nothing to test. If it outperforms the fine-tuned NeMo model on Quran data, switch.

**Keep WeNet as the fallback** for maximum control over tokenization and if you need the deepest Go/K8s integration via Sherpa-ONNX's native bindings.

**All three frameworks converge on the same deployment path**: export to ONNX → serve via Sherpa-ONNX → Go WebSocket server on K8s → Flutter client with Dart bindings. The training framework choice doesn't lock you into a deployment stack.
