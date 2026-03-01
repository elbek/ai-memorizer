# Quran Recitation ASR — Technical Research & Architecture Options

## 1. Problem Statement

Build a cloud-hosted, subscription-based mobile service that:

1. **Listens** to a user's live Quran recitation via streaming audio from a mobile app
2. **Matches** the recitation in real-time to the correct position in the Quran text (word-by-word tracking)
3. **Detects mistakes** at the harakat level — including wrong damma/kasra/fatha, missing madd, incorrect ghunnah, tajweed violations, makhraj errors, and word substitutions
4. **Understands English** — the user may issue voice commands or ask questions in English mid-session
5. **Generates a mistake report** at the end of each recitation session
6. **Scales** to thousands of concurrent users

The core technical challenge is producing an ASR transcript that preserves **full diacritization (tashkeel)** — fatha, kasra, damma, sukun, shadda, tanween, and madd markers — because without these, you cannot distinguish correct from incorrect recitation. Standard Arabic ASR strips diacritics; Quran ASR must retain them.

---

## 2. How Tarteel Solved It

Tarteel is the market leader with 15M+ downloads and 250K+ audio hours transcribed per month. Their published technical journey reveals a multi-year evolution:

### 2.1 Data Collection (2018–present)

Tarteel started as a **crowdsourced data collection challenge** at the 2018 "Muslim Hacks 2.0" hackathon. They built a Django web app that displayed random verses for users to record. Key learnings from their blog:

- They collected 10K recordings in the first month, then scaled to 50K+
- They discovered that **professional reciters sound fundamentally different from average Muslims** — different tajweed levels, background noise, age/gender distributions. Training on professional audio alone doesn't generalize.
- Audio was uploaded in 20-second chunks linked to user sessions, validated with PyDub
- Metadata collection was initially poor (many null demographic fields), which they later improved
- They built a community of annotators but found crowdsourcing Quran annotation extremely hard — "you can't Mechanical Turk your way with labelling Quran recitations; not everyone understands Arabic and even fewer are capable of understanding and correcting Quran recitation"

### 2.2 Production ML Stack

According to the CoreWeave case study and NVIDIA customer story, Tarteel's production stack is:

| Component | Technology |
|---|---|
| **ASR Model Training** | NVIDIA NeMo (FastConformer architecture, fine-tuned on their proprietary dataset) |
| **Inference Serving** | NVIDIA Riva SDK with Triton Inference Server |
| **Streaming** | gRPC-based streaming with configurable audio chunk size, padding, and VAD |
| **Infrastructure** | CoreWeave GPU cloud (migrated from AWS — 22% latency reduction, 56% cost savings) |
| **Deployment** | Kubernetes via Zeet, Docker containers, autoscaling |
| **Optimization** | TensorRT for model optimization, multi-GPU scaling via Triton |

### 2.3 Key Technical Decisions

- They chose **NeMo's Conformer/FastConformer** models (not Whisper) for production because NeMo + Riva provides a complete pipeline from training to low-latency streaming inference
- They also released fine-tuned **Whisper models** on HuggingFace (`tarteel-ai/whisper-base-ar-quran`, `tarteel-ai/whisper-tiny-ar-quran`) as research artifacts, achieving ~5.75% WER
- Riva's streaming capability was critical — it allows real-time word-by-word highlighting as the user recites
- They use VAD (Voice Activity Detection) tuning to handle pauses between ayat naturally

### 2.4 What Tarteel Does NOT Do (Yet)

Based on App Store descriptions and reviews, Tarteel detects **word-level** errors and incorrect tashkeel, but users note it is not a substitute for a Sheikh with ijazah. Harakat-level precision at the level of detecting every single wrong damma vs. kasra is still aspirational for any system.

---

## 3. Model Options — Detailed Comparison

### Option A: Fine-Tuned Whisper (OpenAI)

**Architecture:** Encoder-decoder transformer, trained on 680K hours of multilingual audio.

**Approach:** Fine-tune `whisper-large-v3` or `whisper-medium` on Quran recitation datasets using LoRA or full fine-tuning, with diacritized transcripts as targets.

| Aspect | Details |
|---|---|
| **Base Model** | `openai/whisper-large-v3` (1.55B params), `whisper-medium` (769M) |
| **Existing Work** | `tarteel-ai/whisper-base-ar-quran` (WER 5.75%), `KheemP/whisper-base-quran-lora` (WER 5.98% with diacritics) |
| **Training Data** | Can use OpenSLR-132 (24GB, multiple reciters), Buraaq/quran-audio-text-dataset (187K ayah samples, 30 reciters), Salama1429/tarteel-ai-everyayah-Quran (90K samples with diacritization, 36 reciters) |
| **Fine-tuning Method** | LoRA (efficient, ~5-10% params), QLoRA, or full fine-tuning |
| **Diacritics** | Whisper CAN learn to output diacritized text if the training transcripts include tashkeel — demonstrated by existing Quran Whisper models |
| **Streaming** | Whisper is natively an offline/batch model. Real-time streaming requires chunked inference with overlapping windows (using faster-whisper/WhisperX) or sliding window approaches. Latency of 1-3 seconds achievable with faster-whisper + CTranslate2 |
| **English Support** | Native — Whisper is multilingual. Can handle code-switching between Arabic recitation and English commands |
| **Scalability** | faster-whisper with CTranslate2 gives ~70x real-time on GPU. Can serve on Triton or vLLM |

**Pros:** Multilingual out of the box (Arabic + English in one model), large community, excellent existing fine-tuned checkpoints, straightforward HuggingFace training pipeline.

**Cons:** Not designed for streaming — adding real-time streaming requires engineering workarounds. Encoder-decoder architecture has higher latency than CTC-based models. 30-second window limit requires chunking logic.

---

### Option B: NVIDIA NeMo FastConformer + Riva (Tarteel's Approach)

**Architecture:** FastConformer encoder with CTC, RNN-T (Transducer), or TDT decoder. ~115M params for "large" variant.

**Approach:** Fine-tune `stt_en_fastconformer_hybrid_large` or train from scratch on Arabic Quran data using NeMo toolkit, deploy via Riva.

| Aspect | Details |
|---|---|
| **Base Model** | NVIDIA Parakeet/FastConformer (CTC or Transducer variants) |
| **Existing Work** | `MostafaAhmed98/Conformer-CTC-Arabic-ASR` (CommonVoice Arabic), NVIDIA's Persian FastConformer (similar fine-tuning path) |
| **Training** | NeMo framework with YAML config, supports BPE/character tokenizers. Character-based tokenizer recommended for Arabic to capture individual harakat |
| **Streaming** | **Native streaming support** via Riva SDK — this is the killer feature. gRPC streaming with configurable chunk sizes, padding, and VAD. Sub-150ms latency |
| **Deployment** | Riva + Triton Inference Server, containerized, Kubernetes-native, auto-scaling to thousands of streams |
| **English Support** | Requires separate English model or Canary (NVIDIA's multilingual model supporting 25 languages) |
| **Scalability** | Production-proven at Tarteel's scale (250K+ hours/month). TensorRT optimization, multi-GPU, batched inference |

**Pros:** Purpose-built for streaming ASR. Production-proven in Quran recitation at scale. Best latency characteristics. Riva provides complete deployment pipeline. CTC/Transducer architecture is ideal for real-time word-by-word tracking.

**Cons:** Steeper learning curve (NeMo ecosystem). NVIDIA GPU lock-in. No native multilingual support in a single model — would need separate English model or use Canary. Smaller community than Whisper. Proprietary deployment stack (Riva is not fully open-source).

---

### Option C: WeNet — Community-Driven NeMo Alternative ⭐

**Architecture:** U2 (Unified Two-Pass) — hybrid CTC/attention with Conformer or Transformer encoder. The CTC decoder provides streaming first-pass results, while an attention decoder rescores for higher accuracy in a second pass.

**Approach:** Train a Conformer encoder with character-level diacritized Arabic output using WeNet's training pipeline, export to ONNX/LibTorch for production deployment.

| Aspect | Details |
|---|---|
| **Base Architecture** | Conformer encoder + hybrid CTC/Attention decoder (U2/U2++ framework) |
| **Model Size** | Configurable — typically 30M-120M params depending on encoder depth/width |
| **Training** | Pure PyTorch, YAML-based config, on-the-fly feature extraction via Torchaudio, DDP distributed training |
| **Streaming** | **Native streaming via dynamic chunk-based attention** — allows self-attention to focus on limited right context with configurable chunk sizes. Streaming and non-streaming unified in a single model — no need to train two separate models |
| **Deployment** | LibTorch C++ runtime, ONNX export, PyTorch quantization (INT8), runtimes for x86 servers, ARM Android, iOS |
| **English Support** | Train separate model or build bilingual model with shared encoder |
| **Scalability** | Production-proven at Tencent scale. LibTorch C++ runtime is efficient for server deployment |
| **License** | Apache 2.0, fully open source, no vendor lock-in |

**Why WeNet is compelling for this project:**

1. **U2 Two-Pass is ideal for Quran tracking.** The CTC first pass emits tokens incrementally for real-time word highlighting (low latency). The attention decoder then rescores for higher accuracy — this more accurate transcript is what you use for harakat error detection. You get both fast UX and accurate error checking from the same model.

2. **No NVIDIA lock-in.** Runs on any GPU (or even CPU). No dependency on Riva, Triton, or TensorRT. This means you can deploy on cheaper cloud providers, AMD GPUs, or even run on CPU for cost optimization.

3. **Same architecture quality as NeMo.** WeNet uses the same Conformer architecture and CTC/Transducer decoders. The accuracy gap with NeMo is negligible — both achieve SOTA results on standard benchmarks. The difference is purely in the deployment tooling.

4. **Production-first design.** Unlike ESPnet or SpeechBrain which are research-first, WeNet was designed from day one for production. The C++ LibTorch runtime, quantization support, and server deployment patterns are baked in.

5. **Clean PyTorch codebase.** No Kaldi dependency, no complex build systems. If something breaks, you can debug it because the code is straightforward PyTorch.

**What you need to build yourself (vs NeMo+Riva):**

- A gRPC/WebSocket streaming server around the LibTorch/ONNX runtime (NeMo gets this free with Riva)
- VAD integration (WeNet has basic VAD but not as configurable as Riva's)
- Load balancing and autoscaling for concurrent streams (Riva+Triton handles this natively)

**Recommended WeNet configuration for Quran ASR:**

```yaml
# Key training config decisions
encoder: conformer
encoder_conf:
    output_size: 256        # or 512 for larger model
    num_blocks: 12          # 12-18 layers
    attention_heads: 4
    linear_units: 2048
    cnn_module_kernel: 15   # convolution kernel size
    
decoder: transformer      # attention decoder for second pass
ctc_weight: 0.3           # CTC loss weight (0.3-0.5 typical)

# Character-level tokenizer for Arabic with harakat
tokenizer: char           # NOT BPE — use character-level to preserve each diacritic
vocab_size: ~200           # Arabic letters + all harakat marks + special tokens

# Streaming config
chunk_size: [16, 32, -1]   # Dynamic chunk training: 16/32 frames or full attention
                           # At inference: use 16 for low-latency streaming
```

---

### Option D: Next-Gen Kaldi (K2 + Icefall + Sherpa-ONNX)

**Architecture:** Zipformer (their own architecture, faster than Conformer) with CTC or Transducer decoder. Three repos work together: K2 (GPU FSA engine), Icefall (training recipes), Sherpa-ONNX (deployment runtime).

| Aspect | Details |
|---|---|
| **Base Architecture** | Zipformer encoder (novel, faster than Conformer) + CTC or Transducer |
| **Training** | Icefall (PyTorch training recipes, similar to ESPnet/WeNet) |
| **Streaming** | Streaming Zipformer-CTC and Zipformer-Transducer with WebSocket server |
| **Deployment** | Sherpa-ONNX — ONNX runtime with WebSocket server in C++ and Go, supports Android/iOS/Flutter/HarmonyOS, Raspberry Pi, RISC-V |
| **Key Advantage** | Can load models from NeMo, WeNet, or Icefall — it's a universal deployment runtime |
| **Mobile** | Flutter streaming ASR examples, Android/iOS SDKs, runs on CPU without internet |
| **Language Support** | 12 programming languages (C++, Python, Go, Swift, Kotlin, Java, C#, Dart, JavaScript, Rust, Pascal, etc.) |
| **License** | Apache 2.0 |
| **GitHub Stars** | ~10.3K (sherpa-onnx — very active, backed by Xiaomi) |

**Why Sherpa-ONNX matters even if you don't train with Icefall:**

Sherpa-ONNX is framework-agnostic. You can train your Quran model in WeNet, NeMo, or ESPnet, export to ONNX, and deploy via Sherpa-ONNX. This gives you:
- A Go-based WebSocket server (fits naturally into your existing Go microservices stack)
- Flutter mobile SDK for streaming ASR (if your mobile app uses Flutter)
- INT8 quantization for CPU-only deployment (dramatically reduces GPU costs)
- Built-in VAD integration

**The pragmatic combo: Train in WeNet → Deploy in Sherpa-ONNX** gives you the best of both worlds — WeNet's clean training pipeline with Sherpa's battle-tested deployment runtime and mobile SDKs.

---

### Option E: Wav2Vec2-BERT / HuBERT + CTC Fine-Tuning

**Architecture:** Self-supervised speech representation model (pre-trained on 4.5M hours including 110K Arabic hours) with CTC head.

**Approach:** Fine-tune wav2vec2-BERT or mHuBERT on Quran data with a custom phonetic output script.

| Aspect | Details |
|---|---|
| **Base Model** | `facebook/w2v-bert-2.0` (600M params, 4.5M hours pre-training) |
| **Existing Work** | `TBOGamer22/wav2vec2-quran-phonetics` (phonetic output), arxiv:2509.00094 (Quran Phonetic Script with wav2vec2-BERT for segmentation and error detection) |
| **Key Innovation** | The 2509.00094 paper introduces **Quran Phonetic Script (QPS)** — a custom phonetic representation that encodes tajweed rules (madd lengths, ghunnah types, etc.) at the phoneme level. This is the most sophisticated approach to tajweed error detection published |
| **Training Data** | They produced 850+ hours / 300K annotated utterances with an automated pipeline |
| **Streaming** | CTC-based models are naturally streamable — emit output token-by-token as audio arrives |
| **English Support** | Would need separate model or multilingual wav2vec2 variant |
| **Pronunciation Assessment** | CTC models + forced alignment enable **Goodness of Pronunciation (GOP) scores** for each phoneme — directly applicable to harakat-level error detection |

**Pros:** Best architecture for phoneme-level pronunciation assessment. CTC naturally supports streaming. QPS approach is the most academically rigorous for tajweed error detection. GOP scoring gives granular error diagnosis. mHuBERT shows strong results (86-87% true acceptance rate on QuranMB.v1 benchmark).

**Cons:** More research-oriented — fewer production deployment tools. Requires building own serving infrastructure. QPS approach requires significant domain expertise to implement the phonetic script.

---

### Option F: ESPnet — Academic Research Platform

**Architecture:** Supports virtually every ASR architecture — Transformer, Conformer, Branchformer, CTC, Transducer, attention-based, and hybrid models.

| Aspect | Details |
|---|---|
| **Scope** | Full speech processing toolkit: ASR, TTS, speech translation, enhancement, diarization, SLU |
| **Streaming** | Streaming Conformer/Transformer with blockwise synchronous beam search |
| **Strengths** | Largest collection of reproducible recipes, hundreds of datasets/languages with published results, can import Whisper/wav2vec2/HuBERT as components |
| **Deployment** | Not production-ready out of the box — requires custom work. Best used for training, then export model for deployment elsewhere |
| **Use For This Project** | Experimentation with QPS phonetic approach, running ablation studies on different architectures, benchmarking |
| **License** | Apache 2.0 |

---

### Option G: SpeechBrain — Cleanest API

**Architecture:** Pure PyTorch toolkit supporting CTC, Transducer, attention-based ASR, plus pronunciation assessment tools.

| Aspect | Details |
|---|---|
| **Strengths** | Most Pythonic API, easiest onboarding, includes pronunciation assessment tools useful for harakat checking, built-in wav2vec2/HuBERT integration |
| **Streaming** | No native streaming runtime |
| **Deployment** | Weakest production story — no built-in server, no mobile SDK |
| **Use For This Project** | Rapid prototyping, pronunciation assessment R&D |
| **License** | Apache 2.0 |

---

## 4. Full Comparison Matrix

| Criterion | Whisper (Fine-tuned) | NeMo + Riva | WeNet | Sherpa-ONNX (K2) | Wav2Vec2-BERT | ESPnet | SpeechBrain |
|---|---|---|---|---|---|---|---|
| **Streaming Latency** | 1-3s (chunked) | <150ms (native) | <200ms (U2 CTC pass) | <200ms (native) | <300ms (CTC) | ~500ms (blockwise) | No streaming |
| **D-WER Potential** | 1-3% (large model) | 1-3% (proven) | 1-3% (same arch) | 1-3% (same arch) | 2-4% (estimated) | 2-4% | 3-5% |
| **English Support** | Native (multilingual) | Separate model | Separate model | Separate model | Separate model | Separate model | Separate model |
| **Tajweed Detection** | Word-level | Word + harakat | Word + harakat | Word + harakat | Phoneme-level (QPS) | Phoneme-level | Pronunciation scoring |
| **Production Runtime** | faster-whisper | Riva + Triton | LibTorch C++, ONNX | ONNX + WebSocket C++/Go | Custom needed | Custom needed | Custom needed |
| **Mobile SDK** | No | No (server only) | Android/iOS (LibTorch) | Android/iOS/Flutter/HarmonyOS | No | No | No |
| **GPU Vendor Lock-in** | None | NVIDIA (Riva/TensorRT) | None | None | None | None | None |
| **Training Complexity** | Low (HuggingFace) | Medium (NeMo YAML) | Medium (YAML) | Medium (Icefall recipes) | Medium | Medium (recipes) | Low (Pythonic) |
| **Deployment Complexity** | Low | Medium (Riva setup) | Medium (build server) | Low (server included) | High | High | High |
| **Concurrent Users/GPU** | 4-8 (large model) | 50-100 | 50-100 | 50-100+ (INT8 CPU possible) | 20-40 | 20-40 | N/A |
| **Community Size** | Largest | NVIDIA enterprise | Medium (Tencent-backed) | Large (10K+ stars, Xiaomi-backed) | Large (Meta) | Large (academic) | Large (academic) |
| **License** | MIT | Mixed (Riva proprietary) | Apache 2.0 | Apache 2.0 | MIT | Apache 2.0 | Apache 2.0 |
| **Go Integration** | No | gRPC client | Build own | **Go WebSocket server** | No | No | No |
| **Time to MVP** | 1-2 months | 2-3 months | 2-3 months | 2-3 months | 3-4 months | 3-4 months | 2-3 months |
| **Production Readiness** | Medium | High | High | High | Low | Low | Low |

---

## 5. Recommended Architecture: WeNet + Sherpa-ONNX Hybrid

Given your Go microservices stack and Kubernetes infrastructure, here is the recommended approach:

### Training Pipeline: WeNet

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│                                                         │
│  Quran Audio Datasets ──► Preprocessing ──► WeNet       │
│  (OpenSLR-132,              (16kHz,         Training    │
│   EveryAyah,                 char-level     (Conformer  │
│   Quran-MD)                  tokenizer       U2/U2++)   │
│                              with harakat)       │      │
│                                                  │      │
│                                         ┌────────▼──┐   │
│                                         │ .pt model │   │
│                                         └────────┬──┘   │
│                                                  │      │
│                              ┌───────────────────┤      │
│                              ▼                   ▼      │
│                        ONNX Export         LibTorch JIT  │
│                              │                   │      │
│                              ▼                   ▼      │
│                     Sherpa-ONNX            WeNet C++     │
│                     Deployment             Runtime       │
└─────────────────────────────────────────────────────────┘
```

### Production Architecture: Sherpa-ONNX on Your K8s Stack

```
┌──────────────┐     WebSocket (streaming)   ┌─────────────────────────────┐
│  Mobile App  │ ◄──────────────────────────► │     API Gateway             │
│  (iOS/Andr)  │    audio chunks +            │     (Kong/Envoy)            │
│              │    real-time results          │                             │
└──────────────┘                              └─────────┬───────────────────┘
                                                        │
                                              ┌─────────▼───────────────────┐
                                              │   Audio Stream Router       │
                                              │   (Language ID + VAD)       │
                                              │   Lightweight Go service    │
                                              └──┬──────────────────────┬───┘
                                                 │                      │
                                        Arabic Recitation          English Speech
                                                 │                      │
                                       ┌─────────▼──────────┐  ┌───────▼────────┐
                                       │  Quran ASR Service  │  │  English ASR   │
                                       │                     │  │                │
                                       │  Sherpa-ONNX        │  │  Whisper-small │
                                       │  Go WebSocket       │  │  via faster-   │
                                       │  Server             │  │  whisper       │
                                       │                     │  │                │
                                       │  WeNet Conformer    │  │  Handles       │
                                       │  U2 model (ONNX)    │  │  commands &    │
                                       │                     │  │  questions     │
                                       │  CTC pass: real-    │  │                │
                                       │  time tracking      │  │                │
                                       │  Attn pass: error   │  │                │
                                       │  detection          │  │                │
                                       └─────────┬──────────┘  └───────┬────────┘
                                                 │                      │
                                       ┌─────────▼──────────┐  ┌───────▼────────┐
                                       │  Position Matcher   │  │  Command       │
                                       │  (edit-distance     │  │  Processor     │
                                       │   alignment against │  │  (NLU/Intent)  │
                                       │   Quran text DB)    │  │                │
                                       └─────────┬──────────┘  └───────┬────────┘
                                                 │                      │
                                       ┌─────────▼──────────┐          │
                                       │  Error Detector     │          │
                                       │  • Harakat diff     │          │
                                       │  • Tajweed rules    │          │
                                       │  • Madd checker     │          │
                                       │  • GOP scores       │          │
                                       └─────────┬──────────┘          │
                                                 │                      │
                                       ┌─────────▼──────────────────────▼───┐
                                       │         Session Manager            │
                                       │  • Real-time position tracking     │
                                       │  • Mistake accumulator             │
                                       │  • Session state (Redis)           │
                                       │  • Report generator                │
                                       └────────────────────┬──────────────┘
                                                            │
                                                  ┌─────────▼──────────┐
                                                  │   PostgreSQL +     │
                                                  │   NATS JetStream   │
                                                  │   (session events, │
                                                  │    user progress)  │
                                                  └────────────────────┘
```

### Why This Stack Works for You

1. **Sherpa-ONNX has a native Go WebSocket server** — fits directly into your Go microservices. No Python server needed for inference.
2. **WeNet's U2 two-pass** gives you streaming + accuracy in one model — CTC first pass for real-time highlights, attention rescore for the error report.
3. **ONNX runtime with INT8 quantization** means you can run on CPU for low-traffic periods, scaling to GPU only when concurrent streams demand it. This dramatically reduces your subscription service costs.
4. **Sherpa-ONNX has Flutter/mobile SDKs** — if you ever want to offer an offline mode where the model runs on-device, the path is already there.
5. **No vendor lock-in** — everything is Apache 2.0. You can deploy on any cloud, any GPU vendor, or even CPU-only.
6. **NATS JetStream integration** — your existing event streaming infrastructure handles session events and real-time updates to the mobile app.

---

## 6. Evaluation Metrics System

### 6.1 Standard ASR Metrics

| Metric | Formula | Use Case |
|---|---|---|
| **WER (Word Error Rate)** | (S + D + I) / N, where S=substitutions, D=deletions, I=insertions, N=total words | Overall word-level accuracy. Industry standard. |
| **CER (Character Error Rate)** | Same formula but at character level | More granular — critical for Arabic where single character changes alter meaning |
| **Diacritized WER (D-WER)** | WER computed on fully diacritized text (tashkeel included, no normalization) | **Primary metric** — measures whether harakat are correct. The KheemP/whisper-base-quran-lora model reports this at 5.98% |
| **Diacritized CER (D-CER)** | CER with diacritics preserved | Most granular standard metric for harakat precision |

### 6.2 Tajweed-Specific Metrics

These are novel metrics you should develop for your evaluation framework:

| Metric | Description |
|---|---|
| **Harakat Accuracy (HA)** | Percentage of words where all harakat (fatha, kasra, damma, sukun, shadda, tanween) are correctly identified. Computed by comparing predicted vs. reference diacritics per word |
| **Madd Detection Rate (MDR)** | Precision/recall for detecting madd rules — natural madd (2 beats), connected madd (4-5 beats), separated madd. Binary classification: was the madd present and the correct type? |
| **Ghunnah Detection Rate (GDR)** | Precision/recall for ghunnah (nasalization) in noon sakinah/tanween rules (idgham, ikhfa, iqlab) |
| **Tajweed Rule F1** | Per-rule F1 score across the major tajweed categories: idgham, ikhfa, iqlab, idhar, qalqalah, madd types |
| **True Acceptance (TA)** | Percentage of correctly recited segments that the system correctly accepts (from QuranMB.v1 benchmark) |
| **False Rejection (FR)** | Percentage of correct recitation wrongly flagged as error |
| **False Acceptance (FA)** | Percentage of incorrect recitation wrongly accepted — **this is the most dangerous metric for a Quran app** |
| **Correct Diagnosis (CD)** | When an error IS detected, how often is the specific error type correctly identified? |
| **Latency (P50/P95/P99)** | End-to-end latency from audio chunk received to transcript emitted. Target: <300ms for real-time feel |
| **RTF (Real-Time Factor)** | Processing time / audio duration. Must be < 1.0 for real-time. Target: < 0.3 |

### 6.3 Evaluation Dataset Construction

Build a held-out evaluation set with three tiers:

1. **Gold Standard (Expert Reciters):** 500+ ayat from 5+ professional reciters with ijazah. Perfect recitation = ground truth. Tests TA and FR.
2. **Learner Recitations (With Annotated Errors):** 500+ ayat from non-expert reciters with tajweed mistakes annotated by qualified teachers. Each mistake labeled with type (wrong harakat, missing madd, makhraj error, etc.) and location. Tests FA and CD.
3. **Adversarial/Edge Cases:** Similar-sounding verses (mutashabihat), verses with complex tajweed combinations, different qira'at styles, heavy accents, background noise, whispering, children's voices.

Use the **RetaSy/quranic_audio_dataset** (7000 recitations from 1287 non-Arabic speakers, with correct/incorrect labels) as a starting point for tier 2.

### 6.4 Automated Evaluation Pipeline

```
┌─────────────────────────────────────────────────────┐
│                 Evaluation Pipeline                  │
│                                                     │
│  Audio Sample ──► ASR Model ──► Predicted Text      │
│                                      │              │
│  Reference Text ─────────────────────┤              │
│                                      ▼              │
│                              Alignment Engine       │
│                           (forced alignment or      │
│                            edit distance based)     │
│                                      │              │
│                    ┌─────────────────┼──────┐       │
│                    ▼                 ▼      ▼       │
│               Standard          Harakat  Tajweed    │
│             WER / CER          Accuracy   Rule      │
│                                           Checker   │
│                    │                 │      │       │
│                    ▼                 ▼      ▼       │
│              ┌─────────────────────────────────┐    │
│              │     Evaluation Report            │    │
│              │  • Overall WER/CER/D-WER/D-CER  │    │
│              │  • Per-surah breakdown           │    │
│              │  • Per-tajweed-rule F1           │    │
│              │  • Confusion matrix (harakat)    │    │
│              │  • TA/FR/FA/CD rates             │    │
│              │  • Latency percentiles           │    │
│              └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 7. Available Datasets for Fine-Tuning

| Dataset | Size | Content | Diacritized | Source |
|---|---|---|---|---|
| **OpenSLR-132** | 24 GB | Multiple reciters, full Quran, ayah-level segments | Yes (transcripts available with/without harakat) | openslr.org/132 |
| **Buraaq/quran-audio-text-dataset (Quran-MD)** | 187K ayah samples + 77K word samples | 30 reciters, ayah + word-level audio, transliteration, translation | Yes (full tashkeel) | HuggingFace |
| **Salama1429/tarteel-ai-everyayah-Quran** | 90K samples (~930 hours) | 36 reciters, ayah-level, 16kHz | Yes (with tashkeel) | HuggingFace |
| **MohamedRashad/Quran-Recitations** | Multiple reciters | Famous qaris (Sudais, Husary, Minshawi, etc.) | Yes (full tashkeel) | HuggingFace |
| **RetaSy/quranic_audio_dataset** | 7K recitations | Non-native reciters, labeled correct/incorrect with annotation metadata | Yes | HuggingFace |
| **Tarteel Crowdsourced Data** | 244K+ contributions | Diverse global reciters, various quality levels | Partial | tarteel.ai (may need partnership) |
| **Kaggle Quran Ayat STT** | Multiple reciters | Original sampling rates, ayah-level | Yes | kaggle.com |
| **Quran Speech-to-Text Dataset** | 24 GB: 43,652 WAV (7 imams × 6,236) + ~25K user WAVs from Tarteel.io | Full Quran imam recitations with CSV transcripts for 11K subset; ~18K accepted-quality user recordings | Yes | [archive.org](https://archive.org/details/quran-speech-dataset) |

**Recommended Strategy:**
- Use **Salama1429/tarteel-ai-everyayah-Quran** + **OpenSLR-132** as primary training data (largest, cleanest, diacritized)
- Use **Buraaq/quran-audio-text-dataset** for word-level training if doing word-level segmentation
- Use **RetaSy/quranic_audio_dataset** for error detection model training (has correct/incorrect labels from non-native speakers)
- Reserve a portion of each for evaluation

---

## 8. Infrastructure Sizing (Initial)

| Component | Hardware | Instances | Notes |
|---|---|---|---|
| Quran ASR (Sherpa-ONNX) | NVIDIA L4 or CPU (INT8) | 2-4 | Each GPU handles ~50-100 concurrent streams; CPU handles ~10-20 with INT8 |
| English ASR (faster-whisper) | NVIDIA T4 | 1-2 | Lower traffic, simpler model |
| Language ID/VAD | CPU | 2 | Lightweight classifier |
| API Gateway | CPU | 2-3 | Standard web tier |
| Session Manager (Go) | CPU | 2-3 | Redis-backed stateful service |
| Position Matcher / Error Detector | CPU | 2-3 | Text processing, no GPU needed |

Scale horizontally with Kubernetes HPA based on active stream count.

**Cost Optimization with Sherpa-ONNX INT8:**
- During low-traffic hours (late night), scale down to CPU-only nodes with INT8 quantized models
- During peak (prayer times, Ramadan), scale up to GPU nodes
- Estimated 3-5x cost reduction vs. keeping GPUs running 24/7

---

## 9. Implementation Roadmap

### Phase 1: MVP (Months 1-3)
- Fine-tune Whisper-large-v3 on Quran data with diacritized output (fastest path to working prototype)
- Build chunked streaming with faster-whisper + WebSocket
- Implement basic word-level position matching using edit distance against Quran text
- Simple error detection: word substitution and missing word detection
- Basic mobile app with microphone streaming

### Phase 2: Production ASR (Months 3-6)
- Train WeNet Conformer U2 model on Quran data with character-level diacritized output
- Export to ONNX, deploy via Sherpa-ONNX Go WebSocket server on K8s
- Add language router for Arabic/English switching
- Implement harakat-level error detection (compare predicted vs. reference diacritics per word)
- Build evaluation pipeline with all metrics from Section 6

### Phase 3: Advanced Tajweed (Months 6-9)
- Implement Quran Phonetic Script (QPS) approach for phoneme-level tajweed detection
- Add GOP (Goodness of Pronunciation) scoring for makhraj assessment
- Build tajweed rule engine (madd types, ghunnah, idgham, ikhfa, iqlab, qalqalah)
- Generate detailed mistake reports with specific tajweed rule violations
- A/B test against Phase 2 system

### Phase 4: Scale & Polish (Months 9-12)
- Optimize model size/latency with INT8/INT4 quantization
- Add support for different qira'at (Warsh, Qalun, etc.)
- Implement spaced repetition for weak ayat
- Build analytics dashboard for users (mistake heatmaps, progress tracking)
- Load testing for subscription tier scaling
- Optional: On-device model via Sherpa-ONNX mobile SDK for offline mode

---

## 10. Key Technical Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Harakat accuracy insufficient for reliable error detection | High — false positives destroy user trust, false negatives miss real mistakes | Start with word-level matching (high confidence), gradually add harakat checking with configurable sensitivity levels. Let users adjust strictness. |
| Streaming latency too high for real-time feel | Medium — users expect instant word highlighting | Use CTC-based models (WeNet U2 CTC first pass) which emit tokens incrementally. Avoid encoder-decoder models for the streaming path. |
| Dialect/accent variation in reciters | Medium — model trained on one accent fails on others | Train on diverse reciter datasets (OpenSLR-132 has multiple reciters). Augment with noise, speed perturbation, pitch shifting. |
| Scaling GPU costs for subscription model | High — GPU inference is expensive | Use Sherpa-ONNX INT8 quantization on CPU for low-traffic periods. Scale GPU only during peak. L4 GPUs are cost-effective for inference. |
| Arabic/English code-switching detection | Low-medium — misrouting causes bad UX | Fine-tune a lightweight language ID model on Arabic recitation vs. English speech. Use silence/pause as switching boundary. |
| WeNet community smaller than NeMo for Arabic | Medium — less Arabic-specific support | WeNet uses identical Conformer architecture. Arabic-specific work is in the data/tokenizer, not the framework. NeMo Arabic fine-tuning examples transfer directly. |

---

## 11. Final Recommendation Summary

| Phase | Training Framework | Deployment Runtime | Quran ASR Model | English Model |
|---|---|---|---|---|
| **Phase 1 (MVP)** | HuggingFace (Whisper fine-tune) | faster-whisper + WebSocket | Whisper-large-v3 fine-tuned | Same model (multilingual) |
| **Phase 2 (Production)** | **WeNet** | **Sherpa-ONNX Go WebSocket** | Conformer U2 (char-level, diacritized) | Whisper-small via faster-whisper |
| **Phase 3 (Tajweed)** | WeNet + wav2vec2-BERT for QPS | Sherpa-ONNX + custom tajweed engine | U2 + phonetic error detector | Same |
| **Phase 4 (Scale)** | Same | Same + INT8 CPU scaling | Quantized U2 | Same |

**North Star Metric:** False Acceptance Rate (FA) — a Quran app that approves incorrect recitation is worse than having no app at all. Design every decision around minimizing FA while keeping False Rejection (FR) tolerable.
