# Fine-Tuning ASR Models for Quran Recitation Error Detection

## Technical Deep-Dive: Models, Methods, and Tashkeel Error Detection

---

## Table of Contents

1. [Fine-tuning NVIDIA FastConformer for Quran](#1-fine-tuning-nvidia-fastconformer-for-quran)
2. [Fine-tuning FunASR Models for Arabic/Quran](#2-fine-tuning-funasr-models-for-arabicquran)
3. [Error Detection Approaches](#3-error-detection-approaches)
4. [Training Data Strategy for Error Detection](#4-training-data-strategy-for-error-detection)
5. [Architecture Options](#5-architecture-options)
6. [State of the Art](#6-state-of-the-art)

---

## 1. Fine-tuning NVIDIA FastConformer for Quran

### 1.1 Model Overview

The target model `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` is specifically designed for Arabic with diacritical marks:

| Property | Value |
|----------|-------|
| Architecture | FastConformer (optimized Conformer with 8x depthwise-separable conv downsampling) |
| Parameters | ~115M |
| Training Loss | Hybrid: Transducer (RNNT, default) + CTC |
| Model Class | `EncDecHybridRNNTCTCBPEModel` |
| Tokenizer | Google SentencePiece (Unigram) |
| Vocabulary Size | 1,024 BPE tokens |
| Training Data | ~1,100 hours (MASC 690h, MCV 65h, FLEURS 5h, TarteelAI EveryAyah 390h) |
| Training Epochs | 200 |
| Framework | NeMo 2.0.0 |
| License | CC-BY-4.0 |

**Key distinction:** The `pcd` variant outputs Arabic text **with** diacritical marks (fatha, kasra, damma, shadda, sukun, tanween). The `pc` variant outputs only punctuation/capitalization. For Quran error detection, `pcd` is essential.

**Baseline Performance (Transducer decoder, greedy):**

| Dataset | WER (%) |
|---------|---------|
| MASC Test | 16.67 |
| MCV Test | 25.60 |
| FLEURS Test | 12.94 |
| EveryAyah Test | 6.55 |

The 6.55% WER on EveryAyah (Quranic data) is already strong because the model was partially trained on TarteelAI EveryAyah data.

### 1.2 NeMo Manifest Format

NeMo uses JSONL manifest files. Each line is a JSON object:

```json
{"audio_filepath": "/data/quran/al_fatiha_001.wav", "duration": 4.2, "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"}
```

Required fields:
- `audio_filepath`: Absolute or relative path to a mono-channel, 16kHz WAV file
- `duration`: Duration in seconds (float)
- `text`: Transcription with full diacritization (for the pcd model)

For the hybrid model, you need separate manifest files for train, validation, and test splits.

### 1.3 Tokenizer Considerations for Arabic with Diacritics

**Critical issue:** NeMo has had a documented regression (Issue #3795) where Arabic diacritics are lost during training. The root cause is SentencePiece's default Unicode normalization (`nmt_nfkc_cf`), which can compose or strip combining characters like Arabic diacritics.

**Workarounds and fixes:**

1. **Use `--no_lower_case` flag** when building the tokenizer to prevent NFKD normalization:
   ```bash
   python scripts/tokenizers/process_asr_text_tokenizer.py \
     --manifest=train_manifest.json \
     --vocab_size=1024 \
     --data_root=./tokenizer_spe_unigram_v1024_ar \
     --tokenizer="spe" \
     --spe_type="unigram" \
     --spe_character_coverage=1.0 \
     --no_lower_case
   ```

2. **Set normalization rule to `identity`** in SentencePiece to preserve raw character-level inputs (PR #13006 added this capability):
   ```python
   # When building tokenizer, disable normalization
   spm.SentencePieceTrainer.train(
       input=text_file,
       model_prefix=prefix,
       normalization_rule_name='identity',  # Preserve diacritics exactly
       ...
   )
   ```

3. **Verify round-trip encoding** before training:
   ```python
   test_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
   ids = model.tokenizer.text_to_ids(test_text)
   recovered = model.tokenizer.ids_to_text(ids)
   assert test_text == recovered, f"Diacritics lost: {recovered}"
   ```

4. **Force UTF-8 locale** during training: `LC_ALL=en_US.UTF-8 python train.py ...`

**When fine-tuning the existing pcd model:** The pre-trained model already has a tokenizer that supports diacritics (vocab size 1024). If your Quranic text uses the same diacritic characters, you can keep the existing tokenizer. If you need to expand the vocabulary (e.g., for specialized tajweed notation), you must rebuild the tokenizer and reinitialize the decoder head.

### 1.4 Training Configuration

**Loading the pre-trained model:**
```python
import nemo.collections.asr as nemo_asr
import copy
from omegaconf import open_dict

asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)
```

**Configuring data loaders:**
```python
cfg = copy.deepcopy(asr_model.cfg)

with open_dict(cfg):
    # Training data
    cfg.train_ds.manifest_filepath = "/data/quran_train_manifest.json"
    cfg.train_ds.batch_size = 32
    cfg.train_ds.num_workers = 8
    cfg.train_ds.pin_memory = True
    cfg.train_ds.trim_silence = True
    cfg.train_ds.normalize_transcripts = False  # Critical for Arabic diacritics

    # Validation data
    cfg.validation_ds.manifest_filepath = "/data/quran_val_manifest.json"
    cfg.validation_ds.batch_size = 8
    cfg.validation_ds.num_workers = 8

asr_model.setup_training_data(cfg.train_ds)
asr_model.setup_multiple_validation_data(cfg.validation_ds)
```

**Optimizer and learning rate:**
```python
with open_dict(cfg):
    cfg.optim.lr = 1e-4          # Lower than from-scratch (1e-3). Fine-tuning needs smaller LR.
    cfg.optim.betas = [0.95, 0.5]
    cfg.optim.weight_decay = 1e-3
    cfg.optim.sched.warmup_steps = None
    cfg.optim.sched.warmup_ratio = 0.05   # 5% warmup
    cfg.optim.sched.min_lr = 1e-6
```

**Recommended training parameters for Quran fine-tuning:**

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Learning rate | 1e-4 to 5e-5 | Small LR to avoid catastrophic forgetting of diacritics knowledge |
| Warmup ratio | 0.05 - 0.10 | 5-10% of total training |
| Batch size | 16-32 | Depends on GPU memory. Use bf16 precision to double effective batch |
| Epochs | 20-50 | Quranic text is highly structured; overfitting risk is real |
| Precision | bf16 (if supported) or fp16 | Allows larger batches, faster training |
| SpecAugment | freq_masks=2, freq_width=25, time_masks=10, time_width=0.05 | Regularization against overfitting |

**Encoder freezing strategy:**
```python
# Option A: Freeze encoder, unfreeze batch normalization (low-resource)
def enable_bn_se(module):
    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        for param in module.parameters():
            param.requires_grad = True

asr_model.encoder.freeze()
asr_model.encoder.apply(enable_bn_se)

# Option B: Full fine-tuning (recommended if you have 100+ hours of Quran data)
# Don't freeze anything — train the whole model
```

**Training script (command-line):**
```bash
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  --config-path=conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe \
  +init_from_pretrained_model="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0" \
  model.train_ds.manifest_filepath="/data/quran_train.json" \
  model.validation_ds.manifest_filepath="/data/quran_val.json" \
  model.optim.lr=0.0001 \
  model.optim.sched.warmup_ratio=0.05 \
  trainer.max_epochs=50 \
  trainer.precision=bf16 \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  exp_manager.name="fastconformer_quran_finetune"
```

### 1.5 Preserving/Improving Tashkeel Prediction During Fine-Tuning

The core risk during fine-tuning is that the model "forgets" how to predict diacritics if:
- Training data has inconsistent or missing diacritization
- The tokenizer normalizes away diacritics
- The learning rate is too high, destroying pre-trained weights

**Strategies to preserve tashkeel quality:**

1. **Use fully diacritized Quran text as ground truth.** The Quran has a canonical fully-vowelized text. Every training sample must have complete tashkeel (fatha, kasra, damma, sukun, shadda, tanween on every applicable letter).

2. **Validate diacritization consistency** in your training data:
   ```python
   import unicodedata

   ARABIC_DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0670')

   def count_diacritics(text):
       return sum(1 for ch in text if ch in ARABIC_DIACRITICS)

   def diacritic_ratio(text):
       letters = sum(1 for ch in text if unicodedata.category(ch).startswith('L'))
       diacritics = count_diacritics(text)
       return diacritics / max(letters, 1)

   # Quran text should have ~0.8-1.0 diacritics per letter
   ```

3. **Add a diacritic-aware loss term** (advanced): Weight the CTC/RNNT loss higher for diacritic tokens in the vocabulary. This requires modifying the loss function to apply class weights.

4. **Monitor Diacritic Error Rate (DER)** during validation, not just WER:
   ```python
   def diacritic_error_rate(predicted, reference):
       """Compare diacritics at each position, ignoring base consonants."""
       pred_diacritics = extract_diacritics(predicted)
       ref_diacritics = extract_diacritics(reference)
       # Use edit distance on diacritic sequences
       ...
   ```

5. **Use the NeMo Forced Aligner** to verify alignment quality on validation samples (see Section 3.2).

### 1.6 Existing Resources for FastConformer Arabic Fine-Tuning

- **NeMo CTC Language Fine-tuning Tutorial**: [ASR_CTC_Language_Finetuning.ipynb](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb)
- **NeMo CommonVoice Fine-tuning Example**: [ASR_Example_CommonVoice_Finetuning.ipynb](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_Example_CommonVoice_Finetuning.ipynb)
- **NVIDIA Riva Fine-Tuning Guide**: [Riva ASR Fine-Tune with NeMo](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/asr-finetune-parakeet-nemo.html)
- **FastConformer Hybrid Config**: [fastconformer_hybrid_transducer_ctc_bpe.yaml](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe.yaml)
- **Arabic Diacritics Bug**: [NeMo Issue #3795](https://github.com/NVIDIA/NeMo/issues/3795)
- **SentencePiece Normalization Fix**: [NeMo PR #13006](https://github.com/NVIDIA/NeMo/pull/13006)
- **Tarteel's approach**: They use NeMo FastConformer + Riva for their production system (from NVIDIA case study)

---

## 2. Fine-tuning FunASR Models for Arabic/Quran

### 2.1 Fun-ASR-MLT-Nano-2512 Overview

| Property | Value |
|----------|-------|
| Architecture | Audio encoder (0.2B params) + LLM decoder (0.6B params, Qwen3-0.6B) |
| Languages | 31 languages (including Arabic) |
| Training Data | Tens of millions of hours |
| Inference | Low-latency real-time transcription |
| Framework | FunASR / PyTorch |

The Fun-ASR model family is fundamentally different from FastConformer: it is an **encoder-decoder LLM architecture** where the decoder is a full language model (Qwen3-0.6B). This means it can potentially handle Arabic diacritics through the LLM's language understanding, but fine-tuning requires different considerations.

### 2.2 Data Format and Training Pipeline

**Step 1: Prepare WAV SCP and text files**

`train_wav.scp`:
```
quran_001_001  /data/audio/al_fatiha_001.wav
quran_001_002  /data/audio/al_fatiha_002.wav
```

`train_text.txt`:
```
quran_001_001  بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
quran_001_002  الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ
```

**Step 2: Convert to JSONL format**

```bash
python tools/scp2jsonl.py \
  --scp_file train_wav.scp \
  --text_file train_text.txt \
  --output train_example.jsonl
```

The resulting JSONL uses a ChatML-like format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Speech transcription:<|startofspeech|>!/data/audio/al_fatiha_001.wav<|endofspeech|>"},
    {"role": "assistant", "content": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"}
  ],
  "speech_length": 672,
  "text_length": 15
}
```

**Step 3: Configure finetune.sh**

Key parameters in `finetune.sh`:
```bash
# Model to fine-tune
model_name_or_dir="FunAudioLLM/Fun-ASR-MLT-Nano-2512"

# Training data
train_data="/data/quran/train_example.jsonl"

# Components to unfreeze (set freeze=false for what you want to train)
audio_encoder_conf.freeze=false   # Unfreeze for <5000h data
audio_adaptor_conf.freeze=false   # Always unfreeze for fine-tuning
llm_conf.freeze=true              # Keep LLM frozen unless you have >10000h
```

**Recommended freeze strategy by data size:**

| Data Size | audio_encoder | audio_adaptor | LLM |
|-----------|--------------|---------------|-----|
| < 1,000h  | Frozen       | **Unfrozen**  | Frozen |
| < 5,000h  | **Unfrozen** | **Unfrozen**  | Frozen |
| > 10,000h | **Unfrozen** | **Unfrozen**  | **Unfrozen** |

For Quran data (likely 100-1000 hours), unfreeze only `audio_adaptor`.

**Step 4: Run training**
```bash
pip install funasr>=1.3.0
bash finetune.sh
```

**Step 5: Evaluate**
```bash
python decode.py \
  ++model_dir=/path/to/finetuned_model \
  ++scp_file=data/val_wav.scp \
  ++output_file=output.txt

# Calculate WER
python tools/whisper_mix_normalize.py data/val_text.txt data/val_norm.txt
python tools/whisper_mix_normalize.py output.txt output_norm.txt
compute-wer data/val_norm.txt output_norm.txt cer.txt
```

### 2.3 SenseVoice: Not Suitable for Arabic/Quran

**SenseVoice does NOT explicitly support Arabic.** Its documented languages are Mandarin, Cantonese, English, Japanese, and Korean. While it claims "50+ languages," Arabic is not listed as a primary language, and there is no evidence of Arabic-specific training or evaluation.

SenseVoice's architecture (non-autoregressive CTC-based encoder-only) is fast but lacks the language modeling capacity needed for complex diacritization tasks. It is designed for emotion recognition, event detection, and language ID alongside ASR -- capabilities irrelevant for Quran recitation.

**Verdict:** Do not use SenseVoice for Quran. Use Fun-ASR-MLT-Nano or the FunASR Paraformer instead.

### 2.4 Paraformer for Arabic

Paraformer is a non-autoregressive end-to-end ASR model with advantages in accuracy and efficiency. However, the primary Paraformer models (`funasr/paraformer-zh`) are Chinese-focused. There is no publicly available Arabic Paraformer.

The FunASR toolkit does support fine-tuning Paraformer on custom data using similar SCP/text input format, but for Arabic Quran work, **Fun-ASR-MLT-Nano-2512 is the better choice** because it already supports Arabic in its 31-language capability.

### 2.5 FunASR Fine-Tuning Documentation

- **Fun-ASR Fine-tuning Guide**: [finetune.md](https://github.com/FunAudioLLM/Fun-ASR/blob/main/docs/finetune.md)
- **SenseVoice Fine-tuning Script**: [finetune.sh](https://github.com/FunAudioLLM/SenseVoice/blob/main/finetune.sh)
- **FunASR Main Repository**: [github.com/modelscope/FunASR](https://github.com/modelscope/FunASR)
- **Fun-ASR Technical Report**: [arxiv.org/html/2509.12508v4](https://arxiv.org/html/2509.12508v4)

---

## 3. Error Detection Approaches

### 3.1 ASR-Based Error Detection (Compare ASR Output to Ground Truth)

This is the most straightforward approach: transcribe the recitation with a diacritics-aware ASR model, then compare the output to the known correct Quranic text.

**How it works:**

```
Audio Input → ASR Model (with tashkeel) → Predicted Text → Alignment → Compare → Errors
                                            Ground Truth Quran Text ↗
```

**Detecting error types via text comparison:**

| Error Type | Detection Method |
|------------|-----------------|
| **Wrong word** | Word-level edit distance (Levenshtein). A substitution in the alignment. |
| **Missing word** | A deletion in the alignment — word present in ground truth but absent in prediction. |
| **Extra word** | An insertion in the alignment — word in prediction not in ground truth. |
| **Tashkeel error** | Strip base consonants, compare only diacritic sequences at each position. Base text matches but diacritics differ. |

**Algorithm for tashkeel-specific error detection:**

```python
import re

DIACRITICS = re.compile(r'[\u064B-\u0652\u0670]')

def split_base_and_diacritics(text):
    """Split Arabic text into (base_consonants, diacritic_positions)."""
    result = []
    current_base = None
    current_diacritics = []
    for ch in text:
        if DIACRITICS.match(ch):
            current_diacritics.append(ch)
        else:
            if current_base is not None:
                result.append((current_base, tuple(current_diacritics)))
            current_base = ch
            current_diacritics = []
    if current_base is not None:
        result.append((current_base, tuple(current_diacritics)))
    return result

def detect_tashkeel_errors(predicted, reference):
    """Compare predicted vs reference at the harakat level."""
    pred_pairs = split_base_and_diacritics(predicted)
    ref_pairs = split_base_and_diacritics(reference)

    errors = []
    # Align using word-level then character-level alignment
    for i, (pred, ref) in enumerate(zip(pred_pairs, ref_pairs)):
        pred_base, pred_diacs = pred
        ref_base, ref_diacs = ref
        if pred_base == ref_base and pred_diacs != ref_diacs:
            errors.append({
                'position': i,
                'letter': ref_base,
                'expected_harakat': ref_diacs,
                'predicted_harakat': pred_diacs,
                'type': 'tashkeel_error'
            })
        elif pred_base != ref_base:
            errors.append({
                'position': i,
                'expected': ref_base,
                'predicted': pred_base,
                'type': 'letter_error'
            })
    return errors
```

**Limitations of pure ASR comparison:**
- ASR models have inherent WER. Even the best Quran ASR (Tarteel at ~4% WER) makes mistakes, which would show up as false positives.
- The model may "hallucinate" the correct text if it has memorized the Quran, masking actual pronunciation errors.
- Subtle tashkeel errors (e.g., fatha vs. no vowel) may not be reflected in acoustic differences that the ASR can capture.

### 3.2 Forced Alignment Approach

Forced alignment uses CTC/attention models to align known text to audio at the phoneme/word level, then analyzes alignment quality to detect mispronunciations.

**NeMo Forced Aligner (NFA):**

NFA uses CTC-based ASR models to generate token-, word-, and segment-level timestamps. It works via Viterbi decoding on CTC log-probabilities.

```bash
python tools/nfa/align.py \
  pretrained_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0" \
  manifest_filepath=recitation_manifest.json \
  output_dir=./alignment_output
```

The manifest must contain the reference text (the correct Quran verse). NFA will output:
- CTM files with `<utt_id> 1 <start_time> <duration> <token>` format
- Confidence scores for each aligned token

**Key point:** NFA currently supports only CTC models or Hybrid CTC-Transducer models in CTC mode. The FastConformer pcd model qualifies.

**Limitations:** NFA is designed for temporal alignment, not pronunciation scoring. It does not natively compute GOP scores or flag mispronunciations. You would need to extract the CTC log-probabilities and compute GOP scores yourself.

**Montreal Forced Aligner (MFA):**

MFA is a standalone forced alignment tool that uses Kaldi under the hood. It supports Arabic acoustic models but requires phoneme-level dictionaries. For Quran, you would need to create a pronunciation dictionary mapping each Arabic word (with diacritics) to its phoneme sequence. The Quranic Phonemizer (see Section 6) can provide this mapping.

**Kaldi Forced Alignment:**

Kaldi is the traditional tool for forced alignment and GOP scoring. It provides the most control but requires the most setup. The classic GOP pipeline is:
1. Forced-align the reference transcription to the audio
2. For each phoneme segment, compute log-likelihood ratio: P(correct phoneme) / P(best alternative phoneme)
3. Threshold the ratio to flag mispronunciations

### 3.3 GOP (Goodness of Pronunciation) Scoring

GOP is the standard metric for pronunciation assessment. It measures how well the expected phone matches what was actually pronounced.

**Formula:**
```
GOP(p) = log P(p | o_t) - max_{q ≠ p} log P(q | o_t)
```

Where:
- `p` is the expected (canonical) phoneme
- `q` is any alternative phoneme
- `o_t` is the acoustic observation at time `t`

**Implementation with CTC models:**

```python
def compute_gop_from_ctc(ctc_logprobs, forced_alignment, phoneme_map):
    """
    ctc_logprobs: (T, V) tensor of CTC log-probabilities
    forced_alignment: list of (phoneme_id, start_frame, end_frame)
    phoneme_map: dict mapping phoneme_id to vocabulary index
    """
    gop_scores = []
    for phoneme_id, start, end in forced_alignment:
        target_idx = phoneme_map[phoneme_id]
        segment_logprobs = ctc_logprobs[start:end]  # (segment_len, V)

        # Average log-prob of target phoneme over segment
        target_score = segment_logprobs[:, target_idx].mean()

        # Average log-prob of best non-target phoneme
        mask = torch.ones(segment_logprobs.shape[1], dtype=torch.bool)
        mask[target_idx] = False
        best_other = segment_logprobs[:, mask].max(dim=1).values.mean()

        gop = target_score - best_other
        gop_scores.append({
            'phoneme': phoneme_id,
            'gop_score': gop.item(),
            'start': start,
            'end': end
        })
    return gop_scores
```

**Recent advances in GOP:**

A "Segmentation-Free Goodness of Pronunciation" approach (arxiv:2507.16838) eliminates the need for forced alignment entirely, computing GOP directly from CTC logits. This is promising for real-time Quran assessment.

A 17MB pronunciation scorer using CTC forced alignment + GOP scoring + ensemble heads, built on a quantized NeMo Citrinet-256 backbone, has achieved phone-level accuracy exceeding human inter-annotator agreement.

### 3.4 Dedicated Pronunciation Assessment Models

**Microsoft Pronunciation Assessment API:**

Microsoft's approach uses a specialized speech-to-text model trained on 100,000+ hours from native speakers. It evaluates:
- Accuracy (phoneme-level)
- Fluency (rhythm and timing)
- Completeness (missing content)
- Prosody (intonation patterns)

It returns scores at phoneme, syllable, word, sentence, and article levels, with timestamp-linked error annotations. However, it supports primarily English and a few other languages -- **Arabic support is limited**, and it has no Quran-specific capabilities.

**wav2vec2-Based Pronunciation Scoring:**

This is the most promising approach for Quran. Recent papers show:

1. Fine-tune wav2vec2 for phoneme-level binary classification (correct/mispronounced)
2. Use CTC loss on phoneme sequences
3. Extract per-phoneme confidence scores from CTC logits

**Existing Quran-specific models on HuggingFace:**

| Model | Architecture | Purpose |
|-------|-------------|---------|
| `tarteel-ai/whisper-base-ar-quran` | Whisper (fine-tuned) | Quran ASR, WER ~5.75% |
| `KheemP/whisper-base-quran-lora` | Whisper + LoRA | Diacritic-sensitive Quran ASR, WER ~5.98% |
| `Nuwaisir/Quran_speech_recognizer` | wav2vec2-large-xlsr-53 | Quran position identification |
| `TBOGamer22/wav2vec2-quran-phonetics` | wav2vec2 | Phonetic transcription of Quran |
| `basharalrfooh/Fine-Tashkeel` | ByT5 | Text diacritization (DER 0.95, WER 2.49) |

### 3.5 Tashkeel-Specific Error Detection

Detecting tashkeel errors requires a different approach than detecting word-level errors because:
- The base consonant is correct, only the vowel mark is wrong
- Multiple valid tashkeel patterns may exist for the same word in different contexts
- Some diacritics are acoustically subtle (e.g., difference between short vowels)

**Position-Level Alignment for Harakat Comparison:**

1. **Normalize both texts** to the same encoding (NFC or NFD). Arabic combining marks can be encoded differently.
2. **Extract character-diacritic pairs** from both predicted and reference texts.
3. **Align at the character level** using edit distance.
4. **Compare diacritics** only at positions where base characters match.

```python
import unicodedata

def normalize_arabic(text):
    """Normalize to NFC form for consistent comparison."""
    return unicodedata.normalize('NFC', text)

def extract_char_diacritic_pairs(text):
    """Extract (base_char, [diacritics]) pairs."""
    text = normalize_arabic(text)
    pairs = []
    i = 0
    while i < len(text):
        ch = text[i]
        diacritics = []
        i += 1
        # Collect all following diacritics
        while i < len(text) and unicodedata.category(text[i]) == 'Mn':
            diacritics.append(text[i])
            i += 1
        pairs.append((ch, tuple(sorted(diacritics))))
    return pairs

def compare_tashkeel(predicted, reference):
    """
    Compare at harakat level.
    Returns list of errors with position, type, expected, and actual.
    """
    pred_pairs = extract_char_diacritic_pairs(predicted)
    ref_pairs = extract_char_diacritic_pairs(reference)

    errors = []
    # Use sequence alignment (e.g., Needleman-Wunsch) on base characters
    # Then compare diacritics at aligned positions
    # ... (alignment code omitted for brevity)

    return errors
```

**Handling edge cases:**
- **Shadda + vowel**: A letter can have both shadda (gemination) and a vowel. These are two separate diacritics on the same letter. Ensure your comparison handles multi-diacritic positions.
- **Tanween**: Tanween (double vowels like fathatain, kasratain, dammatain) occupies a single unicode codepoint but represents a nasal vowel. It must be compared as a unit.
- **Alef with hamza**: Hamza placement above or below alef affects meaning. This is a letter-level difference, not a diacritic difference.
- **Waqf marks**: End-of-verse pause marks are not tashkeel and should be excluded from comparison.

---

## 4. Training Data Strategy for Error Detection

### 4.1 Available Datasets

| Dataset | Size | Content | Labels |
|---------|------|---------|--------|
| **obadx/mualem-recitations-annotated** | 890h | Expert Quran recitations | Phoneme-level annotations, fully diacritized |
| **RetaSy/quranic_audio_dataset** | ~7000 samples | Non-native speaker recitations | correct/incorrect + error categories |
| **tarteel-ai/everyayah** | ~390h | Professional reciters (per verse) | Verse-level diacritized text |
| **Salama1429/tarteel-ai-everyayah-Quran** | Full Quran | Multiple reciters, all 6,236 verses | Verse-level text |
| **QuranMB.v1** | ~2.2h | 18 speakers, deliberate errors | Phoneme-level mispronunciation labels |
| **Common Voice Arabic** | 65h+ | General Arabic speech | Transcriptions (no diacritics) |

### 4.2 Creating Training Data with Intentional Errors

For error detection, you need both correct and incorrect examples. Approaches:

**a) Use the RetaSy dataset directly:**

The RetaSy Quranic Audio Dataset contains ~7000 recitations from 1287 non-Arabic speakers across 11 countries, with 1166 annotated in six categories. The `final_label` field indicates correct/incorrect classification with 0.89 algorithm-expert agreement.

```python
from datasets import load_dataset

ds = load_dataset("RetaSy/quranic_audio_dataset")
# Each sample has: audio, aya (verse text), duration_ms, final_label, golden (expert-labeled?)
```

**b) Synthetic error generation via text perturbation + TTS:**

The QuranMB researchers generated a 52-hour synthetic dataset:
1. Start with fully diacritized Quran text
2. For each sample, randomly select 4 characters and/or diacritics
3. Apply modifications based on a **confusion matrix** derived from common mispronunciation patterns
4. Synthesize audio using TTS models trained on vowelized Arabic text
5. Label as "error" with phoneme-level annotations

**Confusion matrix approach:**
```python
# Common Arabic mispronunciation pairs (based on articulatory similarity)
CONFUSION_MAP = {
    'ص': ['س', 'ض'],           # Saad ↔ Seen, Daad
    'ض': ['ظ', 'د', 'ص'],     # Daad ↔ Thaa, Daal, Saad
    'ط': ['ت', 'ظ'],           # Taa ↔ Taa, Thaa
    'ظ': ['ذ', 'ض', 'ز'],     # Thaa ↔ Dhaal, Daad, Zayn
    'ق': ['ك', 'غ'],           # Qaaf ↔ Kaaf, Ghayn
    'ع': ['أ', 'ح'],           # Ayn ↔ Hamza, Haa
    'ح': ['ه', 'خ'],           # Haa ↔ Haa, Khaa
    'خ': ['ح', 'غ'],           # Khaa ↔ Haa, Ghayn
    'ذ': ['ز', 'ظ'],           # Dhaal ↔ Zayn, Thaa
    'ث': ['س', 'ت'],           # Thaa ↔ Seen, Taa
    # Diacritic confusions
    '\u064E': ['\u064F', '\u0650', '\u0652'],  # Fatha ↔ Damma, Kasra, Sukun
    '\u064F': ['\u064E', '\u0650'],              # Damma ↔ Fatha, Kasra
    '\u0650': ['\u064E', '\u064F'],              # Kasra ↔ Fatha, Damma
}
```

**c) SpeechBlender approach:**

SpeechBlender is a fine-grained data augmentation pipeline for generating mispronunciation errors that overcomes data scarcity. It works at the phoneme level:
1. Get phoneme transcription of correct audio
2. Perturb phonemes using confusion rules
3. Synthesize the perturbed phoneme sequence back to audio
4. The result sounds like a mispronunciation of the original

**d) Audio-level augmentation:**

- Speed perturbation (0.9x, 1.1x) to simulate different recitation speeds
- Noise addition to simulate real recording conditions
- Room impulse response convolution for reverb simulation
- Pitch shifting for different voice characteristics

### 4.3 Data Split Strategy

For Quran error detection, a recommended split:

| Split | Purpose | Composition |
|-------|---------|-------------|
| Train | Model training | 80% correct (professional reciters) + synthetic errors |
| Validation | Hyperparameter tuning | Mix of correct and real errors (RetaSy subset) |
| Test | Final evaluation | QuranMB.v1 + held-out RetaSy samples |

**Class balance:** Pronunciation assessment datasets are inherently imbalanced (most phonemes are correct). Use:
- Focal loss to handle class imbalance
- Oversampling of error examples
- Stratified sampling by error type

---

## 5. Architecture Options

### 5.1 Option A: Fine-tune Existing ASR + Post-Process Comparison

**Architecture:**
```
Audio → Fine-tuned ASR (FastConformer pcd) → Predicted text with tashkeel
                                                      ↓
Ground Truth Quran Text → Text Alignment Engine → Error Report
```

**Pros:**
- Simplest to implement
- Leverages well-trained ASR model
- No custom model training needed for error detection

**Cons:**
- Error detection quality is bounded by ASR quality
- ASR may memorize Quran text, masking real errors
- Cannot detect subtle phonetic errors that don't change the text

**Implementation effort:** Low (weeks)

### 5.2 Option B: Binary Classifier on ASR Embeddings

**Architecture:**
```
Audio → ASR Encoder (frozen) → Frame-level Embeddings
                                       ↓
Reference Phonemes → Forced Alignment → Aligned Segments
                                       ↓
                          Per-phoneme Embedding Extraction
                                       ↓
                          Binary Classifier (correct/incorrect)
                                       ↓
                               Error Report
```

**Pros:**
- Uses powerful pre-trained representations
- Can detect subtle pronunciation differences
- Separate classifier is lightweight and fast to train

**Cons:**
- Requires good forced alignment
- Two-stage pipeline adds latency
- Need labeled pronunciation data for classifier training

**Implementation effort:** Medium (1-2 months)

### 5.3 Option C: Multi-Task Learning (ASR + Error Detection Jointly)

**Architecture:**
```
Audio → Shared Encoder (FastConformer/wav2vec2)
              ↓                    ↓
     ASR Head (CTC/RNNT)   Error Detection Head (binary per-phoneme)
              ↓                    ↓
     Transcription         Pronunciation Scores
```

Recent research (ResearchGate: "A Joint Model for Pronunciation Assessment and Mispronunciation Detection") shows that joint training of ASR and MDD improves both tasks because they are "highly correlated."

**Joint loss function:**
```python
loss = alpha * ctc_loss + beta * error_detection_loss
# Where error_detection_loss uses Focal loss for class imbalance
```

**Pros:**
- Shared encoder captures both transcription and pronunciation features
- Joint training improves both tasks
- End-to-end trainable

**Cons:**
- Most complex to implement
- Requires phoneme-level error labels for training
- Balancing two loss terms requires careful tuning

**Implementation effort:** High (2-3 months)

### 5.4 Option D: Two-Stage Pipeline (ASR → Alignment → Scoring)

**Architecture:**
```
Stage 1: ASR
Audio → FastConformer → CTC Logits → Transcription + Token timestamps

Stage 2: Scoring
CTC Logits + Reference Phonemes → GOP Scoring → Per-phoneme scores
                                                      ↓
                                              Threshold → Error flags
```

This is the approach used by the QuranMB researchers and is closest to the traditional CAPT (Computer-Aided Pronunciation Training) pipeline.

**Pros:**
- Well-understood pipeline with decades of research
- GOP scoring is interpretable
- Can use existing ASR model without modification

**Cons:**
- GOP requires careful threshold calibration
- Forced alignment errors propagate to scoring
- May not capture all error types (esp. prosodic)

**Implementation effort:** Medium (1-2 months)

### 5.5 Option E: The QPS Approach (from obadx/prepare-quran-dataset)

**Architecture:**
```
Audio → Multi-level CTC Model → Quran Phonetic Script (QPS) output
                                        ↓
                              Two-level comparison:
                              Level 1: Phoneme sequence comparison
                              Level 2: Sifa (articulation) comparison
                                        ↓
                              Detailed error report with tajweed rule violations
```

The Quran Phonetic Script (QPS) encodes:
- **Phoneme level**: Arabic letters with short/long vowels
- **Sifa level**: Articulation characteristics of every phoneme (e.g., hams, jahr, shidda, rakhawa)

This is the most Quran-specific approach and achieved 0.16% PER on their test set. However, the model and training code are "coming soon" as of late 2025.

### 5.6 Recommended Architecture

For a production system, I recommend a **hybrid of Options A and D**:

1. **Primary path (Option A):** Fine-tune FastConformer pcd → text comparison for word/tashkeel errors
2. **Secondary path (Option D):** Use CTC logits from the same model → GOP scoring for phoneme-level confidence
3. **Combine:** Use text comparison for high-level error detection and GOP scores to gauge severity/confidence

This approach:
- Reuses a single model (FastConformer pcd) for both paths
- Provides both text-level and phoneme-level error information
- Is implementable in 1-2 months
- Can be enhanced later with the QPS approach or multi-task learning

---

## 6. State of the Art

### 6.1 Key Papers

**1. "Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning" (arXiv:2509.00094)**
- Authors: Abdullah (obadx) et al.
- Key contribution: QPS encoding, multi-level CTC model, 850h dataset, Tasmeea verification algorithm
- Performance: 0.16% PER on test set
- Open source: [obadx.github.io/prepare-quran-dataset](https://obadx.github.io/prepare-quran-dataset/)
- Status: Code/models "coming soon"

**2. "Towards a Unified Benchmark for Arabic Pronunciation Assessment: Qur'anic Recitation as Case Study" (Interspeech 2025, arXiv:2506.07722)**
- Key contribution: QuranMB.v1 benchmark, comparison of SSL models
- Architecture: 2-layer 1024-unit Bi-LSTM with CTC loss on frozen SSL features
- Best model: mHuBERT (multilingual, 147 languages)
- Performance: F1 ~30%, True Acceptance 87.35%
- Finding: "Current models achieve modest performance (F1 <= 30%)" -- the problem is far from solved

**3. "Mispronunciation Detection of Basic Quranic Recitation Rules using Deep Learning" (arXiv:2305.06429)**
- Focus on detecting specific tajweed rule violations
- Uses deep learning on audio features

**4. "Quran Recitation Recognition using End-to-End Deep Learning" (arXiv:2305.07034)**
- CNN-BiGRU encoder with CTC and character-based beam search
- End-to-end approach to recognizing Quran recitation

**5. Iqra'Eval Shared Task @ ArabicNLP 2025 (co-located with EMNLP 2025)**
- First shared task specifically for Quranic MDD
- Subtasks: error localization, detailed error diagnosis
- Top system (BAIC) used synthetic data augmentation
- Second place (Hafs2Vec) used real Quranic recitation data
- Published at: [aclanthology.org/2025.arabicnlp-sharedtasks.61](https://aclanthology.org/2025.arabicnlp-sharedtasks.61/)

**6. "Qur'anic Phonemizer: Bringing Tajweed-Aware Phonemes to Qur'anic Machine Learning" (NeurIPS 2025 Workshop)**
- 71-symbol phoneme inventory encoding tajweed rules
- Open source: [github.com/Hetchy/Quranic-Phonemizer](https://github.com/Hetchy/Quranic-Phonemizer)
- PyPI: `pip install quran-phonemizer`
- PER: 3.9% on Iqra'Eval Qur'an, 2.6% on professional reciters

### 6.2 Tarteel AI's Approach

Tarteel is the market leader (15M+ downloads). Their technical stack:

| Component | Technology |
|-----------|-----------|
| ASR Training | NVIDIA NeMo (FastConformer architecture) |
| Inference | NVIDIA Riva SDK + Triton Inference Server |
| Streaming | gRPC with configurable chunking |
| Infrastructure | CoreWeave (primary) + AWS/GCP/Linode |
| WER | ~4% (proprietary model) |
| Training Data | 75,000+ minutes proprietary |
| Open Models | `tarteel-ai/whisper-base-ar-quran` (WER 5.75%) |
| Open Datasets | `tarteel-ai/everyayah` (per-verse Quran audio) |

**Key technical decisions by Tarteel:**
- Chose NeMo FastConformer over Whisper for production (Riva enables streaming)
- Used proprietary crowdsourced data (not just professional reciters)
- Discovered that training on professional audio alone does not generalize to average users
- Achieved 22% latency reduction and 56% cost savings by migrating to CoreWeave

### 6.3 Open-Source Implementations

| Project | Link | Status |
|---------|------|--------|
| Al-Muallim Al-Qurani (obadx) | [github.com/obadx](https://github.com/obadx) | Dataset released; model/code coming soon |
| Quranic Phonemizer | [github.com/Hetchy/Quranic-Phonemizer](https://github.com/Hetchy/Quranic-Phonemizer) | Available, pip installable |
| Quran Tajweed Annotations | [github.com/cpfair/quran-tajweed](https://github.com/cpfair/quran-tajweed) | Available (JSON tajweed annotations) |
| Tarteel Whisper (Quran) | [HuggingFace](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) | Available |
| Whisper Quran LoRA | [HuggingFace](https://huggingface.co/KheemP/whisper-base-quran-lora) | Available (WER ~5.98%) |
| wav2vec2 Quran Phonetics | [HuggingFace](https://huggingface.co/TBOGamer22/wav2vec2-quran-phonetics) | Available |
| ArTST v1.5 | [github.com/mbzuai-nlp/ArTST](https://github.com/mbzuai-nlp/ArTST) | MSA with diacritics, 17 checkpoints |
| IqraEval Shared Task | [HuggingFace Space](https://huggingface.co/spaces/IqraEval/SharedTask_ArabicNLP2025) | Benchmark available |

### 6.4 Key Findings and Open Challenges

1. **The problem is not solved.** Best F1 on QuranMB.v1 is ~30%. There is enormous room for improvement.

2. **Multilingual SSL models outperform monolingual ones.** mHuBERT (trained on 147 languages) beats English-only wav2vec2/HuBERT for Arabic pronunciation assessment.

3. **Synthetic data helps but is not sufficient.** The QuranMB study found that combining real and synthetic data (131 hours total) gives the best results.

4. **Tashkeel preservation in ASR is fragile.** NeMo's SentencePiece tokenizer has a known bug that strips diacritics. Careful Unicode normalization handling is essential.

5. **The QPS approach is the most promising.** By encoding tajweed rules directly into the phoneme set, the multi-level CTC model can detect errors at the articulation level, not just the letter level.

6. **No production-grade open-source system exists yet.** Tarteel's system is proprietary. The obadx project is the most complete open-source effort but models are not yet released.

---

## Sources

### Models
- [nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)
- [tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran)
- [FunAudioLLM/Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
- [KheemP/whisper-base-quran-lora](https://huggingface.co/KheemP/whisper-base-quran-lora)
- [TBOGamer22/wav2vec2-quran-phonetics](https://huggingface.co/TBOGamer22/wav2vec2-quran-phonetics)
- [basharalrfooh/Fine-Tashkeel](https://huggingface.co/basharalrfooh/Fine-Tashkeel)

### Papers
- [Automatic Pronunciation Error Detection and Correction of the Holy Quran (arXiv:2509.00094)](https://arxiv.org/abs/2509.00094)
- [Unified Benchmark for Arabic Pronunciation Assessment (arXiv:2506.07722)](https://arxiv.org/abs/2506.07722)
- [Mispronunciation Detection of Basic Quranic Recitation Rules (arXiv:2305.06429)](https://arxiv.org/abs/2305.06429)
- [Quran Recitation Recognition using End-to-End Deep Learning (arXiv:2305.07034)](https://arxiv.org/abs/2305.07034)
- [Segmentation-Free GOP (arXiv:2507.16838)](https://arxiv.org/html/2507.16838)
- [Iqra'Eval Shared Task](https://aclanthology.org/2025.arabicnlp-sharedtasks.61/)
- [Fun-ASR Technical Report (arXiv:2509.12508)](https://arxiv.org/html/2509.12508v4)

### Documentation
- [NeMo ASR Configuration](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html)
- [NeMo Forced Aligner](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html)
- [NeMo CTC Language Fine-tuning Tutorial](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb)
- [Fun-ASR Fine-tuning Guide](https://github.com/FunAudioLLM/Fun-ASR/blob/main/docs/finetune.md)
- [SenseVoice Repository](https://github.com/FunAudioLLM/SenseVoice)
- [FunASR Repository](https://github.com/modelscope/FunASR)
- [Microsoft Pronunciation Assessment](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-pronunciation-assessment)
- [NVIDIA Riva Fine-Tuning](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/asr-finetune-parakeet-nemo.html)

### Tools and Datasets
- [Quranic Phonemizer](https://github.com/Hetchy/Quranic-Phonemizer)
- [Al-Muallim Al-Qurani Project](https://obadx.github.io/prepare-quran-dataset/)
- [Quran Tajweed Annotations](https://github.com/cpfair/quran-tajweed)
- [RetaSy/quranic_audio_dataset](https://huggingface.co/datasets/RetaSy/quranic_audio_dataset)
- [IqraEval Shared Task](https://huggingface.co/spaces/IqraEval/SharedTask_ArabicNLP2025)
- [NeMo Arabic Diacritics Issue #3795](https://github.com/NVIDIA/NeMo/issues/3795)
- [Tarteel NVIDIA Case Study](https://www.nvidia.com/en-us/case-studies/automating-real-time-arabic-speech-recognition/)
