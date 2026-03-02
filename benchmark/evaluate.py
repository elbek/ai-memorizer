"""Run ASR model inference and save predictions."""

import json
from pathlib import Path

from tqdm import tqdm


def _load_manifest(data_dir: str, manifest: str | None = None) -> list[dict]:
    """Load manifest JSONL. Uses explicit path if given, else data_dir/manifest.jsonl."""
    manifest_path = Path(manifest) if manifest else Path(data_dir) / "manifest.jsonl"
    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _run_nemo(manifest: list[dict], output_path: Path, batch_size: int) -> None:
    """Run NeMo FastConformer model on manifest entries."""
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    )
    model.eval()

    audio_paths = [rec["audio_path"] for rec in manifest]
    results = model.transcribe(audio=audio_paths, batch_size=batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec, result in zip(manifest, results):
            hypothesis = result.text if hasattr(result, "text") else str(result)
            prediction = {
                "audio_path": rec["audio_path"],
                "reference": rec["text"],
                "hypothesis": hypothesis,
                "source": rec["source"],
                "model": "nemo_fastconformer",
            }
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def _run_funasr(manifest: list[dict], output_path: Path, batch_size: int) -> None:
    """Run FunASR MLT Nano model on manifest entries."""
    from funasr import AutoModel

    model = AutoModel(
        model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        trust_remote_code=True,
        remote_code="./model.py",
        device="cpu",
        hub="hf",
    )

    audio_paths = [rec["audio_path"] for rec in manifest]
    predictions = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="funasr"):
        batch_paths = audio_paths[i : i + batch_size]
        results = model.generate(
            input=batch_paths, cache={}, batch_size=batch_size, language="Arabic"
        )
        for result in results:
            hypothesis = result.get("text", "") if isinstance(result, dict) else str(result)
            predictions.append(hypothesis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec, hypothesis in zip(manifest, predictions):
            prediction = {
                "audio_path": rec["audio_path"],
                "reference": rec["text"],
                "hypothesis": hypothesis,
                "source": rec["source"],
                "model": "funasr_mlt_nano",
            }
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def _run_parakeet(manifest: list[dict], output_path: Path, batch_size: int,
                  model_name: str = "nvidia/parakeet-tdt-0.6b-v2") -> None:
    """Run NeMo Parakeet TDT model on manifest entries."""
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model.eval()

    audio_paths = [rec["audio_path"] for rec in manifest]
    results = model.transcribe(audio=audio_paths, batch_size=batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec, result in zip(manifest, results):
            hypothesis = result.text if hasattr(result, "text") else str(result)
            prediction = {
                "audio_path": rec["audio_path"],
                "reference": rec["text"],
                "hypothesis": hypothesis,
                "source": rec["source"],
                "model": "parakeet_tdt",
            }
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def _model_label(model_id: str) -> str:
    """Derive a short label from a HF model_id (part after /)."""
    return model_id.split("/")[-1]


def _load_whisper(model_id: str):
    """Load a Whisper model+processor, auto-detecting LoRA adapters."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    try:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_id)
        print(f"LoRA adapter detected for {model_id}, loading base + adapter")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path
        )
        model = PeftModel.from_pretrained(base_model, model_id)
        model = model.merge_and_unload()
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
    except Exception:
        print(f"Loading {model_id} as full model")
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        processor = WhisperProcessor.from_pretrained(model_id)

    return model, processor


def _run_whisper_model(
    manifest: list[dict], output_path: Path, batch_size: int, model_id: str
) -> None:
    """Run any HF Whisper model (full or LoRA) on manifest entries."""
    import torch

    model, processor = _load_whisper(model_id)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    label = _model_label(model_id)
    print(f"Whisper ({label}) using device: {device}")

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ar", task="transcribe"
    )
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    import soundfile as sf

    # Resume support: skip already-processed samples
    done_paths = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_paths.add(rec["audio_path"])
        if done_paths:
            print(f"Resuming: {len(done_paths)} samples already done, "
                  f"{len(manifest) - len(done_paths)} remaining")

    remaining = [rec for rec in manifest if rec["audio_path"] not in done_paths]
    if not remaining:
        print(f"All {len(manifest)} samples already processed.")
        return

    with open(output_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(remaining), batch_size), desc=f"whisper/{label}"):
            batch = remaining[i : i + batch_size]
            arrays = []
            for rec in batch:
                audio, sr = sf.read(rec["audio_path"])
                arrays.append(audio)

            inputs = processor(
                arrays, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(**inputs)
            hypotheses = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for rec, hypothesis in zip(batch, hypotheses):
                prediction = {
                    "audio_path": rec["audio_path"],
                    "reference": rec["text"],
                    "hypothesis": hypothesis,
                    "source": rec["source"],
                    "model": f"whisper_{label}",
                }
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            f.flush()

            # Clear MPS cache to prevent OOM on long sequences
            if device == "mps":
                torch.mps.empty_cache()


def _run_qwen3_asr(manifest: list[dict], output_path: Path, batch_size: int,
                    model_id: str = "Qwen/Qwen3-ASR-0.6B") -> None:
    """Run Qwen3-ASR model on manifest entries."""
    import torch
    from qwen_asr import Qwen3ASRModel

    device = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    label = _model_label(model_id)
    print(f"Qwen3-ASR ({label}) using device: {device}, dtype: {dtype}")

    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=256,
    )

    # Resume support
    done_paths = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_paths.add(rec["audio_path"])
        if done_paths:
            print(f"Resuming: {len(done_paths)} samples already done, "
                  f"{len(manifest) - len(done_paths)} remaining")

    remaining = [rec for rec in manifest if rec["audio_path"] not in done_paths]
    if not remaining:
        print(f"All {len(manifest)} samples already processed.")
        return

    with open(output_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(remaining), batch_size), desc=f"qwen3/{label}"):
            batch = remaining[i : i + batch_size]
            audio_paths = [rec["audio_path"] for rec in batch]

            results = model.transcribe(audio=audio_paths, language="Arabic")

            for rec, result in zip(batch, results):
                hypothesis = result.text if hasattr(result, "text") else str(result)
                prediction = {
                    "audio_path": rec["audio_path"],
                    "reference": rec["text"],
                    "hypothesis": hypothesis,
                    "source": rec["source"],
                    "model": f"qwen3_{label}",
                }
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            f.flush()


def run_evaluate(model: str, data_dir: str, output_dir: str, batch_size: int,
                 manifest_path: str | None = None,
                 model_id: str | None = None) -> None:
    """Dispatch evaluation to the appropriate model runner."""
    manifest = _load_manifest(data_dir, manifest_path)
    out = Path(output_dir)

    if model == "nemo":
        _run_nemo(manifest, out / "nemo_predictions.jsonl", batch_size)
    elif model == "parakeet":
        _run_parakeet(manifest, out / "parakeet_predictions.jsonl", batch_size)
    elif model == "funasr":
        _run_funasr(manifest, out / "funasr_predictions.jsonl", batch_size)
    elif model == "qwen3":
        mid = model_id or "Qwen/Qwen3-ASR-0.6B"
        label = _model_label(mid)
        output_file = out / f"{label}_predictions.jsonl"
        _run_qwen3_asr(manifest, output_file, batch_size, mid)
    elif model == "whisper":
        if not model_id:
            raise ValueError("--model-id is required when --model is whisper")
        label = _model_label(model_id)
        output_file = out / f"{label}_predictions.jsonl"
        _run_whisper_model(manifest, output_file, batch_size, model_id)
