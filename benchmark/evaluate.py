"""Run ASR model inference and save predictions."""

import json
from pathlib import Path

from tqdm import tqdm


def _load_manifest(data_dir: str) -> list[dict]:
    """Load manifest.jsonl from data_dir, return list of dicts."""
    manifest_path = Path(data_dir) / "manifest.jsonl"
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


def run_evaluate(model: str, data_dir: str, output_dir: str, batch_size: int) -> None:
    """Dispatch evaluation to the appropriate model runner."""
    manifest = _load_manifest(data_dir)
    out = Path(output_dir)

    if model == "nemo":
        _run_nemo(manifest, out / "nemo_predictions.jsonl", batch_size)
    elif model == "funasr":
        _run_funasr(manifest, out / "funasr_predictions.jsonl", batch_size)
