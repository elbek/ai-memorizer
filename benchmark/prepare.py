"""Download and prepare Quran audio datasets into a unified manifest."""

import json
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def _save_audio(audio, path: Path) -> None:
    """Write audio dict (array + sampling_rate) to WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio["array"], audio["sampling_rate"])


def _process_tarteel(output_dir: Path, n_samples: int) -> list[dict]:
    """Process tarteel-ai/everyayah dataset."""
    ds = load_dataset("tarteel-ai/everyayah", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))

    records = []
    audio_dir = output_dir / "audio" / "tarteel"
    for i, row in enumerate(tqdm(ds, desc="tarteel")):
        path = audio_dir / f"{i:06d}.wav"
        _save_audio(row["audio"], path)
        records.append({
            "audio_path": str(path),
            "text": row["text"],
            "source": "tarteel",
            "reciter": row.get("reciter", ""),
            "surah": None,
            "ayah": None,
        })
    return records


def _process_buraaq(output_dir: Path, n_samples: int) -> list[dict]:
    """Process Buraaq/quran-md-ayahs dataset."""
    ds = load_dataset("Buraaq/quran-md-ayahs", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    records = []
    audio_dir = output_dir / "audio" / "buraaq"
    for i, row in enumerate(tqdm(ds, desc="buraaq")):
        path = audio_dir / f"{i:06d}.wav"
        _save_audio(row["audio"], path)
        records.append({
            "audio_path": str(path),
            "text": row["ayah_ar"],
            "source": "buraaq",
            "reciter": str(row.get("reciter_id", "")),
            "surah": row.get("surah_id"),
            "ayah": row.get("ayah_id"),
        })
    return records


def _process_retasy(output_dir: Path, n_samples: int) -> list[dict]:
    """Process RetaSy/quranic_audio_dataset (correct only)."""
    ds = load_dataset("RetaSy/quranic_audio_dataset", split="train")
    ds = ds.filter(lambda x: x["final_label"] == "correct")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    records = []
    audio_dir = output_dir / "audio" / "retasy"
    for i, row in enumerate(tqdm(ds, desc="retasy")):
        path = audio_dir / f"{i:06d}.wav"
        _save_audio(row["audio"], path)
        records.append({
            "audio_path": str(path),
            "text": row["Aya"],
            "source": "retasy",
            "reciter": str(row.get("reciter_id", "")),
            "surah": row.get("Surah"),
            "ayah": None,
        })
    return records


def run_prepare(output_dir: str, max_samples: int) -> None:
    """Download datasets, save audio as WAV, and write unified manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_tarteel = int(max_samples * 0.4)
    n_buraaq = int(max_samples * 0.4)
    n_retasy = max_samples - n_tarteel - n_buraaq

    records = []
    records.extend(_process_tarteel(out, n_tarteel))
    records.extend(_process_buraaq(out, n_buraaq))
    records.extend(_process_retasy(out, n_retasy))

    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    counts = {}
    for rec in records:
        counts[rec["source"]] = counts.get(rec["source"], 0) + 1

    print(f"\nDataset preparation complete: {len(records)} total samples")
    for source, count in counts.items():
        print(f"  {source}: {count}")
    print(f"Manifest written to {manifest_path}")
