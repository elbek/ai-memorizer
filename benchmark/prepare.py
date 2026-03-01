"""Download and prepare Quran audio datasets into a unified manifest."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


def _save_audio(audio, path: Path) -> None:
    """Write audio dict (array + sampling_rate) to WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio["array"], audio["sampling_rate"])


def _process_dataset(ds, output_dir: Path, source: str, text_key: str,
                     reciter_key: str, surah_key: str | None,
                     ayah_key: str | None) -> list[dict]:
    """Process a dataset, skipping corrupt audio samples."""
    records = []
    skipped = 0
    audio_dir = output_dir / "audio" / source
    pbar = tqdm(range(len(ds)), desc=source)
    for i in pbar:
        path = audio_dir / f"{i:06d}.wav"
        try:
            row = ds[i]
            _save_audio(row["audio"], path)
        except (RuntimeError, Exception) as e:
            skipped += 1
            pbar.write(f"  skipping {source} sample {i}: {e}")
            continue
        records.append({
            "audio_path": str(path),
            "text": row[text_key],
            "source": source,
            "reciter": str(row.get(reciter_key, "")),
            "surah": row.get(surah_key) if surah_key else None,
            "ayah": row.get(ayah_key) if ayah_key else None,
        })
    if skipped:
        print(f"  {source}: skipped {skipped} corrupt samples")
    return records


def _process_tarteel(output_dir: Path, n_samples: int) -> list[dict]:
    """Process tarteel-ai/everyayah dataset."""
    ds = load_dataset("tarteel-ai/everyayah", split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.select(range(min(n_samples, len(ds))))
    return _process_dataset(ds, output_dir, "tarteel", "text", "reciter",
                            None, None)


def _process_buraaq(output_dir: Path, n_samples: int) -> list[dict]:
    """Process Buraaq/quran-md-ayahs dataset."""
    ds = load_dataset("Buraaq/quran-md-ayahs", split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    return _process_dataset(ds, output_dir, "buraaq", "ayah_ar", "reciter_id",
                            "surah_id", "ayah_id")


def _process_retasy(output_dir: Path, n_samples: int) -> list[dict]:
    """Process RetaSy/quranic_audio_dataset (correct only)."""
    ds = load_dataset("RetaSy/quranic_audio_dataset", split="train")
    # Filter using Arrow table directly to avoid decoding corrupt audio
    mask = pc.equal(ds.data.column("final_label"), "correct")
    correct_indices = [i for i, v in enumerate(mask) if v.as_py()]
    ds = ds.select(correct_indices)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    return _process_dataset(ds, output_dir, "retasy", "Aya", "reciter_id",
                            "Surah", None)


def _row_to_record(row: dict, tusers_dir: Path) -> dict | None:
    """Convert a tusers CSV row to a manifest record."""
    wav_path = tusers_dir / row["wav_filename"]
    if not wav_path.exists():
        return None
    parts = wav_path.stem.split("_")
    return {
        "audio_path": str(wav_path),
        "text": row["transcript"],
        "source": "tusers",
        "reciter": parts[2] if len(parts) >= 3 else "",
        "surah": int(parts[0]) if len(parts) >= 1 else None,
        "ayah": int(parts[1]) if len(parts) >= 2 else None,
    }


def _split_tusers(tusers_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Split tusers by user ID: 70% eval, 15% val, 15% test."""
    csv_path = tusers_dir / "tusers_filtered.csv"
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    user_rows = defaultdict(list)
    for row in rows:
        user_id = Path(row["wav_filename"]).stem.split("_")[2]
        user_rows[user_id].append(row)

    users = list(user_rows.keys())
    random.seed(42)
    random.shuffle(users)

    n_eval = int(len(users) * 0.7)
    n_val = int(len(users) * 0.15)
    eval_users = users[:n_eval]
    val_users = users[n_eval:n_eval + n_val]
    test_users = users[n_eval + n_val:]

    def collect(user_list):
        recs = []
        for uid in user_list:
            for row in user_rows[uid]:
                rec = _row_to_record(row, tusers_dir)
                if rec:
                    recs.append(rec)
        return recs

    return collect(eval_users), collect(val_users), collect(test_users)


def _process_tusers(output_dir: Path, tusers_dir: Path,
                    n_samples: int) -> list[dict]:
    """Process local tusers dataset from CSV + WAV files (eval split only)."""
    eval_recs, _, _ = _split_tusers(tusers_dir)
    return eval_recs[:n_samples]


def run_prepare(output_dir: str, max_samples: int,
                tusers_dir: str | None = None) -> None:
    """Download datasets, save audio as WAV, and write unified manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []

    if tusers_dir:
        # When tusers is provided, use it as primary source
        tusers_path = Path(tusers_dir)
        eval_recs, val_recs, test_recs = _split_tusers(tusers_path)

        n_tusers = min(int(max_samples * 0.5), len(eval_recs))
        n_tarteel = int(max_samples * 0.25)
        n_buraaq = int(max_samples * 0.2)
        n_retasy = max_samples - n_tusers - n_tarteel - n_buraaq

        records.extend(eval_recs[:n_tusers])
        records.extend(_process_tarteel(out, n_tarteel))
        records.extend(_process_buraaq(out, n_buraaq))
        records.extend(_process_retasy(out, n_retasy))

        # Write val/test manifests for fine-tuning
        for name, recs in [("manifest_val", val_recs), ("manifest_test", test_recs)]:
            path = out / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  {name}: {len(recs)} samples -> {path}")
    else:
        n_tarteel = int(max_samples * 0.4)
        n_buraaq = int(max_samples * 0.4)
        n_retasy = max_samples - n_tarteel - n_buraaq
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
