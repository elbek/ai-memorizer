# Quran ASR Benchmarking Tool — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evaluate NVIDIA FastConformer and FunASR MLT-Nano on Quran recitation with full diacritics-aware metrics.

**Architecture:** Single Python CLI package (`benchmark/`) with three subcommands: `prepare` (download/normalize datasets), `evaluate` (run model inference), `report` (compute metrics and generate comparison tables). All data flows through a unified JSONL manifest format.

**Tech Stack:** Python 3.12, datasets (HuggingFace), nemo_toolkit[asr], funasr, jiwer, tqdm, soundfile, librosa

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `benchmark/__init__.py`
- Create: `benchmark/__main__.py`
- Create: `benchmark/requirements.txt`
- Create: `.gitignore`

**Step 1: Create `.gitignore`**

```gitignore
data/
results/
__pycache__/
*.pyc
.venv/
*.egg-info/
```

**Step 2: Create `benchmark/requirements.txt`**

```
datasets>=2.14.0
soundfile>=0.12.0
librosa>=0.10.0
jiwer>=3.0.0
tqdm>=4.65.0
nemo_toolkit[asr]>=2.0.0
funasr>=1.3.0
```

**Step 3: Create `benchmark/__init__.py`**

```python
```

**Step 4: Create `benchmark/__main__.py` with CLI skeleton**

```python
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Quran ASR Benchmarking Tool",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    prep = subparsers.add_parser("prepare", help="Download and prepare datasets")
    prep.add_argument("--output-dir", default="data", help="Output directory for prepared data")
    prep.add_argument("--max-samples", type=int, default=10000, help="Max total samples across all sources")

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Run model inference")
    ev.add_argument("--model", required=True, choices=["nemo", "funasr"], help="Model to evaluate")
    ev.add_argument("--data-dir", default="data", help="Directory with prepared data")
    ev.add_argument("--output-dir", default="results", help="Output directory for predictions")
    ev.add_argument("--batch-size", type=int, default=8, help="Inference batch size")

    # report
    rep = subparsers.add_parser("report", help="Compute metrics and generate comparison")
    rep.add_argument("--results-dir", default="results", help="Directory with prediction files")
    rep.add_argument("--data-dir", default="data", help="Directory with ground truth manifest")

    args = parser.parse_args()

    if args.command == "prepare":
        from benchmark.prepare import run_prepare
        run_prepare(args.output_dir, args.max_samples)
    elif args.command == "evaluate":
        from benchmark.evaluate import run_evaluate
        run_evaluate(args.model, args.data_dir, args.output_dir, args.batch_size)
    elif args.command == "report":
        from benchmark.report import run_report
        run_report(args.results_dir, args.data_dir)


if __name__ == "__main__":
    main()
```

**Step 5: Commit**

```bash
git add .gitignore benchmark/
git commit -m "feat: scaffold benchmark CLI with subcommands"
```

---

### Task 2: Arabic text utilities

**Files:**
- Create: `benchmark/arabic_utils.py`
- Create: `tests/test_arabic_utils.py`

**Step 1: Write tests for Arabic utilities**

```python
import pytest
from benchmark.arabic_utils import strip_tashkeel, normalize_arabic


class TestStripTashkeel:
    def test_removes_fatha(self):
        assert strip_tashkeel("بَ") == "ب"

    def test_removes_damma(self):
        assert strip_tashkeel("بُ") == "ب"

    def test_removes_kasra(self):
        assert strip_tashkeel("بِ") == "ب"

    def test_removes_sukun(self):
        assert strip_tashkeel("بْ") == "ب"

    def test_removes_shadda(self):
        assert strip_tashkeel("بّ") == "ب"

    def test_removes_tanwin_fath(self):
        assert strip_tashkeel("بً") == "ب"

    def test_removes_tanwin_damm(self):
        assert strip_tashkeel("بٌ") == "ب"

    def test_removes_tanwin_kasr(self):
        assert strip_tashkeel("بٍ") == "ب"

    def test_removes_superscript_alef(self):
        assert strip_tashkeel("رَحْمٰنِ") == "رحمن"

    def test_full_basmala(self):
        text = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"
        expected = "بسم الله الرحمن الرحيم"
        assert strip_tashkeel(text) == expected

    def test_no_diacritics(self):
        text = "بسم الله"
        assert strip_tashkeel(text) == "بسم الله"

    def test_empty_string(self):
        assert strip_tashkeel("") == ""


class TestNormalizeArabic:
    def test_removes_tatweel(self):
        assert normalize_arabic("اللـــه") == "الله"

    def test_normalizes_alef_variants(self):
        assert normalize_arabic("إبراهيم") == "ابراهيم"
        assert normalize_arabic("أحمد") == "احمد"
        assert normalize_arabic("آمن") == "امن"

    def test_normalizes_teh_marbuta(self):
        assert normalize_arabic("رحمة") == "رحمه"

    def test_strips_punctuation(self):
        assert normalize_arabic("بسم الله.") == "بسم الله"
        assert normalize_arabic("بسم الله،") == "بسم الله"

    def test_collapses_whitespace(self):
        assert normalize_arabic("بسم   الله") == "بسم الله"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_arabic_utils.py -v`
Expected: FAIL (module not found)

**Step 3: Implement `benchmark/arabic_utils.py`**

```python
import re
import unicodedata

# Unicode ranges for Arabic diacritical marks (tashkeel)
_TASHKEEL = re.compile(
    "[\u0610-\u061A"   # Small annotations
    "\u064B-\u065F"    # Fathatan through Wavy Hamza Below
    "\u0670"           # Superscript Alef
    "\u06D6-\u06DC"    # Small high ligatures
    "\u06DF-\u06E4"    # More small annotations
    "\u06E7-\u06E8"    # Small annotations
    "\u06EA-\u06ED"    # More small annotations
    "\uFE70-\uFE74"    # Presentation forms
    "\uFE76-\uFE7F"    # Presentation forms
    "]+"
)

# Alef variants
_ALEF_VARIANTS = re.compile("[\u0622\u0623\u0625\u0671]")  # Alef Madda, Hamza Above, Hamza Below, Wasla

_TATWEEL = "\u0640"
_TEH_MARBUTA = "\u0629"
_HEH = "\u0647"
_PUNCTUATION = re.compile(r"[.،؟!:؛\-\"\'\(\)\[\]]")
_MULTI_SPACE = re.compile(r"\s+")


def strip_tashkeel(text: str) -> str:
    """Remove all Arabic diacritical marks from text."""
    return _TASHKEEL.sub("", text)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison: strip tashkeel, normalize alef/teh marbuta, remove punctuation."""
    text = strip_tashkeel(text)
    text = text.replace(_TATWEEL, "")
    text = _ALEF_VARIANTS.sub("\u0627", text)  # Normalize to plain Alef
    text = text.replace(_TEH_MARBUTA, _HEH)
    text = _PUNCTUATION.sub("", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_arabic_utils.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add benchmark/arabic_utils.py tests/test_arabic_utils.py
git commit -m "feat: add Arabic text utilities for tashkeel and normalization"
```

---

### Task 3: Metrics engine

**Files:**
- Create: `benchmark/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write tests for metrics**

```python
import pytest
from benchmark.metrics import compute_metrics


class TestComputeMetrics:
    def test_perfect_match(self):
        refs = ["بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"]
        hyps = ["بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"]
        m = compute_metrics(refs, hyps)
        assert m["wer"] == 0.0
        assert m["cer"] == 0.0
        assert m["d_wer"] == 0.0
        assert m["d_cer"] == 0.0

    def test_totally_wrong(self):
        refs = ["بسم"]
        hyps = ["كتب"]
        m = compute_metrics(refs, hyps)
        assert m["wer"] == 1.0
        assert m["cer"] > 0.0

    def test_diacritics_only_error(self):
        """Correct consonants but wrong diacritics should show WER > 0 but D-WER == 0."""
        refs = ["بِسْمِ"]
        hyps = ["بَسْمُ"]
        m = compute_metrics(refs, hyps)
        assert m["wer"] > 0.0   # Full text differs
        assert m["d_wer"] == 0.0  # Without diacritics, same

    def test_multiple_sentences(self):
        refs = ["بسم الله", "الحمد لله"]
        hyps = ["بسم الله", "الحمد لله"]
        m = compute_metrics(refs, hyps)
        assert m["wer"] == 0.0

    def test_empty_hypothesis(self):
        refs = ["بسم الله"]
        hyps = [""]
        m = compute_metrics(refs, hyps)
        assert m["wer"] == 1.0

    def test_returns_all_keys(self):
        m = compute_metrics(["بسم"], ["بسم"])
        expected_keys = {"wer", "cer", "d_wer", "d_cer", "harakat_accuracy", "num_samples"}
        assert expected_keys.issubset(set(m.keys()))


class TestHarakatAccuracy:
    def test_perfect_diacritics(self):
        refs = ["بِسْمِ"]
        hyps = ["بِسْمِ"]
        m = compute_metrics(refs, hyps)
        assert m["harakat_accuracy"] == 1.0

    def test_missing_diacritics(self):
        refs = ["بِسْمِ"]
        hyps = ["بسم"]
        m = compute_metrics(refs, hyps)
        assert m["harakat_accuracy"] < 1.0

    def test_no_diacritics_in_ref(self):
        refs = ["بسم"]
        hyps = ["بسم"]
        m = compute_metrics(refs, hyps)
        assert m["harakat_accuracy"] == 1.0  # No diacritics to compare
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL (module not found)

**Step 3: Implement `benchmark/metrics.py`**

```python
import re
from jiwer import wer, cer
from benchmark.arabic_utils import strip_tashkeel, normalize_arabic

_TASHKEEL_CHARS = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]"
)


def _extract_harakat(text: str) -> list[tuple[str, str]]:
    """Extract (base_char, diacritic_or_empty) pairs from Arabic text.

    For each base character, collect the diacritical mark(s) following it.
    Returns list of (base_char, diacritics_string) tuples.
    """
    pairs = []
    i = 0
    while i < len(text):
        ch = text[i]
        if _TASHKEEL_CHARS.match(ch):
            # Orphan diacritic at start - skip
            i += 1
            continue
        diacritics = ""
        j = i + 1
        while j < len(text) and _TASHKEEL_CHARS.match(text[j]):
            diacritics += text[j]
            j += 1
        if not ch.isspace():
            pairs.append((ch, diacritics))
        i = j
    return pairs


def _harakat_accuracy(refs: list[str], hyps: list[str]) -> float:
    """Compute position-by-position diacritical mark accuracy.

    Aligns by base consonants (stripped text) and compares diacritics.
    """
    total = 0
    correct = 0
    for ref, hyp in zip(refs, hyps):
        ref_pairs = _extract_harakat(ref)
        hyp_pairs = _extract_harakat(hyp)
        # Only compare up to the shorter length (mismatched lengths penalized by WER/CER)
        for rp, hp in zip(ref_pairs, hyp_pairs):
            if rp[1]:  # Only count positions where ref has diacritics
                total += 1
                if rp[1] == hp[1]:
                    correct += 1
    if total == 0:
        return 1.0
    return correct / total


def compute_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """Compute full Quran ASR metrics suite.

    Returns dict with: wer, cer, d_wer, d_cer, harakat_accuracy, num_samples
    """
    # Filter out pairs where both ref and hyp are empty
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not pairs:
        return {"wer": 0.0, "cer": 0.0, "d_wer": 0.0, "d_cer": 0.0,
                "harakat_accuracy": 1.0, "num_samples": 0}

    refs, hyps = zip(*pairs)
    refs, hyps = list(refs), list(hyps)

    # Replace empty hypotheses with a placeholder to avoid jiwer errors
    hyps_safe = [h if h.strip() else "▌" for h in hyps]

    # Full diacritized metrics
    full_wer = wer(refs, hyps_safe)
    full_cer = cer(refs, hyps_safe)

    # Diacritics-stripped metrics
    refs_stripped = [normalize_arabic(r) for r in refs]
    hyps_stripped = [normalize_arabic(h) for h in hyps_safe]
    d_wer_val = wer(refs_stripped, hyps_stripped)
    d_cer_val = cer(refs_stripped, hyps_stripped)

    # Harakat accuracy
    har_acc = _harakat_accuracy(refs, hyps)

    return {
        "wer": round(full_wer, 4),
        "cer": round(full_cer, 4),
        "d_wer": round(d_wer_val, 4),
        "d_cer": round(d_cer_val, 4),
        "harakat_accuracy": round(har_acc, 4),
        "num_samples": len(refs),
    }
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add benchmark/metrics.py tests/test_metrics.py
git commit -m "feat: add metrics engine with WER, CER, D-WER, D-CER, harakat accuracy"
```

---

### Task 4: Dataset preparation

**Files:**
- Create: `benchmark/prepare.py`

This task downloads from three HuggingFace datasets, normalizes them into a unified JSONL manifest, and saves audio as 16kHz mono WAV files.

**Step 1: Implement `benchmark/prepare.py`**

```python
import json
import os
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def _save_audio(sample_audio: dict, output_path: str) -> None:
    """Save HuggingFace audio dict to 16kHz mono WAV."""
    sf.write(output_path, sample_audio["array"], sample_audio["sampling_rate"])


def _prepare_tarteel(output_dir: Path, max_samples: int) -> list[dict]:
    """Load tarteel-ai/everyayah test split."""
    print(f"Loading tarteel-ai/everyayah (test split, max {max_samples})...")
    ds = load_dataset("tarteel-ai/everyayah", split="test")

    audio_dir = output_dir / "audio" / "tarteel"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, sample in enumerate(tqdm(ds, total=min(max_samples, len(ds)), desc="tarteel")):
        if i >= max_samples:
            break
        audio_path = str(audio_dir / f"{i:06d}.wav")
        _save_audio(sample["audio"], audio_path)
        manifest.append({
            "audio_path": audio_path,
            "text": sample["text"],
            "source": "tarteel",
            "reciter": sample.get("reciter", "unknown"),
        })
    return manifest


def _prepare_buraaq(output_dir: Path, max_samples: int) -> list[dict]:
    """Load Buraaq/quran-md-ayahs, sample from train split."""
    print(f"Loading Buraaq/quran-md-ayahs (max {max_samples})...")
    ds = load_dataset("Buraaq/quran-md-ayahs", split="train")

    # Shuffle deterministically and take max_samples
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    audio_dir = output_dir / "audio" / "buraaq"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, sample in enumerate(tqdm(ds, total=len(ds), desc="buraaq")):
        audio_path = str(audio_dir / f"{i:06d}.wav")
        _save_audio(sample["audio"], audio_path)
        manifest.append({
            "audio_path": audio_path,
            "text": sample["ayah_ar"],
            "source": "buraaq",
            "surah": sample.get("surah_id"),
            "ayah": sample.get("ayah_id"),
            "reciter": sample.get("reciter_id", "unknown"),
        })
    return manifest


def _prepare_retasy(output_dir: Path, max_samples: int) -> list[dict]:
    """Load RetaSy/quranic_audio_dataset, correct-only."""
    print(f"Loading RetaSy/quranic_audio_dataset (correct only, max {max_samples})...")
    ds = load_dataset("RetaSy/quranic_audio_dataset", split="train")
    ds = ds.filter(lambda x: x["final_label"] == "correct")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    audio_dir = output_dir / "audio" / "retasy"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, sample in enumerate(tqdm(ds, total=len(ds), desc="retasy")):
        audio_path = str(audio_dir / f"{i:06d}.wav")
        _save_audio(sample["audio"], audio_path)
        manifest.append({
            "audio_path": audio_path,
            "text": sample["Aya"],
            "source": "retasy",
            "surah": sample.get("Surah"),
            "reciter": sample.get("reciter_id", "unknown"),
        })
    return manifest


def run_prepare(output_dir: str, max_samples: int) -> None:
    """Download and prepare all datasets into unified manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Split max_samples roughly: 40% tarteel, 40% buraaq, 20% retasy
    n_tarteel = int(max_samples * 0.4)
    n_buraaq = int(max_samples * 0.4)
    n_retasy = max_samples - n_tarteel - n_buraaq

    manifest = []
    manifest.extend(_prepare_tarteel(out, n_tarteel))
    manifest.extend(_prepare_buraaq(out, n_buraaq))
    manifest.extend(_prepare_retasy(out, n_retasy))

    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nPrepared {len(manifest)} samples -> {manifest_path}")
    print(f"  tarteel: {sum(1 for m in manifest if m['source'] == 'tarteel')}")
    print(f"  buraaq:  {sum(1 for m in manifest if m['source'] == 'buraaq')}")
    print(f"  retasy:  {sum(1 for m in manifest if m['source'] == 'retasy')}")
```

**Step 2: Commit**

```bash
git add benchmark/prepare.py
git commit -m "feat: add dataset preparation for tarteel, buraaq, retasy"
```

---

### Task 5: Model evaluation runners

**Files:**
- Create: `benchmark/evaluate.py`

Two model runners: NeMo FastConformer and FunASR MLT-Nano.

**Step 1: Implement `benchmark/evaluate.py`**

```python
import json
import os
from pathlib import Path

from tqdm import tqdm


def _load_manifest(data_dir: str) -> list[dict]:
    """Load the prepared manifest."""
    manifest_path = Path(data_dir) / "manifest.jsonl"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _run_nemo(manifest: list[dict], output_path: Path, batch_size: int) -> None:
    """Run NVIDIA FastConformer inference."""
    import nemo.collections.asr as nemo_asr

    print("Loading NVIDIA FastConformer model...")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    )
    model.eval()

    audio_paths = [entry["audio_path"] for entry in manifest]

    print(f"Transcribing {len(audio_paths)} files (batch_size={batch_size})...")
    results = model.transcribe(audio=audio_paths, batch_size=batch_size)

    predictions = []
    for entry, result in zip(manifest, results):
        text = result.text if hasattr(result, "text") else str(result)
        predictions.append({
            "audio_path": entry["audio_path"],
            "reference": entry["text"],
            "hypothesis": text,
            "source": entry["source"],
            "model": "nemo_fastconformer",
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"Saved {len(predictions)} predictions -> {output_path}")


def _run_funasr(manifest: list[dict], output_path: Path, batch_size: int) -> None:
    """Run FunASR MLT-Nano inference."""
    from funasr import AutoModel

    print("Loading FunASR MLT-Nano model...")
    model = AutoModel(
        model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        trust_remote_code=True,
        remote_code="./model.py",
        device="cpu",
        hub="hf",
    )

    audio_paths = [entry["audio_path"] for entry in manifest]

    predictions = []
    # Process in batches
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="funasr"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_entries = manifest[i : i + batch_size]
        results = model.generate(
            input=batch_paths,
            cache={},
            batch_size=batch_size,
            language="Arabic",
        )
        for entry, result in zip(batch_entries, results):
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            predictions.append({
                "audio_path": entry["audio_path"],
                "reference": entry["text"],
                "hypothesis": text,
                "source": entry["source"],
                "model": "funasr_mlt_nano",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"Saved {len(predictions)} predictions -> {output_path}")


def run_evaluate(model: str, data_dir: str, output_dir: str, batch_size: int) -> None:
    """Run inference for the specified model."""
    manifest = _load_manifest(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if model == "nemo":
        _run_nemo(manifest, out / "nemo_predictions.jsonl", batch_size)
    elif model == "funasr":
        _run_funasr(manifest, out / "funasr_predictions.jsonl", batch_size)
    else:
        raise ValueError(f"Unknown model: {model}")
```

**Step 2: Commit**

```bash
git add benchmark/evaluate.py
git commit -m "feat: add model evaluation runners for NeMo and FunASR"
```

---

### Task 6: Report generation

**Files:**
- Create: `benchmark/report.py`

**Step 1: Implement `benchmark/report.py`**

```python
import json
from collections import defaultdict
from pathlib import Path

from benchmark.metrics import compute_metrics


def _load_predictions(path: Path) -> list[dict]:
    """Load prediction JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _format_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def run_report(results_dir: str, data_dir: str) -> None:
    """Compute metrics and generate comparison report."""
    results_path = Path(results_dir)
    pred_files = list(results_path.glob("*_predictions.jsonl"))

    if not pred_files:
        print(f"No prediction files found in {results_dir}")
        return

    all_results = {}

    for pred_file in pred_files:
        model_name = pred_file.stem.replace("_predictions", "")
        predictions = _load_predictions(pred_file)

        refs = [p["reference"] for p in predictions]
        hyps = [p["hypothesis"] for p in predictions]

        # Overall metrics
        overall = compute_metrics(refs, hyps)
        all_results[model_name] = {"overall": overall}

        # Per-source metrics
        by_source = defaultdict(lambda: ([], []))
        for p in predictions:
            r, h = by_source[p["source"]]
            r.append(p["reference"])
            h.append(p["hypothesis"])

        source_metrics = {}
        for source, (r, h) in by_source.items():
            source_metrics[source] = compute_metrics(r, h)
        all_results[model_name]["by_source"] = source_metrics

    # Save JSON results
    json_path = results_path / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {json_path}")

    # Generate markdown report
    report_lines = ["# Quran ASR Benchmark Results\n"]

    # Overall comparison table
    report_lines.append("## Overall Comparison\n")
    report_lines.append("| Metric | " + " | ".join(all_results.keys()) + " |")
    report_lines.append("|--------|" + "|".join(["--------"] * len(all_results)) + "|")

    metric_names = [("wer", "WER"), ("cer", "CER"), ("d_wer", "D-WER"),
                    ("d_cer", "D-CER"), ("harakat_accuracy", "Harakat Acc."),
                    ("num_samples", "Samples")]
    for key, label in metric_names:
        row = f"| {label} |"
        for model in all_results:
            val = all_results[model]["overall"][key]
            if key == "num_samples":
                row += f" {val} |"
            elif key == "harakat_accuracy":
                row += f" {_format_pct(val)} |"
            else:
                row += f" {_format_pct(val)} |"
        report_lines.append(row)

    # Per-source breakdown
    report_lines.append("\n## Per-Source Breakdown\n")
    for model_name, data in all_results.items():
        report_lines.append(f"### {model_name}\n")
        report_lines.append("| Source | WER | CER | D-WER | D-CER | Harakat Acc. | Samples |")
        report_lines.append("|--------|-----|-----|-------|-------|--------------|---------|")
        for source, m in data["by_source"].items():
            report_lines.append(
                f"| {source} | {_format_pct(m['wer'])} | {_format_pct(m['cer'])} "
                f"| {_format_pct(m['d_wer'])} | {_format_pct(m['d_cer'])} "
                f"| {_format_pct(m['harakat_accuracy'])} | {m['num_samples']} |"
            )

    report_md = "\n".join(report_lines) + "\n"
    report_path = results_path / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Report saved to {report_path}")
    print("\n" + report_md)
```

**Step 2: Commit**

```bash
git add benchmark/report.py
git commit -m "feat: add report generation with comparison tables"
```

---

### Task 7: Create virtual environment, install dependencies, run tests

**Step 1: Create venv and install dependencies**

```bash
cd /Users/elbekkamol/work/ai-memorizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmark/requirements.txt
pip install pytest
```

Note: `nemo_toolkit[asr]` install may have issues on macOS. If it fails, install without nemo first and test the other components:
```bash
pip install datasets soundfile librosa jiwer tqdm funasr pytest
pip install 'nemo_toolkit[asr]' || echo "NeMo install failed - will debug separately"
```

**Step 2: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass

**Step 3: Commit any test fixes if needed**

---

### Task 8: Run dataset preparation

**Step 1: Run prepare command**

```bash
python -m benchmark prepare --max-samples 10000 --output-dir data
```

This will download from HuggingFace and create:
- `data/audio/tarteel/` (~4000 WAV files)
- `data/audio/buraaq/` (~4000 WAV files)
- `data/audio/retasy/` (~2000 WAV files)
- `data/manifest.jsonl`

**Step 2: Verify manifest**

```bash
wc -l data/manifest.jsonl
head -1 data/manifest.jsonl | python -m json.tool
```

---

### Task 9: Run model evaluations

**Step 1: Run NeMo evaluation**

```bash
python -m benchmark evaluate --model nemo --batch-size 4
```

Note: CPU inference will be slow. If it's too slow, reduce batch to try a subset first:
```bash
head -100 data/manifest.jsonl > data/manifest_small.jsonl
```

**Step 2: Run FunASR evaluation**

```bash
python -m benchmark evaluate --model funasr --batch-size 4
```

**Step 3: Generate comparison report**

```bash
python -m benchmark report
```

This produces `results/report.md` and `results/metrics.json`.

**Step 4: Commit results report**

```bash
git add results/report.md results/metrics.json
git commit -m "feat: add initial benchmark results"
```
