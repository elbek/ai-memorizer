#!/usr/bin/env python3
"""Convert manifests to NeMo format and train a bilingual SentencePiece tokenizer."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import soundfile as sf
import sentencepiece as spm


def convert_manifest(input_path: Path, output_path: Path, base_dir: Path, max_samples: int | None = None):
    """Convert manifest to NeMo format: audio_path→audio_filepath (absolute), add duration."""
    entries = []
    with open(input_path) as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)
            if max_samples and len(entries) >= max_samples:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w") as out:
        for entry in entries:
            audio_rel = entry.pop("audio_path")
            audio_abs = str((base_dir / audio_rel).resolve())
            info = sf.info(audio_abs)
            nemo_entry = {
                "audio_filepath": audio_abs,
                "text": entry["text"],
                "duration": round(info.duration, 3),
            }
            out.write(json.dumps(nemo_entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"  {input_path.name} → {output_path.name}: {written} samples")
    return written


def collect_texts(manifest_path: Path) -> list[str]:
    """Extract all text lines from a manifest."""
    texts = []
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry["text"])
    return texts


def get_parakeet_english_texts() -> list[str]:
    """Extract English vocabulary words from parakeet-tdt-0.6b-v3's tokenizer."""
    try:
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        tokenizer = model.tokenizer
        # Get all tokens from the vocab
        vocab = []
        for i in range(tokenizer.vocab_size):
            token = tokenizer.ids_to_tokens([i])
            if token:
                vocab.extend(token)
        # Build synthetic English sentences from vocab tokens
        # Filter to actual word-like tokens (strip SentencePiece prefix)
        words = []
        for t in vocab:
            clean = t.replace("▁", " ").strip()
            if clean and clean.isascii() and any(c.isalpha() for c in clean):
                words.append(clean)
        # Return as lines for SentencePiece training
        english_lines = []
        chunk_size = 20
        for i in range(0, len(words), chunk_size):
            english_lines.append(" ".join(words[i : i + chunk_size]))
        del model
        return english_lines
    except Exception as e:
        print(f"  Warning: Could not load parakeet vocab ({e}), training Arabic-only tokenizer")
        return []


def train_tokenizer(texts: list[str], output_dir: Path, vocab_size: int):
    """Train SentencePiece BPE tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write training text to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        for line in texts:
            tmp.write(line + "\n")
        tmp_path = tmp.name

    model_prefix = str(output_dir / "tokenizer")
    try:
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,  # full coverage for Arabic diacritics
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            # Ensure tashkeel characters are not merged away
            split_by_unicode_script=True,
            split_digits=True,
            byte_fallback=True,
        )
        print(f"  Tokenizer trained: {model_prefix}.model (vocab_size={vocab_size})")
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare NeMo manifests and tokenizer")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Input data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/nemo"), help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Tokenizer vocab size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training set size")
    args = parser.parse_args()

    base_dir = args.data_dir.parent if args.data_dir.name == "data" else args.data_dir
    # base_dir should be the project root so audio_path resolves correctly
    base_dir = Path.cwd()

    # --- Convert manifests ---
    print("Converting manifests to NeMo format...")
    splits = {
        "train": ("manifest.jsonl", args.max_samples),
        "val": ("manifest_val.jsonl", None),
        "test": ("manifest_test.jsonl", None),
    }
    for split_name, (filename, limit) in splits.items():
        input_path = args.data_dir / filename
        if not input_path.exists():
            print(f"  Skipping {filename} (not found)")
            continue
        output_path = args.output_dir / f"{split_name}.jsonl"
        convert_manifest(input_path, output_path, base_dir, max_samples=limit)

    # --- Train tokenizer ---
    print("Collecting training texts...")
    train_manifest = args.output_dir / "train.jsonl"
    if not train_manifest.exists():
        print("  Error: train manifest not found, cannot train tokenizer")
        return

    arabic_texts = collect_texts(train_manifest)
    print(f"  Arabic texts: {len(arabic_texts)} lines")

    english_texts = get_parakeet_english_texts()
    if english_texts:
        print(f"  English texts: {len(english_texts)} lines")

    all_texts = arabic_texts + english_texts

    print("Training SentencePiece tokenizer...")
    tokenizer_dir = args.output_dir / "tokenizer"
    train_tokenizer(all_texts, tokenizer_dir, args.vocab_size)

    print("Done.")


if __name__ == "__main__":
    main()
