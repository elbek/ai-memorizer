"""Metrics engine for Quran ASR benchmarking."""

import re
import jiwer
from benchmark.arabic_utils import strip_tashkeel, normalize_arabic, normalize_quran_text

# Same tashkeel pattern from arabic_utils for extracting diacritics
_TASHKEEL_RE = re.compile(
    "["
    "\u0610-\u061A"
    "\u064B-\u065F"
    "\u0670"
    "\u06D6-\u06DC"
    "\u06DF-\u06E4"
    "\u06E7-\u06E8"
    "\u06EA-\u06ED"
    "\uFE70-\uFE74"
    "\uFE76-\uFE7F"
    "]"
)


def _extract_harakat_pairs(text: str) -> list[tuple[str, str]]:
    """Extract (base_char, diacritics) pairs from Arabic text.

    Walks through the text character by character. For each non-diacritic
    character, collects any following diacritical marks as a group.
    Returns a list of (base_char, diacritics_string) tuples.
    """
    pairs = []
    i = 0
    while i < len(text):
        ch = text[i]
        if _TASHKEEL_RE.match(ch):
            # Orphan diacritic without a base char -- skip
            i += 1
            continue
        diacritics = []
        j = i + 1
        while j < len(text) and _TASHKEEL_RE.match(text[j]):
            diacritics.append(text[j])
            j += 1
        pairs.append((ch, "".join(diacritics)))
        i = j
    return pairs


def _harakat_accuracy(references: list[str], hypotheses: list[str]) -> float:
    """Compute position-by-position diacritical mark accuracy.

    For each (ref, hyp) pair, extract (base_char, diacritics) pairs and
    compare at each position. Only positions where the reference has
    diacritics are evaluated.

    If the reference has no diacritics at all (across all pairs), return 1.0.
    """
    total = 0
    correct = 0

    for ref, hyp in zip(references, hypotheses):
        ref_pairs = _extract_harakat_pairs(ref)
        hyp_pairs = _extract_harakat_pairs(hyp)

        for idx, (base, ref_diac) in enumerate(ref_pairs):
            if not ref_diac:
                continue  # no diacritic at this position in reference
            total += 1
            if idx < len(hyp_pairs) and hyp_pairs[idx][1] == ref_diac:
                correct += 1

    if total == 0:
        return 1.0
    return correct / total


def compute_metrics(
    references: list[str], hypotheses: list[str]
) -> dict[str, float | int]:
    """Compute ASR evaluation metrics.

    Args:
        references: Ground truth transcriptions.
        hypotheses: Model predictions (same length as references).

    Returns:
        Dict with keys: wer, cer, d_wer, d_cer, harakat_accuracy, num_samples.
    """
    # Filter out empty references
    filtered_refs = []
    filtered_hyps = []
    for ref, hyp in zip(references, hypotheses):
        if ref.strip():
            filtered_refs.append(ref)
            filtered_hyps.append(hyp)

    num_samples = len(filtered_refs)

    if num_samples == 0:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "d_wer": 0.0,
            "d_cer": 0.0,
            "harakat_accuracy": 1.0,
            "num_samples": 0,
        }

    # Normalize Quran-specific Unicode in both ref and hyp before all comparisons
    filtered_refs = [normalize_quran_text(r) for r in filtered_refs]
    filtered_hyps = [normalize_quran_text(h) for h in filtered_hyps]

    # Replace empty hypotheses with a placeholder so jiwer doesn't crash
    safe_hyps = [h if h.strip() else "\u200B" for h in filtered_hyps]

    # Raw metrics on full diacritized text
    wer = jiwer.wer(filtered_refs, safe_hyps)
    cer = jiwer.cer(filtered_refs, safe_hyps)

    # Normalized metrics (consonant-level, diacritics stripped)
    norm_refs = [normalize_arabic(r) for r in filtered_refs]
    norm_hyps = [normalize_arabic(h) for h in filtered_hyps]

    # Safe normalized hyps (in case normalization produces empty string)
    safe_norm_hyps = [h if h.strip() else "\u200B" for h in norm_hyps]

    d_wer = jiwer.wer(norm_refs, safe_norm_hyps)
    d_cer = jiwer.cer(norm_refs, safe_norm_hyps)

    # Harakat accuracy
    h_acc = _harakat_accuracy(filtered_refs, filtered_hyps)

    return {
        "wer": wer,
        "cer": cer,
        "d_wer": d_wer,
        "d_cer": d_cer,
        "harakat_accuracy": h_acc,
        "num_samples": num_samples,
    }
