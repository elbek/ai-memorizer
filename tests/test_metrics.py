import pytest
from benchmark.metrics import compute_metrics


class TestComputeMetricsPerfectMatch:
    """Perfect match: WER=0, CER=0, harakat_accuracy=1.0."""

    def test_perfect_match_single(self):
        refs = ["بِسْمِ اللَّهِ"]
        hyps = ["بِسْمِ اللَّهِ"]
        m = compute_metrics(refs, hyps)
        assert m["wer"] == 0.0
        assert m["cer"] == 0.0
        assert m["d_wer"] == 0.0
        assert m["d_cer"] == 0.0
        assert m["harakat_accuracy"] == 1.0
        assert m["num_samples"] == 1

    def test_perfect_match_multiple(self):
        refs = ["بِسْمِ", "اللَّهِ"]
        hyps = ["بِسْمِ", "اللَّهِ"]
        m = compute_metrics(refs, hyps)
        assert m["wer"] == 0.0
        assert m["cer"] == 0.0
        assert m["harakat_accuracy"] == 1.0
        assert m["num_samples"] == 2


class TestComputeMetricsTotalMismatch:
    """Total mismatch: all error rates should be high."""

    def test_total_mismatch(self):
        refs = ["بسم الله"]
        hyps = ["كتاب جديد"]
        m = compute_metrics(refs, hyps)
        assert m["wer"] > 0.0
        assert m["cer"] > 0.0
        assert m["num_samples"] == 1


class TestComputeMetricsDiacriticsOnly:
    """Diacritics-only errors: WER>0 but D-WER==0."""

    def test_diacritics_only_error(self):
        refs = ["بِسْمِ اللَّهِ"]
        hyps = ["بَسَمَ اللَّهِ"]  # wrong diacritics, same consonants after normalization
        m = compute_metrics(refs, hyps)
        # Raw WER/CER should reflect the diacritic differences
        assert m["wer"] > 0.0 or m["cer"] > 0.0
        # After normalization, consonants are the same
        assert m["d_wer"] == 0.0
        assert m["d_cer"] == 0.0


class TestComputeMetricsEmptyHypothesis:
    """Empty hypothesis should not crash jiwer."""

    def test_empty_hypothesis(self):
        refs = ["بسم الله"]
        hyps = [""]
        m = compute_metrics(refs, hyps)
        assert m["wer"] > 0.0
        assert m["num_samples"] == 1

    def test_empty_hypothesis_among_valid(self):
        refs = ["بسم", "الله"]
        hyps = ["بسم", ""]
        m = compute_metrics(refs, hyps)
        assert m["num_samples"] == 2


class TestComputeMetricsEmptyReference:
    """Empty reference should be skipped."""

    def test_empty_reference_skipped(self):
        refs = ["", "بسم الله"]
        hyps = ["بسم", "بسم الله"]
        m = compute_metrics(refs, hyps)
        # Only the second pair should be counted
        assert m["num_samples"] == 1
        assert m["wer"] == 0.0

    def test_all_empty_references(self):
        refs = ["", ""]
        hyps = ["بسم", "الله"]
        m = compute_metrics(refs, hyps)
        assert m["num_samples"] == 0


class TestHarakatAccuracy:
    """Tests for position-by-position harakat accuracy."""

    def test_perfect_harakat(self):
        refs = ["بِسْمِ"]
        hyps = ["بِسْمِ"]
        m = compute_metrics(refs, hyps)
        assert m["harakat_accuracy"] == 1.0

    def test_missing_harakat_in_hyp(self):
        """Hypothesis has no diacritics but reference does."""
        refs = ["بِسْمِ"]
        hyps = ["بسم"]
        m = compute_metrics(refs, hyps)
        assert m["harakat_accuracy"] < 1.0

    def test_no_harakat_in_ref(self):
        """If reference has no diacritics, harakat_accuracy = 1.0."""
        refs = ["بسم الله"]
        hyps = ["بسم الله"]
        m = compute_metrics(refs, hyps)
        assert m["harakat_accuracy"] == 1.0

    def test_partial_harakat_match(self):
        """Some diacritics match, some don't."""
        refs = ["بِسْ"]
        hyps = ["بِسَ"]  # first diacritic matches, second doesn't
        m = compute_metrics(refs, hyps)
        assert 0.0 < m["harakat_accuracy"] < 1.0


class TestComputeMetricsMultipleSentences:
    """Test with multiple sentences."""

    def test_multiple_sentences(self):
        refs = [
            "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        ]
        hyps = [
            "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        ]
        m = compute_metrics(refs, hyps)
        assert m["wer"] == 0.0
        assert m["cer"] == 0.0
        assert m["num_samples"] == 2

    def test_mixed_quality(self):
        refs = ["بسم", "الله"]
        hyps = ["بسم", "كتاب"]
        m = compute_metrics(refs, hyps)
        assert 0.0 < m["wer"] <= 1.0
        assert m["num_samples"] == 2


class TestReturnedKeys:
    """Ensure the returned dict has all expected keys."""

    def test_all_keys_present(self):
        m = compute_metrics(["بسم"], ["بسم"])
        expected_keys = {"wer", "cer", "d_wer", "d_cer", "harakat_accuracy", "num_samples"}
        assert set(m.keys()) == expected_keys
