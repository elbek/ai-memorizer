"""Tests for benchmark.report module."""

import json
import tempfile
from pathlib import Path

import pytest
from benchmark.report import _load_predictions, _format_pct, run_report


class TestLoadPredictions:
    def test_loads_jsonl(self, tmp_path):
        pred_file = tmp_path / "preds.jsonl"
        pred_file.write_text(
            '{"reference": "بسم", "hypothesis": "بسم", "source": "test"}\n'
            '{"reference": "الله", "hypothesis": "الله", "source": "test"}\n',
            encoding="utf-8",
        )
        records = _load_predictions(pred_file)
        assert len(records) == 2
        assert records[0]["reference"] == "بسم"

    def test_skips_blank_lines(self, tmp_path):
        pred_file = tmp_path / "preds.jsonl"
        pred_file.write_text(
            '{"reference": "بسم", "hypothesis": "بسم", "source": "test"}\n'
            "\n"
            '{"reference": "الله", "hypothesis": "الله", "source": "test"}\n',
            encoding="utf-8",
        )
        records = _load_predictions(pred_file)
        assert len(records) == 2

    def test_empty_file(self, tmp_path):
        pred_file = tmp_path / "preds.jsonl"
        pred_file.write_text("", encoding="utf-8")
        records = _load_predictions(pred_file)
        assert records == []


class TestFormatPct:
    def test_zero(self):
        assert _format_pct(0.0) == "0.00%"

    def test_one(self):
        assert _format_pct(1.0) == "100.00%"

    def test_decimal(self):
        assert _format_pct(0.1234) == "12.34%"

    def test_small(self):
        assert _format_pct(0.005) == "0.50%"


class TestRunReport:
    def _write_predictions(self, path, model_name, pairs):
        pred_file = path / f"{model_name}_predictions.jsonl"
        with open(pred_file, "w", encoding="utf-8") as f:
            for ref, hyp, source in pairs:
                f.write(json.dumps({
                    "audio_path": "dummy.wav",
                    "reference": ref,
                    "hypothesis": hyp,
                    "source": source,
                    "model": model_name,
                }, ensure_ascii=False) + "\n")

    def test_single_model_perfect(self, tmp_path):
        self._write_predictions(tmp_path, "test_model", [
            ("بسم الله", "بسم الله", "tarteel"),
            ("الحمد لله", "الحمد لله", "tarteel"),
        ])
        run_report(str(tmp_path), str(tmp_path))

        # Check metrics.json was created
        metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert "test_model" in metrics
        assert metrics["test_model"]["overall"]["wer"] == 0.0
        assert metrics["test_model"]["overall"]["num_samples"] == 2

        # Check report.md was created
        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "test_model" in report
        assert "0.00%" in report

    def test_two_models_comparison(self, tmp_path):
        self._write_predictions(tmp_path, "good", [
            ("بسم الله", "بسم الله", "tarteel"),
        ])
        self._write_predictions(tmp_path, "bad", [
            ("بسم الله", "كتب شيء", "tarteel"),
        ])
        run_report(str(tmp_path), str(tmp_path))

        metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["good"]["overall"]["wer"] == 0.0
        assert metrics["bad"]["overall"]["wer"] > 0.0

        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "good" in report
        assert "bad" in report

    def test_per_source_breakdown(self, tmp_path):
        self._write_predictions(tmp_path, "model", [
            ("بسم الله", "بسم الله", "tarteel"),
            ("الحمد لله", "الحمد", "buraaq"),
        ])
        run_report(str(tmp_path), str(tmp_path))

        metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        per_source = metrics["model"]["per_source"]
        assert "tarteel" in per_source
        assert "buraaq" in per_source
        assert per_source["tarteel"]["wer"] == 0.0
        assert per_source["buraaq"]["wer"] > 0.0

    def test_no_prediction_files(self, tmp_path, capsys):
        run_report(str(tmp_path), str(tmp_path))
        # Should not crash, no files to process
        assert not (tmp_path / "metrics.json").exists()
