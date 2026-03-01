"""Load predictions, compute metrics, and generate a comparison report."""

import json
from collections import defaultdict
from pathlib import Path

from benchmark.metrics import compute_metrics


def _load_predictions(path):
    """Load JSONL file, return list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _format_pct(val):
    """Format float as percentage string."""
    return f"{val * 100:.2f}%"


def run_report(results_dir, data_dir):
    """Compute metrics from prediction files and generate a comparison report."""
    results_path = Path(results_dir)
    pred_files = sorted(results_path.glob("*_predictions.jsonl"))

    all_results = {}

    for pred_file in pred_files:
        model_name = pred_file.stem.replace("_predictions", "")
        preds = _load_predictions(pred_file)

        refs = [p["ref"] for p in preds]
        hyps = [p["hyp"] for p in preds]

        overall = compute_metrics(refs, hyps)

        # Group by source
        by_source = defaultdict(lambda: {"refs": [], "hyps": []})
        for p in preds:
            by_source[p["source"]]["refs"].append(p["ref"])
            by_source[p["source"]]["hyps"].append(p["hyp"])

        per_source = {}
        for source, data in sorted(by_source.items()):
            per_source[source] = compute_metrics(data["refs"], data["hyps"])

        all_results[model_name] = {
            "overall": overall,
            "per_source": per_source,
        }

    # Save JSON results
    metrics_path = results_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Generate markdown report
    models = sorted(all_results.keys())
    metric_keys = [
        ("wer", "WER"),
        ("cer", "CER"),
        ("d_wer", "D-WER"),
        ("d_cer", "D-CER"),
        ("harakat_acc", "Harakat Acc."),
        ("samples", "Samples"),
    ]

    lines = []
    lines.append("# Benchmark Report\n")

    # Overall comparison table
    lines.append("## Overall Comparison\n")
    header = "| Metric | " + " | ".join(models) + " |"
    sep = "| --- | " + " | ".join("---" for _ in models) + " |"
    lines.append(header)
    lines.append(sep)
    for key, label in metric_keys:
        row = f"| {label} |"
        for model in models:
            val = all_results[model]["overall"][key]
            if key == "samples":
                row += f" {val} |"
            else:
                row += f" {_format_pct(val)} |"
        lines.append(row)
    lines.append("")

    # Per-source breakdown per model
    for model in models:
        lines.append(f"## {model} — Per-Source Breakdown\n")
        header = "| Source | WER | CER | D-WER | D-CER | Harakat Acc. | Samples |"
        sep = "| --- | --- | --- | --- | --- | --- | --- |"
        lines.append(header)
        lines.append(sep)
        for source, metrics in sorted(all_results[model]["per_source"].items()):
            row = f"| {source}"
            for key, _ in metric_keys:
                val = metrics[key]
                if key == "samples":
                    row += f" | {val}"
                else:
                    row += f" | {_format_pct(val)}"
            row += " |"
            lines.append(row)
        lines.append("")

    report = "\n".join(lines)

    # Save markdown report
    report_path = results_path / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
