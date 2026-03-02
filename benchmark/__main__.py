"""Quran ASR Benchmarking Tool — CLI entry point.

Usage:
    python -m benchmark prepare     Download and prepare evaluation datasets
    python -m benchmark evaluate    Run ASR model inference on prepared data
    python -m benchmark report      Compute metrics and generate comparison
    python -m benchmark run-all     Run full pipeline: prepare → evaluate → report
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Quran ASR Benchmarking Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m benchmark prepare --max-samples 10000
  python -m benchmark evaluate --model nemo --batch-size 4
  python -m benchmark evaluate --model funasr --batch-size 4
  python -m benchmark report
  python -m benchmark run-all --max-samples 1000 --batch-size 4
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    prep = subparsers.add_parser(
        "prepare",
        help="Download and prepare evaluation datasets",
        description="Download from tarteel-ai/everyayah, Buraaq/quran-md-ayahs, "
        "and RetaSy/quranic_audio_dataset. Saves 16kHz WAV files and a "
        "unified JSONL manifest.",
    )
    prep.add_argument(
        "--output-dir", default="data",
        help="Output directory for prepared data (default: data)",
    )
    prep.add_argument(
        "--max-samples", type=int, default=10000,
        help="Max total samples across all sources (default: 10000). "
        "Split 40%% tarteel / 40%% buraaq / 20%% retasy.",
    )
    prep.add_argument(
        "--tusers-dir",
        help="Path to local tusers dataset directory (contains tusers_filtered.csv + wav/)",
    )

    # evaluate
    ev = subparsers.add_parser(
        "evaluate",
        help="Run ASR model inference on prepared data",
        description="Run a model on the prepared dataset and save predictions. "
        "Models: nemo (NVIDIA FastConformer with tashkeel), "
        "funasr (FunASR MLT-Nano multilingual).",
    )
    ev.add_argument(
        "--model", required=True, choices=["nemo", "funasr", "whisper", "parakeet", "qwen3"],
        help="Model to evaluate: nemo or funasr",
    )
    ev.add_argument(
        "--data-dir", default="data",
        help="Directory with prepared data (default: data)",
    )
    ev.add_argument(
        "--output-dir", default="results",
        help="Output directory for predictions (default: results)",
    )
    ev.add_argument(
        "--batch-size", type=int, default=8,
        help="Inference batch size (default: 8). Lower if running out of memory.",
    )
    ev.add_argument(
        "--manifest",
        help="Path to a custom manifest JSONL (overrides data-dir/manifest.jsonl)",
    )
    ev.add_argument(
        "--model-id",
        help="HuggingFace model ID (required for --model whisper, "
        "e.g. tarteel-ai/whisper-base-ar-quran)",
    )

    # report
    rep = subparsers.add_parser(
        "report",
        help="Compute metrics and generate comparison report",
        description="Load prediction files, compute WER/CER/D-WER/D-CER/Harakat "
        "accuracy, and generate a markdown comparison table.",
    )
    rep.add_argument(
        "--results-dir", default="results",
        help="Directory with prediction JSONL files (default: results)",
    )
    rep.add_argument(
        "--data-dir", default="data",
        help="Directory with ground truth manifest (default: data)",
    )

    # run-all
    ra = subparsers.add_parser(
        "run-all",
        help="Run full pipeline: prepare → evaluate both models → report",
        description="Convenience command to run the entire benchmark pipeline.",
    )
    ra.add_argument(
        "--data-dir", default="data",
        help="Data directory (default: data)",
    )
    ra.add_argument(
        "--results-dir", default="results",
        help="Results directory (default: results)",
    )
    ra.add_argument(
        "--max-samples", type=int, default=10000,
        help="Max samples for preparation (default: 10000)",
    )
    ra.add_argument(
        "--batch-size", type=int, default=8,
        help="Inference batch size (default: 8)",
    )
    ra.add_argument(
        "--models", nargs="+", default=["nemo", "whisper"],
        choices=["nemo", "funasr", "whisper", "parakeet", "qwen3"],
        help="Models to evaluate (default: both)",
    )
    ra.add_argument(
        "--tusers-dir",
        help="Path to local tusers dataset directory",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        from benchmark.prepare import run_prepare
        run_prepare(args.output_dir, args.max_samples, args.tusers_dir)

    elif args.command == "evaluate":
        from benchmark.evaluate import run_evaluate
        if args.model == "whisper" and not args.model_id:
            parser.error("--model-id is required when --model is whisper")
        run_evaluate(args.model, args.data_dir, args.output_dir, args.batch_size,
                     getattr(args, 'manifest', None),
                     model_id=getattr(args, 'model_id', None))

    elif args.command == "report":
        from benchmark.report import run_report
        run_report(args.results_dir, args.data_dir)

    elif args.command == "run-all":
        from benchmark.prepare import run_prepare
        from benchmark.evaluate import run_evaluate
        from benchmark.report import run_report

        print("=" * 60)
        print("Step 1/3: Preparing datasets")
        print("=" * 60)
        run_prepare(args.data_dir, args.max_samples,
                    getattr(args, 'tusers_dir', None))

        for model in args.models:
            print()
            print("=" * 60)
            print(f"Step 2/3: Evaluating {model}")
            print("=" * 60)
            run_evaluate(model, args.data_dir, args.results_dir, args.batch_size)

        print()
        print("=" * 60)
        print("Step 3/3: Generating report")
        print("=" * 60)
        run_report(args.results_dir, args.data_dir)


if __name__ == "__main__":
    main()
