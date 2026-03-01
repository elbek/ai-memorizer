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
