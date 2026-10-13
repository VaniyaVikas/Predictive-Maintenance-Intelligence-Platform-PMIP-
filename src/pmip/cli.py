"""Command line interface: ``python -m pmip.cli {download,synthetic,eda,train}``."""
from __future__ import annotations

import argparse
import json

from .config import load_settings
from .logging_utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(prog="pmip")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download the UCI AI4I 2020 dataset")
    syn = sub.add_parser("synthetic", help="write a SYNTHETIC AI4I-like CSV for smoke tests (not real data)")
    syn.add_argument("--rows", type=int, default=6000)
    sub.add_parser("eda", help="run exploratory data analysis")
    tr = sub.add_parser("train", help="train, evaluate and save the model")
    tr.add_argument("--no-mlflow", action="store_true")
    tr.add_argument("--fast", action="store_true", help="skip hyper-parameter tuning")
    args = p.parse_args(argv)
    s = load_settings()
    raw = s.resolve(s.paths.raw_data)

    if args.cmd == "download":
        from .download import download_ai4i

        print(f"Saved dataset to {download_ai4i(raw)}")
    elif args.cmd == "synthetic":
        from .synthetic import generate_synthetic_ai4i

        raw.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_ai4i(args.rows, s.seed).to_csv(raw, index=False)
        print(f"WROTE SYNTHETIC (non-real) data to {raw}")
    elif args.cmd == "eda":
        from .data import load_raw_dataset
        from .eda import run_eda

        run_eda(load_raw_dataset(raw), s.resolve(s.paths.reports_dir))
        print(f"EDA written to {s.resolve(s.paths.reports_dir)}")
    elif args.cmd == "train":
        from .train import train_pipeline

        if args.fast:
            s.training.tuning_enabled = False
        res = train_pipeline(s, use_mlflow=not args.no_mlflow)
        print(res.comparison.round(3).to_string(index=False))
        print(json.dumps({"version": res.version, "threshold": res.metrics["decision_threshold"],
                          "test_metrics": res.metrics["test_metrics"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
