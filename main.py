"""CLI entry: python main.py --ticker AAPL --all  (no interactive input)."""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from analysis_model.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run equity analysis pipelines.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--dupont", action="store_true")
    parser.add_argument("--ccc", action="store_true")
    parser.add_argument("--dcf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    any_flag = args.zscore or args.dupont or args.ccc or args.dcf
    run_all = args.all or not any_flag
    result = run_pipeline(
        ticker=args.ticker,
        run_zscore=run_all or args.zscore,
        run_dupont=run_all or args.dupont,
        run_ccc=run_all or args.ccc,
        run_dcf=run_all or args.dcf,
    )
    if result.get("error"):
        print(result["error"])
        sys.exit(1)
    print(result.get("metrics"))


if __name__ == "__main__":
    main()
