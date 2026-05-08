"""
Run the computational experiments over every JSON instance in a folder.

Usage:
    python scripts/run_experiments.py instances/ --solver gurobi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snd_rti.experiments import run_experiments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_dir",
                    help="Folder with *.json files produced by instance_generator")
    ap.add_argument("--out", default="results", help="Output folder for CSV")
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--time-limit-small", type=float, default=600.0)
    ap.add_argument("--time-limit-medium", type=float, default=3600.0)
    ap.add_argument("--time-limit-large", type=float, default=7200.0)
    ap.add_argument("--mip-gap", type=float, default=0.005)
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip the one-to-one baseline comparison")
    ap.add_argument("--pattern", default="*.json")
    args = ap.parse_args()

    run_experiments(
        instance_dir=args.instance_dir,
        out_dir=args.out,
        solver=args.solver,
        time_limit_small=args.time_limit_small,
        time_limit_medium=args.time_limit_medium,
        time_limit_large=args.time_limit_large,
        mip_gap=args.mip_gap,
        run_baseline=not args.no_baseline,
        pattern=args.pattern,
        verbose=True,
    )


if __name__ == "__main__":
    main()
