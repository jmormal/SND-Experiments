"""
Solve a single SND-RTI instance.

Usage:
    python scripts/run_model.py path/to/instance.json
    python scripts/run_model.py path/to/instance.json --baseline
    python scripts/run_model.py path/to/instance.json --solver highs --time-limit 600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snd_rti.instance import load_instance
from snd_rti.model import build_model
from snd_rti.baseline import build_baseline_model
from snd_rti.solve import solve_model, extract_solution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", help="JSON file produced by instance_generator.py")
    ap.add_argument("--baseline", action="store_true",
                    help="Solve the one-to-one baseline instead")
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--mip-gap", type=float, default=0.005)
    ap.add_argument("--out", default=None,
                    help="Optional JSON output file for the solution")
    ap.add_argument("--tee", action="store_true")
    args = ap.parse_args()

    inst = load_instance(args.instance)
    print(inst.summary())

    if args.baseline:
        m = build_baseline_model(inst, verbose=True)
    else:
        m = build_model(inst, verbose=True)

    res = solve_model(m, solver=args.solver,
                      time_limit=args.time_limit,
                      mip_gap=args.mip_gap,
                      tee=args.tee)
    print("\n--- solver ---")
    for k, v in res.items():
        print(f"  {k}: {v}")

    if res["has_solution"]:
        sol = extract_solution(m)
        print("\n--- solution summary ---")
        for k, v in sol["costs"].items():
            print(f"  {k}: {v:.2f}")
        print(f"  active routes : {sol['n_active_routes']}")
        print(f"  active hubs   : {sol['n_active_hubs']}")
        print(f"  RTIs purchased: {sol['total_pool']:.1f}  (by type: "
              f"{ {r: round(v,1) for r,v in sol['pool_purchased'].items()} })")

        if args.out:
            with open(args.out, "w") as fp:
                json.dump(sol, fp, indent=2, default=str)
            print(f"\nSolution written to {args.out}")


if __name__ == "__main__":
    main()
