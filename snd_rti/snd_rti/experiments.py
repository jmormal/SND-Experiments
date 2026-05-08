"""
Computational experiments for the SND-RTI paper.

Implements:
    * Experiment 1 — computational scalability across Small / Medium /
      Large instance classes (Section 4.4).
    * Experiment 2 — cost savings of the optimised SND-RTI solution
      relative to the one-to-one baseline (Section 4.5).

Outputs two CSV files:
    results_scalability.csv
    results_baseline.csv
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from pyomo.environ import value

from .instance import InstanceData, load_instance
from .model import build_model
from .baseline import build_baseline_model
from .solve import solve_model, extract_solution


# -----------------------------------------------------------------------
#  Single-instance runner
# -----------------------------------------------------------------------


def run_instance(
    inst_path: str | Path,
    solver: str = "gurobi",
    time_limit: float = 3600.0,
    mip_gap: float = 0.005,
    run_baseline: bool = True,
    tee: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Build, solve and compare on a single instance."""
    inst = load_instance(inst_path)
    record: Dict[str, Any] = {
        "instance": inst.name,
        "size_class": inst.size_class,
        "seed": inst.seed,
        "n_V": len(inst.V),
        "n_H": len(inst.H),
        "n_E_f": len(inst.E_f),
        "n_E_all": len(inst.E_all),
        "n_R": len(inst.R),
        "n_P": len(inst.P),
        "n_M": len(inst.M),
    }

    # -------- OPTIMISED MODEL --------
    if verbose:
        print(f"[{inst.name}] Building SND-RTI model ...")
    t0 = time.time()
    m_opt = build_model(inst, allow_hubs=True,
                        one_to_one_empties=False, verbose=verbose)
    build_t = time.time() - t0
    if verbose:
        print(f"[{inst.name}] Solving SND-RTI ...")
    res_opt = solve_model(m_opt, solver=solver, time_limit=time_limit,
                           mip_gap=mip_gap, tee=tee)
    sol_opt = extract_solution(m_opt) if res_opt["has_solution"] else None

    record.update({
        "opt_build_time_s": build_t,
        "opt_solve_time_s": res_opt["wall_time_s"],
        "opt_status": res_opt["termination_condition"],
        "opt_gap": res_opt["mip_gap"],
        "opt_objective": res_opt.get("objective"),
    })
    if sol_opt:
        record.update({
            "opt_TC": sol_opt["costs"]["total_transport"],
            "opt_CH": sol_opt["costs"]["total_hub"],
            "opt_PC": sol_opt["costs"]["total_purchase"],
            "opt_n_routes": sol_opt["n_active_routes"],
            "opt_n_hubs": sol_opt["n_active_hubs"],
            "opt_total_pool": sol_opt["total_pool"],
        })

    # -------- BASELINE --------
    if run_baseline:
        if verbose:
            print(f"[{inst.name}] Building baseline model ...")
        t0 = time.time()
        m_base = build_baseline_model(inst, verbose=verbose)
        b_build_t = time.time() - t0
        if verbose:
            print(f"[{inst.name}] Solving baseline ...")
        res_base = solve_model(m_base, solver=solver, time_limit=time_limit,
                               mip_gap=mip_gap, tee=tee)
        sol_base = extract_solution(m_base) if res_base["has_solution"] else None

        record.update({
            "base_build_time_s": b_build_t,
            "base_solve_time_s": res_base["wall_time_s"],
            "base_status": res_base["termination_condition"],
            "base_gap": res_base["mip_gap"],
            "base_objective": res_base.get("objective"),
        })
        if sol_base:
            record.update({
                "base_TC": sol_base["costs"]["total_transport"],
                "base_CH": sol_base["costs"]["total_hub"],
                "base_PC": sol_base["costs"]["total_purchase"],
                "base_n_routes": sol_base["n_active_routes"],
                "base_total_pool": sol_base["total_pool"],
            })

        # savings decomposition
        if sol_opt and sol_base:
            dTC = (sol_base["costs"]["total_transport"]
                   - sol_opt["costs"]["total_transport"])
            dPC = (sol_base["costs"]["total_purchase"]
                   - sol_opt["costs"]["total_purchase"])
            dCH = -sol_opt["costs"]["total_hub"]          # baseline has 0
            rel_TC = dTC / sol_base["costs"]["total_transport"] \
                if sol_base["costs"]["total_transport"] > 1e-9 else 0.0
            rel_PC = dPC / sol_base["costs"]["total_purchase"] \
                if sol_base["costs"]["total_purchase"] > 1e-9 else 0.0
            rel_total = (res_base["objective"] - res_opt["objective"]) \
                / res_base["objective"] if res_base["objective"] > 1e-9 else 0.0
            record.update({
                "delta_TC_pct": 100 * rel_TC,
                "delta_PC_pct": 100 * rel_PC,
                "delta_CH_eur_per_day": sol_opt["costs"]["total_hub"],
                "total_savings_pct": 100 * rel_total,
            })

    return record


# -----------------------------------------------------------------------
#  Batch runner
# -----------------------------------------------------------------------


def run_experiments(
    instance_dir: str | Path,
    out_dir: str | Path = "results",
    solver: str = "gurobi",
    time_limit_small: float = 600.0,
    time_limit_medium: float = 3600.0,
    time_limit_large: float = 7200.0,
    mip_gap: float = 0.005,
    run_baseline: bool = True,
    pattern: str = "*.json",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Loop over every instance JSON found in ``instance_dir``."""
    instance_dir = Path(instance_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    files = sorted(instance_dir.glob(pattern))
    if verbose:
        print(f"Found {len(files)} instance files in {instance_dir}")

    for fp in files:
        with open(fp) as f:
            header = json.load(f)
        size = header.get("size_class", "small")
        tl = {"small": time_limit_small,
              "medium": time_limit_medium,
              "large": time_limit_large}.get(size, time_limit_medium)

        try:
            rec = run_instance(
                fp, solver=solver, time_limit=tl, mip_gap=mip_gap,
                run_baseline=run_baseline, tee=False, verbose=verbose,
            )
        except Exception as e:
            rec = {"instance": fp.stem, "error": str(e)}
        records.append(rec)

        # incrementally write CSV in case we crash
        _write_csv(out_dir / "results_all.csv", records)

    # also write the per-experiment tables
    _write_scalability_table(out_dir, records)
    _write_baseline_table(out_dir, records)
    return records


# -----------------------------------------------------------------------
#  CSV writers
# -----------------------------------------------------------------------


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    keys: List[str] = []
    for r in records:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in keys})


def _write_scalability_table(out_dir: Path, records) -> None:
    fields = ["instance", "size_class", "seed",
              "n_V", "n_E_f", "n_E_all", "n_R", "n_P",
              "opt_build_time_s", "opt_solve_time_s",
              "opt_status", "opt_gap", "opt_objective"]
    rows = [{k: r.get(k, "") for k in fields} for r in records]
    with open(out_dir / "results_scalability.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_baseline_table(out_dir: Path, records) -> None:
    fields = ["instance", "size_class", "seed",
              "n_V", "n_E_f",
              "opt_objective", "base_objective",
              "delta_TC_pct", "delta_PC_pct",
              "delta_CH_eur_per_day", "total_savings_pct"]
    rows = [{k: r.get(k, "") for k in fields} for r in records
            if "base_objective" in r]
    with open(out_dir / "results_baseline.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
