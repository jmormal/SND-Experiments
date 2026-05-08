"""
Solver wrapper and solution extractor.

Tries the provided solver (default gurobi) first, falls back to
``appsi_highs`` and finally ``cbc`` if available.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import pyomo.environ as pyo
from pyomo.environ import SolverFactory, value
from pyomo.opt import SolverStatus, TerminationCondition


# -----------------------------------------------------------------------
#  Solver selection
# -----------------------------------------------------------------------


def _get_solver(solver_name: str, time_limit: float, mip_gap: float):
    """Return a configured Pyomo SolverFactory instance."""
    opt = SolverFactory(solver_name)
    if not opt.available(exception_flag=False):
        raise RuntimeError(f"Solver '{solver_name}' is not available.")

    # Map common options per solver
    if solver_name.lower() in ("gurobi", "gurobi_direct", "gurobi_persistent"):
        opt.options["TimeLimit"] = time_limit
        opt.options["MIPGap"] = mip_gap
        opt.options["OutputFlag"] = 1
    elif solver_name.lower() in ("cbc",):
        opt.options["sec"] = time_limit
        opt.options["ratio"] = mip_gap
    elif solver_name.lower() in ("glpk",):
        opt.options["tmlim"] = int(time_limit)
        opt.options["mipgap"] = mip_gap
    elif solver_name.lower() in ("appsi_highs", "highs"):
        opt.options["time_limit"] = time_limit
        opt.options["mip_rel_gap"] = mip_gap
    return opt


def solve_model(
    model,
    solver: str = "gurobi",
    time_limit: float = 3600.0,
    mip_gap: float = 0.005,
    tee: bool = True,
    fallback_solvers: Tuple[str, ...] = ("appsi_highs", "cbc", "glpk"),
) -> dict:
    """Solve a built Pyomo model and return a results dict.

    Handles the case where a MIP solver hits its time limit *without*
    finding any feasible solution (HiGHS raises an exception there,
    Gurobi does not).
    """
    last_err = None
    opt = None
    sname = solver
    for sname in (solver, *fallback_solvers):
        try:
            opt = _get_solver(sname, time_limit, mip_gap)
            break
        except RuntimeError as e:
            last_err = e
            opt = None
            print(e)
    if opt is None:
        raise RuntimeError(f"No solver available. Last error: {last_err}")

    print(solver, tee, opt)

    t0 = time.time()
    err_msg = None
    results = None
    try:
        # Ask Pyomo not to auto-load the solution; we do it ourselves
        # after checking the termination condition.
        results = opt.solve(model, tee=True, load_solutions=False)
    except TypeError:
        # Some solver interfaces don't support load_solutions kwarg
        try:
            results = opt.solve(model, tee=True)
        except Exception as e:  # no feasible solution found
            err_msg = str(e)
    except Exception as e:
        err_msg = str(e)
    t_elapsed = time.time() - t0

    if results is None:
        return {
            "solver": sname,
            "termination_condition": "no_feasible_solution_in_time",
            "wall_time_s": t_elapsed,
            "mip_gap": None,
            "has_solution": False,
            "error": err_msg,
        }

    term = results.solver.termination_condition
    status_ok = term in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    )
    # Accept "maxTimeLimit" only if a feasible solution was actually found
    if term == TerminationCondition.maxTimeLimit:
        try:
            # Best feasible upper bound present?
            ub = results.problem.upper_bound
            status_ok = ub is not None and abs(ub) < float("inf")
        except Exception:
            status_ok = False

    # Load the solution only if one exists
    if status_ok:
        try:
            model.solutions.load_from(results)
        except Exception as e:
            err_msg = f"solution present but failed to load: {e}"
            status_ok = False

    # Try to recover a MIP gap from the solver output
    gap = None
    try:
        ub = results.problem.upper_bound
        lb = results.problem.lower_bound
        if ub is not None and lb is not None and abs(ub) > 1e-9:
            gap = (ub - lb) / abs(ub)
    except Exception:
        pass

    info = {
        "solver": sname,
        "termination_condition": str(term),
        "wall_time_s": t_elapsed,
        "mip_gap": gap,
        "has_solution": status_ok,
    }
    if err_msg:
        info["error"] = err_msg
    if status_ok:
        try:
            info["objective"] = value(model.obj)
        except Exception:
            info["objective"] = None
    return info


# -----------------------------------------------------------------------
#  Solution extraction
# -----------------------------------------------------------------------


def extract_solution(model) -> dict:
    """Extract a compact summary from a solved model."""
    inst = model._instance

    def _v(x, default=0.0):
        try:
            return value(x)
        except Exception:
            return default

    # active routes
    active_routes = []
    for i, j, mm, q in model.IJMQ:
        y = _v(model.Y[i, j, mm, q])
        if y is not None and y > 0.5:
            active_routes.append(
                {"origin": i, "destination": j, "mode": mm, "q": q, "tau": inst.tau[q]}
            )

    # active hubs
    active_hubs = [h for h in inst.H if _v(model.XH[h]) > 0.5]

    # pool purchased
    pool = {r: _v(model.Ppool[r]) for r in inst.R}

    # cost components
    costs = {
        "total_transport": _v(model.total_TC),
        "total_hub": _v(model.total_CH),
        "total_purchase": _v(model.total_PC),
        "objective": _v(model.obj),
    }

    # per-route flow summary
    flow_summary = []
    for i, j, mm, q in model.IJMQ:
        y = _v(model.Y[i, j, mm, q])
        if y is None or y <= 0.5:
            continue
        tcs = _v(model.TCS[i, j, mm, q])
        tc = _v(model.TC[i, j, mm, q])
        # total volume in a single shipment
        vs = 0.0
        for r in inst.R:
            vs += _v(model.VSE[i, j, mm, q, r])
            if (i, j) in inst.P_ij:
                for p in inst.P_ij[(i, j)]:
                    key = (i, j, mm, p, q, r)
                    if key in model.XF_idx:
                        vs += _v(model.VSF[key])
            # inlay
            for p_, _, _, _, _, _ in []:  # placeholder
                pass
        flow_summary.append(
            {
                "origin": i,
                "destination": j,
                "mode": mm,
                "q": q,
                "tau": inst.tau[q],
                "shipment_volume_m3": vs,
                "TCS": tcs,
                "TC": tc,
            }
        )

    return {
        "costs": costs,
        "active_routes": active_routes,
        "n_active_routes": len(active_routes),
        "active_hubs": active_hubs,
        "n_active_hubs": len(active_hubs),
        "pool_purchased": pool,
        "total_pool": sum(pool.values()),
        "flow_summary": flow_summary,
    }
