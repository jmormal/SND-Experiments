"""
Pyomo MILP for the SND-RTI problem (linearised formulation).

This module builds the mixed-integer model described in
``models/base_new.tex`` and linearised in ``models/base_new_linear.tex``.

Design choices
--------------
* Sparse indexing. Variables are only created over feasible tuples
  (i.e. for which the corresponding b-parameter is 1). This keeps the
  model compact and avoids huge blocks of structural zeros.
* The variable X_{ijmqr} (route active with mode m, queue q, RTI type r)
  carries the one-mode/one-queue-per-route logic through aggregate
  derived binaries. Those aggregate binaries are created as auxiliary
  Pyomo variables only where they actually appear in constraints.
* Full-RTI routes are fixed to E_f (procurement contracts); empty-RTI
  arcs may use any pair in V x V, which is what enables consolidation
  via hubs.
* Demand is already expressed in RTI/day in the instance generator,
  so kappa_{pr} is implicitly 1. The safety stock ss_{ijp} is likewise
  in RTI units.

Only the cost objective is implemented; the CO2 objective is a
straightforward extension if emission parameters are added to the
instance.
"""
from __future__ import annotations

import pyomo.environ as pyo
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Objective,
    NonNegativeReals, NonNegativeIntegers, Binary, minimize, quicksum,
    Suffix, value,
)

from .instance import InstanceData


# =======================================================================
#  Helpers
# =======================================================================


def _big_M(inst: InstanceData) -> float:
    """Crude but safe big-M for flow-to-binary linking constraints.

    Upper bound for any per-shipment RTI count.  A route at q_max days
    between dispatches carries q_max * total_daily_demand RTIs at most.
    """
    max_d = max(inst.d.values()) if inst.d else 1.0
    q_max = max(inst.tau.values())
    # inflate to cover empty returns + inlays + slack
    return 50.0 * max_d * q_max + 1000.0


# =======================================================================
#  Model builder
# =======================================================================


def build_model(
    inst: InstanceData,
    allow_hubs: bool = True,
    one_to_one_empties: bool = False,
    verbose: bool = False,
) -> ConcreteModel:
    """Build the Pyomo ConcreteModel.

    Parameters
    ----------
    inst : InstanceData
        Parsed instance.
    allow_hubs : bool
        If False, no hub can be activated and the non-hub exclusivity
        (constraint 'hub-excl') reduces to the trivial case.
    one_to_one_empties : bool
        If True, empty-return flows are forced to mirror full-RTI routes
        (baseline). Any flow on an arc (j,i) not in the reverse of E_f
        is forbidden. No consolidation is allowed.
    verbose : bool
        Prints model statistics during construction.
    """
    m = ConcreteModel(name=inst.name)

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    m.V = Set(initialize=inst.V, ordered=True)
    m.H = Set(initialize=inst.H, ordered=True)
    m.P = Set(initialize=inst.P, ordered=True)
    m.R = Set(initialize=inst.R, ordered=True)
    m.M = Set(initialize=inst.M, ordered=True)
    m.Q = Set(initialize=inst.Q, ordered=True)

    m.E_f = Set(initialize=inst.E_f, dimen=2, ordered=True)
    m.E_all = Set(initialize=inst.E_all, dimen=2, ordered=True)

    # Mode-queue pairs that are valid on any route
    m.MQ = Set(
        initialize=[(mm, q) for mm in inst.M for q in inst.Q_m[mm]],
        dimen=2, ordered=True,
    )
    mq_valid = {(mm, q) for (mm, q) in m.MQ}

    # (i,j,m) combinations where mode m is available on (i,j)
    IJM = [(i, j, mm) for (i, j) in inst.E_all for mm in inst.modes_on[(i, j)]]
    m.IJM = Set(initialize=IJM, dimen=3, ordered=True)

    # (i,j,m,q) — the fundamental "decision" of activating a route with a given mode/q
    IJMQ = [(i, j, mm, q) for (i, j, mm) in IJM for q in inst.Q_m[mm]]
    m.IJMQ = Set(initialize=IJMQ, dimen=4, ordered=True)

    # (i,j,m,q,r) for X, N, V, NS, VS, TNC
    IJMQR = [(i, j, mm, q, r) for (i, j, mm, q) in IJMQ for r in inst.R]
    m.IJMQR = Set(initialize=IJMQR, dimen=5, ordered=True)

    # XF domain: (i,j) in E_f, p in P_ij, r in R_p, m in modes(i,j), q in Q_m
    XF_idx = []
    for (i, j) in inst.E_f:
        for p in inst.P_ij[(i, j)]:
            for r in inst.R_p[p]:
                for mm in inst.modes_on[(i, j)]:
                    for q in inst.Q_m[mm]:
                        XF_idx.append((i, j, mm, p, q, r))
    m.XF_idx = Set(initialize=XF_idx, dimen=6, ordered=True)

    # XI (inlay) domain: reverse of E_f. XI carries inlay for product p
    # back on (j,i) for every (i,j) in E_f with ric>0 and r in R_p
    XI_idx = []
    for (i, j) in inst.E_f:
        for p in inst.P_ij[(i, j)]:
            if inst.ric.get((i, j, p), 0.0) <= 0:
                continue
            for r in inst.R_p[p]:
                for mm in inst.modes_on[(j, i)]:          # reverse arc
                    for q in inst.Q_m[mm]:
                        XI_idx.append((j, i, mm, p, q, r))
    m.XI_idx = Set(initialize=XI_idx, dimen=6, ordered=True)

    if verbose:
        print(f"[model] |IJM|={len(IJM)}  |IJMQR|={len(IJMQR)}  "
              f"|XF_idx|={len(XF_idx)}  |XI_idx|={len(XI_idx)}")

    # ------------------------------------------------------------------
    # Parameters (as plain dicts for fast lookup; we reference them
    # directly inside constraints rather than as Pyomo Param objects
    # to avoid Param indexing overhead)
    # ------------------------------------------------------------------
    tau = dict(inst.tau)
    d_dem = dict(inst.d)
    ric = dict(inst.ric)
    ss = dict(inst.ss)
    ser = dict(inst.ser)
    vf = dict(inst.v_full)
    ve = dict(inst.v_empty)
    alpha = dict(inst.alpha)
    p_rti = dict(inst.p_rti)
    s_rti = dict(inst.s_rti)
    L_r = dict(inst.L_r)
    l_tr = dict(inst.l)
    c_var = dict(inst.c)
    f_fix = dict(inst.f)
    delta = dict(inst.delta)
    omega = dict(inst.omega)
    fhc = dict(inst.fhc)
    vhc = dict(inst.vhc)

    BIG_M = _big_M(inst)

    # ------------------------------------------------------------------
    # Root decision variables
    # ------------------------------------------------------------------
    # Binaries
    m.XF = Var(m.XF_idx, within=Binary)
    m.XI = Var(m.XI_idx, within=Binary)
    m.XE = Var(m.IJMQR, within=Binary)
    m.X = Var(m.IJMQR, within=Binary)

    # Daily average flow of RTIs (continuous >=0)
    m.NF = Var(m.XF_idx, within=NonNegativeReals)
    m.NI = Var(m.XI_idx, within=NonNegativeReals)
    m.NE = Var(m.IJMQR, within=NonNegativeReals)

    # Per-shipment counts (continuous >=0; scaled by tau_q, linear)
    m.NSF = Var(m.XF_idx, within=NonNegativeReals)
    m.NSI = Var(m.XI_idx, within=NonNegativeReals)
    m.NSE = Var(m.IJMQR, within=NonNegativeReals)

    # Per-shipment volumes (continuous >=0)
    m.VSF = Var(m.XF_idx, within=NonNegativeReals)
    m.VSI = Var(m.XI_idx, within=NonNegativeReals)
    m.VSE = Var(m.IJMQR, within=NonNegativeReals)

    # Total pool-covering RTIs allocated to each (i,j,m,q,r)
    m.TNCF = Var(m.XF_idx, within=NonNegativeReals)
    m.TNCI = Var(m.XI_idx, within=NonNegativeReals)
    m.TNCE = Var(m.IJMQR, within=NonNegativeReals)

    # Per-RTI-type purchased pool
    m.Ppool = Var(m.R, within=NonNegativeReals)

    # Transport costs (per day) per (i,j,m,q)
    m.TCS = Var(m.IJMQ, within=NonNegativeReals)     # per-shipment cost
    m.TC = Var(m.IJMQ, within=NonNegativeReals)      # daily cost

    # Hub activation and cost
    m.XH = Var(m.H, within=Binary)
    m.CH = Var(m.H, within=NonNegativeReals)

    # RTI amortised daily purchase cost
    m.PC = Var(m.R, within=NonNegativeReals)

    # Forbid hubs if requested
    if not allow_hubs:
        for h in m.H:
            m.XH[h].fix(0)

    # ------------------------------------------------------------------
    # Helper: aggregate expressions (Rule 1 — sum)
    # ------------------------------------------------------------------
    # Daily flows aggregated over (p) for full and inlay
    def NF_sum_p(i, j, mm, q, r):
        return quicksum(
            m.NF[i, j, mm, p, q, r]
            for p in inst.P_ij.get((i, j), [])
            if (i, j, mm, p, q, r) in XF_idx_set
        )
    XF_idx_set = set(XF_idx)

    def NI_sum_p(i, j, mm, q, r):
        return quicksum(
            m.NI[i, j, mm, p, q, r]
            for (i2, j2, mm2, p, q2, r2) in XI_idx
            if (i2, j2, mm2, q2, r2) == (i, j, mm, q, r)
        )
    # precompute product list per (i,j,mm,q,r) in XI_idx
    xi_products = {}
    for (i, j, mm, p, q, r) in XI_idx:
        xi_products.setdefault((i, j, mm, q, r), []).append(p)

    # Replace with faster version
    def NI_sum_p(i, j, mm, q, r):
        plist = xi_products.get((i, j, mm, q, r), [])
        return quicksum(m.NI[i, j, mm, p, q, r] for p in plist)

    # Volume per shipment: VSF (full) = NSF * v^f, VSE = NSE * v^e, VSI = NSI * v^f
    # We define these as equality constraints rather than substituting to
    # keep the model readable.

    # ------------------------------------------------------------------
    # C1 — Per-shipment counts: NS = N * tau_q  (linear, tau is parameter)
    # ------------------------------------------------------------------
    def c_NSF_rule(m, i, j, mm, p, q, r):
        return m.NSF[i, j, mm, p, q, r] == m.NF[i, j, mm, p, q, r] * tau[q]
    m.c_NSF = Constraint(m.XF_idx, rule=c_NSF_rule)

    def c_NSI_rule(m, i, j, mm, p, q, r):
        return m.NSI[i, j, mm, p, q, r] == m.NI[i, j, mm, p, q, r] * tau[q]
    m.c_NSI = Constraint(m.XI_idx, rule=c_NSI_rule)

    def c_NSE_rule(m, i, j, mm, q, r):
        return m.NSE[i, j, mm, q, r] == m.NE[i, j, mm, q, r] * tau[q]
    m.c_NSE = Constraint(m.IJMQR, rule=c_NSE_rule)

    # ------------------------------------------------------------------
    # C2 — Per-shipment volumes
    # ------------------------------------------------------------------
    def c_VSF_rule(m, i, j, mm, p, q, r):
        return m.VSF[i, j, mm, p, q, r] == m.NSF[i, j, mm, p, q, r] * vf[r]
    m.c_VSF = Constraint(m.XF_idx, rule=c_VSF_rule)

    def c_VSI_rule(m, i, j, mm, p, q, r):
        return m.VSI[i, j, mm, p, q, r] == m.NSI[i, j, mm, p, q, r] * vf[r]
    m.c_VSI = Constraint(m.XI_idx, rule=c_VSI_rule)

    def c_VSE_rule(m, i, j, mm, q, r):
        return m.VSE[i, j, mm, q, r] == m.NSE[i, j, mm, q, r] * ve[r]
    m.c_VSE = Constraint(m.IJMQR, rule=c_VSE_rule)

    # ------------------------------------------------------------------
    # C3 — Flow-binary linking (big-M lower + domain upper)
    # ------------------------------------------------------------------
    def c_NF_M_rule(m, i, j, mm, p, q, r):
        return m.NF[i, j, mm, p, q, r] <= BIG_M * m.XF[i, j, mm, p, q, r]
    m.c_NF_M = Constraint(m.XF_idx, rule=c_NF_M_rule)

    def c_NI_M_rule(m, i, j, mm, p, q, r):
        return m.NI[i, j, mm, p, q, r] <= BIG_M * m.XI[i, j, mm, p, q, r]
    m.c_NI_M = Constraint(m.XI_idx, rule=c_NI_M_rule)

    def c_NE_M_rule(m, i, j, mm, q, r):
        return m.NE[i, j, mm, q, r] <= BIG_M * m.XE[i, j, mm, q, r]
    m.c_NE_M = Constraint(m.IJMQR, rule=c_NE_M_rule)

    # ------------------------------------------------------------------
    # C4 — Route activation X_{ijmqr} (OR of XF_sum_p, XI_sum_p, XE)
    # ------------------------------------------------------------------
    def XF_sum_p(i, j, mm, q, r):
        plist = [p for p in inst.P_ij.get((i, j), [])
                 if (i, j, mm, p, q, r) in XF_idx_set]
        return quicksum(m.XF[i, j, mm, p, q, r] for p in plist)

    def XI_sum_p(i, j, mm, q, r):
        plist = xi_products.get((i, j, mm, q, r), [])
        return quicksum(m.XI[i, j, mm, p, q, r] for p in plist)

    # X = OR (XF_any_p, XI_any_p, XE).  We use the formulation:
    #   X  >= XF_any / |P_ij|     (X=1 if any XF is 1)
    #   X  <= XF_any + XI_any + XE  (X=0 if all 0)
    # With |P| possibly small, a cleaner form is:
    #   X  >=  XF[ijmpqr]  for every p with (i,j,m,p,q,r)∈XF_idx
    #   X  >=  XI[ijmpqr]  for every p with (i,j,m,p,q,r)∈XI_idx
    #   X  >=  XE[ijmqr]
    #   X  <=  XF_sum_p + XI_sum_p + XE

    # Upper bound: X <= sum of flow binaries (if nothing flows, X=0)
    def c_X_ub_rule(m, i, j, mm, q, r):
        return m.X[i, j, mm, q, r] <= XF_sum_p(i, j, mm, q, r) \
            + XI_sum_p(i, j, mm, q, r) + m.XE[i, j, mm, q, r]
    m.c_X_ub = Constraint(m.IJMQR, rule=c_X_ub_rule)

    # Lower bounds: X dominates each component
    def c_X_lb_E_rule(m, i, j, mm, q, r):
        return m.X[i, j, mm, q, r] >= m.XE[i, j, mm, q, r]
    m.c_X_lb_E = Constraint(m.IJMQR, rule=c_X_lb_E_rule)

    def c_X_lb_F_rule(m, i, j, mm, p, q, r):
        return m.X[i, j, mm, q, r] >= m.XF[i, j, mm, p, q, r]
    m.c_X_lb_F = Constraint(m.XF_idx, rule=c_X_lb_F_rule)

    def c_X_lb_I_rule(m, i, j, mm, p, q, r):
        return m.X[i, j, mm, q, r] >= m.XI[i, j, mm, p, q, r]
    m.c_X_lb_I = Constraint(m.XI_idx, rule=c_X_lb_I_rule)

    # ------------------------------------------------------------------
    # C5 — One (m,q) pair per route (combines eqs. mode-amo + only-one-q)
    #
    # The paper asks for two separate at-most-one constraints on m and q,
    # but since a route only exists when a specific (m,q) is chosen, we
    # directly enforce  sum over (m,q,r) of X <= |R|  and  sum over (m,q)
    # of X_any_r <= 1.   We also keep the per-mode bm availability
    # implicit in IJMQ (infeasible (m,q) are simply not in the set).
    # ------------------------------------------------------------------
    # For each (i,j) introduce aggregate binary Y_{ij,m,q} = OR over r
    # of X_{ijmqr}; at most one (m,q) combination per (i,j).
    m.Y = Var(m.IJMQ, within=Binary)       # Y[i,j,m,q] = route active with (m,q)

    def c_Y_def_rule(m, i, j, mm, q, r):
        return m.Y[i, j, mm, q] >= m.X[i, j, mm, q, r]
    m.c_Y_def = Constraint(m.IJMQR, rule=c_Y_def_rule)

    # Y(i,j,m,q) = 0  if all X(i,j,m,q,·) = 0   (via route-activation expression below)
    def c_Y_ub_rule(m, i, j, mm, q):
        return m.Y[i, j, mm, q] <= quicksum(
            m.X[i, j, mm, q, r] for r in inst.R
        )
    m.c_Y_ub = Constraint(m.IJMQ, rule=c_Y_ub_rule)

    def c_one_mq_per_route_rule(m, i, j):
        # sum over (m,q) <= 1: at most one mode-queue combo for each (i,j)
        return quicksum(
            m.Y[i, j, mm, q]
            for mm in inst.modes_on[(i, j)]
            for q in inst.Q_m[mm]
        ) <= 1
    m.c_one_mq_per_route = Constraint(m.E_all, rule=c_one_mq_per_route_rule)

    # ------------------------------------------------------------------
    # C6 — Full-route activation:  for every (i,j) in E_f and p in P_ij,
    #      exactly one (m,q,r) combination must carry product p.
    # ------------------------------------------------------------------
    def c_full_activation_rule(m, i, j, p):
        if (i, j) not in inst.E_f or p not in inst.P_ij[(i, j)]:
            return Constraint.Skip
        return quicksum(
            m.XF[i, j, mm, p, q, r]
            for mm in inst.modes_on[(i, j)]
            for q in inst.Q_m[mm]
            for r in inst.R_p[p]
        ) == 1

    m.c_full_activation = Constraint(
        [(i, j, p) for (i, j) in inst.E_f for p in inst.P_ij[(i, j)]],
        rule=c_full_activation_rule,
    )

    # ------------------------------------------------------------------
    # C7 — Demand satisfaction:  NF * kappa >= d * XF   (kappa=1)
    # ------------------------------------------------------------------
    def c_demand_rule(m, i, j, mm, p, q, r):
        return m.NF[i, j, mm, p, q, r] >= d_dem[(i, j, p)] * m.XF[i, j, mm, p, q, r]
    m.c_demand = Constraint(m.XF_idx, rule=c_demand_rule)

    # ------------------------------------------------------------------
    # C8 — Inlay return flows. When product p on (i,j) uses RTI r with
    #      (m,q), an inlay return must exist on (j,i).
    #      We keep the simpler form from the paper:
    #          sum over (m,q) of XI[j,i,m,p,q,r] = sum over (m,q) of XF[i,j,m,p,q,r]
    #      and  sum over (m,q) of NI[j,i,m,p,q,r] >= ric * sum over (m,q) of NF[i,j,m,p,q,r]
    # ------------------------------------------------------------------
    def c_inlay_bin_rule(m, i, j, p, r):
        if (i, j) not in inst.E_f or p not in inst.P_ij[(i, j)]:
            return Constraint.Skip
        if ric.get((i, j, p), 0.0) <= 0:
            return Constraint.Skip
        if r not in inst.R_p[p]:
            return Constraint.Skip
        lhs = quicksum(
            m.XI[j, i, mm, p, q, r]
            for mm in inst.modes_on[(j, i)]
            for q in inst.Q_m[mm]
        )
        rhs = quicksum(
            m.XF[i, j, mm, p, q, r]
            for mm in inst.modes_on[(i, j)]
            for q in inst.Q_m[mm]
        )
        return lhs == rhs

    c_inlay_bin_idx = [
        (i, j, p, r) for (i, j) in inst.E_f
        for p in inst.P_ij[(i, j)]
        if ric.get((i, j, p), 0.0) > 0
        for r in inst.R_p[p]
    ]
    m.c_inlay_bin = Constraint(c_inlay_bin_idx, rule=c_inlay_bin_rule)

    def c_inlay_flow_rule(m, i, j, p, r):
        lhs = quicksum(
            m.NI[j, i, mm, p, q, r]
            for mm in inst.modes_on[(j, i)]
            for q in inst.Q_m[mm]
        )
        rhs = quicksum(
            m.NF[i, j, mm, p, q, r]
            for mm in inst.modes_on[(i, j)]
            for q in inst.Q_m[mm]
        )
        return lhs >= ric[(i, j, p)] * rhs

    m.c_inlay_flow = Constraint(c_inlay_bin_idx, rule=c_inlay_flow_rule)

    # ------------------------------------------------------------------
    # C9 — Flow conservation (per node, per RTI type)
    # ------------------------------------------------------------------
    # Total outflow = total inflow of RTIs of type r at node i.
    # We sum the combined flow N = NF_sum_p + NI_sum_p + NE across m,q.
    def node_flow_expr(i, r, direction):
        expr = 0
        for (ii, jj) in inst.E_all:
            if direction == "out" and ii != i:
                continue
            if direction == "in" and jj != i:
                continue
            for mm in inst.modes_on[(ii, jj)]:
                for q in inst.Q_m[mm]:
                    expr += m.NE[ii, jj, mm, q, r]
                    # Full RTIs (from XF_idx, only on E_f)
                    if (ii, jj) in inst.P_ij:
                        for p in inst.P_ij[(ii, jj)]:
                            if (ii, jj, mm, p, q, r) in XF_idx_set:
                                expr += m.NF[ii, jj, mm, p, q, r]
                    # Inlay flows
                    plist = xi_products.get((ii, jj, mm, q, r), [])
                    for p in plist:
                        expr += m.NI[ii, jj, mm, p, q, r]
        return expr

    def c_conservation_rule(m, i, r):
        return node_flow_expr(i, r, "out") == node_flow_expr(i, r, "in")
    m.c_conservation = Constraint(m.V, m.R, rule=c_conservation_rule)

    # ------------------------------------------------------------------
    # C10 — Shipment volume bounds (per active (i,j,m,q))
    # ------------------------------------------------------------------
    def total_VS(i, j, mm, q):
        expr = 0
        for r in inst.R:
            expr += m.VSE[i, j, mm, q, r]
            if (i, j) in inst.P_ij:
                for p in inst.P_ij[(i, j)]:
                    if (i, j, mm, p, q, r) in XF_idx_set:
                        expr += m.VSF[i, j, mm, p, q, r]
            plist = xi_products.get((i, j, mm, q, r), [])
            for p in plist:
                expr += m.VSI[i, j, mm, p, q, r]
        return expr

    def c_vol_max_rule(m, i, j, mm, q):
        return total_VS(i, j, mm, q) <= delta[mm] * m.Y[i, j, mm, q]
    m.c_vol_max = Constraint(m.IJMQ, rule=c_vol_max_rule)

    def c_vol_min_rule(m, i, j, mm, q):
        return total_VS(i, j, mm, q) >= omega[mm] * m.Y[i, j, mm, q]
    m.c_vol_min = Constraint(m.IJMQ, rule=c_vol_min_rule)

    # ------------------------------------------------------------------
    # C11 — Hub exclusivity for non-hub nodes (per RTI type)
    #
    # A non-hub node cannot simultaneously send and receive empties of
    # the same type r.  Written with OR-aggregates XE_out / XE_in:
    # ------------------------------------------------------------------
    m.XE_out = Var(m.V, m.R, within=Binary)   # dispatches any empties of type r
    m.XE_in = Var(m.V, m.R, within=Binary)    # receives any empties of type r

    def c_XE_out_ub_rule(m, i, r):
        return m.XE_out[i, r] <= quicksum(
            m.XE[i, j, mm, q, r]
            for j in inst.V if j != i
            for mm in inst.modes_on[(i, j)]
            for q in inst.Q_m[mm]
        )

    def c_XE_out_lb_rule(m, i, j, mm, q, r):
        if i == j:
            return Constraint.Skip
        return m.XE_out[i, r] >= m.XE[i, j, mm, q, r]

    def c_XE_in_ub_rule(m, j, r):
        return m.XE_in[j, r] <= quicksum(
            m.XE[i, j, mm, q, r]
            for i in inst.V if i != j
            for mm in inst.modes_on[(i, j)]
            for q in inst.Q_m[mm]
        )

    def c_XE_in_lb_rule(m, i, j, mm, q, r):
        if i == j:
            return Constraint.Skip
        return m.XE_in[j, r] >= m.XE[i, j, mm, q, r]

    m.c_XE_out_ub = Constraint(m.V, m.R, rule=c_XE_out_ub_rule)
    m.c_XE_out_lb = Constraint(m.IJMQR, rule=c_XE_out_lb_rule)
    m.c_XE_in_ub = Constraint(m.V, m.R, rule=c_XE_in_ub_rule)
    m.c_XE_in_lb = Constraint(m.IJMQR, rule=c_XE_in_lb_rule)

    def c_non_hub_excl_rule(m, i, r):
        if i in inst.H:
            return Constraint.Skip
        return m.XE_out[i, r] + m.XE_in[i, r] <= 1
    m.c_non_hub_excl = Constraint(m.V, m.R, rule=c_non_hub_excl_rule)

    # ------------------------------------------------------------------
    # C12 — Hub activation XH and handling cost CH
    # ------------------------------------------------------------------
    # XH = OR over (m,q) of Y[i,*,m,q] — hub i active if it dispatches
    # anything in either direction.  Use a route-level OR of Y.
    def hub_activity(i):
        out_expr = quicksum(
            m.Y[i, j, mm, q]
            for j in inst.V if j != i
            for mm in inst.modes_on[(i, j)]
            for q in inst.Q_m[mm]
        )
        in_expr = quicksum(
            m.Y[j, i, mm, q]
            for j in inst.V if j != i
            for mm in inst.modes_on[(j, i)]
            for q in inst.Q_m[mm]
        )
        return out_expr + in_expr

    def c_hub_act_rule(m, h):
        # Activity cannot exceed BIG_M * XH
        return hub_activity(h) <= BIG_M * m.XH[h]
    m.c_hub_act = Constraint(m.H, rule=c_hub_act_rule)

    # Daily hub volume (continuous) — sum of VSE passing through hub's outflow
    # converted to daily by dividing VSE / tau.  Since VSE = VE * tau,
    # the daily volume is just VE (= NE * v^e).   Similarly include
    # incoming handled volume (empty only — full / inlay already sit at
    # their endpoint).
    def hub_daily_vol(h):
        expr = 0
        for j in inst.V:
            if j == h:
                continue
            for mm in inst.modes_on[(h, j)]:
                for q in inst.Q_m[mm]:
                    for r in inst.R:
                        expr += m.NE[h, j, mm, q, r] * ve[r]
        return expr

    def c_CH_rule(m, h):
        return m.CH[h] == fhc[h] * m.XH[h] + vhc[h] * hub_daily_vol(h)
    m.c_CH = Constraint(m.H, rule=c_CH_rule)

    # ------------------------------------------------------------------
    # C13 — RTI pool (total number of RTIs allocated, per arc / type)
    # ------------------------------------------------------------------
    def c_TNCF_rule(m, i, j, mm, p, q, r):
        return m.TNCF[i, j, mm, p, q, r] == (
            m.NSF[i, j, mm, p, q, r]
            + l_tr[(i, j, mm)] * m.NF[i, j, mm, p, q, r]
            + m.XF[i, j, mm, p, q, r] * ss[(i, j, p)]
        )
    m.c_TNCF = Constraint(m.XF_idx, rule=c_TNCF_rule)

    def c_TNCI_rule(m, i, j, mm, p, q, r):
        # inlay: no safety stock
        return m.TNCI[i, j, mm, p, q, r] == (
            m.NSI[i, j, mm, p, q, r]
            + l_tr[(i, j, mm)] * m.NI[i, j, mm, p, q, r]
        )
    m.c_TNCI = Constraint(m.XI_idx, rule=c_TNCI_rule)

    def c_TNCE_rule(m, i, j, mm, q, r):
        # safety empties at the origin node: ss_{jip} * ser_{jip} / kappa
        # In the paper the empty-safety term is attached to the reverse arc.
        # We associate it with the outgoing empty arc (i,j) coming from
        # the full route terminus:
        empty_ss = 0
        if (j, i) in inst.P_ij:          # reverse exists as full route
            for p in inst.P_ij[(j, i)]:
                if r in inst.R_p[p]:
                    for mm2 in inst.modes_on[(j, i)]:
                        for q2 in inst.Q_m[mm2]:
                            empty_ss += (
                                m.XF[j, i, mm2, p, q2, r]
                                * ss[(j, i, p)] * ser[(j, i, p)]
                            )
        return m.TNCE[i, j, mm, q, r] == (
            m.NSE[i, j, mm, q, r]
            + l_tr[(i, j, mm)] * m.NE[i, j, mm, q, r]
            + empty_ss
        )
    m.c_TNCE = Constraint(m.IJMQR, rule=c_TNCE_rule)

    # Total pool required of type r across the network
    def total_pool_r(r):
        expr = 0
        for (i, j, mm, p, q, rr) in XF_idx:
            if rr != r:
                continue
            expr += (1 + alpha[r]) * m.TNCF[i, j, mm, p, q, rr]
        for (i, j, mm, p, q, rr) in XI_idx:
            if rr != r:
                continue
            expr += (1 + alpha[r]) * m.TNCI[i, j, mm, p, q, rr]
        for (i, j, mm, q, rr) in IJMQR:
            if rr != r:
                continue
            expr += (1 + alpha[r]) * m.TNCE[i, j, mm, q, rr]
        return expr

    # RTI pool balance: existing pool + purchased >= required
    def c_pool_rule(m, r):
        return s_rti[r] + m.Ppool[r] >= total_pool_r(r)
    m.c_pool = Constraint(m.R, rule=c_pool_rule)

    # Amortised daily purchase cost
    def c_PC_rule(m, r):
        return m.PC[r] == p_rti[r] * m.Ppool[r] / L_r[r]
    m.c_PC = Constraint(m.R, rule=c_PC_rule)

    # ------------------------------------------------------------------
    # C14 — Transport cost:  TCS = c*VS + f*Y ;  TC * tau = TCS
    # ------------------------------------------------------------------
    def c_TCS_rule(m, i, j, mm, q):
        return m.TCS[i, j, mm, q] == (
            c_var[(i, j, mm)] * total_VS(i, j, mm, q)
            + f_fix[(i, j, mm)] * m.Y[i, j, mm, q]
        )
    m.c_TCS = Constraint(m.IJMQ, rule=c_TCS_rule)

    def c_TC_rule(m, i, j, mm, q):
        return m.TC[i, j, mm, q] * tau[q] == m.TCS[i, j, mm, q]
    m.c_TC = Constraint(m.IJMQ, rule=c_TC_rule)

    # ------------------------------------------------------------------
    # C15 (optional) — One-to-one empties (baseline mode)
    # ------------------------------------------------------------------
    if one_to_one_empties:
        # Allow XE on (j,i) only if (i,j) ∈ E_f.
        reverse_full = {(j, i) for (i, j) in inst.E_f}
        for (i, j, mm, q, r) in IJMQR:
            if (i, j) not in reverse_full:
                m.XE[i, j, mm, q, r].fix(0)
                m.NE[i, j, mm, q, r].fix(0)
        # Hubs off
        for h in inst.H:
            m.XH[h].fix(0)

    # ------------------------------------------------------------------
    # Objective: minimise daily economic cost
    # ------------------------------------------------------------------
    m.total_TC = Var(within=NonNegativeReals)
    m.total_CH = Var(within=NonNegativeReals)
    m.total_PC = Var(within=NonNegativeReals)

    def c_total_TC_rule(m):
        return m.total_TC == quicksum(
            m.TC[i, j, mm, q] for (i, j, mm, q) in IJMQ
        )
    m.c_total_TC = Constraint(rule=c_total_TC_rule)

    def c_total_CH_rule(m):
        return m.total_CH == quicksum(m.CH[h] for h in inst.H)
    m.c_total_CH = Constraint(rule=c_total_CH_rule)

    def c_total_PC_rule(m):
        return m.total_PC == quicksum(m.PC[r] for r in inst.R)
    m.c_total_PC = Constraint(rule=c_total_PC_rule)

    m.obj = Objective(
        expr=m.total_TC + m.total_CH + m.total_PC,
        sense=minimize,
    )

    # Attach the instance for later use
    m._instance = inst
    m._options = {
        "allow_hubs": allow_hubs,
        "one_to_one_empties": one_to_one_empties,
    }

    if verbose:
        n_bin = sum(1 for v_ in m.component_data_objects(Var)
                    if v_.is_binary() and not v_.fixed)
        n_cont = sum(1 for v_ in m.component_data_objects(Var)
                     if not v_.is_binary() and not v_.fixed)
        n_con = sum(1 for _ in m.component_data_objects(Constraint))
        print(f"[model] binaries={n_bin}  continuous={n_cont}  constraints={n_con}")

    return m
