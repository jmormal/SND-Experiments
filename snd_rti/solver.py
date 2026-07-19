#!/usr/bin/env python3
"""
solve_snd_rti.py — MILP solver for the SND-RTI network design problem.

Implements the *linearised* model of base_new.tex / base_new_linear.tex:
the queue time q_{ijm} is discretised over a set Q of queue periods with
frequency tau_q (days), so that all per-shipment quantities
(NSF = NF * tau_q, VSF = VF * tau_q, ...) and the daily transport costs
(TC = TCS / tau_q) become linear.

Notation-convention variables (overline / widecheck / widehat) are not
materialised: sums are written inline, OR-variables are replaced by the
equivalent linking inequalities, and the at-most-one constraint
(eq:full-activation) is written directly as
        sum_{m,q,r} XF_{ijmpqr} = bf_{ijp}.

Derived (substituted, not declared) quantities — kept as affine
expressions to shrink the model:
    VF = NF * v^f_r            VE = NE * v^e_r          (eq:vol-F..E)
    NSF = NF * tau_q  etc.                              (eq:ship-*)
    VS_{ijmq} = tau_q * dailyVol_{ijmq}                 (eq:ship-VS)
    TCS = VS*vtc + Y*ftc ;  TC = TCS / tau_q            (eq:cost-*)
    TNC* pool terms                                     (eq:pool-*)

Usage
-----
  # solve a generated JSON instance
  python solve_snd_rti.py --json SND_RTI_small_42.json

  # generate + solve in one go (imports your instance_generator.py)
  python solve_snd_rti.py --generate small --seed 42

  # bi-objective handling
  python solve_snd_rti.py --json inst.json --weight 1.0          # pure cost
  python solve_snd_rti.py --json inst.json --weight 0.0          # pure CO2
  python solve_snd_rti.py --json inst.json --weight 1.0 --eps-co2 5000
                                                                 # eps-constraint

  # solvers (PuLP backends): CBC (default, bundled), HiGHS, GUROBI, CPLEX
  python solve_snd_rti.py --json inst.json --solver HIGHS

Requires:  pip install pulp        (CBC bundled)
Place this file next to classes_pydantic.py / instance_generator.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import pulp


# ════════════════════════════════════════════════════════════════════
#  ADAPTERS — tolerate both live pydantic objects and JSON round-trips
# ════════════════════════════════════════════════════════════════════


def obj_name(x) -> str:
    """Name of a Product / RTI / plain string key."""
    n = getattr(x, "name", None)
    return str(n) if n is not None else str(x)


def as_mid(k) -> int:
    """Mode id from a Mode object, int, or stringified key."""
    if hasattr(k, "id"):
        return int(k.id)
    try:
        return int(k)
    except (TypeError, ValueError):
        s = str(k)
        for tok in s.replace("=", " ").replace(",", " ").split():
            if tok.isdigit():
                return int(tok)
        raise ValueError(f"Cannot extract a mode id from key {k!r}")


def mode_keyed(d) -> dict:
    """Re-key a {Mode|id|str: value} dict by integer mode id."""
    return {as_mid(k): v for k, v in (d or {}).items()}


def rti_volumes(rti) -> tuple[float, float]:
    """(v^f_r, v^e_r): volume of a full and of a folded/empty RTI."""
    vf = getattr(rti, "volume_full", None)
    if vf is None:
        vf = rti.length * rti.width * rti.full_height
    ve = getattr(rti, "volume_folded", None) or getattr(rti, "volume_empty", None)
    if ve is None:
        ve = rti.length * rti.width * rti.folded_height
    return float(vf), float(ve)


def rti_compat(rti) -> dict[str, tuple[float, float]]:
    """
    Return {product_name: (kappa_pr, ric_pr)} for the products this RTI can
    carry, read from RTI.product_capacity and RTI.return_inlay_fraction
    (both keyed by Product objects, as defined in classes_pydantic.py).
    An RTI with no compatible products simply never carries fulls/inlays.
    """
    inl = {
        obj_name(k): float(v)
        for k, v in (getattr(rti, "return_inlay_fraction", {}) or {}).items()
    }
    res: dict[str, tuple[float, float]] = {}
    for k, cap in (getattr(rti, "product_capacity", {}) or {}).items():
        name = obj_name(k)
        if float(cap) >= 1.0:
            res[name] = (float(cap), inl.get(name, 0.0))
    return res


# ════════════════════════════════════════════════════════════════════
#  INSTANCE LOADING
# ════════════════════════════════════════════════════════════════════


def load_network(args):
    if args.json:
        from classes_pydantic import Network

        # Network.load_json rebuilds Product/Mode-keyed dicts from the JSON
        if hasattr(Network, "load_json"):
            return Network.load_json(args.json)
        raw = open(args.json).read()
        for loader in ("model_validate_json", "parse_raw"):
            fn = getattr(Network, loader, None)
            if fn:
                try:
                    return fn(raw)
                except Exception:
                    pass
        raise RuntimeError(f"Could not parse {args.json} as a Network.")
    elif args.generate:
        from instance_generator import generate_instance

        return generate_instance(args.generate, args.seed)
    else:
        raise SystemExit("Provide --json PATH or --generate {small,medium,large}.")


# ════════════════════════════════════════════════════════════════════
#  DATA MODEL FOR THE SOLVER
# ════════════════════════════════════════════════════════════════════


def node_p(pid) -> str:
    return f"P{int(pid)}"


def node_h(hid) -> str:
    return f"H{int(hid)}"


@dataclass
class Arc:
    """A directed edge (i,j) in E with mode-indexed parameters."""

    src: str
    dst: str
    dist: float
    kind: str  # "demand" | "reverse" | "hub"
    # mid -> {lt, ftc, vtc, fte, vte}
    modes: dict = field(default_factory=dict)
    # pname -> d_ijp  (bf_ijp = d>0)
    demand: dict = field(default_factory=dict)
    ss: dict = field(default_factory=dict)  # pname -> ss_ijp  [days]
    ser: dict = field(default_factory=dict)  # pname -> ser_ijp [days]
    allow_empty: bool = True  # be_ijr (uniform across r)


def build_data(net, args):
    # ---- nodes --------------------------------------------------------
    plants = {int(p.id): p for p in net.plants.values()}
    hubs = {int(h.id): h for h in net.hubs.values()}
    coords = {node_p(i): (p.x, p.y) for i, p in plants.items()}
    coords.update({node_h(i): (h.x, h.y) for i, h in hubs.items()})
    hub_nodes = sorted(node_h(i) for i in hubs)

    # ---- modes / queue periods Q ---------------------------------------
    modes = {as_mid(getattr(m, "id", k)): m for k, m in net.modes.items()}
    Qm: dict[int, list[int]] = {}
    for mid, m in modes.items():
        qs = list(range(int(m.min_q), int(m.max_q) + 1))
        if args.q_grid:
            qs = [q for q in qs if q in args.q_grid]
            if not qs:
                qs = [int(m.min_q)]
        Qm[mid] = qs  # tau_q = q (days)

    # ---- products / RTIs -----------------------------------------------
    products = {obj_name(p): p for p in net.products.values()}
    rtis = {int(r.id): r for r in net.rtis.values()}
    vf, ve, life_days, stock, pc, pe = {}, {}, {}, {}, {}, {}
    compat_pr: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for rid, r in rtis.items():
        vf[rid], ve[rid] = rti_volumes(r)
        life_days[rid] = float(r.average_useful_life) * 365.0  # years -> days
        stock[rid] = float(r.current_stock)
        pc[rid] = float(r.purchase_cost)
        pe[rid] = float(r.embodied_co2)
        for pname, (kappa, ric) in rti_compat(r).items():
            # bp_pr, kappa, ric
            compat_pr[pname].append((rid, kappa, ric))

    # ---- arcs ------------------------------------------------------------
    def dist_between(a, b):
        (x1, y1), (x2, y2) = coords[a], coords[b]
        return math.hypot(x1 - x2, y1 - y2)

    def synth_mode_params(dist):
        out = {}
        for mid, m in modes.items():
            out[mid] = dict(
                lt=float(dist // args.v_avg + 1),
                ftc=float(m.fixed_economical_cost),
                vtc=float(m.economic_cost_per_km * dist),
                fte=float(m.fixed_environmental_cost),
                vte=float(m.environmental_cost_per_km * dist),
            )
        return out

    arcs: dict[tuple[str, str], Arc] = {}

    def node_of(obj) -> str:
        """Plant -> P{id}, Hub -> H{id} (Edge.source/target can be either)."""
        return node_h(obj.id) if type(obj).__name__ == "Hub" else node_p(obj.id)

    # arcs straight from the instance (demand arcs + any pre-built empty arcs,
    # e.g. those created by Network.add_edges_model)
    for e in net.edges.values():
        s, t = node_of(e.source), node_of(e.target)
        lt = mode_keyed(e.lead_time)
        fec = mode_keyed(e.fixed_economic_cost_per_volume)
        vec = mode_keyed(e.variable_economic_cost_per_volume)
        fco = mode_keyed(e.fixed_co2_cost_per_volume)
        vco = mode_keyed(e.variable_co2_cost_per_volume)
        a = arcs.setdefault((s, t), Arc(s, t, dist_between(s, t), "given"))
        a.kind = "demand" if e.demand else a.kind
        for mk in e.allowed_modes or list(modes.values()):
            mid = as_mid(mk)
            a.modes[mid] = dict(
                lt=float(lt.get(mid, a.dist // args.v_avg + 1)),
                ftc=float(fec.get(mid, modes[mid].fixed_economical_cost)),
                vtc=float(vec.get(mid, modes[mid].economic_cost_per_km * a.dist)),
                fte=float(fco.get(mid, modes[mid].fixed_environmental_cost)),
                vte=float(vco.get(mid, modes[mid].environmental_cost_per_km * a.dist)),
            )
        for pk, d in e.demand.items():
            pname = obj_name(pk)
            d = float(d)
            if d <= 0:
                continue
            a.demand[pname] = a.demand.get(pname, 0.0) + d  # bf_ijp = 1
            ssf = {
                obj_name(k): float(v) for k, v in (e.safety_stocks_fulls or {}).items()
            }
            sse = {
                obj_name(k): float(v) for k, v in (e.safety_stock_empties or {}).items()
            }
            # instance stores safety stocks in *units*; the model wants days
            a.ss[pname] = ssf.get(pname, 0.0) / d
            a.ser[pname] = sse.get(pname, 0.0) / d

    # reverse arcs (inlay returns + empty repositioning)
    for s, t in list(arcs.keys()):
        if (t, s) not in arcs:
            d = dist_between(t, s)
            arcs[(t, s)] = Arc(t, s, d, "reverse", modes=synth_mode_params(d))

    # plant <-> hub arcs for empty consolidation (k nearest hubs)
    if hub_nodes and args.hub_k > 0:
        plant_nodes = sorted(node_p(i) for i in plants)
        for pn in plant_nodes:
            nearest = sorted(hub_nodes, key=lambda h: dist_between(pn, h))[: args.hub_k]
            for hn in nearest:
                for s, t in ((pn, hn), (hn, pn)):
                    if (s, t) not in arcs:
                        d = dist_between(s, t)
                        arcs[(s, t)] = Arc(s, t, d, "hub", modes=synth_mode_params(d))

    # ---- big-M for empty/inlay flows: total full-RTI circulation bound ----
    M_flow = sum(d for a in arcs.values() for d in a.demand.values()) + 1.0

    nodes = sorted({n for (s, t) in arcs for n in (s, t)})
    return dict(
        nodes=nodes,
        hub_nodes=hub_nodes,
        hubs=hubs,
        arcs=arcs,
        modes=modes,
        Qm=Qm,
        products=products,
        rtis=rtis,
        vf=vf,
        ve=ve,
        life_days=life_days,
        stock=stock,
        pc=pc,
        pe=pe,
        compat=compat_pr,
        M_flow=M_flow,
    )


# ════════════════════════════════════════════════════════════════════
#  MODEL
# ════════════════════════════════════════════════════════════════════


def vname(prefix, *key):
    return prefix + "_" + "_".join(str(k) for k in key)


def build_model(data, args):
    arcs, Qm = data["arcs"], data["Qm"]
    vf, ve, compat = data["vf"], data["ve"], data["compat"]
    M_flow = data["M_flow"]
    R = sorted(data["rtis"].keys())

    prob = pulp.LpProblem("SND_RTI_linear", pulp.LpMinimize)

    # ---- index sets (sparse) -------------------------------------------
    AMQ = []  # (a, m, q): admissible route/mode/frequency
    FULL = []  # (a, m, q, p, r)
    INLAY = []  # (a, m, q, p, r) on the *reverse* arc
    EMPTY = []  # (a, m, q, r)
    for ak, a in arcs.items():
        for mid in a.modes:
            for q in Qm[mid]:
                AMQ.append((ak, mid, q))
                if a.allow_empty:  # be_ijr * bm_ijm
                    EMPTY += [(ak, mid, q, r) for r in R]
                for p in a.demand:  # bf_ijp * bm * bp
                    FULL += [(ak, mid, q, p, r) for (r, _, _) in compat.get(p, [])]

    # inlays travel on the reverse arc of each demand arc (eq:inlay)
    inlay_pairs = []  # (fwd_arc, rev_arc, p, r, ric)
    for ak, a in arcs.items():
        if a.kind != "demand":
            continue
        rk = (ak[1], ak[0])
        for p in a.demand:
            for r, _, ric in compat.get(p, []):
                if ric > 0:
                    inlay_pairs.append((ak, rk, p, r, ric))
                    for mid in arcs[rk].modes:
                        for q in Qm[mid]:
                            INLAY.append((rk, mid, q, p, r))
    INLAY = sorted(set(INLAY))

    kappa = {(p, r): k for p, lst in compat.items() for (r, k, _) in lst}

    # ---- variables ------------------------------------------------------
    B, C = pulp.LpBinary, pulp.LpContinuous
    Y = {k: pulp.LpVariable(vname("Y", *k[0], k[1], k[2]), cat=B) for k in AMQ}
    XF = {k: pulp.LpVariable(vname("XF", *k[0], *k[1:]), cat=B) for k in FULL}
    NF = {
        k: pulp.LpVariable(vname("NF", *k[0], *k[1:]), lowBound=0, cat=C) for k in FULL
    }
    XE = {k: pulp.LpVariable(vname("XE", *k[0], *k[1:]), cat=B) for k in EMPTY}
    NE = {
        k: pulp.LpVariable(vname("NE", *k[0], *k[1:]), lowBound=0, cat=C) for k in EMPTY
    }
    XI = {k: pulp.LpVariable(vname("XI", *k[0], *k[1:]), cat=B) for k in INLAY}
    NI = {
        k: pulp.LpVariable(vname("NI", *k[0], *k[1:]), lowBound=0, cat=C) for k in INLAY
    }
    P = {
        r: pulp.LpVariable(vname("Purch", r), lowBound=0, cat=pulp.LpInteger) for r in R
    }
    XH = {h: pulp.LpVariable(vname("XH", h), cat=B) for h in data["hub_nodes"]}

    # ---- mode + frequency selection: at most one (m,q) per arc ----------
    # (eq:mode-amo + eq:only-one-q, with bm encoded in the index set)
    byarc_amq = defaultdict(list)
    for ak, mid, q in AMQ:
        byarc_amq[ak].append((mid, q))
    for ak, lst in byarc_amq.items():
        prob += (
            pulp.lpSum(Y[(ak, mid, q)] for (mid, q) in lst) <= 1,
            vname("oneMQ", *ak),
        )

    # ---- flow–binary consistency (eq:binary-*) ---------------------------
    for k in FULL:
        ak, mid, q, p, r = k
        d = arcs[ak].demand[p]
        prob += NF[k] <= d * XF[k]  # M = d (kappa >= 1)
        prob += XF[k] <= Y[(ak, mid, q)]
        # demand coverage when this (m,q,r) is chosen (eq:demand)
        prob += kappa[(p, r)] * NF[k] >= d * XF[k]
    for k in EMPTY:
        ak, mid, q, r = k
        prob += NE[k] <= M_flow * XE[k]
        prob += XE[k] <= Y[(ak, mid, q)]
    for k in INLAY:
        ak, mid, q, p, r = k
        prob += NI[k] <= M_flow * XI[k]
        prob += XI[k] <= Y[(ak, mid, q)]

    # ---- full-route activation (eq:full-activation): widehat{XF}_{ijp}=bf -
    byarc_full = defaultdict(list)
    for k in FULL:
        byarc_full[(k[0], k[3])].append(k)
    for (ak, p), keys in byarc_full.items():
        prob += pulp.lpSum(XF[k] for k in keys) == 1, vname("assign", *ak, p)

    # ---- inlay returns (eq:inlay) ----------------------------------------
    for fwd, rev, p, r, ric in inlay_pairs:
        f_keys = [k for k in byarc_full[(fwd, p)] if k[4] == r]
        i_keys = [k for k in INLAY if k[0] == rev and k[3] == p and k[4] == r]
        prob += (
            (pulp.lpSum(XI[k] for k in i_keys) == pulp.lpSum(XF[k] for k in f_keys)),
            vname("inlX", *fwd, p, r),
        )
        prob += (
            (
                pulp.lpSum(NI[k] for k in i_keys)
                >= ric * pulp.lpSum(NF[k] for k in f_keys)
            ),
            vname("inlN", *fwd, p, r),
        )

    # ---- daily volume and per-shipment volume per (a,m,q) ----------------
    full_by_amq, empty_by_amq, inlay_by_amq = (defaultdict(list) for _ in range(3))
    for k in FULL:
        full_by_amq[(k[0], k[1], k[2])].append(k)
    for k in EMPTY:
        empty_by_amq[(k[0], k[1], k[2])].append(k)
    for k in INLAY:
        inlay_by_amq[(k[0], k[1], k[2])].append(k)

    dailyVol = {}  # sum_r V_{ijmqr}  (eq:vol-*)
    for amq in AMQ:
        dailyVol[amq] = (
            pulp.lpSum(NF[k] * vf[k[4]] for k in full_by_amq[amq])
            + pulp.lpSum(NI[k] * vf[k[4]] for k in inlay_by_amq[amq])
            + pulp.lpSum(NE[k] * ve[k[3]] for k in empty_by_amq[amq])
        )

    # shipment volume bounds (eq:vol-bounds): VS = tau_q * dailyVol
    for ak, mid, q in AMQ:
        m = data["modes"][mid]
        y = Y[(ak, mid, q)]
        prob += (
            q * dailyVol[(ak, mid, q)] >= float(m.min_vol) * y,
            vname("vmin", *ak, mid, q),
        )
        prob += (
            q * dailyVol[(ak, mid, q)] <= float(m.max_vol) * y,
            vname("vmax", *ak, mid, q),
        )

    # ---- flow conservation per node and RTI type (eq:conservation) -------
    out_arcs, in_arcs = defaultdict(list), defaultdict(list)
    for ak in arcs:
        out_arcs[ak[0]].append(ak)
        in_arcs[ak[1]].append(ak)

    def node_flow(keys_dict, idx_r, ak_set, r):
        return pulp.lpSum(
            v for k, v in keys_dict.items() if k[0] in ak_set and k[idx_r] == r
        )

    for n in data["nodes"]:
        oset, iset = set(out_arcs[n]), set(in_arcs[n])
        for r in R:
            outflow = (
                node_flow(NF, 4, oset, r)
                + node_flow(NI, 4, oset, r)
                + node_flow(NE, 3, oset, r)
            )
            inflow = (
                node_flow(NF, 4, iset, r)
                + node_flow(NI, 4, iset, r)
                + node_flow(NE, 3, iset, r)
            )
            prob += outflow == inflow, vname("cons", n, r)

    # ---- non-hub empty exclusivity (eq:hub-excl) --------------------------
    nonhub = [n for n in data["nodes"] if n not in data["hub_nodes"]]
    zin = {
        (n, r): pulp.LpVariable(vname("ZEin", n, r), cat=B) for n in nonhub for r in R
    }
    zout = {
        (n, r): pulp.LpVariable(vname("ZEout", n, r), cat=B) for n in nonhub for r in R
    }
    for k in EMPTY:
        ak, _, _, r = k
        if (ak[0], r) in zout:
            prob += XE[k] <= zout[(ak[0], r)]
        if (ak[1], r) in zin:
            prob += XE[k] <= zin[(ak[1], r)]
    for n in nonhub:
        for r in R:
            prob += zin[(n, r)] + zout[(n, r)] <= 1, vname("hexcl", n, r)

    # ---- hub activation and handling costs (eq:hub-XH, eq:hub-CH-*) ------
    HC, HE = {}, {}
    for h in data["hub_nodes"]:
        hub = data["hubs"][int(h[1:])]
        for ak in out_arcs[h]:
            for mid in arcs[ak].modes:
                for q in Qm[mid]:
                    prob += XH[h] >= Y[(ak, mid, q)], vname("hubAct", *ak, mid, q)
        outVol = pulp.lpSum(
            dailyVol[(ak, mid, q)]
            for ak in out_arcs[h]
            for mid in arcs[ak].modes
            for q in Qm[mid]
        )
        fhc = float(getattr(hub, "fixed_economic_cost_per_volume", 0.0))
        fhe = float(getattr(hub, "fixed_co2_cost_per_volume", 0.0))
        vhc = float(getattr(hub, "variable_economic_cost_per_volume", 0.0))
        vhe = float(getattr(hub, "variable_co2_cost_per_volume", 0.0))
        HC[h] = fhc * XH[h] + vhc * outVol
        HE[h] = fhe * XH[h] + vhe * outVol

    # ---- RTI pool and purchases (eq:pool-*, eq:pc-*) ----------------------
    pool = {r: [] for r in R}
    for k in FULL:  # TNCF >= NF*(tau_q + l_ijm + ss_ijp)
        ak, mid, q, p, r = k
        lt = arcs[ak].modes[mid]["lt"]
        pool[r].append(NF[k] * (q + lt + arcs[ak].ss.get(p, 0.0)))
        # safety empties term of eq:pool-empty: + NF * ser_ijp
        pool[r].append(NF[k] * arcs[ak].ser.get(p, 0.0))
    for k in INLAY:  # TNCI >= NI*(tau_q + l_ijm)
        ak, mid, q, p, r = k
        pool[r].append(NI[k] * (q + arcs[ak].modes[mid]["lt"]))
    for k in EMPTY:  # TNCE >= NE*(tau_q + l_ijm)
        ak, mid, q, r = k
        pool[r].append(NE[k] * (q + arcs[ak].modes[mid]["lt"]))

    PCr, PEr = {}, {}
    for r in R:
        total = (1.0 + args.alpha) * pulp.lpSum(pool[r])  # eq:pool-total
        prob += total <= data["stock"][r] + P[r], vname("pool", r)
        PCr[r] = data["pc"][r] / data["life_days"][r] * P[r]  # eq:pc-money
        PEr[r] = data["pe"][r] / data["life_days"][r] * P[r]  # eq:pc-emissions

    # ---- daily transport cost / emissions (eq:cost-*) ----------------------
    # TC = TCS / tau_q = dailyVol*vtc + Y*ftc/tau_q   (exact, since VS=tau*V)
    TC = pulp.lpSum(
        dailyVol[(ak, mid, q)] * arcs[ak].modes[mid]["vtc"]
        + Y[(ak, mid, q)] * arcs[ak].modes[mid]["ftc"] / q
        for (ak, mid, q) in AMQ
    )
    TE = pulp.lpSum(
        dailyVol[(ak, mid, q)] * arcs[ak].modes[mid]["vte"]
        + Y[(ak, mid, q)] * arcs[ak].modes[mid]["fte"] / q
        for (ak, mid, q) in AMQ
    )

    ZC = TC + pulp.lpSum(PCr.values()) + pulp.lpSum(HC.values())  # eq:objective-cost
    ZE = (
        TE + pulp.lpSum(PEr.values()) + pulp.lpSum(HE.values())
    )  # eq:objective-emissions

    # ---- objective: weighted sum / epsilon-constraint ----------------------
    w = args.weight
    prob += w * ZC + (1.0 - w) * ZE
    if args.eps_co2 is not None:
        prob += ZE <= args.eps_co2, "eps_emissions"

    handles = dict(
        prob=prob,
        Y=Y,
        XF=XF,
        NF=NF,
        XE=XE,
        NE=NE,
        XI=XI,
        NI=NI,
        P=P,
        XH=XH,
        dailyVol=dailyVol,
        ZC=ZC,
        ZE=ZE,
        TC=TC,
        TE=TE,
        HC=HC,
        HE=HE,
        PCr=PCr,
        PEr=PEr,
        AMQ=AMQ,
    )
    return handles


# ════════════════════════════════════════════════════════════════════
#  SOLVE + REPORT
# ════════════════════════════════════════════════════════════════════


def get_solver(args):
    name = args.solver.upper()
    opts = dict(msg=args.verbose, timeLimit=args.time_limit)
    if name == "CBC":
        return pulp.PULP_CBC_CMD(gapRel=args.mip_gap, **opts)
    if name == "HIGHS":
        return pulp.HiGHS_CMD(**opts)
    if name == "GUROBI":
        return pulp.GUROBI_CMD(options=[("MIPGap", args.mip_gap)], **opts)
    if name == "CPLEX":
        return pulp.CPLEX_CMD(**opts)
    raise SystemExit(f"Unknown solver {args.solver}")


def val(x):
    v = pulp.value(x)
    return 0.0 if v is None else float(v)


def report(data, h, args):
    prob = h["prob"]
    status = pulp.LpStatus[prob.status]
    print(f"\nStatus: {status}")
    if prob.status != pulp.LpStatusOptimal:
        if prob.status == pulp.LpStatusInfeasible:
            print(
                "Model infeasible — relax --eps-co2, use a denser --q-grid, "
                "increase --hub-k, or check mode min-volume bounds vs. demand."
            )
        else:
            print(
                "No proven-optimal solution (time limit / solver issue). "
                "Re-run with --verbose, a larger --time-limit, or another --solver."
            )
        return None

    ZC, ZE = val(h["ZC"]), val(h["ZE"])
    print(f"\nObjectives (daily averages)")
    print(f"  ZC  total cost      : {ZC:12.2f} EUR/day")
    print(f"    transport         : {val(h['TC']):12.2f}")
    # print(f"    RTI amortisation  : {sum(val(v)
    #       for v in h['PCr'].values()):12.2f}")
    # print(f"    hub handling      : {sum(val(v)
    #       for v in h['HC'].values()):12.2f}")
    print(f"  ZE  total emissions : {ZE:12.2f} kg CO2/day")
    print(f"    transport         : {val(h['TE']):12.2f}")
    # print(f"    RTI amortisation  : {sum(val(v)
    #       for v in h['PEr'].values()):12.2f}")
    # print(f"    hub handling      : {sum(val(v)
    #       for v in h['HE'].values()):12.2f}")

    print("\nRTI purchases (P_r):")
    for r, var in h["P"].items():
        rt = data["rtis"][r]
        # print(f"  {obj_name(rt):<20s} buy {
        #       val(var):8.0f}   (stock {data['stock'][r]:.0f})")

    routes = []
    print("\nActive routes  (i -> j, mode, q [days], shipment vol [m3], daily cost):")
    for ak, mid, q in h["AMQ"]:
        if val(h["Y"][(ak, mid, q)]) > 0.5:
            dv = val(h["dailyVol"][(ak, mid, q)])
            arc = data["arcs"][ak]
            dcost = dv * arc.modes[mid]["vtc"] + arc.modes[mid]["ftc"] / q
            routes.append(
                dict(
                    src=ak[0],
                    dst=ak[1],
                    mode=mid,
                    q=q,
                    shipment_volume=round(q * dv, 3),
                    daily_volume=round(dv, 3),
                    daily_cost=round(dcost, 2),
                    kind=arc.kind,
                )
            )
            print(
                f"  {ak[0]:>5s} -> {ak[1]:<5s}  m{mid}  q={q:<2d}  "
                f"VS={q * dv:8.2f}  TC={dcost:8.2f}  [{arc.kind}]"
            )

    hubs_on = [hn for hn, v in h["XH"].items() if val(v) > 0.5]
    print(f"\nActivated hubs: {hubs_on or 'none'}")

    sol = dict(
        status=status,
        ZC=ZC,
        ZE=ZE,
        weight=args.weight,
        eps_co2=args.eps_co2,
        purchases={obj_name(data["rtis"][r]): val(v) for r, v in h["P"].items()},
        hubs_activated=hubs_on,
        routes=routes,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(sol, f, indent=2)
        print(f"\nSolution written to {args.out}")
    return sol


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Solve the linearised SND-RTI MILP.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--json", help="instance JSON produced by Network.save_json")
    src.add_argument(
        "--generate",
        choices=["small", "medium", "large"],
        help="generate an instance on the fly (needs instance_generator.py)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="objective = w*ZC + (1-w)*ZE  (1.0 = pure cost)",
    )
    ap.add_argument(
        "--eps-co2",
        type=float,
        default=None,
        help="epsilon-constraint cap on ZE [kg CO2/day]",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.03,
        help="maintenance/loss factor (eq:pool-total)",
    )
    ap.add_argument(
        "--q-grid",
        type=lambda s: {int(x) for x in s.split(",")},
        default=None,
        help="restrict Q, e.g. 1,2,3,5,7,10 (smaller model)",
    )
    ap.add_argument(
        "--hub-k",
        type=int,
        default=2,
        help="connect each plant to its k nearest hubs for empties (0 = off)",
    )
    ap.add_argument(
        "--v-avg",
        type=float,
        default=500.0,
        help="km/day used for synthetic (reverse/hub) arc lead times",
    )
    ap.add_argument("--solver", default="CBC", help="CBC | HIGHS | GUROBI | CPLEX")
    ap.add_argument("--mip-gap", type=float, default=0.01)
    ap.add_argument("--time-limit", type=int, default=600)
    ap.add_argument("--out", default="solution.json")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    net = load_network(args)
    t0 = time.time()
    data = build_data(net, args)
    h = build_model(data, args)
    nv = len(h["prob"].variables())
    nc = len(h["prob"].constraints)
    # print(f"Model built in {time.time()-t0:.1f}s — {nv} variables, {nc} constraints, "
    #       f"{len(data['arcs'])} arcs, |Q| per mode = "
    #       f"{{m: len(qs) for m, qs in data['Qm'].items()} }")
    h["prob"].solve(get_solver(args))
    report(data, h, args)


if __name__ == "__main__":
    main()
