"""
Instance loader for the SND-RTI model.

Reads the JSON produced by `instance_generator.py` and precomputes every
derived set, mapping and parameter the model needs:

    * V, H, P, R, M
    * Q_m       : valid queue periods for mode m
    * E_f       : full-RTI routes (fixed by procurement contracts)
    * E_all     : V x V minus self-loops (empty-RTI arcs can use any pair)
    * P_ij      : products on full-RTI route (i,j)
    * R_p       : RTI types compatible with product p
    * modes(ij) : modes available on (i,j) given its distance
    * dist[ij]  : euclidean distance
    * l_ijm     : transit time in days
    * c_ijm     : variable transport cost  (EUR/m3/shipment)
    * f_ijm     : fixed  transport cost    (EUR/shipment)
    * d_ijp     : daily RTI demand of product p on route (i,j)
    * ric_ijpr  : daily inlay fraction for product p / RTI type r
    * ss_ijp    : safety stock (RTI units) at both endpoints

The InstanceData object is immutable once built.

Note on units.  The instance generator already expresses demand in
*RTI units per day* (one RTI per day corresponds to the product quantity
that fits in one container).  Consequently we treat kappa_{pr} = 1 in
the model: the demand constraint simplifies to
    NF_{ijmpqr} >= d_{ijp} * XF_{ijmpqr}.
Safety stocks are also expressed directly in RTI units.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
#  Dataclass
# ---------------------------------------------------------------------------


@dataclass
class InstanceData:
    """All data needed to build the Pyomo model."""

    # --- identity ---
    name: str
    size_class: str
    seed: int

    # --- sets ---
    V: List[int]  # plant ids
    H: List[int]  # hub ids (subset of V)
    P: List[int]  # products
    R: List[int]  # RTI types
    M: List[str]  # modes, e.g. ['FTL','LTL']
    Q: List[int]  # union of queue periods
    Q_m: Dict[str, List[int]]  # valid queue periods per mode
    tau: Dict[int, int]  # tau_q (= q, days between dispatches)

    # --- graph topology ---
    E_f: List[Tuple[int, int]]  # full-RTI routes (arcs)
    E_all: List[Tuple[int, int]]  # V x V minus diagonal
    route_of: Dict[Tuple[int, int], dict]

    # --- edge attributes ---
    dist: Dict[Tuple[int, int], float]
    modes_on: Dict[Tuple[int, int], List[str]]
    l: Dict[Tuple[int, int, str], float]  # transit time in days
    c: Dict[Tuple[int, int, str], float]  # EUR per m3
    f: Dict[Tuple[int, int, str], float]  # EUR fixed per shipment
    delta: Dict[str, float]  # max m3 per mode
    omega: Dict[str, float]  # min m3 per mode (load factor)

    # --- product / route-product data ---
    P_ij: Dict[Tuple[int, int], List[int]]  # products on (i,j) in E_f
    R_p: Dict[int, List[int]]  # compatible RTI types per product
    d: Dict[Tuple[int, int, int], float]  # daily RTI demand d_{ijp}
    ric: Dict[Tuple[int, int, int], float]  # inlay fraction ric_{ijp}
    ss: Dict[Tuple[int, int, int], float]  # safety stock ss_{ijp} (RTI)
    ser: Dict[Tuple[int, int, int], float]  # safety empties ratio

    # --- RTI parameters ---
    v_full: Dict[int, float]  # v^f_r
    v_empty: Dict[int, float]  # v^e_r
    p_rti: Dict[int, float]  # purchase cost per RTI
    s_rti: Dict[int, float]  # existing pool
    L_r: Dict[int, float]  # useful life, days
    alpha: Dict[int, float]  # maintenance loss factor (combined)

    # --- hub parameters ---
    fhc: Dict[int, float]
    vhc: Dict[int, float]

    # --- misc ---
    ss_days: int
    plant_coords: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"Instance {self.name}",
            f"  |V|={len(self.V)}  |H|={len(self.H)}  |P|={len(self.P)}  "
            f"|R|={len(self.R)}  |M|={len(self.M)}  |Q|={len(self.Q)}",
            f"  |E_f|={len(self.E_f)}  |E_all|={len(self.E_all)}",
            f"  Q[FTL]={self.Q_m.get('FTL', [])}  Q[LTL]={self.Q_m.get('LTL', [])}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Loader
# ---------------------------------------------------------------------------


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_instance(path: str | Path) -> InstanceData:
    """Parse the JSON produced by ``instance_generator.py``."""
    with open(path, "r") as fp:
        raw = json.load(fp)

    params = raw["params"]

    # -------- basic sets --------
    V = [p["id"] for p in raw["plants"]]
    H = list(raw["hub_ids"])
    n_products = int(raw["n_products"])
    n_rti = int(raw["n_rti_types"])
    P = list(range(n_products))
    R = list(range(n_rti))
    M = list(raw["modes"])

    # -------- queue periods --------
    q_min_ftl, q_max_ftl = raw["q_bounds"]["FTL"]
    q_min_ltl, q_max_ltl = raw["q_bounds"]["LTL"]
    Q_m = {
        "FTL": list(range(int(q_min_ftl), int(q_max_ftl) + 1)),
        "LTL": list(range(int(q_min_ltl), int(q_max_ltl) + 1)),
    }
    Q = sorted(set(Q_m["FTL"]) | set(Q_m["LTL"]))
    tau = {q: q for q in Q}  # one day per unit of q

    # -------- coordinates & distances --------
    plant_coords = {p["id"]: (float(p["x"]), float(p["y"])) for p in raw["plants"]}

    # -------- full routes & derived E_all --------
    route_of: Dict[Tuple[int, int], dict] = {}
    E_f: List[Tuple[int, int]] = []
    P_ij: Dict[Tuple[int, int], List[int]] = {}
    d: Dict[Tuple[int, int, int], float] = {}
    ric: Dict[Tuple[int, int, int], float] = {}
    ss: Dict[Tuple[int, int, int], float] = {}
    ser: Dict[Tuple[int, int, int], float] = {}

    ss_days = int(raw["ss_days"])
    ser_default = 1.0  # one empty returned per full consumed

    for r in raw["routes"]:
        i, j = int(r["origin"]), int(r["destination"])
        E_f.append((i, j))
        route_of[(i, j)] = r
        prods = [int(p) for p in r["products"]]
        P_ij[(i, j)] = prods
        for p in prods:
            demand = float(r["demand"][str(p)])
            d[(i, j, p)] = demand
            ric[(i, j, p)] = float(r["inlay_frac"].get(str(p), 0.0))
            ss[(i, j, p)] = ss_days * demand
            ser[(i, j, p)] = ser_default

    E_all = [(i, j) for i in V for j in V if i != j]

    # -------- per-edge attributes --------
    d_ftl_min = float(params["d_ftl_min"])
    d_ltl_max = float(params["d_ltl_max"])
    v_avg = float(params["v_avg"])

    dist: Dict[Tuple[int, int], float] = {}
    modes_on: Dict[Tuple[int, int], List[str]] = {}
    l_: Dict[Tuple[int, int, str], float] = {}
    c: Dict[Tuple[int, int, str], float] = {}
    f: Dict[Tuple[int, int, str], float] = {}

    c_ftl = float(params["c_ftl"])
    c_ltl = float(params["c_ltl"])
    f_ftl = float(params["f_ftl"])
    f_ltl = float(params["f_ltl"])

    for i, j in E_all:
        dij = _euclid(plant_coords[i], plant_coords[j])
        dist[(i, j)] = dij

        allowed: List[str] = []
        if dij < d_ftl_min:
            allowed = ["LTL"]
        elif dij > d_ltl_max:
            allowed = ["FTL"]
        else:
            allowed = ["FTL", "LTL"]
        modes_on[(i, j)] = allowed

        # transit time: at least one day even for very short routes
        for m in allowed:
            l_[(i, j, m)] = max(dij / v_avg, 1e-3)
            c[(i, j, m)] = (c_ftl if m == "FTL" else c_ltl) * dij
            f[(i, j, m)] = f_ftl if m == "FTL" else f_ltl

    delta = {"FTL": float(params["delta_ftl"]), "LTL": float(params["delta_ltl"])}
    omega = {"FTL": float(params["omega_ftl"]), "LTL": float(params["omega_ltl"])}

    # -------- products / RTI compatibility --------
    R_p: Dict[int, List[int]] = {
        int(k): [int(r_) for r_ in v] for k, v in raw["product_rti_compat"].items()
    }

    # -------- RTI parameters --------
    v_full = {r_: float(raw["v_full"][r_]) for r_ in R}
    v_empty = {r_: float(raw["v_empty"][r_]) for r_ in R}
    p_rti = {r_: float(raw["p_rti"][r_]) for r_ in R}
    s_rti = {r_: float(raw["s_rti"][r_]) for r_ in R}
    alpha_f = {r_: float(raw["alpha_f"][r_]) for r_ in R}
    alpha_e = {r_: float(raw["alpha_e"][r_]) for r_ in R}
    alpha = {r_: alpha_f[r_] + alpha_e[r_] for r_ in R}  # combined loss factor
    L_default = 3 * 365.0  # 3 years RTI life
    L_r = {r_: L_default for r_ in R}

    # -------- hubs --------
    fhc = {int(k): float(v) for k, v in raw["fhc"].items()}
    vhc = {int(k): float(v) for k, v in raw["vhc"].items()}

    return InstanceData(
        name=raw["name"],
        size_class=raw["size_class"],
        seed=int(raw["seed"]),
        V=V,
        H=H,
        P=P,
        R=R,
        M=M,
        Q=Q,
        Q_m=Q_m,
        tau=tau,
        E_f=E_f,
        E_all=E_all,
        route_of=route_of,
        dist=dist,
        modes_on=modes_on,
        l=l_,
        c=c,
        f=f,
        delta=delta,
        omega=omega,
        P_ij=P_ij,
        R_p=R_p,
        d=d,
        ric=ric,
        ss=ss,
        ser=ser,
        v_full=v_full,
        v_empty=v_empty,
        p_rti=p_rti,
        s_rti=s_rti,
        L_r=L_r,
        alpha=alpha,
        fhc=fhc,
        vhc=vhc,
        ss_days=ss_days,
        plant_coords=plant_coords,
    )
