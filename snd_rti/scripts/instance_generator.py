"""
SND-RTI Instance Generator
===========================
Generates random instances for the Service Network Design for
Returnable Transport Items model. Produces a JSON instance file
and a network plot.

Usage:
    python instance_generator.py --size small --seed 42 --plot
    python instance_generator.py --size medium --seed 1 --plot --out instances/
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

try:
    import folium
    from folium import plugins

    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ═══════════════════════════════════════════════════════════════════
#  INSTANCE CLASS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

INSTANCE_CONFIGS = {
    "small": {
        "n_zones": (2, 3),
        "n_large_per_zone": (1, 2),
        "n_small_per_zone": (2, 4),
        "n_hubs_per_zone": 1,
        "L": 500,
        "n_rti_types": 2,
        "n_products": (3, 5),
        "max_rti_compat": 2,
        "max_products_per_route": 2,
    },
    "medium": {
        "n_zones": (4, 6),
        "n_large_per_zone": (2, 3),
        "n_small_per_zone": (4, 8),
        "n_hubs_per_zone": 2,
        "L": 1000,
        "n_rti_types": (3, 4),
        "n_products": (6, 10),
        "max_rti_compat": 3,
        "max_products_per_route": 3,
    },
    "large": {
        "n_zones": (8, 12),
        "n_large_per_zone": (3, 5),
        "n_small_per_zone": (6, 12),
        "n_hubs_per_zone": 3,
        "L": 1500,
        "n_rti_types": (4, 6),
        "n_products": (10, 20),
        "max_rti_compat": 4,
        "max_products_per_route": 4,
    },
}

PARAMS = {
    "sigma_intra": (5, 15),
    "k_avg_suppliers": (2, 4),
    "p_inter": 0.7,
    "p_intra": 0.7,
    "zone_importance_range": (0.5, 2.5),
    "zone_imbalance_range": (0.2, 0.8),
    "mu_d": 2.0,
    "sigma_d": 0.8,
    "r_max": 0.3,
    "p_inlay": 0.05,
    "c_ftl": 0.05,
    "c_ltl": 0.12,
    "f_ftl": 150.0,
    "f_ltl": 40.0,
    "delta_ftl": 100.0,     # realistic EU trailer ~100 m3
    "delta_ltl": 30.0,
    "omega_ftl": 10.0,      # loose; mode choice driven by fixed-cost gap
    "omega_ltl": 0.5,       # groupage has effectively no min
    "v_avg": 500.0,
    "d_ftl_min": 50.0,
    "d_ltl_max": 800.0,
    "q_min_ftl": 3,
    "q_max_ftl": 10,
    "q_min_ltl": 1,
    "q_max_ltl": 5,
    "v_full_range": (0.3, 0.8),
    "fold_ratio_range": (0.2, 0.5),
    "p_rti_range": (30, 120),
    "s_rti": 0,
    "ss_days": 3,
    "alpha_f_range": (0.02, 0.05),
    "alpha_e_range": (0.02, 0.05),
    "fhc_range": (200, 500),
    "vhc_range": (0.5, 2.0),
}


@dataclass
class Plant:
    id: int
    x: float
    y: float
    zone_id: int
    is_large: bool
    is_hub: bool = False


@dataclass
class Route:
    origin: int
    destination: int
    products: List[int]
    demand: Dict[int, float]
    inlay_frac: Dict[int, float]
    distance: float


@dataclass
class Instance:
    name: str
    size_class: str
    seed: int
    L: float
    plants: List[dict]
    zones: List[dict]
    routes: List[dict]
    hub_ids: List[int]
    n_rti_types: int
    n_products: int
    n_modes: int
    product_rti_compat: Dict[int, List[int]]
    v_full: List[float]
    v_empty: List[float]
    p_rti: List[float]
    s_rti: List[int]
    alpha_f: List[float]
    alpha_e: List[float]
    modes: List[str]
    q_bounds: Dict[str, Tuple[int, int]]
    fhc: Dict[int, float]
    vhc: Dict[int, float]
    ss_days: int
    params: dict


def _sample_range(rng, r, as_int=True):
    if isinstance(r, tuple):
        if as_int:
            return rng.integers(r[0], r[1] + 1)
        else:
            return rng.uniform(r[0], r[1])
    return r


def euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def generate_instance(size: str, seed: int):
    rng = np.random.default_rng(seed)
    cfg = INSTANCE_CONFIGS[size]
    L = cfg["L"]

    n_zones = _sample_range(rng, cfg["n_zones"])
    zone_centres = [(rng.uniform(0, L), rng.uniform(0, L)) for _ in range(n_zones)]

    plants: List[Plant] = []
    hub_ids = []
    zone_info = []
    pid = 0
    for z in range(n_zones):
        cx, cy = zone_centres[z]
        sigma = _sample_range(rng, PARAMS["sigma_intra"], as_int=False)
        n_large = _sample_range(rng, cfg["n_large_per_zone"])
        n_small = _sample_range(rng, cfg["n_small_per_zone"])
        n_hubs = cfg["n_hubs_per_zone"]
        importance = round(rng.uniform(*PARAMS["zone_importance_range"]), 2)
        imbalance = rng.uniform(*PARAMS["zone_imbalance_range"])
        weight_in = round(importance * imbalance * 2, 2)
        weight_out = round(importance * (1 - imbalance) * 2, 2)
        zone_plants = []

        for _ in range(n_large):
            px = np.clip(cx + rng.normal(0, sigma), 0, L)
            py = np.clip(cy + rng.normal(0, sigma), 0, L)
            plants.append(Plant(id=pid, x=px, y=py, zone_id=z, is_large=True))
            zone_plants.append(pid); pid += 1
        for _ in range(n_small):
            px = np.clip(cx + rng.normal(0, sigma), 0, L)
            py = np.clip(cy + rng.normal(0, sigma), 0, L)
            plants.append(Plant(id=pid, x=px, y=py, zone_id=z, is_large=False))
            zone_plants.append(pid); pid += 1
        for _ in range(n_hubs):
            px = np.clip(cx + rng.normal(0, sigma), 0, L)
            py = np.clip(cy + rng.normal(0, sigma), 0, L)
            plants.append(Plant(id=pid, x=px, y=py, zone_id=z, is_large=False, is_hub=True))
            zone_plants.append(pid); hub_ids.append(pid); pid += 1
        zone_info.append({"zone_id": z, "cx": cx, "cy": cy,
                          "importance": importance, "weight_in": weight_in,
                          "weight_out": weight_out, "plant_ids": zone_plants,
                          "n_large": n_large, "n_small": n_small, "n_hubs": n_hubs})

    n_plants = len(plants)
    k_avg = _sample_range(rng, PARAMS["k_avg_suppliers"])
    routes: List[Route] = []
    n_products = _sample_range(rng, cfg["n_products"])

    for j_plant in plants:
        if j_plant.is_hub: continue
        w_in = zone_info[j_plant.zone_id]["weight_in"]
        n_suppliers = max(1, rng.poisson(k_avg * w_in))
        n_suppliers = min(n_suppliers, n_plants - len(hub_ids) - 1)
        same_zone = [p for p in plants
                     if p.zone_id == j_plant.zone_id and p.id != j_plant.id and not p.is_hub]
        other_zone = [p for p in plants if p.zone_id != j_plant.zone_id and not p.is_hub]
        p_inter = PARAMS["p_inter"] if j_plant.is_large else 1.0 - PARAMS["p_intra"]
        if len(other_zone) > 0:
            ow = np.array([zone_info[p.zone_id]["weight_out"] for p in other_zone])
            op = ow / ow.sum() if ow.sum() > 0 else np.ones(len(other_zone)) / len(other_zone)
        else:
            op = None
        if len(same_zone) > 0:
            sw = np.array([zone_info[p.zone_id]["weight_out"] for p in same_zone])
            sp = sw / sw.sum() if sw.sum() > 0 else np.ones(len(same_zone)) / len(same_zone)
        else:
            sp = None
        suppliers = []
        for _ in range(n_suppliers):
            s = None
            if rng.random() < p_inter and len(other_zone) > 0:
                s = other_zone[rng.choice(len(other_zone), p=op)]
            elif len(same_zone) > 0:
                s = same_zone[rng.choice(len(same_zone), p=sp)]
            elif len(other_zone) > 0:
                s = other_zone[rng.choice(len(other_zone), p=op)]
            if s and s.id not in [sup.id for sup in suppliers] and s.id != j_plant.id:
                suppliers.append(s)

        for s in suppliers:
            dist = euclidean(s, j_plant)
            if j_plant.is_large:
                n_prods = rng.integers(1, cfg["max_products_per_route"] + 1)
            else:
                n_prods = 1
            prods = rng.choice(n_products, size=min(n_prods, n_products), replace=False).tolist()
            route_has_inlay = rng.random() < PARAMS["p_inlay"]
            demand, inlay_frac = {}, {}
            for p in prods:
                if j_plant.is_large:
                    d = np.exp(rng.normal(PARAMS["mu_d"] + 0.5 * PARAMS["sigma_d"],
                                          PARAMS["sigma_d"] * 0.6))
                else:
                    d = np.exp(rng.normal(PARAMS["mu_d"], PARAMS["sigma_d"]))
                demand[p] = max(1, int(round(d)))
                inlay_frac[p] = round(rng.uniform(0, PARAMS["r_max"]), 3) if route_has_inlay else 0.0

            # --- Ensure per-route feasibility at tightest (mode, q) combo ---
            # A route with total demand D must satisfy
            #   D * q_min_m * max(v_full) <= delta_m   for the chosen mode m.
            # Use the most permissive capacity (FTL, q_min_FTL) as the cap.
            v_max_tmp = 0.8  # upper bound used by the generator (see v_full_range)
            cap = PARAMS["delta_ftl"] / (PARAMS["q_min_ftl"] * v_max_tmp)
            total_d = sum(demand.values())
            if total_d > cap:
                scale = cap / total_d
                demand = {p: max(1, int(v * scale)) for p, v in demand.items()}
            routes.append(Route(origin=s.id, destination=j_plant.id,
                                products=prods, demand=demand, inlay_frac=inlay_frac,
                                distance=round(dist, 2)))

    seen = set(); unique_routes = []
    for r in routes:
        key = (r.origin, r.destination)
        if key not in seen:
            seen.add(key); unique_routes.append(r)
    routes = unique_routes

    n_rti = _sample_range(rng, cfg["n_rti_types"])
    product_rti_compat = {}
    frac_multi = 0.2
    for p in range(n_products):
        if rng.random() < frac_multi and n_rti > 1:
            n_compat = rng.integers(2, min(cfg["max_rti_compat"], n_rti) + 1)
            compat = sorted(rng.choice(n_rti, size=n_compat, replace=False).tolist())
        else:
            compat = [int(rng.integers(0, n_rti))]
        product_rti_compat[p] = compat

    v_full = [round(rng.uniform(*PARAMS["v_full_range"]), 3) for _ in range(n_rti)]
    fold_ratios = [round(rng.uniform(*PARAMS["fold_ratio_range"]), 3) for _ in range(n_rti)]
    v_empty = [round(v_full[r] * fold_ratios[r], 4) for r in range(n_rti)]
    p_rti = [round(rng.uniform(*PARAMS["p_rti_range"]), 2) for _ in range(n_rti)]
    s_rti = [0] * n_rti
    alpha_f = [round(rng.uniform(*PARAMS["alpha_f_range"]), 4) for _ in range(n_rti)]
    alpha_e = [round(rng.uniform(*PARAMS["alpha_e_range"]), 4) for _ in range(n_rti)]
    fhc = {h: round(rng.uniform(*PARAMS["fhc_range"]), 2) for h in hub_ids}
    vhc = {h: round(rng.uniform(*PARAMS["vhc_range"]), 3) for h in hub_ids}

    instance = Instance(
        name=f"SND_RTI_{size}_{seed}", size_class=size, seed=seed, L=L,
        plants=[{"id": p.id, "x": round(p.x, 2), "y": round(p.y, 2),
                 "zone_id": p.zone_id, "is_large": p.is_large, "is_hub": p.is_hub}
                for p in plants],
        zones=zone_info,
        routes=[{"origin": r.origin, "destination": r.destination,
                 "products": r.products,
                 "demand": {str(k): v for k, v in r.demand.items()},
                 "inlay_frac": {str(k): v for k, v in r.inlay_frac.items()},
                 "distance": r.distance} for r in routes],
        hub_ids=hub_ids, n_rti_types=n_rti, n_products=n_products, n_modes=2,
        product_rti_compat={str(k): v for k, v in product_rti_compat.items()},
        v_full=v_full, v_empty=v_empty, p_rti=p_rti, s_rti=s_rti,
        alpha_f=alpha_f, alpha_e=alpha_e,
        modes=["FTL", "LTL"],
        q_bounds={"FTL": (PARAMS["q_min_ftl"], PARAMS["q_max_ftl"]),
                  "LTL": (PARAMS["q_min_ltl"], PARAMS["q_max_ltl"])},
        fhc={str(k): v for k, v in fhc.items()},
        vhc={str(k): v for k, v in vhc.items()},
        ss_days=PARAMS["ss_days"], params=PARAMS,
    )
    return instance, plants, routes, zone_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=".")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    inst, plants, routes, zinfo = generate_instance(args.size, args.seed)
    json_path = os.path.join(args.out, f"{inst.name}.json")
    with open(json_path, "w") as f:
        json.dump(asdict(inst), f, indent=2, default=str)
    print(f"Instance saved to {json_path}")
    print(f"  |V|={len(plants)}  |E|={len(routes)}  |R|={inst.n_rti_types}  |P|={inst.n_products}")


if __name__ == "__main__":
    main()
