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
        "max_rti_compat": 2,  # max |R_p| for multi-compat products
        "max_products_per_route": 2,  # for large-plant routes
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

# ═══════════════════════════════════════════════════════════════════
#  FIXED PARAMETERS
# ═══════════════════════════════════════════════════════════════════

PARAMS = {
    # Network generation
    "sigma_intra": (5, 15),  # km, Gaussian spread within zone
    "k_avg_suppliers": (2, 4),  # avg supply routes per plant
    "p_inter": 0.7,  # large plants: prob of inter-zone
    "p_intra": 0.7,  # small plants: prob of intra-zone
    # Demand
    "mu_d": 2.0,  # log-normal mean (log scale)
    "sigma_d": 0.8,  # log-normal std  (log scale)
    "r_max": 0.3,  # max inlay fraction
    # Transport
    "c_ftl": 0.05,  # €/km/m³
    "c_ltl": 0.12,  # €/km/m³
    "f_ftl": 150.0,  # €/shipment
    "f_ltl": 40.0,  # €/shipment
    "delta_ftl": 80.0,  # m³ capacity
    "delta_ltl": 30.0,  # m³ capacity
    "omega_ftl": 40.0,  # m³ min load
    "omega_ltl": 5.0,  # m³ min load
    "v_avg": 500.0,  # km/day average speed
    "d_ftl_min": 50.0,  # km, FTL minimum distance
    "d_ltl_max": 800.0,  # km, LTL maximum distance
    "q_min_ftl": 3,
    "q_max_ftl": 10,  # days
    "q_min_ltl": 1,
    "q_max_ltl": 5,  # days
    # RTI & inventory
    "v_full_range": (0.3, 0.8),  # m³ per full RTI
    "fold_ratio_range": (0.2, 0.5),  # v_e / v_f
    "p_rti_range": (30, 120),  # € per RTI
    "s_rti": 0,  # existing pool
    "ss_days": 3,  # safety stock days
    "alpha_f_range": (0.02, 0.05),
    "alpha_e_range": (0.02, 0.05),
    # Hubs
    "fhc_range": (200, 500),  # €/day fixed hub cost
    "vhc_range": (0.5, 2.0),  # €/m³ variable hub cost
}


# ═══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════


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
    products: List[int]  # product indices assigned
    demand: Dict[int, float]  # product -> daily RTI demand
    inlay_frac: Dict[int, float]  # product -> ric
    distance: float


@dataclass
class Instance:
    name: str
    size_class: str
    seed: int
    L: float

    # Sets
    plants: List[dict]
    zones: List[dict]
    routes: List[dict]
    hub_ids: List[int]
    n_rti_types: int
    n_products: int
    n_modes: int

    # Product-RTI compatibility: product_id -> list of RTI types
    product_rti_compat: Dict[int, List[int]]

    # RTI parameters (indexed by r)
    v_full: List[float]
    v_empty: List[float]
    p_rti: List[float]
    s_rti: List[int]
    alpha_f: List[float]
    alpha_e: List[float]

    # Transport parameters
    modes: List[str]
    q_bounds: Dict[str, Tuple[int, int]]

    # Hub parameters (indexed by hub plant id)
    fhc: Dict[int, float]
    vhc: Dict[int, float]

    # Global
    ss_days: int
    params: dict


# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def _sample_range(rng: np.random.Generator, r, as_int=True):
    """Sample from a (lo, hi) tuple or return scalar."""
    if isinstance(r, tuple):
        if as_int:
            return rng.integers(r[0], r[1] + 1)
        else:
            return rng.uniform(r[0], r[1])
    return r


def euclidean(p1: Plant, p2: Plant) -> float:
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# ═══════════════════════════════════════════════════════════════════
#  INSTANCE GENERATION
# ═══════════════════════════════════════════════════════════════════


def generate_instance(size: str, seed: int) -> Instance:
    rng = np.random.default_rng(seed)
    cfg = INSTANCE_CONFIGS[size]
    L = cfg["L"]

    # ── 1. Industrial zones ──────────────────────────────────────
    n_zones = _sample_range(rng, cfg["n_zones"])
    zone_centres = []
    for z in range(n_zones):
        zone_centres.append((rng.uniform(0, L), rng.uniform(0, L)))

    # ── 2. Place plants ──────────────────────────────────────────
    plants: List[Plant] = []
    zone_info = []
    pid = 0
    for z in range(n_zones):
        cx, cy = zone_centres[z]
        sigma = _sample_range(rng, PARAMS["sigma_intra"], as_int=False)
        n_large = _sample_range(rng, cfg["n_large_per_zone"])
        n_small = _sample_range(rng, cfg["n_small_per_zone"])

        zone_plants = []
        for _ in range(n_large):
            px = cx + rng.normal(0, sigma)
            py = cy + rng.normal(0, sigma)
            px = np.clip(px, 0, L)
            py = np.clip(py, 0, L)
            plants.append(Plant(id=pid, x=px, y=py, zone_id=z, is_large=True))
            zone_plants.append(pid)
            pid += 1
        for _ in range(n_small):
            px = cx + rng.normal(0, sigma)
            py = cy + rng.normal(0, sigma)
            px = np.clip(px, 0, L)
            py = np.clip(py, 0, L)
            plants.append(Plant(id=pid, x=px, y=py, zone_id=z, is_large=False))
            zone_plants.append(pid)
            pid += 1

        zone_info.append(
            {
                "zone_id": z,
                "cx": cx,
                "cy": cy,
                "plant_ids": zone_plants,
                "n_large": n_large,
                "n_small": n_small,
            }
        )

    n_plants = len(plants)

    # ── 3. Generate full-RTI routes ──────────────────────────────
    k_avg = _sample_range(rng, PARAMS["k_avg_suppliers"])
    routes: List[Route] = []
    n_products = _sample_range(rng, cfg["n_products"])

    for j_plant in plants:
        # How many suppliers feed into this plant
        n_suppliers = max(1, rng.poisson(k_avg))
        n_suppliers = min(n_suppliers, n_plants - 1)

        # Build candidate supplier list
        same_zone = [
            p for p in plants if p.zone_id == j_plant.zone_id and p.id != j_plant.id
        ]
        other_zone = [p for p in plants if p.zone_id != j_plant.zone_id]

        if j_plant.is_large:
            p_inter = PARAMS["p_inter"]
        else:
            p_inter = 1.0 - PARAMS["p_intra"]

        suppliers = []
        for _ in range(n_suppliers):
            if rng.random() < p_inter and len(other_zone) > 0:
                s = rng.choice(other_zone)
            elif len(same_zone) > 0:
                s = rng.choice(same_zone)
            elif len(other_zone) > 0:
                s = rng.choice(other_zone)
            else:
                continue
            if s.id not in [sup.id for sup in suppliers] and s.id != j_plant.id:
                suppliers.append(s)

        for s in suppliers:
            dist = euclidean(s, j_plant)

            # Products on this route
            if j_plant.is_large:
                n_prods = rng.integers(1, cfg["max_products_per_route"] + 1)
            else:
                n_prods = 1

            prods = rng.choice(
                n_products, size=min(n_prods, n_products), replace=False
            ).tolist()

            demand = {}
            inlay_frac = {}
            for p in prods:
                if j_plant.is_large:
                    # Upper quartile for large plants
                    d = np.exp(
                        rng.normal(
                            PARAMS["mu_d"] + 0.5 * PARAMS["sigma_d"],
                            PARAMS["sigma_d"] * 0.6,
                        )
                    )
                else:
                    d = np.exp(rng.normal(PARAMS["mu_d"], PARAMS["sigma_d"]))
                demand[p] = max(1, int(round(d)))
                inlay_frac[p] = round(rng.uniform(0, PARAMS["r_max"]), 3)

            routes.append(
                Route(
                    origin=s.id,
                    destination=j_plant.id,
                    products=prods,
                    demand=demand,
                    inlay_frac=inlay_frac,
                    distance=round(dist, 2),
                )
            )

    # Remove duplicate routes (same origin-destination)
    seen = set()
    unique_routes = []
    for r in routes:
        key = (r.origin, r.destination)
        if key not in seen:
            seen.add(key)
            unique_routes.append(r)
    routes = unique_routes

    # ── 4. Hub designation ───────────────────────────────────────
    # Compute throughput per plant
    throughput = {p.id: 0.0 for p in plants}
    for r in routes:
        total_demand = sum(r.demand.values())
        throughput[r.origin] += total_demand
        throughput[r.destination] += total_demand

    hub_ids = []
    n_hubs_per_zone = cfg["n_hubs_per_zone"]
    for z_info in zone_info:
        z_plants = z_info["plant_ids"]
        # Sort by throughput descending
        z_plants_sorted = sorted(
            z_plants, key=lambda pid: throughput[pid], reverse=True
        )
        n_h = min(n_hubs_per_zone, len(z_plants_sorted))
        for i in range(n_h):
            plants[z_plants_sorted[i]].is_hub = True
            hub_ids.append(z_plants_sorted[i])

    # ── 5. RTI types and product compatibility ───────────────────
    n_rti = _sample_range(rng, cfg["n_rti_types"])
    product_rti_compat = {}
    frac_multi = 0.2  # 20% of products have multiple compatible types

    for p in range(n_products):
        if rng.random() < frac_multi and n_rti > 1:
            n_compat = rng.integers(2, min(cfg["max_rti_compat"], n_rti) + 1)
            compat = sorted(rng.choice(n_rti, size=n_compat, replace=False).tolist())
        else:
            compat = [int(rng.integers(0, n_rti))]
        product_rti_compat[p] = compat

    # ── 6. RTI parameters ────────────────────────────────────────
    v_full = [round(rng.uniform(*PARAMS["v_full_range"]), 3) for _ in range(n_rti)]
    fold_ratios = [
        round(rng.uniform(*PARAMS["fold_ratio_range"]), 3) for _ in range(n_rti)
    ]
    v_empty = [round(v_full[r] * fold_ratios[r], 4) for r in range(n_rti)]
    p_rti = [round(rng.uniform(*PARAMS["p_rti_range"]), 2) for _ in range(n_rti)]
    s_rti = [0] * n_rti
    alpha_f = [round(rng.uniform(*PARAMS["alpha_f_range"]), 4) for _ in range(n_rti)]
    alpha_e = [round(rng.uniform(*PARAMS["alpha_e_range"]), 4) for _ in range(n_rti)]

    # ── 7. Hub parameters ────────────────────────────────────────
    fhc = {}
    vhc = {}
    for h in hub_ids:
        fhc[h] = round(rng.uniform(*PARAMS["fhc_range"]), 2)
        vhc[h] = round(rng.uniform(*PARAMS["vhc_range"]), 3)

    # ── 8. Build instance ────────────────────────────────────────
    instance = Instance(
        name=f"SND_RTI_{size}_{seed}",
        size_class=size,
        seed=seed,
        L=L,
        plants=[
            {
                "id": p.id,
                "x": round(p.x, 2),
                "y": round(p.y, 2),
                "zone_id": p.zone_id,
                "is_large": p.is_large,
                "is_hub": p.is_hub,
            }
            for p in plants
        ],
        zones=zone_info,
        routes=[
            {
                "origin": r.origin,
                "destination": r.destination,
                "products": r.products,
                "demand": {str(k): v for k, v in r.demand.items()},
                "inlay_frac": {str(k): v for k, v in r.inlay_frac.items()},
                "distance": r.distance,
            }
            for r in routes
        ],
        hub_ids=hub_ids,
        n_rti_types=n_rti,
        n_products=n_products,
        n_modes=2,
        product_rti_compat={str(k): v for k, v in product_rti_compat.items()},
        v_full=v_full,
        v_empty=v_empty,
        p_rti=p_rti,
        s_rti=s_rti,
        alpha_f=alpha_f,
        alpha_e=alpha_e,
        modes=["FTL", "LTL"],
        q_bounds={
            "FTL": (PARAMS["q_min_ftl"], PARAMS["q_max_ftl"]),
            "LTL": (PARAMS["q_min_ltl"], PARAMS["q_max_ltl"]),
        },
        fhc={str(k): v for k, v in fhc.items()},
        vhc={str(k): v for k, v in vhc.items()},
        ss_days=PARAMS["ss_days"],
        params=PARAMS,
    )

    return instance, plants, routes, zone_info


# ═══════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════


def plot_instance(plants, routes, zone_info, instance, save_path=None):
    """Plot the supply chain network with industrial zones and routes."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    L = instance.L

    # ── Zone backgrounds ─────────────────────────────────────────
    zone_colors = plt.cm.Set3(np.linspace(0, 1, len(zone_info)))
    for z_info, zc in zip(zone_info, zone_colors):
        z_plants = [plants[pid] for pid in z_info["plant_ids"]]
        xs = [p.x for p in z_plants]
        ys = [p.y for p in z_plants]
        cx, cy = z_info["cx"], z_info["cy"]

        # Draw zone as an ellipse around the plants
        if len(xs) > 1:
            rx = max(max(xs) - min(xs), 30) * 0.7 + 20
            ry = max(max(ys) - min(ys), 30) * 0.7 + 20
        else:
            rx, ry = 30, 30

        ellipse = mpatches.Ellipse(
            (cx, cy),
            width=rx * 2,
            height=ry * 2,
            facecolor=zc,
            edgecolor="gray",
            alpha=0.18,
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(ellipse)
        ax.text(
            cx,
            cy + ry + 8,
            f"Zone {z_info['zone_id']}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="gray",
        )

    # ── Routes ───────────────────────────────────────────────────
    # Color routes by number of products
    max_prods = max(len(r.products) for r in routes) if routes else 1
    route_cmap = plt.cm.YlOrRd

    for r in routes:
        p_orig = plants[r.origin]
        p_dest = plants[r.destination]
        n_prods = len(r.products)
        total_demand = sum(r.demand.values())

        # Color intensity by number of products
        color = route_cmap(0.3 + 0.7 * (n_prods / max(max_prods, 1)))
        lw = 0.6 + 1.4 * (total_demand / 30)  # scale by demand
        lw = min(lw, 3.5)

        ax.annotate(
            "",
            xy=(p_dest.x, p_dest.y),
            xytext=(p_orig.x, p_orig.y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=0.55,
                connectionstyle="arc3,rad=0.08",
            ),
        )

    # ── Plant nodes ──────────────────────────────────────────────
    for p in plants:
        if p.is_hub:
            marker, ms, ec, zorder = "D", 11, "black", 10
            fc = "#e74c3c"
        elif p.is_large:
            marker, ms, ec, zorder = "s", 9, "#2c3e50", 8
            fc = "#3498db"
        else:
            marker, ms, ec, zorder = "o", 6, "#7f8c8d", 6
            fc = "#95a5a6"

        ax.plot(
            p.x,
            p.y,
            marker=marker,
            markersize=ms,
            markerfacecolor=fc,
            markeredgecolor=ec,
            markeredgewidth=1.2,
            zorder=zorder,
        )

    # ── Node labels ──────────────────────────────────────────────
    for p in plants:
        offset = 6 if p.is_large else 4
        ax.text(
            p.x + offset,
            p.y + offset,
            str(p.id),
            fontsize=6,
            color="#2c3e50",
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=15,
        )

    # ── Legend ────────────────────────────────────────────────────
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#e74c3c",
            markeredgecolor="black",
            markersize=10,
            label="Hub node",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#3498db",
            markeredgecolor="#2c3e50",
            markersize=9,
            label="Large plant",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#95a5a6",
            markeredgecolor="#7f8c8d",
            markersize=7,
            label="Small plant",
        ),
        Line2D([0], [0], color=route_cmap(0.35), lw=1.2, alpha=0.7, label="1 product"),
        Line2D(
            [0],
            [0],
            color=route_cmap(0.7),
            lw=2.0,
            alpha=0.7,
            label=f"{max_prods // 2 + 1} products",
        ),
        Line2D(
            [0],
            [0],
            color=route_cmap(1.0),
            lw=3.0,
            alpha=0.7,
            label=f"{max_prods} products",
        ),
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="gray",
            alpha=0.3,
            linestyle="--",
            label="Industrial zone",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        title="Legend",
        title_fontsize=9,
    )

    # ── Formatting ───────────────────────────────────────────────
    ax.set_xlim(-0.05 * L, 1.05 * L)
    ax.set_ylim(-0.05 * L, 1.05 * L)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)", fontsize=11)
    ax.set_ylabel("y (km)", fontsize=11)
    ax.set_title(
        f"SND-RTI Instance: {instance.name}\n"
        f"|V|={len(plants)}  |E|={len(routes)}  "
        f"|R|={instance.n_rti_types}  |P|={instance.n_products}  "
        f"|H|={len(instance.hub_ids)}  zones={len(zone_info)}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def print_summary(instance, plants, routes, zone_info):
    """Print a summary table of the generated instance."""
    print("=" * 65)
    print(f"  Instance: {instance.name}")
    print("=" * 65)
    print(f"  Size class       : {instance.size_class}")
    print(f"  Seed             : {instance.seed}")
    print(f"  Area             : {instance.L} x {instance.L} km²")
    print(f"  Industrial zones : {len(zone_info)}")
    print(f"  Total plants |V| : {len(plants)}")

    n_large = sum(1 for p in plants if p.is_large)
    n_small = sum(1 for p in plants if not p.is_large)
    print(f"    Large plants   : {n_large}")
    print(f"    Small plants   : {n_small}")
    print(f"  Hub candidates   : {len(instance.hub_ids)}")
    print(f"  Full-RTI routes  : {len(routes)}")

    # Products per route distribution
    prods_per_route = [len(r.products) for r in routes]
    print(
        f"  Products/route   : min={min(prods_per_route)}, "
        f"max={max(prods_per_route)}, "
        f"avg={np.mean(prods_per_route):.1f}"
    )

    # Total daily demand
    total_demand = sum(sum(r.demand.values()) for r in routes)
    print(f"  Total daily demand: {total_demand} RTIs")

    print(f"  RTI types |R|    : {instance.n_rti_types}")
    print(f"  Products |P|     : {instance.n_products}")
    print(f"  Transport modes  : {instance.modes}")

    # Distance stats
    dists = [r.distance for r in routes]
    print(
        f"  Route distances  : min={min(dists):.0f}, "
        f"max={max(dists):.0f}, avg={np.mean(dists):.0f} km"
    )

    # Mode availability
    n_ftl_only = sum(1 for d in dists if d >= PARAMS["d_ltl_max"])
    n_ltl_only = sum(1 for d in dists if d < PARAMS["d_ftl_min"])
    n_both = len(dists) - n_ftl_only - n_ltl_only
    print(
        f"  Mode availability: FTL-only={n_ftl_only}, "
        f"LTL-only={n_ltl_only}, both={n_both}"
    )

    # Zone breakdown
    print("-" * 65)
    print(f"  {'Zone':>4}  {'Large':>5}  {'Small':>5}  {'Hubs':>4}  {'Plants':>6}")
    print("-" * 65)
    for z in zone_info:
        n_h = sum(1 for pid in z["plant_ids"] if plants[pid].is_hub)
        print(
            f"  {z['zone_id']:>4}  {z['n_large']:>5}  {z['n_small']:>5}  "
            f"{n_h:>4}  {len(z['plant_ids']):>6}"
        )
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="SND-RTI Instance Generator")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Instance class",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Generate plot")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Generate
    instance, plants, routes, zone_info = generate_instance(args.size, args.seed)

    # Summary
    print_summary(instance, plants, routes, zone_info)

    # Save JSON
    json_path = os.path.join(args.out, f"{instance.name}.json")
    with open(json_path, "w") as f:
        json.dump(asdict(instance), f, indent=2, default=str)
    print(f"\nInstance saved to {json_path}")

    # Plot
    if args.plot:
        plot_path = os.path.join(args.out, f"{instance.name}.png")
        plot_instance(plants, routes, zone_info, instance, save_path=plot_path)
        plt.show()


if __name__ == "__main__":
    main()
