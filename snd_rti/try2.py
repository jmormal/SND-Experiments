"""
Instance generator for the SND-RTI network design problem.
Automotive sector: OEM assembly plants, tier-1/2 suppliers, and
returnable transport items (KLT, GLT, stillages, pallets, etc.).
Generates a Network object using the pydantic data model in classes_pydantic.py.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from classes_pydantic import Edge, Hub, Mode, Network, Plant, Product, RTI, Zone

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
    # Spatial
    "sigma_intra": (5, 15),
    # Supplier counts
    "k_avg_suppliers_large": (4, 6),
    "k_avg_suppliers_small": (2, 4),
    # Inter/intra-zone probabilities
    "p_inter": 0.7,
    "p_intra": 0.7,
    # Zone weights
    "zone_importance_range": (0.5, 2.5),
    "zone_imbalance_range": (0.2, 0.8),
    # Demand
    "mu_d": 2.0,
    "sigma_d": 0.8,
    # Transport cost parameters
    "c_ftl": 0.05,
    "c_ltl": 0.12,
    "f_ftl": 150.0,
    "f_ltl": 40.0,
    # Mode capacity / volume
    "delta_ftl": 100.0,
    "delta_ltl": 30.0,
    "omega_ftl": 10.0,
    "omega_ltl": 0.5,
    # Speed & distance
    "v_avg": 500.0,
    "d_ftl_min": 50.0,
    "d_ltl_max": 800.0,
    # Frequency bounds
    "q_min_ftl": 3,
    "q_max_ftl": 10,
    "q_min_ltl": 1,
    "q_max_ltl": 5,
    # RTI physical dimensions (metres) – automotive containers
    # Covers KLT small-load carriers up to large metal stillages
    "rti_length_range": (0.4, 1.6),
    "rti_width_range": (0.3, 1.2),
    "rti_full_height_range": (0.15, 0.97),
    "rti_fold_ratio_range": (0.20, 0.50),
    # RTI lifecycle (automotive pools typically 5–15 years)
    "rti_life_range": (5.0, 15.0),
    "rti_stock_range": (500, 10000),
    # Automotive RTI archetypes used for naming / typing
    "rti_archetypes": [
        {"name": "KLT-3214", "type": "small_load_carrier"},
        {"name": "KLT-4328", "type": "small_load_carrier"},
        {"name": "KLT-6429", "type": "small_load_carrier"},
        {"name": "GLT-1210", "type": "large_load_carrier"},
        {"name": "GLT-1208", "type": "large_load_carrier"},
        {"name": "Stillage-A", "type": "metal_stillage"},
        {"name": "Stillage-B", "type": "metal_stillage"},
        {"name": "WireMesh-1", "type": "wire_mesh_container"},
        {"name": "EUR-Pallet", "type": "pallet"},
        {"name": "DollyCart-1", "type": "dolly"},
    ],
    # Hub costs
    "fhc_range": (200, 500),
    "vhc_range": (0.5, 2.0),
    # Environmental cost multipliers
    "env_cost_ratio": (0.15, 0.35),
    # Safety-stock days
    "ss_days": 3,
    # Automotive part categories and example part names per category
    "product_categories": [
        "Powertrain",
        "Body",
        "Chassis",
        "Interior",
        "Electronics",
        "Stamping",
    ],
    "product_names": {
        "Powertrain": [
            "EngineBlock",
            "Crankshaft",
            "Cylinder Head",
            "Turbocharger",
            "TransmissionCase",
            "DriveShaft",
            "ExhaustManifold",
            "FuelRail",
        ],
        "Body": [
            "DoorPanel",
            "Hood",
            "Fender",
            "BumperCover",
            "Tailgate",
            "RoofPanel",
            "SidePanel",
            "Pillar",
        ],
        "Chassis": [
            "SubFrame",
            "ControlArm",
            "SteeringKnuckle",
            "BrakeDisc",
            "SpringAssembly",
            "StabilizerBar",
            "WheelHub",
            "AxleBeam",
        ],
        "Interior": [
            "Dashboard",
            "SeatFrame",
            "CenterConsole",
            "DoorTrim",
            "HeadlinerAssembly",
            "AirbagModule",
            "SteeringWheel",
            "Carpet",
        ],
        "Electronics": [
            "WiringHarness",
            "ECU",
            "SensorCluster",
            "HeadlampModule",
            "TaillightAssembly",
            "BatteryModule",
            "InverterUnit",
            "DisplayUnit",
        ],
        "Stamping": [
            "CrossMember",
            "ReinforcementPlate",
            "BracketSet",
            "HingeAssembly",
            "MountingPlate",
            "ShieldPanel",
            "SillPlate",
            "WheelArch",
        ],
    },
    # Volume per unit (m³) by category – drives RTI capacity and demand feasibility
    "product_volume_range": {
        "Powertrain": (0.020, 0.150),  # engine blocks, transmission cases
        "Body": (0.040, 0.250),  # door panels, hoods – large but flat
        "Chassis": (0.008, 0.080),  # control arms, subframes
        "Interior": (0.010, 0.100),  # dashboards to trim pieces
        "Electronics": (0.002, 0.030),  # ECUs, wiring harnesses
        "Stamping": (0.003, 0.050),  # brackets to cross members
    },
}


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════


def _sample_range(rng: np.random.Generator, r, as_int: bool = True):
    """Return a sample from a (lo, hi) tuple or the scalar itself."""
    if isinstance(r, tuple):
        return rng.integers(r[0], r[1] + 1) if as_int else rng.uniform(r[0], r[1])
    return r


def _euclidean(a, b) -> float:
    return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2))


# ═══════════════════════════════════════════════════════════════════
#  MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════


def generate_instance(size: str, seed: int) -> Network:
    rng = np.random.default_rng(seed)
    cfg = INSTANCE_CONFIGS[size]
    L = cfg["L"]

    n_zones = _sample_range(rng, cfg["n_zones"])

    # ------------------------------------------------------------------
    # 1. Products
    # ------------------------------------------------------------------
    n_products = _sample_range(rng, cfg["n_products"])
    categories = PARAMS["product_categories"]
    part_names = PARAMS["product_names"]
    vol_ranges = PARAMS["product_volume_range"]
    products: dict[str, Product] = {}
    # Track how many parts we've drawn from each category to avoid repeats
    cat_counters: dict[str, int] = {c: 0 for c in categories}
    for i in range(n_products):
        cat = categories[i % len(categories)]
        pool = part_names[cat]
        idx = cat_counters[cat] % len(pool)
        name = pool[idx]
        # Deduplicate if instance is very large
        if name in products:
            name = f"{name}_{cat_counters[cat]}"
        vol = round(rng.uniform(*vol_ranges[cat]), 4)
        products[name] = Product(name=name, category=cat, volume=vol)
        cat_counters[cat] += 1
    product_list = list(products.values())  # indexed 0..n_products-1

    # ------------------------------------------------------------------
    # 2. Modes  (FTL = 0, LTL = 1)
    # ------------------------------------------------------------------
    env_ratio = rng.uniform(*PARAMS["env_cost_ratio"])
    ftl = Mode(
        id=0,
        max_vol=PARAMS["delta_ftl"],
        min_vol=PARAMS["omega_ftl"],
        min_q=PARAMS["q_min_ftl"],
        max_q=PARAMS["q_max_ftl"],
        fixed_economical_cost=PARAMS["f_ftl"],
        fixed_environmental_cost=round(PARAMS["f_ftl"] * env_ratio, 2),
        economic_cost_per_km=PARAMS["c_ftl"],
        environmental_cost_per_km=round(PARAMS["c_ftl"] * env_ratio, 4),
    )
    ltl = Mode(
        id=1,
        max_vol=PARAMS["delta_ltl"],
        min_vol=PARAMS["omega_ltl"],
        min_q=PARAMS["q_min_ltl"],
        max_q=PARAMS["q_max_ltl"],
        fixed_economical_cost=PARAMS["f_ltl"],
        fixed_environmental_cost=round(PARAMS["f_ltl"] * env_ratio, 2),
        economic_cost_per_km=PARAMS["c_ltl"],
        environmental_cost_per_km=round(PARAMS["c_ltl"] * env_ratio, 4),
    )
    modes: dict[int, Mode] = {0: ftl, 1: ltl}
    mode_list = [ftl, ltl]

    # ------------------------------------------------------------------
    # 3. RTI types
    # ------------------------------------------------------------------
    n_rti = _sample_range(rng, cfg["n_rti_types"])
    rtis: dict[int, RTI] = {}
    archetypes = PARAMS["rti_archetypes"]
    for r in range(n_rti):
        arch = archetypes[r % len(archetypes)]
        length = round(rng.uniform(*PARAMS["rti_length_range"]), 3)
        width = round(rng.uniform(*PARAMS["rti_width_range"]), 3)
        full_h = round(rng.uniform(*PARAMS["rti_full_height_range"]), 3)
        fold_ratio = rng.uniform(*PARAMS["rti_fold_ratio_range"])
        folded_h = round(full_h * fold_ratio, 3)
        life = round(rng.uniform(*PARAMS["rti_life_range"]), 1)
        stock = int(rng.integers(*PARAMS["rti_stock_range"]))

        # Suffix the archetype name if we wrap around
        suffix = f"_{r // len(archetypes)}" if r >= len(archetypes) else ""

        rtis[r] = RTI(
            id=r,
            name=f"{arch['name']}{suffix}",
            type=arch["type"],
            length=length,
            width=width,
            full_height=full_h,
            folded_height=folded_h,
            average_useful_life=life,
            current_stock=stock,
        )

    # Product ↔ RTI compatibility  (also sets capacity & inlay on RTI)
    # Capacity is derived: floor(RTI.volume_full / product.volume).
    # If a product doesn't fit (capacity < 1), the pairing is dropped.
    frac_multi = 0.2
    product_rti_compat: dict[int, list[int]] = {}
    for pi, prod in enumerate(product_list):
        if rng.random() < frac_multi and n_rti > 1:
            n_compat = rng.integers(2, min(cfg["max_rti_compat"], n_rti) + 1)
            candidates = sorted(
                rng.choice(n_rti, size=n_compat, replace=False).tolist()
            )
        else:
            candidates = [int(rng.integers(0, n_rti))]

        inlay_bool = rng.uniform() < 0.1
        inlay = 0.0

        compat: list[int] = []
        for rid in candidates:
            cap = int(rtis[rid].volume_full // prod.volume)
            if cap < 1:
                continue  # product doesn't physically fit in this RTI
            if inlay_bool:
                inlay = round(rng.uniform(0.10, 0.25), 3)
            rtis[rid].add_product(prod, float(cap), inlay)
            compat.append(rid)

        # Guarantee at least one compatible RTI: pick the largest available
        if not compat:
            rid_largest = max(rtis, key=lambda r: rtis[r].volume_full)
            cap = max(1, int(rtis[rid_largest].volume_full // prod.volume))
            rtis[rid_largest].add_product(prod, float(cap), inlay)
            compat = [rid_largest]

        product_rti_compat[pi] = compat

    # ------------------------------------------------------------------
    # 4. Zones, Plants, Hubs
    #    Zones = regional clusters (e.g. Stuttgart, Munich, Wolfsburg)
    #    Large plants = OEM assembly plants / major tier-1 sites
    #    Small plants = tier-1/tier-2 component suppliers
    #    Hubs = cross-dock / packaging-pool depots
    # ------------------------------------------------------------------
    plants: dict[int, Plant] = {}
    hubs: dict[int, Hub] = {}
    zones: dict[int, Zone] = {}

    pid = 0  # running plant id
    hid = 0  # running hub id

    for z in range(n_zones):
        cx, cy = rng.uniform(0, L), rng.uniform(0, L)
        sigma = _sample_range(rng, PARAMS["sigma_intra"], as_int=False)
        n_large = _sample_range(rng, cfg["n_large_per_zone"])
        n_small = _sample_range(rng, cfg["n_small_per_zone"])
        n_hubs = cfg["n_hubs_per_zone"]

        importance = round(rng.uniform(*PARAMS["zone_importance_range"]), 2)
        imbalance = rng.uniform(*PARAMS["zone_imbalance_range"])
        weight_in = round(importance * imbalance * 2, 2)
        weight_out = round(importance * (1 - imbalance) * 2, 2)

        zone_plants: list[Plant] = []
        zone_hubs: list[Hub] = []

        # OEM / large tier-1 assembly plants
        for _ in range(n_large):
            px = float(np.clip(cx + rng.normal(0, sigma), 0, L))
            py = float(np.clip(cy + rng.normal(0, sigma), 0, L))
            p = Plant(id=pid, x=round(px, 2), y=round(py, 2), zone_id=z, is_large=True)
            plants[pid] = p
            zone_plants.append(p)
            pid += 1

        # Tier-1 / tier-2 component suppliers
        for _ in range(n_small):
            px = float(np.clip(cx + rng.normal(0, sigma), 0, L))
            py = float(np.clip(cy + rng.normal(0, sigma), 0, L))
            p = Plant(id=pid, x=round(px, 2), y=round(py, 2), zone_id=z, is_large=False)
            plants[pid] = p
            zone_plants.append(p)
            pid += 1

        # Cross-dock / packaging-pool depots
        for _ in range(n_hubs):
            hx = float(np.clip(cx + rng.normal(0, sigma), 0, L))
            hy = float(np.clip(cy + rng.normal(0, sigma), 0, L))
            h = Hub(
                id=hid,
                x=round(hx, 2),
                y=round(hy, 2),
                zone_id=z,
                fixed_economic_cost_per_volume=0,
                fixed_co2_cost_per_volume=0,
                variable_economic_cost_per_volume=0,
                variable_co2_cost_per_volume=0,  # TODO : Add good ones
            )
            hubs[hid] = h
            zone_hubs.append(h)
            hid += 1

        zones[z] = Zone(
            id=z,
            x=round(cx, 2),
            y=round(cy, 2),
            importance=importance,
            weight_in=weight_in,
            weight_out=weight_out,
            plants=zone_plants,
            hubs=zone_hubs,
        )

    # ------------------------------------------------------------------
    # 5. Edges  (supplier → OEM/destination plant, carrying parts in RTIs)
    # ------------------------------------------------------------------
    all_plants = list(plants.values())
    n_total_plants = len(all_plants)
    k_avg_large = _sample_range(rng, PARAMS["k_avg_suppliers_large"])
    k_avg_small = _sample_range(rng, PARAMS["k_avg_suppliers_small"])

    edges: dict[int, Edge] = {}
    eid = 0
    seen_pairs: set[tuple[int, int]] = set()

    for dest in all_plants:
        k_avg = k_avg_large if dest.is_large else k_avg_small
        dest_zone = zones[dest.zone_id]

        n_suppliers = max(1, int(rng.poisson(k_avg * dest_zone.weight_in)))
        n_suppliers = min(n_suppliers, n_total_plants - 1)

        # Separate candidate suppliers into same-zone and other-zone
        same_zone_plants = [
            p for p in all_plants if p.zone_id == dest.zone_id and p.id != dest.id
        ]
        other_zone_plants = [p for p in all_plants if p.zone_id != dest.zone_id]

        # Probability weights for choosing inter-zone suppliers
        p_inter = PARAMS["p_inter"] if dest.is_large else (1.0 - PARAMS["p_intra"])

        if other_zone_plants:
            ow = np.array([zones[p.zone_id].weight_out for p in other_zone_plants])
            op = ow / ow.sum() if ow.sum() > 0 else np.ones(len(ow)) / len(ow)
        else:
            op = None

        if same_zone_plants:
            sw = np.ones(len(same_zone_plants)) / len(same_zone_plants)
        else:
            sw = None

        # Sample suppliers
        suppliers: list[Plant] = []
        for _ in range(n_suppliers):
            chosen = None
            if rng.random() < p_inter and other_zone_plants:
                idx = rng.choice(len(other_zone_plants), p=op)
                chosen = other_zone_plants[idx]
            elif same_zone_plants:
                idx = rng.choice(len(same_zone_plants), p=sw)
                chosen = same_zone_plants[idx]
            elif other_zone_plants:
                idx = rng.choice(len(other_zone_plants), p=op)
                chosen = other_zone_plants[idx]

            if chosen and chosen.id != dest.id:
                if chosen.id not in {s.id for s in suppliers}:
                    suppliers.append(chosen)

        # Build an Edge for each supplier → dest pair
        for src in suppliers:
            pair_key = (src.id, dest.id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            dist = round(_euclidean(src, dest), 2)

            # Pick products for this edge
            if dest.is_large:
                n_prods = rng.integers(1, cfg["max_products_per_route"] + 1)
            else:
                n_prods = 1
            prod_indices = rng.choice(
                n_products, size=min(n_prods, n_products), replace=False
            ).tolist()

            demand: dict[Product, float] = {}
            for pi in prod_indices:
                prod = product_list[pi]
                if dest.is_large:
                    d = np.exp(
                        rng.normal(
                            PARAMS["mu_d"] + 0.5 * PARAMS["sigma_d"],
                            PARAMS["sigma_d"] * 0.6,
                        )
                    )
                else:
                    d = np.exp(rng.normal(PARAMS["mu_d"], PARAMS["sigma_d"]))
                demand[prod] = max(1.0, round(d))

            # Cap total demand so the total volume per period is feasible
            # at the tightest combo: FTL mode at minimum frequency.
            # Volume per period = sum(d_p * v_p), must fit in q_min shipments
            # of delta_ftl each.
            max_volume = PARAMS["delta_ftl"] * PARAMS["q_min_ftl"]
            total_vol = sum(d * p.volume for p, d in demand.items())
            if total_vol > max_volume:
                scale = max_volume / total_vol
                demand = {p: max(1.0, round(d * scale)) for p, d in demand.items()}

            # Safety stocks (simple: ss_days × demand)
            ss_factor = PARAMS["ss_days"]
            safety_stock_empties = {
                p: round(v * ss_factor, 1) for p, v in demand.items()
            }
            safety_stocks_fulls = {
                p: round(v * ss_factor * 0.5, 1) for p, v in demand.items()
            }

            # Allowed RTIs for this edge = union of compatible RTIs for
            # all products on this edge
            allowed_empties = []

            # Lead times & costs per mode
            lead_time: dict[Mode, float] = {}
            fixed_econ: dict[Mode, float] = {}
            fixed_co2: dict[Mode, float] = {}
            var_econ: dict[Mode, float] = {}
            var_co2: dict[Mode, float] = {}

            for m in mode_list:
                # Filter modes by distance rules
                if m.id == 0 and dist < PARAMS["d_ftl_min"]:
                    continue
                if m.id == 1 and dist > PARAMS["d_ltl_max"]:
                    continue

                lt = round(dist / PARAMS["v_avg"], 3)
                lead_time[m] = lt

                fc = round(m.fixed_economical_cost + dist * m.economic_cost_per_km, 2)
                fixed_econ[m] = fc
                fixed_co2[m] = round(
                    m.fixed_environmental_cost + dist * m.environmental_cost_per_km, 4
                )
                var_econ[m] = round(m.economic_cost_per_km * dist * 0.1, 4)
                var_co2[m] = round(m.environmental_cost_per_km * dist * 0.1, 4)

            # If no mode passed the distance filter, allow at least LTL
            if not lead_time:
                m = ltl
                lt = round(dist / PARAMS["v_avg"], 3)
                lead_time[m] = lt
                fixed_econ[m] = round(
                    m.fixed_economical_cost + dist * m.economic_cost_per_km, 2
                )
                fixed_co2[m] = round(
                    m.fixed_environmental_cost + dist * m.environmental_cost_per_km, 4
                )
                var_econ[m] = round(m.economic_cost_per_km * dist * 0.1, 4)
                var_co2[m] = round(m.environmental_cost_per_km * dist * 0.1, 4)

            edges[eid] = Edge(
                id=eid,
                source=src,
                target=dest,
                zone_id=dest.zone_id,
                demand=demand,
                safety_stock_empties=safety_stock_empties,
                safety_stocks_fulls=safety_stocks_fulls,
                allowed_empties=allowed_empties,
                allowed_modes=list(lead_time.keys()),
                lead_time=lead_time,
                fixed_economic_cost_per_volume=fixed_econ,
                fixed_co2_cost_per_volume=fixed_co2,
                variable_economic_cost_per_volume=var_econ,
                variable_co2_cost_per_volume=var_co2,
            )
            eid += 1

    # ------------------------------------------------------------------
    # 6. Assemble Network
    # ------------------------------------------------------------------
    network = Network(
        products=products,
        rtis=rtis,
        plants=plants,
        hubs=hubs,
        zones=zones,
        modes=modes,
        edges=edges,
    )
    return network


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Generate an SND-RTI network instance."
    )
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=".")
    args = parser.parse_args()

    print("Generating")
    os.makedirs(args.out, exist_ok=True)
    network = generate_instance(args.size, args.seed)

    name = f"SND_RTI_{args.size}_{args.seed}"
    json_path = os.path.join(args.out, f"{name}.json")
    network.save_json(json_path)

    print(f"Instance '{name}' saved to {json_path}")
    print(
        f"  |Plants|={len(network.plants)}  "
        f"|Hubs|={len(network.hubs)}  "
        f"|Edges|={len(network.edges)}  "
        f"|RTI types|={len(network.rtis)}  "
        f"|Products|={len(network.products)}"
    )


if __name__ == "__main__":
    main()
