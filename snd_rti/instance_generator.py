"""
Instance generator for the SND-RTI network design problem.
Automotive sector: OEM assembly plants, tier-1/2 suppliers, and
returnable transport items (KLT, GLT, stillages, pallets, etc.).
Generates a Network object using the pydantic data model in classes_pydantic.py.

Environmental parameter sources:
  - Transport emissions: EEA (2023) "Specific CO2 emissions per tonne-km
    and per mode of transport in Europe"; GLEC Framework v3.0 (Smart
    Freight Centre, 2023).
  - RTI embodied emissions: ecoinvent v3.10 datasets for injection-moulded
    PP containers, steel stillages, and wood pallets; PlasticsEurope
    Eco-profiles (2022).
  - Hub/warehouse emissions: IEA "Tracking Buildings" (2023) for warehouse
    energy intensity; EU electricity grid emission factor ≈ 0.25 kg CO2/kWh
    (2023 average, Ember).
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
from instance_config import PARAMS, INSTANCE_CONFIGS

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
    cat_counters: dict[str, int] = {c: 0 for c in categories}
    for i in range(n_products):
        cat = categories[i % len(categories)]
        pool = part_names[cat]
        idx = cat_counters[cat] % len(pool)
        name = pool[idx]
        if name in products:
            name = f"{name}_{cat_counters[cat]}"
        vol = round(rng.uniform(*vol_ranges[cat]), 4)
        products[name] = Product(name=name, category=cat, volume=vol)
        cat_counters[cat] += 1
    product_list = list(products.values())

    # ------------------------------------------------------------------
    # 2. Modes  (FTL = 0, LTL = 1)
    # ------------------------------------------------------------------
    ftl = Mode(
        id=0,
        max_vol=PARAMS["delta_ftl"],
        min_vol=PARAMS["omega_ftl"],
        min_q=PARAMS["q_min_ftl"],
        max_q=PARAMS["q_max_ftl"],
        fixed_economical_cost=PARAMS["f_ftl"],
        fixed_environmental_cost=PARAMS["f_co2_ftl"],
        economic_cost_per_km=PARAMS["c_ftl"],
        environmental_cost_per_km=PARAMS["e_ftl"],
    )
    ltl = Mode(
        id=1,
        max_vol=PARAMS["delta_ltl"],
        min_vol=PARAMS["omega_ltl"],
        min_q=PARAMS["q_min_ltl"],
        max_q=PARAMS["q_max_ltl"],
        fixed_economical_cost=PARAMS["f_ltl"],
        fixed_environmental_cost=PARAMS["f_co2_ltl"],
        economic_cost_per_km=PARAMS["c_ltl"],
        environmental_cost_per_km=PARAMS["e_ltl"],
    )
    modes: dict[int, Mode] = {0: ftl, 1: ltl}
    mode_list = [ftl, ltl]

    # ------------------------------------------------------------------
    # 3. RTI types
    # ------------------------------------------------------------------
    # Select archetypes randomly to ensure diverse container mixes across
    # instances.  Guarantee at least one small container (KLT) and one
    # large container (GLT/FLC/stillage/Gitterbox) when n_rti >= 2,
    # reflecting real automotive supply chains that move both small
    # components (fasteners, ECUs) and bulky parts (dashboards, subframes).
    n_rti = _sample_range(rng, cfg["n_rti_types"])
    rtis: dict[int, RTI] = {}
    archetypes = PARAMS["rti_archetypes"]
    type_params_map = PARAMS["rti_type_params"]

    small_archs = [a for a in archetypes if a["type"] == "small_load_carrier"]
    large_archs = [a for a in archetypes if a["type"] != "small_load_carrier"]

    if n_rti >= 2 and small_archs and large_archs:
        # Guarantee one from each size class
        picked = [
            small_archs[rng.integers(len(small_archs))],
            large_archs[rng.integers(len(large_archs))],
        ]
        remaining_pool = [a for a in archetypes if a not in picked]
        n_extra = n_rti - 2
        if n_extra > 0 and remaining_pool:
            extra_idx = rng.choice(
                len(remaining_pool),
                size=min(n_extra, len(remaining_pool)),
                replace=False,
            )
            picked.extend(remaining_pool[i] for i in extra_idx)
        # If we still need more (n_rti > len(archetypes)), wrap with repeats
        while len(picked) < n_rti:
            picked.append(archetypes[rng.integers(len(archetypes))])
        rng.shuffle(picked)
    else:
        # n_rti == 1: pick any archetype at random
        picked = [archetypes[rng.integers(len(archetypes))]]

    # Track name usage for deduplication
    name_counts: dict[str, int] = {}
    for r, arch in enumerate(picked):
        rti_type = arch["type"]
        tp = type_params_map[rti_type]
        dims = tp["dims"]

        # Physical dimensions from type-specific ranges
        length = round(rng.uniform(*dims["length"]), 3)
        width = round(rng.uniform(*dims["width"]), 3)
        full_h = round(rng.uniform(*dims["full_height"]), 3)
        fold_ratio = rng.uniform(*tp["fold_ratio"])
        folded_h = round(full_h * fold_ratio, 3)
        life = round(rng.uniform(*PARAMS["rti_life_range"]), 1)
        stock = int(rng.integers(*PARAMS["rti_stock_range"]))

        # Tare weight from type-specific range
        weight = round(rng.uniform(*tp["weight_range"]), 2)

        # Purchase cost from type-specific range
        purchase_cost = round(rng.uniform(*tp["cost_range"]), 2)

        # Embodied CO2 computed from weight × material factor
        material = tp["material"]
        co2_factor = PARAMS[f"co2_factor_{material}"]
        embodied_co2 = round(weight * co2_factor, 2)

        # Deduplicate names if the same archetype appears more than once
        base_name = arch["name"]
        count = name_counts.get(base_name, 0)
        name_counts[base_name] = count + 1
        name = f"{base_name}_{count}" if count > 0 else base_name

        rtis[r] = RTI(
            id=r,
            name=name,
            type=rti_type,
            length=length,
            width=width,
            full_height=full_h,
            folded_height=folded_h,
            average_useful_life=life,
            current_stock=stock,
            purchase_cost=purchase_cost,
            embodied_co2=embodied_co2,
        )

    # Product ↔ RTI compatibility
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
                continue
            if inlay_bool:
                inlay = round(rng.uniform(0.10, 0.25), 3)
            rtis[rid].add_product(prod, float(cap), inlay)
            compat.append(rid)

        if not compat:
            rid_largest = max(rtis, key=lambda r: rtis[r].volume_full)
            cap = max(1, int(rtis[rid_largest].volume_full // prod.volume))
            rtis[rid_largest].add_product(prod, float(cap), inlay)
            compat = [rid_largest]

        product_rti_compat[pi] = compat

    # ------------------------------------------------------------------
    # 4. Zones, Plants, Hubs
    # ------------------------------------------------------------------
    plants: dict[int, Plant] = {}
    hubs: dict[int, Hub] = {}
    zones: dict[int, Zone] = {}

    pid = 0
    hid = 0

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

        for _ in range(n_large):
            px = float(np.clip(cx + rng.normal(0, sigma), 0, L))
            py = float(np.clip(cy + rng.normal(0, sigma), 0, L))
            p = Plant(id=pid, x=round(px, 2), y=round(py, 2), zone_id=z, is_large=True)
            plants[pid] = p
            zone_plants.append(p)
            pid += 1

        for _ in range(n_small):
            px = float(np.clip(cx + rng.normal(0, sigma), 0, L))
            py = float(np.clip(cy + rng.normal(0, sigma), 0, L))
            p = Plant(id=pid, x=round(px, 2), y=round(py, 2), zone_id=z, is_large=False)
            plants[pid] = p
            zone_plants.append(p)
            pid += 1

        # Hubs with both economic and environmental costs
        for _ in range(n_hubs):
            hx = float(np.clip(cx + rng.normal(0, sigma), 0, L))
            hy = float(np.clip(cy + rng.normal(0, sigma), 0, L))
            h = Hub(
                id=hid,
                x=round(hx, 2),
                y=round(hy, 2),
                zone_id=z,
                fixed_economic_cost_per_volume=round(
                    rng.uniform(*PARAMS["fhc_range"]), 2
                ),
                fixed_co2_cost_per_volume=round(
                    rng.uniform(*PARAMS["fhc_co2_range"]), 2
                ),
                variable_economic_cost_per_volume=round(
                    rng.uniform(*PARAMS["vhc_range"]), 4
                ),
                variable_co2_cost_per_volume=round(
                    rng.uniform(*PARAMS["vhc_co2_range"]), 4
                ),
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
    # 5. Edges
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

        same_zone_plants = [
            p for p in all_plants if p.zone_id == dest.zone_id and p.id != dest.id
        ]
        other_zone_plants = [p for p in all_plants if p.zone_id != dest.zone_id]

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

        for src in suppliers:
            pair_key = (src.id, dest.id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            dist = round(_euclidean(src, dest), 2)

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

            max_volume = PARAMS["delta_ftl"] * PARAMS["q_min_ftl"]
            total_vol = sum(d * p.volume for p, d in demand.items())
            if total_vol > max_volume:
                scale = max_volume / total_vol
                demand = {p: max(1.0, round(d * scale)) for p, d in demand.items()}

            ss_factor = PARAMS["ss_days"]
            safety_stock_empties = {
                p: round(v * ss_factor, 1) for p, v in demand.items()
            }
            safety_stocks_fulls = {
                p: round(v * ss_factor * 0.5, 1) for p, v in demand.items()
            }

            allowed_empties = []

            # Lead times & costs per mode
            lead_time: dict[Mode, float] = {}
            fixed_econ: dict[Mode, float] = {}
            fixed_co2: dict[Mode, float] = {}
            var_econ: dict[Mode, float] = {}
            var_co2: dict[Mode, float] = {}

            for m in mode_list:
                lt = round(dist // PARAMS["v_avg"] + 1, 3)
                lead_time[m] = lt

                # Economic costs
                fixed_econ[m] = round(m.fixed_economical_cost, 2)
                var_econ[m] = round(m.economic_cost_per_km * dist, 4)

                # Environmental costs (independent, not a ratio of economic)
                fixed_co2[m] = round(m.fixed_environmental_cost, 4)
                var_co2[m] = round(m.environmental_cost_per_km * dist, 4)

            # Fallback: ensure at least one mode
            if not lead_time:
                m = ltl
                lt = round(dist // PARAMS["v_avg"] + 1, 3)
                lead_time[m] = lt
                fixed_econ[m] = round(m.fixed_economical_cost, 2)
                fixed_co2[m] = round(dist * m.environmental_cost_per_km, 4)
                var_econ[m] = round(m.economic_cost_per_km * dist, 4)
                var_co2[m] = round(m.environmental_cost_per_km * dist, 4)

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
    # network.add_edges_model()
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

    # Print emission summary
    total_rti_co2 = sum(r.embodied_co2 for r in network.rtis.values())
    avg_rti_co2 = total_rti_co2 / len(network.rtis) if network.rtis else 0
    print(
        f"  Avg RTI embodied CO2={avg_rti_co2:.1f} kg  "
        f"Hub fixed CO2 range="
        f"[{min(h.fixed_co2_cost_per_volume for h in network.hubs.values()):.1f}, "
        f"{max(h.fixed_co2_cost_per_volume for h in network.hubs.values()):.1f}] kg/day"
    )


if __name__ == "__main__":
    main()
