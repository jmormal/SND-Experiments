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
    # ── Spatial ────────────────────────────────────────────────────
    "sigma_intra": (5, 15),
    # ── Supplier counts ───────────────────────────────────────────
    "k_avg_suppliers_large": (4, 6),
    "k_avg_suppliers_small": (2, 4),
    # ── Inter/intra-zone probabilities ────────────────────────────
    "p_inter": 0.7,
    "p_intra": 0.7,
    # ── Zone weights ──────────────────────────────────────────────
    "zone_importance_range": (0.5, 2.5),
    "zone_imbalance_range": (0.2, 0.8),
    # ── Demand ────────────────────────────────────────────────────
    "mu_d": 2.0,
    "sigma_d": 0.8,
    # ==============================================================
    # TRANSPORT — ECONOMIC COST
    # ==============================================================
    # Variable cost per km per m³ (EUR/km/m³)
    #   FTL: ~0.04–0.06 EUR/km/m³ (full-truckload, high utilisation)
    #   LTL: ~0.10–0.15 EUR/km/m³ (groupage, lower utilisation, extra handling)
    # Source: typical Western-European road freight rates (2022–2024).
    "c_ftl": 0.05,  # EUR/km/m³
    "c_ltl": 0.12,  # EUR/km/m³
    # Fixed dispatch cost per shipment (EUR/shipment)
    #   FTL: 120–180 EUR (driver dispatch, truck allocation, admin)
    #   LTL: 30–50 EUR  (collection stop cost)
    "f_ftl": 150.0,  # EUR/shipment
    "f_ltl": 40.0,  # EUR/shipment
    # ==============================================================
    # TRANSPORT — ENVIRONMENTAL COST (CO2)
    # ==============================================================
    # Variable emission intensity (kg CO2 / km / m³)
    #
    # Derivation (GLEC Framework v3.0, EEA 2023):
    #   A 40-t articulated truck emits ~0.062 kg CO2 per tonne-km
    #   (tank-to-wheel, EU average 2022).  Assuming average cargo
    #   density 250–350 kg/m³ and typical load factor:
    #     FTL (load factor ~85%): 0.062 * 300 / 1000 / 0.85 ≈ 0.022
    #       → we use 0.025 to account for well-to-wheel (WTW adds ~20%)
    #     LTL (load factor ~55%, extra routing +30% km):
    #       0.062 * 300 / 1000 / 0.55 * 1.3 ≈ 0.044
    #       → we use 0.045
    #
    # These are mid-range estimates for European road freight.
    # For sensitivity, ±30% is reasonable.
    "e_ftl": 0.025,  # kg CO2 / km / m³  (FTL, WTW)
    "e_ltl": 0.045,  # kg CO2 / km / m³  (LTL, WTW)
    # Fixed dispatch emissions (kg CO2 / shipment)
    #   Covers engine cold-start, depot manoeuvring, loading/unloading
    #   equipment.  Estimated from fuel consumption of ~2–4 litres
    #   diesel per dispatch event (2.68 kg CO2 / litre diesel).
    #     FTL: ~3 L * 2.68 ≈ 8 kg CO2
    #     LTL: ~1 L * 2.68 ≈ 3 kg CO2 (smaller vehicle, shorter idle)
    "f_co2_ftl": 5.0,  # kg CO2 / shipment
    "f_co2_ltl": 2.0,  # kg CO2 / shipment
    # ==============================================================
    # TRANSPORT — MODE CAPACITY & OPERATIONS
    # ==============================================================
    "delta_ftl": 100.0,  # m³ max volume per FTL shipment
    "delta_ltl": 30.0,  # m³ max volume per LTL shipment
    "omega_ftl": 10.0,  # m³ min volume per FTL shipment
    "omega_ltl": 0.5,  # m³ min volume per LTL shipment
    # Speed & distance
    "v_avg": 500.0,  # km/day average road speed
    "d_ftl_min": 50.0,  # km — below this, FTL is not offered
    "d_ltl_max": 800.0,  # km — above this, LTL is not offered
    # Frequency bounds (days between dispatches)
    "q_min_ftl": 3,
    "q_max_ftl": 10,
    "q_min_ltl": 1,
    "q_max_ltl": 5,
    # ==============================================================
    # RTI — LIFECYCLE
    # ==============================================================
    "rti_life_range": (5.0, 15.0),  # years
    "rti_stock_range": (500, 10000),
    # ==============================================================
    # RTI — PURCHASE COST, WEIGHT, AND EMBODIED EMISSIONS BY TYPE
    # ==============================================================
    # CO2 estimation methodology:
    #   Embodied CO2 is computed as:
    #       embodied_co2 = tare_weight_kg × co2_factor_per_kg
    #   where the factor depends on the primary material:
    #
    #   PP (polypropylene) containers — KLT, GLT, Magnum Optimum:
    #     Raw material:   1.65–1.78 kg CO2/kg PP cradle-to-gate
    #                     (PlasticsEurope Eco-profiles 2022;
    #                      ScienceDirect doi:10.1016/j.jclepro.2020.120520)
    #     Manufacturing:  +15–25% for injection moulding energy
    #                     (ecoinvent v3.10, injection moulding, RER)
    #     Total:          ~2.1 kg CO2/kg finished PP container
    #
    #   Steel containers — stillages, wire mesh / Gitterbox:
    #     Raw material:   1.8 kg CO2/kg hot-rolled steel section
    #                     (ecoinvent v3.10, steel hot rolling, RER)
    #     Manufacturing:  +10–15% for welding, bending, assembly
    #     Total:          ~2.0 kg CO2/kg finished steel container
    #
    # Purchase cost:
    #   KLT: retail prices from Salesbridges, Traydon, Eurobox
    #        (VDA 4500 standard, 2023–2025 catalogue prices excl. VAT)
    #   GLT: B2B list prices, 2023–2024
    #   Magnum Optimum: Schoeller Allibert / Transoplast B2B pricing
    #        (~£148 used Grade-A from plasticboxsales.co.uk;
    #         new units typically €150–300 depending on configuration)
    #   Steel: industry quotes, 2023 Central European market
    #
    # Container weights from manufacturer datasheets:
    #   KLT-3214 (300×200×147mm): 0.57 kg (VDA R-KLT, Traydon)
    #   KLT-4328 (400×300×280mm): 2.0 kg  (VDA R-KLT, SSI Schaefer)
    #   C-KLT-6428 (600×400×280mm): 4.4 kg (VDA C-KLT, Eurobox)
    #   GLT-1210 rigid: 35–50 kg
    #   Magnum Optimum 1210: 30–42 kg PP (Schoeller Allibert)
    #   Magnum Optimum 1208: 25–35 kg PP
    #   Gitterbox EUR: ~85 kg steel
    #   Metal stillage: 60–90 kg steel
    "co2_factor_pp": 2.1,  # kg CO2 per kg finished PP container
    "co2_factor_steel": 2.0,  # kg CO2 per kg finished steel container
    "rti_type_params": {
        "small_load_carrier": {
            # KLT series (VDA 4500): injection-moulded PP, 300×200 to 600×400 base
            "material": "pp",
            # EUR — €6 for KLT-3214 to €30 for C-KLT-6428
            "cost_range": (6, 30),
            "weight_range": (0.5, 4.5),  # kg — 0.57 kg (3214) to 4.4 kg (6428)
            "dims": {  # physical dimensions in metres
                "length": (0.30, 0.60),
                "width": (0.20, 0.40),
                "full_height": (0.14, 0.30),
            },
            # non-foldable KLTs nest ~35-40% height
            "fold_ratio": (0.30, 0.45),
        },
        "large_load_carrier": {
            # GLT series: rigid PP pallet boxes, 1200×1000 or 1200×800
            "material": "pp",
            "cost_range": (60, 150),  # EUR — rigid GLT
            "weight_range": (25, 50),  # kg
            "dims": {
                "length": (1.20, 1.20),
                "width": (0.80, 1.00),
                "full_height": (0.59, 0.97),
            },
            # rigid GLTs don't fold much; stacking ratio
            "fold_ratio": (0.35, 0.50),
        },
        "foldable_large_container": {
            # Magnum Optimum (Schoeller Allibert / IPL): foldable PP FLC
            # The automotive industry's preferred FLC.
            # Folds to 295mm when empty → fold ratio ~0.30–0.35
            # 750 kg unit load, 3200 kg stack load
            "material": "pp",
            "cost_range": (150, 300),  # EUR — new B2B price
            "weight_range": (25, 42),  # kg — lighter than rigid GLT due to
            #       welded double-wall PP construction
            "dims": {
                "length": (1.20, 1.20),
                "width": (0.80, 1.00),
                "full_height": (0.75, 1.00),
            },
            "fold_ratio": (0.28, 0.35),  # key advantage: folds to ~295mm
        },
        "metal_stillage": {
            # Welded steel frames for heavy automotive parts (engines, subframes)
            "material": "steel",
            "cost_range": (100, 250),  # EUR
            "weight_range": (60, 90),  # kg
            "dims": {
                "length": (1.20, 1.60),
                "width": (0.80, 1.20),
                "full_height": (0.60, 0.97),
            },
            "fold_ratio": (0.40, 0.55),  # limited collapsibility
        },
        "wire_mesh_container": {
            # EUR Gitterbox and similar wire-mesh containers
            "material": "steel",
            "cost_range": (80, 160),  # EUR
            "weight_range": (70, 85),  # kg — standard Gitterbox ~85 kg
            "dims": {
                "length": (1.20, 1.24),  # EUR Gitterbox = 1240×835×970mm
                "width": (0.80, 1.00),
                "full_height": (0.80, 0.97),
            },
            "fold_ratio": (0.35, 0.45),  # half-flap folding
        },
    },
    # RTI archetypes — containers and boxes only (no pallets, no dollies)
    "rti_archetypes": [
        {"name": "KLT-3214", "type": "small_load_carrier"},
        {"name": "KLT-4328", "type": "small_load_carrier"},
        {"name": "C-KLT-6428", "type": "small_load_carrier"},
        {"name": "GLT-1210", "type": "large_load_carrier"},
        {"name": "GLT-1208", "type": "large_load_carrier"},
        {"name": "MagnumOpt-1210", "type": "foldable_large_container"},
        {"name": "MagnumOpt-1208", "type": "foldable_large_container"},
        {"name": "Stillage-A", "type": "metal_stillage"},
        {"name": "Gitterbox-EUR", "type": "wire_mesh_container"},
    ],
    # ==============================================================
    # HUBS — ECONOMIC AND ENVIRONMENTAL COST
    # ==============================================================
    # Fixed activation cost (EUR/day and kg CO2/day):
    #   Covers building energy (lighting, HVAC, IT), staffing overhead.
    #   A medium cross-dock warehouse (2 000–5 000 m²) in Central Europe
    #   consumes 150–400 kWh/day of electricity + gas.
    #   At EU-average grid factor ~0.25 kg CO2/kWh (Ember, 2023):
    #     low:  150 kWh * 0.25 = 38 → round to 40 kg CO2/day
    #     high: 400 kWh * 0.25 = 100 → round to 120 kg CO2/day
    #   Monetary cost: 200–500 EUR/day (rent, energy, base labour).
    "fhc_range": (200, 500),  # EUR/day — fixed economic cost
    "fhc_co2_range": (40.0, 120.0),  # kg CO2/day — fixed emissions
    # Variable handling cost (EUR/m³ and kg CO2/m³):
    #   Forklift operations, scanning, sorting, re-palletising.
    #   A diesel forklift uses ~3–5 L/hour, handles ~40–60 m³/hour:
    #     (4 L/h * 2.68 kg CO2/L) / 50 m³/h ≈ 0.21 kg CO2/m³
    #   Electric forklifts: ~8 kWh/h / 50 m³/h * 0.25 ≈ 0.04 kg CO2/m³
    #   We use a mixed fleet range of 0.08–0.30 kg CO2/m³.
    #   Monetary cost: 0.5–2.0 EUR/m³ (labour + equipment + energy).
    "vhc_range": (0.5, 2.0),  # EUR/m³ — variable economic cost
    "vhc_co2_range": (0.08, 0.30),  # kg CO2/m³ — variable emissions
    # ==============================================================
    # MAINTENANCE / LOSS FACTOR
    # ==============================================================
    "alpha_range": (0.02, 0.05),
    # ── Safety-stock days ─────────────────────────────────────────
    "ss_days": 3,
    # ==============================================================
    # AUTOMOTIVE PRODUCT CONFIGURATION (unchanged)
    # ==============================================================
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
    "product_volume_range": {
        "Powertrain": (0.020, 0.150),
        "Body": (0.040, 0.250),
        "Chassis": (0.008, 0.080),
        "Interior": (0.010, 0.100),
        "Electronics": (0.002, 0.030),
        "Stamping": (0.003, 0.050),
    },
}
