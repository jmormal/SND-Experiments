# ═══════════════════════════════════════════════════════════════════
#  Instance generator for the SND-RTI network design problem.
#  Julia port of generate_instance.py, targeting structs.jl.
#  Configuration values mirror the Python generator (INSTANCE_CONFIGS
#  + PARAMS) 1:1 where possible.
#
#  Automotive setting: OEM plants + tier-1/2 suppliers, RTIs
#  (KLT, GLT, Magnum FLC, stillage, Gitterbox).
#
#  Sources for environmental params: EEA (2023) tkm emission factors,
#  GLEC v3.0; ecoinvent v3.10 / PlasticsEurope for embodied CO2.
#
#  Usage:
#      include("structs.jl"); include("generator.jl")
#      inst = generate_instance(:small, 42)
#      save_instance("SND_RTI_small_42.json", inst)
# ═══════════════════════════════════════════════════════════════════

using Random
using Statistics      # mean
using Distributions   # Poisson, LogNormal
using JSON3
isdefined(Main, :Instance) || include(joinpath(@__DIR__, "structs.jl"))

# ───────────────────────────────────────────────────────────────────
#  INSTANCE CLASS CONFIGURATION  (== Python INSTANCE_CONFIGS)
# ───────────────────────────────────────────────────────────────────

const INSTANCE_CONFIGS = Dict(
    :small => (L = 500.0,  n_zones = 2:3,  n_large = 1:2, n_small = 2:4,
               n_hubs = 1, n_rti = 2:2,    n_products = 3:5,
               max_rti_compat = 2, max_products_per_route = 2),
    :medium => (L = 1000.0, n_zones = 4:6, n_large = 2:3, n_small = 4:8,
               n_hubs = 2, n_rti = 3:4,    n_products = 6:10,
               max_rti_compat = 3, max_products_per_route = 3),
    :large => (L = 1500.0, n_zones = 8:12, n_large = 3:5, n_small = 6:12,
               n_hubs = 3, n_rti = 4:6,    n_products = 10:20,
               max_rti_compat = 4, max_products_per_route = 4),
)

VOLUME_TRUCK = 13.6 * 2.4 * 3
MAX_WEIGHT_TRUCK = 24


# ───────────────────────────────────────────────────────────────────
#  FIXED PARAMETERS  (== Python PARAMS)
# ───────────────────────────────────────────────────────────────────

const PARAMS = (
    # -- spatial ------------------------------------------------------
    sigma_intra = (5.0, 15.0),

    # -- supplier counts (avg. per plant, scaled by zone weight) ------
    k_avg_large = (4.0, 6.0),
    k_avg_small = (2.0, 4.0),

    # -- inter/intra-zone probabilities -------------------------------
    p_inter = 0.7,                    # large plants: prob. supplier in other zone
    p_intra = 0.7,                    # small plants: prob. supplier in same zone

    # -- zone weights -------------------------------------------------
    zone_importance = (0.5, 2.5),
    zone_imbalance  = (0.2, 0.8),

    # -- demand (log-normal, parts/day) -------------------------------
    μ_d = 2.0, σ_d = 0.8,

    # -- transport modes: FTL and LTL ---------------------------------
    # Variable cost/emissions are per km per m³ (EUR/km/m³, kgCO2/km/m³),
    # fixed cost/emissions per shipment. Sources: GLEC v3.0, EEA 2023,
    # WTW; see the Python generator for the full derivation.
    # ρ ≈ 333 kg/m³ chargeable-weight rule (not in Python config; kept).
    modes = [
        (name = "FTL", ρ = MAX_WEIGHT_TRUCK / VOLUME_TRUCK, qmin =0, qmax = 14.0,
         fixed_cost = 1.2, fixed_emis = 2.0,
         cost_km_m3 = 0, emis_km_m3 = 0.10,
         min_vol = 0.0, max_vol = VOLUME_TRUCK,      # ω / δ  (m³ per shipment)
         min_weight = 0.0, max_weight = MAX_WEIGHT_TRUCK,      # ω / δ  (m³ per shipment)
         d_min = 0, d_max = Inf),           # FTL not offered below 50 km
        (name = "LTL", ρ =  MAX_WEIGHT_TRUCK / VOLUME_TRUCK, qmin = 0, qmax = 14.0,
         fixed_cost = 40.0, fixed_emis = 5,
         cost_km_m3 = 1.8/24, emis_km_m3 = 0.18,
         min_vol = 0, max_vol = VOLUME_TRUCK,
         min_weight = 0, max_weight = MAX_WEIGHT_TRUCK,    # ω / δ  (m³ per shipment)
         d_min = 0.0, d_max = Inf),          # LTL not offered above 800 km
    ],
    v_avg = 500.0,          # km/day → transit = floor(dist/v_avg)+1

    # -- RTI lifecycle ------------------------------------------------
    rti_life_years = (5.0, 15.0),     # converted to days (×365) for the model
    rti_pool = 500:10_000,            # existing stock

    # -- RTI embodied CO2 (kg CO2 per kg finished container) ----------
    co2_factor = Dict(:pp => 2.1, :steel => 2.0),

    # -- RTI type params (cost EUR, weight kg, dims m, fold ratio) ----
    rti_type_params = Dict(
        :small_load_carrier => (material = :pp,
            cost = (6.0, 30.0),  wt = (0.5, 4.5),
            dims = ((0.30, 0.60), (0.20, 0.40), (0.14, 0.30)),
            fold = (0.30, 0.45)),
        :large_load_carrier => (material = :pp,
            cost = (60.0, 150.0), wt = (25.0, 50.0),
            dims = ((1.20, 1.20), (0.80, 1.00), (0.59, 0.97)),
            fold = (0.35, 0.50)),
        :foldable_large_container => (material = :pp,
            cost = (150.0, 300.0), wt = (25.0, 42.0),
            dims = ((1.20, 1.20), (0.80, 1.00), (0.75, 1.00)),
            fold = (0.28, 0.35)),
        :metal_stillage => (material = :steel,
            cost = (100.0, 250.0), wt = (60.0, 90.0),
            dims = ((1.20, 1.60), (0.80, 1.20), (0.60, 0.97)),
            fold = (0.40, 0.55)),
        :wire_mesh_container => (material = :steel,
            cost = (80.0, 160.0), wt = (70.0, 85.0),
            dims = ((1.20, 1.24), (0.80, 1.00), (0.80, 0.97)),
            fold = (0.35, 0.45)),
    ),
    rti_archetypes = [
        (name = "KLT-3214",       type = :small_load_carrier),
        (name = "KLT-4328",       type = :small_load_carrier),
        (name = "C-KLT-6428",     type = :small_load_carrier),
        (name = "GLT-1210",       type = :large_load_carrier),
        (name = "GLT-1208",       type = :large_load_carrier),
        (name = "MagnumOpt-1210", type = :foldable_large_container),
        (name = "MagnumOpt-1208", type = :foldable_large_container),
        (name = "Stillage-A",     type = :metal_stillage),
        (name = "Gitterbox-EUR",  type = :wire_mesh_container),
    ],

    # -- compatibility / inlays (kept from previous version; not in
    #    the Python PARAMS block) -------------------------------------
    frac_multi_rti = 0.2,
    frac_inlay     = 0.1,
    ric_range      = (0.10, 0.25),
    inlay_wt_frac  = (0.05, 0.20),    # gI as fraction of RTI tare

    # -- hubs ---------------------------------------------------------
    fhc_range = (50.0, 100.0),  fhe_range = (40.0, 120.0),   # /day
    vhc_range = (0.5, 2.0),      vhe_range = (0.08, 0.30),    # /m³

    # -- maintenance / loss factor (sampled per instance) -------------
    α_range = (0.02, 0.05),

    # -- safety stocks, DAYS (model convention) -----------------------
    ss_days  = 3.0,               # fixed, as in the Python config
    ser_days = (0.5, 2.0),        # not in Python config; kept

    # -- global -------------------------------------------------------
    τ = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0],

    # -- automotive product configuration -----------------------------
    product_cats = ["Powertrain", "Body", "Chassis", "Interior",
                    "Electronics", "Stamping"],
    product_names = Dict(
        "Powertrain"  => ["EngineBlock", "Crankshaft", "CylinderHead",
                          "Turbocharger", "TransmissionCase", "DriveShaft",
                          "ExhaustManifold", "FuelRail"],
        "Body"        => ["DoorPanel", "Hood", "Fender", "BumperCover",
                          "Tailgate", "RoofPanel", "SidePanel", "Pillar"],
        "Chassis"     => ["SubFrame", "ControlArm", "SteeringKnuckle",
                          "BrakeDisc", "SpringAssembly", "StabilizerBar",
                          "WheelHub", "AxleBeam"],
        "Interior"    => ["Dashboard", "SeatFrame", "CenterConsole",
                          "DoorTrim", "HeadlinerAssembly", "AirbagModule",
                          "SteeringWheel", "Carpet"],
        "Electronics" => ["WiringHarness", "ECU", "SensorCluster",
                          "HeadlampModule", "TaillightAssembly",
                          "BatteryModule", "InverterUnit", "DisplayUnit"],
        "Stamping"    => ["CrossMember", "ReinforcementPlate", "BracketSet",
                          "HingeAssembly", "MountingPlate", "ShieldPanel",
                          "SillPlate", "WheelArch"],
    ),
    product_volume = Dict(          # m³ per part
        "Powertrain"  => (0.020, 0.150),
        "Body"        => (0.040, 0.250),
        "Chassis"     => (0.008, 0.080),
        "Interior"    => (0.010, 0.100),
        "Electronics" => (0.002, 0.030),
        "Stamping"    => (0.003, 0.050),
    ),
    # NOTE: the Python config carries no product weights, but the
    # Product struct needs one. Weight is derived as volume × an
    # effective packaged density per category (kg/m³), jittered ±30%.
    product_density = Dict(
        "Powertrain"  => 1500.0,
        "Body"        => 250.0,
        "Chassis"     => 1200.0,
        "Interior"    => 200.0,
        "Electronics" => 400.0,
        "Stamping"    => 900.0,
    ),
)

# ───────────────────────────────────────────────────────────────────
#  HELPERS
# ───────────────────────────────────────────────────────────────────

_u(rng, r::Tuple) = r[1] == r[2] ? float(r[1]) : rand(rng, Uniform(r[1], r[2]))
_i(rng, r::UnitRange) = rand(rng, r)
_euclid(a::Node, b::Node) = sqrt((a.x - b.x)^2 + (a.y - b.y)^2)

# ───────────────────────────────────────────────────────────────────
#  GENERATOR
# ───────────────────────────────────────────────────────────────────

function generate_instance(size::Symbol, seed::Int)::Instance
    rng = MersenneTwister(seed)
    cfg = INSTANCE_CONFIGS[size]
    P = PARAMS
    L = cfg.L

    # -- 1. Products (volume is generator-local; only weight enters the model)
    n_products = _i(rng, cfg.n_products)
    cats = P.product_cats
    products = Dict{Int,Product}()
    prod_volume = Dict{Int,Float64}()
    cat_counter = Dict(c => 0 for c in cats)
    for pid in 1:n_products
        cat = cats[mod1(pid, length(cats))]
        pool = P.product_names[cat]
        k = cat_counter[cat]; cat_counter[cat] += 1
        base = pool[mod1(k + 1, length(pool))]
        name = k < length(pool) ? base : "$(base)_$(k)"
        vol = round(_u(rng, P.product_volume[cat]), digits = 5)
        wt  = round(vol * P.product_density[cat] * _u(rng, (0.7, 1.3)), digits = 3)/1000
        products[pid] = Product(pid, name, wt)
        prod_volume[pid] = vol
    end

    # -- 2. Transport modes
    modes = Dict{Int,TransportMode}()
    for (mid, m) in enumerate(P.modes)
        modes[mid] = TransportMode(mid, m.name, m.ρ, m.qmin, m.qmax)
    end

    # -- 3. RTI types: guarantee ≥1 small and ≥1 large when n_rti ≥ 2
    #    ("small" = small_load_carrier archetypes, "large" = the rest)
    n_rti = _i(rng, cfg.n_rti)
    smalls = filter(a -> a.type == :small_load_carrier, P.rti_archetypes)
    larges = filter(a -> a.type != :small_load_carrier, P.rti_archetypes)
    picked = if n_rti ≥ 2
        sel = Any[rand(rng, smalls), rand(rng, larges)]
        remaining = [a for a in P.rti_archetypes if !(a in sel)]
        while length(sel) < n_rti
            pool = isempty(remaining) ? P.rti_archetypes : remaining
        a = rand(rng, pool)
            push!(sel, a)
            remaining = [x for x in remaining if x !== a]
        end
        shuffle!(rng, sel)
        sel
    else
        Any[rand(rng, P.rti_archetypes)]
    end

    rtis = Dict{Int,RTIType}()
    name_count = Dict{String,Int}()
    for (rid, arch) in enumerate(picked)
        tp = P.rti_type_params[arch.type]
        len = _u(rng, tp.dims[1]); wid = _u(rng, tp.dims[2]); h = _u(rng, tp.dims[3])
        fold = _u(rng, tp.fold)
        v_full  = round(len * wid * h, digits = 4)
        v_empty = round(len * wid * h * fold, digits = 4)
        wt = round(_u(rng, tp.wt), digits = 2)/1000
        c = get(name_count, arch.name, 0); name_count[arch.name] = c + 1
        rtis[rid] = RTIType(rid,
            c == 0 ? arch.name : "$(arch.name)_$c",
            v_full, v_empty, wt,
            round(_u(rng, P.rti_life_years) * 365, digits = 1),   # years → days
            # _i(rng, P.rti_pool),
            0,
            round(_u(rng, tp.cost), digits = 2),
            round(wt * P.co2_factor[tp.material], digits = 2))
    end

    # -- 3b. Product ↔ RTI compatibility
    compat = Dict{Tuple{Int,Int},Compat}()
    for pid in 1:n_products
        cand = if rand(rng) < P.frac_multi_rti && n_rti > 1
            nc = rand(rng, 2:min(cfg.max_rti_compat, n_rti))
            shuffle(rng, collect(1:n_rti))[1:nc]
        else
            [rand(rng, 1:n_rti)]
        end
        ric = rand(rng) < P.frac_inlay ? round(_u(rng, P.ric_range), digits = 3) : 0.0
        ok = Int[]
        for rid in cand
            κ = floor(rtis[rid].v_full / prod_volume[pid])
            κ ≥ 1 || continue
            g_inlay = ric > 0 ? round(rtis[rid].weight * _u(rng, P.inlay_wt_frac), digits = 3) : 0.0
            compat[(pid, rid)] = Compat(κ, ric, g_inlay)
            push!(ok, rid)
        end
        if isempty(ok)   # fallback: largest RTI, κ ≥ 1 forced
            rid = argmax(r -> rtis[r].v_full, collect(keys(rtis)))
            κ = max(1.0, floor(rtis[rid].v_full / prod_volume[pid]))
            g_inlay = ric > 0 ? round(rtis[rid].weight * _u(rng, P.inlay_wt_frac), digits = 3) : 0.0
            compat[(pid, rid)] = Compat(κ, ric, g_inlay)
        end
    end

    # -- 4. Zones and nodes (hubs are nodes with HubData)
    n_zones = _i(rng, cfg.n_zones)
    nodes = Dict{Int,Node}()
    zones = Zone[]
    plants_by_zone = Dict{Int,Vector{Int}}()
    large_ids = Set{Int}()
    nid = 0
    for z in 1:n_zones
        cx, cy = rand(rng) * L, rand(rng) * L
        σ = _u(rng, P.sigma_intra)
        imp = round(_u(rng, P.zone_importance), digits = 2)
        imb = _u(rng, P.zone_imbalance)
        push!(zones, Zone(z, "Z$z", round(cx, digits = 2), round(cy, digits = 2),
                          imp, round(imp * imb * 2, digits = 2), round(imp * (1 - imb) * 2, digits = 2)))
        plants_by_zone[z] = Int[]
        jitter() = (clamp(cx + σ * randn(rng), 0, L), clamp(cy + σ * randn(rng), 0, L))
        for _ in 1:_i(rng, cfg.n_large)
            nid += 1; x, y = jitter()
            nodes[nid] = Node(nid, "OEM_$nid", "Z$z", round(x, digits = 2), round(y, digits = 2), nothing)
            push!(plants_by_zone[z], nid); push!(large_ids, nid)
        end
        for _ in 1:_i(rng, cfg.n_small)
            nid += 1; x, y = jitter()
            nodes[nid] = Node(nid, "supplier_$nid", "Z$z", round(x, digits = 2), round(y, digits = 2), nothing)
            push!(plants_by_zone[z], nid)
        end
        for _ in 1:cfg.n_hubs
            nid += 1; x, y = jitter()
            nodes[nid] = Node(nid, "hub_$nid", "Z$z", round(x, digits = 2), round(y, digits = 2),
                HubData(round(_u(rng, P.fhc_range), digits = 2),
                        round(_u(rng, P.fhe_range), digits = 2),
                        round(_u(rng, P.vhc_range), digits = 4),
                        round(_u(rng, P.vhe_range), digits = 4)))
        end
    end
    plant_ids = sort(vcat(values(plants_by_zone)...))
    zone_of(id) = parse(Int, nodes[id].zone[2:end])

    # -- helper: ArcModes for a given distance.
    #    A mode is offered on an arc only if d_min ≤ dist ≤ d_max
    #    (FTL not below 50 km, LTL not above 800 km).
    function arcmodes(dist;scale_costs = _u(rng, (0.8, 1.3)),
               scale_emissions = _u(rng, (0.6, 1.6)))
        d = Dict{Int,ArcMode}()
        for (mid, m) in enumerate(P.modes)
            m.d_min ≤ dist ≤ m.d_max || continue
            d[mid] = ArcMode(floor(dist / P.v_avg) + 1,
                             round((m.fixed_cost *( m.name != "FTL") + m.fixed_cost * dist*( m.name == "FTL") ) * scale_costs, digits = 2),
                             round(m.fixed_emis * scale_emissions, digits = 4),
                             round(m.cost_km_m3 * dist * scale_costs, digits = 4),   # EUR/m³
                             round(m.emis_km_m3 * dist * scale_emissions, digits = 4),   # kgCO2/m³
                             m.min_vol, m.max_vol,
                             m.min_weight, m.max_weight) # weight cap via ρ
        end
        return d
    end

    # -- 5. Demand arcs (supplier → plant)
    arcs = Dict{Tuple{Int,Int},Arc}()
    for dest in plant_ids
        is_large = dest in large_ids
        k_avg = _u(rng, is_large ? P.k_avg_large : P.k_avg_small)
        zin = zones[zone_of(dest)].weight_in
        n_sup = clamp(rand(rng, Poisson(k_avg * zin)), 1, length(plant_ids) - 1)

        same  = [p for p in plants_by_zone[zone_of(dest)] if p != dest]
        other = [p for p in plant_ids if zone_of(p) != zone_of(dest)]
        p_int = is_large ? P.p_inter : 1.0 - P.p_intra
        ow = [zones[zone_of(p)].weight_out for p in other]
        wsum = sum(ow)

        suppliers = Int[]
        for _ in 1:n_sup
            src = if (rand(rng) < p_int || isempty(same)) && !isempty(other)
                idx = searchsortedfirst(cumsum(ow ./ wsum), rand(rng))
                other[min(idx, length(other))]
            elseif !isempty(same)
                rand(rng, same)
            else
                continue
            end
            src != dest && !(src in suppliers) && push!(suppliers, src)
        end

        for src in suppliers
            haskey(arcs, (src, dest)) && continue
            dist = round(_euclid(nodes[src], nodes[dest]), digits = 2)

            n_p = is_large ? rand(rng, 1:cfg.max_products_per_route) : 1
            pids = shuffle(rng, collect(1:n_products))[1:min(n_p, n_products)]

            demand = Dict{Int,Float64}()
            for pid in pids
                μ = is_large ? P.μ_d + 0.5 * P.σ_d : P.μ_d
                σ = is_large ? 0.6 * P.σ_d : P.σ_d
                demand[pid] = max(1.0, round(rand(rng, LogNormal(μ, σ))))
            end
            # cap daily demand volume so at least one mode is feasible:
            # a shipment every q days carries q·tv m³ and must fit max_vol,
            # so tv ≤ max_m (max_vol_m / qmin_m).  Best-fit RTI per product.
            vol(pid, d) = minimum(d / compat[(pid, r)].κ * rtis[r].v_full
                                  for r in 1:n_rti if haskey(compat, (pid, r)))
            tv = sum(vol(pid, d) for (pid, d) in demand)
            cap = maximum(m.max_vol / m.qmin for m in P.modes)
            if tv > cap
                s = cap / tv
                for pid in keys(demand)
                    demand[pid] = max(1.0, round(demand[pid] * s))
                end
            end

            ss  = Dict(pid => P.ss_days for pid in keys(demand))
            ser = Dict(pid => round(_u(rng, P.ser_days), digits = 1) for pid in keys(demand))

            arcs[(src, dest)] = Arc(src, dest, arcmodes(dist), demand, ss, ser, Set{Int}())
        end
    end

    # -- 5b. Empty-repositioning arcs (hub ↔ plant, hub ↔ hub),
    #        mirroring Python Network.add_edges_model.
    hub_ids = sort!(NodeId[id for (id, n) in nodes if is_hub(n)])

    # Products each plant ships out / receives, derived from the demand arcs
    out_products = Dict{NodeId,Set{ProductId}}(id => Set{ProductId}() for id in keys(nodes))
    in_products  = Dict{NodeId,Set{ProductId}}(id => Set{ProductId}() for id in keys(nodes))
    for ((i, j), a) in arcs
        union!(out_products[i], keys(a.demand))
        union!(in_products[j],  keys(a.demand))
    end

    # RTIs compatible with at least one product in the given set
    compat_rtis(ps::Set{ProductId}) =
        Set{RTIId}(r for r in keys(rtis) if any(haskey(compat, (p, r)) for p in ps))

    # Create (or extend) an arc that only carries empties
    function empty_arc!(src::NodeId, dst::NodeId, rset::Set{RTIId})
        isempty(rset) && return
        key = ArcKey((src, dst))
        if haskey(arcs, key)
            union!(arcs[key].empties_ok, rset)   # demand arc reused for empties
            return
        end
        dist = round(_euclid(nodes[src], nodes[dst]), digits = 2)
        arcs[key] = Arc(src, dst, arcmodes(dist),
                        Dict{ProductId,Float64}(),   # no full-goods demand
                        Dict{ProductId,Float64}(),   # ss
                        Dict{ProductId,Float64}(),   # ser
                        rset)
    end

    all_rtis = Set{RTIId}(keys(rtis))
  # -- empty-flow bounds per (plant, RTI): prune arcs the solver can't use
    only_rti(p) = (rs = [r for r in keys(rtis) if haskey(compat, (p, r))];
                   length(rs) == 1 ? rs[1] : 0)

    # Genenrate max, consume min, consume_max. Node , Rtis to quantity
    gen_max  = Dict{Tuple{NodeId,RTIId},Float64}()
    gen_min  = Dict{Tuple{NodeId,RTIId},Float64}()
    cons_max = Dict{Tuple{NodeId,RTIId},Float64}()
    cons_min = Dict{Tuple{NodeId,RTIId},Float64}()
    for ((i, j), a) in arcs, (p, d) in a.demand, r in keys(rtis)
        haskey(compat, (p, r)) || continue
        u = d / compat[(p, r)].κ
        forced = only_rti(p) == r
        gen_max[(j, r)]  = get(gen_max,  (j, r), 0.0) + u
        cons_max[(i, r)] = get(cons_max, (i, r), 0.0) + u
        forced && (gen_min[(j, r)]  = get(gen_min,  (j, r), 0.0) + u)
        forced && (cons_min[(i, r)] = get(cons_min, (i, r), 0.0) + u)
    end

    # plant n can ever have surplus / deficit of RTI r?
    can_send(n, r)    = get(gen_max, (n, r), 0.0) > get(cons_min, (n, r), 0.0)
    can_receive(n, r) = get(cons_max, (n, r), 0.0) > get(gen_min, (n, r), 0.0)
    # for src in plant_ids, dst in plant_ids
    #     src == dst && continue
    #     empty_arc!(src, dst, compat_rtis(in_products[src]) ∩ compat_rtis(out_products[dst]))
    # end
  
    # for h in hub_ids, pl in plant_ids
    #     # Plant needs empties delivered to pack its OUTGOING products
    #     empty_arc!(h, pl, compat_rtis(out_products[pl]))
    #     # RTIs that arrived full (incoming products) leave the plant empty
    #     empty_arc!(pl, h, compat_rtis(in_products[pl]))
    # end

    # for h in hub_ids, pl in plant_ids
    #     nodes[h].zone != nodes[pl].zone && continue
    #     # Plant needs empties delivered to pack its OUTGOING products
    #     empty_arc!(h, pl, compat_rtis(out_products[pl]))
    #     # RTIs that arrived full (incoming products) leave the plant empty
    #     empty_arc!(pl, h, compat_rtis(in_products[pl]))
    # end

    for h in hub_ids, pl in plant_ids
        nodes[h].zone != nodes[pl].zone && continue
        empty_arc!(h, pl, Set{RTIId}(r for r in compat_rtis(out_products[pl]) if can_receive(pl, r)))
        empty_arc!(pl, h, Set{RTIId}(r for r in compat_rtis(in_products[pl]) if can_send(pl, r)))
    end

    for src in plant_ids, dst in plant_ids
        src == dst && continue
        rset = Set{RTIId}(r for r in compat_rtis(in_products[src]) ∩ compat_rtis(out_products[dst])
                          if can_send(src, r) && can_receive(dst, r))
        empty_arc!(src, dst, rset)
    end
    for h1 in hub_ids, h2 in hub_ids
        h1 == h2 && continue
        empty_arc!(h1, h2, all_rtis)             # hub ↔ hub: all RTI types
    end
# -- 5c. Guarantee reverse arcs for inlay returns (ric > 0).
    #    Inlays must travel back on (j, i) with the same mode; create the
    #    reverse arc if the demand/empty loops didn't already.
    for ((i, j), a) in collect(arcs)          # collect: we mutate `arcs` below
        needs_rev = any(haskey(compat, (p, r)) && compat[(p, r)].ric > 0
                        for p in keys(a.demand), r in keys(rtis))
        needs_rev || continue
        haskey(arcs, (j, i)) && continue
        dist = round(_euclid(nodes[i], nodes[j]), digits = 2)
        arcs[(j, i)] = Arc(j, i, arcmodes(dist),
                           Dict{ProductId,Float64}(),
                           Dict{ProductId,Float64}(),
                           Dict{ProductId,Float64}(),
                           Set{RTIId}())      # no empties; carries inlays only
    end
    # -- 6. Maintenance / loss factor (sampled per instance)
    α = round(_u(rng, P.α_range), digits = 4)

    return build_instance(nodes, arcs, rtis, products, modes, compat,
                          copy(P.τ), α)
end

# ───────────────────────────────────────────────────────────────────
#  SAVE / LOAD  (raw data only; derived sets are recomputed on load)
# ───────────────────────────────────────────────────────────────────

function save_instance(path::String, inst::Instance)
    js = Dict(
        "alpha" => inst.α,
        "tau"   => inst.τ,
        "nodes" => [Dict("id" => n.id, "name" => n.name, "zone" => n.zone,
                         "x" => n.x, "y" => n.y,
                         "hub" => n.hub === nothing ? nothing : Dict(
                             "fixed_hub_cost" => n.hub.fixed_hub_cost,
                             "fixed_hub_emissions" => n.hub.fixed_hub_emissions,
                             "variable_hub_cost" => n.hub.variable_hub_cost,
                             "variable_hub_emissions" => n.hub.variable_hub_emissions))
                    for n in values(inst.nodes)],
        "rtis" => [Dict("id" => r.id, "name" => r.name, "v_full" => r.v_full,
                        "v_empty" => r.v_empty, "weight" => r.weight, "life" => r.life,
                        "pool" => r.pool, "purchase_cost" => r.purchase_cost,
                        "purchase_emissions" => r.purchase_emissions)
                   for r in values(inst.rtis)],
        "products" => [Dict("id" => p.id, "name" => p.name, "weight" => p.weight)
                       for p in values(inst.products)],
        "modes" => [Dict("id" => m.id, "name" => m.name, "rho" => m.ρ,
                         "qmin" => m.qmin, "qmax" => m.qmax)
                    for m in values(inst.modes)],
        "compat" => [Dict("p" => p, "r" => r, "kappa" => c.κ, "ric" => c.ric,
                          "g_inlay" => c.g_inlay)
                     for ((p, r), c) in inst.compat],
        "arcs" => [Dict("i" => a.i, "j" => a.j,
                        "modes" => [Dict("m" => m, "transit" => am.transit,
                                         "fixed_transport_cost" => am.fixed_transport_cost,
                                         "fixed_transport_emissions" => am.fixed_transport_emissions,
                                         "variable_transport_cost" => am.variable_transport_cost,
                                         "variable_transport_emissions" => am.variable_transport_emissions,
                                         "min_volume" => am.min_volume, "max_volume" => am.max_volume,
                                         "min_weight" => am.min_weight, "max_weight" => am.max_weight)
                                    for (m, am) in a.modes],
                        "demand" => [Dict("p" => p, "d" => d,
                                          "ss" => a.ss[p], "ser" => a.ser[p])
                                     for (p, d) in a.demand],
                        "empties_ok" => sort!(collect(a.empties_ok)))
                   for a in values(inst.arcs)],
    )
    open(path, "w") do io
        JSON3.pretty(io, js)
    end
end

function load_instance(path::String)::Instance
    js = JSON3.read(read(path, String))
    nodes = Dict{Int,Node}(n.id => Node(n.id, n.name, n.zone, n.x, n.y,
        n.hub === nothing ? nothing :
            HubData(n.hub.fixed_hub_cost, n.hub.fixed_hub_emissions,
                    n.hub.variable_hub_cost, n.hub.variable_hub_emissions))
        for n in js.nodes)
    rtis = Dict{Int,RTIType}(r.id => RTIType(r.id, r.name, r.v_full, r.v_empty,
        r.weight, r.life, r.pool, r.purchase_cost, r.purchase_emissions) for r in js.rtis)
    products = Dict{Int,Product}(p.id => Product(p.id, p.name, p.weight) for p in js.products)
    modes = Dict{Int,TransportMode}(m.id => TransportMode(m.id, m.name, m.rho, m.qmin, m.qmax)
        for m in js.modes)
    compat = Dict{Tuple{Int,Int},Compat}((c.p, c.r) => Compat(c.kappa, c.ric, c.g_inlay)
        for c in js.compat)
    arcs = Dict{Tuple{Int,Int},Arc}()
    for a in js.arcs
        arcs[(a.i, a.j)] = Arc(a.i, a.j,
            Dict{Int,ArcMode}(am.m => ArcMode(am.transit,
                am.fixed_transport_cost, am.fixed_transport_emissions,
                am.variable_transport_cost, am.variable_transport_emissions,
                am.min_volume, am.max_volume, am.min_weight, am.max_weight)
                for am in a.modes),
            Dict{Int,Float64}(d.p => d.d for d in a.demand),
            Dict{Int,Float64}(d.p => d.ss for d in a.demand),
            Dict{Int,Float64}(d.p => d.ser for d in a.demand),
            Set{Int}(a.empties_ok))
    end
    return build_instance(nodes, arcs, rtis, products, modes, compat,
                          collect(Float64, js.tau), Float64(js.alpha))
end

# ───────────────────────────────────────────────────────────────────
#  CLI:
#    julia generator.jl --size small --seed 42 --out .
#    julia generator.jl --experiments 10 --out instances/
#      → generates 10 instances per size class (small, medium, large),
#        seeds 1..10, into the given directory (created if missing)
# ───────────────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    args = Dict(ARGS[i] => ARGS[i+1] for i in 1:2:length(ARGS)-1)
    out = get(args, "--out", ".")
    isdir(out) || mkpath(out)

    function _gen(size::Symbol, seed::Int)
        inst = generate_instance(size, seed)
        path = joinpath(out, "SND_RTI_$(size)_$(seed).json")
        save_instance(path, inst)
        nh = length(inst.hubs)
        println("Instance saved to $path")
        println("  |nodes|=$(length(inst.nodes)) (hubs=$nh)  |arcs|=$(length(inst.arcs))")
        println("  |RTI|=$(length(inst.rtis))  |products|=$(length(inst.products))")
        println("  |FULL|=$(length(inst.FULL))  |INLAY|=$(length(inst.INLAY))  |EMPTY|=$(length(inst.EMPTY))")
    end

    if haskey(args, "--experiments")
        n_exp = parse(Int, args["--experiments"])
        for size in sort!(collect(keys(INSTANCE_CONFIGS))), seed in 1:n_exp
            _gen(size, seed)
        end
        println("\nGenerated $n_exp experiments per class ($(n_exp * length(INSTANCE_CONFIGS)) total) in $out")
    else
        size = Symbol(get(args, "--size", "small"))
        seed = parse(Int, get(args, "--seed", "42"))
        _gen(size, seed)
    end
end
