# ═══════════════════════════════════════════════════════════════════
#  SND-RTI data model
# ═══════════════════════════════════════════════════════════════════

# ---- id aliases (documentation only; all are plain Ints) ----------
const NodeId    = Int
const RTIId     = Int
const ProductId = Int
const ModeId    = Int

const ArcKey    = Tuple{NodeId,NodeId}          # (i, j)
const CompatKey = Tuple{ProductId,RTIId}        # (p, r)

const FullKey    = Tuple{NodeId,NodeId,ModeId,ProductId,RTIId}  # (i,j,m,p,r)
const InlayKey   = Tuple{NodeId,NodeId,ModeId,ProductId,RTIId}  # (j,i,m,p,r)
const EmptyKey   = Tuple{NodeId,NodeId,ModeId,RTIId}            # (i,j,m,r)
const ArcModeKey = Tuple{NodeId,NodeId,ModeId}                  # (i,j,m)

struct RTIType
    id::RTIId
    name::String
    v_full::Float64      # v^f_r
    v_empty::Float64     # v^e_r
    weight::Float64      # g_r (tare)
    life::Float64        # l_r (days)
    pool::Int            # s^RTI_r
    purchase_cost::Float64
    purchase_emissions::Float64
end

struct Product
    id::ProductId
    name::String
    weight::Float64      # g_p
end

struct TransportMode
    id::ModeId
    name::String
    ρ::Float64           # volumetric factor (kg chargeable per m³)
    qmin::Float64
    qmax::Float64
end

struct Compat
    κ::Float64           # units per RTI
    ric::Float64         # inlay return fraction
    g_inlay::Float64     # gI_{pr}
end

struct ArcMode
    transit::Float64
    fixed_transport_cost::Float64
    fixed_transport_emissions::Float64
    variable_transport_cost::Float64
    variable_transport_emissions::Float64
    min_volume::Float64; max_volume::Float64
    min_weight::Float64; max_weight::Float64
end

struct HubData
    fixed_hub_cost::Float64
    fixed_hub_emissions::Float64
    variable_hub_cost::Float64
    variable_hub_emissions::Float64
end

struct Node
    id::NodeId
    name::String
    zone::String
    x::Float64
    y::Float64
    hub::Union{Nothing,HubData}
end

is_hub(n::Node) = n.hub !== nothing

struct Zone
    id::Int
    name::String
    x::Float64
    y::Float64
    importance::Float64
    weight_in::Float64
    weight_out::Float64
end

struct Arc
    i::NodeId
    j::NodeId
    modes::Dict{ModeId,ArcMode}
    demand::Dict{ProductId,Float64}   # p => d_{ijp} > 0  (this IS bf)
    ss::Dict{ProductId,Float64}       # p => safety stock, DAYS
    ser::Dict{ProductId,Float64}      # p => safety empties, DAYS
    empties_ok::Set{RTIId}            # r with be_{ijr}=1
end

struct Instance
    nodes::Dict{NodeId,Node}
    arcs::Dict{ArcKey,Arc}
    rtis::Dict{RTIId,RTIType}
    products::Dict{ProductId,Product}
    modes::Dict{ModeId,TransportMode}
    compat::Dict{CompatKey,Compat}
    τ::Vector{Float64}
    α::Float64
    hubs::Vector{NodeId}
    FULL::Vector{FullKey}
    INLAY::Vector{InlayKey}
    EMPTY::Vector{EmptyKey}
    ARCMODE::Vector{ArcModeKey}
    out_arcs::Dict{NodeId,Vector{ArcKey}}
    in_arcs::Dict{NodeId,Vector{ArcKey}}
end

# ───────────────────────────────────────────────────────────────────
#  Constructor: validates raw data and computes all derived sets.
#  The ONLY sanctioned way to create an Instance.
# ───────────────────────────────────────────────────────────────────
function build_instance(nodes::Dict{NodeId,Node},
                        arcs::Dict{ArcKey,Arc},
                        rtis::Dict{RTIId,RTIType},
                        products::Dict{ProductId,Product},
                        modes::Dict{ModeId,TransportMode},
                        compat::Dict{CompatKey,Compat},
                        τ::Vector{Float64},
                        α::Float64)::Instance

    # ---- validation -------------------------------------------------
    for ((i, j), a) in arcs
        (a.i, a.j) == (i, j)   || error("Arc key ($i,$j) ≠ contents ($(a.i),$(a.j))")
        haskey(nodes, i)       || error("Arc ($i,$j): unknown source node $i")
        haskey(nodes, j)       || error("Arc ($i,$j): unknown target node $j")
        for m in keys(a.modes)
            haskey(modes, m)   || error("Arc ($i,$j): unknown mode $m")
        end
        for (p, d) in a.demand
            d > 0              || error("Arc ($i,$j): demand for p=$p must be > 0 (drop the key instead)")
            haskey(products, p)|| error("Arc ($i,$j): unknown product $p")
            any(haskey(compat, (p, r)) for r in keys(rtis)) ||
                error("Arc ($i,$j): product $p has no compatible RTI type")
            haskey(a.ss, p)    || error("Arc ($i,$j): missing ss for product $p")
            haskey(a.ser, p)   || error("Arc ($i,$j): missing ser for product $p")
        end
        for r in a.empties_ok
            haskey(rtis, r)    || error("Arc ($i,$j): unknown RTI $r in empties_ok")
        end
    end
    for ((p, r), c) in compat
        haskey(products, p) && haskey(rtis, r) || error("compat ($p,$r): unknown id")
        c.κ ≥ 1 || error("compat ($p,$r): κ = $(c.κ) < 1")
    end
    for (m, md) in modes
        md.qmin ≤ md.qmax || error("mode $m: qmin > qmax")
    end
    for t in τ
        any(md.qmin ≤ t ≤ md.qmax for md in values(modes)) ||
            @warn "τ = $t lies outside every mode's [qmin, qmax]; it will never be usable"
    end

    # ---- derived sets ----------------------------------------------
    hubs = sort!([id for (id, n) in nodes if is_hub(n)])

    FULL    = FullKey[]
    INLAY   = InlayKey[]
    EMPTY   = EmptyKey[]
    ARCMODE = ArcModeKey[]

    for ((i, j), a) in arcs
        for m in keys(a.modes)
            push!(ARCMODE, (i, j, m))
            for (p, _) in a.demand, r in keys(rtis)
                haskey(compat, (p, r)) || continue
                push!(FULL, (i, j, m, p, r))
                if compat[(p, r)].ric > 0
                    # inlay returns on (j,i); require the mode there
                    rev = get(arcs, (j, i), nothing)
                    (rev !== nothing && haskey(rev.modes, m)) ||
                        error("Product $p (ric>0) on ($i,$j,m=$m) needs reverse arc ($j,$i) with mode $m")
                    push!(INLAY, (j, i, m, p, r))
                end
            end
            for r in a.empties_ok
                push!(EMPTY, (i, j, m, r))
            end
        end
    end
    sort!(FULL); sort!(unique!(INLAY)); sort!(EMPTY); sort!(ARCMODE)

    out_arcs = Dict{NodeId,Vector{ArcKey}}(id => ArcKey[] for id in keys(nodes))
    in_arcs  = Dict{NodeId,Vector{ArcKey}}(id => ArcKey[] for id in keys(nodes))
    for (i, j) in keys(arcs)
        push!(out_arcs[i], (i, j))
        push!(in_arcs[j], (i, j))
    end

    return Instance(nodes, arcs, rtis, products, modes, compat, τ, α,
                    hubs, FULL, INLAY, EMPTY, ARCMODE, out_arcs, in_arcs)
end
