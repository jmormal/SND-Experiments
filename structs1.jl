struct RTIType
    id::Int
    name::String
    v_full::Float64      # v^f_r
    v_empty::Float64     # v^e_r
    weight::Float64        # g_r
    life::Float64        # l_r (days)
    pool::Int            # s^RTI_r
    purchase_cost::Float64          # purchase cost
    purchase_emissions::Float64          # purchase emissions
end

struct Product
    id::Int
    name::String
    weight::Float64      # g_p
end

struct TransportMode
    id::Int
    name::String
    ρ::Float64           # volumetric factor
    qmin::Float64
    qmax::Float64
end

struct Compat
    κ::Float64           # units per RTI
    ric::Float64         # inlay return fraction
    g_inlay::Float64     # gI_{pr}
end

compat::Dict{Tuple{Int,Int},Compat}   # (p, r) => Compat

struct ArcMode                # parameters for one (i,j,m)
    transit::Float64          # l_{ijm}
    fixed_transport_cost::Float64
    fixed_transport_emissions::Float64
    variable_transport_cost::Float64 
    variable_transport_emissions::Float64
    min_volume::Float64; max_volume::Float64  # volume min/max
    min_weight::Float64; max_weight::Float64  # weight min/max
end

struct HubData
    fixed_hub_cost::Float64
    fixed_hub_emissions::Float64
    variable_hub_cost::Float64
    variable_hub_emissions::Float64
end

struct Node
    id::Int
    name::String
    zone::String
    x::Float64
    y::Float64
    hub::Union{Nothing, HubData}   # nothing ⟹ regular node
end

struct Zone
  id::Int
  name::String
  x::Float64
  y::Float64
  importance::Float64
  weight_in::Float64
  weight_out::Float64
end

is_hub(n::Node) = n.hub !== nothing

struct Arc
    i::Int
    j::Int
    modes::Dict{Int,ArcMode}          # only modes with bm_{ijm}=1
    demand::Dict{Int,Float64}         # p => d_{ijp}, only p with d>0 (this IS bf)
    ss::Dict{Int,Float64}             # p => safety stock days
    ser::Dict{Int,Float64}            # p => safety empties days
    empties_ok::Set{Int}              # RTI types r with be_{ijr}=1
end


struct Instance
    nodes::Dict{Int,Node}             # ALL nodes, hubs included
    arcs::Dict{Tuple{Int,Int},Arc}
    rtis::Dict{Int,RTIType}
    products::Dict{Int,Product}
    modes::Dict{Int,TransportMode}
    compat::Dict{Tuple{Int,Int},Compat}
    τ::Vector{Float64}
    α::Float64

    hubs::Vector{Int}                 # precomputed: [id for (id,n) in nodes if is_hub(n)]
    FULL::Vector{NTuple{5,Int}}
    INLAY::Vector{NTuple{5,Int}}
    EMPTY::Vector{NTuple{4,Int}}
    ARCMODE::Vector{NTuple{3,Int}}
    out_arcs::Dict{Int,Vector{Tuple{Int,Int}}}
    in_arcs::Dict{Int,Vector{Tuple{Int,Int}}}
end
