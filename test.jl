# model.jl ─ Linearised RTI-SND model (queue-period formulation)
#
# Convention: variables exist ONLY for tuples in the sparse index sets.
# Any (i,j,m,p,q,r) not present is implicitly zero — sums simply iterate
# over the tuples that exist.

using JuMP

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

"Group a vector of tuples by a key function: key => Vector{tuple}."
function groupby(f, xs)
    d = Dict{Any,Vector{eltype(xs)}}()
    for x in xs
        push!(get!(d, f(x), eltype(xs)[]), x)
    end
    return d
end

"Queue periods q admissible for mode m (replaces q-bounds constraint)."
admissible_q(inst, m) =
    [q for q in eachindex(inst.τ)
       if inst.modes[m].qmin <= inst.τ[q] <= inst.modes[m].qmax]

# ──────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────

function build_model(inst::Instance, optimizer; objective::Symbol = :cost)

    # ---- q-expanded sparse index sets --------------------------------
    # ARCMODEQ: (i,j,m,q)  — a "service": mode m on (i,j) with frequency τ_q
    ARCMODEQ = [(i,j,m,q) for (i,j,m) in inst.ARCMODE
                          for q in admissible_q(inst, m)]

    FULLQ  = [(i,j,m,p,q,r) for (i,j,m,p,r) in inst.FULL
                            for q in admissible_q(inst, m)]
    INLAYQ = [(i,j,m,p,q,r) for (i,j,m,p,r) in inst.INLAY
                            for q in admissible_q(inst, m)]
    EMPTYQ = [(i,j,m,q,r)   for (i,j,m,r) in inst.EMPTY
                            for q in admissible_q(inst, m)]

    # ---- lookup tables (avoid O(n) scans inside constraints) ---------
    fullq_by_arc      = groupby(t -> (t[1], t[2]), FULLQ)              # (i,j)
    fullq_by_service  = groupby(t -> (t[1], t[2], t[3], t[5]), FULLQ)  # (i,j,m,q)
    fullq_by_odp      = groupby(t -> (t[1], t[2], t[4]), FULLQ)        # (i,j,p)
    fullq_by_odpr     = groupby(t -> (t[1], t[2], t[4], t[6]), FULLQ)  # (i,j,p,r)
    inlayq_by_service = groupby(t -> (t[1], t[2], t[3], t[5]), INLAYQ)
    inlayq_by_odpr    = groupby(t -> (t[1], t[2], t[4], t[6]), INLAYQ)
    emptyq_by_service = groupby(t -> (t[1], t[2], t[3], t[4]), EMPTYQ)
    emptyq_out        = groupby(t -> (t[1], t[5]), EMPTYQ)             # (i, r) dispatch
    emptyq_in         = groupby(t -> (t[2], t[5]), EMPTYQ)             # (i, r) receive
    amq_by_arc        = groupby(t -> (t[1], t[2]), ARCMODEQ)
    amq_by_node_out   = groupby(t -> t[1], ARCMODEQ)

    get0(d, k) = get(d, k, valtype(d)())   # empty vector if key absent

    model = Model(optimizer)

    # ──────────────────────────────────────────────────────────────
    # Variables (sparse!)
    # ──────────────────────────────────────────────────────────────
    @variable(model, NF[FULLQ]  >= 0)   # daily full flow
    @variable(model, NI[INLAYQ] >= 0)   # daily inlay-return flow
    @variable(model, NE[EMPTYQ] >= 0)   # daily empty flow

    @variable(model, XF[FULLQ],  Bin)
    @variable(model, XI[INLAYQ], Bin)
    @variable(model, XE[EMPTYQ], Bin)

    # Y = \widecheck{X}_{ijmq}: service (i,j,m,q) is active
    @variable(model, Y[ARCMODEQ], Bin)

    @variable(model, XH[inst.hubs], Bin)                  # hub activation
    @variable(model, Prti[r in keys(inst.rtis)] >= 0, Int) # P_r purchases
    @variable(model, CW[ARCMODEQ] >= 0)                   # chargeable weight/shipment

    # Empties in/out indicator per (node, r) — the \widecheck{XE}_{i·r} / _{·ir}
    NR_out = collect(keys(emptyq_out))
    NR_in  = collect(keys(emptyq_in))
    @variable(model, ZEout[NR_out], Bin)
    @variable(model, ZEin[NR_in],  Bin)

    # ──────────────────────────────────────────────────────────────
    # Derived quantities as expressions (the \overline{} accents)
    # ──────────────────────────────────────────────────────────────
    vf(r) = inst.rtis[r].v_full
    ve(r) = inst.rtis[r].v_empty
    gr(r) = inst.rtis[r].weight
    gp(p) = inst.products[p].weight
    gi(p,r) = inst.compat[(p,r)].g_inlay
    κ(p,r)  = inst.compat[(p,r)].κ
    ric(p,r) = inst.compat[(p,r)].ric
    arc(i,j) = inst.arcs[(i,j)]

    # Per-shipment volume and weight per service — LINEAR because τ_q is data
    @expression(model, VS[s in ARCMODEQ],
        sum(NF[t] * vf(t[6]) * inst.τ[s[4]] for t in get0(fullq_by_service, s)) +
        sum(NI[t] * vf(t[6]) * inst.τ[s[4]] for t in get0(inlayq_by_service, s)) +
        sum(NE[t] * ve(t[5]) * inst.τ[s[4]] for t in get0(emptyq_by_service, s)))

    @expression(model, WS[s in ARCMODEQ],
        sum(NF[t] * (gp(t[4]) + gr(t[6])) * inst.τ[s[4]] for t in get0(fullq_by_service, s)) +
        sum(NI[t] * gi(t[4], t[6])        * inst.τ[s[4]] for t in get0(inlayq_by_service, s)) +
        sum(NE[t] * gr(t[5])              * inst.τ[s[4]] for t in get0(emptyq_by_service, s)))

    # Daily volume dispatched from a node (for hub variable handling)
    @expression(model, Vout[i in inst.hubs],
        sum(NF[t] * vf(t[6]) for s in get0(amq_by_node_out, i)
                             for t in get0(fullq_by_service, s)) +
        sum(NI[t] * vf(t[6]) for s in get0(amq_by_node_out, i)
                             for t in get0(inlayq_by_service, s)) +
        sum(NE[t] * ve(t[5]) for s in get0(amq_by_node_out, i)
                             for t in get0(emptyq_by_service, s)))

    # ──────────────────────────────────────────────────────────────
    # Service selection: at most one (mode, queue) per arc
    # ──────────────────────────────────────────────────────────────
    @constraint(model, [a in keys(amq_by_arc)],
        sum(Y[s] for s in amq_by_arc[a]) <= 1)

    # Flows may only use the selected service
    @constraint(model, [t in FULLQ],  XF[t] <= Y[(t[1],t[2],t[3],t[5])])
    @constraint(model, [t in INLAYQ], XI[t] <= Y[(t[1],t[2],t[3],t[5])])
    @constraint(model, [t in EMPTYQ], XE[t] <= Y[(t[1],t[2],t[3],t[4])])

    # Y active only if it carries something (Rule 3 upper part)
    @constraint(model, [s in ARCMODEQ],
        Y[s] <= sum(XF[t] for t in get0(fullq_by_service, s)) +
                sum(XI[t] for t in get0(inlayq_by_service, s)) +
                sum(XE[t] for t in get0(emptyq_by_service, s)))

    # ──────────────────────────────────────────────────────────────
    # Full-route activation + demand  (eq:full-activation, eq:demand)
    # Exactly one (m, q, r) serves each (i,j,p) with demand.
    # ──────────────────────────────────────────────────────────────
    @constraint(model, [odp in keys(fullq_by_odp)],
        sum(XF[t] for t in fullq_by_odp[odp]) == 1)

    for t in FULLQ
        (i,j,m,p,q,r) = t
        d = arc(i,j).demand[p]
        @constraint(model, NF[t] * κ(p,r) >= d * XF[t])
        @constraint(model, NF[t] <= (d / κ(p,r)) * XF[t])   # tight big-M
    end

    # Empties flow-binary link (tight-ish big-M: total demand-driven flow bound)
    Mempty = sum(d / minimum(κ(p,r) for r in keys(inst.rtis) if haskey(inst.compat,(p,r)))
                 for a in values(inst.arcs) for (p,d) in a.demand; init = 0.0)
    @constraint(model, [t in EMPTYQ], NE[t] <= Mempty * XE[t])

    # ──────────────────────────────────────────────────────────────
    # Inlay returns  (eq:inlay + XI/XF tie, aggregated over m,q)
    # ──────────────────────────────────────────────────────────────
    for (odpr, fw) in fullq_by_odpr
        (i,j,p,r) = odpr
        ric(p,r) == 0 && continue
        rv = get0(inlayq_by_odpr, (j,i,p,r))
        @constraint(model, sum(XI[t] for t in rv) == sum(XF[t] for t in fw))
        @constraint(model, sum(NI[t] for t in rv) >= ric(p,r) * sum(NF[t] for t in fw))
    end

    # ──────────────────────────────────────────────────────────────
    # Flow conservation per (node, RTI type)  (eq:conservation)
    # ──────────────────────────────────────────────────────────────
    nflow_out = groupby(t -> (t[1], t[6]), FULLQ)
    nflow_in  = groupby(t -> (t[2], t[6]), FULLQ)
    iflow_out = groupby(t -> (t[1], t[6]), INLAYQ)
    iflow_in  = groupby(t -> (t[2], t[6]), INLAYQ)
    for i in keys(inst.nodes), r in keys(inst.rtis)
        @constraint(model,
            sum(NF[t] for t in get0(nflow_out, (i,r))) +
            sum(NI[t] for t in get0(iflow_out, (i,r))) +
            sum(NE[t] for t in get0(emptyq_out, (i,r)))
            ==
            sum(NF[t] for t in get0(nflow_in, (i,r))) +
            sum(NI[t] for t in get0(iflow_in, (i,r))) +
            sum(NE[t] for t in get0(emptyq_in, (i,r))))
    end

    # ──────────────────────────────────────────────────────────────
    # Shipment load bounds + chargeable weight
    # ──────────────────────────────────────────────────────────────
    for s in ARCMODEQ
        (i,j,m,q) = s
        am = arc(i,j).modes[m]
        @constraint(model, VS[s] >= am.min_volume * Y[s])
        @constraint(model, VS[s] <= am.max_volume * Y[s])
        @constraint(model, WS[s] >= am.min_weight * Y[s])
        @constraint(model, WS[s] <= am.max_weight * Y[s])
        @constraint(model, CW[s] >= WS[s])
        @constraint(model, CW[s] >= inst.modes[m].ρ * VS[s])
    end

    # ──────────────────────────────────────────────────────────────
    # Hubs  (eq:hub-XH, eq:hub-excl)
    # ──────────────────────────────────────────────────────────────
    for i in inst.hubs
        S = get0(amq_by_node_out, i)
        @constraint(model, XH[i] <= sum(Y[s] for s in S; init = 0))
        for s in S
            @constraint(model, XH[i] >= Y[s])
        end
    end

    # Non-hub nodes: no simultaneous receive & dispatch of empties of type r
    for (k, ts) in emptyq_out
        (i, r) = k
        @constraint(model, ZEout[k] <= sum(XE[t] for t in ts))
        @constraint(model, sum(XE[t] for t in ts) <= length(ts) * ZEout[k])
    end
    for (k, ts) in emptyq_in
        (i, r) = k
        @constraint(model, ZEin[k] <= sum(XE[t] for t in ts))
        @constraint(model, sum(XE[t] for t in ts) <= length(ts) * ZEin[k])
    end
    for i in keys(inst.nodes), r in keys(inst.rtis)
        is_hub(inst.nodes[i]) && continue
        zo = haskey(emptyq_out, (i,r)) ? ZEout[(i,r)] : 0
        zi = haskey(emptyq_in,  (i,r)) ? ZEin[(i,r)]  : 0
        (zo === 0 && zi === 0) && continue
        @constraint(model, zo + zi <= 1)
    end

    # ──────────────────────────────────────────────────────────────
    # RTI pool  (eq:pool-*, purchase link) — all linear expressions
    # ──────────────────────────────────────────────────────────────
    @expression(model, TNCF[t in FULLQ],
        NF[t] * (inst.τ[t[5]] + arc(t[1],t[2]).modes[t[3]].transit +
                 arc(t[1],t[2]).ss[t[4]]))
    @expression(model, TNCI[t in INLAYQ],
        NI[t] * (inst.τ[t[5]] + arc(t[1],t[2]).modes[t[3]].transit))
    @expression(model, TNCE[t in EMPTYQ],
        NE[t] * (inst.τ[t[4]] + arc(t[1],t[2]).modes[t[3]].transit))
    # safety empties: driven by full flows at the packing plant
    @expression(model, SEmp[t in FULLQ],
        NF[t] * get(arc(t[1],t[2]).ser, t[4], 0.0))

    fullq_by_r  = groupby(t -> t[6], FULLQ)
    inlayq_by_r = groupby(t -> t[6], INLAYQ)
    emptyq_by_r = groupby(t -> t[5], EMPTYQ)
    @constraint(model, pool[r in keys(inst.rtis)],
        (1 + inst.α) * (
            sum(TNCF[t] + SEmp[t] for t in get0(fullq_by_r, r)) +
            sum(TNCI[t] for t in get0(inlayq_by_r, r)) +
            sum(TNCE[t] for t in get0(emptyq_by_r, r)))
        <= inst.rtis[r].pool + Prti[r])

    @expression(model, PC[r in keys(inst.rtis)],
        inst.rtis[r].purchase_cost * Prti[r] / inst.rtis[r].life)
    @expression(model, PE[r in keys(inst.rtis)],
        inst.rtis[r].purchase_emissions * Prti[r] / inst.rtis[r].life)

    # ──────────────────────────────────────────────────────────────
    # Transport cost & emissions — daily = per-shipment / τ_q (linear!)
    # ──────────────────────────────────────────────────────────────
    @expression(model, TC[s in ARCMODEQ],
        (CW[s] * arc(s[1],s[2]).modes[s[3]].variable_transport_cost +
         Y[s]  * arc(s[1],s[2]).modes[s[3]].fixed_transport_cost) / inst.τ[s[4]])
    @expression(model, TE[s in ARCMODEQ],
        (WS[s] * arc(s[1],s[2]).modes[s[3]].variable_transport_emissions +
         Y[s]  * arc(s[1],s[2]).modes[s[3]].fixed_transport_emissions) / inst.τ[s[4]])

    @expression(model, HC[i in inst.hubs],
        inst.nodes[i].hub.fixed_hub_cost * XH[i] +
        inst.nodes[i].hub.variable_hub_cost * Vout[i])
    @expression(model, HE[i in inst.hubs],
        inst.nodes[i].hub.fixed_hub_emissions * XH[i] +
        inst.nodes[i].hub.variable_hub_emissions * Vout[i])

    @expression(model, ZC, sum(TC) + sum(PC) + sum(HC; init = 0))
    @expression(model, ZE, sum(TE) + sum(PE) + sum(HE; init = 0))

    if objective == :cost
        @objective(model, Min, ZC)
    elseif objective == :emissions
        @objective(model, Min, ZE)
    else
        error("objective must be :cost or :emissions")
    end

    return model
end

# ε-constraint helper for the bi-objective analysis:
#   m = build_model(inst, HiGHS.Optimizer; objective = :cost)
#   @constraint(m, m[:ZE] <= ε)
#   optimize!(m)
