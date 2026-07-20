# model.jl ─ Linearised RTI-SND model (queue-period formulation)
#
# Convention: variables exist ONLY for tuples in the sparse index sets.
# Any index not present is implicitly zero — sums simply iterate over
# the tuples that exist.
#
# Index tuples are NamedTuples:
#   service  s ∈ ARCMODEQ : (i, j, m, q)
#   full     t ∈ FULLQ    : (i, j, m, p, q, r)
#   inlay    t ∈ INLAYQ   : (i, j, m, p, q, r)
#   empty    t ∈ EMPTYQ   : (i, j, m, q, r)
# so constraints read t.i, t.p, t.r etc.

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

"The service (i,j,m,q) a full/inlay/empty tuple rides on."
svc(t) = (i = t.i, j = t.j, m = t.m, q = t.q)

# ──────────────────────────────────────────────────────────────────
# Human-readable labels
# ──────────────────────────────────────────────────────────────────

node_name(inst, i)  = inst.nodes[i].name
mode_name(inst, m)  = inst.modes[m].name
prod_name(inst, p)  = inst.products[p].name
rti_name(inst, r)   = inst.rtis[r].name
q_label(inst, q)    = "q=$(inst.τ[q])d"

describe_service(inst, s) =
    "$(node_name(inst, s.i))→$(node_name(inst, s.j)), " *
    "$(mode_name(inst, s.m)), $(q_label(inst, s.q))"

describe_full(inst, t) =            # also for inlay tuples (same fields)
    "$(node_name(inst, t.i))→$(node_name(inst, t.j)), " *
    "$(mode_name(inst, t.m)), $(prod_name(inst, t.p)), " *
    "$(q_label(inst, t.q)), $(rti_name(inst, t.r))"

describe_empty(inst, t) =
    "$(node_name(inst, t.i))→$(node_name(inst, t.j)), " *
    "$(mode_name(inst, t.m)), $(q_label(inst, t.q)), $(rti_name(inst, t.r))"

"""
Print the IIS with instance names. Build the model with
`verbose_names = true`, optimize (with DualReductions = 0), then call this.
"""
function print_iis(model)
    compute_conflict!(model)
    println("── IIS ─────────────────────────────────────────────")
    for (F, S) in list_of_constraint_types(model), c in all_constraints(model, F, S)
        if MOI.get(model, MOI.ConstraintConflictStatus(), c) == MOI.IN_CONFLICT
            println(c)
        end
    end
    println("────────────────────────────────────────────────────")
end

# ──────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────

function build_model(inst::Instance, optimizer;
                     objective::Symbol = :cost,
                     verbose_names::Bool = false, one_to_one::Bool = false)

    # ---- q-expanded sparse index sets (NamedTuples) ------------------
    ARCMODEQ = [(i = i, j = j, m = m, q = q)
                for (i, j, m) in inst.ARCMODE
                for q in admissible_q(inst, m)]

    FULLQ  = [(i = i, j = j, m = m, p = p, q = q, r = r)
              for (i, j, m, p, r) in inst.FULL
              for q in admissible_q(inst, m)]
    INLAYQ = [(i = i, j = j, m = m, p = p, q = q, r = r)
              for (i, j, m, p, r) in inst.INLAY
              for q in admissible_q(inst, m)]
    EMPTYQ = [(i = i, j = j, m = m, q = q, r = r)
              for (i, j, m, r) in inst.EMPTY
              for q in admissible_q(inst, m)]

    # ---- lookup tables (avoid O(n) scans inside constraints) ---------
    fullq_by_service  = groupby(svc, FULLQ)                       # (i,j,m,q)
    fullq_by_odp      = groupby(t -> (t.i, t.j, t.p), FULLQ)
    fullq_by_odpr     = groupby(t -> (t.i, t.j, t.p, t.r), FULLQ)
    inlayq_by_service = groupby(svc, INLAYQ)
    inlayq_by_odpr    = groupby(t -> (t.i, t.j, t.p, t.r), INLAYQ)
    emptyq_by_service = groupby(svc, EMPTYQ)
    emptyq_out        = groupby(t -> (t.i, t.r), EMPTYQ)          # dispatch
    emptyq_in         = groupby(t -> (t.j, t.r), EMPTYQ)          # receive
    amq_by_arc        = groupby(s -> (s.i, s.j), ARCMODEQ)
    amq_by_node_out   = groupby(s -> s.i, ARCMODEQ)

    get0(d, k) = get(d, k, valtype(d)())   # empty vector if key absent

    model = Model(optimizer)

    # Register index sets for downstream code (solution extraction etc.).
    model[:ARCMODEQ] = ARCMODEQ
    model[:FULLQ]    = FULLQ
    model[:INLAYQ]   = INLAYQ
    model[:EMPTYQ]   = EMPTYQ

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

    @variable(model, XH[inst.hubs], Bin)                   # hub activation
    @variable(model, Prti[r in keys(inst.rtis)] >= 0, Int) # P_r purchases
    @variable(model, CW[ARCMODEQ] >= 0)                    # chargeable weight/shipment

    # Empties in/out indicator per (node, r) — \widecheck{XE}_{i·r} / _{·ir}
    NR_out = collect(keys(emptyq_out))
    NR_in  = collect(keys(emptyq_in))
    @variable(model, ZEout[NR_out], Bin)
    @variable(model, ZEin[NR_in],  Bin)

    # ---- optional: embed instance names into variable names ----------
    # Makes Gurobi logs, IIS dumps, and write_to_file(model, "m.lp")
    # directly readable. Off by default (naming costs build time).
    if verbose_names
        for t in FULLQ
            set_name(NF[t], "NF[$(describe_full(inst, t))]")
            set_name(XF[t], "XF[$(describe_full(inst, t))]")
        end
        for t in INLAYQ
            set_name(NI[t], "NI[$(describe_full(inst, t))]")
            set_name(XI[t], "XI[$(describe_full(inst, t))]")
        end
        for t in EMPTYQ
            set_name(NE[t], "NE[$(describe_empty(inst, t))]")
            set_name(XE[t], "XE[$(describe_empty(inst, t))]")
        end
        for s in ARCMODEQ
            set_name(Y[s],  "Y[$(describe_service(inst, s))]")
            set_name(CW[s], "CW[$(describe_service(inst, s))]")
        end
        for i in inst.hubs
            set_name(XH[i], "XH[$(node_name(inst, i))]")
        end
        for r in keys(inst.rtis)
            set_name(Prti[r], "Prti[$(rti_name(inst, r))]")
        end
        for k in NR_out
            set_name(ZEout[k], "ZEout[$(node_name(inst, k[1])), $(rti_name(inst, k[2]))]")
        end
        for k in NR_in
            set_name(ZEin[k], "ZEin[$(node_name(inst, k[1])), $(rti_name(inst, k[2]))]")
        end
    end

    # ──────────────────────────────────────────────────────────────
    # Data accessors
    # ──────────────────────────────────────────────────────────────
    vf(r) = inst.rtis[r].v_full
    ve(r) = inst.rtis[r].v_empty
    gr(r) = inst.rtis[r].weight
    gp(p) = inst.products[p].weight
    gi(p, r)  = inst.compat[(p, r)].g_inlay
    κ(p, r)   = inst.compat[(p, r)].κ
    ric(p, r) = inst.compat[(p, r)].ric
    arc(i, j) = inst.arcs[(i, j)]

    # Per-shipment volume and weight per service — LINEAR because τ_q is data
    @expression(model, VS[s in ARCMODEQ],
        sum(NF[t] * vf(t.r) * inst.τ[s.q] for t in get0(fullq_by_service, s)) +
        sum(NI[t] * vf(t.r) * inst.τ[s.q] for t in get0(inlayq_by_service, s)) +
        sum(NE[t] * ve(t.r) * inst.τ[s.q] for t in get0(emptyq_by_service, s)))

    @expression(model, WS[s in ARCMODEQ],
        sum(NF[t] * (gp(t.p) + gr(t.r)) * inst.τ[s.q] for t in get0(fullq_by_service, s)) +
        sum(NI[t] * gi(t.p, t.r)        * inst.τ[s.q] for t in get0(inlayq_by_service, s)) +
        sum(NE[t] * gr(t.r)             * inst.τ[s.q] for t in get0(emptyq_by_service, s)))

    # Daily volume dispatched from a node (for hub variable handling)
    @expression(model, Vout[i in inst.hubs],
        sum(NF[t] * vf(t.r) for s in get0(amq_by_node_out, i)
                            for t in get0(fullq_by_service, s)) +
        sum(NI[t] * vf(t.r) for s in get0(amq_by_node_out, i)
                            for t in get0(inlayq_by_service, s)) +
        sum(NE[t] * ve(t.r) for s in get0(amq_by_node_out, i)
                            for t in get0(emptyq_by_service, s)))

    if one_to_one
        emptyq_by_arcr = groupby(t -> (t.i, t.j, t.r), EMPTYQ)
        inlayq_by_arcr = groupby(t -> (t.i, t.j, t.r), INLAYQ)
        for ((i, j, r), fw) in groupby(t -> (t.i, t.j, t.r), FULLQ)
            rv = get0(emptyq_by_arcr, (j, i, r))
            iv = get0(inlayq_by_arcr, (j, i, r))
            @constraint(model,
                sum(NE[t] for t in rv) + sum(NI[t] for t in iv)
                == sum(NF[t] for t in fw))
        end
    end
  
    # ──────────────────────────────────────────────────────────────
    # Service selection: at most one (mode, queue) per arc
    # ──────────────────────────────────────────────────────────────
    @constraint(model, [a in keys(amq_by_arc)],
        sum(Y[s] for s in amq_by_arc[a]) <= 1)

    # Flows may only use the selected service
    @constraint(model, [t in FULLQ],  XF[t] <= Y[svc(t)])
    @constraint(model, [t in INLAYQ], XI[t] <= Y[svc(t)])
    @constraint(model, [t in EMPTYQ], XE[t] <= Y[svc(t)])

    # Y active only if it carries something (Rule 3 upper part)
    @constraint(model, [s in ARCMODEQ],
        Y[s] <= sum(XF[t] for t in get0(fullq_by_service, s)) +
                sum(XI[t] for t in get0(inlayq_by_service, s)) +
                sum(XE[t] for t in get0(emptyq_by_service, s)))

    # ──────────────────────────────────────────────────────────────
    # Full-route activation + demand  (eq:full-activation, eq:demand)
    # Exactly one (m, q, r) serves each (i,j,p) with demand.
    # ──────────────────────────────────────────────────────────────
    demand_served = @constraint(model, [odp in keys(fullq_by_odp)],
        sum(XF[t] for t in fullq_by_odp[odp]) == 1)
    if verbose_names
        for (i, j, p) in keys(fullq_by_odp)
            set_name(demand_served[(i, j, p)],
                "demand_served[$(node_name(inst, i))→$(node_name(inst, j)), " *
                "$(prod_name(inst, p))]")
        end
    end

    for t in FULLQ
        d = arc(t.i, t.j).demand[t.p]
        c1 = @constraint(model, NF[t] * κ(t.p, t.r) >= d * XF[t])
        c2 = @constraint(model, NF[t] <= (d / κ(t.p, t.r)) * XF[t])  # tight big-M
        if verbose_names
            set_name(c1, "demand_lb[$(describe_full(inst, t))]")
            set_name(c2, "demand_ub[$(describe_full(inst, t))]")
        end
    end

    # Empties flow-binary link (tight-ish big-M: total demand-driven flow bound)
    Mempty = sum(d / minimum(κ(p, r) for r in keys(inst.rtis) if haskey(inst.compat, (p, r)))
                 for a in values(inst.arcs) for (p, d) in a.demand; init = 0.0)
    @constraint(model, [t in EMPTYQ], NE[t] <= Mempty * XE[t])

    # ──────────────────────────────────────────────────────────────
    # Inlay returns  (eq:inlay + XI/XF tie, aggregated over m,q)
    # ──────────────────────────────────────────────────────────────
    for (odpr, fw) in fullq_by_odpr
        (i, j, p, r) = odpr
        ric(p, r) == 0 && continue
        rv = get0(inlayq_by_odpr, (j, i, p, r))
        c1 = @constraint(model, sum(XI[t] for t in rv) == sum(XF[t] for t in fw))
        c2 = @constraint(model, sum(NI[t] for t in rv) >= ric(p, r) * sum(NF[t] for t in fw))
        if verbose_names
            lbl = "$(node_name(inst, i))→$(node_name(inst, j)), " *
                  "$(prod_name(inst, p)), $(rti_name(inst, r))"
            set_name(c1, "inlay_tie[$lbl]")
            set_name(c2, "inlay_flow[$lbl]")
        end
    end

    # ──────────────────────────────────────────────────────────────
    # Flow conservation per (node, RTI type)  (eq:conservation)
    # ──────────────────────────────────────────────────────────────
    nflow_out = groupby(t -> (t.i, t.r), FULLQ)
    nflow_in  = groupby(t -> (t.j, t.r), FULLQ)
    iflow_out = groupby(t -> (t.i, t.r), INLAYQ)
    iflow_in  = groupby(t -> (t.j, t.r), INLAYQ)
    for i in keys(inst.nodes), r in keys(inst.rtis)
        c = @constraint(model,
            sum(NF[t] for t in get0(nflow_out, (i, r))) +
            sum(NI[t] for t in get0(iflow_out, (i, r))) +
            sum(NE[t] for t in get0(emptyq_out, (i, r)))
            ==
            sum(NF[t] for t in get0(nflow_in, (i, r))) +
            sum(NI[t] for t in get0(iflow_in, (i, r))) +
            sum(NE[t] for t in get0(emptyq_in, (i, r))))
        verbose_names && set_name(c,
            "conservation[$(node_name(inst, i)), $(rti_name(inst, r))]")
    end

    # ──────────────────────────────────────────────────────────────
    # Shipment load bounds + chargeable weight
    # ──────────────────────────────────────────────────────────────
    for s in ARCMODEQ
        am = arc(s.i, s.j).modes[s.m]
        # c1 = @constraint(model, VS[s] >= am.min_volume * Y[s])
        # c2 = @constraint(model, VS[s] <= am.max_volume * Y[s])
        # c3 = @constraint(model, ws[s] >= am.min_weight * y[s])
        # c4 = @constraint(model, ws[s] <= am.max_weight * y[s])
        c3 = @constraint(model, CW[s] >= am.min_weight * Y[s])
        c4 = @constraint(model, CW[s] <= am.max_weight * Y[s])
        @constraint(model, CW[s] >= WS[s])
        @constraint(model, CW[s] >= inst.modes[s.m].ρ * VS[s])
        if verbose_names
            lbl = describe_service(inst, s)
            # set_name(c1, "vol_min[$lbl]"); set_name(c2, "vol_max[$lbl]")
            # set_name(c3, "wt_min[$lbl]");  set_name(c4, "wt_max[$lbl]")
        end
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
        @constraint(model, ZEout[k] <= sum(XE[t] for t in ts))
        @constraint(model, sum(XE[t] for t in ts) <= length(ts) * ZEout[k])
    end
    for (k, ts) in emptyq_in
        @constraint(model, ZEin[k] <= sum(XE[t] for t in ts))
        @constraint(model, sum(XE[t] for t in ts) <= length(ts) * ZEin[k])
    end

    if !one_to_one
      for i in keys(inst.nodes), r in keys(inst.rtis)
          is_hub(inst.nodes[i]) && continue
          zo = haskey(emptyq_out, (i, r)) ? ZEout[(i, r)] : 0
          zi = haskey(emptyq_in,  (i, r)) ? ZEin[(i, r)]  : 0
          (zo === 0 && zi === 0) && continue
          c = @constraint(model, zo + zi <= 1)
          verbose_names && set_name(c,
              "no_crossdock[$(node_name(inst, i)), $(rti_name(inst, r))]")
      end
    end

    # ──────────────────────────────────────────────────────────────
    # RTI pool  (eq:pool-*, purchase link) — all linear expressions
    # ──────────────────────────────────────────────────────────────
    @expression(model, TNCF[t in FULLQ],
        NF[t] * (inst.τ[t.q] + arc(t.i, t.j).modes[t.m].transit +
                 arc(t.i, t.j).ss[t.p]))
    @expression(model, TNCI[t in INLAYQ],
        NI[t] * (inst.τ[t.q] + arc(t.i, t.j).modes[t.m].transit))
    @expression(model, TNCE[t in EMPTYQ],
        NE[t] * (inst.τ[t.q] + arc(t.i, t.j).modes[t.m].transit))
    # safety empties: driven by full flows at the packing plant
    @expression(model, SEmp[t in FULLQ],
        NF[t] * get(arc(t.i, t.j).ser, t.p, 0.0))

    fullq_by_r  = groupby(t -> t.r, FULLQ)
    inlayq_by_r = groupby(t -> t.r, INLAYQ)
    emptyq_by_r = groupby(t -> t.r, EMPTYQ)
    @constraint(model, pool[r in keys(inst.rtis)],
        (1 + inst.α) * (
            sum(TNCF[t] + SEmp[t] for t in get0(fullq_by_r, r)) +
            sum(TNCI[t] for t in get0(inlayq_by_r, r)) +
            sum(TNCE[t] for t in get0(emptyq_by_r, r)))
        <= inst.rtis[r].pool + Prti[r])
    if verbose_names
        for r in keys(inst.rtis)
            set_name(pool[r], "pool[$(rti_name(inst, r))]")
        end
    end

    @expression(model, PC[r in keys(inst.rtis)],
        inst.rtis[r].purchase_cost * Prti[r] / inst.rtis[r].life)
    @expression(model, PE[r in keys(inst.rtis)],
        inst.rtis[r].purchase_emissions * Prti[r] / inst.rtis[r].life)

    # ──────────────────────────────────────────────────────────────
    # Transport cost & emissions — daily = per-shipment / τ_q (linear!)
    # ──────────────────────────────────────────────────────────────
    @expression(model, TC[s in ARCMODEQ],
        (CW[s] * arc(s.i, s.j).modes[s.m].variable_transport_cost +
         Y[s]  * arc(s.i, s.j).modes[s.m].fixed_transport_cost) / inst.τ[s.q])
    @expression(model, TE[s in ARCMODEQ],
        (WS[s] * arc(s.i, s.j).modes[s.m].variable_transport_emissions +
         Y[s]  * arc(s.i, s.j).modes[s.m].fixed_transport_emissions) / inst.τ[s.q])

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
    elseif objective == :both
        # Vector objective for MultiObjectiveAlgorithms.jl (ε-constraint etc.)
        @objective(model, Min, [ZC, ZE])
    else
        error("objective must be :cost, :emissions, or :both")
    end

    return model
end

# Bi-objective usage (MultiObjectiveAlgorithms.jl):
#   import MultiObjectiveAlgorithms as MOA
#   m = build_model(inst, () -> MOA.Optimizer(Gurobi.Optimizer); objective = :both)
#   set_attribute(m, MOA.Algorithm(), MOA.EpsilonConstraint())
#   set_attribute(m, MOA.SolutionLimit(), 15)
#   optimize!(m)
#
# Infeasibility debugging with readable names:
#   m = build_model(inst, Gurobi.Optimizer; verbose_names = true)
#   set_optimizer_attribute(m, "DualReductions", 0)
#   optimize!(m)
#   termination_status(m) == MOI.INFEASIBLE && print_iis(m)
