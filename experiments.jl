# ═══════════════════════════════════════════════════════════════════
#  experiments.jl ─ Bi-objective (cost, emissions) experiment runner
#
#  Solves every instance JSON in a directory with the ε-constraint
#  method (MultiObjectiveAlgorithms.jl over Gurobi), logs to MLflow,
#  and saves per instance:
#    {name}_summary.json    — objective values per Pareto point only
#    {name}_solutions.json  — nodes + full per-arc detail per solution
#
#  Usage:
#    julia experiments.jl --dir instances/ --out results/ \
#          --mlflow http://localhost:5000 --points 15 --timelimit 3600
# ═══════════════════════════════════════════════════════════════════

using JuMP, Gurobi, JSON3, Printf, Dates
import MultiObjectiveAlgorithms as MOA
import MLFlowClient

isdefined(Main, :Instance)      || include(joinpath(@__DIR__, "structs.jl"))
isdefined(Main, :build_model)   || include(joinpath(@__DIR__, "model.jl"))
isdefined(Main, :load_instance) || include(joinpath(@__DIR__, "generator.jl"))

# One Gurobi environment for the whole batch (single license checkout)
const GRB_ENV = Gurobi.Env()

const TOL = 1e-6
r3(x) = round(x; digits = 3)

# ──────────────────────────────────────────────────────────────────
#  Solution extraction
# ──────────────────────────────────────────────────────────────────

"""
Per-arc/service detail for result k. One entry per ACTIVE service
(i,j,m,q): shipment volume/weight, chargeable weight, and the full /
inlay / empty flows riding it, with RTI and product counts.
"""
function extract_services(model, inst, k)
    full_by_svc  = Dict{Any,Vector{Any}}()
    inlay_by_svc = Dict{Any,Vector{Any}}()
    empty_by_svc = Dict{Any,Vector{Any}}()
    for t in model[:FULLQ];  push!(get!(full_by_svc,  svc(t), []), t); end
    for t in model[:INLAYQ]; push!(get!(inlay_by_svc, svc(t), []), t); end
    for t in model[:EMPTYQ]; push!(get!(empty_by_svc, svc(t), []), t); end

    services = Vector{Dict{String,Any}}()
    for s in model[:ARCMODEQ]
        value(model[:Y][s]; result = k) > 0.5 || continue
        τ = inst.τ[s.q]

        fulls = [t for t in get(full_by_svc, s, [])
                 if value(model[:NF][t]; result = k) > TOL]
        inlays = [t for t in get(inlay_by_svc, s, [])
                  if value(model[:NI][t]; result = k) > TOL]
        empties = [t for t in get(empty_by_svc, s, [])
                   if value(model[:NE][t]; result = k) > TOL]

        nf(t) = value(model[:NF][t]; result = k)
        ni(t) = value(model[:NI][t]; result = k)
        ne(t) = value(model[:NE][t]; result = k)

        # RTIs per shipment, by type and by state
        rti_ship = Dict{String,Dict{String,Float64}}()
        addr!(r, state, v) = begin
            d = get!(rti_ship, inst.rtis[r].name,
                     Dict("full" => 0.0, "inlay" => 0.0, "empty" => 0.0))
            d[state] = r3(d[state] + v * τ)
        end
        foreach(t -> addr!(t.r, "full",  nf(t)), fulls)
        foreach(t -> addr!(t.r, "inlay", ni(t)), inlays)
        foreach(t -> addr!(t.r, "empty", ne(t)), empties)

        # Products per shipment (units = RTIs × κ)
        prod_ship = Dict{String,Float64}()
        for t in fulls
            key = inst.products[t.p].name
            prod_ship[key] = r3(get(prod_ship, key, 0.0) +
                                nf(t) * inst.compat[(t.p, t.r)].κ * τ)
        end

        am = inst.arcs[(s.i, s.j)].modes[s.m]
        push!(services, Dict{String,Any}(
            "from"      => inst.nodes[s.i].name,
            "to"        => inst.nodes[s.j].name,
            "from_id"   => s.i, "to_id" => s.j,
            "mode"      => inst.modes[s.m].name,
            "frequency_days" => τ,
            "transit_days"   => am.transit,
            "shipment" => Dict(
                "volume_m3"  => r3(value(model[:VS][s]; result = k)),
                "volume_cap" => am.max_volume,
                "weight_t"   => r3(value(model[:WS][s]; result = k)),
                "weight_cap" => am.max_weight,
                "chargeable_weight_t" => r3(value(model[:CW][s]; result = k)),
                "rtis_total" => r3(sum(nf(t) for t in fulls;  init = 0.0) * τ +
                                   sum(ni(t) for t in inlays; init = 0.0) * τ +
                                   sum(ne(t) for t in empties; init = 0.0) * τ)),
            "rtis_per_shipment"     => rti_ship,
            "products_per_shipment" => prod_ship,
            "daily" => Dict(
                "full_rtis"  => r3(sum(nf(t) for t in fulls;   init = 0.0)),
                "inlay_rtis" => r3(sum(ni(t) for t in inlays;  init = 0.0)),
                "empty_rtis" => r3(sum(ne(t) for t in empties; init = 0.0)),
                "cost"      => r3(value(model[:TC][s]; result = k)),
                "emissions" => r3(value(model[:TE][s]; result = k))),
            "full_flows" => [Dict(
                "product" => inst.products[t.p].name,
                "rti"     => inst.rtis[t.r].name,
                "rtis_per_day"  => r3(nf(t)),
                "parts_per_day" => r3(nf(t) * inst.compat[(t.p, t.r)].κ))
                for t in fulls],
        ))
    end
    sort!(services, by = d -> (d["from_id"], d["to_id"], d["mode"]))
    return services
end

"One Pareto point: objectives + network decisions + per-arc detail."
function extract_solution(model, inst, k)
    zc = value(model[:ZC]; result = k)
    ze = value(model[:ZE]; result = k)
    Dict{String,Any}(
        "ZC" => r3(zc), "ZE" => r3(ze),
        "purchases" => Dict(inst.rtis[r].name =>
                            round(Int, value(model[:Prti][r]; result = k))
                            for r in keys(inst.rtis)
                            if value(model[:Prti][r]; result = k) > 0.5),
        "hubs_open" => [inst.nodes[i].name for i in inst.hubs
                        if value(model[:XH][i]; result = k) > 0.5],
        "services"  => extract_services(model, inst, k),
    )
end

# ──────────────────────────────────────────────────────────────────
#  Solve one instance
# ──────────────────────────────────────────────────────────────────
function solve_instance(inst::Instance;
                        n_points::Int = 15,
                        time_limit::Float64 = 3600.0,
                        mip_gap::Float64 = 1e-4,
                        threads::Int = 0)

    gurobi = optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV),
        "MIPGap"     => mip_gap,
        "Threads"    => threads,
        "OutputFlag" => 0,
        "TimeLimit"  => max(60.0, time_limit / n_points))  # per-subproblem cap

    model = build_model(inst, () -> MOA.Optimizer(gurobi); objective = :both)
    set_attribute(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    set_attribute(model, MOA.SolutionLimit(), n_points)
    set_time_limit_sec(model, time_limit)
    set_silent(model)

    t = @elapsed optimize!(model)
    nres = result_count(model)

    solutions = [extract_solution(model, inst, k) for k in 1:nres]
    sort!(solutions, by = s -> s["ZC"])

    return Dict{String,Any}(
        "status"       => string(termination_status(model)),
        "n_points"     => nres,
        "solve_time_s" => r3(t),
        "solutions"    => solutions,
        "model_stats"  => Dict(
            "n_vars"    => num_variables(model),
            "n_bin"     => num_constraints(model, VariableRef, MOI.ZeroOne),
            "n_constrs" => sum(num_constraints(model, F, S)
                               for (F, S) in list_of_constraint_types(model)
                               if F != VariableRef; init = 0)),
    )
end

# ──────────────────────────────────────────────────────────────────
#  Output files: one summary + one full-detail file per instance
# ──────────────────────────────────────────────────────────────────

"Compact summary: run info + objective values per Pareto point only."
function summary_payload(name, inst, res)
    Dict{String,Any}(
        "instance"     => name,
        "timestamp"    => string(now()),
        "status"       => res["status"],
        "solve_time_s" => res["solve_time_s"],
        "n_points"     => res["n_points"],
        "model_stats"  => res["model_stats"],
        "points" => [Dict(
            "ZC" => s["ZC"], "ZE" => s["ZE"],
            "n_services"      => length(s["services"]),
            "hubs_open"       => length(s["hubs_open"]),
            "rtis_purchased"  => sum(values(s["purchases"]); init = 0))
            for s in res["solutions"]],
    )
end

"Full detail: instance nodes + every solution with per-arc breakdown."
function detail_payload(name, inst, res)
    Dict{String,Any}(
        "instance"  => name,
        "timestamp" => string(now()),
        "nodes" => [Dict("id" => n.id, "name" => n.name, "zone" => n.zone,
                         "x" => n.x, "y" => n.y, "is_hub" => is_hub(n))
                    for n in sort!(collect(values(inst.nodes)); by = n -> n.id)],
        "rtis" => [Dict("id" => r.id, "name" => r.name,
                        "v_full" => r.v_full, "v_empty" => r.v_empty,
                        "weight_t" => r.weight)
                   for r in sort!(collect(values(inst.rtis)); by = r -> r.id)],
        "solutions" => res["solutions"],
    )
end

function save_results(out, name, inst, res)
    sum_path = joinpath(out, "$(name)_summary.json")
    det_path = joinpath(out, "$(name)_solutions.json")
    open(sum_path, "w") do io; JSON3.pretty(io, summary_payload(name, inst, res)); end
    open(det_path, "w") do io; JSON3.pretty(io, detail_payload(name, inst, res)); end
    return sum_path, det_path
end

# ──────────────────────────────────────────────────────────────────
#  MLflow helpers (degrade gracefully if the server is down)
# ──────────────────────────────────────────────────────────────────
struct Tracker
    mlf::Union{Nothing,MLFlowClient.MLFlow}
    exp_id::Union{Nothing,String}
end

function init_tracker(uri::Union{Nothing,String}, exp_name::String)
    uri === nothing && return Tracker(nothing, nothing)
    try
        mlf = MLFlowClient.MLFlow(uri)
        exp = MLFlowClient.getorcreateexperiment(mlf, exp_name)
        return Tracker(mlf, exp.experiment_id)
    catch e
        @warn "MLflow unreachable at $uri — logging to disk only" exception = e
        return Tracker(nothing, nothing)
    end
end

function log_run!(tr::Tracker, name::String, params::Dict, res::Dict,
                  artifacts::Vector{String})
    tr.mlf === nothing && return
    try
        run = MLFlowClient.createrun(tr.mlf, tr.exp_id; run_name = name)
        for (k, v) in params
            MLFlowClient.logparam(tr.mlf, run, string(k), string(v))
        end
        MLFlowClient.logparam(tr.mlf, run, "status", res["status"])
        res["solve_time_s"] !== nothing &&
            MLFlowClient.logmetric(tr.mlf, run, "solve_time_s", res["solve_time_s"])
        MLFlowClient.logmetric(tr.mlf, run, "n_pareto_points", res["n_points"])
        zcs = [s["ZC"] for s in res["solutions"]]
        zes = [s["ZE"] for s in res["solutions"]]
        if !isempty(zcs)
            MLFlowClient.logmetric(tr.mlf, run, "zc_min", minimum(zcs))
            MLFlowClient.logmetric(tr.mlf, run, "zc_max", maximum(zcs))
            MLFlowClient.logmetric(tr.mlf, run, "ze_min", minimum(zes))
            MLFlowClient.logmetric(tr.mlf, run, "ze_max", maximum(zes))
        end
        for (k, s) in enumerate(res["solutions"])
            MLFlowClient.logmetric(tr.mlf, run, "pareto_ZC", s["ZC"]; step = k)
            MLFlowClient.logmetric(tr.mlf, run, "pareto_ZE", s["ZE"]; step = k)
        end
        for a in artifacts
            MLFlowClient.logartifact(tr.mlf, run, a)
        end
        MLFlowClient.updaterun(tr.mlf, run, "FINISHED")
    catch e
        @warn "MLflow logging failed for $name" exception = e
    end
end

# ──────────────────────────────────────────────────────────────────
#  Batch driver
# ──────────────────────────────────────────────────────────────────
function run_experiments(; dir::String, out::String,
                         mlflow_uri::Union{Nothing,String} = nothing,
                         exp_name::String = "SND-RTI",
                         n_points::Int = 15,
                         time_limit::Float64 = 3600.0,
                         mip_gap::Float64 = 1e-4)

    isdir(out) || mkpath(out)
    tracker = init_tracker(mlflow_uri, exp_name)
    files = sort(filter(f -> endswith(f, ".json"), readdir(dir; join = true)))
    isempty(files) && error("No instance .json files found in $dir")

    batch = Vector{Dict{String,Any}}()
    for (n, f) in enumerate(files)
        name = splitext(basename(f))[1]
        @info "[$n/$(length(files))] Solving $name"
        inst = load_instance(f)

        res = try
            solve_instance(inst; n_points, time_limit, mip_gap)
        catch e
            @error "Solve failed for $name" exception = (e, catch_backtrace())
            Dict{String,Any}("status" => "ERROR: $(sprint(showerror, e))",
                             "n_points" => 0, "solve_time_s" => nothing,
                             "solutions" => Dict{String,Any}[],
                             "model_stats" => Dict("n_vars" => 0, "n_bin" => 0,
                                                   "n_constrs" => 0))
        end

        sum_path, det_path = save_results(out, name, inst, res)

        params = Dict("instance" => name,
                      "n_points_requested" => n_points,
                      "time_limit" => time_limit, "mip_gap" => mip_gap,
                      "n_nodes" => length(inst.nodes), "n_arcs" => length(inst.arcs),
                      "n_rtis" => length(inst.rtis),
                      "n_products" => length(inst.products))
        log_run!(tracker, name, params, res, [sum_path, det_path])

        push!(batch, Dict("instance" => name, "status" => res["status"],
                          "time" => res["solve_time_s"],
                          "points" => res["n_points"]))
        t_str = res["solve_time_s"] === nothing ? "—" :
                @sprintf("%.1f s", res["solve_time_s"])
        @info "  → $(res["status"]) | $t_str | $(res["n_points"]) Pareto points"
    end

    open(joinpath(out, "batch_summary.json"), "w") do io
        JSON3.pretty(io, batch)
    end
    @info "Done. Results in $out"
    return batch
end

# ──────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    args = Dict(ARGS[i] => ARGS[i+1] for i in 1:2:length(ARGS)-1)
    run_experiments(
        dir        = get(args, "--dir", "instances"),
        out        = get(args, "--out", "results"),
        mlflow_uri = get(args, "--mlflow", nothing),
        exp_name   = get(args, "--exp", "SND-RTI"),
        n_points   = parse(Int, get(args, "--points", "15")),
        time_limit = parse(Float64, get(args, "--timelimit", "3600")),
        mip_gap    = parse(Float64, get(args, "--gap", "1e-4")),
    )
end
