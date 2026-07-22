# ═══════════════════════════════════════════════════════════════════
#  analysis.jl ─ Post-processing of ε-constraint experiment results
#
#  Reads {name}_summary.json / {name}_solutions.json from --dir and
#  produces in --out:
#    table1_group_means.csv        — mean of ZC,ZE,TC,PC,HC,TE,PE,HE per
#                                    group: (a) mean over all Pareto pts,
#                                    (b) min-ZC point, (c) min-ZE point
#    table2_one_to_one.csv         — ZC/ZE base vs 1-to-1 + % reduction,
#                                    per group × optimized objective
#    net_{instance}_{minZC|minZE}.png — network drawings
#    pareto_{instance}.png         — Pareto front per instance (base vs 1to1)
#
#  Usage:
#    julia --project analysis.jl --dir results/ --out analysis/
#
#  Deps: JSON3, DataFrames, CSV, PrettyTables, Plots, Statistics
# ═══════════════════════════════════════════════════════════════════

using JSON3, DataFrames, CSV, PrettyTables, Plots, Statistics, Printf

gr()  # backend

const OBJ_FIELDS = ["ZC", "ZE", "TC", "PC", "HC", "TE", "PE", "HE"]

# ──────────────────────────────────────────────────────────────────
#  Loading & classification
# ──────────────────────────────────────────────────────────────────

"Infer size group from instance name."
function group_of(name::AbstractString)
    n = lowercase(name)
    occursin("small",  n) && return "small"
    occursin("medium", n) && return "medium"
    occursin("large",  n) && return "large"
    return "other"
end

is_one_to_one(name) = occursin("one_to_one", name)
base_name(name)     = replace(name, "one_to_one" => "")

"Load every *_summary.json into a vector of NamedTuples."
function load_summaries(dir)
    files = filter(f -> endswith(f, "_summary.json") && f != "batch_summary.json",
                   readdir(dir))
    out = []
    for f in files
        js = JSON3.read(read(joinpath(dir, f), String))
        name = String(js.instance)
        pts  = [Dict(String(k) => v for (k, v) in pairs(p)) for p in js.points]
        isempty(pts) && (@warn "No Pareto points in $name, skipping"; continue)
        push!(out, (name = name,
                    base = base_name(name),
                    group = group_of(name),
                    one_to_one = is_one_to_one(name),
                    points = pts))
    end
    return out
end

"Point minimizing ZC (⇒ solution when optimizing cost only)."
min_zc_point(pts) = argmin(p -> p["ZC"], pts)
"Point minimizing ZE (⇒ solution when optimizing emissions only)."
min_ze_point(pts) = argmin(p -> p["ZE"], pts)

# ──────────────────────────────────────────────────────────────────
#  Table 1: group means (all points / min-ZC / min-ZE)
# ──────────────────────────────────────────────────────────────────

function table1(summaries)
    rows = DataFrame(group = String[], selection = String[], n_instances = Int[];
                     [Symbol(f) => Float64[] for f in OBJ_FIELDS]...)

    for grp in ["small", "medium", "large", "other"]
        S = [s for s in summaries if s.group == grp && !s.one_to_one]
        isempty(S) && continue

        # (a) mean over ALL Pareto points of all instances in the group
        allpts = reduce(vcat, [s.points for s in S])
        push!(rows, (group = grp, selection = "mean_all_points",
                     n_instances = length(S),
                     (Symbol(f) => mean(get(p, f, NaN) for p in allpts)
                      for f in OBJ_FIELDS)...))

        # (b) optimizing ZC only → min-ZC point of each instance, averaged
        zc_pts = [min_zc_point(s.points) for s in S]
        push!(rows, (group = grp, selection = "opt_ZC",
                     n_instances = length(S),
                     (Symbol(f) => mean(get(p, f, NaN) for p in zc_pts)
                      for f in OBJ_FIELDS)...))

        # (c) optimizing ZE only → min-ZE point of each instance, averaged
        ze_pts = [min_ze_point(s.points) for s in S]
        push!(rows, (group = grp, selection = "opt_ZE",
                     n_instances = length(S),
                     (Symbol(f) => mean(get(p, f, NaN) for p in ze_pts)
                      for f in OBJ_FIELDS)...))
    end
    return rows
end

# ──────────────────────────────────────────────────────────────────
#  Table 2: base vs one-to-one, ZC/ZE + % reduction
# ──────────────────────────────────────────────────────────────────

function table2(summaries)
    # pair base instance ↔ its one_to_one variant
    base_map = Dict(s.base => s for s in summaries if !s.one_to_one)
    o2o_map  = Dict(s.base => s for s in summaries if s.one_to_one)

    per_inst = DataFrame(instance = String[], group = String[], opt = String[],
                         ZC_base = Float64[], ZE_base = Float64[],
                         ZC_1to1 = Float64[], ZE_1to1 = Float64[],
                         red_ZC_pct = Float64[], red_ZE_pct = Float64[])

    for (bn, sb) in base_map
        haskey(o2o_map, bn) || (@warn "No one_to_one twin for $bn"; continue)
        so = o2o_map[bn]
        for (opt, pick) in (("opt_ZC", min_zc_point), ("opt_ZE", min_ze_point))
            pb, po = pick(sb.points), pick(so.points)
            # reduction achieved by the flexible network vs the 1-to-1 baseline
            rzc = 100 * (po["ZC"] - pb["ZC"]) / po["ZC"]
            rze = 100 * (po["ZE"] - pb["ZE"]) / po["ZE"]
            push!(per_inst, (bn, sb.group, opt,
                             pb["ZC"], pb["ZE"], po["ZC"], po["ZE"], rzc, rze))
        end
    end

    # group-level aggregation
    grp = combine(groupby(per_inst, [:group, :opt]),
                  :ZC_base => mean => :ZC_base,
                  :ZE_base => mean => :ZE_base,
                  :ZC_1to1 => mean => :ZC_1to1,
                  :ZE_1to1 => mean => :ZE_1to1,
                  :red_ZC_pct => mean => :red_ZC_pct,
                  :red_ZE_pct => mean => :red_ZE_pct,
                  nrow => :n_instances)
    return per_inst, grp
end

# ──────────────────────────────────────────────────────────────────
#  Network drawings (from *_solutions.json)
# ──────────────────────────────────────────────────────────────────

const MODE_COLORS = Dict{String,Symbol}()  # filled lazily
const PALETTE = [:steelblue, :darkorange, :seagreen, :purple, :crimson, :teal]

mode_color(m) = get!(MODE_COLORS, m, PALETTE[mod1(length(MODE_COLORS)+1, length(PALETTE))])

"""
Draw one solution's network: nodes at (x,y), hubs highlighted, active
services as arrows colored by mode, line width ∝ shipment volume.
"""
function draw_network(det, sol; title = "")
    nodes = Dict(n.id => n for n in det.nodes)
    plt = plot(; title, legend = :outertopright, aspect_ratio = :equal,
               grid = false, framestyle = :box, size = (900, 700))

    open_hubs = Set(String.(sol["hubs_open"]))
    for svc in sol["services"]
        a, b = nodes[svc["from_id"]], nodes[svc["to_id"]]
        m = String(svc["mode"])
        vol = svc["shipment"]["volume_m3"]
        plot!(plt, [a.x, b.x], [a.y, b.y];
              lw = 0.5 + 3 * sqrt(vol) / 10, alpha = 0.75,
              color = mode_color(m), arrow = true,
              label = m ∈ keys(MODE_COLORS) && !any(s -> s[:label] == m, plt.series_list) ? m : "")
    end

    for n in det.nodes
        ishub = n.is_hub
        isopen = String(n.name) in open_hubs
        scatter!(plt, [n.x], [n.y];
                 marker = ishub ? :square : :circle,
                 ms = ishub ? (isopen ? 10 : 7) : 5,
                 color = ishub ? (isopen ? :red : :lightgray) : :black,
                 label = "")
        annotate!(plt, n.x, n.y, text(" " * String(n.name), 7, :left, :bottom))
    end
    return plt
end

"Draw min-ZC and min-ZE solutions for every base instance."
function draw_all_networks(dir, out)
    for f in filter(f -> endswith(f, "_solutions.json"), readdir(dir))
        det = JSON3.read(read(joinpath(dir, f), String))
        name = String(det.instance)
        sols = [Dict(String(k) => v for (k, v) in pairs(s)) for s in det.solutions]
        isempty(sols) && continue
        for (tag, pick) in (("minZC", min_zc_point), ("minZE", min_ze_point))
            s = pick(sols)
            plt = draw_network(det, s;
                title = "$name — $tag  (ZC=$(round(s["ZC"], digits=1)), ZE=$(round(s["ZE"], digits=1)))")
            savefig(plt, joinpath(out, "net_$(name)_$(tag).png"))
        end
    end
end

# ──────────────────────────────────────────────────────────────────
#  Pareto fronts
# ──────────────────────────────────────────────────────────────────

"One plot per base instance, overlaying base and 1-to-1 frontiers."
function draw_pareto_fronts(summaries, out)
    bases = unique(s.base for s in summaries)
    for bn in bases
        plt = plot(; title = "Pareto front — $bn",
                   xlabel = "Cost  ZC", ylabel = "Emissions  ZE",
                   legend = :topright, size = (750, 550))
        drew = false
        for s in summaries
            s.base == bn || continue
            pts = sort(s.points, by = p -> p["ZC"])
            xs = [p["ZC"] for p in pts]; ys = [p["ZE"] for p in pts]
            plot!(plt, xs, ys; marker = :circle, ms = 4,
                  ls = s.one_to_one ? :dash : :solid,
                  color = s.one_to_one ? :darkorange : :steelblue,
                  label = s.one_to_one ? "1-to-1" : "flexible")
            drew = true
        end
        drew && savefig(plt, joinpath(out, "pareto_$(bn).png"))
    end
end

# ──────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────

function main(; dir = "results", out = "analysis")
    isdir(out) || mkpath(out)
    summaries = load_summaries(dir)
    isempty(summaries) && error("No *_summary.json files found in $dir")
    @info "Loaded $(length(summaries)) summaries " *
          "($(count(s -> s.one_to_one, summaries)) one-to-one variants)"

    # Table 1
    t1 = table1(summaries)
    CSV.write(joinpath(out, "table1_group_means.csv"), t1)
    pretty_table(t1; formatters = ft_printf("%.1f", 4:11), crop = :none)

    # Table 2
    t2_inst, t2_grp = table2(summaries)
    CSV.write(joinpath(out, "table2_one_to_one_per_instance.csv"), t2_inst)
    CSV.write(joinpath(out, "table2_one_to_one_by_group.csv"), t2_grp)
    pretty_table(t2_grp; formatters = ft_printf("%.1f", 3:8), crop = :none)

    # Plots
    draw_pareto_fronts(summaries, out)
    draw_all_networks(dir, out)
    @info "Done. Tables and figures written to $out/"
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = Dict(ARGS[i] => ARGS[i+1] for i in 1:2:length(ARGS)-1)
    main(dir = get(args, "--dir", "results"),
         out = get(args, "--out", "analysis"))
end
