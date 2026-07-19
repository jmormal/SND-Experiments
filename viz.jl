# ═══════════════════════════════════════════════════════════════════
#  viz.jl ─ Instance visualisation for the SND-RTI problem
#
#  Figure 1 (network):  nodes (OEM / supplier / hub) + three arc
#  layers you can toggle via the legend:
#    • full routes   (arcs with product demand)
#    • empty routes  (arcs with empties_ok ≠ ∅)
#    • inlay routes  (reverse arcs implied by ric > 0)
#  Figure 2: product × RTI compatibility heatmap (κ, with ric flag).
#
#  Usage:
#    include("structs.jl"); include("generator.jl"); include("viz.jl")
#    inst = generate_instance(:small, 42)         # or load_instance(path)
#    visualize(inst; out = "instance_viz")        # → instance_viz_network.html
#                                                 #   instance_viz_compat.html
#  CLI:
#    julia viz.jl --file SND_RTI_small_42.json --out viz/small_42
#
#  Requires: Pkg.add("PlotlyJS")
# ═══════════════════════════════════════════════════════════════════

using PlotlyJS
isdefined(Main, :Instance)      || include(joinpath(@__DIR__, "structs.jl"))
isdefined(Main, :load_instance) || include(joinpath(@__DIR__, "generator.jl"))

# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────

# Slight quadratic bend so (i,j) and (j,i) don't overlap; also keeps
# parallel layers (full/empty/inlay on the same arc) visually separate.
function _bent(x1, y1, x2, y2; bend = 0.06, n = 20)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    px, py = mx - bend * dy, my + bend * dx      # midpoint pushed sideways
    ts = range(0, 1; length = n)
    xs = [(1 - t)^2 * x1 + 2t * (1 - t) * px + t^2 * x2 for t in ts]
    ys = [(1 - t)^2 * y1 + 2t * (1 - t) * py + t^2 * y2 for t in ts]
    return xs, ys
end

# One scatter trace per arc-layer with NaN separators (fast) is not
# hover-friendly per arc, so we emit one trace per arc but share a
# legendgroup — the legend shows a single toggle per layer.
function _arc_traces(inst, arc_list, hovers; name, color, dash, bend, width = 1.6)
    traces = GenericTrace[]
    first_trace = true
    for (k, (i, j)) in enumerate(arc_list)
        ni, nj = inst.nodes[i], inst.nodes[j]
        xs, ys = _bent(ni.x, ni.y, nj.x, nj.y; bend)
        push!(traces, scatter(
            x = xs, y = ys, mode = "lines",
            line = attr(color = color, width = width, dash = dash),
            opacity = 0.75,
            name = name, legendgroup = name, showlegend = first_trace,
            hoverinfo = "text", text = hovers[k], hoverlabel = attr(namelength = -1)))
        first_trace = false
    end
    return traces
end

# ──────────────────────────────────────────────────────────────────
#  Figure 1: network
# ──────────────────────────────────────────────────────────────────
function network_figure(inst::Instance)
    hubs = Set(inst.hubs)

    # --- classify arcs straight from the instance data --------------
    full_arcs  = Tuple{Int,Int}[]
    full_hover = String[]
    empty_arcs  = Tuple{Int,Int}[]
    empty_hover = String[]
    for ((i, j), a) in sort(collect(inst.arcs); by = first)
        if !isempty(a.demand)
            push!(full_arcs, (i, j))
            lines = ["$(inst.products[p].name): $(round(d; digits=1))/day"
                     for (p, d) in sort(collect(a.demand); by = first)]
            push!(full_hover,
                  "<b>FULL</b> $(inst.nodes[i].name) → $(inst.nodes[j].name)<br>" *
                  join(lines, "<br>"))
        end
        if !isempty(a.empties_ok)
            push!(empty_arcs, (i, j))
            rn = join((inst.rtis[r].name for r in sort!(collect(a.empties_ok))), ", ")
            push!(empty_hover,
                  "<b>EMPTY</b> $(inst.nodes[i].name) → $(inst.nodes[j].name)<br>RTIs: $rn")
        end
    end

    # Inlay routes: reverse (j,i) of any full route whose (p,r) has ric > 0.
    inlay_set = Dict{Tuple{Int,Int},Set{String}}()
    for ((i, j), a) in inst.arcs, (p, _) in a.demand, r in keys(inst.rtis)
        c = get(inst.compat, (p, r), nothing)
        (c === nothing || c.ric == 0) && continue
        push!(get!(inlay_set, (j, i), Set{String}()),
              "$(inst.products[p].name)/$(inst.rtis[r].name) (ric=$(c.ric))")
    end
    inlay_arcs  = sort!(collect(keys(inlay_set)))
    inlay_hover = ["<b>INLAY</b> $(inst.nodes[i].name) → $(inst.nodes[j].name)<br>" *
                   join(sort!(collect(inlay_set[(i, j)])), "<br>")
                   for (i, j) in inlay_arcs]

    traces = GenericTrace[]
    append!(traces, _arc_traces(inst, full_arcs, full_hover;
        name = "Full routes", color = "#2166ac", dash = "solid", bend = 0.05, width = 2.0))
    append!(traces, _arc_traces(inst, empty_arcs, empty_hover;
        name = "Empty routes", color = "#1a9850", dash = "dot", bend = 0.10))
    append!(traces, _arc_traces(inst, inlay_arcs, inlay_hover;
        name = "Inlay returns", color = "#d73027", dash = "dash", bend = 0.15, width = 1.3))

    # --- nodes: three classes, drawn last (on top) -------------------
    for (label, sym, color, ids) in (
            ("OEM plant", "square", "#4a1486",
             [n.id for n in values(inst.nodes) if startswith(n.name, "OEM")]),
            ("Supplier", "circle", "#807dba",
             [n.id for n in values(inst.nodes) if startswith(n.name, "supplier")]),
            ("Hub", "diamond", "#e08214",
             collect(hubs)))
        isempty(ids) && continue
        push!(traces, scatter(
            x = [inst.nodes[i].x for i in ids],
            y = [inst.nodes[i].y for i in ids],
            mode = "markers+text",
            marker = attr(symbol = sym, size = 14, color = color,
                          line = attr(width = 1, color = "white")),
            text = [inst.nodes[i].name for i in ids],
            textposition = "top center", textfont = attr(size = 9),
            name = label, legendgroup = label,
            hoverinfo = "text",
            hovertext = ["$(inst.nodes[i].name) (zone $(inst.nodes[i].zone))"
                         for i in ids]))
    end

    layout = Layout(
        title = "SND-RTI network — full / empty / inlay route layers " *
                "(click legend entries to toggle)",
        xaxis = attr(title = "x (km)", zeroline = false, scaleanchor = "y"),
        yaxis = attr(title = "y (km)", zeroline = false),
        hovermode = "closest",
        legend = attr(orientation = "h", y = -0.12),
        plot_bgcolor = "white", width = 1000, height = 850)
    return plot(traces, layout)
end

# ──────────────────────────────────────────────────────────────────
#  Figure 2: product × RTI compatibility heatmap
# ──────────────────────────────────────────────────────────────────
function compat_figure(inst::Instance)
    pids = sort!(collect(keys(inst.products)))
    rids = sort!(collect(keys(inst.rtis)))
    z    = [haskey(inst.compat, (p, r)) ? inst.compat[(p, r)].κ : NaN
            for p in pids, r in rids]
    txt  = [begin
                c = get(inst.compat, (p, r), nothing)
                c === nothing ? "" :
                    (c.ric > 0 ? "κ=$(Int(c.κ))<br>ric=$(c.ric)" : "κ=$(Int(c.κ))")
            end for p in pids, r in rids]

    hm = heatmap(
        z = z, text = txt, texttemplate = "%{text}",
        x = [inst.rtis[r].name for r in rids],
        y = [inst.products[p].name for p in pids],
        colorscale = "Blues", colorbar = attr(title = "κ (parts/RTI)"),
        hovertemplate = "%{y} in %{x}<br>%{text}<extra></extra>",
        xgap = 2, ygap = 2)

    layout = Layout(
        title = "Product ↔ RTI compatibility (blank = incompatible; " *
                "ric shown where inlay return required)",
        xaxis = attr(title = "RTI type", tickangle = -30),
        yaxis = attr(title = "Product", autorange = "reversed"),
        width = 250 + 130 * length(rids),
        height = 200 + 32 * length(pids))
    return plot(hm, layout)
end

# ──────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────
function visualize(inst::Instance; out::String = "instance_viz", open_browser::Bool = false)
    p1 = network_figure(inst)
    p2 = compat_figure(inst)
    savefig(p1, "$(out)_network.html")
    savefig(p2, "$(out)_compat.html")
    println("Saved $(out)_network.html and $(out)_compat.html")
    if open_browser
        display(p1); display(p2)
    end
    return p1, p2
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = Dict(ARGS[i] => ARGS[i+1] for i in 1:2:length(ARGS)-1)
    file = get(args, "--file", nothing)
    file === nothing && error("Usage: julia viz.jl --file instance.json [--out prefix]")
    inst = load_instance(file)
    visualize(inst; out = get(args, "--out", splitext(file)[1]))
end
