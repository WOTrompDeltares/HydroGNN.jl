cd(@__DIR__)
using Pkg
Pkg.activate(".")

using TOML, CairoMakie

# ─── Load results ─────────────────────────────────────────────────────────────
log_file = "models/hparam_search/results.toml"
results  = TOML.parsefile(log_file)

# ─── Detect format ────────────────────────────────────────────────────────────
first_entry = first(values(results))
has_skip    = haskey(first_entry, "skip_connections")
has_noise   = haskey(first_entry, "noise_scale")

# ─── Extract unique axis values ───────────────────────────────────────────────
dh_vals    = sort(unique(r["dhidden"] for r in values(results)))
nh_vals    = sort(unique(r["nhidden"] for r in values(results)))
skip_vals  = has_skip  ? sort(unique(r["skip_connections"] for r in values(results))) : [nothing]
noise_vals = has_noise ? sort(unique(r["noise_scale"]      for r in values(results))) : [nothing]

# ─── Helper: build dh × nh matrix ─────────────────────────────────────────────
function build_matrix(metric, skip, noise)
    M = fill(NaN, length(dh_vals), length(nh_vals))
    for (i, dh) in enumerate(dh_vals), (j, nh) in enumerate(nh_vals)
        for r in values(results)
            skip_ok  = isnothing(skip)  || r["skip_connections"] == skip
            noise_ok = isnothing(noise) || r["noise_scale"]      == noise
            if r["dhidden"] == dh && r["nhidden"] == nh && skip_ok && noise_ok
                M[i, j] = r[metric]
                break
            end
        end
    end
    return M
end

# ─── Plot ─────────────────────────────────────────────────────────────────────
for (metric, label, colormap) in [
        ("best_val_loss", "Best validation loss", :viridis),
        ("train_seconds", "Training time (s)",    :plasma),
    ]

    ncols = length(noise_vals)
    nrows = length(skip_vals)

    fig = Figure(size=(300 * ncols + 120, 280 * nrows + 80))
    title_str = has_skip || has_noise ?
        "$label  (rows = skip, cols = noise scale)" : label
    Label(fig[0, :], title_str; fontsize=16, font=:bold)

    mats = [build_matrix(metric, s, n) for s in skip_vals, n in noise_vals]
    cmin = minimum(x for M in mats for x in M if !isnan(x))
    cmax = maximum(x for M in mats for x in M if !isnan(x))

    for (ri, skip) in enumerate(skip_vals), (ci, noise) in enumerate(noise_vals)
        M = mats[ri, ci]
        panel_title = if has_skip && has_noise
            "skip=$skip  noise=$noise"
        elseif has_skip
            "skip=$skip"
        elseif has_noise
            "noise=$noise"
        else
            label
        end
        ax = Axis(fig[ri, ci];
                  title  = panel_title,
                  xlabel = "dhidden",
                  ylabel = "nhidden",
                  xticks = (1:length(dh_vals), string.(dh_vals)),
                  yticks = (1:length(nh_vals), string.(nh_vals)))
        hm = heatmap!(ax, 1:length(dh_vals), 1:length(nh_vals), M;
                      colormap=colormap, colorrange=(cmin, cmax), nan_color=:lightgray)

        for (i, _) in enumerate(dh_vals), (j, _) in enumerate(nh_vals)
            isnan(M[i, j]) && continue
            text!(ax, i, j;
                  text     = string(round(M[i, j]; sigdigits=3)),
                  align    = (:center, :center),
                  fontsize = 9,
                  color    = (M[i, j] - cmin) / (cmax - cmin) > 0.6 ? :white : :black)
        end

        if ri == 1 && ci == ncols
            Colorbar(fig[ri, ncols + 1], hm; label=label)
        end
    end

    out = "models/hparam_search/summary_$(metric).png"
    save(out, fig)
    println("Saved: $out")
end
