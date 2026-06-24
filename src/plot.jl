using CairoMakie

function plot_loss(train_loss, noiseless_loss, val_loss, fn; val_loss_1step=nothing)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="MSE Loss", title="Losses", yscale=log10,
        xminorgridvisible=true, yminorgridvisible=true)
    lines!(ax, 1:length(train_loss),     train_loss,     color=:blue,   label="Train Loss")
    lines!(ax, 1:length(val_loss),       val_loss,       color=:red,    label="Val Loss (strategy)")
    if val_loss_1step !== nothing
        lines!(ax, 1:length(val_loss_1step), val_loss_1step, color=:orange, linestyle=:dash, label="Val Loss (1-step)")
    end
    lines!(ax, 1:length(noiseless_loss), noiseless_loss, color=:green,  label="Noiseless Loss")
    axislegend(ax, position=:rt)
    display(fig)
    save(fn, fig)
end


function movie_graphs(graphs, fn)
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel="Mesh Position", ylabel="Water Level", xminorgridvisible=true, yminorgridvisible=true)
    ax2 = Axis(fig[1, 2], xlabel="Mesh Position", ylabel="Velocity", xminorgridvisible=true, yminorgridvisible=true)

    step = Observable(1)

    h = @lift(graphs[$step].ndata.dynamic[1,:])
    v = @lift(graphs[$step].ndata.dynamic[2,:])

    lines!(ax1, h, color=:blue)
    scatter!(ax1, h, color=:blue)
    lines!(ax2, v, color=:red)
    scatter!(ax2, v, color=:red)

    record(fig, fn, 1:length(graphs)) do i
        step[] = i
    end


end

# Compute y-limits from GT + predicted values up to the first blowup step.
# A blowup is detected when any predicted value becomes non-finite or exceeds
# blowup_factor times the GT range beyond the GT extremes.
function _stable_ylims(gt_vecs, pred_vecs; margin=0.1, blowup_factor=3.0)
    gt_flat   = vcat(gt_vecs...)
    finite_gt = filter(isfinite, gt_flat)
    gt_lo, gt_hi = minimum(finite_gt), maximum(finite_gt)
    gt_span   = max(gt_hi - gt_lo, 1f-6)
    threshold = max(abs(gt_lo), abs(gt_hi)) + blowup_factor * gt_span

    stable_up_to = length(pred_vecs)
    for (i, pv) in enumerate(pred_vecs)
        if any(!isfinite, pv) || maximum(abs, pv) > threshold
            stable_up_to = i - 1
            break
        end
    end

    if stable_up_to == 0
        lo_raw, hi_raw = gt_lo, gt_hi
    else
        pred_stable    = vcat(pred_vecs[1:stable_up_to]...)
        all_vals       = vcat(finite_gt, filter(isfinite, pred_stable))
        lo_raw, hi_raw = minimum(all_vals), maximum(all_vals)
    end
    span = max(hi_raw - lo_raw, 1f-6)
    return lo_raw - margin * span, hi_raw + margin * span
end

function movie_graphs_comp(graph_gt, graph_pred, fn)

    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel="Mesh Position", ylabel="Water Level", xminorgridvisible=true, yminorgridvisible=true)
    ax2 = Axis(fig[1, 2], xlabel="Mesh Position", ylabel="Velocity", xminorgridvisible=true, yminorgridvisible=true)
    ax3 = Axis(fig[1,3], xlabel="Time step", ylabel="MSE Loss", title="MSE Loss", xminorgridvisible=true, yminorgridvisible=true)

    step = Observable(1)

    h_gt = @lift(graph_gt[$step].ndata.dynamic[1,:])
    v_gt = @lift(graph_gt[$step].ndata.dynamic[2,:])
    h_pred = @lift(graph_pred[$step].ndata.dynamic[1,:])
    v_pred = @lift(graph_pred[$step].ndata.dynamic[2,:])

    # mse_loss = @lift(mean(($h_gt .- $h_pred).^2 .+ ($v_gt .- $v_pred).^2))
    # mse_loss = mean((graph_gt.ndata.dynamic .- graph_pred.ndata.dynamic) .^ 2, dims=1)
    mse_loss = (x,y) -> mean((x.ndata.dynamic .- y.ndata.dynamic).^2)
    losses = (mse_loss.(graph_gt, graph_pred)) |> vec
    mse_pt = @lift(losses[$step])


    h_gt_vecs   = [graph_gt[i].ndata.dynamic[1,:]   for i in 1:length(graph_gt)]
    v_gt_vecs   = [graph_gt[i].ndata.dynamic[2,:]   for i in 1:length(graph_gt)]
    h_pred_vecs = [graph_pred[i].ndata.dynamic[1,:] for i in 1:length(graph_pred)]
    v_pred_vecs = [graph_pred[i].ndata.dynamic[2,:] for i in 1:length(graph_pred)]
    h_lo, h_hi  = _stable_ylims(h_gt_vecs, h_pred_vecs)
    v_lo, v_hi  = _stable_ylims(v_gt_vecs, v_pred_vecs)

    lines!(ax1, h_gt, color=:blue, label="Ground Truth")
    lines!(ax1, h_pred, color=:red, label="Prediction")
    axislegend(ax1, position=:rt)
    ylims!(ax1, h_lo, h_hi)
    lines!(ax2, v_gt, color=:blue, label="Ground Truth")
    lines!(ax2, v_pred, color=:red, label="Prediction")
    axislegend(ax2, position=:rt)
    ylims!(ax2, v_lo, v_hi)
    lines!(ax3, 1:length(graph_gt), losses, color=:green, label="MSE Loss")
    scatter!(ax3, step, mse_pt, color=:green)
    ylims!(ax3, 0,  minimum([maximum(filter(!isnan, losses)), 3]))

    record(fig, fn, 1:length(graph_gt)) do i
        step[] = i
    end


end