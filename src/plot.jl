using CairoMakie

function plot_loss(train_loss, noiseless_loss, val_loss, fn)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="MSE Loss", title="Losses", yscale=log10,
        xminorgridvisible=true, yminorgridvisible=true)
    lines!(ax, 1:length(train_loss), train_loss, color=:blue, label="Train Loss")
    lines!(ax, 1:length(val_loss), val_loss, color=:red, label="Validation Loss")
    lines!(ax, 1:length(noiseless_loss), noiseless_loss, color=:green, label="Noiseless Loss")
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


    lines!(ax1, h_gt, color=:blue, label="Ground Truth")
    lines!(ax1, h_pred, color=:red, label="Prediction")
    axislegend(ax1, position=:rt)
    ylims!(ax1, -1,2.5)
    lines!(ax2, v_gt, color=:blue, label="Ground Truth")
    lines!(ax2, v_pred, color=:red, label="Prediction")
    axislegend(ax2, position=:rt)
    ylims!(ax2, -1,1)
    lines!(ax3, 1:length(graph_gt), losses, color=:green, label="MSE Loss")
    scatter!(ax3, step, mse_pt, color=:green)
    ylims!(ax3, 0,  minimum([maximum(filter(!isnan, losses)), 3]))

    record(fig, fn, 1:length(graph_gt)) do i
        step[] = i
    end


end