using CairoMakie

function plot_loss(train_loss, noiseless_loss, val_loss)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="MSE Loss", title="Losses", yscale=log10,
        xminorgridvisible=true, yminorgridvisible=true)
    lines!(ax, 1:length(train_loss), train_loss, color=:blue, label="Train Loss")
    lines!(ax, 1:length(val_loss), val_loss, color=:red, label="Validation Loss")
    lines!(ax, 1:length(noiseless_loss), noiseless_loss, color=:green, label="Noiseless Loss")
    axislegend(ax, position=:rt)
    display(fig)
    
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