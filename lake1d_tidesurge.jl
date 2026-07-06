cd(@__DIR__)
using Pkg
Pkg.activate(".")

using OrdinaryDiffEq
using ComponentArrays
using Interpolations
using Random
using StatsBase
using JLD2
using CairoMakie

rng = MersenneTwister(1234)


################################
# Helper structs & functions ODE
################################

struct Wave1DSurge_cpu
    g::Float64 # Gravity
    D::Vector{Float64} # Depth variable in space
    L::Float64 # Length
    W::Float64 #  Width
    dx::Float64 # Spatial step
    nx::Int64 # Number of spatial points
    rho::Float64 # Density
    C::Float64 # Chezy friction factor
    tau::Function # Function of time t: Float64 -> Float64; wind stress as funcion of time
    tide::Function # Function of time t: Float64 -> Float64; tide level as function of time
    tide_derivative::Function # Function of time t: Float64 -> Float64; derivative of tide level as function of time
    open_right_bc::Bool # true = Sommerfeld radiation BC (open); false = closed wall (u=0)
    # q_left::Function # Inflow at left boundary
    # q_right::Function # Outflow at right boundary
end

function depth_profile(grid_u, D_min, D_max, D_edges, D_widths)
    D = [ D_min +
        (D_max-D_min) * 0.5 * (1 + tanh((x-D_edges[1]) / D_widths[1])) + 
        (D_min-D_max) * 0.5 * (1 + tanh((x-D_edges[2]) / D_widths[2]))
        for x in grid_u
    ]
    return D
end

function initial_state_bump(f::Wave1DSurge_cpu, h0=1.0, w=0.05, c=0.5, u0=0.0)
    x_center = f.L * c
    width = f.L * w
    x_h = f.dx/2 : f.dx : (f.L-f.dx/2)
    x_u = 0.0:f.dx:f.L
    h = h0.*exp.(-((x_h .- x_center).^2) ./ (2*width^2))
    u = u0.*exp.(-((x_u .- x_center).^2) ./ (2*width^2))
    x = ComponentVector(h=h,u=u)
    return x    
end

function dh_dx!(∂h∂x, h, dx)
    nx = length(h)
    for i in 2:nx
        ∂h∂x[i] = (h[i] - h[i-1])/dx
    end
    ∂h∂x[1] = 0.0
    ∂h∂x[end] = 0.0
end

function avg_h!(h_avg, h, dx)
    nx = length(h)
    for i in 2:nx
        h_avg[i] = (h[i] + h[i-1])/2.0
    end
    h_avg[1] = h[1]
    h_avg[end] = h[end]
end

function du_dx!(∂u∂x, u, dx)
    nx = length(u) - 1
    for i in 1:nx
        ∂u∂x[i] = (u[i+1] - u[i]) / dx
    end
end

function (f::Wave1DSurge_cpu)(dx_dt, x, p, t)
    g = f.g
    D = f.D
    dx = f.dx
    rho = f.rho
    C = f.C
    tau_val = f.tau(t)
    W = f.W
    tide_val = f.tide(t)
    tide_der_val = f.tide_derivative(t)
    # q_left_val = f.q_left(t)
    # q_right_val = f.q_right(t)

    ∂h∂x = similar(x.u)
    ∂H∂x = similar(x.h)
    h_avg = similar(x.u)

    x.h[1] = tide_val
    avg_h!(h_avg, x.h, dx)
    H = h_avg .+ D
    u = x.u
    # u[1] = q_left_val / (H[1]*W)
    # u[end] = q_right_val / (H[end]*W)
    # u[1] = tide_val * sqrt(f.g/(x.h[1]+D[1]))
    # u[end] = x.h[end] * sqrt(f.g/(x.h[end]+D[end]))
    Hu = H .* u

    du_dx!(∂H∂x, Hu, dx)
    dh_dx!(∂h∂x, x.h, dx)

    @. dx_dt.u = -g * ∂h∂x + tau_val/(rho*H) - (g*x.u*abs(x.u))/(C*C*rho*H)
    @. dx_dt.h = -∂H∂x
    dx_dt.h[1] = tide_der_val
    # dx_dt.u[1] = tide_der_val*sqrt(f.g/(x.h[1]+D[1])) + tide_val*(-0.5)*sqrt(f.g/(x.h[1]+D[1])^3)*tide_der_val
                    # tau_val/(rho*H[1]) - (g*x.u[1]*abs(x.u[1]))/(C*C*rho*H[1])   
    dx_dt.u[1] = dx_dt.u[2]
    if f.open_right_bc
        # Sommerfeld outgoing radiation: du/dt = -c * du/dx  (one-way wave equation)
        c_right = sqrt(g * H[end])
        dx_dt.u[end] = -c_right * (x.u[end] - x.u[end-1]) / dx
    else
        # Closed wall: zero normal velocity, no flux
        dx_dt.u[end] = 0.0
    end
end

function tide_func(t,a, phi)
    return a*sin.(2*pi*t/(12.25*3600.0) .+ phi)
    
end

tides(t) = tide_func(t, 0.1, 0.0)

function tide_derivative(t, a, phi)
    return (2*pi/(12.25*3600.0))*a*cos.(2*pi*t/(12.25*3600.0) .+ phi)
end

tides_der(t) = tide_derivative(t, 0.1, 0.0)

function tau_func(t,a=1.0,T=3600.0)
    return (a*sin(2*π*t/T))^2
end

function constant_func(t,value=0.0)
    return value
end

tau(t) = tau_func(t, 0.5, 3600.0)
# tau(t) = constant_func(t, 0.0)

####################
# Visualisation helpers (call manually on a single sol)
####################

function movie_solution(sol, grid_h, grid_u, D_center, times, filename="solution.mp4")
    h_all = [sol(t).h for t in times]
    u_all = [sol(t).u for t in times]

    h_min = minimum(minimum.(h_all)) - 0.05
    h_max = maximum(maximum.(h_all)) + 0.05
    u_min = minimum(minimum.(u_all)) - 0.01
    u_max = maximum(maximum.(u_all)) + 0.01

    fig = CairoMakie.Figure(size=(900, 400))
    ax_h = CairoMakie.Axis(fig[1, 1], xlabel="x (km)", ylabel="Water level (m)", title="Water level",
                            width=350)
    ax_u = CairoMakie.Axis(fig[1, 2], xlabel="x (km)", ylabel="Velocity (m/s)", title="Velocity",
                            width=350)

    lines!(ax_h, grid_h ./ 1e3, -D_center, color=:brown, linestyle=:dash, label="Bed")
    ylims!(ax_h, h_min, h_max)
    ylims!(ax_u, u_min, u_max)

    h_obs = Observable(h_all[1])
    u_obs = Observable(u_all[1])
    t_obs = Observable(Float64(times[1]))

    lines!(ax_h, grid_h ./ 1e3, h_obs, color=:blue, label="η")
    lines!(ax_u, grid_u ./ 1e3, u_obs, color=:red)

    CairoMakie.Label(fig[0, 1], @lift("t = $(round($t_obs / 3600, digits=2)) h"), fontsize=14)

    CairoMakie.record(fig, filename, enumerate(times); framerate=30) do (i, t)
        h_obs[] = h_all[i]
        u_obs[] = u_all[i]
        t_obs[] = Float64(t)
    end
    println("Saved to $filename")
end

function plot_timeseries(sol, grid_h, grid_u, times, x_probe; filename="timeseries.png")
    ih = argmin(abs.(grid_h .- x_probe))
    iu = argmin(abs.(grid_u .- x_probe))

    t_hours = collect(times) ./ 3600.0
    h_ts = [sol(t).h[ih] for t in times]
    u_ts = [sol(t).u[iu] for t in times]

    fig = CairoMakie.Figure(size=(800, 400))
    ax_h = CairoMakie.Axis(fig[1, 1], xlabel="Time (h)", ylabel="Water level (m)",
                            title="x = $(round(x_probe/1e3, digits=1)) km")
    ax_u = CairoMakie.Axis(fig[1, 2], xlabel="Time (h)", ylabel="Velocity (m/s)",
                            title="x = $(round(x_probe/1e3, digits=1)) km")

    lines!(ax_h, t_hours, h_ts, color=:blue)
    lines!(ax_u, t_hours, u_ts, color=:red)
    hlines!(ax_h, [0.0], color=:black, linestyle=:dash, linewidth=0.5)
    hlines!(ax_u, [0.0], color=:black, linestyle=:dash, linewidth=0.5)

    save(filename, fig)
    println("Saved to $filename")
end

####################
# Data generation
####################

datadir  = "data/lake1d_tidesurge_short"
if !isdir(datadir)
    mkpath(datadir)
end
moviesdir = joinpath(datadir, "movies")
if !isdir(moviesdir)
    mkpath(moviesdir)
end

train_frac = 0.7
val_frac   = 0.2

# Domain / physics
nx     = 100
L      = 100.0e3
dx     = L / nx
rho    = 1000.0
g      = 10.0
grid_u = collect(0.0:dx:L)
grid_h = collect(dx/2:dx:(L-dx/2))
D_min    = 10.0
D_max    = 30.0
D_edges  = [0.3, 0.7] .* L
D_widths = [0.05, 0.05] .* L
D        = depth_profile(grid_u, D_min, D_max, D_edges, D_widths)
D_center = Float32.(depth_profile(grid_h, D_min, D_max, D_edges, D_widths))
W = 100.0
C = 60.0

# Timing: 48 h total, first 4 h spin-up discarded
t_start    = 0.0
t_end      = 24 * 3600.0
spin_up    = 4  * 3600.0
dt_out     = 60.0
times_save = spin_up:dt_out:t_end

# Mesh metadata
# node_type: 0 = interior, 1 = tidal boundary node (left edge)
mesh_pos       = zeros(Float32, 2, length(grid_h))
mesh_pos[1,:] .= grid_h
node_type      = zeros(Int32, length(grid_h))
node_type[1]   = 1   # left boundary node

edges         = zeros(Int32, 2, length(grid_h) - 1)
edges[1, :]   = 1:(length(grid_h) - 1)
edges[2, :]   = edges[1, :] .+ 1

n_bc_nodes = sum(node_type .== 1)   # 1 for this domain

# Sweep parameters
tau_amps    = [0.0, 0.1, 0.25, 0.5]  # N/m²: calm → strong wind
tau_periods = [6.0, 12.0, 18.0]     # hours: sea-breeze / M2-resonant / diurnal
tide_amps   = [0.1, 0.25, 0.5, 1.0] # m: no tide → mesotidal

n_traj = length(Iterators.product(tau_amps, tau_periods, tide_amps))
@info "Total trajectories: $n_traj"

# Train / val / test split (symmetric like wave1d.jl)
train_inds = sample(rng, 1:n_traj÷2, floor(Int, train_frac * n_traj / 2), replace=false)
train_inds = vcat(train_inds, (n_traj + 1) .- train_inds)
rem_inds   = setdiff(1:n_traj, train_inds)
val_inds   = sample(rng, rem_inds, floor(Int, length(rem_inds) * val_frac / (1 - train_frac)), replace=false)
test_inds  = setdiff(rem_inds, val_inds)

# Clear existing files
for file in ["train", "valid", "test"]
    p = joinpath(datadir, "$file.jld2")
    isfile(p) && rm(p)
end

for (ind, (tau_amp, tau_period_h, tide_amp)) in enumerate(Iterators.product(tau_amps, tau_periods, tide_amps))

    tau_period_s  = tau_period_h * 3600.0
    tau_f(t)      = tau_func(t, tau_amp, tau_period_s)
    tide_f(t)     = tide_func(t, tide_amp, 0.0)
    tide_der_f(t) = tide_derivative(t, tide_amp, 0.0)

    f_ode = Wave1DSurge_cpu(g, D, L, W, dx, nx, rho, C, tau_f, tide_f, tide_der_f, false)  # true = open right BC (Sommerfeld)
    x0    = ComponentVector(h=zeros(length(grid_h)), u=zeros(length(grid_u)))

    prob = ODEProblem(f_ode, x0, (t_start, t_end))
    @time sol = solve(prob, Tsit5(), adaptive=false, dt=30.0)

    # Interpolate u to h-grid for saved times
    interp_u = zeros(length(grid_h), length(times_save))
    for (kk, time) in enumerate(times_save)
        lin_interp      = linear_interpolation(grid_u, sol(time).u)
        interp_u[:, kk] = lin_interp(grid_h)
    end

    velocity_node = Float32.(interp_u)
    # Exclude virtual left boundary u-node (index 1)
    velocity_edge = Float32.(cat([sol(t).u[2:end] for t in times_save]..., dims=2))
    waterlevel    = Float32.(cat([sol(t).h         for t in times_save]..., dims=2))

    # Wind forcing replicated across all nodes (n_nodes × n_times)
    forcing = repeat(Float32.(tau_f.(times_save))', length(grid_h), 1)

    # Boundary conditions: one row per BC node (n_bc_nodes × n_times)
    bound_cond = Float32.(reshape(tide_f.(times_save), 1, :))  # (1 × n_times)

    function save_traj!(path, traj_ind)
        jldopen(path, "a") do jf
            grp = JLD2.Group(jf, "trajectory_$(traj_ind)")
            grp["edges"]         = edges
            grp["velocity_edge"] = velocity_edge
            grp["velocity_node"] = velocity_node
            grp["waterlevel"]    = waterlevel
            grp["mesh_pos"]      = mesh_pos
            grp["node_type"]     = node_type
            grp["bathymetry"]    = D_center
            grp["n_nodes"]       = size(mesh_pos, 2)
            grp["forcing"]       = forcing
            grp["bound_cond"]    = bound_cond
            grp["bc_dyn_indices"] = Int32[1]   # row 1 of bound_cond → dynamic feature 1 (waterlevel)
        end
    end

    split_label = if ind in train_inds
        println("Train! ($ind/$n_traj)  tau_amp=$tau_amp  tau_T=$(tau_period_h)h  tide_amp=$tide_amp")
        save_traj!(joinpath(datadir, "train.jld2"), findfirst(train_inds .== ind))
        "train"
    elseif ind in val_inds
        println("Valid! ($ind/$n_traj)  tau_amp=$tau_amp  tau_T=$(tau_period_h)h  tide_amp=$tide_amp")
        save_traj!(joinpath(datadir, "valid.jld2"), findfirst(val_inds .== ind))
        "valid"
    else
        println("Test!  ($ind/$n_traj)  tau_amp=$tau_amp  tau_T=$(tau_period_h)h  tide_amp=$tide_amp")
        save_traj!(joinpath(datadir, "test.jld2"),  findfirst(test_inds .== ind))
        "test"
    end

    movie_fn = joinpath(moviesdir, "$(split_label)_traj$(ind)_tauA$(tau_amp)_tauT$(tau_period_h)h_tideA$(tide_amp).mp4")
    movie_solution(sol, grid_h, grid_u, D_center, times_save, movie_fn)
end
