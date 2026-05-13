cd(@__DIR__)
using Pkg
Pkg.activate(".")

using OrdinaryDiffEq
using ComponentArrays
using Interpolations
using Random
using StatsBase
using JLD2

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
    q_left::Function # Inflow at left boundary
    q_right::Function # Outflow at right boundary
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
    q_left_val = f.q_left(t)
    q_right_val = f.q_right(t)

    ∂h∂x = similar(x.u)
    ∂H∂x = similar(x.h)
    h_avg = similar(x.u)

    avg_h!(h_avg, x.h, dx)
    H = h_avg .+ D
    u = x.u
    u[1] = q_left_val / (H[1]*W)
    u[end] = q_right_val / (H[end]*W)
    Hu = H .* u

    du_dx!(∂H∂x, Hu, dx)
    dh_dx!(∂h∂x, x.h, dx)

    @. dx_dt.u = -g * ∂h∂x + tau_val/(rho*H) - (g*x.u*abs(x.u))/(C*C*rho*H)
    dx_dt.u[1] = 0.0
    dx_dt.u[end] = 0.0
    @. dx_dt.h = -∂H∂x
end

###########################
# Helper functions TFRecord
###########################

function convert_array_to_uint8(array)
    # convert to array of uint8 vectors
    array_new = [reinterpret(UInt8, [x]) for x in array]
    array_new = [UInt8.(x) for x in array_new]
    # flatten into a singleton vector{vector{UInt8}} as desired by TFRecord.jl
    return [collect(Iterators.flatten(array_new))] 
end

####################
# Setting up Problem
####################
write_x0 = 1
# setting up the problem
# parameters
nx=100     # number of spatial points
L=100.0e3  # length of the domain
dx=L/nx    # spatial step
rho=1000.0 # density of water
g=10.0     # acceleration due to gravity
grid_u=collect(0.0:dx:L) # grid for velocity
grid_h=collect(dx/2:dx:(L-dx/2)) # grid for water levels
D_min=10.0 # minimum depth of the water
D_max=30.0 # maximum depth of the water
D_edges=[0.3,0.7] .* L # edges of the depth change
D_widths=[0.05,0.05] .* L # widths of the depth change
D=depth_profile(grid_u,D_min,D_max,D_edges,D_widths)
D_center=Float32.(depth_profile(grid_h, D_min, D_max, D_edges, D_widths))
W=100.0 # width
C=60.0 # Chezy friction factor

datadir = "data/lake1d_surge"
if !isdir(datadir)
    mkpath(datadir)
end

train_frac = 0.7
val_frac = 0.2

# time span
t_start=0.0
t_end=6*3600.0 # hours

# cfl_out = 1.0
# dt_out = cfl_out*f.dx/(sqrt(maximum(f.D)*f.g))
dt_out = 60.0
times = 0.0:dt_out:t_end

# spatial grid 
x_h = grid_h
x_u = grid_u
x=ComponentVector(h=x_h,u=x_u)

# ini_pos = 0.25:0.01:0.75
ini_amp = 0.1:0.05:1
ini_period = 0.25:0.25:2.0
n_traj = length(Iterators.product(ini_amp, ini_period))

# This makes a symmetric train data set
# Val, test sets are not necessarily symmetric
train_inds = sample(rng, 1:n_traj÷2, floor(Int, train_frac*n_traj/2), replace=false)
train_inds = vcat(train_inds, (n_traj+1).-train_inds)
rem_inds = setdiff(1:n_traj, train_inds)
val_inds = sample(rng, rem_inds, floor(Int, length(rem_inds)*val_frac/(1-train_frac)), replace=false)
test_inds = setdiff(rem_inds, val_inds)

# Create mesh_pos array to be saved to tfrecord
# This uses staggered grid, we will interpolate everything to the waterlevel grid
mesh_pos = zeros(Float32,2,length(grid_h))
mesh_pos[1,:] .= grid_h

# Since everything is on the (smaller) waterlevel grid
# There are no boundary points, so every node has node_type 0
node_type = zeros(Int32, length(grid_h))


edges = zeros(Int32, 2, length(grid_h)-1)
edges[1,:] = 1:(length(grid_h)-1)
edges[2,:] = edges[1,:] .+ 1

# wind stress as a function of time
# wind stress forcing
function tau_func(t,a=1.0,T=3600.0)
    return (a*sin(2*π*t/T))^2
end
# tau(t)=tau_func(t,1.0,8*3600.0) # set parameters

function constant_func(t,value=0.0)
    return value
end
# tau(t) = constant_func(t, 0.0)
q_left(t)=constant_func(t,0.0)
q_right(t)=constant_func(t,0.0) #Note q_left-q_right is the nett influx

for file in ["train", "valid", "test"]
    if isfile(joinpath(datadir, "$file.jld2"))
        rm(joinpath(datadir, "$file.jld2"))
    end
end

# for (ind, pos) in enumerate(ini_pos)
for (ind, (amp, period)) in enumerate(Iterators.product(ini_amp, ini_period))
    
    tau(t) = tau_func(t, amp, period*3600.0)

    f = Wave1DSurge_cpu(g,D,L,W,dx,nx,rho,C,tau,q_left,q_right)
    # x0 = initial_state_bump(f,1,0.05,pos,0.0) # h bump
    #x0=initial_state_bump(f,0.0,0.05,0.3,0.1) # u bump
    
    x0 = ComponentVector(
        h = zeros(length(grid_h)),
        u = zeros(length(grid_u))
    )

    prob = ODEProblem(f, x0, (t_start, t_end))
    @time sol = solve(prob, Tsit5())
    
    # Interp u to h grid
    interp_u = zeros((length(grid_h), length(times)))
    for (kk,time) in enumerate(times)
        lin_interp = linear_interpolation(grid_u, sol(time).u)
        interp_u[:,kk] = lin_interp(grid_h)
    
    end

    velocity_node = Float32.(interp_u)
    velocity_edge = Float32.(cat([x.u for x in sol(times).u]..., dims=2))
    waterlevel = Float32.(cat([x.h for x in sol(times).u]..., dims=2))

    winds = Float32.(tau.(times))
    winds = repeat(winds', size(waterlevel,1), 1)

    if ind in train_inds
        println("Train!")
        jldopen(joinpath(datadir, "train.jld2"), "a") do f
            traj_ind = findfirst(train_inds.==ind)
            traj_group = JLD2.Group(f, "trajectory_$(traj_ind)")

            traj_group["edges"] = edges
            traj_group["velocity_edge"] = velocity_edge
            traj_group["velocity_node"] = velocity_node
            traj_group["waterlevel"] = waterlevel
            traj_group["mesh_pos"] = mesh_pos
            traj_group["node_type"] = node_type
            traj_group["bathymetry"] = D_center
            traj_group["n_nodes"] = size(mesh_pos,2)
            traj_group["tau"] = winds
        end
    elseif ind in val_inds
        println("Valid!")
        jldopen(joinpath(datadir, "valid.jld2"), "a") do f
            traj_ind = findfirst(val_inds.==ind)
            traj_group = JLD2.Group(f, "trajectory_$(traj_ind)")

            traj_group["edges"] = edges
            traj_group["velocity_edge"] = velocity_edge
            traj_group["velocity_node"] = velocity_node
            traj_group["waterlevel"] = waterlevel
            traj_group["mesh_pos"] = mesh_pos
            traj_group["node_type"] = node_type
            traj_group["bathymetry"] = D_center
            traj_group["n_nodes"] = size(mesh_pos,2)
            traj_group["tau"] = winds
        end
    else
        println("Test!")
        jldopen(joinpath(datadir, "test.jld2"), "a") do f
            traj_ind = findfirst(test_inds.==ind)
            traj_group = JLD2.Group(f, "trajectory_$(traj_ind)")

            traj_group["edges"] = edges
            traj_group["velocity_edge"] = velocity_edge
            traj_group["velocity_node"] = velocity_node
            traj_group["waterlevel"] = waterlevel
            traj_group["mesh_pos"] = mesh_pos
            traj_group["node_type"] = node_type
            traj_group["bathymetry"] = D_center
            traj_group["n_nodes"] = size(mesh_pos,2)
            traj_group["tau"] = winds

        end
    end
end

