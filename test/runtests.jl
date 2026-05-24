using Test

@testset "HydroGNN" begin
    include("test_gnn_model.jl")
    include("test_normalization.jl")
    include("test_training.jl")
end
