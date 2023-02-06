export DEMConfig, DEMConfigDefault

@option struct DEMConfig{I, F}
    domain_min::SVector{3, F} = SVector{3, F}(-200.0, -200.0, -30.0)
    domain_max::SVector{3, F} = SVector{3, F}(200.0, 200.0, 90.0)
    init_particles::String = joinpath(@__DIR__, "..", "Resources", "bunny.p4p")
    particle_contact_radius_multiplier::F = 1.1
    neighboring_search_safety_factor::F = 1.01
    particle_elastic_modulus::F = 7e10
    particle_poisson_ratio::F = 0.25
    wall_normal::SVector{3, F} = SVector{3, F}(0.0, 0.0, -1.0)
    wall_distance::F = 25.0
    wall_density::F = 7800
    wall_elastic_modulus::F = 2e11
    wall_poisson_ratio::F = 0.25

    bond_radius_ratio::F = 0.5
    bond_elastic_modulus::F = 28e9
    bond_poisson_ratio::F = 0.2
    bond_compressive_strength::F = 3e8
    bond_tensile_strength::F = 6e7
    bond_shear_strength::F = 6e7

    pp_coefficient_friction::F = 0.3
    pp_coefficient_restitution::F = 0.9
    pp_coefficient_rolling_resistance::F = 0.01

    pw_coefficient_friction::F = 0.35
    pw_coefficient_restitution::F = 0.7
    pw_coefficient_rolling_resistance::F = 0.01

    max_coordinate_number::I = 64
    collision_pair_init_capacity_factor::I = 128

    tolerance::F = 1e-12
    max_particle_count::I = 1000000000

    gravity::SVector{3, F} = SVector{3, F}(0.0, 0.0, -9.81)
    global_damping::F = 0.0
    dt::F = 2.63e-5
    target_time::F = 0.5
    saving_interval_time::F = 0.05
end

DEMConfigDefault = DEMConfig{Int, Float64}

GLOBAL_CONFIG = Ref{Union{DEMConfig, Nothing}}(nothing)
GLOBAL_EX = Ref{Any}(ThreadedEx())

function load_config(file)
    GLOBAL_CONFIG[] = from_toml(DEMConfigDefault, file)
end

function use_cpu()
    GLOBAL_EX[] = ThreadedEx()
end

function use_gpu()
    GLOBAL_EX[] = CUDAEx()
end
