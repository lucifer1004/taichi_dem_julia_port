Vec2 = SVector{2, Float64}
Vec3 = SVector{3, Float64}
Vec4 = SVector{4, Float64}
Vec2i = SVector{2, Int32}
Vec3i = SVector{3, Int32}
Vec8i = SVector{8, Int32}
Mat3x3 = SMatrix{3, 3, Float64, 9}

struct Material{T <: AbstractFloat}
    density::T
    elastic_modulus::T
    poisson_ratio::T
end

struct Surface{T <: AbstractFloat}
    coefficient_friction::T
    coefficient_restitution::T
    coefficient_rolling_resistance::T
    radius_ratio::T
    elastic_modulus::T
    poisson_ratio::T
    compressive_strength::T
    tensile_strength::T
    shear_strength::T
end

struct Grain{I <: Integer, F <: AbstractFloat}
    id::I
    material_type::I
    radius::F
    contact_radius::F
    position::SVector{3, F}
    velocity::SVector{3, F}
    acceleration::SVector{3, F}
    force::SVector{3, F}
    quaternion::SVector{4, F}
    omega::SVector{3, F}
    omega_dot::SVector{3, F}
    inertia::SMatrix{3, 3, F, 9}
    moment::SVector{3, F}
end

struct Wall{I <: Integer, F <: AbstractFloat}
    normal::SVector{3, F}
    distance::F
    material_type::I
end

struct Contact{I <: Integer, F <: AbstractFloat}
    i::I
    j::I
    material_type_i::I
    material_type_j::I
    position::SVector{3, F}
    force_a::SVector{3, F}
    moment_a::SVector{3, F}
    moment_b::SVector{3, F}
    shear_displacement::SVector{3, F}
    is_active::Bool
    is_bonded::Bool
end

struct IOContact{I <: Integer, F <: AbstractFloat}
    i::I
    j::I
    position::SVector{3, F}
    force_a::SVector{3, F}
    is_bonded::Bool
    is_active::Bool
end

struct HashCell{T <: Integer}
    offset::T
    count::T
    current::T
end

Base.zero(::Type{HashCell{T}}) where {T} = HashCell(zero(T), zero(T), zero(T))
