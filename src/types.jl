const Vec2 = SVector{2, Float64}
const Vec3 = SVector{3, Float64}
const Vec4 = SVector{4, Float64}
const Vec2i = SVector{2, Int32}
const Vec3i = SVector{3, Int32}
const Vec8i = SVector{8, Int32}
const Mat3x3 = SMatrix{3, 3, Float64, 9}

struct Material{T <: AbstractFloat}
    ρ::T # Density
    E::T # Elastic modulus
    ν::T # Poisson's ratio
end

const MaterialDefault = Material{Float64}

struct Surface{T <: AbstractFloat}
    μ::T # Coefficient of friction
    e::T # Coefficient of restitution
    μᵣ::T # Coefficient of rolling resistance
    ρ::T # Radius ratio
    E::T # Elastic modulus
    ν::T # Poisson's ratio
    σ𝑐::T # Compressive strength
    σ𝑡::T # Tensile strength
    σ𝑠::T # Shear strength
end

const SurfaceDefault = Surface{Float64}

struct Grain{I <: Integer, F <: AbstractFloat}
    id::I
    gid::I
    mid::I
    V::F
    m::F
    r::F
    r₀::F
    𝐤::SVector{3, F}
    𝐯::SVector{3, F}
    𝐚::SVector{3, F}
    𝐪::Quaternion{F}
    𝛚::SVector{3, F}
    d𝛚::SVector{3, F}
    𝐈::SMatrix{3, 3, F, 9}
end

const GrainDefault = Grain{Int32, Float64}

struct IOGrain{I <: Integer, F <: AbstractFloat}
    id::I
    V::F
    m::F
    𝐤::SVector{3, F}
    𝐯::SVector{3, F}
end

const IOGrainDefault = IOGrain{Int32, Float64}

struct Wall{I <: Integer, F <: AbstractFloat}
    𝐧::SVector{3, F}
    d::F
    mid::I
end

WallDefault = Wall{Int32, Float64}

struct Contact{I <: Integer, F <: AbstractFloat}
    i::I
    j::I
    midᵢ::I
    midⱼ::I
    𝐤::SVector{3, F}
    𝐅ᵢ::SVector{3, F}
    𝛕ᵢ::SVector{3, F}
    𝛕ⱼ::SVector{3, F}
    𝐬::SVector{3, F} # Shear displacement
end

const ContactDefault = Contact{Int32, Float64}
const ContactDefaultZero = Contact{Int32, Float64}(0, 0, 0, 0, zero(Vec3), zero(Vec3),
                                                   zero(Vec3), zero(Vec3), zero(Vec3))

struct IOContact{I <: Integer, F <: AbstractFloat}
    i::I
    j::I
    𝐤::SVector{3, F}
    𝐅ᵢ::SVector{3, F}
    bonded::Bool
end

const IOContactDefault = IOContact{Int32, Float64}
