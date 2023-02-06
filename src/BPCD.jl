@enumx BPCDType Implicit Explicit

struct BPCD
    cell_size
    domain_min
    cell_count
    cell_prefix_sum
    particle_id
    cp_list
    cp_range
end

Adapt.@adapt_structure BPCD

function BPCD(particle_count::T, hash_table_size::T, max_radius::Number, domain_min::AbstractVector, type::BPCDType.T) where {T <: Integer}
    if type == BPCDType.Explicit && !isnothing(GLOBAL_CONFIG)
        cp_list_size = GLOBAL_CONFIG[].collision_pair_init_capacity_factor * particle_count
        cp_range_size = particle_count
    else
        cp_list_size = 0
        cp_range_size = 0
    end

    return BPCD(
        max_radius * 4,
        domain_min,
        zeros(T, hash_table_size),
        zeros(T, hash_table_size),
        zeros(T, particle_count),
        zeros(T, cp_list_size),
        zeros(HashCell{T}, cp_range_size),
    )
end

function BPCD(particle_count, max_radius, domain_min::AbstractVector, domain_max::AbstractVector, type::BPCDType.T = BPCDType.Implicit)
    v = (domain_max - domain_min) / (4 * max_radius)
    sz = clamp(prod(ceil.(Int, v)), 1 << 20, 1 << 22)
    return BPCD(particle_count, sz, max_radius, domain_min, type)
end

cell(pos, domain_min, cell_size) = cartesian3morton(@. floor(Int32, (pos - domain_min) / cell_size) + Int32(1))

function count_particles!(domain_min, cell_size, cell_count, positions)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(positions)
        idx = cell(positions[i], domain_min, cell_size) % cell_count + 1
        CUDA.atomic_add!(pointer(cell_count, idx), 1)
    end
end

function setup!(bpcd::BPCD, positions)
    fill!(bpcd.cell_count, zero(eltype(bpcd.cell_count)))
    CUDA.@sync @cuda threads=256 blocks=ceil(Int, length(positions) / 256) count_particles!(bpcd.domain_min, bpcd.cell_size, bpcd.cell_count, positions)
    accumulate!(+, bpcd.cell_prefix_sum, bpcd.cell_count)
end

function detect_collision!(bpcd::BPCD, positions, callback)

end
