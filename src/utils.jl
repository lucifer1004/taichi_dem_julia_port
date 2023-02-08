cell(pos, domain_min, cell_size) = @. ceil(Int32, (pos - domain_min) / cell_size)
hashcode(ijk, hash_table_size) = cartesian3morton(ijk) % hash_table_size + 1
center(ijk, domain_min, cell_size) = @. domain_min + (ijk - 0.5) * cell_size
function neighbors(o, ijk, c, hash_table_size)
    dijk = @SVector [o[i] > c[i] ? 1 : -1 for i in 1:3]
    return @SVector [
        hashcode(ijk, hash_table_size),
        hashcode(ijk + @SVector([dijk[1], 0, 0]), hash_table_size),
        hashcode(ijk + @SVector([0, dijk[2], 0]), hash_table_size),
        hashcode(ijk + @SVector([0, 0, dijk[3]]), hash_table_size),
        hashcode(ijk + @SVector([dijk[1], dijk[2], 0]), hash_table_size),
        hashcode(ijk + @SVector([dijk[1], 0, dijk[3]]), hash_table_size),
        hashcode(ijk + @SVector([0, dijk[2], dijk[3]]), hash_table_size),
        hashcode(ijk + @SVector([dijk[1], dijk[2], dijk[3]]), hash_table_size),
    ]
end

@inline function get_bit(bs, idx)
    return (bs[idx >> 5 + 1] & (UInt32(1) << (idx & 31))) != 0
end

@inline function set_bit!(bs, idx)
    CUDA.atomic_or!(pointer(bs, idx >> 5 + 1), UInt32(1) << (idx & 31))
end

@inline function clear_bit!(bs, idx)
    CUDA.atomic_and!(pointer(bs, idx >> 5 + 1), ~(UInt32(1) << (idx & 31)))
end

function save_single(grains, contacts, total_contacts, contact_active, contact_bonded, p4p, p4c, t)
    @info "Start saving..."
    t₁ = time_ns()
    n = length(grains)

    println(p4p, "TIMESTEP PARTICLES")
    println(p4p, "$t $n")
    println(p4p, "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ")

    g = Array(grains)
    for i in 1:n
        println(p4p,
                "$(g[i].id) $(g[i].gid) $(g[i].V) $(g[i].m) $(g[i].𝐤[1]) $(g[i].𝐤[2]) $(g[i].𝐤[3]) $(g[i].𝐯[1]) $(g[i].𝐯[2]) $(g[i].𝐯[3])")
    end

    println(p4c, "TIMESTEP CONTACTS")

    c = Array(contacts[1:total_contacts])
    ca = Array(contact_active)
    cb = Array(contact_bonded)
    cache = String[]
    for contact in c
        i = contact.i
        j = contact.j
        ij = UInt64(i - 1) * n +  j
        if get_bit(ca, ij)
            bonded = get_bit(cb, ij)
            𝐤 = contact.𝐤
            𝐅ᵢ = contact.𝐅ᵢ
            push!(cache,
                    "$i $j $(𝐤[1]) $(𝐤[2]) $(𝐤[3]) $(𝐅ᵢ[1]) $(𝐅ᵢ[2]) $(𝐅ᵢ[3]) $(Int(bonded))\n")
        end
    end

    println(p4c, "$t $(length(cache))")
    println(p4c, "P1  P2  CX  CY  CZ  FX  FY  FZ  CONTACT_IS_BONDED")
    println(p4c, join(cache))

    t₂ = time_ns()
    Δt = (t₂ - t₁) * 1e-9
    @info "Checkpoint saved! $(length(cache)) active contacts" Δt
end

function snapshot(grains, step)
    grains = Array(grains)
    𝐤 = [g.𝐤 for g in grains]
    x = [k[1] for k in 𝐤]
    y = [k[2] for k in 𝐤]
    z = [k[3] for k in 𝐤]
    r = [g.r for g in grains]
    fig, _ = meshscatter(x,
                         y,
                         z;
                         markersize = r,
                         color = z,
                         axis = (;
                                 type = Axis3,
                                 aspect = :data,
                                 azimuth = 7.3,
                                 elevation = 0.189,
                                 perspectiveness = 0.5),
                         figure = (; resolution = (1200, 800)))
    save("snapshot_$step.png", fig)
end
