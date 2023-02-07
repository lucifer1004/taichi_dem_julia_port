function main(cfg_filename)
    cfg = from_toml(DEMConfig{Int32,Float64}, cfg_filename)
    (;
        domain_min,
        domain_max,
        init_particles,
        max_coordinate_number,
        particle_contact_radius_multiplier,
        collision_pair_init_capacity_factor,
        dt,
        target_time,
        saving_interval_time,
        particle_elastic_modulus,
        particle_poisson_ratio,
        wall_density,
        wall_elastic_modulus,
        wall_poisson_ratio,
        pp_coefficient_friction,
        pp_coefficient_restitution,
        pp_coefficient_rolling_resistance,
        bond_radius_ratio,
        bond_elastic_modulus,
        bond_poisson_ratio,
        bond_compressive_strength,
        bond_tensile_strength,
        bond_shear_strength,
        pw_coefficient_friction,
        pw_coefficient_restitution,
        pw_coefficient_rolling_resistance,
        wall_normal,
        wall_distance,
    ) = cfg

    lines = split(read(init_particles, String), "\n")

    n = parse(Int, split(lines[2])[2]) # $timestamp $particles
    grains_cpu = map(lines[4:n+3]) do line
        # FIXME: group ID is omitted
        id, _gid, V, m, k_x, k_y, k_z, v_x, v_y, v_z = parse.(Float64, split(line))
        k = @SVector [k_x, k_y, k_z]
        v = @SVector [v_x, v_y, v_z]
        r = ∛(V / (4 / 3 * π))
        I = @SMatrix [
            2/5*m*r^2 0 0
            0 2/5*m*r^2 0
            0 0 2/5*m*r^2
        ]

        # FIXME: material type is hard-coded
        GrainDefault(
            id,
            1,
            V,
            m,
            r,
            r * particle_contact_radius_multiplier,
            k,
            v,
            zero(Vec3),
            zero(Quaternion{Float64}),
            zero(Vec3),
            zero(Vec3),
            I,
        )
    end
    material_density = grains_cpu[1].m / grains_cpu[1].V
    rₘₐₓ = maximum(g -> g.r, grains_cpu)
    grains = cu(grains_cpu)

    # Translational attributes, all in GLOBAL coordinates
    kid = CUDA.zeros(Vec3i, n) # Position id
    hid = CUDA.zeros(Int32, n) # Position hash
    kcenter = CUDA.zeros(Vec3, n) # Position center
    neighbor_cells = CUDA.zeros(Vec8i, n) # Position neighbor cells

    F = CUDA.zeros(Vec3, n) # Force
    τ = CUDA.zeros(Vec3, n) # Moment

    total_steps = trunc(Int, target_time / dt)
    save_per_steps = trunc(Int, saving_interval_time / dt)

    cell_size = 4rₘₐₓ
    hash_table_size =
        clamp(prod(ceil.(Int32, (domain_max - domain_min) / cell_size)), 1 << 20, 1 << 22)
    hash_table = CUDA.zeros(Int32, hash_table_size)
    hash_table_current = similar(hash_table)
    hash_table_prefix_sum = similar(hash_table)

    pid = CUDA.zeros(Int32, n)
    cp_list = CUDA.zeros(Int32, n * collision_pair_init_capacity_factor)
    cp_range = CUDA.zeros(Int32, n)
    cp_range_current = similar(cp_range)
    cp_range_prefix_sum = similar(cp_range)

    contacts = CuVector{ContactDefault}(undef, n * max_coordinate_number)
    contact_active = CUDA.zeros(Bool, n * max_coordinate_number)
    contact_bonded = CUDA.zeros(Bool, n * max_coordinate_number)
    contact_count = CUDA.zeros(Int32, n)

    # FIXME: materials are hard-coded
    materials = cu([
        MaterialDefault(material_density, particle_elastic_modulus, particle_poisson_ratio),
        MaterialDefault(wall_density, wall_elastic_modulus, wall_poisson_ratio),
    ])

    # FIXME: surfaces are hard-coded
    surfaces_cpu = Matrix{SurfaceDefault}(undef, 2, 2)
    surfaces_cpu[1, 1] = SurfaceDefault(
        pp_coefficient_friction,
        pp_coefficient_restitution,
        pp_coefficient_rolling_resistance,
        bond_radius_ratio,
        bond_elastic_modulus,
        bond_poisson_ratio,
        bond_compressive_strength,
        bond_tensile_strength,
        bond_shear_strength,
    )
    surfaces_cpu[1, 2] =
        surfaces_cpu[2, 1] = SurfaceDefault(
            pw_coefficient_friction,
            pw_coefficient_restitution,
            pw_coefficient_rolling_resistance,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    surfaces = cu(surfaces_cpu)

    # FIXME: walls are hard-coded
    walls = cu([WallDefault(wall_normal, wall_distance, 2)])

    threads = 512
    blocks = ceil(Int, n / threads)

    function clear_state()
        fill!(F, zero(Vec3))
        fill!(τ, zero(Vec3))
    end

    cell(pos, domain_min, cell_size) = @. ceil(Int32, (pos - domain_min) / cell_size)
    hashcode(ijk) = cartesian3morton(ijk) % length(hash_table) + 1
    center(ijk, domain_min, cell_size) = @. domain_min + (ijk - 0.5) * cell_size
    function neighbors(o, ijk, c)
        dijk = @SVector [o[i] > c[i] ? 1 : -1 for i = 1:3]
        return @SVector [
            hashcode(ijk),
            hashcode(ijk + @SVector([dijk[1], 0, 0])),
            hashcode(ijk + @SVector([0, dijk[2], 0])),
            hashcode(ijk + @SVector([0, 0, dijk[3]])),
            hashcode(ijk + @SVector([dijk[1], dijk[2], 0])),
            hashcode(ijk + @SVector([dijk[1], 0, dijk[3]])),
            hashcode(ijk + @SVector([0, dijk[2], dijk[3]])),
            hashcode(ijk + @SVector([dijk[1], dijk[2], dijk[3]])),
        ]
    end

    function count_particles!(hash_table, hid)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        for i = index:stride:length(hid)
            CUDA.atomic_add!(pointer(hash_table, hid[i]), Int32(1))
        end
    end

    function get_particle_id!(pid, hash_table_current, hid)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        for i = index:stride:length(hid)
            idx = hid[i]
            pid[hash_table_current[idx]] = i
            CUDA.atomic_add!(pointer(hash_table_current, idx), Int32(-1))
        end
    end

    function search_hash_table!(cp_range, hash_table, hash_table_current, neighbors, pid)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        for i = index:stride:length(pid)
            for idx in neighbors[i]
                for k = hash_table_current[idx]+1:hash_table_current[idx]+hash_table[idx]
                    j = pid[k]
                    if i < j
                        CUDA.atomic_add!(pointer(cp_range, i), Int32(1))
                    end
                end
            end
        end
    end

    function update_cp_list!(
        cp_list,
        cp_range_current,
        hash_table,
        hash_table_current,
        neighbors,
        pid,
    )
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        for i = index:stride:length(pid)
            for idx in neighbors[i]
                for k = hash_table_current[idx]+1:hash_table_current[idx]+hash_table[idx]
                    j = pid[k]
                    if i < j
                        current = cp_range_current[i]
                        cp_list[current] = j
                        CUDA.atomic_add!(pointer(cp_range_current, i), Int32(-1))
                    end
                end
            end
        end
    end

    # function append_contact_offset!(cfn, i)
    #     offset = CUDA.atomic_add!(pointer(cfn, i), Int32(1))
    #     if offset <= max_coordinate_number
    #         return (i - 1) * max_coordinate_number + offset
    #     end
    #     return 0
    # end

    # function search_active_contact_offset(i, j, cf, cfa, cfn, max_coordinate_number)
    #     for k in (i - 1) * max_coordinate_number .+ (1:cfn[i])
    #         if cf[k].j == j && cfa[k]
    #             return k
    #         end
    #     end
    #     return 0
    # end

    function resolve_collision!(
        contacts,
        contact_active,
        contact_bonded,
        contact_count,
        cp_list,
        cp_range,
        cp_range_current,
        grains,
        materials,
        surfaces,
        max_coordinate_number,
    )
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        for i = index:stride:length(pid)
            for k = cp_range_current[i]+1:cp_range_current[i]+cp_range[i]
                j = cp_list[k]

                ev = false
                offset = 0
                for k in (i - 1) * max_coordinate_number .+ (1:contact_count[i])
                    if contacts[k].j == j && contact_active[k]
                        offset = k
                        break
                    end
                end

                if offset > 0
                    if contact_bonded[offset]
                        ev = true
                    elseif norm(grains[i].𝐤 - grains[j].𝐤) < grains[i].r + grains[j].r
                        # Use PFC's gap < 0 criterion
                        ev = true
                    else
                        contact_active[offset] = false
                    end
                elseif norm(grains[i].𝐤 - grains[j].𝐤) < grains[i].r + grains[j].r
                    # Note that atomic_add! returns the old value
                    offset = CUDA.atomic_add!(pointer(contact_count, i), Int32(1)) + 1
                    if offset <= max_coordinate_number
                        offset += (i - 1) * max_coordinate_number
                    else
                        offset = 0
                    end

                    if offset > 0
                        z = SVector{3,Float64}(0.0, 0.0, 0.0)
                        contacts[offset] = ContactDefault(i, j, 1, 1, z, z, z, z, z)
                        contact_active[offset] = true
                        contact_bonded[offset] = false
                        ev = true
                    end
                end
            end
        end
    end

    function contact()
        map!(g -> cell(g.𝐤, domain_min, cell_size), kid, grains)
        map!(hashcode, hid, kid)
        map!(ijk -> center(ijk, domain_min, cell_size), kcenter, kid)

        # Setup hash table
        fill!(hash_table, 0)
        @cuda threads = threads blocks = blocks count_particles!(hash_table, hid)
        accumulate!(+, hash_table_prefix_sum, hash_table)
        copyto!(hash_table_current, hash_table_prefix_sum)
        @cuda threads = threads blocks = blocks get_particle_id!(
            pid,
            hash_table_current,
            hid,
        )

        # Detect collision
        fill!(cp_range, 0)
        map!((g, ijk, c) -> neighbors(g.𝐤, ijk, c), neighbor_cells, grains, kid, kcenter)
        @cuda threads = threads blocks = blocks search_hash_table!(
            cp_range,
            hash_table,
            hash_table_current,
            neighbor_cells,
            pid,
        )
        accumulate!(+, cp_range_prefix_sum, cp_range)
        copyto!(cp_range_current, cp_range_prefix_sum)

        total = CUDA.@allowscalar cp_range_prefix_sum[end]
        cap = length(cp_list)
        while cap < total
            cap <<= 1
        end
        if cap > length(cp_list)
            resize!(cp_list, cap)
        end

        @cuda threads = threads blocks = blocks update_cp_list!(
            cp_list,
            cp_range_current,
            hash_table,
            hash_table_current,
            neighbor_cells,
            pid,
        )

        # Resolve collision
        @cuda threads = threads blocks = blocks resolve_collision!(
            contacts,
            contact_active,
            contact_bonded,
            contact_count,
            cp_list,
            cp_range,
            cp_range_current,
            grains,
            materials,
            surfaces,
            max_coordinate_number,
        )
    end

    function resolve_wall() end

    function apply_body_force() end

    function update() end

    function late_clear_state() end

    function simulate()
        clear_state()
        contact()
        resolve_wall()
        apply_body_force()
        update()
        late_clear_state()
    end

    function save_single(grains, p4p, p4c, t)
        t₁ = time_ns()

        println(p4p, "TIMESTEP PARTICLES")
        println(p4p, "$t $n")
        println(p4p, "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ")

        g = Array(grains)
        for i = 1:n
            println(
                p4p,
                "$(g[i].id) $(g[i].mid) $(g[i].V) $(g[i].m) $(g[i].𝐤[1]) $(g[i].𝐤[2]) $(g[i].𝐤[3]) $(g[i].𝐯[1]) $(g[i].𝐯[2]) $(g[i].𝐯[3])",
            )
        end

        println(p4c, "TIMESTEP CONTACTS")
        println(p4c, "$t 0")
        println(p4c, "P1  P2  CX  CY  CZ  FX  FY  FZ  CONTACT_IS_BONDED")

        t₂ = time_ns()
        println("save time cost = $((t₂ - t₁) * 1e-9)s")
    end

    step = 0
    p4p = open("output.p4p", "w")
    p4c = open("output.p4c", "w")
    save_single(grains, p4p, p4c, 0.0)
    while step < total_steps
        t₁ = time_ns()
        for _ = 1:save_per_steps
            step += 1
            simulate()
        end
        t₂ = time_ns()

        println(">>>")
        println("solved steps: $step last-$(save_per_steps)-sim: $((t₂ - t₁) * 1e-9)s")
        save_single(grains, p4p, p4c, step * dt)
    end

    close(p4p)
    close(p4c)
end
