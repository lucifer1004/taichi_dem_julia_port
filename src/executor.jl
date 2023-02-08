function solve(cfg_filename; save_snapshot = false, save_information = true)
    start_time = time_ns()

    cfg = from_toml(DEMConfig{Int32, Float64}, cfg_filename)
    (;
    domain_min,
    domain_max,
    init_particles,
    max_coordinate_number,
    particle_contact_radius_multiplier,
    collision_pair_init_capacity_factor,
    neighboring_search_safety_factor,
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
    tolerance,
    gravity,
    global_damping) = cfg

    lines = split(read(init_particles, String), "\n")

    n = parse(Int, split(lines[2])[2]) # $timestamp $particles
    grains_cpu = map(lines[4:(n + 3)]) do line
        id, gid, V, m, k₁, k₂, k₃, v₁, v₂, v₃ = parse.(Float64, split(line))

        # TODO: implement other shapes
        r = ∛(V / (4 / 3 * π))
        𝐈 = @SMatrix [2/5*m*r^2 0 0
                      0 2/5*m*r^2 0
                      0 0 2/5*m*r^2]

        # FIXME: material type is hard-coded
        GrainDefault(id,
                     gid,
                     1,
                     V,
                     m,
                     r,
                     r * particle_contact_radius_multiplier,
                     Vec3(k₁, k₂, k₃),
                     Vec3(v₁, v₂, v₃),
                     zero(Vec3),
                     one(Quaternion{Float64}),
                     zero(Vec3),
                     zero(Vec3),
                     𝐈)
    end
    material_density = grains_cpu[1].m / grains_cpu[1].V
    # Calculate neighbor search radius
    rₘₐₓ = maximum(g -> g.r, grains_cpu) *
           particle_contact_radius_multiplier *
           (1.0 + bond_tensile_strength / bond_elastic_modulus) *
           neighboring_search_safety_factor
    grains = cu(grains_cpu)

    # Translational attributes, all in GLOBAL coordinates
    kid = CUDA.zeros(Vec3i, n) # Position id
    hid = CUDA.zeros(Int32, n) # Position hash
    kcenter = CUDA.zeros(Vec3, n) # Position center
    neighbor_cells = CUDA.zeros(Vec8i, n) # Position neighbor cells

    forces = CUDA.zeros(Float64, n * 3)
    moments = CUDA.zeros(Float64, n * 3)

    total_steps = trunc(Int, target_time / dt)
    save_per_steps = trunc(Int, saving_interval_time / dt)

    cell_size = 4rₘₐₓ
    hash_table_size = clamp(prod(ceil.(Int32, (domain_max - domain_min) / cell_size)),
                            1 << 20, 1 << 22)
    hash_table = CUDA.zeros(Int32, hash_table_size)
    hash_table_current = similar(hash_table)

    pid = CUDA.zeros(Int32, n)
    cp_list = CUDA.zeros(Vec2i, n * collision_pair_init_capacity_factor)
    cp_range = CUDA.zeros(Int32, n)
    cp_range_current = similar(cp_range)

    contacts = CuVector{ContactDefault}(undef, n * max_coordinate_number)
    contact_active = CUDA.zeros(Bool, n * max_coordinate_number)
    contact_bonded = CUDA.zeros(Bool, n * max_coordinate_number)
    contact_count = CUDA.zeros(Int32, n)

    # FIXME: materials are hard-coded
    materials = cu([
                       MaterialDefault(material_density, particle_elastic_modulus,
                                       particle_poisson_ratio),
                       MaterialDefault(wall_density, wall_elastic_modulus,
                                       wall_poisson_ratio),
                   ])

    # FIXME: surfaces are hard-coded
    surfaces_cpu = Matrix{SurfaceDefault}(undef, 2, 2)
    surfaces_cpu[1, 1] = SurfaceDefault(pp_coefficient_friction,
                                        pp_coefficient_restitution,
                                        pp_coefficient_rolling_resistance,
                                        bond_radius_ratio,
                                        bond_elastic_modulus,
                                        bond_poisson_ratio,
                                        bond_compressive_strength,
                                        bond_tensile_strength,
                                        bond_shear_strength)
    surfaces_cpu[1, 2] = surfaces_cpu[2, 1] = SurfaceDefault(pw_coefficient_friction,
                                                             pw_coefficient_restitution,
                                                             pw_coefficient_rolling_resistance,
                                                             0,
                                                             0,
                                                             0,
                                                             0,
                                                             0,
                                                             0)
    surfaces = cu(surfaces_cpu)

    # FIXME: walls are hard-coded
    walls = cu([WallDefault(wall_normal, wall_distance, 2)])
    nwall = length(walls)
    wall_contacts = CuMatrix{ContactDefault}(undef, nwall, n)

    threads = 256
    blocks = cld(n, threads)

    @info "Setting:" hash_table_size cell_size

    function clear_state()
        fill!(forces, 0)
        fill!(moments, 0)
    end

    cell_m(g) = cell(g.𝐤, domain_min, cell_size)
    hashcode_m(ijk) = hashcode(ijk, hash_table_size)
    center_m(ijk) = center(ijk, domain_min, cell_size)
    neighbors_m(g, ijk, c) = neighbors(g.𝐤, ijk, c, hash_table_size)

    function contact(; init = false)
        # Precalculate cell ijk, hash code, center and neighbors
        map!(cell_m, kid, grains)
        map!(hashcode_m, hid, kid)
        map!(center_m, kcenter, kid)
        map!(neighbors_m, neighbor_cells, grains, kid, kcenter)

        # Setup hash table
        fill!(hash_table, 0)
        @cuda threads=threads blocks=blocks count_particles!(hash_table, hid)
        accumulate!(+, hash_table_current, hash_table)
        @cuda threads=threads blocks=blocks get_particle_id!(pid,
                                                             hash_table_current,
                                                             hid)

        # Detect collision
        fill!(cp_range, 0)
        @cuda threads=threads blocks=blocks search_hash_table!(cp_range,
                                                               hash_table,
                                                               hash_table_current,
                                                               neighbor_cells,
                                                               pid)
        accumulate!(+, cp_range_current, cp_range)
        total = CUDA.@allowscalar cp_range_current[end]
        cap = length(cp_list)
        while cap < total
            cap <<= 1
        end
        if cap > length(cp_list)
            resize!(cp_list, cap)
            @info "Resize CP List: $(length(cp_list))"
        end

        @cuda threads=threads blocks=blocks update_cp_list!(cp_list,
                                                            cp_range_current,
                                                            hash_table,
                                                            hash_table_current,
                                                            neighbor_cells,
                                                            pid)

        # Resolve collision
        if init
            @cuda threads=threads blocks=cld(total, threads) init_bonds!(contacts,
                                                                         contact_active,
                                                                         contact_bonded,
                                                                         contact_count,
                                                                         cp_list,
                                                                         total,
                                                                         grains,
                                                                         max_coordinate_number)
        else
            @cuda threads=threads blocks=cld(total, threads) resolve_collision!(contacts,
                                                                                contact_active,
                                                                                contact_bonded,
                                                                                contact_count,
                                                                                forces,
                                                                                moments,
                                                                                cp_list,
                                                                                total,
                                                                                grains,
                                                                                materials,
                                                                                surfaces,
                                                                                max_coordinate_number,
                                                                                dt,
                                                                                tolerance)
        end
    end

    function resolve_wall()
        @cuda threads=threads blocks=blocks resolve_wall!(wall_contacts,
                                                          forces,
                                                          moments,
                                                          grains,
                                                          walls,
                                                          nwall,
                                                          materials,
                                                          surfaces,
                                                          dt,
                                                          tolerance)
    end

    function apply_body_force()
        @cuda threads=threads blocks=blocks apply_body_force!(forces,
                                                              moments,
                                                              grains,
                                                              gravity,
                                                              global_damping)
    end

    function update()
        @cuda threads=threads blocks=blocks update!(grains, forces, moments, dt)
    end

    function late_clear_state()
        @cuda threads=threads blocks=blocks remove_inactive_contact!(contacts,
                                                                     contact_active,
                                                                     contact_bonded,
                                                                     contact_count,
                                                                     grains,
                                                                     max_coordinate_number)
    end

    function simulate(; init = false)
        clear_state()
        contact(init = init)
        resolve_wall()
        apply_body_force()
        update()
        late_clear_state()
    end

    step = 0
    simulate(init = true)

    if save_information
        p4p = open("output.p4p", "w")
        p4c = open("output.p4c", "w")
        save_single(grains, contacts, contact_active, contact_bonded, p4p, p4c, 0.0)
    end

    # # Debug use
    # total_steps = 100
    # save_per_steps = 100
    while step < total_steps
        t₁ = time_ns()
        for _ in 1:save_per_steps
            step += 1
            simulate()
        end
        t₂ = time_ns()

        Δt = (t₂ - t₁) * 1e-9
        @info "Solving..." step Δt

        if save_snapshot
            snapshot(grains, step)
        end

        if save_information
            save_single(grains, contacts, contact_active, contact_bonded, p4p, p4c,
                        step * dt)
        end
    end

    if save_information
        close(p4p)
        close(p4c)
    end

    @info "Total time: $((time_ns() - start_time) * 1e-9) s"
end
