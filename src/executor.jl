mutable struct GlobalData
    grains::CuArray{GrainDefault, 1, CUDA.Mem.DeviceBuffer}
    kid::CuArray{Vec3i, 1, CUDA.Mem.DeviceBuffer}
    hid::CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}
    kcenter::CuArray{Vec3, 1, CUDA.Mem.DeviceBuffer}
    neighbor_cells::CuArray{Vec8i, 1, CUDA.Mem.DeviceBuffer}

    materials::CuArray{MaterialDefault, 1, CUDA.Mem.DeviceBuffer}
    surfaces::CuArray{SurfaceDefault, 2, CUDA.Mem.DeviceBuffer}
    walls::CuArray{WallDefault, 1, CUDA.Mem.DeviceBuffer}
    wall_contacts::CuArray{ContactDefault, 2, CUDA.Mem.DeviceBuffer}

    forces::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    moments::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}

    hash_table::CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}
    hash_table_current::CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}
    pid::CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}

    cp_list::CuArray{Vec2i, 1, CUDA.Mem.DeviceBuffer}
    cp_range::CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}
    cp_range_current::CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}

    contacts::CuArray{ContactDefault, 1, CUDA.Mem.DeviceBuffer}
    contact_ptr::CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}
    contacts_temp::CuArray{ContactDefault, 1, CUDA.Mem.DeviceBuffer}
    contact_active::CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}
    contact_bonded::CuArray{UInt32, 1, CUDA.Mem.DeviceBuffer}
end

function clear_state!(g::GlobalData)
    fill!(g.forces, 0)
    fill!(g.moments, 0)
end

function contact!(g::GlobalData, threads; domain_min, cell_size, hash_table_size, dt,
                  tolerance, init = false)
    # Precalculate cell ijk, hash code, center and neighbors
    map!(g -> cell(g.𝐤, domain_min, cell_size), g.kid, g.grains)
    map!(ijk -> hashcode(ijk, hash_table_size), g.hid, g.kid)
    map!(ijk -> center(ijk, domain_min, cell_size), g.kcenter, g.kid)
    map!((g, ijk, c) -> neighbors(g.𝐤, ijk, c, hash_table_size), g.neighbor_cells, g.grains,
         g.kid, g.kcenter)

    # Setup hash table
    fill!(g.hash_table, 0)
    n = length(g.grains)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks count_particles!(g.hash_table, g.hid)
    accumulate!(+, g.hash_table_current, g.hash_table)
    @cuda threads=threads blocks=blocks get_particle_id!(g.pid,
                                                         g.hash_table_current,
                                                         g.hid)

    # Detect collision
    fill!(g.cp_range, 0)
    @cuda threads=threads blocks=blocks search_hash_table!(g.cp_range,
                                                           g.hash_table,
                                                           g.hash_table_current,
                                                           g.neighbor_cells,
                                                           g.pid)
    accumulate!(+, g.cp_range_current, g.cp_range)
    total_cps = CUDA.@allowscalar g.cp_range_current[end]
    cap = length(g.cp_list)
    while cap < total_cps
        cap <<= 1
    end
    if cap > length(g.cp_list)
        resize!(g.cp_list, cap)
        @info "Resize contact-pair list to $(length(g.cp_list))"
    end

    @cuda threads=threads blocks=blocks update_cp_list!(g.cp_list,
                                                        g.cp_range_current,
                                                        g.hash_table,
                                                        g.hash_table_current,
                                                        g.neighbor_cells,
                                                        g.pid)

    if init
        @cuda threads=threads blocks=cld(total_cps, threads) init_bonds!(g.contacts,
                                                                         g.contact_ptr,
                                                                         g.contact_active,
                                                                         g.contact_bonded,
                                                                         g.cp_list,
                                                                         total_cps,
                                                                         g.grains)
    else
        @cuda threads=threads blocks=cld(total_cps, threads) update_contacts!(g.contacts,
                                                                              g.contact_ptr,
                                                                              g.contact_active,
                                                                              g.contact_bonded,
                                                                              g.cp_list,
                                                                              total_cps,
                                                                              g.grains)

        # Resolve collision
        total_contacts = CUDA.@allowscalar g.contact_ptr[1]
        if total_contacts > 0
            @cuda threads=threads blocks=cld(total_contacts, threads) resolve_collision!(g.contacts,
                                                                                         total_contacts,
                                                                                         g.contact_active,
                                                                                         g.contact_bonded,
                                                                                         g.forces,
                                                                                         g.moments,
                                                                                         g.grains,
                                                                                         g.materials,
                                                                                         g.surfaces,
                                                                                         dt,
                                                                                         tolerance)
        end
    end
end

function resolve_wall!(g::GlobalData, threads; dt, tolerance)
    @cuda threads=threads blocks=cld(length(g.grains), threads) resolve_wall!(g.wall_contacts,
                                                                              g.forces,
                                                                              g.moments,
                                                                              g.grains,
                                                                              g.walls,
                                                                              g.materials,
                                                                              g.surfaces,
                                                                              dt,
                                                                              tolerance)
end

function apply_body_force!(g::GlobalData, threads; gravity, global_damping)
    @cuda threads=threads blocks=cld(length(g.grains), threads) apply_body_force!(g.forces,
                                                                                  g.moments,
                                                                                  g.grains,
                                                                                  gravity,
                                                                                  global_damping)
end

function update!(g::GlobalData, threads; dt)
    @cuda threads=threads blocks=cld(length(g.grains), threads) update!(g.grains, g.forces,
                                                                        g.moments, dt)
end

function late_clear_state!(g::GlobalData, threads)
    total_contacts = CUDA.@allowscalar g.contact_ptr[1]

    if total_contacts > 0
        fill!(g.contact_ptr, 0)
        @cuda threads=threads blocks=cld(total_contacts, threads) remove_inactive_contact!(g.contacts,
                                                                                           g.contacts_temp,
                                                                                           g.contact_ptr,
                                                                                           g.contact_active,
                                                                                           total_contacts,
                                                                                           g.grains)

        @debug begin
            inactive = total_contacts - CUDA.@allowscalar g.contact_ptr[1]
            if !iszero(inactive)
                @show Int(inactive)
            end
        end

        g.contacts, g.contacts_temp = g.contacts_temp, g.contacts
    end
end

function initialize!(global_data::GlobalData, threads; domain_min, cell_size,
                     hash_table_size,
                     dt, tolerance)
    contact!(global_data, threads; domain_min = domain_min, cell_size = cell_size,
             hash_table_size = hash_table_size, dt = dt, tolerance = tolerance, init = true)
end

function simulate!(global_data::GlobalData, threads; domain_min, cell_size, hash_table_size,
                   dt, tolerance, gravity, global_damping)
    clear_state!(global_data)
    contact!(global_data, threads; domain_min = domain_min, cell_size = cell_size,
             hash_table_size = hash_table_size, dt = dt, tolerance = tolerance)
    resolve_wall!(global_data, threads; dt = dt, tolerance = tolerance)
    apply_body_force!(global_data, threads; gravity = gravity,
                      global_damping = global_damping)
    update!(global_data, threads; dt = dt)
    late_clear_state!(global_data, threads)
end

function solve(cfg_filename; save_snapshot = false, save_information = true)
    # Release memory in case there are leftovers from previous runs
    GC.gc(true)
    CUDA.reclaim()
    @info "Current memory usage:"
    CUDA.memory_status()

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

    # FIXME: To match baseline solution's wrong convention
    material_density = grains_cpu[end].m / grains_cpu[end].V
    map!(grains_cpu, grains_cpu) do g
        @set g.m = g.V * material_density
    end

    # Calculate neighbor search radius
    rₘₐₓ = maximum(g -> g.r, grains_cpu) *
           particle_contact_radius_multiplier *
           (1.0 + bond_tensile_strength / bond_elastic_modulus) *
           neighboring_search_safety_factor
    grains = cu(grains_cpu)

    # Translational attributes, all in GLOBAL coordinates
    kid = CUDA.zeros(Vec3i, n) # Position id
    hid = CUDA.zeros(UInt32, n) # Position hash
    kcenter = CUDA.zeros(Vec3, n) # Position center
    neighbor_cells = CUDA.zeros(Vec8i, n) # Position neighbor cells

    forces = CUDA.zeros(Float64, n * 3)
    moments = CUDA.zeros(Float64, n * 3)

    total_steps = trunc(Int, target_time / dt)
    save_per_steps = trunc(Int, saving_interval_time / dt)

    cell_size = 4rₘₐₓ
    hash_table_size = clamp(prod(ceil.(Int32, (domain_max - domain_min) / cell_size)),
                            1 << 20, 1 << 22)
    hash_table = CUDA.zeros(UInt32, hash_table_size)
    hash_table_current = similar(hash_table)

    pid = CUDA.zeros(UInt32, n)
    cp_list = CUDA.zeros(Vec2i, n * collision_pair_init_capacity_factor)
    cp_range = CUDA.zeros(UInt32, n)
    cp_range_current = similar(cp_range)

    contact_ptr = CUDA.zeros(UInt32, 1)
    contacts = CUDA.fill(ContactDefaultZero, n * max_coordinate_number)
    contacts_temp = CUDA.fill(ContactDefaultZero, n * max_coordinate_number)
    contact_active = CUDA.zeros(UInt32, (nextpow(2, n)^2) >> 5)
    contact_bonded = CUDA.zeros(UInt32, (nextpow(2, n)^2) >> 5)

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
    wall_contacts = CUDA.fill(ContactDefaultZero, nwall, n)

    threads = 256
    global_data = GlobalData(grains,
                             kid,
                             hid,
                             kcenter,
                             neighbor_cells,
                             materials,
                             surfaces,
                             walls,
                             wall_contacts,
                             forces,
                             moments,
                             hash_table,
                             hash_table_current,
                             pid,
                             cp_list,
                             cp_range,
                             cp_range_current,
                             contacts,
                             contact_ptr,
                             contacts_temp,
                             contact_active,
                             contact_bonded)

    @info "Setting:" hash_table_size cell_size

    step = 0
    initialize!(global_data, threads; domain_min = domain_min, cell_size = cell_size,
                hash_table_size = hash_table_size, dt = dt, tolerance = tolerance)

    if save_information
        p4p = open("output.p4p", "w")
        p4c = open("output.p4c", "w")
        total_contacts = CUDA.@allowscalar global_data.contact_ptr[1]
        save_single(grains, global_data.contacts, total_contacts, contact_active,
                    contact_bonded, p4p,
                    p4c, 0.0)
    end

    # # Debug use
    # total_steps = 100
    # save_per_steps = 100
    while step < total_steps
        t₁ = time_ns()
        for _ in 1:save_per_steps
            step += 1
            simulate!(global_data, threads; domain_min = domain_min, cell_size = cell_size,
                      hash_table_size = hash_table_size, dt = dt, tolerance = tolerance,
                      gravity = gravity, global_damping = global_damping)
        end
        t₂ = time_ns()

        Δt = (t₂ - t₁) * 1e-9
        @info "Solving..." step Δt

        if save_snapshot
            snapshot(grains, step)
        end

        if save_information
            total_contacts = CUDA.@allowscalar global_data.contact_ptr[1]
            save_single(grains, global_data.contacts, total_contacts, contact_active,
                        contact_bonded,
                        p4p, p4c, step * dt)
        end
    end

    if save_information
        close(p4p)
        close(p4c)
    end

    @info "Total time: $((time_ns() - start_time) * 1e-9) s"
end
