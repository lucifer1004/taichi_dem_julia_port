function main(cfg_filename)
    cfg = from_toml(DEMConfig{Int32, Float64}, cfg_filename)
    (; domain_min, domain_max, init_particles, particle_contact_radius_multiplier, collision_pair_init_capacity_factor, dt, target_time, saving_interval_time) = cfg

    lines = split(read(init_particles, String), "\n")
    
    n = parse(Int, split(lines[2])[2]) # $timestamp $particles
    id = zeros(Int, n)
    mid = zeros(Int, n)
    ρ = zeros(n)
    r = zeros(n)
    V = zeros(n)
    m = zeros(n)
    k = zeros(Vec3, n)
    v = zeros(Vec3, n)
    I = zeros(Mat3x3, n)

    for i in 1:n
        id[i], mid[i], V[i], m[i], k_x, k_y, k_z, v_x, v_y, v_z = parse.(Float64, split(lines[i + 3]))
        ρ[i] = m[i] / V[i]
        k[i] = @SVector [k_x, k_y, k_z]
        v[i] = @SVector [v_x, v_y, v_z]
        r[i] = ∛(V[i] / (4 / 3 * π))
        I[i] = @SMatrix [
            2 / 5 * m[i] * r[i]^2 0 0
            0 2 / 5 * m[i] * r[i]^2 0
            0 0 2 / 5 * m[i] * r[i]^2
        ]
    end

    rₘₐₓ = maximum(r)
    id # Particle ID
    mid # Material ID
    r = cu(r) # Radius
    r₀ = r .* particle_contact_radius_multiplier # Contact radius
    
    # Translational attributes, all in GLOBAL coordinates
    k = cu(k) # Position
    kid = CUDA.zeros(Vec3i, n) # Position id
    hid = CUDA.zeros(Int32, n) # Position hash
    kcenter = CUDA.zeros(Vec3, n) # Position center
    neighbor_cells = CUDA.zeros(Vec8i, n) # Position neighbor cells
    v = cu(v) # Velocity
    a = CUDA.zeros(Vec3, n) # Acceleration
    F = CUDA.zeros(Vec3, n) # Total force

    # Rotational attributes, all in GLOBAL coordinates
    q = cu(zeros(Quaternion{Float64}, n)) # Quarternion
    ω = CUDA.zeros(Vec3, n) # Angular velocity
    dω = CUDA.zeros(Vec3, n) # Angular acceleration
    I = cu(I) # Moment of intertia
    τ = CUDA.zeros(Vec3, n) # Total moment (including torque)

    total_steps = trunc(Int, target_time / dt)
    save_per_steps = trunc(Int, saving_interval_time / dt)

    cell_size = 4rₘₐₓ
    hash_table_size = clamp(prod(ceil.(Int32, (domain_max - domain_min) / cell_size)), 1 << 20, 1 << 22)
    hash_table = CUDA.zeros(Int32, hash_table_size)
    hash_table_current = similar(hash_table)
    hash_table_prefix_sum = similar(hash_table)
    
    pid = CUDA.zeros(Int32, n)
    cp_list = CUDA.zeros(Int32, n * collision_pair_init_capacity_factor)
    cp_range = CUDA.zeros(Int32, n)
    cp_range_current = similar(cp_range)
    cp_range_prefix_sum = similar(cp_range)

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
        dijk = @SVector [o[i] > c[i] ? 1 : -1 for i in 1:3]
        return @SVector [
            hashcode(ijk),
            hashcode(ijk + @SVector([dijk[1], 0, 0])),
            hashcode(ijk + @SVector([0, dijk[2], 0])),
            hashcode(ijk + @SVector([0, 0, dijk[3]])),
            hashcode(ijk + @SVector([dijk[1], dijk[2], 0])),
            hashcode(ijk + @SVector([dijk[1], 0, dijk[3]])),
            hashcode(ijk + @SVector([0, dijk[2], dijk[3]])),
            hashcode(ijk + @SVector([dijk[1], dijk[2], dijk[3]]))
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
                for k in hash_table_current[idx]+1:hash_table_current[idx]+hash_table[idx]
                    j = pid[k]
                    if i < j
                        CUDA.atomic_add!(pointer(cp_range, i), Int32(1))
                    end
                end
            end
        end
    end

    function update_cp_list!(cp_list, cp_range_current, hash_table, hash_table_current, neighbors, pid)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        for i = index:stride:length(pid)
            for idx in neighbors[i]
                for k in hash_table_current[idx]+1:hash_table_current[idx]+hash_table[idx]
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

    function resolve(i, j)
    end

    function resolve_collision(cp_list, cp_range, cp_range_current)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        for i = index:stride:length(pid)
            for k in cp_range_current[i]+1:cp_range_current[i]+cp_range[i]
                j = cp_list[k]
                resolve(i, j)
            end
        end
    end

    function contact()
        map!(pos -> cell(pos, domain_min, cell_size), kid, k)
        map!(hashcode, hid, kid)
        map!(ijk -> center(ijk, domain_min, cell_size), kcenter, kid)

        # Setup hash table
        fill!(hash_table, 0)
        @cuda threads=threads blocks=blocks count_particles!(hash_table, hid)
        accumulate!(+, hash_table_prefix_sum, hash_table)
        copyto!(hash_table_current, hash_table_prefix_sum)
        @cuda threads=threads blocks=blocks get_particle_id!(pid, hash_table_current, hid)

        # Detect collision
        fill!(cp_range, 0)
        map!(neighbors, neighbor_cells, k, kid, kcenter)
        @cuda threads=threads blocks=blocks search_hash_table!(cp_range, hash_table, hash_table_current, neighbor_cells, pid)
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

        @cuda threads=threads blocks=blocks update_cp_list!(cp_list, cp_range_current, hash_table, hash_table_current, neighbor_cells, pid)

        # Resolve collision
        @cuda threads=threads blocks=blocks resolve_collision(cp_list, cp_range, cp_range_current)
    end

    function resolve_wall()
    end

    function apply_body_force()
    end

    function update()
    end

    function late_clear_state()
    end

    function simulate()
        clear_state()
        contact()
        resolve_wall()
        apply_body_force()
        update()
        late_clear_state()
    end

    function save_single(p4p, p4c, t)
        t₁ = time_ns()

        println(p4p, "TIMESTEP PARTICLES")
        println(p4p, "$t $n")
        println(p4p, "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ")

        k_cpu = Array(k)
        v_cpu = Array(v)

        for i in 1:n
            println(p4p, "$(id[i]) $(mid[i]) $(V[i]) $(m[i]) $(k_cpu[i][1]) $(k_cpu[i][2]) $(k_cpu[i][3]) $(v_cpu[i][1]) $(v_cpu[i][2]) $(v_cpu[i][3])")
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
    save_single(p4p, p4c, 0.0)
    while step < total_steps
        t₁ = time_ns()
        for _ in 1:save_per_steps
            step += 1
            simulate()
        end
        t₂ = time_ns()

        println(">>>")
        println("solved steps: $step last-$(save_per_steps)-sim: $((t₂ - t₁) * 1e-9)s")
        save_single(p4p, p4c, step * dt)
    end

    close(p4p)
    close(p4c)
end
