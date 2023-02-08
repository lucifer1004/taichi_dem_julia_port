function atomic_add_vec3!(container, start, value)
    CUDA.atomic_add!(pointer(container, start), value[1])
    CUDA.atomic_add!(pointer(container, start + 1), value[2])
    CUDA.atomic_add!(pointer(container, start + 2), value[3])
end

function count_particles!(hash_table, hid)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(hid)
        CUDA.atomic_add!(pointer(hash_table, hid[i]), Int32(1))
    end
end

function get_particle_id!(pid, hash_table_current, hid)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(hid)
        id = CUDA.atomic_add!(pointer(hash_table_current, hid[i]), Int32(-1))
        pid[id] = i
    end
end

function search_hash_table!(cp_range, hash_table, hash_table_current, neighbors, pid)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(pid)
        for idx in neighbors[i]
            for k in (hash_table_current[idx] + 1):(hash_table_current[idx] + hash_table[idx])
                j = pid[k]
                if i < j
                    CUDA.atomic_add!(pointer(cp_range, i), Int32(1))
                end
            end
        end
    end
end

function update_cp_list!(cp_list,
                         cp_range_current,
                         hash_table,
                         hash_table_current,
                         neighbors,
                         pid)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(pid)
        for idx in neighbors[i]
            for k in (hash_table_current[idx] + 1):(hash_table_current[idx] + hash_table[idx])
                j = pid[k]
                if i < j
                    current = CUDA.atomic_add!(pointer(cp_range_current, i), Int32(-1))
                    cp_list[current] = Vec2i(i, j)
                end
            end
        end
    end
end

function init_bonds!(contacts,
                     contact_active,
                     contact_bonded,
                     contact_count,
                     cp_list,
                     total,
                     grains,
                     max_coordinate_number)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for idx in index:stride:total
        i, j = cp_list[idx]
        if norm(grains[i].𝐤 - grains[j].𝐤) < grains[i].r₀ + grains[j].r₀
            offset = CUDA.atomic_add!(pointer(contact_count, i), Int32(1)) + 1
            if offset <= max_coordinate_number
                offset += (i - 1) * max_coordinate_number
            else
                offset = 0
            end

            if offset > 0
                # FIXME: material type is hard-coded
                contacts[offset] = ContactDefault(i,
                                                  j,
                                                  1,
                                                  1,
                                                  zero(Vec3),
                                                  zero(Vec3),
                                                  zero(Vec3),
                                                  zero(Vec3),
                                                  zero(Vec3))
                contact_active[offset] = true
                contact_bonded[offset] = true
            end
        end
    end
end

function resolve_collision!(contacts,
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
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for idx in index:stride:total
        i, j = cp_list[idx]
        ev = false
        offset = 0

        # TODO: Find a better way to do this
        for k in (i - 1) * max_coordinate_number .+ (1:contact_count[i])
            if contacts[k].j == j && contact_active[k]
                offset = k
                break
            end
        end

        if offset > 0
            if contact_bonded[offset] ||
               norm(grains[i].𝐤 - grains[j].𝐤) < grains[i].r + grains[j].r
                ev = true
            else
                contact_active[offset] = false
            end
        elseif norm(grains[i].𝐤 - grains[j].𝐤) < grains[i].r + grains[j].r
            offset = CUDA.atomic_add!(pointer(contact_count, i), Int32(1)) + 1
            if offset <= max_coordinate_number
                offset += (i - 1) * max_coordinate_number
            else
                offset = 0
            end

            if offset > 0
                # FIXME: material type is hard-coded
                contacts[offset] = ContactDefault(i,
                                                  j,
                                                  1,
                                                  1,
                                                  zero(Vec3),
                                                  zero(Vec3),
                                                  zero(Vec3),
                                                  zero(Vec3),
                                                  zero(Vec3))
                contact_active[offset] = true
                contact_bonded[offset] = false
                ev = true
            end
        end

        if ev
            a = normalize(grains[j].𝐤 - grains[i].𝐤)
            b = @SVector [1.0, 0.0, 0.0] # Local x coordinate
            v = a × b
            s = norm(v)
            c = a ⋅ b

            if s < tolerance
                sign = c > 0.0 ? 1.0 : -1.0
                𝐑 = @SMatrix [sign 0.0 0.0
                              0.0 1.0 0.0
                              0.0 0.0 sign]
            else
                vx = @SMatrix [0.0 -v[3] v[2]
                               v[3] 0.0 -v[1]
                               -v[2] v[1] 0.0]
                𝐑 = @SMatrix([1.0 0.0 0.0
                              0.0 1.0 0.0
                              0.0 0.0 1.0]) +
                    vx +
                    vx^2 * (1.0 - c) / s^2
            end

            Lᵢ = norm(grains[j].𝐤 - grains[i].𝐤)

            # Contact evaluation (with contact model)
            if contact_bonded[offset]
                𝐤 = (grains[i].𝐤 + grains[j].𝐤) * 0.5
                𝐝ᵢ = 𝐑 * grains[i].𝐯 * dt
                𝐝ⱼ = 𝐑 * grains[j].𝐯 * dt
                𝛉ᵢ = 𝐑 * grains[i].𝛚 * dt
                𝛉ⱼ = 𝐑 * grains[j].𝛚 * dt
                midᵢ = grains[i].mid
                midⱼ = grains[j].mid
                rⱼ = surfaces[midᵢ, midⱼ].ρ * min(grains[i].r, grains[j].r)
                Lⱼ = Lᵢ
                Eⱼ = surfaces[midᵢ, midⱼ].E
                ν = surfaces[midᵢ, midⱼ].ν
                Iⱼ = rⱼ^4 * π / 4
                ϕ = 20.0 / 3.0 * rⱼ^2 / Lⱼ^2 * (1.0 + ν)
                Aⱼ = rⱼ^2 * π
                k₁ = Eⱼ * Aⱼ / Lⱼ
                k₂ = 12.0 * Eⱼ * Iⱼ / Lⱼ^3 / (1.0 + ϕ)
                k₃ = 6.0 * Eⱼ * Iⱼ / Lⱼ^2 / (1.0 + ϕ)
                k₄ = Eⱼ * Iⱼ / Lⱼ / (1.0 + ν)
                k₅ = Eⱼ * Iⱼ * (4.0 + ϕ) / Lⱼ / (1.0 + ϕ)
                k₆ = Eⱼ * Iⱼ * (2.0 - ϕ) / Lⱼ / (1.0 + ϕ)

                Δ𝐅ᵢ = Vec3(k₁ * (𝐝ᵢ[1] - 𝐝ⱼ[1]),
                           k₂ * (𝐝ᵢ[2] - 𝐝ⱼ[2]) + k₃ * (𝛉ᵢ[3] + 𝛉ⱼ[3]),
                           k₂ * (𝐝ᵢ[3] - 𝐝ⱼ[3]) - k₃ * (𝛉ᵢ[2] + 𝛉ⱼ[2]))
                Δ𝛕ᵢ = Vec3(k₄ * (𝛉ᵢ[1] - 𝛉ⱼ[1]),
                           k₃ * (𝐝ⱼ[3] - 𝐝ᵢ[3]) + k₅ * 𝛉ᵢ[2] + k₆ * 𝛉ⱼ[2],
                           k₃ * (𝐝ᵢ[2] - 𝐝ⱼ[2]) + k₅ * 𝛉ᵢ[3] + k₆ * 𝛉ⱼ[3])
                Δ𝛕ⱼ = Vec3(k₄ * (𝛉ⱼ[1] - 𝛉ᵢ[1]),
                           k₃ * (𝐝ⱼ[3] - 𝐝ᵢ[3]) + k₆ * 𝛉ᵢ[2] + k₅ * 𝛉ⱼ[2],
                           k₃ * (𝐝ᵢ[2] - 𝐝ⱼ[2]) + k₆ * 𝛉ᵢ[3] + k₅ * 𝛉ⱼ[3])

                𝐅ᵢ = contacts[offset].𝐅ᵢ + Δ𝐅ᵢ
                𝐅ⱼ = -𝐅ᵢ
                𝛕ᵢ = contacts[offset].𝛕ᵢ + Δ𝛕ᵢ
                𝛕ⱼ = contacts[offset].𝛕ⱼ + Δ𝛕ⱼ

                # TODO: should it be 𝐅ᵢ[1]?
                σ𝑐ᵢ = 𝐅ⱼ[1] / Aⱼ - rⱼ / Iⱼ * √(𝛕ᵢ[2]^2 + 𝛕ᵢ[3]^2)
                σ𝑐ⱼ = 𝐅ⱼ[1] / Aⱼ - rⱼ / Iⱼ * √(𝛕ⱼ[2]^2 + 𝛕ⱼ[3]^2)
                σ𝑐 = -min(σ𝑐ᵢ, σ𝑐ⱼ)

                σ𝑡ᵢ = σ𝑐ᵢ
                σ𝑡ⱼ = σ𝑐ⱼ
                σ𝑡 = max(σ𝑡ᵢ, σ𝑡ⱼ)

                σ𝑠 = abs(𝛕ᵢ[1]) * rⱼ / 2.0 / Iⱼ + 4.0 / 3.0 / Aⱼ * √(𝐅ᵢ[2]^2 + 𝐅ᵢ[3]^2)
                if σ𝑐 >= surfaces[midᵢ, midⱼ].σ𝑐 ||
                   σ𝑡 >= surfaces[midᵢ, midⱼ].σ𝑡 ||
                   σ𝑠 >= surfaces[midᵢ, midⱼ].σ𝑠
                    contact_active[offset] = false
                    contact_bonded[offset] = false
                else
                    𝐑⁻¹ = inv(𝐑)
                    atomic_add_vec3!(forces, 3 * i - 2, 𝐑⁻¹ * -𝐅ᵢ)
                    atomic_add_vec3!(forces, 3 * j - 2, 𝐑⁻¹ * -𝐅ⱼ)
                    atomic_add_vec3!(moments, 3 * i - 2, 𝐑⁻¹ * -𝛕ᵢ)
                    atomic_add_vec3!(moments, 3 * j - 2, 𝐑⁻¹ * -𝛕ⱼ)
                end

                contacts[offset] = ContactDefault(contacts[offset].i,
                                                  contacts[offset].j,
                                                  contacts[offset].midᵢ,
                                                  contacts[offset].midⱼ,
                                                  𝐤,
                                                  𝐅ᵢ,
                                                  𝛕ᵢ,
                                                  𝛕ⱼ,
                                                  zero(Vec3))
            else # Non-bonded, use Hertz-Mindlin
                gap = Lᵢ - grains[i].r - grains[j].r # gap must be negative to ensure an intact contact
                Δn = abs(gap)
                𝐤 = grains[i].𝐤 + normalize(grains[j].𝐤 - grains[i].𝐤) * (grains[i].r - Δn)
                𝐤ᵢ = 𝐤 - grains[i].𝐤
                𝐤ⱼ = 𝐤 - grains[j].𝐤
                𝐯𝑐ᵢ = grains[i].𝛚 × 𝐤ᵢ + grains[i].𝐯
                𝐯𝑐ⱼ = grains[j].𝛚 × 𝐤ⱼ + grains[j].𝐯
                𝐯𝑐 = 𝐑 * (𝐯𝑐ⱼ - 𝐯𝑐ᵢ)

                midᵢ = grains[i].mid
                midⱼ = grains[j].mid
                νᵢ = materials[midᵢ].ν
                Eᵢ = materials[midᵢ].E
                νⱼ = materials[midⱼ].ν
                Eⱼ = materials[midⱼ].E
                Y✶ = 1.0 / ((1.0 - νᵢ^2) / Eᵢ + (1.0 - νⱼ^2) / Eⱼ)
                G✶ = 1.0 / (2.0 * (2.0 - νᵢ) * (1.0 + νᵢ) / Eᵢ +
                      2.0 * (2.0 - νⱼ) * (1.0 + νⱼ) / Eⱼ)
                R✶ = 1.0 / (1.0 / grains[i].r + 1.0 / grains[j].r)
                m✶ = 1.0 / (1.0 / grains[i].m + 1.0 / grains[j].m)
                β = log(surfaces[midᵢ, midⱼ].e) / √(log(surfaces[midᵢ, midⱼ].e)^2 + π^2)
                Sₙ = 2.0 * Y✶ * √(R✶ * Δn)
                Sₜ = 8.0 * G✶ * √(R✶ * Δn)
                kₙ = 4.0 / 3.0 * Y✶ * √(R✶ * Δn)

                # TODO: Check whether gamma_n >= 0
                γₙ = -2.0 * β * √(5.0 / 6.0 * Sₙ * m✶)
                kₜ = 8.0 * G✶ * √(R✶ * Δn)

                # TODO: Check whether gamma_t >= 0
                γₜ = -2.0 * β * √(5.0 / 6.0 * Sₜ * m✶)

                # Shear displacement increments (remove the normal direction)
                Δ𝐬 = 𝐯𝑐 .* Vec3(0.0, dt, dt)
                𝐬 = contacts[offset].𝐬 + Δ𝐬
                F₁ = -kₙ * gap - γₙ * 𝐯𝑐[1]
                𝐅𝑡 = -kₜ * 𝐬

                if norm(𝐅𝑡) >= surfaces[midᵢ, midⱼ].μ * F₁ # Sliding
                    ratio = surfaces[midᵢ, midⱼ].μ * F₁ / norm(𝐅𝑡)
                    F₂ = ratio * 𝐅𝑡[2]
                    F₃ = ratio * 𝐅𝑡[3]
                    𝐬 = Vec3(𝐬[1], F₂ / kₜ, F₃ / kₜ)
                else # No sliding
                    F₂ = 𝐅𝑡[2] - γₜ * 𝐯𝑐[2]
                    F₃ = 𝐅𝑡[3] - γₜ * 𝐯𝑐[3]
                end

                𝐅ᵢ = Vec3(F₁, F₂, F₃)
                𝐅ᵢ𝑔 = inv(𝐑) * -𝐅ᵢ
                atomic_add_vec3!(forces, 3 * i - 2, 𝐅ᵢ𝑔)
                atomic_add_vec3!(forces, 3 * j - 2, -𝐅ᵢ𝑔)
                atomic_add_vec3!(moments, 3 * i - 2, 𝐤ᵢ × 𝐅ᵢ𝑔)
                atomic_add_vec3!(moments, 3 * j - 2, 𝐤ⱼ × -𝐅ᵢ𝑔)

                contacts[offset] = ContactDefault(contacts[offset].i,
                                                  contacts[offset].j,
                                                  contacts[offset].midᵢ,
                                                  contacts[offset].midⱼ,
                                                  𝐤,
                                                  𝐅ᵢ,
                                                  contacts[offset].𝛕ᵢ,
                                                  contacts[offset].𝛕ⱼ,
                                                  𝐬)
            end
        end
    end
end

function resolve_wall!(wall_contacts,
                       forces,
                       moments,
                       grains,
                       walls,
                       nwall,
                       materials,
                       surfaces,
                       dt,
                       tolerance)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(grains)
        for j in 1:nwall
            if abs(grains[i].𝐤 ⋅ walls[j].𝐧 - walls[j].d) < grains[i].r
                a = walls[j].𝐧
                b = @SVector [1.0, 0.0, 0.0] # Local x coordinate
                v = a × b
                s = norm(v)
                c = a ⋅ b

                if s < tolerance
                    sign = c > 0.0 ? 1.0 : -1.0
                    𝐑 = @SMatrix [sign 0.0 0.0
                                  0.0 1.0 0.0
                                  0.0 0.0 sign]
                else
                    vx = @SMatrix [0.0 -v[3] v[2]
                                   v[3] 0.0 -v[1]
                                   -v[2] v[1] 0.0]
                    𝐑 = @SMatrix([1.0 0.0 0.0
                                  0.0 1.0 0.0
                                  0.0 0.0 1.0]) +
                        vx +
                        vx^2 * (1.0 - c) / s^2
                end

                L = grains[i].𝐤 ⋅ walls[j].𝐧 - walls[j].d
                gap = abs(L) - grains[i].r
                Δn = abs(gap)

                𝐤ᵢ = -L * walls[j].𝐧 / abs(L) * (abs(L) + Δn / 2.0)
                𝐤 = grains[i].𝐤 + 𝐤ᵢ
                𝐯𝑐ᵢ = grains[i].𝛚 × 𝐤ᵢ + grains[i].𝐯
                𝐯𝑐 = 𝐑 * -𝐯𝑐ᵢ

                midᵢ = grains[i].mid
                midⱼ = walls[j].mid
                νᵢ = materials[midᵢ].ν
                Eᵢ = materials[midᵢ].E
                νⱼ = materials[midⱼ].ν
                Eⱼ = materials[midⱼ].E
                Y✶ = 1.0 / ((1.0 - νᵢ^2) / Eᵢ + (1.0 - νⱼ^2) / Eⱼ)
                G✶ = 1.0 / (2.0 * (2.0 - νᵢ) * (1.0 + νᵢ) / Eᵢ +
                      2.0 * (2.0 - νⱼ) * (1.0 + νⱼ) / Eⱼ)
                R✶ = grains[i].r
                m✶ = grains[i].m
                β = log(surfaces[midᵢ, midⱼ].e) / √(log(surfaces[midᵢ, midⱼ].e)^2 + π^2)
                Sₙ = 2.0 * Y✶ * √(R✶ * Δn)
                Sₜ = 8.0 * G✶ * √(R✶ * Δn)
                kₙ = 4.0 / 3.0 * Y✶ * √(R✶ * Δn)

                # TODO: Check whether gamma_n >= 0
                γₙ = -2.0 * β * √(5.0 / 6.0 * Sₙ * m✶)
                kₜ = 8.0 * G✶ * √(R✶ * Δn)

                # TODO: Check whether gamma_t >= 0
                γₜ = -2.0 * β * √(5.0 / 6.0 * Sₜ * m✶)

                # Shear displacement increments (remove the normal direction)
                Δ𝐬 = 𝐯𝑐 .* Vec3(0.0, dt, dt)
                𝐬 = wall_contacts[j, i].𝐬 + Δ𝐬
                F₁ = -kₙ * gap - γₙ * 𝐯𝑐[1]
                𝐅𝑡 = -kₜ * 𝐬
                if norm(𝐅𝑡) >= surfaces[midᵢ, midⱼ].μ * F₁ # Sliding
                    ratio = surfaces[midᵢ, midⱼ].μ * F₁ / norm(𝐅𝑡)
                    F₂ = ratio * 𝐅𝑡[2]
                    F₃ = ratio * 𝐅𝑡[3]
                    𝐬 = Vec3(𝐬[1], F₂ / kₜ, F₃ / kₜ)
                else # No sliding
                    F₂ = 𝐅𝑡[2] - γₜ * 𝐯𝑐[2]
                    F₃ = 𝐅𝑡[3] - γₜ * 𝐯𝑐[3]
                end

                𝐅ᵢ = Vec3(F₁, F₂, F₃)
                𝐅ᵢ𝑔 = inv(𝐑) * -𝐅ᵢ
                atomic_add_vec3!(forces, 3 * i - 2, 𝐅ᵢ𝑔)
                atomic_add_vec3!(moments, 3 * i - 2, 𝐤ᵢ × 𝐅ᵢ𝑔)

                wall_contacts[j, i] = ContactDefault(i,
                                                     0,
                                                     midᵢ,
                                                     midⱼ,
                                                     𝐤,
                                                     𝐅ᵢ,
                                                     wall_contacts[j, i].𝛕ᵢ,
                                                     wall_contacts[j, i].𝛕ⱼ,
                                                     𝐬)
            end
        end
    end
end

function apply_body_force!(forces, moments, grains, gravity, global_damping)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(grains)
        # Add gravity
        atomic_add_vec3!(forces, 3 * i - 2, gravity * grains[i].m)

        # # Disable due to zero global damping
        # 𝐅 = Vec3(forces[3 * i - 2], forces[3 * i - 1], forces[3 * i])
        # 𝛕 = Vec3(moments[3 * i - 2], moments[3 * i - 1], moments[3 * i])
        # Δ𝐅 = @. -global_damping * abs(𝐅) * sign(grains[i].𝐯)
        # Δ𝛕 = @. -global_damping * abs(𝛕) * sign(grains[i].𝛕)
        # atomic_add_vec3!(forces, 3 * i - 2, Δ𝐅)
        # atomic_add_vec3!(moments, 3 * i - 2, Δ𝛕)
    end
end

function update!(grains, forces, moments, dt)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(grains)
        𝐅 = Vec3(forces[3 * i - 2], forces[3 * i - 1], forces[3 * i])
        𝐚 = 𝐅 / grains[i].m
        𝐤 = grains[i].𝐤 + grains[i].𝐯 * dt + 0.5 * 𝐚 * dt^2
        𝐯 = grains[i].𝐯 + 𝐚 * dt

        𝐪 = grains[i].𝐪
        𝛚 = grains[i].𝛚
        𝐑 = to_rotation_matrix(𝐪)
        𝛕 = Vec3(moments[3 * i - 2], moments[3 * i - 1], moments[3 * i])
        𝛕𝑙 = 𝐑 * 𝛕 # Local angular moment
        𝛚𝑙 = 𝐑 * 𝛚 # Local angular velocity
        d𝛚𝑙 = inv(grains[i].𝐈) * (𝛕𝑙 - 𝛚𝑙 × (grains[i].𝐈 * 𝛚𝑙)) # Local angular acceleration
        d𝛚 = inv(𝐑) * d𝛚𝑙 # Global angular acceleration

        Δq₁ = -0.5 * (𝐪[2] * 𝛚[1] + 𝐪[3] * 𝛚[2] + 𝐪[4] * 𝛚[3])
        Δq₂ = 0.5 * (𝐪[1] * 𝛚[1] + 𝐪[3] * 𝛚[3] - 𝐪[4] * 𝛚[2])
        Δq₃ = 0.5 * (𝐪[1] * 𝛚[2] + 𝐪[2] * 𝛚[3] + 𝐪[4] * 𝛚[1])
        Δq₄ = 0.5 * (𝐪[1] * 𝛚[3] + 𝐪[2] * 𝛚[2] - 𝐪[3] * 𝛚[1])
        𝐪 = normalize(𝐪 + Quaternion(Δq₁, Δq₂, Δq₃, Δq₄))
        𝛚 += d𝛚 * dt

        grains[i] = GrainDefault(grains[i].id,
                                 grains[i].gid,
                                 grains[i].mid,
                                 grains[i].V,
                                 grains[i].m,
                                 grains[i].r,
                                 grains[i].r₀,
                                 𝐤,
                                 𝐯,
                                 𝐚,
                                 𝐪,
                                 𝛚,
                                 d𝛚,
                                 grains[i].𝐈)
    end
end

function remove_inactive_contact!(contacts,
                                  contact_active,
                                  contact_bonded,
                                  contact_count,
                                  grains,
                                  max_coordinate_number)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(grains)
        active_count = 0
        base = (i - 1) * max_coordinate_number
        for j in 1:contact_count[i]
            if contact_active[base + j]
                active_count += 1
            end
        end

        offset = 1
        for j in 1:contact_count[i]
            if contact_active[base + j]
                contact_active[base + offset] = true
                contact_bonded[base + offset] = contact_bonded[base + j]
                contacts[base + offset] = contacts[base + j]
                offset += 1
                if offset > active_count
                    break
                end
            end
        end

        for j in (active_count + 1):contact_count[i]
            contact_active[(i - 1) * max_coordinate_number + j] = false
        end
        contact_count[i] = active_count
    end
end
