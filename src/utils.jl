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

function save_single(grains, contacts, total_contacts, contact_active, contact_bonded, p4p,
                     p4c, t)
    @info "Start saving..."
    tâ = time_ns()
    n = length(grains)

    println(p4p, "TIMESTEP PARTICLES")
    println(p4p, "$t $n")
    println(p4p, "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ")

    g = Array(grains)
    for i in 1:n
        println(p4p,
                "$(g[i].id) $(g[i].gid) $(g[i].V) $(g[i].m) $(g[i].ğ¤[1]) $(g[i].ğ¤[2]) $(g[i].ğ¤[3]) $(g[i].ğ¯[1]) $(g[i].ğ¯[2]) $(g[i].ğ¯[3])")
    end

    println(p4c, "TIMESTEP CONTACTS")

    c = Array(contacts[1:total_contacts])
    ca = Array(contact_active)
    cb = Array(contact_bonded)
    cache = String[]
    s = Set()
    for contact in c
        i = contact.i
        j = contact.j
        ij = UInt64(i - 1) * n + j
        if get_bit(ca, ij)
            if (i, j) in s
                error("Duplicate contact pairs!")
            end
            push!(s, (i, j))

            bonded = get_bit(cb, ij)
            ğ¤ = contact.ğ¤
            ğáµ¢ = contact.ğáµ¢
            push!(cache,
                  "$i $j $(ğ¤[1]) $(ğ¤[2]) $(ğ¤[3]) $(ğáµ¢[1]) $(ğáµ¢[2]) $(ğáµ¢[3]) $(Int(bonded))\n")
        end
    end

    println(p4c, "$t $(length(cache))")
    println(p4c, "P1  P2  CX  CY  CZ  FX  FY  FZ  CONTACT_IS_BONDED")
    print(p4c, join(cache)) # `cache` has a trailing newline

    tâ = time_ns()
    Ît = (tâ - tâ) * 1e-9
    @info "Checkpoint saved! $(length(cache)) active contacts" Ît
end

function _snapshot(x, y, z, r, pid)
    fig, _ = meshscatter(x,
    y,
    z;
    markersize = r,
    color = pid,
    axis = (;
            type = Axis3,
            aspect = :data,
            azimuth = 7.3,
            elevation = 0.189,
            perspectiveness = 0.5),
    figure = (; resolution = (1200, 800)))
    fig
end

function snapshot(grains, step)
    grains = Array(grains)
    pid = [g.id for g in grains]
    ğ¤ = [g.ğ¤ for g in grains]
    x = [k[1] for k in ğ¤]
    y = [k[2] for k in ğ¤]
    z = [k[3] for k in ğ¤]
    r = [g.r for g in grains]
    fig = _snapshot(x, y, z, r, pid)
    save("snapshot_$step.png", fig)
end

function plot_p4p(p4pfile::String)
    records = read_p4p(p4pfile)
    plot_p4p(records)
end

function plot_p4p(records::AbstractVector{Tuple{Float64, Vector{IOGrainDefault}}})
    foreach(enumerate(records)) do (i, (_timestep, grains))
        pid = [g.id for g in grains]
        ğ¤ = [g.ğ¤ for g in grains]
        x = [k[1] for k in ğ¤]
        y = [k[2] for k in ğ¤]
        z = [k[3] for k in ğ¤]
        r = [â(g.V / (4 / 3 * Ï)) for g in grains]
        fig = _snapshot(x, y, z, r, pid)
        save("snapshot_$i.png", fig)
    end
end

function read_p4p(p4pfile)
    records = Tuple{Float64, Vector{IOGrainDefault}}[]
    lines = readlines(p4pfile)
    i = 1
    while i <= length(lines)
        @assert startswith(lines[i], "TIMESTEP")
        timestep, n = parse.(Float64, split(lines[i + 1]))
        n = Int(n)
        grains = Vector{IOGrainDefault}(undef, n)
        for j in 1:n
            line = lines[i + 2 + j]
            pid, _gid, V, m, px, py, pz, vx, vy, vz = parse.(Float64, split(line)[1:10])
            grains[Int(pid)] = IOGrainDefault(Int(pid), V, m, SVector(px, py, pz),
                                              SVector(vx, vy, vz))
        end
        push!(records, (timestep, grains))
        i += 3 + n
    end
    return records
end

function read_p4c(p4cfile)
    records = []
    lines = readlines(p4cfile)
    i = 1
    while i <= length(lines)
        @assert startswith(lines[i], "TIMESTEP")
        timestep, n = parse.(Float64, split(lines[i + 1]))
        n = Int(n)
        current = Dict{Tuple{Int, Int}, IOContactDefault}()
        for j in 1:n
            line = lines[i + 2 + j]
            p1, p2, cx, cy, cz, fx, fy, fz, bonded = parse.(Float64, split(line)[1:9])
            contact = IOContactDefault(Int(p1), Int(p2), SVector(cx, cy, cz),
                                       SVector(fx, fy, fz), Bool(bonded))
            current[(Int(p1), Int(p2))] = contact
        end
        sorted_results = sort(collect(current), by = x -> x[1])
        push!(records, (timestep, sorted_results))
        i += 3 + n
    end
    return records
end

function compare_p4p(p4pfile1::String, p4pfile2::String)
    records1 = read_p4p(p4pfile1)
    records2 = read_p4p(p4pfile2)
    compare_p4p(records1, records2)
end

function compare_p4p(records1, records2; atol=0, rtol=atol>0 ? 0 : â(eps(Float64)))
    @assert length(records1) == length(records2)
    for i in eachindex(records1)
        @assert records1[i][1]==records2[i][1] "[timestep] i = $i, left = $(records1[i][1]), right = $(records2[i][1])"
        @assert length(records1[i][2])==length(records2[i][2]) "[length] i = $i, left = $(length(records1[i][2])) right = $(length(records2[i][2]))"
        for j in 1:length(records1[i][2])
            @assert records1[i][2][j].id==records2[i][2][j].id "[pid] i = $i, j = $j, left = $(records1[i][2][j].id), right = $(records2[i][2][j].id)"
            @assert isapprox(records1[i][2][j].V, records2[i][2][j].V; atol=atol, rtol=rtol) "[V] i = $i, j = $j, left = $(records1[i][2][j].V), right = $(records2[i][2][j].V)"
            @assert isapprox(records1[i][2][j].m, records2[i][2][j].m; atol=atol, rtol=rtol) "[m] i = $i, j = $j, left = $(records1[i][2][j].m), right = $(records2[i][2][j].m)"
            @assert isapprox(records1[i][2][j].ğ¤, records2[i][2][j].ğ¤; atol=atol, rtol=rtol) "[k] i = $i, j = $j, left = $(records1[i][2][j].ğ¤), right = $(records2[i][2][j].ğ¤)"
            @assert isapprox(records1[i][2][j].ğ¯, records2[i][2][j].ğ¯; atol=atol, rtol=rtol) "[v] i = $i, j = $j, left = $(records1[i][2][j].ğ¯), right = $(records2[i][2][j].ğ¯)"
        end
    end
end

function compare_p4c(p4cfile1::String, p4cfile2::String)
    records1 = read_p4c(p4cfile1)
    records2 = read_p4c(p4cfile2)
    compare_p4c(records1, records2)
end

function compare_p4c(records1, records2; atol=0, rtol=atol>0 ? 0 : â(eps(Float64)))
    @assert length(records1) == length(records2)
    for i in eachindex(records1)
        @assert records1[i][1]==records2[i][1] "[timestep] i = $i, left = $(records1[i][1]), right = $(records2[i][1])"
        @assert length(records1[i][2])==length(records2[i][2]) "[length] i = $i, left = $(length(records1[i][2])) right = $(length(records2[i][2]))"
        for j in 1:length(records1[i][2])
            @assert records1[i][2][j][1]==records2[i][2][j][1] "[pair] i = $i, j = $j, left = $(records1[i][2][j][1]), right = $(records2[i][2][j][1])"
            @assert isapprox(records1[i][2][j][2].ğ¤, records2[i][2][j][2].ğ¤; rtol=rtol, atol=atol) "[position] i = $i, j = $j, left = $(records1[i][2][j][2].ğ¤), right = $(records2[i][2][j][2].ğ¤)"
            @assert isapprox(records1[i][2][j][2].ğáµ¢, records2[i][2][j][2].ğáµ¢; rtol=rtol, atol=atol) "[force] i = $i, j = $j, left = $(records1[i][2][j][2].ğáµ¢), right = $(records2[i][2][j][2].ğáµ¢)"
            @assert records1[i][2][j][2].bonded==records2[i][2][j][2].bonded "[bonded] i = $i, j = $j, left = $(records1[i][2][j][2].bonded), right = $(records2[i][2][j][2].bonded)"
        end
    end
end
