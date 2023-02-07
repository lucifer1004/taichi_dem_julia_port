Base.@kwdef struct Timer
    first::Bool = true
    on::Bool = false
    start::Float64 = 0.0
    total::Float64 = 0.0
end

@as_record Timer

function tick(t::Timer)
    @match t begin
        Timer(first, false, _, total) => Timer(first, true, time(), total)
        Timer(false, true, start, total) => Timer(false, false, start,
                                                  total + time() - start)
        Timer(true, true, _, _) => Timer(false, false, time(), 0.0)
    end
end

mutable struct Statistics
    solve::Timer
    broad_phase_detection::Timer
    hash_table_setup::Timer
    prefix_sum::Timer
    collision_pair_setup::Timer
    contact_resolve::Timer
    contact::Timer
    resolve_wall::Timer
    apply_force::Timer
    update::Timer
end

function Statistics()
    Statistics(Timer(), Timer(), Timer(), Timer(), Timer(), Timer(), Timer(), Timer(),
               Timer(), Timer())
end
