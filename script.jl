import Pkg; Pkg.activate(".")

tick = time_ns()
@info "Loading packages..."
using DEMPort
@info "Loading time: $((time_ns() - tick) * 1e-9) s"

# Do not do warm up locally
if !haskey(ENV, "LOCAL")
    @info "Warming up..."
    DEMPort.solve("cfg/6_warmup.toml"; save_information = false)

    @info "Running benchmark..."
    DEMPort.solve("cfg/6.toml")
else
    @info "Running benchmark locally..."
    DEMPort.solve("cfg/6_local.toml")
end
