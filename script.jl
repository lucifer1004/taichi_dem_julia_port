import Pkg;
Pkg.activate(".");

if !haskey(ENV, "LOCAL")
    # To run GLMakie on server
    ENV["DISPLAY"] = ":0"
end

tick = time_ns()
@info "Loading packages..."
using DEMPort
@info "Loading time: $((time_ns() - tick) * 1e-9) s"

@info "Warming up..."
DEMPort.solve("cfg/6_warmup.toml"; save_information = false)

@info "Running benchmark..."
DEMPort.solve("cfg/6.toml")
