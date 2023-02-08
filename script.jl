import Pkg; Pkg.activate(".")

tick = time_ns()
@info "Loading packages..."
using DEMPort
@info "Loading time: $((time_ns() - tick) * 1e-9) s"

@info "Warming up..."
DEMPort.solve("cfg/6_warmup.toml"; save_information = false)

@info "Running from start..."
DEMPort.solve("cfg/6.toml")
