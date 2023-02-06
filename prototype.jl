### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 2d15f5ce-a4f1-11ed-04e1-35be261012a0
import Pkg; Pkg.activate(".")

# ╔═╡ e721fb78-ecf3-46a7-8cba-bf839713232b
using CUDA, FoldsCUDA, FLoops, LinearAlgebra, Morton, StaticArrays

# ╔═╡ 67777012-633d-4135-ba08-8a4fe3b608d2
domain_min = @SVector [-200.0, -200.0, -200.0]

# ╔═╡ 2b4868b8-057d-4bcb-b6a7-aee5e6d4abd9
domain_max = @SVector [200.0, 200.0, 200.0]

# ╔═╡ be8c270a-e842-44c9-b0a7-c0509bd2a0a5
max_radius = 1.0

# ╔═╡ 91636c26-5237-4686-835d-44d97fcf9f05
cell_size = 4 * max_radius

# ╔═╡ fefb3123-063a-4a9c-97e5-a76c3309a85d
cell(pos, domain_min, cell_size) = cartesian3morton(@. floor(Int32, (pos - domain_min) / cell_size) + Int32(1))

# ╔═╡ 26fab79f-b6af-40a7-a1e0-54c67ab0ca8c
@code_warntype cell((@SVector [1.0, 2.0, 3.0]), domain_min, 4.0)

# ╔═╡ a422787b-d15a-4d7b-867a-acba5cc8f72d
cell((0.2, 0.5, 1.0), domain_min, cell_size)

# ╔═╡ 41ca797d-ae66-4958-850d-3819ff1b2b73
positions = CUDA.rand(SVector{3, Float32}, 1000000) .* 200

# ╔═╡ 042c1afd-3bc9-4f68-a144-17b35b9e1e29
cell.(positions, Ref(domain_min), Ref(cell_size))

# ╔═╡ 627b7937-08e9-44a8-ae76-6c6b45a2ad6b
begin
	v = (domain_max - domain_min) / (4 * max_radius)
    sz = clamp(prod(ceil.(Int, v)), 1 << 20, 1 << 22)
	cell_count = CUDA.zeros(Int32, sz)
	cell_prefix_sum = CUDA.zeros(Int32, sz)
end

# ╔═╡ 585fa365-c66b-4ceb-8a8a-853188ee410a
begin
	fill!(cell_count, zero(eltype(cell_count)))
    @floop CUDAEx() for pos in positions
        idx = cartesian3morton(cell(pos))
        CUDA.atomic_add!(pointer(cell_count, idx), 1)
    end
    accumulate!(+, cell_prefix_sum, cell_count)
end

# ╔═╡ Cell order:
# ╠═2d15f5ce-a4f1-11ed-04e1-35be261012a0
# ╠═e721fb78-ecf3-46a7-8cba-bf839713232b
# ╠═67777012-633d-4135-ba08-8a4fe3b608d2
# ╠═2b4868b8-057d-4bcb-b6a7-aee5e6d4abd9
# ╠═be8c270a-e842-44c9-b0a7-c0509bd2a0a5
# ╠═91636c26-5237-4686-835d-44d97fcf9f05
# ╠═fefb3123-063a-4a9c-97e5-a76c3309a85d
# ╠═26fab79f-b6af-40a7-a1e0-54c67ab0ca8c
# ╠═a422787b-d15a-4d7b-867a-acba5cc8f72d
# ╠═042c1afd-3bc9-4f68-a144-17b35b9e1e29
# ╠═41ca797d-ae66-4958-850d-3819ff1b2b73
# ╠═627b7937-08e9-44a8-ae76-6c6b45a2ad6b
# ╠═585fa365-c66b-4ceb-8a8a-853188ee410a
