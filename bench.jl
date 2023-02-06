using BenchmarkTools
using FoldsCUDA, CUDA, FLoops

xs = CUDA.rand(1000000)
ys = CUDA.copy(xs)
zs = Array(xs)
ws = Array(xs)

function f1(xs, ex = CUDAEx())
    @floop ex for i in eachindex(xs)
        xs[i] *= 1.00001
    end
end

function f2(xs)
    xs .*= 1.00001
end

function f3(xs)
    function f(xs)
        blk = blockDim().x * gridDim().x
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        for i in idx:blk:length(xs)
            xs[i] *= 1.00001
        end
    end

    @cuda threads=1024 blocks=72 f(xs)
end

# @btime CUDA.@sync f1($xs);
@btime CUDA.@sync f3($ys);

# @btime f1($ws, $(ThreadedEx()));
# @btime f2($zs);
