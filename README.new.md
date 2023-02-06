# 与原始版本的区别

> 本文为 Steins; Gate 第零届 PKU HPCGame DEM 赛题 writeup，此文由本团队原创。

- `CUDA.jl` 支持直接对数组进行并行的 `accumulate`，不需要自行实现并行前缀和。
- `Mortan.jl` 计算结果遵循了 Julia 下标从 `1` 开始的规范。因此 `cartesian3mortan([2,3,4])` 的结果等于原版的 `mortan3d32(1, 2, 3) + 1`。
