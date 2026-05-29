using Test
using Statistics
using cuTile
using MLIRKernels
using cuTile: BFloat16
const ct = cuTile

# Mirrors examples/vadd.jl (1D variant) from cuTile.jl. The kernel itself is
# pure cuTile syntax — identical to what the CUDA backend compiles.
function vadd_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                     c::ct.TileArray{T,1}, tile::Int) where {T}
    bid = ct.bid(1)
    a_tile = ct.load(a; index=bid, shape=(tile,))
    b_tile = ct.load(b; index=bid, shape=(tile,))
    ct.store(c; index=bid, tile=a_tile + b_tile)
    return
end

# Scatter-only variant of vadd: contiguous tile loads for inputs, ct.scatter
# for the output. Verifies the scatter path in isolation from gather (the
# gather/scatter test combines both).
function vadd_scatter_only_kernel(a::ct.TileArray{Float32,1},
                                  b::ct.TileArray{Float32,1},
                                  c::ct.TileArray{Float32,1}, tile::Int)
    bid = ct.bid(1)
    a_tile = ct.load(a; index=bid, shape=(tile,))
    b_tile = ct.load(b; index=bid, shape=(tile,))
    sum_tile = a_tile + b_tile
    offsets = ct.arange(tile)
    base = ct.Tile((bid - Int32(1)) * Int32(tile))
    indices = ct.broadcast_to(base, (tile,)) .+ offsets
    ct.scatter(c, indices, sum_tile)
    return
end

# Same vadd, but lowered via gather/scatter rather than contiguous tile loads.
# Exercises Intrinsics.iota → vector.step, Intrinsics.broadcast / arange /
# offset tracked-only views, Intrinsics.load_ptr_tko → vector.gather, and
# Intrinsics.store_ptr_tko → vector.scatter. The output should be bit-exact
# to a .+ b for F32 (contiguous-index gather/scatter is just a regular load
# / store at the LLVM level after canonicalisation).
function vadd_gather_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                            c::ct.TileArray{Float32,1}, tile::Int)
    bid = ct.bid(1)
    offsets = ct.arange(tile)
    base = ct.Tile((bid - Int32(1)) * Int32(tile))
    indices = ct.broadcast_to(base, (tile,)) .+ offsets

    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)
    sum_tile = a_tile + b_tile
    ct.scatter(c, indices, sum_tile)
    return
end

# --- Kernels for the new feature tests. Defined at top level so cuTile's
# inference doesn't see them as local closures, which trips a `Vararg`
# canonicalisation path.
function if_branch_kernel(a::ct.TileArray{Float32,1},
                          b::ct.TileArray{Float32,1}, flag::Int32)
    bid = ct.bid(1)
    tile = ct.load(a; index=bid, shape=(16,))
    if flag == Int32(0)
        return nothing
    end
    ct.store(b; index=bid, tile = tile * 2.0f0)
    return
end

function counted_loop_kernel(a::ct.TileArray{Float32,1},
                             b::ct.TileArray{Float32,1}, n::Int32)
    bid = ct.bid(1)
    acc = zeros(Float32, (16,))
    for j in Int32(1):n
        acc = acc + ct.load(a; index=bid, shape=(16,))
    end
    ct.store(b; index=bid, tile=acc)
    return
end

function row_sum_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
    bid = ct.bid(1)
    tile = ct.load(a; index=(bid, 1), shape=(1, 128))
    s = sum(tile; dims=2)
    ct.store(b; index=bid, tile=s)
    return
end

# Same shape as row_sum_kernel but parametric on element type — used for the
# BFloat16 reduction test. Verifies vector.multi_reduction <add> on bf16.
function row_sum_kernel_T(a::ct.TileArray{T,2}, b::ct.TileArray{T,1}) where {T}
    bid = ct.bid(1)
    tile = ct.load(a; index=(bid, 1), shape=(1, 128))
    s = sum(tile; dims=2)
    ct.store(b; index=bid, tile=s)
    return
end

# Row-layernorm kernel — one row per block, single (1, BLOCK_N) tile.
# Exercises Intrinsics.rsqrt → math.rsqrt, scalar→(1,1)tile vector.broadcast
# on a runtime Float32 kernel arg (`eps`), tile/tile mulf for Δ.*Δ, and the
# rest of the softmax-style mean/broadcast machinery.
function layernorm_kernel(X::ct.TileArray{Float32,2}, Y::ct.TileArray{Float32,2},
                          eps::Float32, BLOCK_N::Int)
    bid = ct.bid(1)
    x   = ct.load(X; index=(bid, 1), shape=(1, BLOCK_N))
    n   = Float32(BLOCK_N)
    μ   = sum(x; dims=2) ./ n
    Δ   = x .- μ
    σ²  = sum(Δ .* Δ; dims=2) ./ n
    inv = ct.rsqrt.(σ² .+ eps)
    y   = Δ .* inv
    ct.store(Y; index=(bid, 1), tile=y)
    return
end

# Row-layernorm backward kernel — one row per block. Given x and dy (gradient
# w.r.t. y), recomputes the row's mean / variance / inverse std, then produces
# dx = (dy - sum(dy)/N - x_hat * sum(dy * x_hat)/N) * inv_std. This is the
# canonical no-affine layernorm backward formula. Exercises everything the
# forward kernel does, plus two additional row-reductions (sum(dy) and
# sum(dy .* x_hat)) and a broadcast-multiply-subtract chain.
function layernorm_bwd_kernel(X::ct.TileArray{Float32,2},
                              dY::ct.TileArray{Float32,2},
                              dX::ct.TileArray{Float32,2},
                              eps::Float32, BLOCK_N::Int)
    bid = ct.bid(1)
    x  = ct.load(X;  index=(bid, 1), shape=(1, BLOCK_N))
    dy = ct.load(dY; index=(bid, 1), shape=(1, BLOCK_N))

    n  = Float32(BLOCK_N)
    μ  = sum(x; dims=2) ./ n
    Δ  = x .- μ
    σ² = sum(Δ .* Δ; dims=2) ./ n
    inv_std = ct.rsqrt.(σ² .+ eps)
    x_hat = Δ .* inv_std

    sum1 = sum(dy; dims=2)
    sum2 = sum(dy .* x_hat; dims=2)
    dx = (dy .- sum1 ./ n .- x_hat .* (sum2 ./ n)) .* inv_std

    ct.store(dX; index=(bid, 1), tile=dx)
    return
end

# Row-softmax kernel — one row per block, single (1, BLOCK_N) tile.
# Exercises Intrinsics.reduce(max) → vector.multi_reduction <maxnumf>,
# tile→tile vector.broadcast across the reduction axis, math.exp on a 2-D
# vector, and arith.divf elementwise.
function softmax_kernel(A::ct.TileArray{Float32,2}, Y::ct.TileArray{Float32,2},
                       BLOCK_N::Int)
    bid = ct.bid(1)
    row = ct.load(A; index=(bid, 1), shape=(1, BLOCK_N))
    m   = maximum(row; dims=2)
    shifted = row .- m
    e   = exp.(shifted)
    s   = sum(e; dims=2)
    y   = e ./ s
    ct.store(Y; index=(bid, 1), tile=y)
    return
end

# 1-stage matmul kernel — 2-D grid via bid(1)/bid(2), 64×64 inner tile.
# Exercises Intrinsics.mma → vector.contract, Intrinsics.cldi → arith.ceildivsi,
# memref-rooted Base.getfield (`size(A, 2)`), and a 2-D scf.parallel.
function matmul_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                       BM::Int, BN::Int, BK::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    acc = zeros(T, (BM, BN))
    for k in 1:cld(size(A, 2), BK)
        a_tile = ct.load(A; index=(bid_m, k), shape=(BM, BK))
        b_tile = ct.load(B; index=(k, bid_n), shape=(BK, BN))
        acc = muladd(a_tile, b_tile, acc)
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return
end

# Register-tile / rank-1 outer-product matmul kernel. This is the
# **recommended** matmul pattern for CPU targets — it lets LLVM keep the
# accumulator in vector registers across the K-loop (carried as a phi node)
# and emits a tight inner loop instead of a multi-thousand-FMA unrolled body.
#
# At RM=RN=16 the inner body is ~16 vector FMAs on `vector<16xf32>`,
# producing a few hundred lines of LLVM IR and matching the BLAS microkernel
# pattern (rank-1 outer-product update over K). See
# `bench/perf_research/README.md` for the cross-comparison with OpenBLAS's
# Cooperlake sgemm_kernel.
function matmul_reg_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2},
                           C::ct.TileArray{T,2},
                           RM::Int, RN::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    acc = zeros(T, (RM, RN))
    for k in 1:size(A, 2)
        a_col = ct.load(A; index=(bid_m, k), shape=(RM, 1))    # (RM, 1)
        b_row = ct.load(B; index=(k, bid_n), shape=(1, RN))    # (1, RN)
        acc = muladd(a_col, b_row, acc)                         # rank-1 update
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return
end

# Single-head attention kernel (non-flash). One block per output row-tile of O.
# We avoid the K-tile loop by sizing BN to the full sequence length, so the
# whole of K and V is loaded in one shot. Pipeline per block:
#   Q tile (BM,D) and K tile (BN,D) → S = Q @ K^T (BM,BN)
#   P = row-softmax(S) (BM, BN)
#   O tile = P @ V (BM, D)
# Exercises Intrinsics.permute → vector.transpose plus two vector.contract +
# softmax (max-shift + exp + sum + divf) in one kernel.
function attention_kernel(Q::ct.TileArray{Float32,2}, K::ct.TileArray{Float32,2},
                          V::ct.TileArray{Float32,2}, O::ct.TileArray{Float32,2},
                          BM::Int, BN::Int, D::Int)
    bid = ct.bid(1)
    q = ct.load(Q; index=(bid, 1), shape=(BM, D))
    k = ct.load(K; index=(1, 1),   shape=(BN, D))
    v = ct.load(V; index=(1, 1),   shape=(BN, D))
    s_acc = zeros(Float32, (BM, BN))
    kT = permutedims(k, (2, 1))
    s = muladd(q, kT, s_acc)
    m = maximum(s; dims=2)
    sh = s .- m
    e = exp.(sh)
    ssum = sum(e; dims=2)
    p = e ./ ssum
    o_acc = zeros(Float32, (BM, D))
    o = muladd(p, v, o_acc)
    ct.store(O; index=(bid, 1), tile=o)
    return
end

# Mixed-precision matmul: BF16 inputs accumulating into an F32 tile. This is
# the canonical ML matmul shape — vector.contract lowers an lhs/rhs in bf16
# with an acc/result in f32 directly.
function matmul_mixed_kernel(A::ct.TileArray{BFloat16,2},
                             B::ct.TileArray{BFloat16,2},
                             C::ct.TileArray{Float32,2},
                             BM::Int, BN::Int, BK::Int)
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    acc = zeros(Float32, (BM, BN))
    for k in 1:cld(size(A, 2), BK)
        a_tile = ct.load(A; index=(bid_m, k), shape=(BM, BK))
        b_tile = ct.load(B; index=(k, bid_n), shape=(BK, BN))
        acc = muladd(a_tile, b_tile, acc)
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return
end

# Batched matmul kernel — cuTile-canonical layout: A (M, K, Batch), B (K, N, Batch),
# C (M, N, Batch) in Julia col-major. Per-block tile is (BM, BN, BS) of C, looping
# along K in BK-sized steps. Block grid = (cld(M, BM), cld(N, BN), cld(Batch, BS)).
# Exercises 3-D Intrinsics.mma → vector.contract with batched indexing maps
# (4 iterators: parallel(b), parallel(m), parallel(n), reduction(k)).
function bmm_kernel(A::ct.TileArray{T,3}, B::ct.TileArray{T,3}, C::ct.TileArray{T,3},
                    BM::Int, BN::Int, BK::Int, BS::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    bid_b = ct.bid(3)
    acc = zeros(T, (BM, BN, BS))
    for k in 1:cld(size(A, 2), BK)
        a = ct.load(A; index=(bid_m, k,     bid_b), shape=(BM, BK, BS))
        b = ct.load(B; index=(k,     bid_n, bid_b), shape=(BK, BN, BS))
        acc = muladd(a, b, acc)
    end
    ct.store(C; index=(bid_m, bid_n, bid_b), tile=acc)
    return
end

# Flash attention (with online softmax). One block per BM-row tile of O.
# K and V are processed in BN-sized tiles along the sequence axis, with the
# running (m, l, o) statistics carried as iter_args through `scf.for`.
# Exercises: 3 iter_args of mixed vector shapes ((BM, D), (BM, 1), (BM, 1)),
# Intrinsics.maxf → arith.maxnumf (element-wise tile/tile max), Intrinsics.fma
# → math.fma (the `α*l + sum(p)` and `α*o + p@v` accumulation steps), and a
# `-Inf32` splat constant.
function flash_attn_kernel(Q::ct.TileArray{Float32,2}, K::ct.TileArray{Float32,2},
                           V::ct.TileArray{Float32,2}, O::ct.TileArray{Float32,2},
                           BM::Int, BN::Int, D::Int, N_KV::Int)
    bid = ct.bid(1)
    q = ct.load(Q; index=(bid, 1), shape=(BM, D))
    m = fill(-Inf32, (BM, 1))
    l = zeros(Float32, (BM, 1))
    o = zeros(Float32, (BM, D))
    for kbi in 1:cld(N_KV, BN)
        k = ct.load(K; index=(kbi, 1), shape=(BN, D))
        v = ct.load(V; index=(kbi, 1), shape=(BN, D))
        kT = permutedims(k, (2, 1))
        s = muladd(q, kT, zeros(Float32, (BM, BN)))
        m_new_chunk = maximum(s; dims=2)
        m_new = max.(m, m_new_chunk)
        α = exp.(m .- m_new)
        p = exp.(s .- m_new)
        l = α .* l .+ sum(p; dims=2)
        o = α .* o .+ muladd(p, v, zeros(Float32, (BM, D)))
        m = m_new
    end
    o = o ./ l
    ct.store(O; index=(bid, 1), tile=o)
    return
end

# ----------------------------------------------------------------------------
# Atomic RMW kernels for the new atomic_add / atomic_max / atomic_min tests.
# All three kernels use the scalar-index form (`ct.atomic_*(arr, scalar_idx,
# val)`) which cuTile lowers to a 0-D pointer-tile via `Intrinsics.offset`
# followed by an `Intrinsics.atomic_*` SCI op. On the MLIRKernels walker this
# maps to a single `memref.atomic_rmw <kind>` per block.
function atomic_count_kernel(counter::ct.TileArray{Int32,1})
    bid = ct.bid(1)
    ct.atomic_add(counter, 1, Int32(1); memory_order=ct.MemoryOrder.AcqRel)
    return
end

function atomic_count_f32_kernel(counter::ct.TileArray{Float32,1}, val::Float32)
    bid = ct.bid(1)
    ct.atomic_add(counter, 1, val; memory_order=ct.MemoryOrder.AcqRel)
    return
end

function atomic_max_kernel(out::ct.TileArray{Int32,1})
    bid = ct.bid(1)
    ct.atomic_max(out, 1, bid; memory_order=ct.MemoryOrder.AcqRel)
    return
end

function atomic_min_kernel(out::ct.TileArray{Int32,1})
    bid = ct.bid(1)
    ct.atomic_min(out, 1, bid; memory_order=ct.MemoryOrder.AcqRel)
    return
end

# Histogram-style atomic_add: one block per input value; each block reads
# `values[bid]`, computes a bucket index in 1..n_buckets, and atomically
# bumps `counts[bucket]`. Exercises the 0-D pointer-tile-from-tile-load
# path (scalar getindex → reshape (1,) → ()) and `Intrinsics.remi`.
function atomic_hist_kernel(values::ct.TileArray{Int32,1},
                            counts::ct.TileArray{Int32,1}, n_buckets::Int)
    bid = ct.bid(1)
    v = values[bid]
    bucket = rem(v, Int32(n_buckets)) + Int32(1)
    ct.atomic_add(counts, bucket, Int32(1); memory_order=ct.MemoryOrder.AcqRel)
    return
end

# Bitwise atomic kernels. All take a 1-element accumulator; each block runs
# one atomic_{or,and,xor} of `bid` against the slot. Walker maps to
# `memref.atomic_rmw {ori,andi,xori}`.
function atomic_or_kernel(out::ct.TileArray{Int32,1})
    bid = ct.bid(1)
    ct.atomic_or(out, 1, bid; memory_order=ct.MemoryOrder.AcqRel)
    return
end

function atomic_and_kernel(out::ct.TileArray{Int32,1})
    bid = ct.bid(1)
    ct.atomic_and(out, 1, bid; memory_order=ct.MemoryOrder.AcqRel)
    return
end

function atomic_xor_kernel(out::ct.TileArray{Int32,1})
    bid = ct.bid(1)
    ct.atomic_xor(out, 1, bid; memory_order=ct.MemoryOrder.AcqRel)
    return
end

# Atomic exchange: each block stores its bid into slot 1. The final value is
# the bid of whichever block ran last (non-deterministic but in 1..n_blocks).
# Walker maps to `memref.atomic_rmw assign`.
function atomic_xchg_kernel(out::ct.TileArray{Int32,1})
    bid = ct.bid(1)
    ct.atomic_xchg(out, 1, bid; memory_order=ct.MemoryOrder.AcqRel)
    return
end

# Atomic compare-and-swap: first block to land sets slot 1 to its bid; later
# blocks see a non-zero value and the CAS leaves it unchanged. Final value
# is non-zero and in 1..n_blocks. Walker maps to `memref.generic_atomic_rmw`
# with a compare/select region body.
function atomic_cas_kernel(locks::ct.TileArray{Int32,1})
    bid = ct.bid(1)
    ct.atomic_cas(locks, 1, Int32(0), bid; memory_order=ct.MemoryOrder.AcqRel)
    return
end

# Counter + per-block bid recorder. Each block atomically increments
# `counter[1]`, reads the prior value, and writes its bid at `bids[prior+1]`.
# Verifies both that the counter hits `n_blocks` (no lost updates under
# contention) and that every block's bid appears exactly once in `bids`
# (the prior value is unique per block — atomic_add returns a monotonic
# sequence of prior values 0, 1, 2, …, n_blocks-1).
function atomic_counter_record_kernel(counter::ct.TileArray{Int32,1},
                                      bids::ct.TileArray{Int32,1})
    bid = ct.bid(1)
    prior = ct.atomic_add(counter, 1, Int32(1); memory_order=ct.MemoryOrder.AcqRel)
    bids[prior + Int32(1)] = bid
    return
end

# Mixed-math kernel — exercises sin/cos/log/tanh/floor/abs in one tile pass.
# Every op should lower to a single `math.*` MLIR instruction on the tile
# `vector<16xf32>`, then the math-to-llvm pass takes it from there.
function math_mix_kernel(a::ct.TileArray{Float32,1},
                         b::ct.TileArray{Float32,1}, tile::Int)
    bid = ct.bid(1)
    x = ct.load(a; index=bid, shape=(tile,))
    y = abs.(floor.(sin.(x) .+ tanh.(log.(x) .+ cos.(x))))
    ct.store(b; index=bid, tile=y)
    return
end

# Matrix transpose: B[j, i] = A[i, j]. One tile per (bid_m, bid_n) output
# block. cuTile lowers `permutedims(tile, (2, 1))` to `Intrinsics.permute`
# which maps to `vector.transpose` in the walker (already used by attention).
# This test exercises that path in isolation on a non-square shape.
function transpose_kernel(A::ct.TileArray{Float32,2}, B::ct.TileArray{Float32,2},
                          BM::Int, BN::Int)
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    a_tile = ct.load(A; index=(bid_m, bid_n), shape=(BM, BN))
    b_tile = permutedims(a_tile, (2, 1))
    ct.store(B; index=(bid_n, bid_m), tile=b_tile)
    return
end

# RNG kernels — exercise the cuTile Philox2x32-7 path through the implicit
# `KernelState.seed` parameter that lower_to_mlir always emits as the trailing
# i32 arg.
function rand_uniform_kernel(out::ct.TileArray{Float32,1}, tile::Int)
    bid = ct.bid(1)
    r = rand(Float32, (tile,))
    ct.store(out; index=bid, tile=r)
    return
end

function rand_normal_kernel(out::ct.TileArray{Float32,1}, tile::Int)
    bid = ct.bid(1)
    r = randn(Float32, (tile,))
    ct.store(out; index=bid, tile=r)
    return
end

# ----------------------------------------------------------------------------
# SPMD-mode kernels (ISPC-style: plain Julia scalar code, lifted to lanes)
# ----------------------------------------------------------------------------
#
# These kernels look like normal Julia — no Tile/ct.* types. The walker's
# SPMD mode (`MLIRKernels.spmd_function`) lifts the trailing lane-index arg
# to a `lane_width`-wide vector and turns each scalar op into a vector op.

function vadd_spmd(a::Vector{Float32}, b::Vector{Float32},
                   c::Vector{Float32}, i::Int)
    @inbounds c[i] = a[i] + b[i]
    return
end

# 1-stage complex DFT kernel: Y = W @ X computed as two real matmuls per part.
# Complex values are packed (real, imag) along a leading dim of size 2. Exercises
# `ct.extract` on a 3-D tile + the rank-2 mma path with asymmetric shapes
# (W is (N,N), X reshaped to (N,BS) — gives (N,N) × (N,BS) → (N,BS)).
function dft_kernel(X_packed::ct.TileArray{Float32,3},   # (2, N, BS)
                    Y_packed::ct.TileArray{Float32,3},   # (2, N, BS)
                    W_packed::ct.TileArray{Float32,3},   # (2, N, N)
                    N::Int, BS::Int)
    bid = ct.bid(1)
    X_ri = ct.load(X_packed; index=(1, 1, 1), shape=(2, N, BS))
    X_r = reshape(ct.extract(X_ri, (1, 1, 1), (1, N, BS)), (N, BS))
    X_i = reshape(ct.extract(X_ri, (2, 1, 1), (1, N, BS)), (N, BS))

    W_ri = ct.load(W_packed; index=(1, 1, 1), shape=(2, N, N))
    W_r = reshape(ct.extract(W_ri, (1, 1, 1), (1, N, N)), (N, N))
    W_i = reshape(ct.extract(W_ri, (2, 1, 1), (1, N, N)), (N, N))

    # (Y_r + i Y_i) = (W_r + i W_i) (X_r + i X_i)
    Y_r = W_r * X_r - W_i * X_i
    Y_i = W_r * X_i + W_i * X_r

    Y_r_packed = reshape(Y_r, (1, N, BS))
    Y_i_packed = reshape(Y_i, (1, N, BS))
    Y_ri = ct.cat((Y_r_packed, Y_i_packed), 1)
    ct.store(Y_packed; index=(1, 1, 1), tile=Y_ri)
    return
end

# 3-stage Cooley-Tukey FFT — Julia port of cuTile.jl/examples/fft.jl. Decomposes
# N = F0*F1*F2 with three batched matrix DFTs (F0, F1, F2) and two twiddle ×
# permute stages between them. Exercises:
#   • `ct.extract` on 3-D tiles to split real/imag.
#   • Rank-3 batched `mma` with broadcasting batch dim (1 → BS).
#   • Reshape + `permutedims` on 4-D tiles.
#   • `ct.cat` to merge the final r/i pair back into the packed (D, N2D, BS) buffer.
function fft_kernel(
    x_in::ct.TileArray{Float32, 3},
    y_out::ct.TileArray{Float32, 3},
    W0::ct.TileArray{Float32, 3},
    W1::ct.TileArray{Float32, 3},
    W2::ct.TileArray{Float32, 3},
    T0::ct.TileArray{Float32, 3},
    T1::ct.TileArray{Float32, 3},
    n_const::Int, f0_const::Int, f1_const::Int, f2_const::Int,
    f0f1_const::Int, f1f2_const::Int, f0f2_const::Int,
    bs_const::Int, d_const::Int, n2d_const::Int)

    N = n_const; F0 = f0_const; F1 = f1_const; F2 = f2_const
    F0F1 = f0f1_const; F1F2 = f1f2_const; F0F2 = f0f2_const
    BS = bs_const; D = d_const; N2D = n2d_const

    bid = ct.bid(1)
    X_ri = reshape(ct.load(x_in; index=(Int32(1), Int32(1), bid), shape=(D, N2D, BS)),
                   (2, N, BS))

    # Split real/imag, factor to 4-D (F2, F1, F0, BS).
    X_r = reshape(ct.extract(X_ri, (1, 1, 1), (1, N, BS)), (F2, F1, F0, BS))
    X_i = reshape(ct.extract(X_ri, (2, 1, 1), (1, N, BS)), (F2, F1, F0, BS))

    # Load DFT matrices: trailing batch dim 1 broadcasts to BS in batched mma.
    W0_ri = reshape(ct.load(W0; index=(1, 1, 1), shape=(2, F0, F0)), (2, F0, F0))
    W0_r = reshape(ct.extract(W0_ri, (1, 1, 1), (1, F0, F0)), (F0, F0, 1))
    W0_i = reshape(ct.extract(W0_ri, (2, 1, 1), (1, F0, F0)), (F0, F0, 1))

    W1_ri = reshape(ct.load(W1; index=(1, 1, 1), shape=(2, F1, F1)), (2, F1, F1))
    W1_r = reshape(ct.extract(W1_ri, (1, 1, 1), (1, F1, F1)), (F1, F1, 1))
    W1_i = reshape(ct.extract(W1_ri, (2, 1, 1), (1, F1, F1)), (F1, F1, 1))

    W2_ri = reshape(ct.load(W2; index=(1, 1, 1), shape=(2, F2, F2)), (2, F2, F2))
    W2_r = reshape(ct.extract(W2_ri, (1, 1, 1), (1, F2, F2)), (F2, F2, 1))
    W2_i = reshape(ct.extract(W2_ri, (2, 1, 1), (1, F2, F2)), (F2, F2, 1))

    # Twiddle factors (flattened to (1, N) and (1, F1F2) for elementwise multiply).
    T0_ri = reshape(ct.load(T0; index=(1, 1, 1), shape=(2, F1F2, F0)), (2, F1F2, F0))
    T0_r = reshape(ct.extract(T0_ri, (1, 1, 1), (1, F1F2, F0)), (1, N))
    T0_i = reshape(ct.extract(T0_ri, (2, 1, 1), (1, F1F2, F0)), (1, N))

    T1_ri = reshape(ct.load(T1; index=(1, 1, 1), shape=(2, F2, F1)), (2, F2, F1))
    T1_r = reshape(ct.extract(T1_ri, (1, 1, 1), (1, F2, F1)), (1, F1F2))
    T1_i = reshape(ct.extract(T1_ri, (2, 1, 1), (1, F2, F1)), (1, F1F2))

    # Stage 0: F0-point DFT.
    X_r = reshape(X_r, (F1F2, F0, BS))
    X_i = reshape(X_i, (F1F2, F0, BS))
    X_r_ = reshape(X_r * W0_r - X_i * W0_i, (1, N, BS))
    X_i_ = reshape(X_r * W0_i + X_i * W0_r, (1, N, BS))

    # Twiddle + permute 0.
    X_r2 = T0_r .* X_r_ .- T0_i .* X_i_
    X_i2 = T0_i .* X_r_ .+ T0_r .* X_i_
    X_r3 = permutedims(reshape(X_r2, (F2, F1, F0, BS)), (3, 1, 2, 4))
    X_i3 = permutedims(reshape(X_i2, (F2, F1, F0, BS)), (3, 1, 2, 4))

    # Stage 1: F1-point DFT.
    X_r4 = reshape(X_r3, (F0F2, F1, BS))
    X_i4 = reshape(X_i3, (F0F2, F1, BS))
    X_r5 = reshape(X_r4 * W1_r - X_i4 * W1_i, (F0, F1F2, BS))
    X_i5 = reshape(X_r4 * W1_i + X_i4 * W1_r, (F0, F1F2, BS))

    # Twiddle + permute 1.
    X_r6 = T1_r .* X_r5 .- T1_i .* X_i5
    X_i6 = T1_i .* X_r5 .+ T1_r .* X_i5
    X_r7 = permutedims(reshape(X_r6, (F0, F2, F1, BS)), (3, 1, 2, 4))
    X_i7 = permutedims(reshape(X_i6, (F0, F2, F1, BS)), (3, 1, 2, 4))

    # Stage 2: F2-point DFT.
    X_r8 = reshape(X_r7, (F0F1, F2, BS))
    X_i8 = reshape(X_i7, (F0F1, F2, BS))
    X_r9 = X_r8 * W2_r - X_i8 * W2_i
    X_i9 = X_r8 * W2_i + X_i8 * W2_r

    # Final permute.
    X_r10 = permutedims(reshape(X_r9, (F1, F0, F2, BS)), (2, 1, 3, 4))
    X_i10 = permutedims(reshape(X_i9, (F1, F0, F2, BS)), (2, 1, 3, 4))

    X_r_final = reshape(X_r10, (1, N, BS))
    X_i_final = reshape(X_i10, (1, N, BS))
    Y_ri = reshape(ct.cat((X_r_final, X_i_final), 1), (D, N2D, BS))
    ct.store(y_out; index=(Int32(1), Int32(1), bid), tile=Y_ri)
    return
end

# Helpers for FFT tests.
function _dft_matrix(s::Int)
    W = zeros(ComplexF32, s, s)
    for i in 0:s-1, j in 0:s-1
        W[i+1, j+1] = exp(-2π * im * i * j / s)
    end
    r = zeros(Float32, 2, s, s)
    r[1, :, :] .= Float32.(real.(W))
    r[2, :, :] .= Float32.(imag.(W))
    return r
end
function _twiddles_T0(F0::Int, F1F2::Int, N::Int)
    T0 = zeros(Float32, 2, F1F2, F0)
    for i in 0:F0-1, j in 0:F1F2-1
        val = exp(-2π * im * i * j / N)
        T0[1, j+1, i+1] = Float32(real(val))
        T0[2, j+1, i+1] = Float32(imag(val))
    end
    return T0
end
function _twiddles_T1(F1::Int, F2::Int, F1F2::Int)
    T1 = zeros(Float32, 2, F2, F1)
    for j in 0:F1-1, k in 0:F2-1
        val = exp(-2π * im * j * k / F1F2)
        T1[1, k+1, j+1] = Float32(real(val))
        T1[2, k+1, j+1] = Float32(imag(val))
    end
    return T1
end

# Hand-rolled DFT — used as the reference oracle for both kernels (avoids
# pulling FFTW into the test deps).
function _dft_ref(x::AbstractMatrix{ComplexF32})
    N, BS = size(x)
    y = zeros(ComplexF32, N, BS)
    for b in 1:BS, k in 0:N-1, n in 0:N-1
        y[k+1, b] += x[n+1, b] * exp(-2π * im * k * n / N)
    end
    return y
end

# MoE routing kernel — Mixture of Experts (option 3: per-block expert dispatch).
# Each block processes one token: reads the token's pre-assigned expert id,
# atomically claims a slot in that expert's output region via atomic_add, then
# loads the token (D, 1), loads the expert's weight matrix (D_out, D, 1) slice
# from the 3-D Wexp tensor (3-D indexed by `expert` along the trailing dim),
# does a single (D_out, D) × (D, 1) → (D_out, 1) matmul, and writes the result
# to Y at the claimed slot. Also records the originating token id in
# `slot_tokens` for host-side verification.
#
# Exercises:
#   • Scalar indexing into a 1-D Int32 TileArray (`expert_ids[bid]`)
#     → load_partition_view with `(1,)` tile shape + reshape to `()`.
#   • atomic_add returning the prior value (the slot index) — same path as the
#     atomic_counter_record_kernel.
#   • Scalar tile store `slot_tokens[slot] = bid` at a *runtime* index.
#   • 3-D load with a runtime trailing index (the expert id) — partition_view
#     shape `(D_out, D, 1)`, indices `(0, 0, expert)`.
#   • reshape (D_out, D, 1) → (D_out, D) and a (D_out, D) × (D, 1) → (D_out, 1)
#     vector.contract.
#   • 2-D store at a runtime atomic-derived column index.
function moe_routing_kernel(
        X::ct.TileArray{Float32,2},          # (D, num_tokens) — input tokens
        Y::ct.TileArray{Float32,2},          # (D_out, MAX_PER_EXPERT*num_experts)
        expert_ids::ct.TileArray{Int32,1},   # (num_tokens,) — per-token expert id
        counters::ct.TileArray{Int32,1},     # (num_experts,) — per-expert slot counter
        slot_tokens::ct.TileArray{Int32,1},  # (MAX_PER_EXPERT*num_experts,) — slot → token id
        Wexp::ct.TileArray{Float32,3},       # (D_out, D, num_experts) — expert weights
        D::Int, D_out::Int, MAX_PER_EXPERT::Int)
    bid = ct.bid(1)
    expert = expert_ids[bid]
    # Atomically claim a slot in this expert's region; `slot` is the prior value
    # (0-based count of tokens already routed to this expert).
    slot = ct.atomic_add(counters, expert, Int32(1);
                         memory_order=ct.MemoryOrder.AcqRel)
    # 1-indexed slot in Y: (expert-1) * MAX_PER_EXPERT + slot + 1.
    slot_in_y = (expert - Int32(1)) * Int32(MAX_PER_EXPERT) + slot + Int32(1)
    # Record this token's id at its slot for host verification.
    slot_tokens[slot_in_y] = bid
    # Load token column X[:, bid] → (D, 1).
    x_tile = ct.load(X; index=(Int32(1), bid), shape=(D, 1))
    # Load expert's weight matrix Wexp[:, :, expert] → (D_out, D, 1).
    w_tile = ct.load(Wexp; index=(Int32(1), Int32(1), expert),
                     shape=(D_out, D, 1))
    w_2d = reshape(w_tile, (D_out, D))
    # Matmul (D_out, D) * (D, 1) → (D_out, 1).
    acc = zeros(Float32, (D_out, 1))
    y_tile = muladd(w_2d, x_tile, acc)
    # Store to Y[:, slot_in_y].
    ct.store(Y; index=(Int32(1), slot_in_y), tile=y_tile)
    return
end

@testset "MLIRKernels" begin

    @testset "aligned_array" begin
        for align in (32, 64, 128, 256)
            a = MLIRKernels.aligned_array(Float32, 1024; alignment=align)
            @test a isa Array{Float32, 1}
            @test length(a) == 1024
            @test UInt(pointer(a)) % align == 0
            # Indexable, broadcastable like any Array
            a .= 1f0
            @test all(==(1f0), a)
        end
    end

    @testset "vadd 1-D Float32" begin
        # Tile size 16 lets us run with small N; alignment 128 matches the
        # cuTile default ArraySpec.
        n = 1024
        tile = 16
        a = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b = MLIRKernels.aligned_array(Float32, n; alignment=128)
        c = MLIRKernels.aligned_array(Float32, n; alignment=128)
        copyto!(a, Float32.(1:n))
        copyto!(b, Float32.(101:100+n))
        fill!(c, 0f0)

        k = MLIRKernels.cpu_function(vadd_kernel, (a, b, c, ct.Constant(tile)))
        k(a, b, c, ct.Constant(tile); blocks=cld(n, tile))

        @test c ≈ a .+ b
    end

    @testset "@parallel_for surface (1-D)" begin
        # Same vadd, launched through the explicit parallel-for. This is the
        # surface kernels should actually use; `cpu_function` + manual call is
        # the underlying mechanism.
        n = 1024
        tile = 16
        a = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b = MLIRKernels.aligned_array(Float32, n; alignment=128)
        c = MLIRKernels.aligned_array(Float32, n; alignment=128)
        copyto!(a, Float32.(1:n))
        copyto!(b, Float32.(101:100+n))
        fill!(c, 0f0)

        # Function form
        MLIRKernels.parallel_for(vadd_kernel,
                               (a, b, c, ct.Constant(tile));
                               blocks = cld(n, tile))
        @test c ≈ a .+ b

        # Macro form — re-run with a different buffer to confirm same effect.
        c2 = MLIRKernels.aligned_array(Float32, n; alignment=128)
        fill!(c2, 0f0)
        MLIRKernels.@parallel_for blocks = cld(n, tile) vadd_kernel(a, b, c2,
                                                                  ct.Constant(tile))
        @test c2 ≈ a .+ b
    end

    @testset "reflection: code_mlir contains expected ops" begin
        n = 1024; tile = 16
        a = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b = MLIRKernels.aligned_array(Float32, n; alignment=128)
        c = MLIRKernels.aligned_array(Float32, n; alignment=128)
        mlir = MLIRKernels.code_mlir(vadd_kernel, (a, b, c, ct.Constant(tile)))
        # Should be a func.func named after the kernel
        @test occursin("func.func @vadd_kernel", mlir)
        # Tile-shaped vectors
        @test occursin("vector<16xf32>", mlir)
        # Parallel grid
        @test occursin("scf.parallel", mlir)
        @test occursin("scf.reduce", mlir)
        # Alignment hint from ArraySpec.alignment
        @test occursin("memref.assume_alignment", mlir)
        # Strided layout from ArraySpec.contiguous
        @test occursin("strided<[1]>", mlir)
        # No unwanted ops
        @test !occursin("memref.dim", mlir)   # we don't need it — nblocks is a runtime arg
        @test !occursin("scf.yield", mlir)    # body of scf.parallel uses scf.reduce
        # 1-D contiguous TileArray: only stride is the leading (unit) dim, so
        # `arg_chain([3, 1])` is empty and no per-stride llvm.intr.assume ops
        # are emitted.
        @test !occursin("llvm.intr.assume", mlir)
        @test !occursin("memref.extract_strided_metadata", mlir)
    end

    @testset "reflection: stride divby assumes (2-D matmul)" begin
        # Drives `emit_stride_divby_assumes!`: each 2-D TileArray arg has a
        # leading contiguous dim (stride=1, no annotation) plus a row stride
        # whose ArraySpec divisibility hint shows up as an
        # `llvm.intr.assume((stride % n) == 0)` triple at func entry. This is
        # the alignment proof LLVM's vectorizer needs for non-leading-dim
        # accesses; absent it the row-stride dim falls back to scalar loads.
        M, N, K = 128, 128, 128
        BM, BN, BK = 64, 64, 64
        raw_a = MLIRKernels.aligned_array(Float32, M*K; alignment=128)
        raw_b = MLIRKernels.aligned_array(Float32, K*N; alignment=128)
        raw_c = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        A = reshape(raw_a, (M, K))
        B = reshape(raw_b, (K, N))
        C = reshape(raw_c, (M, N))
        mlir = MLIRKernels.code_mlir(matmul_kernel,
            (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK));
            n_grid_dims=2)
        # 3 TileArray args × 1 non-contiguous-dim with a DivBy hint = 3 assumes.
        n_assume = length(collect(eachmatch(r"llvm\.intr\.assume", mlir)))
        n_extract = length(collect(eachmatch(r"memref\.extract_strided_metadata", mlir)))
        @test n_assume == 3
        @test n_extract == 3
        # Each assume is built from `arith.remui stride, N == 0`. Spec divisor
        # is `1` for the row stride in the default ArraySpec, projected to its
        # largest power-of-2 — which is `1`, so no annotation is emitted from
        # spec alone. But the underlying matmul ArraySpec carries a
        # `stride_div_by[2]` > 1, so the `arith.remui` form should appear.
        @test occursin("arith.remui", mlir)
        @test occursin("arith.cmpi", mlir)
    end

    @testset "reflection: code_llvm contains __kmpc_*" begin
        n = 1024; tile = 16
        a = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b = MLIRKernels.aligned_array(Float32, n; alignment=128)
        c = MLIRKernels.aligned_array(Float32, n; alignment=128)
        ll = MLIRKernels.code_llvm(vadd_kernel, (a, b, c, ct.Constant(tile)))
        @test occursin("__kmpc_fork_call", ll)   # OpenMP parallelism
        @test occursin("_mlir_ciface_vadd_kernel", ll)  # C-interface wrapper
    end

    @testset "aligned_array meets posted alignment" begin
        # aligned_array is the official path. Verify it gives us pointer
        # alignment matching cuTile's default ArraySpec alignment used by
        # `TileArray(...)`. The launch-time check (pointer_aligned) reads
        # the kernel's param_alignments, which come from each arg's
        # ArraySpec — so this just confirms aligned_array is sufficient
        # for the canonical alignment.
        for align in (32, 64, 128)
            a = MLIRKernels.aligned_array(Float32, 1024; alignment=align)
            @test MLIRKernels.pointer_aligned(a, align)
        end
    end

    @testset "if/else branching (early return)" begin
        # Mirror of test/device/control_flow.jl `early_return_*`. flag==0 keeps
        # `b` untouched (the kernel returns before the store); flag==1 stores
        # tile*2.
        n = 1024; tile = 16
        a  = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b0 = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b1 = MLIRKernels.aligned_array(Float32, n; alignment=128)
        copyto!(a, Float32.(1:n))
        fill!(b0, 7f0); fill!(b1, 7f0)

        k = MLIRKernels.cpu_function(if_branch_kernel, (a, b0, Int32(0)))
        k(a, b0, Int32(0); blocks=cld(n, tile))
        @test b0 == fill(7f0, n)   # nothing stored

        k(a, b1, Int32(1); blocks=cld(n, tile))
        @test b1 ≈ a .* 2f0

        mlir = MLIRKernels.code_mlir(if_branch_kernel, (a, b0, Int32(0)))
        @test occursin("scf.if", mlir)
    end

    @testset "counted for loop" begin
        # Repeated accumulate-from-`a` n times into a zero-tile, then store.
        # Result: b[i] == n * a[i].
        n = 1024; tile = 16
        a = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b = MLIRKernels.aligned_array(Float32, n; alignment=128)
        copyto!(a, Float32.(1:n))
        fill!(b, 0f0)

        reps = Int32(5)
        k = MLIRKernels.cpu_function(counted_loop_kernel, (a, b, reps))
        k(a, b, reps; blocks=cld(n, tile))
        @test b ≈ a .* Float32(reps)

        mlir = MLIRKernels.code_mlir(counted_loop_kernel, (a, b, reps))
        @test occursin("scf.for", mlir)
    end

    @testset "sum reduction (row_sum)" begin
        # 2-D tile of shape (1, 128). One block per row → sum across columns.
        nrows = 16
        ncols = 128
        # 2-D aligned array via reshape of an aligned flat buffer.
        raw = MLIRKernels.aligned_array(Float32, nrows * ncols; alignment=128)
        a2 = reshape(raw, (nrows, ncols))
        copyto!(a2, Float32.(reshape(1:(nrows*ncols), (nrows, ncols))))
        b = MLIRKernels.aligned_array(Float32, nrows; alignment=128)
        fill!(b, 0f0)

        k = MLIRKernels.cpu_function(row_sum_kernel, (a2, b))
        k(a2, b; blocks=nrows)
        @test b ≈ vec(sum(a2; dims=2))

        mlir = MLIRKernels.code_mlir(row_sum_kernel, (a2, b))
        @test occursin("vector.multi_reduction", mlir)
    end

    @testset "matmul Float32" begin
        # 64×64 × 64×64 with 16×16 inner tiles ⇒ 4×4 grid, 4 K-step.
        # Exercises 2-D scf.parallel, Intrinsics.mma → vector.contract,
        # Intrinsics.cldi → arith.ceildivsi, and memref-rooted size(A, k).
        #
        # Tile size 16 keeps the unrolled vector.contract body small
        # (~250 FMAs vs ~4 K at 64-tile) so clang -O2 finishes in ~1 s
        # instead of ~30 s. Still covers every walker path.
        M, N, K = 64, 64, 64
        BM, BN, BK = 16, 16, 16
        raw_a = MLIRKernels.aligned_array(Float32, M*K; alignment=128)
        raw_b = MLIRKernels.aligned_array(Float32, K*N; alignment=128)
        raw_c = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        A = reshape(raw_a, (M, K))
        B = reshape(raw_b, (K, N))
        C = reshape(raw_c, (M, N))
        copyto!(A, Float32.(reshape(1:M*K, (M, K)) ./ Float32(M*K)))
        copyto!(B, Float32.(reshape(1:K*N, (K, N)) ./ Float32(K*N)))
        fill!(C, 0f0)

        # Compile + launch via @parallel_for on a 2-D grid.
        MLIRKernels.@parallel_for blocks = (M ÷ BM, N ÷ BN) matmul_kernel(
            A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK))

        @test C ≈ A * B rtol=1e-4

        # Reflection: should emit vector.contract and ceildivsi.
        mlir = MLIRKernels.code_mlir(matmul_kernel,
            (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK));
            n_grid_dims=2)
        @test occursin("vector.contract", mlir)
        @test occursin("arith.ceildivsi", mlir)
        @test occursin("memref.dim", mlir)
    end

    @testset "matmul (register-tile rank-1, recommended for CPU)" begin
        # The BLAS-style microkernel pattern: a small (RM, 1) × (1, RN)
        # rank-1 outer-product accumulated over an explicit K-loop. The
        # accumulator stays in vector registers across iterations (LLVM
        # emits a `phi [RM x <RN x float>]`) — same structural shape as
        # OpenBLAS's Cooperlake sgemm_kernel inner loop, just at a smaller
        # register tile (16×16 = 16 zmm acc regs, vs BLAS's 12×32 = 24).
        #
        # Bench numbers (see bench/bench_matmul_reg.jl): at M=N=K=1024 this
        # pattern reaches ~65% of OpenBLAS — about 2.4× the multi-thousand-
        # FMA unrolled `vector.contract` path. Compile time is also
        # dramatically lower (~1 s vs ~30 s for big-tile contract).
        M, N, K = 128, 128, 128
        RM, RN = 16, 16
        raw_a = MLIRKernels.aligned_array(Float32, M*K; alignment=128)
        raw_b = MLIRKernels.aligned_array(Float32, K*N; alignment=128)
        raw_c = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        A = reshape(raw_a, (M, K))
        B = reshape(raw_b, (K, N))
        C = reshape(raw_c, (M, N))
        copyto!(A, Float32.(reshape(1:M*K, (M, K)) ./ Float32(M*K)))
        copyto!(B, Float32.(reshape(1:K*N, (K, N)) ./ Float32(K*N)))
        fill!(C, 0f0)

        MLIRKernels.@parallel_for blocks = (M ÷ RM, N ÷ RN) matmul_reg_kernel(
            A, B, C, ct.Constant(RM), ct.Constant(RN))

        @test C ≈ A * B rtol=1e-4

        # The LLVM IR for this kernel should be tight: a real K-loop with
        # a phi-carried accumulator + ~RM vector FMAs per iteration. Lock
        # in the property — if the IR balloons to thousands of FMAs (= the
        # contract path got selected somehow), this test catches it.
        llvm = MLIRKernels.code_llvm(matmul_reg_kernel,
            (A, B, C, ct.Constant(RM), ct.Constant(RN));
            n_grid_dims=2)
        # Phi node carrying the accumulator across the K-loop.
        @test occursin(r"phi\s+\[\d+\s*x\s*<\d+\s*x\s*float>\]", llvm)
        # And no unrolled-blob FMA explosion — well under 100 fmuladd ops.
        @test count(_ -> true, eachmatch(r"\bfmuladd\b", llvm)) < 100
    end

    @testset "batched matmul Float32" begin
        # 4×32×32 × 4×32×32 batched matmul with 2×16×16 inner tiles per block.
        # Exercises 3-D Intrinsics.mma → vector.contract with batched indexing
        # maps (4 iterators: b, m, n, k). Tile sizes shrunk from 32 → 16 to
        # keep clang -O2 fast (batched outer-product unrolling at 32-tile is
        # the costly path).
        M, N, K, Batch = 32, 32, 32, 4
        BM, BN, BK, BS = 16, 16, 16, 2
        A = MLIRKernels.aligned_array(Float32, (M, K, Batch); alignment=128)
        B = MLIRKernels.aligned_array(Float32, (K, N, Batch); alignment=128)
        C = MLIRKernels.aligned_array(Float32, (M, N, Batch); alignment=128)
        copyto!(A, Float32.(reshape(1:M*K*Batch, (M, K, Batch)) ./ Float32(M*K*Batch)))
        copyto!(B, Float32.(reshape(1:K*N*Batch, (K, N, Batch)) ./ Float32(K*N*Batch)))
        fill!(C, 0f0)

        MLIRKernels.@parallel_for blocks = (M ÷ BM, N ÷ BN, Batch ÷ BS) bmm_kernel(
            A, B, C,
            ct.Constant(BM), ct.Constant(BN), ct.Constant(BK), ct.Constant(BS))

        # Julia oracle: per-batch 2-D matmul.
        C_expected = similar(C)
        for b in 1:Batch
            C_expected[:, :, b] = A[:, :, b] * B[:, :, b]
        end
        @test C ≈ C_expected rtol=1e-3
        @test maximum(abs, C .- C_expected) < 1f-3

        # Reflection: should emit a batched vector.contract (4 iterator types,
        # 4-axis affine maps with a leading batch dim).
        mlir = MLIRKernels.code_mlir(bmm_kernel,
            (A, B, C,
             ct.Constant(BM), ct.Constant(BN), ct.Constant(BK), ct.Constant(BS));
            n_grid_dims=3)
        @test occursin("vector.contract", mlir)
        @test occursin("arith.ceildivsi", mlir)
    end

    @testset "softmax (1-D row per block, BLOCK_N=128)" begin
        # M rows, N cols. One block per row → row-softmax. The kernel loads a
        # (1, N) tile, reduces with max along dim=2 to (1, 1), broadcasts back
        # to (1, N) for the shift, applies math.exp, sums to (1, 1), divides
        # back to (1, N), and stores.
        M, N = 64, 128
        raw_a = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        raw_y = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        A = reshape(raw_a, (M, N))
        Y = reshape(raw_y, (M, N))
        copyto!(A, Float32.(reshape(1:M*N, (M, N)) ./ Float32(M*N)))
        fill!(Y, 0f0)

        MLIRKernels.@parallel_for blocks = M softmax_kernel(A, Y, ct.Constant(N))

        # Numerically-stable Julia oracle.
        m = maximum(A; dims=2)
        e = exp.(A .- m)
        Y_expected = e ./ sum(e; dims=2)
        @test Y ≈ Y_expected rtol=1e-4

        # Reflection: emitted MLIR should contain the new ops.
        mlir = MLIRKernels.code_mlir(softmax_kernel, (A, Y, ct.Constant(N)))
        @test occursin("vector.multi_reduction <maxnumf>", mlir)
        @test occursin("math.exp", mlir)
        @test occursin("arith.divf", mlir)
    end

    @testset "layernorm (forward, 1-D row, BLOCK_N=128)" begin
        # M rows, N cols. One block per row → row-layernorm. The kernel loads
        # a (1, N) tile, reduces with + along dim=2 to (1, 1) for mean, computes
        # Δ = x - μ broadcast back to (1, N), squares, reduces again for
        # variance, broadcasts the runtime `eps::Float32` scalar arg to (1, 1),
        # adds to variance, takes rsqrt, broadcasts to (1, N), and finally
        # multiplies elementwise with Δ.
        M, N = 64, 128
        eps = 1f-5
        raw_x = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        raw_y = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        X = reshape(raw_x, (M, N))
        Y = reshape(raw_y, (M, N))
        copyto!(X, Float32.(reshape(1:M*N, (M, N)) ./ Float32(M*N)))
        fill!(Y, 0f0)

        MLIRKernels.@parallel_for blocks = M layernorm_kernel(X, Y, eps,
                                                            ct.Constant(N))

        # Julia oracle: mean → Δ → variance → rsqrt → normalize.
        μ = sum(X; dims=2) ./ Float32(N)
        Δ = X .- μ
        σ² = sum(Δ .* Δ; dims=2) ./ Float32(N)
        Y_expected = Δ ./ sqrt.(σ² .+ eps)

        @test Y ≈ Y_expected rtol=1e-4
        @test all(isfinite, Y)
        @test maximum(abs, Y .- Y_expected) < 1f-4

        # Reflection: emitted MLIR should contain the new ops.
        mlir = MLIRKernels.code_mlir(layernorm_kernel, (X, Y, eps, ct.Constant(N)))
        @test occursin("math.rsqrt", mlir)
        @test occursin("vector.multi_reduction <add>", mlir)
        # Runtime Float32 scalar arg should appear in func signature.
        @test occursin("f32", mlir)
    end

    @testset "layernorm (backward, 1-D row, BLOCK_N=128)" begin
        # Gradient pass for the row-layernorm forward kernel. One block per
        # row. Given x and dy, recomputes per-row mean / variance / inverse
        # std and produces dx via the standard no-affine formula:
        #   dx = (dy - sum(dy)/N - x_hat * sum(dy * x_hat)/N) * inv_std
        # Exercises the forward's ops plus two additional row reductions and
        # a broadcast-multiply-subtract chain.
        M, N = 64, 128
        eps = 1f-5
        raw_x  = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        raw_dy = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        raw_dx = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        X  = reshape(raw_x,  (M, N))
        dY = reshape(raw_dy, (M, N))
        dX = reshape(raw_dx, (M, N))
        # Reproducible pseudo-random inputs via a fixed LCG (avoids touching
        # the global RNG used by neighboring rand/randn testsets).
        s = UInt32(0xCAFE_F00D)
        @inline _lcg32(z) = UInt32(((UInt64(z) * 1103515245) + 12345) & 0xFFFFFFFF)
        for k in 1:length(X)
            s = _lcg32(s); X[k]  = Float32(s) / Float32(typemax(UInt32)) * 2f0 - 1f0
            s = _lcg32(s); dY[k] = Float32(s) / Float32(typemax(UInt32)) * 2f0 - 1f0
        end
        fill!(dX, 0f0)

        MLIRKernels.@parallel_for blocks = M layernorm_bwd_kernel(X, dY, dX, eps,
                                                                ct.Constant(N))

        # Julia oracle: replicate the per-row backward math on host.
        nF = Float32(N)
        μ_h  = sum(X; dims=2) ./ nF
        Δ_h  = X .- μ_h
        σ²_h = sum(Δ_h .* Δ_h; dims=2) ./ nF
        inv_h = 1f0 ./ sqrt.(σ²_h .+ eps)
        x_hat_h = Δ_h .* inv_h
        sum1_h = sum(dY; dims=2)
        sum2_h = sum(dY .* x_hat_h; dims=2)
        dX_expected = (dY .- sum1_h ./ nF .- x_hat_h .* (sum2_h ./ nF)) .* inv_h

        @test dX ≈ dX_expected rtol=1e-3
        @test all(isfinite, dX)
        @test maximum(abs, dX .- dX_expected) < 1f-3

        # Reflection: same row-reduction / rsqrt machinery as forward, plus
        # additional reductions for sum(dy) and sum(dy * x_hat).
        mlir = MLIRKernels.code_mlir(layernorm_bwd_kernel,
                                   (X, dY, dX, eps, ct.Constant(N)))
        @test occursin("math.rsqrt", mlir)
        @test occursin("vector.multi_reduction <add>", mlir)
        @test occursin("f32", mlir)
    end

    @testset "vadd Float16" begin
        # Same shape as the F32 vadd; F16 only has ~3 sig figs of precision.
        n = 1024
        tile = 16
        a = MLIRKernels.aligned_array(Float16, n; alignment=32)
        b = MLIRKernels.aligned_array(Float16, n; alignment=32)
        c = MLIRKernels.aligned_array(Float16, n; alignment=32)
        copyto!(a, Float16.(1:n) ./ Float16(n))
        copyto!(b, Float16.(101:100+n) ./ Float16(n))
        fill!(c, Float16(0))

        k = MLIRKernels.cpu_function(vadd_kernel, (a, b, c, ct.Constant(tile)))
        k(a, b, c, ct.Constant(tile); blocks=cld(n, tile))

        @test Float32.(c) ≈ Float32.(a) .+ Float32.(b) rtol=1e-3

        mlir = MLIRKernels.code_mlir(vadd_kernel, (a, b, c, ct.Constant(tile)))
        @test occursin("vector<16xf16>", mlir)
        @test occursin("arith.addf", mlir)
    end

    @testset "vadd BFloat16" begin
        # BFloat16 = 8-bit exponent, 7-bit mantissa → ~2-3 sig figs of precision.
        n = 1024
        tile = 16
        a = MLIRKernels.aligned_array(BFloat16, n; alignment=32)
        b = MLIRKernels.aligned_array(BFloat16, n; alignment=32)
        c = MLIRKernels.aligned_array(BFloat16, n; alignment=32)
        copyto!(a, BFloat16.((1:n) ./ n))
        copyto!(b, BFloat16.((101:100+n) ./ n))
        fill!(c, BFloat16(0))

        k = MLIRKernels.cpu_function(vadd_kernel, (a, b, c, ct.Constant(tile)))
        k(a, b, c, ct.Constant(tile); blocks=cld(n, tile))

        @test Float32.(c) ≈ Float32.(a) .+ Float32.(b) rtol=1e-2

        mlir = MLIRKernels.code_mlir(vadd_kernel, (a, b, c, ct.Constant(tile)))
        @test occursin("vector<16xbf16>", mlir)
        @test occursin("arith.addf", mlir)
    end

    @testset "matmul BF16 × BF16 → F32 (MLIR only)" begin
        # Mixed-precision matmul: BFloat16 inputs, Float32 accumulator/output.
        # This is the canonical ML matmul shape. The MLIR side accepts a
        # vector.contract with bf16 lhs/rhs and an f32 acc/result directly
        # (no arith.extf needed) — verified here at the IR level. End-to-end
        # JIT execution of this kernel currently hangs in Julia's compile
        # pipeline (separate from MLIR), so we exercise only code emission.
        M, N, K = 32, 32, 32
        BM, BN, BK = 16, 16, 16
        raw_a = MLIRKernels.aligned_array(BFloat16, M*K; alignment=128)
        raw_b = MLIRKernels.aligned_array(BFloat16, K*N; alignment=128)
        raw_c = MLIRKernels.aligned_array(Float32, M*N; alignment=128)
        A = reshape(raw_a, (M, K))
        B = reshape(raw_b, (K, N))
        C = reshape(raw_c, (M, N))

        mlir = MLIRKernels.code_mlir(matmul_mixed_kernel,
            (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK));
            n_grid_dims=2)
        @test occursin("vector.contract", mlir)
        # Operand types should retain bf16, result type should be f32.
        @test occursin("vector<16x16xbf16>", mlir)
        @test occursin("vector<16x16xf32>", mlir)
    end

    @testset "vadd gather/scatter (1-D Float32)" begin
        # Identical to vadd_kernel but lowered through gather/scatter. The
        # walker handles Intrinsics.iota / .offset / .load_ptr_tko /
        # .store_ptr_tko (and Intrinsics.bitcast on the bounds-check mask
        # path). Indices are contiguous, so the result must equal a .+ b
        # bit-exactly for F32 add.
        n = 1024
        tile = 16
        a = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b = MLIRKernels.aligned_array(Float32, n; alignment=128)
        c = MLIRKernels.aligned_array(Float32, n; alignment=128)
        copyto!(a, Float32.(1:n))
        copyto!(b, Float32.(101:100+n))
        fill!(c, 0f0)

        k = MLIRKernels.cpu_function(vadd_gather_kernel,
                                   (a, b, c, ct.Constant(tile)))
        k(a, b, c, ct.Constant(tile); blocks=cld(n, tile))

        @test c ≈ a .+ b rtol=1e-5

        # Reflection: emitted MLIR should contain gather + scatter + step.
        mlir = MLIRKernels.code_mlir(vadd_gather_kernel,
                                   (a, b, c, ct.Constant(tile)))
        @test occursin("vector.gather", mlir)
        @test occursin("vector.scatter", mlir)
        @test occursin("vector.step", mlir)
    end

    @testset "vadd scatter-only (1-D Float32)" begin
        # Contiguous tile loads for inputs, ct.scatter for the output. Verifies
        # the scatter path independently of gather.
        n = 1024
        tile = 16
        a = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b = MLIRKernels.aligned_array(Float32, n; alignment=128)
        c = MLIRKernels.aligned_array(Float32, n; alignment=128)
        copyto!(a, Float32.(1:n))
        copyto!(b, Float32.(101:100+n))
        fill!(c, 0f0)

        k = MLIRKernels.cpu_function(vadd_scatter_only_kernel,
                                   (a, b, c, ct.Constant(tile)))
        k(a, b, c, ct.Constant(tile); blocks=cld(n, tile))

        @test c ≈ a .+ b rtol=1e-5

        mlir = MLIRKernels.code_mlir(vadd_scatter_only_kernel,
                                   (a, b, c, ct.Constant(tile)))
        @test occursin("vector.scatter", mlir)
        @test !occursin("vector.gather", mlir)  # loads stay contiguous
    end

    @testset "attention (single-head, BM=BN=D=32)" begin
        # Single-head, F32, non-flash attention. One block per BM-row tile of O.
        # The kernel loads the full K and V (BN equals the full sequence length),
        # computes S = Q @ K^T, applies a numerically stable row-softmax, and
        # finally O = softmax(S) @ V. Verifies that two vector.contract calls,
        # vector.transpose, and the softmax chain compose correctly in a single
        # kernel. F32 throughout — comparison uses rtol=1e-3 to leave slack for
        # the softmax + chained matmul rounding noise.
        #
        # Shrunk from BM=BN=D=64 → 32 to cut the two unrolled vector.contract
        # bodies from ~4 K FMAs to ~500. Same code paths exercised.
        BM, BN, D = 32, 32, 32
        M, N = 64, BN
        @assert N == BN
        @assert M % BM == 0

        raw_q = MLIRKernels.aligned_array(Float32, M*D; alignment=128)
        raw_k = MLIRKernels.aligned_array(Float32, N*D; alignment=128)
        raw_v = MLIRKernels.aligned_array(Float32, N*D; alignment=128)
        raw_o = MLIRKernels.aligned_array(Float32, M*D; alignment=128)
        Q = reshape(raw_q, (M, D))
        K = reshape(raw_k, (N, D))
        V = reshape(raw_v, (N, D))
        O = reshape(raw_o, (M, D))

        # Deterministic, well-scaled inputs (1/sqrt(D) normalisation keeps S
        # magnitudes near the softmax-stable regime).
        copyto!(Q, Float32.(reshape(sinpi.((1:M*D) ./ Float32(M*D)), (M, D))) ./
                   sqrt(Float32(D)))
        copyto!(K, Float32.(reshape(cospi.((1:N*D) ./ Float32(N*D)), (N, D))) ./
                   sqrt(Float32(D)))
        copyto!(V, Float32.(reshape((1:N*D) ./ Float32(N*D), (N, D))))
        fill!(O, 0f0)

        MLIRKernels.@parallel_for blocks = (M ÷ BM) attention_kernel(
            Q, K, V, O, ct.Constant(BM), ct.Constant(BN), ct.Constant(D))

        # Julia oracle.
        S = Q * K'
        mvals = maximum(S; dims=2)
        E = exp.(S .- mvals)
        P = E ./ sum(E; dims=2)
        O_expected = P * V

        @test O ≈ O_expected rtol=1e-3
        @test all(isfinite, O)

        # Reflection: emitted MLIR should contain vector.transpose plus two
        # vector.contract ops (one for Q*K^T, one for P*V) and a row-softmax.
        mlir = MLIRKernels.code_mlir(attention_kernel,
            (Q, K, V, O, ct.Constant(BM), ct.Constant(BN), ct.Constant(D)))
        @test occursin("vector.transpose", mlir)
        # Two distinct vector.contract sites.
        @test count(_ -> true,
                    eachmatch(r"vector\.contract", mlir)) == 2
        @test occursin("vector.multi_reduction <maxnumf>", mlir)
        @test occursin("math.exp", mlir)
        @test occursin("arith.divf", mlir)
    end

    @testset "flash attention (online softmax)" begin
        # Online-softmax flash attention. M output rows, N_KV keys/values, D
        # head dim. One block per BM-row tile of O; K and V are tiled along
        # the sequence axis with BN — N_KV/BN K/V tiles per block. The kernel
        # carries (m, l, o) as scf.for iter_args (three different vector
        # shapes), uses element-wise tile/tile `max.` (arith.maxnumf), a
        # `-Inf32` splat, and two math.fma sites for the rescale step.
        # Shrunk to keep clang -O2 fast — D=32 halves the inner contract size.
        BM, BN, D, N_KV = 16, 16, 32, 64
        M = 32
        @assert M % BM == 0
        @assert N_KV % BN == 0

        raw_q = MLIRKernels.aligned_array(Float32, M*D; alignment=128)
        raw_k = MLIRKernels.aligned_array(Float32, N_KV*D; alignment=128)
        raw_v = MLIRKernels.aligned_array(Float32, N_KV*D; alignment=128)
        raw_o = MLIRKernels.aligned_array(Float32, M*D; alignment=128)
        Q = reshape(raw_q, (M, D))
        K = reshape(raw_k, (N_KV, D))
        V = reshape(raw_v, (N_KV, D))
        O = reshape(raw_o, (M, D))

        # Same input shaping as the non-flash attention test — keeps S in the
        # softmax-stable range.
        copyto!(Q, Float32.(reshape(sinpi.((1:M*D) ./ Float32(M*D)), (M, D))) ./
                   sqrt(Float32(D)))
        copyto!(K, Float32.(reshape(cospi.((1:N_KV*D) ./ Float32(N_KV*D)),
                                    (N_KV, D))) ./
                   sqrt(Float32(D)))
        copyto!(V, Float32.(reshape((1:N_KV*D) ./ Float32(N_KV*D), (N_KV, D))))
        fill!(O, 0f0)

        MLIRKernels.@parallel_for blocks = (M ÷ BM) flash_attn_kernel(
            Q, K, V, O, ct.Constant(BM), ct.Constant(BN), ct.Constant(D),
            ct.Constant(N_KV))

        # Reference: standard (non-flash) attention.
        S = Q * K'
        mvals = maximum(S; dims=2)
        E = exp.(S .- mvals)
        P = E ./ sum(E; dims=2)
        O_expected = P * V

        @test O ≈ O_expected rtol=1e-3
        @test all(isfinite, O)
        # Online-softmax accumulation introduces some F32 rounding noise from
        # the chunked partial sums — tighter than rtol=1e-3 should still hold.
        @test maximum(abs, O .- O_expected) < 1f-4

        # Reflection: emitted MLIR should contain the new ops for this kernel:
        # arith.maxnumf (element-wise max), math.fma (two sites in the loop
        # body for α*l + sum(p) and α*o + p@v), and a -Inf32 splat constant.
        mlir = MLIRKernels.code_mlir(flash_attn_kernel,
            (Q, K, V, O, ct.Constant(BM), ct.Constant(BN), ct.Constant(D),
             ct.Constant(N_KV)))
        @test occursin("arith.maxnumf", mlir)
        @test occursin("math.fma", mlir)
        # -Inf32 splat for the initial running-max.
        @test occursin("0xFF800000", mlir)
        # Three iter_args: scf.for with a (vector<...x...xf32>, vector<...>,
        # vector<...>) result tuple.
        @test occursin(r"scf\.for[^\n]*iter_args", mlir)
        # Should still contain vector.transpose / two vector.contract sites
        # and a max-reduction.
        @test occursin("vector.transpose", mlir)
        @test count(_ -> true,
                    eachmatch(r"vector\.contract", mlir)) == 2
        @test occursin("vector.multi_reduction <maxnumf>", mlir)
    end

    @testset "row_sum BFloat16" begin
        # 2-D tile (1, 128) → sum across columns. Verifies that
        # vector.multi_reduction <add> lowers correctly for bf16.
        nrows = 16
        ncols = 128
        raw = MLIRKernels.aligned_array(BFloat16, nrows * ncols; alignment=128)
        a2  = reshape(raw, (nrows, ncols))
        # Keep row values small so the BF16-precision sum can stay representable.
        copyto!(a2, BFloat16.(reshape(1:(nrows*ncols), (nrows, ncols)) ./
                              (nrows * ncols)))
        b = MLIRKernels.aligned_array(BFloat16, nrows; alignment=128)
        fill!(b, BFloat16(0))

        k = MLIRKernels.cpu_function(row_sum_kernel_T, (a2, b))
        k(a2, b; blocks=nrows)
        expected = vec(sum(Float32.(a2); dims=2))
        @test Float32.(b) ≈ expected rtol=5e-2

        mlir = MLIRKernels.code_mlir(row_sum_kernel_T, (a2, b))
        @test occursin("vector.multi_reduction", mlir)
        @test occursin("bf16", mlir)
    end

    @testset "atomic_add (counter, Int32)" begin
        # One block per index in 1..N atomically adds 1 to counter[1].
        # Validates the scalar-index atomic_add path: cuTile's SCI emits
        # `Intrinsics.offset(arg.ptr, 0)` (a 0-D pointer tile) followed by
        # `Intrinsics.atomic_add(...)`; the walker maps the RMW to
        # `memref.atomic_rmw addi` on the underlying memref.
        n_blocks = 128
        counter = MLIRKernels.aligned_array(Int32, 1; alignment=128)
        counter[1] = Int32(0)
        k = MLIRKernels.cpu_function(atomic_count_kernel, (counter,))
        k(counter; blocks=n_blocks)
        @test counter[1] == Int32(n_blocks)

        # Reflection: MLIR should contain a single memref.atomic_rmw with
        # the `addi` kind keyword. The atomic op lowers to `llvm.atomicrmw
        # add … acq_rel` in the pipeline (verified end-to-end by the launch
        # check above).
        mlir = MLIRKernels.code_mlir(atomic_count_kernel, (counter,))
        @test occursin("memref.atomic_rmw addi", mlir)
        # Run a larger grid to stress the OMP fork/join + atomic contention.
        counter[1] = Int32(0)
        k(counter; blocks=10_000)
        @test counter[1] == Int32(10_000)
    end

    @testset "atomic_add (counter, Float32)" begin
        # Float atomic_add — cuTile picks `AtomicRMWMode.ADDF` for AbstractFloat
        # element types; the walker selects the `addf` kind keyword for
        # `memref.atomic_rmw`. Sum may have F32 rounding noise, so we compare
        # with a small absolute tolerance per block.
        n_blocks = 100
        val = 1.5f0
        out = MLIRKernels.aligned_array(Float32, 1; alignment=128)
        out[1] = 0f0
        k = MLIRKernels.cpu_function(atomic_count_f32_kernel, (out, val))
        k(out, val; blocks=n_blocks)
        @test out[1] ≈ Float32(n_blocks) * val rtol=1e-5

        mlir = MLIRKernels.code_mlir(atomic_count_f32_kernel, (out, val))
        @test occursin("memref.atomic_rmw addf", mlir)
    end

    @testset "atomic_max (Int32)" begin
        # Each block contributes its bid (1..n_blocks) to out[1] via atomic
        # signed max. Final value is the largest bid that ran. Walker maps
        # to `memref.atomic_rmw maxs` (signed-max kind).
        n_blocks = 128
        out = MLIRKernels.aligned_array(Int32, 1; alignment=128)
        out[1] = Int32(0)
        k = MLIRKernels.cpu_function(atomic_max_kernel, (out,))
        k(out; blocks=n_blocks)
        @test out[1] == Int32(n_blocks)

        mlir = MLIRKernels.code_mlir(atomic_max_kernel, (out,))
        @test occursin("memref.atomic_rmw maxs", mlir)
    end

    @testset "atomic_min (Int32)" begin
        # Inverse of atomic_max — start at a sentinel max value, take a
        # signed min over each block's bid. The minimum is `1` (bid is
        # 1-indexed). Walker maps to `memref.atomic_rmw mins`.
        n_blocks = 128
        out = MLIRKernels.aligned_array(Int32, 1; alignment=128)
        out[1] = typemax(Int32)
        k = MLIRKernels.cpu_function(atomic_min_kernel, (out,))
        k(out; blocks=n_blocks)
        @test out[1] == Int32(1)

        mlir = MLIRKernels.code_mlir(atomic_min_kernel, (out,))
        @test occursin("memref.atomic_rmw mins", mlir)
    end

    @testset "atomic_add (histogram)" begin
        # Per-block-index atomic accumulator: each block reads one element
        # from `values`, bins it modulo `n_buckets`, and bumps `counts`. The
        # bid → value path uses cuTile's scalar getindex (`values[bid]`),
        # which the SCI lowers via a (1,) partition_view + `Intrinsics.reshape
        # (() target shape)` + `Intrinsics.remi`. End state matches a Julia
        # host histogram.
        n_buckets = 16
        n_values = 1024
        values = MLIRKernels.aligned_array(Int32, n_values; alignment=128)
        counts = MLIRKernels.aligned_array(Int32, n_buckets; alignment=128)

        # Seed-stable RNG so the histogram is deterministic across runs.
        rng_state = UInt32(0x12345678)
        @inline function _lcg(s)
            # Numerical Recipes LCG (constants from Park/Miller-ish family —
            # only needs to be deterministic + spread across buckets).
            UInt32(((UInt64(s) * 1664525) + 1013904223) & 0xFFFFFFFF)
        end
        s = rng_state
        for i in 1:n_values
            s = _lcg(s)
            values[i] = Int32(s & 0x7FFFFFFF)
        end
        fill!(counts, Int32(0))

        k = MLIRKernels.cpu_function(atomic_hist_kernel,
                                   (values, counts, ct.Constant(n_buckets)))
        k(values, counts, ct.Constant(n_buckets); blocks=n_values)

        # Host oracle: same `rem` + `+ 1` recipe.
        expected = zeros(Int32, n_buckets)
        for v in values
            expected[mod(v, Int32(n_buckets)) + 1] += Int32(1)
        end
        @test counts == expected
        @test sum(counts) == Int32(n_values)

        mlir = MLIRKernels.code_mlir(atomic_hist_kernel,
            (values, counts, ct.Constant(n_buckets)))
        @test occursin("memref.atomic_rmw addi", mlir)
        @test occursin("arith.remsi", mlir)
    end

    @testset "atomic_or (Int32)" begin
        # OR-reduction over bids 1..n_blocks; final value is the bitwise OR
        # of all bids that ran (every block does run). Order-independent, so
        # the result is fully deterministic.
        n_blocks = 8
        out = MLIRKernels.aligned_array(Int32, 1; alignment=128)
        out[1] = Int32(0)
        k = MLIRKernels.cpu_function(atomic_or_kernel, (out,))
        k(out; blocks=n_blocks)
        expected = Int32(0)
        for i in 1:n_blocks
            expected = expected | Int32(i)
        end
        @test out[1] == expected

        mlir = MLIRKernels.code_mlir(atomic_or_kernel, (out,))
        @test occursin("memref.atomic_rmw ori", mlir)
    end

    @testset "atomic_and (Int32)" begin
        # AND-reduction: start from all-ones, AND in each bid. Order-
        # independent.
        n_blocks = 8
        out = MLIRKernels.aligned_array(Int32, 1; alignment=128)
        out[1] = typemax(Int32)
        k = MLIRKernels.cpu_function(atomic_and_kernel, (out,))
        k(out; blocks=n_blocks)
        expected = typemax(Int32)
        for i in 1:n_blocks
            expected = expected & Int32(i)
        end
        @test out[1] == expected

        mlir = MLIRKernels.code_mlir(atomic_and_kernel, (out,))
        @test occursin("memref.atomic_rmw andi", mlir)
    end

    @testset "atomic_xor (Int32)" begin
        # XOR-reduction: order-independent, even number of equal bits cancel.
        n_blocks = 8
        out = MLIRKernels.aligned_array(Int32, 1; alignment=128)
        out[1] = Int32(0)
        k = MLIRKernels.cpu_function(atomic_xor_kernel, (out,))
        k(out; blocks=n_blocks)
        expected = Int32(0)
        for i in 1:n_blocks
            expected = expected ⊻ Int32(i)
        end
        @test out[1] == expected

        # Upstream `memref.atomic_rmw` enum lacks an `xori` kind, so the
        # walker routes `atomic_xor` through `memref.generic_atomic_rmw`
        # with an `arith.xori` region body.
        mlir = MLIRKernels.code_mlir(atomic_xor_kernel, (out,))
        @test occursin("memref.generic_atomic_rmw", mlir)
        @test occursin("arith.xori", mlir)
    end

    @testset "atomic_xchg (Int32)" begin
        # Each block does an unconditional atomic store-and-swap. Final
        # value is the bid of *some* block — non-deterministic which, but
        # always in 1..n_blocks. Walker maps to `memref.atomic_rmw assign`
        # which lowers to `llvm.atomicrmw xchg`.
        n_blocks = 64
        out = MLIRKernels.aligned_array(Int32, 1; alignment=128)
        out[1] = Int32(0)
        k = MLIRKernels.cpu_function(atomic_xchg_kernel, (out,))
        k(out; blocks=n_blocks)
        @test 1 <= out[1] <= n_blocks

        mlir = MLIRKernels.code_mlir(atomic_xchg_kernel, (out,))
        @test occursin("memref.atomic_rmw assign", mlir)
    end

    @testset "atomic_cas (Int32)" begin
        # CAS-based one-shot election: the first block to land replaces the
        # initial `0` with its bid; subsequent CAS attempts see a non-zero
        # value and the compare fails, leaving the slot unchanged. Final
        # value is non-zero and in 1..n_blocks. Walker maps to
        # `memref.generic_atomic_rmw` with a compare/select region.
        n_blocks = 64
        out = MLIRKernels.aligned_array(Int32, 1; alignment=128)
        out[1] = Int32(0)
        k = MLIRKernels.cpu_function(atomic_cas_kernel, (out,))
        k(out; blocks=n_blocks)
        @test 1 <= out[1] <= n_blocks

        mlir = MLIRKernels.code_mlir(atomic_cas_kernel, (out,))
        @test occursin("memref.generic_atomic_rmw", mlir)
        @test occursin("memref.atomic_yield", mlir)
    end

    @testset "atomic_add counter+record (prior-value)" begin
        # Each block atomically increments `counter[1]` by 1 and uses the
        # returned `prior` value as a 0-based slot index into `bids` (we add
        # 1 for Julia's 1-based indexing). After N blocks:
        #   - counter[1] must equal N (no lost updates).
        #   - bids must be a permutation of 1..N (each block recorded its bid
        #     at a unique slot, since prior values are unique 0..N-1).
        # This exercises the *prior-value-returning* form of atomic_add
        # together with a scalar tile store at an atomic-derived index.
        for n_blocks in (64, 1024)
            counter = MLIRKernels.aligned_array(Int32, 1; alignment=128)
            counter[1] = Int32(0)
            bids = MLIRKernels.aligned_array(Int32, n_blocks; alignment=128)
            fill!(bids, Int32(0))
            k = MLIRKernels.cpu_function(atomic_counter_record_kernel,
                                        (counter, bids))
            k(counter, bids; blocks=n_blocks)
            @test counter[1] == Int32(n_blocks)
            # Every bid 1..n_blocks must appear exactly once.
            @test sort(bids) == collect(Int32, 1:n_blocks)
        end

        # Reflection: the kernel must contain a `memref.atomic_rmw addi`
        # and the prior value must feed back into a store (a `transfer_write`
        # whose index originates from the atomic's i32 result).
        counter = MLIRKernels.aligned_array(Int32, 1; alignment=128)
        bids = MLIRKernels.aligned_array(Int32, 64; alignment=128)
        mlir = MLIRKernels.code_mlir(atomic_counter_record_kernel, (counter, bids))
        @test occursin("memref.atomic_rmw addi", mlir)
        @test occursin("vector.transfer_write", mlir)
    end

    @testset "math intrinsics (sin/cos/log/tanh/floor/abs)" begin
        # Exercises math.sin / math.cos / math.log / math.tanh / math.floor /
        # math.absf in a single kernel. cuTile broadcasts each function over
        # the input tile element-wise via overlay.
        n = 1024
        tile = 16
        a = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b = MLIRKernels.aligned_array(Float32, n; alignment=128)
        copyto!(a, Float32.(range(0.1f0, stop=π, length=n)))
        fill!(b, 0f0)

        MLIRKernels.parallel_for(math_mix_kernel, (a, b, ct.Constant(tile));
                               blocks = cld(n, tile))
        # Reference: same Julia expression on host.
        ref = abs.(floor.(sin.(a) .+ tanh.(log.(a) .+ cos.(a))))
        @test b ≈ ref rtol=1e-4

        mlir = MLIRKernels.code_mlir(math_mix_kernel, (a, b, ct.Constant(tile)))
        for op in ("math.sin", "math.cos", "math.log", "math.tanh",
                   "math.floor", "math.absf")
            @test occursin(op, mlir)
        end
    end

    @testset "rand: uniform Float32" begin
        # Smoke test the Philox2x32-7 RNG. cuTile decomposes `rand(Float32, (tile,))`
        # to raw integer arithmetic on a per-stream `(counter, seed)` pair; the
        # seed is sourced from the implicit trailing `KernelState.seed` param
        # that MLIRKernels's `lower_to_mlir` adds after the user args. The launch
        # plumbs `rand(UInt32)` per call into that slot.
        n = 65536
        tile = 16
        out = MLIRKernels.aligned_array(Float32, n; alignment=128)
        fill!(out, 0f0)

        MLIRKernels.parallel_for(rand_uniform_kernel, (out, ct.Constant(tile));
                               blocks = cld(n, tile))

        @test minimum(out) >= 0f0
        @test maximum(out) <= 1f0
        # Statistical sanity checks: uniform [0, 1) has mean 1/2, variance 1/12.
        # With N=65536 samples, the standard error on the sample mean is
        # 1/sqrt(12 * N) ≈ 0.001, so the 0.02 mean tolerance is ~20σ — passes
        # well over 99.99% of the time. Variance tolerance similarly slack.
        @test abs(Statistics.mean(out) - 0.5f0) < 0.02
        @test abs(Statistics.var(out) - 1/12) < 0.01

        # Two consecutive launches must produce *different* outputs: the host
        # seeds the launch with `Base.rand(UInt32)` per call, so the streams
        # should diverge.
        out2 = MLIRKernels.aligned_array(Float32, n; alignment=128)
        fill!(out2, 0f0)
        MLIRKernels.parallel_for(rand_uniform_kernel, (out2, ct.Constant(tile));
                               blocks = cld(n, tile))
        @test out != out2

        # Reflection: MLIR must reference the trailing i32 seed param and the
        # Philox bit-mixing ops (xori on the key, muli widening into the
        # high-half product).
        mlir = MLIRKernels.code_mlir(rand_uniform_kernel, (out, ct.Constant(tile)))
        @test occursin("arith.xori", mlir)
        @test occursin("vector.shuffle", mlir)
        @test occursin("arith.uitofp", mlir)
    end

    @testset "rand: per-block stream divergence" begin
        # Within a single launch, every block should see a distinct Philox
        # stream — cuTile mixes the block id into the per-stream key (see
        # `rng_key` in cuTile.jl/src/language/random.jl). If the block-id mix
        # were broken, every block's tile would be identical. We launch with
        # 64 blocks of 16 elements each; the per-block tiles must not all be
        # equal.
        n = 64 * 16
        tile = 16
        out = MLIRKernels.aligned_array(Float32, n; alignment=128)
        fill!(out, 0f0)
        MLIRKernels.parallel_for(rand_uniform_kernel, (out, ct.Constant(tile));
                               blocks = cld(n, tile))
        # Reshape to (tile, n_blocks) and compare every pair: at most one pair
        # may happen to coincide (essentially never with random seed).
        per_block = reshape(out, (tile, n ÷ tile))
        n_eq_pairs = 0
        for i in 1:size(per_block, 2)
            for j in (i+1):size(per_block, 2)
                per_block[:, i] == per_block[:, j] && (n_eq_pairs += 1)
            end
        end
        @test n_eq_pairs == 0
    end

    @testset "randn: standard normal Float32" begin
        # `randn` reuses the Philox core plus Box-Muller (-2*log(U) and
        # sin/cos of 2π*V). Mean and variance over N samples have the same
        # error budget as the rand test: std-error ≈ 1/sqrt(N) for both,
        # the 0.05 tolerance is ample with N=65536.
        n = 65536
        tile = 16
        out = MLIRKernels.aligned_array(Float32, n; alignment=128)
        fill!(out, 0f0)

        MLIRKernels.parallel_for(rand_normal_kernel, (out, ct.Constant(tile));
                               blocks = cld(n, tile))

        @test abs(Statistics.mean(out)) < 0.05
        @test abs(Statistics.var(out) - 1f0) < 0.05
        # Distribution should be reasonable: minimum should be deeply negative,
        # maximum deeply positive — but neither saturating wildly. (Realistic
        # extrema for N=65536 standard-normal samples are about ±4.3σ.)
        @test minimum(out) < -2f0
        @test maximum(out) >  2f0

        mlir = MLIRKernels.code_mlir(rand_normal_kernel, (out, ct.Constant(tile)))
        # Box-Muller uses log, sqrt, sin, cos.
        @test occursin("math.log", mlir)
        @test occursin("math.sqrt", mlir)
        @test occursin("math.sin", mlir)
        @test occursin("math.cos", mlir)
    end

    @testset "dft (1-stage matrix DFT, Float32)" begin
        # Complex 1-stage DFT via two real matmuls per part. Verifies:
        #   • `ct.extract` on a 3-D tile splits real/imag halves correctly
        #     (lowers to `vector.extract_strided_slice` with offsets [0,…] /
        #     [1, 0, …] along the leading MLIR dim).
        #   • Two `vector.contract` sites (real-from-W_r×X_r etc.) with
        #     asymmetric shapes ((N, N) × (N, BS) → (N, BS)).
        #   • `ct.cat` along the leading dim re-packs the (1, N, BS) r/i tiles
        #     into a (2, N, BS) packed result for store.
        N = 16
        BS = 2
        X = MLIRKernels.aligned_array(Float32, (2, N, BS); alignment=128)
        Y = MLIRKernels.aligned_array(Float32, (2, N, BS); alignment=128)
        W = MLIRKernels.aligned_array(Float32, (2, N, N); alignment=128)

        # Deterministic complex input.
        for b in 1:BS, n in 1:N
            X[1, n, b] = Float32(n + (b - 1) * N) / Float32(N * BS)
            X[2, n, b] = Float32(b + (n - 1) * BS) / Float32(N * BS)
        end
        # DFT matrix W[k+1, n+1] = exp(-2πi*k*n/N), real & imag stacked.
        for k in 0:N-1, n in 0:N-1
            v = exp(-2π * im * k * n / N)
            W[1, k+1, n+1] = Float32(real(v))
            W[2, k+1, n+1] = Float32(imag(v))
        end
        fill!(Y, 0f0)

        MLIRKernels.parallel_for(dft_kernel,
            (X, Y, W, ct.Constant(N), ct.Constant(BS)); blocks=1)

        # Oracle: hand-rolled DFT via the same W matrix.
        x_cplx = ComplexF32.(view(X, 1, :, :)) .+ im .* ComplexF32.(view(X, 2, :, :))
        y_ref = _dft_ref(x_cplx)
        y_got = ComplexF32.(view(Y, 1, :, :)) .+ im .* ComplexF32.(view(Y, 2, :, :))

        @test y_got ≈ y_ref rtol=1e-3
        @test maximum(abs, y_got .- y_ref) < 1f-4
        @test all(isfinite, Y)

        # Reflection: walker should emit extract_strided_slice for the r/i
        # split, two vector.contract sites for the two real matmul products,
        # and a final insert_strided_slice (from the leading-axis cat).
        mlir = MLIRKernels.code_mlir(dft_kernel,
            (X, Y, W, ct.Constant(N), ct.Constant(BS)))
        @test occursin("vector.extract_strided_slice", mlir)
        @test occursin("vector.insert_strided_slice", mlir)
        @test count(_ -> true,
                    eachmatch(r"vector\.contract", mlir)) == 4  # 4 real matmuls
    end

    @testset "fft (3-stage Cooley-Tukey, Float32)" begin
        # 3-stage Cooley-Tukey FFT, factors (2, 2, 2) → N=8. Mirrors the canonical
        # cuTile.jl/examples/fft.jl. Verifies the end-to-end FFT pipeline:
        #   • 4-D reshape + `permutedims` (vector.transpose on rank-4 vectors)
        #     between the F0/F1/F2 stages.
        #   • Rank-3 batched mma with a broadcasting batch dim (W matrices have
        #     trailing batch dim 1, X has BS).
        #   • Twiddle-factor element-wise multiplies between stages.
        F0, F1, F2 = 2, 2, 2
        N = F0 * F1 * F2
        F0F1 = F0 * F1; F1F2 = F1 * F2; F0F2 = F0 * F2
        BS = 2; D = 2; N2D = N * 2 ÷ D

        x_in  = MLIRKernels.aligned_array(Float32, (D, N2D, BS); alignment=128)
        y_out = MLIRKernels.aligned_array(Float32, (D, N2D, BS); alignment=128)
        W0_   = MLIRKernels.aligned_array(Float32, (2, F0, F0); alignment=128)
        W1_   = MLIRKernels.aligned_array(Float32, (2, F1, F1); alignment=128)
        W2_   = MLIRKernels.aligned_array(Float32, (2, F2, F2); alignment=128)
        T0_   = MLIRKernels.aligned_array(Float32, (2, F1F2, F0); alignment=128)
        T1_   = MLIRKernels.aligned_array(Float32, (2, F2, F1); alignment=128)

        copyto!(W0_, _dft_matrix(F0))
        copyto!(W1_, _dft_matrix(F1))
        copyto!(W2_, _dft_matrix(F2))
        copyto!(T0_, _twiddles_T0(F0, F1F2, N))
        copyto!(T1_, _twiddles_T1(F1, F2, F1F2))

        # Reproducible random input (no Random.seed! to avoid affecting RNG
        # tests above) — use a fixed LCG instead.
        input_cplx = zeros(ComplexF32, N, BS)
        s = UInt32(0xC0DE_FFFF)
        @inline _lcg(z) = UInt32(((UInt64(z) * 1103515245) + 12345) & 0xFFFFFFFF)
        for k in 1:length(input_cplx)
            s = _lcg(s); r = Float32(s) / Float32(typemax(UInt32)) * 2f0 - 1f0
            s = _lcg(s); i = Float32(s) / Float32(typemax(UInt32)) * 2f0 - 1f0
            input_cplx[k] = ComplexF32(r, i)
        end
        # Pack: ComplexF32 (N, BS) ↔ Float32 (2, N, BS).
        x_ri = reinterpret(reshape, Float32, input_cplx)
        if D == 2
            copyto!(x_in, x_ri)
        else
            copyto!(x_in, reshape(x_ri, D, N2D, BS))
        end
        fill!(y_out, 0f0)

        MLIRKernels.parallel_for(fft_kernel,
            (x_in, y_out, W0_, W1_, W2_, T0_, T1_,
             ct.Constant(N), ct.Constant(F0), ct.Constant(F1), ct.Constant(F2),
             ct.Constant(F0F1), ct.Constant(F1F2), ct.Constant(F0F2),
             ct.Constant(BS), ct.Constant(D), ct.Constant(N2D));
            blocks=1)

        # Unpack output and compare with hand-rolled DFT.
        y_ri = D == 2 ? y_out : reshape(y_out, 2, N, BS)
        y_cplx = collect(reinterpret(reshape, ComplexF32, y_ri))
        y_ref  = _dft_ref(input_cplx)

        @test y_cplx ≈ y_ref rtol=1e-3
        @test maximum(abs, y_cplx .- y_ref) < 1f-4
        @test all(isfinite, y_out)

        # Reflection: should contain the new ops + several contracts/transposes.
        mlir = MLIRKernels.code_mlir(fft_kernel,
            (x_in, y_out, W0_, W1_, W2_, T0_, T1_,
             ct.Constant(N), ct.Constant(F0), ct.Constant(F1), ct.Constant(F2),
             ct.Constant(F0F1), ct.Constant(F1F2), ct.Constant(F0F2),
             ct.Constant(BS), ct.Constant(D), ct.Constant(N2D)))
        @test occursin("vector.extract_strided_slice", mlir)
        @test occursin("vector.insert_strided_slice", mlir)
        @test occursin("vector.transpose", mlir)
        # Three stages × four real matmuls (W_r×X_r, W_i×X_i, W_r×X_i, W_i×X_r)
        # = 12 vector.contract sites.
        @test count(_ -> true,
                    eachmatch(r"vector\.contract", mlir)) == 12
    end

    @testset "moe routing (per-block expert dispatch)" begin
        # Simplified Mixture-of-Experts: one block per token, each token has a
        # pre-assigned expert id. Each block atomically claims a slot in its
        # expert's region of `Y`, loads the token, the expert's weight matrix,
        # multiplies, and writes the result to the slot. Exercises end-to-end
        # composition of: scalar-indexed Int32 tile load (`expert_ids[bid]`),
        # atomic_add with prior-value return, scalar tile store at a runtime
        # index (`slot_tokens[slot_in_y] = bid`), a 3-D weight-tensor load
        # whose trailing index is the runtime expert id, reshape (3-D → 2-D),
        # and a (D_out, D) × (D, 1) matmul to a 2-D output column.
        num_tokens = 8
        num_experts = 2
        D = 32
        D_out = 32
        MAX_PER_EXPERT = 8

        X = MLIRKernels.aligned_array(Float32, (D, num_tokens); alignment=128)
        Y = MLIRKernels.aligned_array(Float32, (D_out, MAX_PER_EXPERT * num_experts);
                                    alignment=128)
        expert_ids = MLIRKernels.aligned_array(Int32, num_tokens; alignment=128)
        counters = MLIRKernels.aligned_array(Int32, num_experts; alignment=128)
        slot_tokens = MLIRKernels.aligned_array(Int32, MAX_PER_EXPERT * num_experts;
                                              alignment=128)
        Wexp = MLIRKernels.aligned_array(Float32, (D_out, D, num_experts);
                                       alignment=128)

        # Deterministic inputs.
        copyto!(X, Float32.(reshape(1:D*num_tokens, (D, num_tokens)) ./
                            Float32(D*num_tokens)))
        copyto!(Wexp, Float32.(reshape(1:D_out*D*num_experts, (D_out, D, num_experts)) ./
                               Float32(D_out*D*num_experts)))

        # Round-robin expert assignment: token i → expert ((i-1) mod E) + 1.
        # 4 tokens per expert, MAX_PER_EXPERT=8 leaves slack.
        for i in 1:num_tokens
            expert_ids[i] = Int32(((i - 1) % num_experts) + 1)
        end
        fill!(counters, Int32(0))
        fill!(slot_tokens, Int32(0))
        fill!(Y, 0f0)

        MLIRKernels.@parallel_for blocks = num_tokens moe_routing_kernel(
            X, Y, expert_ids, counters, slot_tokens, Wexp,
            ct.Constant(D), ct.Constant(D_out), ct.Constant(MAX_PER_EXPERT))

        # Counters must match the host-computed per-expert token counts.
        expected_counts = zeros(Int32, num_experts)
        for i in 1:num_tokens
            expected_counts[expert_ids[i]] += Int32(1)
        end
        @test counters == expected_counts
        # Total tokens routed == num_tokens (no lost updates under contention).
        @test sum(counters) == Int32(num_tokens)

        # Each slot's recorded token id must belong to that expert. Within an
        # expert's region (positions 1..counters[e]) every routed token appears
        # exactly once.
        for e in 1:num_experts
            lo = (e-1)*MAX_PER_EXPERT + 1
            hi = (e-1)*MAX_PER_EXPERT + counters[e]
            tokens_in_e = sort(Int.(slot_tokens[lo:hi]))
            expected_tokens = sort([i for i in 1:num_tokens if expert_ids[i] == Int32(e)])
            @test tokens_in_e == expected_tokens
        end

        # Per-slot numerical check: Y[:, slot] == Wexp[:,:,e] * X[:, token].
        for e in 1:num_experts
            for k_slot in 1:Int(counters[e])
                slot_idx = (e - 1) * MAX_PER_EXPERT + k_slot
                tok = Int(slot_tokens[slot_idx])
                Y_ref = Wexp[:, :, e] * X[:, tok]
                Y_got = Y[:, slot_idx]
                @test Y_got ≈ Y_ref rtol=1e-4
            end
        end

        # Unused slots in each expert's region remain at their initial 0f0.
        for e in 1:num_experts
            base = (e - 1) * MAX_PER_EXPERT
            for k_slot in (counters[e] + 1):MAX_PER_EXPERT
                @test all(==(0f0), Y[:, base + k_slot])
            end
        end

        # Reflection: MLIR should contain the key ops for this kernel.
        # The atomic_add → memref.atomic_rmw addi (prior-value), the matmul
        # (vector.contract), the 3-D weight load with a 1-thick trailing dim
        # (vector<1x32x32xf32>), and a final 2-D vector.transfer_write at a
        # runtime column.
        mlir = MLIRKernels.code_mlir(moe_routing_kernel,
            (X, Y, expert_ids, counters, slot_tokens, Wexp,
             ct.Constant(D), ct.Constant(D_out), ct.Constant(MAX_PER_EXPERT)))
        @test occursin("memref.atomic_rmw addi", mlir)
        @test occursin("vector.contract", mlir)
        # The 3-D weight load has a trailing-dim-1 expert slice — should be a
        # vector.transfer_read producing a vector<1x...x...xf32>.
        @test occursin("vector.transfer_read", mlir)
        @test occursin("vector.transfer_write", mlir)
        # Reshape (3-D → 2-D) appears as vector.shape_cast.
        @test occursin("vector.shape_cast", mlir)
    end

    @testset "moe routing (uneven assignment, larger grid)" begin
        # Same kernel as above but with a non-round-robin (deliberately uneven)
        # assignment and a larger token count to stress atomic contention.
        # `MAX_PER_EXPERT` is chosen as the upper bound of any single expert's
        # token count so the per-expert region in Y is large enough.
        num_tokens = 64
        num_experts = 4
        D = 32
        D_out = 32
        MAX_PER_EXPERT = num_tokens  # > any single expert's worst-case count

        X = MLIRKernels.aligned_array(Float32, (D, num_tokens); alignment=128)
        Y = MLIRKernels.aligned_array(Float32, (D_out, MAX_PER_EXPERT * num_experts);
                                    alignment=128)
        expert_ids = MLIRKernels.aligned_array(Int32, num_tokens; alignment=128)
        counters = MLIRKernels.aligned_array(Int32, num_experts; alignment=128)
        slot_tokens = MLIRKernels.aligned_array(Int32, MAX_PER_EXPERT * num_experts;
                                              alignment=128)
        Wexp = MLIRKernels.aligned_array(Float32, (D_out, D, num_experts);
                                       alignment=128)

        copyto!(X, Float32.(reshape(1:D*num_tokens, (D, num_tokens)) ./
                            Float32(D*num_tokens)))
        copyto!(Wexp, Float32.(reshape(1:D_out*D*num_experts, (D_out, D, num_experts)) ./
                               Float32(D_out*D*num_experts)))

        # Deterministic but uneven distribution: tokens 1..40 → expert 1, then
        # cycle 2, 3, 4 for the rest. Stresses atomic contention on expert 1.
        for i in 1:num_tokens
            expert_ids[i] = i <= 40 ? Int32(1) : Int32(2 + ((i - 41) % 3))
        end
        fill!(counters, Int32(0))
        fill!(slot_tokens, Int32(0))
        fill!(Y, 0f0)

        MLIRKernels.@parallel_for blocks = num_tokens moe_routing_kernel(
            X, Y, expert_ids, counters, slot_tokens, Wexp,
            ct.Constant(D), ct.Constant(D_out), ct.Constant(MAX_PER_EXPERT))

        expected_counts = zeros(Int32, num_experts)
        for i in 1:num_tokens
            expected_counts[expert_ids[i]] += Int32(1)
        end
        @test counters == expected_counts
        @test sum(counters) == Int32(num_tokens)

        for e in 1:num_experts
            lo = (e-1)*MAX_PER_EXPERT + 1
            hi = (e-1)*MAX_PER_EXPERT + counters[e]
            tokens_in_e = sort(Int.(slot_tokens[lo:hi]))
            expected_tokens = sort([i for i in 1:num_tokens if expert_ids[i] == Int32(e)])
            @test tokens_in_e == expected_tokens
            for k_slot in 1:Int(counters[e])
                slot_idx = (e - 1) * MAX_PER_EXPERT + k_slot
                tok = Int(slot_tokens[slot_idx])
                Y_ref = Wexp[:, :, e] * X[:, tok]
                Y_got = Y[:, slot_idx]
                @test Y_got ≈ Y_ref rtol=1e-4
            end
        end
    end

    @testset "matrix transpose" begin
        # Non-square 64 × 32 transpose with 16 × 16 tiles. 4 × 2 grid.
        # Verifies permutedims(tile, (2,1)) + the non-square index swap path.
        # Tile shrunk from 32 → 16 to keep vector.transpose cheap.
        BM, BN = 16, 16
        M, N = 64, 32
        A = MLIRKernels.aligned_array(Float32, M, N; alignment=128)
        B = MLIRKernels.aligned_array(Float32, N, M; alignment=128)
        copyto!(A, rand(Float32, M, N))
        fill!(B, 0f0)

        MLIRKernels.parallel_for(transpose_kernel,
                               (A, B, ct.Constant(BM), ct.Constant(BN));
                               blocks = (M ÷ BM, N ÷ BN))
        @test B == permutedims(A, (2, 1))

        mlir = MLIRKernels.code_mlir(transpose_kernel,
                                   (A, B, ct.Constant(BM), ct.Constant(BN)))
        @test occursin("vector.transpose", mlir)
    end

    @testset "SPMD: vadd" begin
        # ISPC-style SPMD mode: the kernel is plain Julia (no Tile/ct.* types,
        # no `ct.bid()`). The trailing `i::Int` arg is the lane index and the
        # walker lifts the body to `LANE_WIDTH`-wide vectors. Each grid block
        # processes 16 consecutive `i`-values in parallel.
        n = 1024
        lane_width = 16
        # Plain Vector{Float32} — SPMD mode doesn't impose an ArraySpec
        # alignment requirement.
        a = rand(Float32, n)
        b = rand(Float32, n)
        c = zeros(Float32, n)

        k = MLIRKernels.spmd_function(vadd_spmd,
            (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
            lane_width)
        # The `0` in the launch is a placeholder — the lane index is
        # synthesised inside the kernel, not read from this arg.
        k(a, b, c, 0; blocks = cld(n, lane_width))
        @test c ≈ a .+ b

        # Reflection: the emitted MLIR should contain the vector arith op and
        # one of the vector memory ops (transfer_read / gather) at the lane
        # width.
        mlir = MLIRKernels.code_mlir(vadd_spmd,
            (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
            spmd=true, lane_width)
        @test occursin("vector<16xf32>", mlir)
        @test occursin("arith.addf", mlir)
        @test occursin("vector.transfer_read", mlir) ||
              occursin("vector.gather", mlir)
        @test occursin("vector.transfer_write", mlir) ||
              occursin("vector.scatter", mlir)
        @test occursin("scf.parallel", mlir)
        # No Tile types / ArraySpec strided layout in SPMD memrefs.
        @test !occursin("strided<", mlir)
    end

    @testset "SPMD: vadd with alignment=128" begin
        # Same kernel as the basic SPMD vadd test, but with the explicit
        # `alignment=128` kwarg. The walker emits memref.assume_alignment
        # and a strided<[1]> layout — matching what the cuTile (TileArray)
        # path gets from ArraySpec. Closes the DRAM-scale perf gap between
        # SPMD and tile (see bench/bench_spmd.jl). Caller is responsible
        # for supplying aligned buffers (aligned_array).
        n = 1024
        lane_width = 16
        a = MLIRKernels.aligned_array(Float32, n; alignment=128)
        b = MLIRKernels.aligned_array(Float32, n; alignment=128)
        c = MLIRKernels.aligned_array(Float32, n; alignment=128)
        copyto!(a, rand(Float32, n))
        copyto!(b, rand(Float32, n))

        k = MLIRKernels.spmd_function(vadd_spmd,
            (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
            lane_width, alignment=128)
        k(a, b, c, 0; blocks = cld(n, lane_width))
        @test c ≈ a .+ b

        # MLIR should now contain the alignment hint AND a strided<[1]>
        # layout on the memref args.
        mlir = MLIRKernels.code_mlir(vadd_spmd,
            (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
            spmd=true, lane_width, alignment=128)
        @test occursin("memref.assume_alignment", mlir)
        @test occursin("strided<[1]>", mlir)

        # And the launcher must reject a misaligned buffer.
        bad = Vector{Float32}(undef, n)   # 16-byte aligned, not 128
        if UInt(pointer(bad)) % 128 != 0
            @test_throws ErrorException k(bad, b, c, 0;
                                          blocks = cld(n, lane_width))
        end
    end

    # KernelAbstractions CPU backend (MLIRBackend <: KA.GPU). Guarded on KA
    # being loadable: the package's own test env (--project=.) has KA only as
    # a weakdep, so this skips there and runs whenever KA is present (e.g. an
    # env that adds it). Validates the fully cuTile-decoupled KA path: KA
    # @kernel → Frontend.structured (own interpreter/intrinsics) → MLIR → clang.
    @testset "KA: vadd via MLIRBackend (CPU, decoupled)" begin
        ka_loaded = try
            @eval using KernelAbstractions
            true
        catch
            false
        end
        if !ka_loaded
            @info "KernelAbstractions not in this env — skipping KA backend test"
            @test true  # placeholder so the testset is non-empty
        else
            KA = KernelAbstractions
            KAExt = Base.get_extension(MLIRKernels, :KernelAbstractionsExt)
            Backend = KAExt.MLIRBackend
            @eval begin
                @kernel function _ka_vadd!(C, A, B)
                    i = @index(Global, Linear)
                    @inbounds C[i] = A[i] + B[i]
                end
            end
            N = 4096; W = 16
            A = MLIRKernels.aligned_array(Float32, N; alignment=128); copyto!(A, rand(Float32, N))
            B = MLIRKernels.aligned_array(Float32, N; alignment=128); copyto!(B, rand(Float32, N))
            C = MLIRKernels.aligned_array(Float32, N; alignment=128); fill!(C, 0f0)
            (@eval _ka_vadd!)(Backend(), W)(C, A, B; ndrange=N)
            @test C ≈ A .+ B
            # The @noinline global_index marker must survive inference under
            # the Frontend interpreter (default opt params) — i.e. appear as a
            # call in the SCI, not be inlined/folded away.
            gpu_body = @eval gpu__ka_vadd!
            ctxT = let
                ndr = KA.NDIteration.StaticSize{(N,)}
                wg  = KA.NDIteration.StaticSize{(W,)}
                grp = KA.NDIteration.StaticSize{(N ÷ W,)}
                ndro = KA.NDIteration.NDRange{1, grp, wg, Nothing, Nothing}
                KA.CompilerMetadata{ndr, KA.NDIteration.NoDynamicCheck, Nothing, Nothing, ndro}
            end
            sci, rt = MLIRKernels.Frontend.structured(gpu_body,
                Tuple{ctxT, Vector{Float32}, Vector{Float32}, Vector{Float32}})
            @test rt === Nothing
            @test occursin("global_index", sprint(show, sci))
        end
    end

    # KA.@atomic — KernelAbstractions' *portable* atomic (= Atomix.@atomic,
    # which CUDA/AMDGPU/oneAPI all override for device arrays). The KA extension
    # overlays `Atomix.modify!(IndexableRef, op, x, ord)` onto the Frontend
    # `atomic_index!` marker, which the walker lowers to `memref.atomic_rmw`.
    # Covers: varying-index float add (per-lane scatter), uniform-slot float add
    # and integer max/min (lane-reduction → single atomic per block), atomicity
    # under cross-block contention, and the float-min/max MLIR-version gate.
    @testset "KA: @atomic via MLIRBackend (Atomix portable path)" begin
        # The portable atomic is Atomix's `@atomic` — KA itself just re-exports
        # it (`import Atomix: @atomic`, cf. KernelAbstractions/examples/
        # histogram.jl), and our KA extension overlays `Atomix.modify!` onto the
        # Frontend `atomic_index!` marker. We import it straight from Atomix so
        # bare `@atomic` in a kernel is unambiguously the portable atomic, NOT
        # Base's scalar `@atomic`.
        ka_loaded = try
            @eval using KernelAbstractions
            @eval using Atomix: @atomic
            true
        catch
            false
        end
        if !ka_loaded
            @info "KernelAbstractions not in this env — skipping KA.@atomic test"
            @test true
        else
            KA = KernelAbstractions
            Backend = Base.get_extension(MLIRKernels, :KernelAbstractionsExt).MLIRBackend
            W = 16
            @eval begin
                @kernel function _ka_hist!(bins, @Const(idx))
                    i = @index(Global, Linear)
                    @inbounds @atomic bins[idx[i]] += 1f0
                end
                @kernel function _ka_amax!(out, @Const(x))
                    i = @index(Global, Linear)
                    @inbounds @atomic out[1] max x[i]
                end
                @kernel function _ka_amin!(out, @Const(x))
                    i = @index(Global, Linear)
                    @inbounds @atomic out[1] min x[i]
                end
            end

            # (a) Histogram: varying per-lane index → per-lane atomic scatter.
            N = 4096; NB = 8
            idx  = Int32[(j % NB) + 1 for j in 0:N-1]
            bins = MLIRKernels.aligned_array(Float32, NB; alignment=128); fill!(bins, 0f0)
            (@eval _ka_hist!)(Backend(), W)(bins, idx; ndrange=N)
            @test all(==(Float32(N ÷ NB)), bins)

            # (b) Atomicity: every lane → one slot; no lost updates across blocks.
            M = 65536
            ones_idx = ones(Int32, M)
            acc = MLIRKernels.aligned_array(Float32, 1; alignment=128); fill!(acc, 0f0)
            (@eval _ka_hist!)(Backend(), W)(acc, ones_idx; ndrange=M)
            @test acc[1] == Float32(M)

            # (c) Integer max/min into a uniform slot → vector.reduction + one
            #     atomic per block (maxs/mins lower on every supported MLIR).
            Ni = 256
            xi = Int32.(collect(1:Ni)); xi[100] = Int32(9999)
            omax = MLIRKernels.aligned_array(Int32, 1; alignment=128); omax[1] = typemin(Int32)
            omin = MLIRKernels.aligned_array(Int32, 1; alignment=128); omin[1] = typemax(Int32)
            (@eval _ka_amax!)(Backend(), W)(omax, xi; ndrange=Ni)
            (@eval _ka_amin!)(Backend(), W)(omin, xi; ndrange=Ni)
            @test omax[1] == 9999
            @test omin[1] == 1

            # (d) The Atomix overlay → atomic_index! marker must survive inference
            #     (DCE would otherwise delete the unused-result call).
            gpu_body = @eval gpu__ka_hist!
            ctxT = let
                ndr = KA.NDIteration.StaticSize{(N,)}; wg = KA.NDIteration.StaticSize{(W,)}
                grp = KA.NDIteration.StaticSize{(N ÷ W,)}
                ndro = KA.NDIteration.NDRange{1, grp, wg, Nothing, Nothing}
                KA.CompilerMetadata{ndr, KA.NDIteration.NoDynamicCheck, Nothing, Nothing, ndro}
            end
            sci, rt = MLIRKernels.Frontend.structured(gpu_body,
                Tuple{ctxT, Vector{Float32}, Vector{Int32}})
            @test occursin("atomic_index!", sprint(show, sci))

            # (e) Float min/max atomics need MLIR ≥ 21 (LLVM 20's memref→llvm
            #     doesn't lower `maxnumf`/`minnumf`). On older MLIR the walker
            #     raises a clear, actionable error rather than emitting IR that
            #     dies at LLVM translation; on MLIR ≥ 21 it just works.
            xf = Float32.(collect(1:Ni)); xf[100] = 9999f0
            of = MLIRKernels.aligned_array(Float32, 1; alignment=128); of[1] = -Inf32
            if MLIRKernels.MLIR.MLIR_VERSION[] < v"21"
                @test_throws Exception (@eval _ka_amax!)(Backend(), W)(of, xf; ndrange=Ni)
            else
                (@eval _ka_amax!)(Backend(), W)(of, xf; ndrange=Ni)
                @test of[1] == 9999f0
            end

            # (f) Counter idiom `@atomic out[1] += c` with a UNIFORM scalar
            #     value. Each of the W lanes runs the statement, so the slot must
            #     gain W*c per block (== ndrange total). A naive single atomic per
            #     block would undercount by exactly W — guard against that
            #     regression (the value is broadcast to W lanes then reduced).
            @eval begin
                @kernel function _ka_ctr!(out)
                    i = @index(Global, Linear)
                    @inbounds @atomic out[1] += 1f0
                end
            end
            ctr = MLIRKernels.aligned_array(Float32, 1; alignment=128); ctr[1] = 0f0
            (@eval _ka_ctr!)(Backend(), W)(ctr; ndrange=256)
            @test ctr[1] == 256f0   # NOT 256/W

            # (g) The backend lowers a 1-D grid/workgroup; multi-dimensional
            #     ndrange or workgroupsize would silently corrupt Local/Group
            #     indices, so the launcher must reject them. And a launch-time
            #     workgroupsize that conflicts with a static one must error
            #     rather than be silently ignored.
            @test_throws Exception (@eval _ka_ctr!)(Backend(), W)(ctr; ndrange=(8, 4))
            @test_throws Exception (@eval _ka_ctr!)(Backend(), W)(ctr; ndrange=256, workgroupsize=2W)
        end
    end

    # Multi-dimensional support: N-D `@index(Global, NTuple)` + N-D array
    # indexing `A[i,j]`. The workgroup is flattened to a 1-D lane vector, per-dim
    # coords reconstructed by column-major unflatten, and `A[i,j]` linearised
    # (column-major) to a gather/scatter over a flattened (`reinterpret_cast`)
    # rank-1 view. 2-D transpose (KA's `naive_transpose`) is the end-to-end gate.
    @testset "KA: multi-dim @index(Global, NTuple) + A[i,j]" begin
        ka_loaded = try; @eval using KernelAbstractions; true; catch; false; end
        if !ka_loaded
            @info "KernelAbstractions not in this env — skipping multi-dim test"
            @test true
        else
            KA = KernelAbstractions
            Backend = Base.get_extension(MLIRKernels, :KernelAbstractionsExt).MLIRBackend
            @eval begin
                @kernel function _ka_transpose!(a, @Const(b))
                    i, j = @index(Global, NTuple)
                    @inbounds a[i, j] = b[j, i]
                end
            end
            M = 8
            b = MLIRKernels.aligned_array(Float32, (M, M); alignment=128)
            copyto!(b, reshape(collect(Float32, 1:(M * M)), M, M))
            a = MLIRKernels.aligned_array(Float32, (M, M); alignment=128)
            fill!(a, 0f0)
            (@eval _ka_transpose!)(Backend(), (4, 4))(a, b; ndrange=(M, M))
            @test a == permutedims(b)            # full N-D index + linearised A[i,j]
        end
    end

    # GPU SIMT path (MLIRCUDABackend): a KA @kernel compiled through the MLIR
    # gpu dialect → PTX and run on the device. Scalar-per-thread, so N-D @index
    # + A[i,j] + a reduction accumulator all work with no uniform/varying
    # harmonization. Guarded on a functional CUDA device (skips otherwise).
    @testset "GPU: KA @kernel on MLIRCUDABackend (SIMT)" begin
        gpu_ok = try
            @eval using CUDA, LLVM
            @eval using KernelAbstractions
            CUDA.functional()
        catch
            false
        end
        if !gpu_ok
            @info "CUDA not functional in this env — skipping GPU backend test"
            @test true
        else
            GPUB = Base.get_extension(MLIRKernels, :MLIRCUDAExt).MLIRCUDABackend
            @eval begin
                @kernel function _g_vadd!(c, @Const(a), @Const(b))
                    i = @index(Global, Linear)
                    @inbounds c[i] = a[i] + b[i]
                end
                @kernel function _g_transpose!(a, @Const(b))
                    i, j = @index(Global, NTuple)
                    @inbounds a[i, j] = b[j, i]
                end
                @kernel function _g_matmul!(out, @Const(a), @Const(b))
                    i, j = @index(Global, NTuple)
                    tmp = zero(eltype(out))
                    for k in 1:size(a, 2)
                        @inbounds tmp += a[i, k] * b[k, j]
                    end
                    @inbounds out[i, j] = tmp
                end
            end
            # 1-D vadd
            N = 4096
            a1 = CUDA.CuArray(rand(Float32, N)); b1 = CUDA.CuArray(rand(Float32, N))
            c1 = CUDA.zeros(Float32, N)
            (@eval _g_vadd!)(GPUB(), 256)(c1, a1, b1; ndrange=N); CUDA.synchronize()
            @test Array(c1) == Array(a1) .+ Array(b1)
            # 2-D transpose (non-square) — catches descriptor dim-order
            bh = reshape(collect(Float32, 1:32), 8, 4)
            bt = CUDA.CuArray(bh); at = CUDA.zeros(Float32, 4, 8)
            (@eval _g_transpose!)(GPUB(), (4, 4))(at, bt; ndrange=(4, 8)); CUDA.synchronize()
            @test Array(at) == permutedims(bh)
            # 2-D matmul (non-square) — scalar accumulator over a for-loop
            ah = rand(Float32, 8, 4); bbh = rand(Float32, 4, 6)
            am = CUDA.CuArray(ah); bm = CUDA.CuArray(bbh); om = CUDA.zeros(Float32, 8, 6)
            (@eval _g_matmul!)(GPUB(), (4, 2))(om, am, bm; ndrange=(8, 6)); CUDA.synchronize()
            @test maximum(abs.(Array(om) .- ah * bbh)) < 1f-3

            # @localmem: workgroup-space memref.global + real gpu.barrier.
            @eval begin
                using KernelAbstractions: @localmem, @synchronize
                using Atomix: @atomic
                # cross-lane: reverse within each block through shared memory
                @kernel function _g_shrev!(out, @Const(inp))
                    gid = @index(Global, Linear); lid = @index(Local, Linear)
                    s = @localmem Float32 (256,)
                    @inbounds s[lid] = inp[gid]
                    @synchronize
                    @inbounds out[gid] = s[256 - lid + 1]
                end
                # atomic-on-shared: per-block reduction
                @kernel function _g_blocksum!(out, @Const(inp))
                    gid = @index(Global, Linear); gi = @index(Group, Linear)
                    acc = @localmem Float32 (1,)
                    @inbounds acc[1] = 0f0
                    @synchronize
                    @atomic acc[1] += inp[gid]
                    @synchronize
                    @inbounds out[gi] = acc[1]
                end
            end
            Nl = 1024; Wl = 256; NBl = Nl ÷ Wl
            inl = CUDA.CuArray(rand(Float32, Nl)); ihl = Array(inl)
            orl = CUDA.zeros(Float32, Nl)
            (@eval _g_shrev!)(GPUB(), Wl)(orl, inl; ndrange=Nl); CUDA.synchronize()
            refrev = similar(ihl)
            for b in 0:(NBl-1), k in 1:Wl; refrev[b*Wl+k] = ihl[b*Wl + (Wl-k+1)]; end
            @test Array(orl) == refrev                       # cross-lane shared + barrier
            osum = CUDA.zeros(Float32, NBl)
            (@eval _g_blocksum!)(GPUB(), Wl)(osum, inl; ndrange=Nl); CUDA.synchronize()
            refsum = [sum(ihl[(b*Wl+1):((b+1)*Wl)]) for b in 0:(NBl-1)]
            @test isapprox(Array(osum), refsum; rtol=1f-4)   # atomic-on-shared

            # The full KA histogram example (verbatim): @localmem + two-level
            # shared→global @atomic + @synchronize + @groupsize + a 1:gs:N
            # step-range loop + divergent per-thread ifs. The headline kernel.
            @eval begin
                using KernelAbstractions: @uniform, @groupsize
                @kernel unsafe_indices=true function _g_histogram!(histogram_output, input)
                    gid = @index(Group, Linear)
                    lid = @index(Local, Linear)
                    @uniform gs = prod(@groupsize())
                    tid = (gid - 1) * gs + lid
                    @uniform Nh = length(histogram_output)
                    shared_histogram = @localmem eltype(input) (gs)
                    for min_element in 1:gs:Nh
                        @inbounds shared_histogram[lid] = 0
                        @synchronize()
                        max_element = min_element + gs
                        if max_element > Nh
                            max_element = Nh + 1
                        end
                        bin = tid <= length(input) ? input[tid] : 0
                        if bin >= min_element && bin < max_element
                            bin -= min_element - 1
                            @atomic shared_histogram[bin] += 1
                        end
                        @synchronize()
                        if ((lid + min_element - 1) <= Nh)
                            @atomic histogram_output[lid + min_element - 1] += shared_histogram[lid]
                        end
                    end
                end
            end
            Lh = 4096; NBINS = 256
            hin = rand(1:NBINS, Lh)
            dhin = CUDA.CuArray(hin); dhout = CUDA.zeros(Int, NBINS)
            (@eval _g_histogram!)(GPUB(), (256,))(dhout, dhin; ndrange=Lh); CUDA.synchronize()
            hist_ref = zeros(Int, NBINS); for v in hin; hist_ref[v] += 1; end
            @test Array(dhout) == hist_ref                   # full KA histogram

            # @private: per-thread storage (default-space alloca). Scalar and
            # array forms; the array kernel also takes a `::Val{M}` dispatch arg
            # (a compile-time constant, not a runtime param).
            @eval begin
                using KernelAbstractions: @private
                @kernel function _g_privrev!(A)
                    @uniform Np = prod(@groupsize())
                    I = @index(Global, Linear); il = @index(Local, Linear)
                    pp = @private Int (1,)
                    @inbounds pp[1] = Np - il + 1
                    @inbounds A[I] = pp[1]
                end
                @kernel function _g_privarr!(out, @Const(A), ::Val{M}) where {M}
                    I = @index(Global, Linear)
                    pp = @private Int (M,)
                    s = 0
                    @inbounds for j in 1:M; pp[j] = A[I] * j; end
                    @inbounds for j in 1:M; s += pp[j]; end
                    @inbounds out[I] = s
                end
            end
            Np = 64; Wp = 16
            ap = CUDA.zeros(Int, Np)
            (@eval _g_privrev!)(GPUB(), Wp)(ap; ndrange=Np); CUDA.synchronize()
            @test Array(ap) == repeat(collect(Wp:-1:1), Np ÷ Wp)   # per-thread scalar
            Mp = 4; inpp = CUDA.CuArray(collect(1:Np)); op = CUDA.zeros(Int, Np)
            (@eval _g_privarr!)(GPUB(), Wp)(op, inpp, Val(Mp); ndrange=Np); CUDA.synchronize()
            @test Array(op) == [i * sum(1:Mp) for i in 1:Np]       # per-thread array + Val arg

            # 2-D @localmem tile: size(tile,d) (the column-major linearisation
            # of tile[i,j]) resolves to the static dims. Copy + cross-lane
            # transpose through a 16x16 shared tile.
            @eval begin
                @kernel function _g_tilecopy!(o, @Const(a))
                    I, J = @index(Global, NTuple); ii, jj = @index(Local, NTuple)
                    t = @localmem Float32 (16, 16)
                    @inbounds t[ii, jj] = a[I, J]
                    @synchronize
                    @inbounds o[I, J] = t[ii, jj]
                end
                @kernel function _g_tiletr!(o, @Const(a))
                    I, J = @index(Global, NTuple); ii, jj = @index(Local, NTuple)
                    t = @localmem Float32 (16, 16)
                    @inbounds t[ii, jj] = a[I, J]
                    @synchronize
                    @inbounds o[I, J] = t[jj, ii]
                end
            end
            Mt = 32
            int = CUDA.CuArray(reshape(collect(Float32, 1:(Mt*Mt)), Mt, Mt)); iht = Array(int)
            otc = CUDA.zeros(Float32, Mt, Mt)
            (@eval _g_tilecopy!)(GPUB(), (16,16))(otc, int; ndrange=(Mt,Mt)); CUDA.synchronize()
            @test Array(otc) == iht                                # 2-D tile copy
            ott = CUDA.zeros(Float32, Mt, Mt)
            (@eval _g_tiletr!)(GPUB(), (16,16))(ott, int; ndrange=(Mt,Mt)); CUDA.synchronize()
            reft = copy(iht)
            for bi in 0:1, bj in 0:1, ai in 1:16, aj in 1:16
                reft[bi*16+ai, bj*16+aj] = iht[bi*16+aj, bj*16+ai]
            end
            @test Array(ott) == reft                               # 2-D tile cross-lane transpose

            # @simd / @unroll loops: the loopinfo hint is dropped (the loop is a
            # plain scf.for); LLVM/ptxas unroll. Both forms over a Val{M} bound.
            @eval begin
                @kernel function _g_simdsum!(o, @Const(a), ::Val{M}) where {M}
                    Ix = @index(Global, Linear); ac = zero(eltype(o))
                    @simd for k in 1:M; @inbounds ac += a[Ix] * k; end
                    @inbounds o[Ix] = ac
                end
                @kernel function _g_unrollsum!(o, @Const(a), ::Val{M}) where {M}
                    Ix = @index(Global, Linear); ac = zero(eltype(o))
                    KernelAbstractions.Extras.@unroll for k in 1:M
                        @inbounds ac += a[Ix] * k
                    end
                    @inbounds o[Ix] = ac
                end
            end
            Ns = 64; Ws = 16; Ms = 4
            as = CUDA.CuArray(collect(1:Ns)); refs = [Array(as)[k]*sum(1:Ms) for k in 1:Ns]
            os1 = CUDA.zeros(Int, Ns)
            (@eval _g_simdsum!)(GPUB(), Ws)(os1, as, Val(Ms); ndrange=Ns); CUDA.synchronize()
            @test Array(os1) == refs                               # @simd
            os2 = CUDA.zeros(Int, Ns)
            (@eval _g_unrollsum!)(GPUB(), Ws)(os2, as, Val(Ms); ndrange=Ns); CUDA.synchronize()
            @test Array(os2) == refs                               # @unroll

            # code_gpu reflection: every codegen level emits its expected IR.
            @eval @kernel function _g_radd!(c, @Const(a), @Const(b))
                i = @index(Global, Linear); @inbounds c[i] = a[i] + b[i]
            end
            ar = CUDA.rand(Float32, 256); br = CUDA.rand(Float32, 256); cr = CUDA.zeros(Float32, 256)
            kr = (@eval _g_radd!)(GPUB(), 256)
            @test occursin("gpu.func",      code_gpu(kr, cr, ar, br; ndrange=256, level=:mlir))
            @test occursin("llvm.",         code_gpu(kr, cr, ar, br; ndrange=256, level=:lowered))
            @test occursin("ptx_kernel",    code_gpu(kr, cr, ar, br; ndrange=256, level=:llvm))
            @test occursin(".visible .entry", code_gpu(kr, cr, ar, br; ndrange=256, level=:ptx))
        end
    end

end
