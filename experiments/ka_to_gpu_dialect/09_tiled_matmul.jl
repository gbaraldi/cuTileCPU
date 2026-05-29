# Tiled matmul: correctness + benchmark vs naive / CUDA.jl KA / CUBLAS.
# The tiled @kernel itself is filled in from the design workflow (see TILED block).
using MLIRKernels, KernelAbstractions, CUDA, LLVM, Atomix, Printf
using LinearAlgebra
const KA = KernelAbstractions
const ext = Base.get_extension(MLIRKernels, :MLIRCUDAExt)
const MLIRB = ext.MLIRCUDABackend
@assert CUDA.functional()

# ---- naive (already validated; one thread per output element) --------------
@kernel function mm_naive!(C, @Const(A), @Const(B))
    i, j = @index(Global, NTuple)
    acc = zero(eltype(C))
    @inbounds for k in 1:size(A, 2)
        acc += A[i, k] * B[k, j]
    end
    @inbounds C[i, j] = acc
end

# ---- TILED (placeholder; replaced after design workflow) -------------------
# --- tiled kernel (safe nested-loop structure) ---
# Tiled matmul (square, K an exact multiple of TILE).
# Structure proven safe by the nested-loop repro: the inner-product accumulator
# `out` is LOCAL to each k-tile iteration (reset each outer iter) and folded
# into the outer register accumulator `acc` AFTER the inner loop — never one
# accumulator carried through both loop levels (that trips the structurizer).
const TILE = 16

@kernel function mm_tiled!(C, @Const(A), @Const(B))
    I, J  = @index(Global, NTuple)
    ii, jj = @index(Local, NTuple)
    tA = @localmem Float32 (TILE, TILE)
    tB = @localmem Float32 (TILE, TILE)
    acc = zero(Float32)
    nkt = size(A, 2) ÷ TILE
    for kt in 1:nkt
        kbase = (kt - 1) * TILE
        @inbounds tA[ii, jj] = A[I, kbase + jj]
        @inbounds tB[ii, jj] = B[kbase + ii, J]
        @synchronize
        out = zero(Float32)
        @inbounds for kk in 1:TILE
            out += tA[ii, kk] * tB[kk, jj]
        end
        acc += out
        @synchronize
    end
    @inbounds C[I, J] = acc
end


function bench(fn; warmup=5, samples=100)
    for _ in 1:warmup; fn(); end
    CUDA.synchronize()
    t = Inf
    for _ in 1:samples; t = min(t, CUDA.@elapsed fn()); end
    return t
end

function run()
    println("# tiled matmul — correctness")
    for n in (256, 512, 1024)
        A = CUDA.rand(Float32, n, n); B = CUDA.rand(Float32, n, n)
        Cref = Array(A) * Array(B)
        Ct = CUDA.zeros(Float32, n, n)
        mm_tiled!(MLIRB(), (TILE, TILE))(Ct, A, B; ndrange=(n, n)); CUDA.synchronize()
        ok = isapprox(Array(Ct), Cref; rtol=1f-2)
        println("  n=$n tiled correct: ", ok, ok ? "" : "  MAXERR=$(maximum(abs.(Array(Ct).-Cref)))")
        @assert ok
    end

    println("\n# perf (GFLOP/s, min of 100)")
    for n in (512, 1024, 2048)
        A = CUDA.rand(Float32, n, n); B = CUDA.rand(Float32, n, n)
        Cn = CUDA.zeros(Float32, n, n); Ct = CUDA.zeros(Float32, n, n)
        CcK = CUDA.zeros(Float32, n, n)
        kNa = mm_naive!(MLIRB(), (TILE, TILE)); kTi = mm_tiled!(MLIRB(), (TILE, TILE))
        kCK = mm_tiled!(CUDABackend(), (TILE, TILE))
        tNa = bench(() -> kNa(Cn, A, B; ndrange=(n, n)))
        tTi = bench(() -> kTi(Ct, A, B; ndrange=(n, n)))
        tCK = bench(() -> kCK(CcK, A, B; ndrange=(n, n)))
        tBl = bench(() -> LinearAlgebra.mul!(Cn, A, B))    # CUBLAS
        g(t) = 2.0*n^3/t/1e9
        @printf("  n=%-5d  MLIR-naive %7.0f  MLIR-tiled %7.0f  CUDA.jl-tiled %7.0f  CUBLAS %8.0f  (GFLOP/s)\n",
                n, g(tNa), g(tTi), g(tCK), g(tBl))
    end
end
run()
println("DONE")
