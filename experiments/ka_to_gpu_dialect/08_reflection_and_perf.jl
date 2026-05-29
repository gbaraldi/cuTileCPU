# Reflection at every codegen level + perf comparison vs CUDA.jl's native KA.
using MLIRKernels, KernelAbstractions, CUDA, LLVM, Atomix, Printf
const KA = KernelAbstractions
const ext = Base.get_extension(MLIRKernels, :MLIRCUDAExt)
const MLIRB = ext.MLIRCUDABackend

@assert CUDA.functional()

# ---- kernels (one body, two backends) --------------------------------------
@kernel function vadd!(c, @Const(a), @Const(b))
    i = @index(Global, Linear)
    @inbounds c[i] = a[i] + b[i]
end

@kernel function matmul!(C, @Const(A), @Const(B))
    i, j = @index(Global, NTuple)
    acc = zero(eltype(C))
    @inbounds for k in 1:size(A, 2)
        acc += A[i, k] * B[k, j]
    end
    @inbounds C[i, j] = acc
end

# ---- reflection: print each codegen level for vadd -------------------------
function show_levels()
    N = 1024
    a = CUDA.rand(Float32, N); b = CUDA.rand(Float32, N); c = CUDA.zeros(Float32, N)
    k = vadd!(MLIRB(), 256)
    for (lvl, head, nlines) in (
            (:sci,     "1. StructuredIRCode (post-inference Julia IR)", 45),
            (:mlir,    "2. High-level `gpu`-dialect MLIR (pre-pipeline)", 250),
            (:lowered, "3. Lowered LLVM/NVVM-dialect MLIR", 250),
            (:llvm,    "4. LLVM IR (.ll)", 250),
            (:ptx,     "5. PTX assembly", 250))
        s = code_gpu(k, c, a, b; ndrange=N, level=lvl)
        nl = count(==('\n'), s) + 1
        println("\n", "="^78, "\n  ", head, "  [", nl, " lines]\n", "="^78)
        lines = split(s, '\n')
        println(join(lines[1:min(nlines, end)], '\n'))
        length(lines) > nlines && println("    … (", length(lines)-nlines, " more lines)")
    end
end

# ---- perf: min over samples after warmup -----------------------------------
function bench(fn; warmup=5, samples=100)
    for _ in 1:warmup; fn(); end
    CUDA.synchronize()
    t = Inf
    for _ in 1:samples
        t = min(t, CUDA.@elapsed fn())
    end
    return t   # seconds
end

function printrow(label, tM, tC, work, unit)
    rM = work/tM/1e9; rC = work/tC/1e9
    println(rpad("  $label", 18),
            rpad("MLIR: $(round(tM*1e6,digits=1))µs ($(round(rM,digits=1)) $unit)", 36),
            rpad("CUDA.jl: $(round(tC*1e6,digits=1))µs ($(round(rC,digits=1)) $unit)", 36),
            "ratio ", round(tM/tC, digits=2), "×")
end

function compare_vadd()
    println("\n", "#"^78, "\n# PERF: vadd  (bandwidth-bound)\n", "#"^78)
    for N in (1<<20, 1<<24)
        a = CUDA.rand(Float32, N); b = CUDA.rand(Float32, N)
        cM = CUDA.zeros(Float32, N); cC = CUDA.zeros(Float32, N)
        kM = vadd!(MLIRB(), 256); kC = vadd!(CUDABackend(), 256)
        tM = bench(() -> kM(cM, a, b; ndrange=N))
        tC = bench(() -> kC(cC, a, b; ndrange=N))
        @assert Array(cM) ≈ Array(cC)
        printrow("N=$N", tM, tC, 3*N*sizeof(Float32), "GB/s")
    end
end

function compare_matmul()
    println("\n", "#"^78, "\n# PERF: matmul  (naive, one thread/element)\n", "#"^78)
    for n in (256, 512, 1024)
        A = CUDA.rand(Float32, n, n); B = CUDA.rand(Float32, n, n)
        CM = CUDA.zeros(Float32, n, n); CC = CUDA.zeros(Float32, n, n)
        wg = (16, 16)
        kM = matmul!(MLIRB(), wg); kC = matmul!(CUDABackend(), wg)
        tM = bench(() -> kM(CM, A, B; ndrange=(n, n)))
        tC = bench(() -> kC(CC, A, B; ndrange=(n, n)))
        @assert isapprox(Array(CM), Array(CC); rtol=1f-2)
        printrow("n=$n", tM, tC, 2.0*n^3, "GFLOP/s")
    end
end

show_levels()
compare_vadd()
compare_matmul()
println("\nDONE")
