# Standalone perf check: vector.contract path vs linalg.matmul path.
# Both .mlir files emit the same kernel shape (1024×1024×1024 F32 matmul,
# 16×16 grid of 64×64 tiles). Compiles each through its appropriate pass
# pipeline, dlopens the .so, ccalls 10× with cache flush between samples.

using Printf, Libdl, LinearAlgebra, Random

const ART_DIR = "/home/gbaraldi/.julia/artifacts/720864e43f58e8fb02ef0ace03311873748fac8d"
const MLIR_OPT   = joinpath(ART_DIR, "tools", "mlir-opt")
const MLIR_TRANSLATE = joinpath(ART_DIR, "tools", "mlir-translate")
const CLANG     = joinpath(ART_DIR, "tools", "clang")
const LIBOMP_DIR = "/home/gbaraldi/.julia/artifacts/899764f4e46a2e2a6f6351ad5c22e6002d24d294/lib"

# === pipelines ===

# Contract path: what cuTileCPU's DEFAULT_PASSES currently uses.
const CONTRACT_PASSES = String[
    "--lower-vector-multi-reduction",
    "--convert-vector-to-scf",
    "--convert-scf-to-openmp",
    "--convert-openmp-to-llvm",
    "--convert-scf-to-cf",
    "--lower-affine",
    "--convert-vector-to-llvm=vector-contract-lowering=outerproduct",
    "--expand-strided-metadata",
    "--finalize-memref-to-llvm",
    "--convert-math-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-ub-to-llvm",
    "--reconcile-unrealized-casts",
]

# Linalg path. Modeled on upstream MLIR's `test-lower-to-llvm` (in
# llvm-project/mlir/test/lib/Dialect/LLVM/TestLowerToLLVM.cpp) plus the
# transform-interpreter/bufferize prelude used by the Linalg CPU integration
# tests (e.g. mlir/test/Integration/Dialect/Linalg/CPU/ArmSVE/matmul.mlir).
# OpenMP conversion is inserted before the cf-lowering so the parallel grid
# stays parallel.
const LINALG_PASSES = String[
    # Front-end: run the embedded transform schedule, then bufferize.
    "--transform-interpreter",
    "--test-transform-dialect-erase-schedule",  # remove the schedule from the IR
    "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true",
    "--buffer-deallocation-pipeline",
    "--canonicalize",
    "--cse",

    # vector.multi_reduction (left over from transform.structured.vectorize)
    # must be lowered while still in vector dialect. `--convert-openmp-to-llvm`
    # cascades the type-conversion through the parallel region and converts
    # surrounding values to !llvm.array, after which lower-vector-multi-reduction
    # can no longer touch them. So do this BEFORE OpenMP lowering.
    "--lower-vector-multi-reduction",

    # Convert scf.parallel → OpenMP omp.parallel, then lower the omp dialect
    # to LLVM. Do this before scf-to-cf so the parallel region keeps its
    # structure for OpenMP runtime calls.
    "--convert-scf-to-openmp",
    "--convert-openmp-to-llvm",

    # Canonical lower-to-LLVM (matches test-lower-to-llvm in upstream).
    "--convert-vector-to-scf",
    "--convert-linalg-to-loops",      # fallback for any leftover linalg
    "--lower-affine",                 # first affine sweep
    "--convert-scf-to-cf",
    "--canonicalize",
    "--cse",
    "--convert-vector-to-llvm=vector-contract-lowering=outerproduct",
    "--convert-math-to-llvm",
    "--expand-strided-metadata",
    "--lower-affine",                 # expand-strided-metadata re-introduces affine
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-index-to-llvm",
    "--convert-ub-to-llvm",
    "--reconcile-unrealized-casts",
]

function compile_so(mlir_path::String, passes::Vector{String}, name::String)
    workdir = "/tmp/perf_compare"
    lowered_path = joinpath(workdir, "$(name).lowered.mlir")
    ll_path = joinpath(workdir, "$(name).ll")
    so_path = joinpath(workdir, "$(name).so")

    @info "  lowering $name"
    t0 = time_ns()
    open(lowered_path, "w") do io
        run(pipeline(`$MLIR_OPT $mlir_path $passes`, stdout=io))
    end
    t_lower = (time_ns() - t0) / 1e9

    @info "  translating to LLVM IR"
    t0 = time_ns()
    run(pipeline(`$MLIR_TRANSLATE $lowered_path --mlir-to-llvmir`, stdout=ll_path))
    t_trans = (time_ns() - t0) / 1e9

    @info "  clang -O2"
    t0 = time_ns()
    run(`$CLANG -O2 -shared -fPIC $ll_path
         -L$LIBOMP_DIR -Wl,-rpath,$LIBOMP_DIR -lomp
         -o $so_path`)
    t_clang = (time_ns() - t0) / 1e9

    @printf("  compile (lower=%.1fs, translate=%.1fs, clang=%.1fs, total=%.1fs)\n",
            t_lower, t_trans, t_clang, t_lower + t_trans + t_clang)
    return so_path
end

# === memref descriptor for 2-D F32 (matches MLIR C-interface ABI) ===

struct MR2D
    allocated::Ptr{Float32}
    aligned::Ptr{Float32}
    offset::Int64
    size0::Int64
    size1::Int64
    stride0::Int64
    stride1::Int64
end

function bench_so(so_path::String, A, B, C, M, N, K; reference::Union{Nothing,Matrix{Float32}}=nothing,
                  label::String="")
    h = Libdl.dlopen(so_path)
    fn = Libdl.dlsym(h, :_mlir_ciface_kernel)

    # Cache flush scratch
    flush_buf = Vector{UInt8}(undef, 256 * 1024 * 1024); fill!(flush_buf, 0x01)
    flush_caches!() = (s = zero(UInt64); @inbounds for i in 1:64:length(flush_buf); s += flush_buf[i]; end; s)

    # MLIR row-major. Our buffers are stored as flat Vector{Float32} with
    # row-major semantics: A[i, k] at index (i-1)*K + k.
    da = Ref(MR2D(pointer(A), pointer(A), 0, M, K, K, 1))
    db = Ref(MR2D(pointer(B), pointer(B), 0, K, N, N, 1))
    dc = Ref(MR2D(pointer(C), pointer(C), 0, M, N, N, 1))

    function call_once()
        GC.@preserve A B C begin
            ccall(fn, Cvoid, (Ptr{MR2D}, Ptr{MR2D}, Ptr{MR2D}), da, db, dc)
        end
    end

    # Correctness: run once, full-array compare against the reference if
    # provided. We treat the C buffer as a row-major M×N matrix.
    fill!(C, 0f0)
    call_once()
    if reference !== nothing
        Cm = collect(reshape(C, N, M)')   # row-major (M,N) → column-major (M,N)
        max_diff = maximum(abs, Cm .- reference)
        rel_err = max_diff / maximum(abs, reference)
        ok = isapprox(Cm, reference; rtol=1f-3)
        @printf("  %s correctness: %s  (max abs diff = %.2e, rel err = %.2e)\n",
                label, ok ? "PASS" : "FAIL", max_diff, rel_err)
        ok || error("$label: numerical mismatch (rtol=1e-3 violated)")
    end

    # Warmup
    for _ in 1:3
        call_once()
    end

    # Time min over 10 samples with cache flush between
    best = typemax(UInt64)
    for _ in 1:10
        flush_caches!()
        t0 = time_ns(); call_once(); t1 = time_ns()
        Δ = t1 - t0
        Δ < best && (best = Δ)
    end
    return Float64(best)
end

# Row-major view helpers (the linear array stores row-major data).
A_view(A, i, j, K) = @inbounds A[(i-1)*K + j]
B_view(B, k, j, N) = @inbounds B[(k-1)*N + j]
C_view(C, i, j, N) = @inbounds C[(i-1)*N + j]

function main()
    M = N = K = 1024
    println("Problem: $M × $N × $K F32 matmul, 64-tile grid")
    println("Julia threads: $(Threads.nthreads())")
    println()

    Random.seed!(0xc0ffee)
    A = Vector{Float32}(undef, M * K); rand!(A)
    B = Vector{Float32}(undef, K * N); rand!(B)
    C = Vector{Float32}(undef, M * N); fill!(C, 0f0)

    # Reference: compute C_ref = A * B in Julia (BLAS) on a column-major view
    # of the same row-major buffer. A_mlir[i,k] at A[(i-1)*K + k] corresponds
    # to column-major A_col[k, i] for an array of shape (K, M). So:
    Am_rm = collect(reshape(A, K, M)')       # explicit M×K column-major
    Bm_rm = collect(reshape(B, N, K)')       # explicit K×N column-major
    C_ref = Am_rm * Bm_rm                     # OpenBLAS oracle, M×N

    @info "Compiling CONTRACT path"
    so_c = compile_so("/tmp/perf_compare/contract.mlir", CONTRACT_PASSES, "contract")
    @info "Compiling LINALG (minimal) path"
    so_l = compile_so("/tmp/perf_compare/linalg.mlir", LINALG_PASSES, "linalg")
    @info "Compiling LINALG (rich) path"
    so_r = compile_so("/tmp/perf_compare/linalg_rich.mlir", LINALG_PASSES, "linalg_rich")

    println()
    println("Correctness + benchmarking (min of 10 samples, cache-flushed):")
    t_c = bench_so(so_c, A, B, C, M, N, K; reference=C_ref, label="contract")
    t_l = bench_so(so_l, A, B, C, M, N, K; reference=C_ref, label="linalg  ")
    t_r = bench_so(so_r, A, B, C, M, N, K; reference=C_ref, label="rich    ")

    # OpenBLAS bench, on the column-major views (which is what mul! consumes).
    Cm = Matrix{Float32}(undef, M, N); fill!(Cm, 0f0)
    mul!(Cm, Am_rm, Bm_rm)
    @assert isapprox(Cm, C_ref; rtol=1f-5) "OpenBLAS oracle disagreement"
    for _ in 1:3; mul!(Cm, Am_rm, Bm_rm); end
    flush_buf = Vector{UInt8}(undef, 256 * 1024 * 1024); fill!(flush_buf, 0x01)
    flush!() = (s = zero(UInt64); @inbounds for i in 1:64:length(flush_buf); s += flush_buf[i]; end; s)
    best = typemax(UInt64)
    for _ in 1:10
        flush!()
        t0 = time_ns(); mul!(Cm, Am_rm, Bm_rm); t1 = time_ns()
        Δ = t1 - t0
        Δ < best && (best = Δ)
    end
    t_blas = Float64(best)

    println()
    flops = 2.0 * M * N * K
    @printf("  contract:       %8.1f μs  %7.1f GFLOPS  (%.0f%% of BLAS)\n",
            t_c/1e3, flops / t_c, 100 * (flops/t_c) / (flops/t_blas))
    @printf("  linalg minimal: %8.1f μs  %7.1f GFLOPS  (%.0f%% of BLAS)\n",
            t_l/1e3, flops / t_l, 100 * (flops/t_l) / (flops/t_blas))
    @printf("  linalg rich:    %8.1f μs  %7.1f GFLOPS  (%.0f%% of BLAS)\n",
            t_r/1e3, flops / t_r, 100 * (flops/t_r) / (flops/t_blas))
    @printf("  OpenBLAS:       %8.1f μs  %7.1f GFLOPS  (reference)\n",
            t_blas/1e3, flops / t_blas)
end

main()
