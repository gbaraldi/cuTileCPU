# A real KernelAbstractions `@kernel` compiled to GPU through the MLIR
# gpu-dialect pipeline and launched on the H100.
#
# This is the original question, fully closed: take an unmodified KA
# kernel — the same source that runs on CUDA.jl / AMDGPU.jl / the CPU —
# and run it on the GPU via Julia → MLIR gpu-dialect → PTX, comparing to
# CUDA.jl's native SIMT.
#
# How it composes the prior pieces:
#   - `cuTileBackend <: KA.GPU` (ext/KernelAbstractionsExt.jl) makes KA
#     emit the SIMT `gpu_*` body and overlays the KA intrinsics in
#     cuTile's method table:
#         KA.__validindex(ctx)          → true
#         KA.__index_Global_Linear(ctx) → __cutilecpu_spmd_lane_id()
#   - `_structured_with_analyses` runs inference (overlays applied) +
#     structurizer → SCI for the `gpu_*` body. The KA intrinsic calls are
#     gone; only the sentinel remains.
#   - `lower_to_mlir_gpu(sci; ctx_arg=2)` (the new KA-shape GPU entrypoint)
#     emits gpu.module: the `__ctx__` slot is the lane, bound to the scalar
#     global thread index; `__cutilecpu_spmd_lane_id()` resolves to it.
#   - then the exp 06 path: gpu→nvvm pipeline → LLVM bitcode → LLVM.jl
#     NVPTX → PTX → CUDA.CuModule → cudacall.
#
# Limitation (MVP): `__validindex → true` means every launched thread runs
# the body with no per-thread bounds mask, so ndrange must be an exact
# multiple of the block size (no tail block). A real `__validindex`
# (gid <= ndrange, threaded through the ctx) would lift this — future work.

using KernelAbstractions
const KA = KernelAbstractions
using cuTile
using cuTileCPU
using MLIR
const IR = MLIR.IR
const MLIRAPI = MLIR.API
using LLVM
using CUDA

const KAExt = Base.get_extension(cuTileCPU, :KernelAbstractionsExt)
const cuTileBackend = KAExt.cuTileBackend

# ---------------------------------------------------------------------------
# The kernel — an unmodified KA @kernel. Same source you'd run on any
# KA backend.
# ---------------------------------------------------------------------------

@kernel function vadd_ka!(C, A, B)
    i = @index(Global, Linear)
    @inbounds C[i] = A[i] + B[i]
end

# Grab the macro-generated SIMT body (`gpu_vadd_ka!`). KA's @kernel emits
# `gpu_<name>` and `cpu_<name>`; cuTileBackend (<: KA.GPU) selects the
# gpu_ one.
const gpu_body = @eval $(Symbol("gpu_vadd_ka!"))

# ---------------------------------------------------------------------------
# Compile: KA gpu_ body → SCI → gpu.module (ctx_arg=2) → PTX → CuFunction
# ---------------------------------------------------------------------------

const GPU_PASSES = String[
    "nvvm-attach-target{chip=sm_90 features=+ptx80}",
    "gpu-kernel-outlining",
    "gpu.module(convert-gpu-to-nvvm)",
    "convert-scf-to-cf", "convert-cf-to-llvm", "convert-arith-to-llvm",
    "expand-strided-metadata", "finalize-memref-to-llvm",
    "convert-nvvm-to-llvm", "reconcile-unrealized-casts",
    "gpu-module-to-binary{format=llvm}",
]

# Build a CompilerMetadata type for the launch shape (ndrange / workgroup).
function ctx_type(N, W)
    ndr   = KA.NDIteration.StaticSize{(N,)}
    wg    = KA.NDIteration.StaticSize{(W,)}
    groups= KA.NDIteration.StaticSize{(cld(N, W),)}
    ndrobj= KA.NDIteration.NDRange{1, groups, wg, Nothing, Nothing}
    return KA.CompilerMetadata{ndr, KA.NDIteration.NoDynamicCheck, Nothing, Nothing, ndrobj}
end

function compile_ka_to_cufunction(gpu_body, N, W; kernel_name="vadd_ka")
    ctxT = ctx_type(N, W)
    AT = Tuple{ctxT, CuArray{Float32,1,CUDA.DeviceMemory},
               CuArray{Float32,1,CUDA.DeviceMemory},
               CuArray{Float32,1,CUDA.DeviceMemory}}
    # NOTE: the SCI walk only needs the *element/shape* class of the array
    # args (it lowers them to memref<?xf32, global>), so plain Vector{Float32}
    # works just as well for inference; use it to avoid device-array inference
    # quirks.
    AT = Tuple{ctxT, Vector{Float32}, Vector{Float32}, Vector{Float32}}

    sci, rettype = cuTileCPU.Frontend.structured(gpu_body, AT)
    rettype === Nothing || @warn "KA gpu body inferred rettype = $rettype (expected Nothing)"
    # ctx is arg-slot 2 (slot 1 is the function itself).
    mod, _, mlir_ctx, kinds =
        cuTileCPU.lower_to_mlir_gpu(sci, AT; kernel_name, ctx_arg=2)

    IR.activate(mlir_ctx)
    println("=== generated gpu.module ===")
    println(sprint(show, mod))
    println("param kinds: ", kinds)

    pm = IR.PassManager()
    parse(IR.OpPassManager(pm), "builtin.module(" * join(GPU_PASSES, ",") * ")")
    MLIRAPI.mlirPassManagerRunOnOp(pm, IR.Operation(mod)).value == 0 &&
        error("GPU pipeline failed")

    bc = let out=nothing
        for op in IR.body(mod)
            IR.name(op) == "gpu.binary" || continue
            objs = IR.getattr(op, "objects")
            o0 = IR.Attribute(MLIRAPI.mlirArrayAttrGetElement(objs, 0))
            sr = MLIRAPI.mlirGPUObjectAttrGetObject(o0)
            out = copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(sr.data), sr.length; own=false))
        end
        out === nothing && error("no gpu.binary"); out
    end
    lctx = LLVM.Context()
    lmod = LLVM.context!(lctx) do; parse(LLVM.Module, bc); end
    triple = "nvptx64-nvidia-cuda"
    LLVM.triple!(lmod, triple)
    tm = LLVM.TargetMachine(LLVM.Target(; triple), triple, "sm_90", "+ptx80")
    LLVM.asm_verbosity!(tm, true)
    ptx = String(LLVM.emit(tm, lmod, LLVM.API.LLVMAssemblyFile))
    cumod = CuModule(ptx)
    return CuFunction(cumod, kernel_name), count(==(:memref), kinds)
end

const N = 16 * 1024 * 1024
const W = 256
@assert N % W == 0  "MVP needs exact-multiple ndrange (no tail block)"

println("=" ^ 60)
println("Compiling KA @kernel `vadd_ka!` → gpu.module → PTX")
println("=" ^ 60)
kernel, nmemref = compile_ka_to_cufunction(gpu_body, N, W)
println("✓ CuFunction handle: ", kernel.handle, "   (", nmemref, " memref args)")

# ---------------------------------------------------------------------------
# Launch + verify
# ---------------------------------------------------------------------------

A_host = rand(Float32, N); B_host = rand(Float32, N)
A = CuArray(A_host); B = CuArray(B_host); C = CUDA.zeros(Float32, N)
function desc(arr::CuArray)
    p = UInt64(UInt(pointer(arr))); (p, p, UInt64(0), UInt64(length(arr)), UInt64(1))
end
# gpu.func params are (C, A, B) in the KA body order gpu_vadd_ka!(ctx,C,A,B).
cd_, ad, bd = desc(C), desc(A), desc(B)
const SIG = Tuple{ntuple(_->Culonglong, 15)...}
args = (cd_..., ad..., bd...)

const GRID = cld(N, W)
println("\n=== Launching KA-generated kernel: grid=$GRID block=$W ===")
cudacall(kernel, SIG, args...; threads=W, blocks=GRID)
CUDA.synchronize()

err = maximum(abs.(Array(C) .- (A_host .+ B_host)))
println("max abs diff = $err")
@assert err == 0 "KA kernel produced wrong results on GPU"
println("✓ A REAL KernelAbstractions @kernel runs CORRECTLY on the H100 via MLIR")

# ---------------------------------------------------------------------------
# Timing vs CUDA.jl's own KA backend (CUDABackend) running the SAME @kernel
# ---------------------------------------------------------------------------

function bench(fn; warmup=3, samples=20)
    for _ in 1:warmup; fn(); end
    CUDA.synchronize()
    best = typemax(Float64)
    for _ in 1:samples; best = min(best, CUDA.@elapsed fn()); end
    return best
end

gb = 3 * N * sizeof(Float32) / 1e9
t_mlir = bench(() -> cudacall(kernel, SIG, args...; threads=W, blocks=GRID))

# Same @kernel, but through CUDA.jl's native KA backend (GPUCompiler).
cuda_kernel = vadd_ka!(CUDABackend(), W)
t_cuda = bench(() -> (cuda_kernel(C, A, B; ndrange=N)))

println("\n" * "=" ^ 60)
println("Same KA @kernel, two backends (N=$N, H100 sm_90)")
println("=" ^ 60)
println(rpad("backend", 40), rpad("μs", 10), "GB/s")
println(rpad("cuTileBackend (MLIR gpu→PTX)", 40),
        rpad(round(t_mlir*1e6, digits=2), 10), round(gb/t_mlir, digits=1))
println(rpad("CUDABackend (GPUCompiler SIMT)", 40),
        rpad(round(t_cuda*1e6, digits=2), 10), round(gb/t_cuda, digits=1))
println("\nratio (MLIR / CUDA.jl) = $(round(t_mlir/t_cuda, digits=3))x")
