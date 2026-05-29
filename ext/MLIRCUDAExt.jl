module MLIRCUDAExt

# KA → GPU SIMT path for MLIRKernels.
#
# `MLIRCUDABackend <: KA.GPU` makes KA emit the SIMT `gpu_*` kernel body; we
# infer it through the decoupled Frontend (the KA-intrinsic overlays live in
# `KernelAbstractionsExt`'s `Frontend.METHOD_TABLE`, populated whenever KA is
# loaded), lower to the `gpu` dialect via `lower_to_mlir_gpu`, run the
# gpu→nvvm pipeline, emit PTX through LLVM.jl's NVPTX backend, and launch with
# `cudacall`. Each GPU thread is one scalar lane — no `vector<W>`, hence none
# of the uniform/varying harmonization the CPU-SIMD path needs.
#
# Productionised from experiments/ka_to_gpu_dialect/07_ka_kernel_on_gpu.jl.

using KernelAbstractions
const KA = KernelAbstractions
const NDI = KernelAbstractions.NDIteration
using MLIRKernels
const MK = MLIRKernels
const FE = MLIRKernels.Frontend
using MLIR
const IR = MLIR.IR
const MLIRAPI = MLIR.API
using LLVM
using CUDA

struct MLIRCUDABackend <: KA.GPU end

# ----------------------------------------------------------------------------
# Backend protocol — allocation/sync defer to CUDA.
# ----------------------------------------------------------------------------

KA.allocate(::MLIRCUDABackend, ::Type{T}, dims::Tuple) where {T} = CuArray{T}(undef, dims)
KA.zeros(::MLIRCUDABackend, ::Type{T}, dims::Tuple) where {T} = CUDA.zeros(T, dims)
KA.ones(::MLIRCUDABackend, ::Type{T}, dims::Tuple) where {T} = CUDA.ones(T, dims)
KA.synchronize(::MLIRCUDABackend) = CUDA.synchronize()
KA.functional(::MLIRCUDABackend) = CUDA.functional()
KA.supports_atomics(::MLIRCUDABackend) = true
# NOTE: no `KA.get_backend(::CuArray)` override — that belongs to CUDA.jl's own
# CUDABackend. Users opt in explicitly: `k(MLIRCUDABackend(), W)(...; ndrange)`.

# ----------------------------------------------------------------------------
# GPU compilation: SCI → gpu.module → PTX → CuFunction (cached).
# ----------------------------------------------------------------------------

const GPU_PASSES = String[
    "nvvm-attach-target{chip=sm_90 features=+ptx80}",
    "gpu-kernel-outlining",
    "gpu.module(convert-gpu-to-nvvm)",
    "convert-scf-to-cf", "convert-cf-to-llvm", "convert-arith-to-llvm",
    "expand-strided-metadata", "finalize-memref-to-llvm",
    "convert-nvvm-to-llvm", "reconcile-unrealized-casts",
    "gpu-module-to-binary{format=llvm}",
]

# PTX identifiers can't contain `!` etc.; sanitise the kernel symbol.
_sym(f) = replace(string(nameof(f)), r"[^A-Za-z0-9_]" => "_")

const _gpu_cache = Dict{Any, Tuple{CUDA.CuFunction, Vector{Symbol}}}()

function _extract_gpu_binary(mod)
    for op in IR.body(mod)
        IR.name(op) == "gpu.binary" || continue
        objs = IR.getattr(op, "objects")
        o0 = IR.Attribute(MLIRAPI.mlirArrayAttrGetElement(objs, 0))
        sr = MLIRAPI.mlirGPUObjectAttrGetObject(o0)
        return copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(sr.data), sr.length; own=false))
    end
    error("MLIRCUDABackend: gpu pipeline produced no gpu.binary")
end

# Run an MLIR pass pipeline (list of pass strings) on `mod` in place.
function _run_passes!(mod, mlir_ctx, passes)
    IR.activate(mlir_ctx)
    pm = IR.PassManager()
    parse(IR.OpPassManager(pm), "builtin.module(" * join(passes, ",") * ")")
    MLIRAPI.mlirPassManagerRunOnOp(pm, IR.Operation(mod)).value == 0 &&
        error("MLIRCUDABackend: GPU pass pipeline failed")
    return mod
end

# gpu.binary{format=llvm} bitcode → (LLVM.Module, PTX string). The driver JITs
# PTX → SASS at module load.
function _bitcode_to_ptx(bc; sm="sm_90", feat="+ptx80")
    lctx = LLVM.Context()
    lmod = LLVM.context!(lctx) do
        parse(LLVM.Module, bc)
    end
    triple = "nvptx64-nvidia-cuda"
    LLVM.triple!(lmod, triple)
    tm = LLVM.TargetMachine(LLVM.Target(; triple), triple, sm, feat)
    LLVM.asm_verbosity!(tm, true)
    ptx = String(LLVM.emit(tm, lmod, LLVM.API.LLVMAssemblyFile))
    return lmod, ptx
end

function _compile(f, full_argtypes; sm="sm_90", feat="+ptx80")
    key = (f, full_argtypes, sm, feat)
    haskey(_gpu_cache, key) && return _gpu_cache[key]
    kname = _sym(f)

    sci, rettype = FE.structured(f, full_argtypes)
    (rettype === Nothing || rettype === Union{}) ||
        @warn "MLIRCUDABackend: kernel inferred rettype = $rettype (expected Nothing)"

    # ctx (`__ctx__`) is arg slot 2 (slot 1 is the function itself).
    mod, _pjt, mlir_ctx, kinds =
        MK.lower_to_mlir_gpu(sci, full_argtypes; kernel_name=kname, ctx_arg=2)

    _run_passes!(mod, mlir_ctx, GPU_PASSES)
    bc = _extract_gpu_binary(mod)
    _lmod, ptx = _bitcode_to_ptx(bc; sm, feat)

    cufn = CuFunction(CuModule(ptx), kname)
    _gpu_cache[key] = (cufn, kinds)
    return cufn, kinds
end

# ----------------------------------------------------------------------------
# Reflection — capture every codegen level for the GPU path. Mirrors the
# `_compile` pipeline but stops at, and returns the text of, each stage.
# ----------------------------------------------------------------------------

# Which passes to run to reach each level. `:lowered` is the full pipeline minus
# the final `gpu-module-to-binary` (so the gpu.module is still readable LLVM/NVVM
# dialect MLIR, not an opaque binary blob).
const _GPU_PASSES_NOBIN = GPU_PASSES[1:end-1]

function _codegen_stages(f, full_argtypes; sm="sm_90", feat="+ptx80",
                         upto::Symbol=:ptx)
    kname = _sym(f)
    order = (:sci, :mlir, :lowered, :llvm, :ptx)
    want = findfirst(==(upto), order)
    want === nothing && error("code_gpu: unknown level :$upto (one of $order)")
    out = Dict{Symbol,String}()

    sci, _ = FE.structured(f, full_argtypes)
    out[:sci] = sprint(show, sci)
    want == 1 && return out

    mod, _pjt, mlir_ctx, _kinds =
        MK.lower_to_mlir_gpu(sci, full_argtypes; kernel_name=kname, ctx_arg=2)
    MK.@with_context mlir_ctx begin
        out[:mlir] = sprint(show, mod)
        want == 2 && return out
        # Lower to LLVM/NVVM dialect (everything but serialise-to-binary).
        _run_passes!(mod, mlir_ctx, _GPU_PASSES_NOBIN)
        out[:lowered] = sprint(show, mod)
        want == 3 && return out
        # Serialise to gpu.binary, extract bitcode → LLVM IR + PTX.
        _run_passes!(mod, mlir_ctx, GPU_PASSES[end:end])
        bc = _extract_gpu_binary(mod)
        lmod, ptx = _bitcode_to_ptx(bc; sm, feat)
        out[:llvm] = string(lmod)   # LLVMPrintModuleToString; `show` gives only the wrapper repr
        want >= 5 && (out[:ptx] = ptx)
    end
    return out
end

# ----------------------------------------------------------------------------
# Launch argument marshalling.
# ----------------------------------------------------------------------------
#
# Each `memref<…>` kernel param lowers (LLVM memref ABI) to the descriptor
# fields {allocated_ptr, aligned_ptr, offset, sizes…, strides…} passed as
# individual scalar params. A `:scalar` param passes its value directly.

function _push_memref!(flat, sig, arr::CuArray)
    p = UInt64(UInt(pointer(arr)))
    push!(flat, p);          push!(sig, Culonglong)   # allocated ptr
    push!(flat, p);          push!(sig, Culonglong)   # aligned ptr
    push!(flat, UInt64(0));  push!(sig, Culonglong)   # offset (elements)
    # The kernel addresses the array via Julia (column-major) linearisation and
    # reads `size(a,k)` as `memref.dim(a, N-k)` (Julia↔MLIR dim reversal). So the
    # LLVM descriptor's sizes/strides must be in REVERSED Julia order. (For a
    # 1-D arg `reverse` is a no-op, so vadd is unchanged.)
    for s in reverse(size(arr));    push!(flat, UInt64(s)); push!(sig, Culonglong); end
    for s in reverse(strides(arr)); push!(flat, UInt64(s)); push!(sig, Culonglong); end
    return nothing
end

function _marshal(args, kinds)
    # `kinds` is one symbol per *param* (memref/scalar). Compile-time-constant
    # args — singletons like `::Val{M}` / `::Type{T}` used only for dispatch —
    # are NOT runtime params (lower_to_mlir_gpu folds them in as Core.Const), so
    # drop them here too; the rest line up 1:1 with `kinds`.
    args = Tuple(a for a in args if !Base.issingletontype(typeof(a)))
    length(kinds) == length(args) ||
        error("MLIRCUDABackend: $(length(kinds)) params vs $(length(args)) args")
    flat = Any[]; sig = DataType[]
    for (a, k) in zip(args, kinds)
        if k === :memref
            a isa CuArray ||
                error("MLIRCUDABackend: memref param expects a CuArray, got $(typeof(a))")
            _push_memref!(flat, sig, a)
        elseif k === :scalar
            push!(flat, a); push!(sig, typeof(a))
        else
            error("MLIRCUDABackend: unsupported param kind :$k")
        end
    end
    return flat, sig
end

# ----------------------------------------------------------------------------
# ctx type + launch geometry.
# ----------------------------------------------------------------------------

# Device array types trip scalar-indexing guards during inference; map them to
# the host `Array` of the same eltype/rank (which lowers identically).
_host_argtype(::Type{<:CuArray{T,N}}) where {T,N} = Array{T,N}
_host_argtype(@nospecialize(T::Type)) = T

function _resolve_wgsize(obj::KA.Kernel{MLIRCUDABackend}, workgroupsize)
    wg_T = KA.workgroupsize(obj)
    if wg_T <: NDI.StaticSize
        static = NDI.get(wg_T)
        if workgroupsize !== nothing
            wg = workgroupsize isa Integer ? (workgroupsize,) : Tuple(workgroupsize)
            wg == static || error(
                "MLIRCUDABackend: workgroupsize=$wg conflicts with the kernel's " *
                "static workgroupsize $static.")
        end
        return static
    end
    workgroupsize === nothing && return (256,)
    workgroupsize isa Integer && return (workgroupsize,)
    return Tuple(workgroupsize)
end

# Build the `CompilerMetadata` *type* for inference. Static sizes give the
# grid dimensionality (used by the N-D `@index` overlays) and a clean ctx.
function _ctx_type(nd::NTuple{D,Int}, wg::NTuple{D,Int}) where {D}
    ndr = NDI.StaticSize{nd}
    wgs = NDI.StaticSize{wg}
    grp = NDI.StaticSize{map(cld, nd, wg)}
    ndobj = NDI.NDRange{D, grp, wgs, Nothing, Nothing}
    return KA.CompilerMetadata{ndr, NDI.NoDynamicCheck, Nothing, Nothing, ndobj}
end

# Shared by the launcher and `code_gpu`: resolve geometry + build the inference
# signature. Infer with HOST array types — the SCI walk only needs each arg's
# eltype/ndims (→ `memref<?×…×T>`), and inferring a kernel body's `A[i]` on a
# `CuArray` trips `GPUArrays.assertscalar`. (Marshalling still uses the real
# device arrays.)
function _launch_setup(obj::KA.Kernel{MLIRCUDABackend}, args, ndrange, workgroupsize)
    ndrange === nothing && error("MLIRCUDABackend: ndrange must be specified")
    wg = _resolve_wgsize(obj, workgroupsize)
    nd = ndrange isa Integer ? (ndrange,) : Tuple(ndrange)
    length(wg) == length(nd) || error(
        "MLIRCUDABackend: ndrange $nd ($(length(nd))-D) and workgroupsize $wg " *
        "($(length(wg))-D) must have the same number of dimensions.")
    all(nd[d] % wg[d] == 0 for d in 1:length(wg)) || error(
        "MLIRCUDABackend: ndrange=$nd not a per-dim multiple of workgroupsize=$wg " *
        "(no tail-block masking yet — __validindex→true).")
    ctxT = _ctx_type(nd, wg)
    full_argtypes = Tuple{ctxT, map(a -> _host_argtype(typeof(a)), args)...}
    return full_argtypes, nd, wg
end

function (obj::KA.Kernel{MLIRCUDABackend})(args...; ndrange=nothing,
                                                     workgroupsize=nothing)
    full_argtypes, nd, wg = _launch_setup(obj, args, ndrange, workgroupsize)
    cufn, kinds = _compile(obj.f, full_argtypes)
    flat, sig = _marshal(args, kinds)
    grid = map(cld, nd, wg)          # blocks per dim
    cudacall(cufn, Tuple{sig...}, flat...; threads=wg, blocks=grid)
    return nothing
end

# ----------------------------------------------------------------------------
# code_gpu — reflection entry points (see MLIRKernels.code_gpu docstring).
# ----------------------------------------------------------------------------

# Low-level form: explicit (gpu_body, full_argtypes::Type).
function MK.code_gpu(@nospecialize(f), full_argtypes::Type; level::Symbol=:ptx,
                     sm="sm_90", feat="+ptx80")
    stages = _codegen_stages(f, full_argtypes; sm, feat, upto=level)
    return stages[level]
end

# Ergonomic form: a KA kernel + launch args (mirrors a `(obj)(args…; ndrange)`).
function MK.code_gpu(obj::KA.Kernel{MLIRCUDABackend}, args...; level::Symbol=:ptx,
                     ndrange=nothing, workgroupsize=nothing, sm="sm_90", feat="+ptx80")
    full_argtypes, _nd, _wg = _launch_setup(obj, args, ndrange, workgroupsize)
    return MK.code_gpu(obj.f, full_argtypes; level, sm, feat)
end

end # module MLIRCUDAExt
