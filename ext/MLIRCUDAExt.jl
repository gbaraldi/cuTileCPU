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
import CUDA_Compiler_jll
import GPUArrays
using GPUArraysCore: GPUArraysCore, AbstractGPUArray, AbstractGPUArrayStyle

struct MLIRCUDABackend <: KA.GPU end

# ----------------------------------------------------------------------------
# MLIRArray — the backend's array type.
# ----------------------------------------------------------------------------
#
# A thin wrapper around a CuArray whose `KA.get_backend` returns MLIRCUDABackend.
# This is what lets backend-agnostic KA code (GPUArrays, AcceleratedKernels) pick
# our backend automatically: those libraries dispatch on `get_backend(array)`, so
# wrapping the device array in an MLIRArray routes their `@kernel` launches here.
# We can't instead define `get_backend(::CuArray)` — that belongs to CUDA.jl's
# CUDABackend. The wrapper is host-indexable (for verification) and marshals by
# unwrapping to the inner CuArray.
#
# `<: AbstractGPUArray` (not just `AbstractArray`) makes GPUArrays' generic
# `broadcast`/`map!`/`fill!`/`mapreduce`/`sort` dispatch here: those are KA
# kernels launched via `get_backend`, so they compile through MLIRKernels. The
# scalar `getindex`/`setindex!` defer to the CuArray (CUDA's scalar-indexing
# guard still fires unless `@allowscalar`).
struct MLIRArray{T,N} <: AbstractGPUArray{T,N}
    data::CuArray{T,N}
end

unwrap(a::MLIRArray) = a.data
unwrap(@nospecialize(a)) = a

Base.size(a::MLIRArray) = size(a.data)
Base.getindex(a::MLIRArray, i::Int...) = getindex(a.data, i...)
Base.setindex!(a::MLIRArray, v, i::Int...) = (setindex!(a.data, v, i...); v)
Base.IndexStyle(::Type{<:MLIRArray}) = IndexLinear()
Base.similar(a::MLIRArray, ::Type{T}, dims::Dims) where {T} = MLIRArray(similar(a.data, T, dims))
Base.similar(::Type{MLIRArray{T}}, dims::Dims) where {T} = MLIRArray(CuArray{T}(undef, dims))
Base.Array(a::MLIRArray) = Array(a.data)
Base.pointer(a::MLIRArray) = pointer(a.data)
Base.strides(a::MLIRArray) = strides(a.data)
Base.elsize(::Type{MLIRArray{T,N}}) where {T,N} = sizeof(T)
Base.unsafe_convert(::Type{CUDA.CuPtr{T}}, a::MLIRArray{T}) where {T} =
    Base.unsafe_convert(CUDA.CuPtr{T}, a.data)

# ---- broadcasting ----------------------------------------------------------
# A broadcast `a .+ b` over MLIRArrays must (1) produce an MLIRArray and (2)
# launch GPUArrays' broadcast kernel through our backend. Both follow from a
# BroadcastStyle that is an `AbstractGPUArrayStyle` (so GPUArrays' overrides win
# over Base's scalar fallback) plus a `similar` that allocates an MLIRArray.
struct MLIRArrayStyle{N} <: AbstractGPUArrayStyle{N} end
MLIRArrayStyle(::Val{N}) where {N} = MLIRArrayStyle{N}()
MLIRArrayStyle{M}(::Val{N}) where {N,M} = MLIRArrayStyle{N}()
Base.Broadcast.BroadcastStyle(::Type{<:MLIRArray{T,N}}) where {T,N} = MLIRArrayStyle{N}()
Base.similar(bc::Base.Broadcast.Broadcasted{MLIRArrayStyle{N}}, ::Type{T}, dims) where {T,N} =
    MLIRArray(CuArray{T}(undef, length.(dims)))
# GPUArrays materialises contiguous views / reshape / reinterpret via `derive`
# (a new array sharing storage). Delegate to the wrapped CuArray and rewrap, so a
# `view(::MLIRArray, …)` stays an MLIRArray (get_backend → our backend) instead of
# erroring. (AcceleratedKernels' block reductions `view` the source.)
GPUArrays.derive(::Type{T}, a::MLIRArray, osize::Dims, offset::Int) where {T} =
    MLIRArray(GPUArrays.derive(T, a.data, osize, offset))
Base.copyto!(d::MLIRArray, s::AbstractArray) = (copyto!(d.data, s); d)
Base.copyto!(d::AbstractArray, s::MLIRArray) = (copyto!(d, s.data); d)
Base.copyto!(d::MLIRArray, s::MLIRArray) = (copyto!(d.data, s.data); d)
# A view of an MLIRArray must copy to the host via the wrapped CuArray: the
# generic `AbstractArray` path is element-wise `getindex`, which trips CUDA's
# scalar-indexing guard. (AK's reduce/scan copy partial results with
# `Vector(@view dst[1:len])`.)
_cuview(s::SubArray{<:Any,<:Any,<:MLIRArray}) = view(parent(s).data, parentindices(s)...)
Base.copyto!(d::Array, s::SubArray{<:Any,<:Any,<:MLIRArray}) = (copyto!(d, _cuview(s)); d)
Base.Array(s::SubArray{T,N,<:MLIRArray}) where {T,N} = Array(_cuview(s))
CUDA.unsafe_free!(a::MLIRArray) = CUDA.unsafe_free!(a.data)

# ----------------------------------------------------------------------------
# Backend protocol — the backend's native array is MLIRArray; storage defers to
# CUDA.
# ----------------------------------------------------------------------------

KA.get_backend(::MLIRArray) = MLIRCUDABackend()
KA.allocate(::MLIRCUDABackend, ::Type{T}, dims::Tuple) where {T} = MLIRArray(CuArray{T}(undef, dims))
KA.zeros(::MLIRCUDABackend, ::Type{T}, dims::Tuple) where {T} = MLIRArray(CUDA.zeros(T, dims))
KA.ones(::MLIRCUDABackend, ::Type{T}, dims::Tuple) where {T} = MLIRArray(CUDA.ones(T, dims))
KA.synchronize(::MLIRCUDABackend) = CUDA.synchronize()
KA.functional(::MLIRCUDABackend) = CUDA.functional()
KA.supports_atomics(::MLIRCUDABackend) = true
# Data movement (host↔device, device↔device) defers to CUDA's copyto!.
KA.copyto!(::MLIRCUDABackend, dst, src) = (Base.copyto!(unwrap(dst), unwrap(src)); dst)

# GPUArrays' generic `map!`/`broadcast` call `KA.launch_config(kernel, ndrange,
# workgroupsize)` and launch with `config[1]`/`config[2]`. Our launcher takes
# `ndrange`/`workgroupsize` directly and pads+masks the grid, so we just
# normalise to tuples and pick a default block size. (iterspace/dynamic — the
# 3rd/4th elements KA's own launcher uses — are unused by GPUArrays.)
@inline function KA.launch_config(::KA.Kernel{MLIRCUDABackend}, ndrange, workgroupsize)
    ndrange isa Integer && (ndrange = (ndrange,))
    workgroupsize isa Integer && (workgroupsize = (workgroupsize,))
    if workgroupsize === nothing
        workgroupsize = ntuple(d -> d == 1 ? min(256, ndrange[d]) : 1, length(ndrange))
    end
    return ndrange, workgroupsize, nothing, nothing
end

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

# Resolve libdevice externs (`__nv_fabsf`, `__nv_sqrtf`, …) that the gpu→nvvm
# pipeline emits for math ops, by linking NVIDIA's libdevice bitcode. We can't
# reuse GPUCompiler's `link_libraries!`/`compile` — those are keyed on a
# `CompilerJob` built from a Julia `MethodInstance`, which we don't have (our IR
# comes from MLIR, not Julia inference). So we link directly with LLVM.jl's
# `link!(...; only_needed=true)` — the job-free path GPUCompiler's own
# deprecation notice recommends — pulling the .bc from `CUDA_Compiler_jll`. The
# NVPTX backend runs NVVMReflect during codegen, resolving libdevice's
# `__nvvm_reflect` calls. No-op when there are no `__nv_*` references.
function _link_libdevice!(lmod)
    any(f -> LLVM.isdeclaration(f) && startswith(LLVM.name(f), "__nv_"),
        LLVM.functions(lmod)) || return
    lib = parse(LLVM.Module, read(CUDA_Compiler_jll.libdevice); lazy=true)
    LLVM.triple!(lib, LLVM.triple(lmod))
    LLVM.datalayout!(lib, LLVM.datalayout(lmod))
    LLVM.link!(lmod, lib; only_needed=true)
    return
end

# gpu.binary{format=llvm} bitcode → (LLVM.Module, PTX string), with libdevice
# linked and LLVM's default -O2 run. The driver JITs PTX → SASS at module load.
# `stages`, when given, captures the LLVM IR before (`:llvm_unopt`) and after
# (`:llvm`) the -O2 pipeline — for reflection / `code_gpu`.
function _bitcode_to_ptx(bc; sm="sm_90", feat="+ptx80",
                         stages::Union{Nothing,Dict{Symbol,String}}=nothing)
    lctx = LLVM.Context()
    return LLVM.context!(lctx) do
        lmod = parse(LLVM.Module, bc)
        triple = "nvptx64-nvidia-cuda"
        LLVM.triple!(lmod, triple)
        _link_libdevice!(lmod)            # parse libdevice in the SAME context, then link
        tm = LLVM.TargetMachine(LLVM.Target(; triple), triple, sm, feat)
        LLVM.asm_verbosity!(tm, true)
        stages === nothing || (stages[:llvm_unopt] = string(lmod))   # linked, pre-O2
        # We emit no LLVM-level optimization of our own; run LLVM's default -O2
        # pipeline (inline/GVN/DCE) — also strips the lazily-linked libdevice down
        # to the referenced functions. LLVM ≥17 weaves NVPTX's NVVMReflect in at
        # PipelineStart, resolving libdevice's `__nvvm_reflect`. Job-free (the
        # GPUCompiler `optimize_module!` is keyed on a CompilerJob we don't have).
        LLVM.run!("default<O2>", lmod, tm)
        stages === nothing || (stages[:llvm] = string(lmod))         # post-O2
        ptx = String(LLVM.emit(tm, lmod, LLVM.API.LLVMAssemblyFile))
        return lmod, ptx
    end
end

# Env-var kernel dumping. `MLIRKERNELS_DUMP` = a comma-separated subset of
# sci,mlir,lowered,llvm_unopt,llvm,ptx (or "all") prints those levels for every
# GPU kernel — `llvm_unopt`/`llvm` are the LLVM IR before/after the -O2 pipeline
# as it compiles; `MLIRKERNELS_DUMP_FILTER=<substr>` restricts to kernels whose
# name contains <substr>. Best-effort (dumps whatever stages succeed) and goes
# to stderr, so it works even when a kernel is launched deep inside a library.
const _DUMP_ORDER = (:sci, :mlir, :lowered, :llvm_unopt, :llvm, :ptx)

function _maybe_dump_kernel(f, full_argtypes, kname; sm, feat, nd_dims, optimize=true)
    spec = get(ENV, "MLIRKERNELS_DUMP", "")
    isempty(spec) && return
    filt = get(ENV, "MLIRKERNELS_DUMP_FILTER", "")
    (isempty(filt) || occursin(filt, String(kname))) || return
    want = spec == "all" ? collect(_DUMP_ORDER) :
           Symbol[Symbol(strip(s)) for s in split(spec, ',') if !isempty(strip(s))]
    idxs = filter(!isnothing, [findfirst(==(l), _DUMP_ORDER) for l in want])
    isempty(idxs) && return
    upto = _DUMP_ORDER[maximum(idxs)]
    stages = try
        _codegen_stages(f, full_argtypes; sm, feat, upto, nd_dims, optimize)
    catch e
        printstyled(stderr, "===== [MLIRKernels dump] $kname: staging failed at :$upto =====\n";
                    color=:red, bold=true)
        showerror(stderr, e); println(stderr)
        return
    end
    for lvl in _DUMP_ORDER
        (lvl in want && haskey(stages, lvl)) || continue
        printstyled(stderr, "===== [MLIRKernels dump] $kname :$lvl =====\n"; color=:cyan, bold=true)
        println(stderr, stages[lvl])
    end
    return
end

function _compile(f, full_argtypes; sm="sm_90", feat="+ptx80", nd_dims=Int[], optimize::Bool=true)
    key = (f, full_argtypes, sm, feat, nd_dims, optimize)
    haskey(_gpu_cache, key) && return _gpu_cache[key]
    kname = _sym(f)
    _maybe_dump_kernel(f, full_argtypes, kname; sm, feat, nd_dims, optimize)

    sci, rettype = FE.structured(f, full_argtypes)
    (rettype === Nothing || rettype === Union{}) ||
        @warn "MLIRCUDABackend: kernel inferred rettype = $rettype (expected Nothing)"

    # ctx (`__ctx__`) is arg slot 2 (slot 1 is the function itself).
    mod, _pjt, mlir_ctx, kinds =
        MK.lower_to_mlir_gpu(sci, full_argtypes; kernel_name=kname, ctx_arg=2, nd_dims, optimize)

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
                         upto::Symbol=:ptx, nd_dims=Int[], optimize::Bool=true)
    kname = _sym(f)
    order = (:sci, :mlir, :lowered, :llvm_unopt, :llvm, :ptx)
    want = findfirst(==(upto), order)
    want === nothing && error("code_gpu: unknown level :$upto (one of $order)")
    out = Dict{Symbol,String}()

    sci, _ = FE.structured(f, full_argtypes)
    # Optimize the SCI here (so the :sci level reflects the toggle); then lower
    # with optimize=false to avoid running the passes twice.
    optimize && MK.SCIOpt.optimize_sci!(sci)
    out[:sci] = sprint(show, sci)
    want == 1 && return out

    mod, _pjt, mlir_ctx, _kinds =
        MK.lower_to_mlir_gpu(sci, full_argtypes; kernel_name=kname, ctx_arg=2, nd_dims, optimize=false)
    MK.@with_context mlir_ctx begin
        out[:mlir] = sprint(show, mod)
        want == 2 && return out
        # Lower to LLVM/NVVM dialect (everything but serialise-to-binary).
        _run_passes!(mod, mlir_ctx, _GPU_PASSES_NOBIN)
        out[:lowered] = sprint(show, mod)
        want == 3 && return out
        # Serialise to gpu.binary, extract bitcode → LLVM IR (pre/post-O2) + PTX.
        _run_passes!(mod, mlir_ctx, GPU_PASSES[end:end])
        bc = _extract_gpu_binary(mod)
        _lmod, ptx = _bitcode_to_ptx(bc; sm, feat, stages=out)  # fills :llvm_unopt + :llvm
        want >= findfirst(==(:ptx), order) && (out[:ptx] = ptx)
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

# Flatten a launch arg to match the flattened param list (see
# lower_to_mlir_gpu): drop singletons (`Val`/`Type`/captured user fns — folded
# as Core.Const), unwrap arrays to their CuArray, and expand a closure/functor
# into its captured array+scalar fields (fieldname order — matching the
# signature flattening).
function _flatten_args!(out, @nospecialize(a))
    Base.issingletontype(typeof(a)) && return out
    au = unwrap(a)
    if au isa CuArray || au isa Number
        push!(out, au)
    elseif isstructtype(typeof(a))
        for fn in fieldnames(typeof(a))
            _flatten_args!(out, getfield(a, fn))
        end
    else
        error("MLIRCUDABackend: cannot marshal arg of type $(typeof(a))")
    end
    return out
end

function _marshal(args, kinds)
    # `kinds` is one symbol per flattened *param* (memref/scalar). Flatten the
    # runtime args the same way the signature was flattened, then they line up.
    flat_vals = Any[]
    for a in args; _flatten_args!(flat_vals, a); end
    length(kinds) == length(flat_vals) ||
        error("MLIRCUDABackend: $(length(kinds)) params vs $(length(flat_vals)) marshalled values")
    flat = Any[]; sig = DataType[]
    for (a, k) in zip(flat_vals, kinds)
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
# Does a type contain a device array anywhere in its parameter tree?
_has_device_array(@nospecialize(x)) = false
function _has_device_array(@nospecialize(T::Type))
    (T <: CuArray || T <: MLIRArray) && return true
    isconcretetype(T) && isstructtype(T) &&
        any(p -> _has_device_array(p), T.parameters)
end

_host_argtype(::Type{<:CuArray{T,N}}) where {T,N} = Array{T,N}
_host_argtype(::Type{<:MLIRArray{T,N}}) where {T,N} = Array{T,N}
function _host_argtype(@nospecialize(T::Type))
    # A wrapper/closure carrying device arrays (a closure's captures, a
    # SubArray's `.parent`, …): rebuild the type with every device-array type
    # param remapped to a host Array, recursively. Then inference indexes those
    # arrays via Array's getindex (no GPUArrays.assertscalar in the kernel IR)
    # and inlines; marshalling still unwraps the real device arrays. Element
    # type/ndims are unchanged, so the flattened memref params match. Only Type
    # params are remapped — value params (ndims, flags) are kept verbatim.
    _has_device_array(T) || return T
    try
        return T.name.wrapper{map(p -> p isa Type ? _host_argtype(p) : p,
                                  collect(T.parameters))...}
    catch
        return T
    end
end

function _resolve_wgsize(obj::KA.Kernel{MLIRCUDABackend}, workgroupsize, nd::Tuple)
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
    # Default block: up to 256 lanes along dim 1, singletons elsewhere — and the
    # SAME rank as the ndrange (GPUArrays' broadcast launches an N-D ndrange with
    # no workgroupsize, so a fixed `(256,)` would mismatch the rank).
    if workgroupsize === nothing
        return ntuple(d -> d == 1 ? min(256, nd[1]) : 1, length(nd))
    end
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
# ndrange: explicit kwarg wins; otherwise fall back to the kernel's STATIC
# ndrange (baked in via `kernel(backend, wg, ndrange)`), as KA's testsuite does.
function _resolve_ndrange(obj::KA.Kernel{MLIRCUDABackend}, ndrange)
    if ndrange !== nothing
        return ndrange isa Integer ? (ndrange,) : Tuple(ndrange)
    end
    nd_T = KA.ndrange(obj)
    nd_T <: NDI.StaticSize ||
        error("MLIRCUDABackend: ndrange must be specified (kernel has no static ndrange)")
    return NDI.get(nd_T)
end

function _launch_setup(obj::KA.Kernel{MLIRCUDABackend}, args, ndrange, workgroupsize)
    nd = _resolve_ndrange(obj, ndrange)
    wg = _resolve_wgsize(obj, workgroupsize, nd)
    length(wg) == length(nd) || error(
        "MLIRCUDABackend: ndrange $nd ($(length(nd))-D) and workgroupsize $wg " *
        "($(length(wg))-D) must have the same number of dimensions.")
    # ndrange need not be a multiple of wg: the grid is padded (`cld`) and
    # `__validindex` masks the tail. (`unsafe_indices=true` skips the mask, so it
    # still needs an exact-multiple ndrange.)
    ctxT = _ctx_type(nd, wg)
    full_argtypes = Tuple{ctxT, map(a -> _host_argtype(typeof(a)), args)...}
    return full_argtypes, nd, wg
end

function (obj::KA.Kernel{MLIRCUDABackend})(args...; ndrange=nothing,
                                                     workgroupsize=nothing)
    full_argtypes, nd, wg = _launch_setup(obj, args, ndrange, workgroupsize)
    cufn, kinds = _compile(obj.f, full_argtypes; nd_dims=Int[nd...])
    flat, sig = _marshal(args, kinds)
    grid = map(cld, nd, wg)          # blocks per dim (padded)
    cudacall(cufn, Tuple{sig...}, flat...; threads=wg, blocks=grid)
    return nothing
end

# ----------------------------------------------------------------------------
# code_gpu — reflection entry points (see MLIRKernels.code_gpu docstring).
# ----------------------------------------------------------------------------

# Low-level form: explicit (gpu_body, full_argtypes::Type). `optimize` toggles the
# SCI optimization passes (DCE/CSE/LICM) — handy for opt-vs-raw codegen diffs.
function MK.code_gpu(@nospecialize(f), full_argtypes::Type; level::Symbol=:ptx,
                     sm="sm_90", feat="+ptx80", nd_dims=Int[], optimize::Bool=true)
    stages = _codegen_stages(f, full_argtypes; sm, feat, upto=level, nd_dims, optimize)
    return stages[level]
end

# Ergonomic form: a KA kernel + launch args (mirrors a `(obj)(args…; ndrange)`).
function MK.code_gpu(obj::KA.Kernel{MLIRCUDABackend}, args...; level::Symbol=:ptx,
                     ndrange=nothing, workgroupsize=nothing, sm="sm_90", feat="+ptx80",
                     optimize::Bool=true)
    full_argtypes, nd, _wg = _launch_setup(obj, args, ndrange, workgroupsize)
    return MK.code_gpu(obj.f, full_argtypes; level, sm, feat, nd_dims=Int[nd...], optimize)
end

end # module MLIRCUDAExt
