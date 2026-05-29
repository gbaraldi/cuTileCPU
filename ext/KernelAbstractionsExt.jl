module KernelAbstractionsExt

# Mirrors the CUDA.jl / AMDGPU.jl / oneAPI.jl / Metal.jl KA backend pattern:
#
#   1. `struct MLIRBackend <: KA.GPU` so KA's `@kernel` macro picks the
#      `gpu_*` (SIMT) function body, not the `cpu_*` (loop-splitting) one.
#
#   2. `@overlay MLIRKernels.Frontend.METHOD_TABLE` redefinitions of the KA
#      intrinsics. Inference runs under MLIRKernels's Frontend interpreter
#      (src/frontend.jl), so the overlays map KA intrinsics onto the
#      `Frontend.Intrinsics` markers the walker recognises by name.
#
#   3. `(::Kernel{MLIRBackend})(args...; ndrange, workgroupsize)` builds the
#      KA `CompilerMetadata` type and calls `ka_function` → `lower_to_mlir_ka`
#      → in-process MLIR pipeline → clang → dlopen, then dispatches the grid
#      via the standard SPMD-style launch path.
#
# The Frontend owns its interpreter, Intrinsics module, and overlay method
# table, so the markers are defined at the package's own precompile and the
# overlays are ordinary precompile-safe method additions (no `__init__` eval).

using KernelAbstractions
const KA = KernelAbstractions
const NDI = KernelAbstractions.NDIteration
using MLIRKernels
const FE = MLIRKernels.Frontend

# Atomix is KA's portable-atomic backend: `KA.@atomic` / `Atomix.@atomic` both
# expand to `Atomix.modify!(referenceable(arr)[i], op, x, order)`. We overlay
# `modify!` (below) onto a Frontend marker so it survives inference as a clean
# atomic instead of inlining down to a raw `atomicrmw` llvmcall.
using Atomix

import Base.Experimental: @overlay

# ----------------------------------------------------------------------------
# 1. Backend
# ----------------------------------------------------------------------------

struct MLIRBackend <: KA.GPU end

# ----------------------------------------------------------------------------
# 2. Overlays into the Frontend method table
# ----------------------------------------------------------------------------
#
# Each maps a KA intrinsic onto a `Frontend.Intrinsics` marker (a @noinline +
# compilerbarrier function that survives inference under the Frontend's
# default-opt-params interpreter) which the walker intercepts. All are plain
# method additions to OUR table — precompile-safe.

# `__index_Global_Linear(ctx)` → global linear thread index (1-based).
@overlay FE.METHOD_TABLE KA.__index_Global_Linear(ctx) = FE.Intrinsics.global_index()

# `@index(Local, Linear)` / `@index(Group, Linear)` expand to one-arg `(ctx)`
# calls (the `:Linear` kind is a macro-stripped literal), so the unary overlay
# is the matching one. KA's 2-arg `(ctx, ::CartesianIndex)` defs in cpu.jl are
# CPU-emit-only and never reached on the Frontend path.
@overlay FE.METHOD_TABLE KA.__index_Local_Linear(ctx) = FE.Intrinsics.local_index()
@overlay FE.METHOD_TABLE KA.__index_Group_Linear(ctx) = FE.Intrinsics.group_index()

# `@groupsize()` → the workgroup size as an `NTuple` (KA semantics). We return
# the STATIC size straight off the ctx type (the workgroup `StaticSize` is the
# NDRange's 3rd type param) so it's a COMPILE-TIME CONSTANT — essential because
# `@localmem T (@groupsize())` and `prod(@groupsize())` feed `Val{Dims}`, which
# needs a constant. The runtime `group_size()` marker (block_dim) is kept only
# for a future DynamicSize path.
@overlay FE.METHOD_TABLE KA.groupsize(
        ctx::KA.CompilerMetadata{A, B, C, D, <:NDI.NDRange{N, BL, WGS}}) where {A, B, C, D, N, BL, WGS} =
    NDI.get(WGS)

# `@index(Global/Local/Group, NTuple)` → unary `__index_*_NTuple(ctx)` returning
# an `NTuple{N,Int}` of 1-based per-dim coords. We read the grid dimensionality
# `N` straight off the ctx type (`CompilerMetadata{…, NDRange{N,…}}`) and pass it
# to the marker so inference sees a concrete-arity tuple; the walker fills in the
# per-dim vectors. (`@index(…, Cartesian)` / N-D `__validindex` masking are TODO.)
@overlay FE.METHOD_TABLE KA.__index_Global_NTuple(
        ctx::KA.CompilerMetadata{A,B,C,D,<:NDI.NDRange{N}}) where {A,B,C,D,N} =
    FE.Intrinsics.global_ntuple(Val(N))
@overlay FE.METHOD_TABLE KA.__index_Local_NTuple(
        ctx::KA.CompilerMetadata{A,B,C,D,<:NDI.NDRange{N}}) where {A,B,C,D,N} =
    FE.Intrinsics.local_ntuple(Val(N))
@overlay FE.METHOD_TABLE KA.__index_Group_NTuple(
        ctx::KA.CompilerMetadata{A,B,C,D,<:NDI.NDRange{N}}) where {A,B,C,D,N} =
    FE.Intrinsics.group_ntuple(Val(N))

# `@index(…, Cartesian)` → a `CartesianIndex{N}` wrapping the same per-dim coords
# as the NTuple form (CartesianIndex's `.I` field IS the NTuple). `A[I]` then
# inlines through `I.I` to the per-dim coords + column-major linearisation.
# GPUArrays uses this form heavily (broadcast/copy/clamp/transpose/…).
@overlay FE.METHOD_TABLE KA.__index_Global_Cartesian(
        ctx::KA.CompilerMetadata{A,B,C,D,<:NDI.NDRange{N}}) where {A,B,C,D,N} =
    CartesianIndex(FE.Intrinsics.global_ntuple(Val(N)))
@overlay FE.METHOD_TABLE KA.__index_Local_Cartesian(
        ctx::KA.CompilerMetadata{A,B,C,D,<:NDI.NDRange{N}}) where {A,B,C,D,N} =
    CartesianIndex(FE.Intrinsics.local_ntuple(Val(N)))
@overlay FE.METHOD_TABLE KA.__index_Group_Cartesian(
        ctx::KA.CompilerMetadata{A,B,C,D,<:NDI.NDRange{N}}) where {A,B,C,D,N} =
    CartesianIndex(FE.Intrinsics.group_ntuple(Val(N)))

# `__validindex(ctx)` — tail-block masking; walker lowers the marker (GPU:
# `∧_d (global_d < ndrange[d])`, CPU: `true`).
@overlay FE.METHOD_TABLE KA.__validindex(ctx) = FE.Intrinsics.valid_index()

# `__synchronize()` → workgroup barrier marker. CPU SIMD has no warp barrier
# so the walker lowers `:barrier` to a no-op; on the GPU SIMT path with no
# cross-lane communication this is correct for the current scope.
@overlay FE.METHOD_TABLE KA.__synchronize() = FE.Intrinsics.barrier()

# `@localmem T dims` → `SharedMemory(T, Val(dims), Val(id))`. Map onto the
# `shared_alloc` marker (dropping the gensym id — the walker emits a distinct
# alloca per call site anyway, and the marker's `donotdelete` blocks CSE).
# The walker lowers it to a workgroup-address-space `memref.alloca` (real GPU
# shared memory on the SIMT path); `@synchronize` becomes a `gpu.barrier`.
@overlay FE.METHOD_TABLE KA.SharedMemory(::Type{T}, ::Val{Dims}, ::Val) where {T, Dims} =
    FE.Intrinsics.shared_alloc(T, Val(Dims))

# `for i in start:step:stop` builds a StepRange whose last element comes from
# `Base.steprange_last`. The default pulls in `ArgumentError`, a `@noinline
# overflow_case`, and `checked_srem_int` — none of which lower. This GPU-safe
# version uses plain unsigned `rem`. Needed e.g. by KA's histogram
# (`for min_element in 1:gs:N`). Must be `@consistent_overlay` + `:foldable` —
# a plain `@overlay` isn't honoured inside the range machinery's :consistent
# context, so the default (un-lowerable) version would leak through.
Base.Experimental.@consistent_overlay FE.METHOD_TABLE function Base.steprange_last(start::T, step::T, stop::T) where {T <: Base.BitInteger}
    stop == start && return stop
    if step > zero(step)
        stop < start && return start - oneunit(step)
        remain = signed(unsigned(stop - start) % unsigned(step))
        return stop - remain
    else
        stop > start && return start + oneunit(step)
        remain = signed(unsigned(start - stop) % unsigned(-step))
        return stop + remain
    end
end

# `@private T dims` → `Scratchpad(__ctx__, T, Val(dims))`. Per-thread private
# storage: map onto `private_alloc` (the walker emits a default-space alloca —
# per-thread `.local` on GPU). Unlike KA's CPU Scratchpad we don't add the
# implicit workitem dimension (SIMT gives each thread its own copy directly).
@overlay FE.METHOD_TABLE KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims} =
    FE.Intrinsics.private_alloc(T, Val(Dims))

# `KA.@atomic` / `Atomix.@atomic arr[i] op= x` — KA's *portable* atomic, the
# form KA docs recommend (CUDA/AMDGPU/oneAPI all override `Atomix.modify!` for
# their device arrays). It expands to `Atomix.modify!(referenceable(arr)[i],
# op, x, order)[2]`. Under the Frontend's default opt params this `modify!`
# (not `@noinline`) would otherwise inline all the way down to raw pointer
# arithmetic + a `UnsafeAtomics` `atomicrmw` llvmcall — opaque to the walker.
# Overlaying it onto our `atomic_index!` marker stops that cascade with the
# array, op, value and 1-based linear index still as clean SSA values; the
# walker (`:atomic_index!`) emits `memref.atomic_rmw`. `modify!` returns the
# `(old, new)` pair Atomix expects; both are bound to `x` since the result is
# discarded by `@atomic … op= …` used as a statement.
@overlay FE.METHOD_TABLE function Atomix.modify!(
        ref::Atomix.Internal.IndexableRef, op::OP, x, ord) where {OP}
    FE.Intrinsics.atomic_index!(ref.data, op, x, ref.indices[1])
    return (x, x)
end

# ----------------------------------------------------------------------------
# 3. KA backend protocol
# ----------------------------------------------------------------------------
#
# We only implement the methods the kernel call needs. Allocation /
# synchronisation / copyto! all fall back to KA's default `<: KA.GPU`
# behaviour on the host (we're CPU-targeting, so host == device).

KA.allocate(::MLIRBackend, ::Type{T}, dims::Tuple) where {T} =
    MLIRKernels.aligned_array(T, dims; alignment=128)

KA.zeros(b::MLIRBackend, ::Type{T}, dims::Tuple) where {T} =
    fill!(KA.allocate(b, T, dims), zero(T))
KA.ones(b::MLIRBackend, ::Type{T}, dims::Tuple) where {T} =
    fill!(KA.allocate(b, T, dims), one(T))

KA.synchronize(::MLIRBackend) = nothing
KA.functional(::MLIRBackend) = true
KA.argconvert(::MLIRBackend, x) = x
# NOTE: deliberately no `KA.get_backend(::Array) = MLIRBackend()`. Plain
# `Array` already dispatches to KA's `CPU()` backend, and overriding here
# would silently steal every `Array`-touching KA call from the default
# (and fails precompile with method-overwriting anyway). Users select the
# backend explicitly: `vadd!(MLIRBackend(), 16)(C, A, B; ndrange=N)`.

# `mkcontext` and `launch_config` follow KA's GPU defaults (no per-backend
# specialisation needed) by reusing the generic methods. We provide a
# minimal stub so KA's `partition` can build the CompilerMetadata type.

KA.mkcontext(kernel::KA.Kernel{MLIRBackend}, ndrange, iterspace) =
    KA.CompilerMetadata{KA.ndrange(kernel), NDI.NoDynamicCheck}(ndrange, iterspace)

function KA.launch_config(kernel::KA.Kernel{MLIRBackend}, ndrange, workgroupsize)
    ndrange isa Integer && (ndrange = (ndrange,))
    workgroupsize isa Integer && (workgroupsize = (workgroupsize,))
    if KA.workgroupsize(kernel) <: KA.NDIteration.DynamicSize && workgroupsize === nothing
        workgroupsize = (16,)  # default lane width for MLIRKernels SPMD lowering
    end
    iterspace, dynamic = KA.partition(kernel, ndrange, workgroupsize)
    return ndrange, workgroupsize, iterspace, dynamic
end

# ----------------------------------------------------------------------------
# 4. The kernel-callable launcher
# ----------------------------------------------------------------------------

# Resolve the effective workgroupsize. KA's StaticSize{(N,)} encodes it in
# the kernel type; DynamicSize falls back to the launch-time kwarg or our
# 16-lane default.
function _resolve_wgsize(obj::KA.Kernel{MLIRBackend}, workgroupsize)
    wg_T = KA.workgroupsize(obj)
    if wg_T <: NDI.StaticSize
        static = NDI.get(wg_T)
        # The workgroup size is baked into the kernel type. Honour an explicit
        # launch-time kwarg only if it AGREES — otherwise it would be silently
        # ignored (the kernel was already specialised for `static`), so error
        # rather than run a size the user didn't ask for.
        if workgroupsize !== nothing
            wg = workgroupsize isa Integer ? (workgroupsize,) : Tuple(workgroupsize)
            wg == static || error(
                "MLIRBackend: workgroupsize=$wg conflicts with the kernel's " *
                "static workgroupsize $static (baked in at `@kernel`-construction " *
                "time). Drop the kwarg or rebuild the kernel with the new size.")
        end
        return static
    end
    workgroupsize === nothing && return (16,)
    workgroupsize isa Integer && return (workgroupsize,)
    return workgroupsize
end

function (obj::KA.Kernel{MLIRBackend})(args...; ndrange=nothing,
                                                  workgroupsize=nothing)
    wg = _resolve_wgsize(obj, workgroupsize)
    nd = ndrange isa Integer ? (ndrange,) : ndrange
    nd === nothing && error("MLIRBackend: ndrange must be specified")

    # N-D model: the workgroup is flattened to a single `vector<prod(wg)>` lane
    # and the grid to a 1-D `scf.parallel` over `prod(nd)/prod(wg)` blocks; the
    # N-D `@index(…,NTuple)` markers reconstruct per-dim coords by column-major
    # unflatten. That reconstruction requires each ndrange dim to be an exact
    # multiple of the matching workgroup dim (no masked/partial blocks yet), and
    # the dimensionalities to agree.
    length(wg) == length(nd) || error(
        "MLIRBackend: ndrange $(nd) ($(length(nd))-D) and workgroupsize $(wg) " *
        "($(length(wg))-D) must have the same number of dimensions.")
    all(nd[d] % wg[d] == 0 for d in 1:length(wg)) || error(
        "MLIRBackend: ndrange=$nd not a per-dim multiple of workgroupsize=$wg " *
        "— masked launches not yet supported.")

    wg_dims = collect(Int, wg)
    nd_dims = collect(Int, nd)
    W = prod(wg_dims)          # flat workgroup (lane) width
    total = prod(nd_dims)      # total work-items

    _, _, iterspace, _ = KA.launch_config(obj, nd, wg)
    ctx = KA.mkcontext(obj, nd, iterspace)
    ctx_T = typeof(ctx)
    arg_types = map(typeof, args)
    full_argtypes = Tuple{ctx_T, arg_types...}

    k = ka_function(obj.f, full_argtypes;
                    lane_width=W, wg_dims=wg_dims, nd_dims=nd_dims,
                    kernel_name=string(nameof(obj.f), "_ka"))
    k(args...; blocks=total ÷ W)
    return nothing
end

end # module KernelAbstractionsExt
