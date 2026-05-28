module KernelAbstractionsExt

# Mirrors the CUDA.jl / AMDGPU.jl / oneAPI.jl / Metal.jl KA backend pattern:
#
#   1. `struct cuTileBackend <: KA.GPU` so KA's `@kernel` macro picks the
#      `gpu_*` (SIMT) function body, not the `cpu_*` (loop-splitting) one.
#
#   2. `@overlay cuTile.cuTileMethodTable` redefinitions of the KA intrinsics
#      consulted by inference inside `ct.emit_julia`. The overlay bodies
#      replace KA calls with expressions our walker already handles —
#      principally the sentinel `__cutilecpu_spmd_lane_id()` which a single
#      walker clause maps to the SPMD lane vector.
#
#   3. `(::Kernel{cuTileBackend})(args...; ndrange, workgroupsize)` builds the
#      KA `CompilerMetadata` type and calls `ka_function` → `lower_to_mlir_ka`
#      → in-process MLIR pipeline → clang → dlopen, then dispatches the grid
#      via the standard SPMD-style launch path.

using KernelAbstractions
const KA = KernelAbstractions
const NDI = KernelAbstractions.NDIteration
using cuTile
const ct = cuTile
using cuTileCPU

import Base.Experimental: @overlay

# ----------------------------------------------------------------------------
# 1. Backend
# ----------------------------------------------------------------------------

struct cuTileBackend <: KA.GPU end

# ----------------------------------------------------------------------------
# 2. Overlays into cuTile's method table
# ----------------------------------------------------------------------------
#
# Sentinel intrinsic. The cuTileCPU walker recognises calls to this (by
# name) in SPMD/KA mode and binds them to the lane value — the lane vector
# on the CPU path, or the scalar global thread index on the GPU path (see
# the `:__cutilecpu_spmd_lane_id` clause in `src/lower.jl`).
#
# It must be defined INSIDE `cuTile.Intrinsics`, with an opaque
# `compilerbarrier(:type, nothing)` body + a `cuTile.tfunc` return-type
# override — i.e. exactly what cuTile's `@intrinsic` macro generates. The
# `Intrinsics`-module membership is load-bearing: cuTile's interpreter
# only marks calls as `CC.NoCallInfo()` (keeping them as `Expr(:call)`,
# un-inlined) when `parentmodule(f) === Intrinsics` (see `isintrinsic` in
# cuTile/src/compiler/interpreter.jl). A function of the same shape defined
# anywhere else gets normal call info and is inlined away — its body
# leaks into the structured IR and the walker never sees a call to
# intercept.
#
# Creating a *new binding* inside `cuTile.Intrinsics` via `eval` is illegal
# during the extension's PRECOMPILE (Julia forbids mutating another
# module's bindings then). So we do it at runtime in `__init__`, where
# cross-module eval is allowed. (A cleaner long-term fix is to vendor
# cuTile's interpreter/intrinsic machinery into a shared layer so cuTileCPU
# owns intrinsic definition — design-doc step 4.) The `__index_Global_Linear`
# overlay references the intrinsic, so it too is installed in `__init__`.
function __init__()
    Core.eval(cuTile.Intrinsics, quote
        @noinline __cutilecpu_spmd_lane_id() = $(Base.compilerbarrier)(:type, nothing)
    end)
    @eval begin
        const __cutilecpu_spmd_lane_id = cuTile.Intrinsics.__cutilecpu_spmd_lane_id
        # Return-type override consulted by cuTileInterpreter (`tfunc(𝕃,f,…)`).
        cuTile.tfunc(𝕃, ::typeof(cuTile.Intrinsics.__cutilecpu_spmd_lane_id)) = Int32
        # `__index_Global_Linear(ctx)` → the global linear thread index.
        @overlay ct.cuTileMethodTable KA.__index_Global_Linear(ctx) =
            cuTile.Intrinsics.__cutilecpu_spmd_lane_id()
    end
    return
end

# `__validindex(ctx)` — for launches where ndrange is a multiple of the
# workgroup size, every lane is valid. Tighter (lane < ndrange) handling is
# a TODO; the kernel would need to thread `ndrange` through as a uniform
# scalar arg and we'd emit a vector compare + `vector.gather`/`scatter` with
# a per-lane mask. (No intrinsic reference → safe at precompile.)
@overlay ct.cuTileMethodTable KA.__validindex(ctx) = true

# `__synchronize()` — CPU SIMD has no warp barrier. For kernels without
# cross-lane communication this is a no-op. Kernels that depend on a real
# barrier (after `@localmem` + reduction patterns) need a different lowering
# strategy that this POC doesn't yet handle.
@overlay ct.cuTileMethodTable KA.__synchronize() = nothing

# `SharedMemory` / `Scratchpad` — not yet wired up.
@overlay ct.cuTileMethodTable KA.SharedMemory(::Type{T}, ::Val, ::Val) where {T} =
    error("cuTileBackend: @localmem / SharedMemory not yet implemented")
@overlay ct.cuTileMethodTable KA.Scratchpad(ctx, ::Type, ::Val) =
    error("cuTileBackend: @private / Scratchpad not yet implemented")

# ----------------------------------------------------------------------------
# 3. KA backend protocol
# ----------------------------------------------------------------------------
#
# We only implement the methods the kernel call needs. Allocation /
# synchronisation / copyto! all fall back to KA's default `<: KA.GPU`
# behaviour on the host (we're CPU-targeting, so host == device).

KA.allocate(::cuTileBackend, ::Type{T}, dims::Tuple) where {T} =
    cuTileCPU.aligned_array(T, dims; alignment=128)

KA.zeros(b::cuTileBackend, ::Type{T}, dims::Tuple) where {T} =
    fill!(KA.allocate(b, T, dims), zero(T))
KA.ones(b::cuTileBackend, ::Type{T}, dims::Tuple) where {T} =
    fill!(KA.allocate(b, T, dims), one(T))

KA.synchronize(::cuTileBackend) = nothing
KA.functional(::cuTileBackend) = true
KA.argconvert(::cuTileBackend, x) = x
# NOTE: deliberately no `KA.get_backend(::Array) = cuTileBackend()`. Plain
# `Array` already dispatches to KA's `CPU()` backend, and overriding here
# would silently steal every `Array`-touching KA call from the default
# (and fails precompile with method-overwriting anyway). Users select the
# backend explicitly: `vadd!(cuTileBackend(), 16)(C, A, B; ndrange=N)`.

# `mkcontext` and `launch_config` follow KA's GPU defaults (no per-backend
# specialisation needed) by reusing the generic methods. We provide a
# minimal stub so KA's `partition` can build the CompilerMetadata type.

KA.mkcontext(kernel::KA.Kernel{cuTileBackend}, ndrange, iterspace) =
    KA.CompilerMetadata{KA.ndrange(kernel), NDI.NoDynamicCheck}(ndrange, iterspace)

function KA.launch_config(kernel::KA.Kernel{cuTileBackend}, ndrange, workgroupsize)
    ndrange isa Integer && (ndrange = (ndrange,))
    workgroupsize isa Integer && (workgroupsize = (workgroupsize,))
    if KA.workgroupsize(kernel) <: KA.NDIteration.DynamicSize && workgroupsize === nothing
        workgroupsize = (16,)  # default lane width for cuTileCPU SPMD lowering
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
function _resolve_wgsize(obj::KA.Kernel{cuTileBackend}, workgroupsize)
    wg_T = KA.workgroupsize(obj)
    if wg_T <: NDI.StaticSize
        return NDI.get(wg_T)
    end
    workgroupsize === nothing && return (16,)
    workgroupsize isa Integer && return (workgroupsize,)
    return workgroupsize
end

function (obj::KA.Kernel{cuTileBackend})(args...; ndrange=nothing,
                                                  workgroupsize=nothing)
    wg = _resolve_wgsize(obj, workgroupsize)
    nd = ndrange isa Integer ? (ndrange,) : ndrange
    _, _, iterspace, _ = KA.launch_config(obj, nd, wg)

    ctx = KA.mkcontext(obj, nd, iterspace)
    ctx_T = typeof(ctx)
    arg_types = map(typeof, args)
    full_argtypes = Tuple{ctx_T, arg_types...}

    # Lane width = workgroup size. ndrange must be a multiple (the POC's
    # `__validindex == true` overlay assumes that).
    W = first(wg)
    total = prod(nd)
    total % W == 0 || error(
        "cuTileBackend: ndrange=$nd not a multiple of workgroupsize=$wg " *
        "— masked launches not yet supported")

    k = ka_function(obj.f, full_argtypes;
                    lane_width=W,
                    kernel_name=string(nameof(obj.f), "_ka"))
    k(args...; blocks=total ÷ W)
    return nothing
end

end # module KernelAbstractionsExt
