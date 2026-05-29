# Standalone Julia → StructuredIRCode frontend for the SPMD / KA / GPU paths.
#
# A minimal `AbstractInterpreter` with DEFAULT optimization params — crucially
# NOT inline-everything — so the `@noinline` marker intrinsics below survive
# inference for the walker to intercept (an inline-everything policy bulldozes
# `@noinline` and leaks the marker bodies into the IR). It carries its own
# `Intrinsics` module and an overlay `MethodTable` that frontends register
# intrinsic mappings into. Inference runs via `Base.code_ircode(f, argtypes;
# interp=…)`, and `IRStructurizer.StructuredIRCode` turns the result into an SCI.
# Depends only on Core.Compiler + IRStructurizer.
module Frontend

const CC = Core.Compiler
using IRStructurizer: StructuredIRCode

# ----------------------------------------------------------------------------
# Frontend intrinsics — markers the MLIRKernels walker recognises by name.
# ----------------------------------------------------------------------------
#
# Each is `@noinline` with a `compilerbarrier(:type, …)` body so that, under
# default optimization, the call SURVIVES inference (no inline, no
# const-fold) with a concrete return type for the walker to replace.
module Intrinsics
    using Base: compilerbarrier

    # Global linear thread index (1-based, Julia semantics). The walker binds
    # this to the SPMD lane vector (CPU) or `gpu.thread_id + block_id*block_dim`
    # (GPU SIMT).
    @noinline global_index() = compilerbarrier(:type, zero(Int32))::Int32

    # Block / workgroup index along a dimension (0-based). `dim` ∈ (0,1,2).
    @noinline block_index(dim::Int32) = compilerbarrier(:type, zero(Int32))::Int32

    # Block (workgroup) dimension along an axis.
    @noinline block_dim(dim::Int32) = compilerbarrier(:type, zero(Int32))::Int32

    # Local linear index within the workgroup (1-based). CPU = lane step+1; GPU = thread_id+1.
    @noinline local_index() = compilerbarrier(:type, zero(Int32))::Int32

    # Group (workgroup/block) linear index (1-based). CPU = bid+1 (uniform); GPU = block_id+1.
    @noinline group_index() = compilerbarrier(:type, zero(Int32))::Int32

    # Workgroup size (count). CPU = lane_width const; GPU = block_dim.
    @noinline group_size() = compilerbarrier(:type, zero(Int32))::Int32

    # Workgroup barrier (CPU: no-op; GPU: gpu.barrier). Returns nothing.
    # `Base.donotdelete` is essential: the barrier has no result and is otherwise
    # effect-free, so without it DCE deletes the call before the walker sees it
    # and `@synchronize` silently vanishes — fatal for cross-lane shared memory
    # (the reads race ahead of the writes). Same fix as `atomic_index!`.
    @noinline function barrier()
        Base.donotdelete(0)
        return compilerbarrier(:type, nothing)
    end

    # Per-thread validity for tail-block masking (KA's `if __validindex(ctx)`).
    # The walker lowers it to `∧_d (global_d < ndrange[d])` on GPU (padded last
    # block's out-of-range threads do nothing), `true` on CPU. `compilerbarrier`
    # keeps it from folding so the guarding `if` survives inference.
    @noinline valid_index() = compilerbarrier(:type, true)::Bool

    # Atomic read-modify-write at a 1-based linear index. The KA extension
    # overlays `Atomix.modify!(IndexableRef, op, x, ord)` — i.e. `KA.@atomic` /
    # `Atomix.@atomic`, KA's *portable* atomic — onto this marker, stopping the
    # default-opt inline cascade before it degrades to raw pointer arithmetic +
    # an `atomicrmw` llvmcall. The walker routes it to the `memref.atomic_rmw`
    # emitter. `op` is the reduction function (+/max/min/&/|), `idx` the 1-based
    # linear index.
    #
    # The `Base.donotdelete` is essential: the marker's result is discarded (the
    # atomic is used for its memory side effect, not its value), and
    # `compilerbarrier` is itself effect-free + nothrow, so without an effect the
    # marker is inferred effect-free and DCE deletes the whole call before the
    # walker ever sees it — the atomic silently vanishes. `donotdelete` makes the
    # method `!effect_free`, so the call is preserved for the walker to rewrite.
    @noinline function atomic_index!(arr, op, val, idx)
        Base.donotdelete(arr, val, idx)
        return compilerbarrier(:type, val)
    end

    # N-D workgroup indices (1-based), as an `NTuple{N,Int}`. The KA extension
    # overlays `__index_{Global,Local,Group}_NTuple(ctx)` onto these, reading the
    # grid dimensionality `N` from the ctx type. The walker reconstructs the per-
    # dim coordinate vectors (column-major unflatten of the flat lane/block) and
    # registers them as the tuple's components, so `i, j = @index(…, NTuple)`
    # binds `i`/`j` to the right per-lane vectors. Returning a concrete-arity
    # `NTuple{N,Int}` is what lets inference destructure the result.
    @noinline global_ntuple(::Val{N}) where {N} =
        compilerbarrier(:type, ntuple(_ -> zero(Int), Val(N)))::NTuple{N,Int}
    @noinline local_ntuple(::Val{N}) where {N} =
        compilerbarrier(:type, ntuple(_ -> zero(Int), Val(N)))::NTuple{N,Int}
    @noinline group_ntuple(::Val{N}) where {N} =
        compilerbarrier(:type, ntuple(_ -> zero(Int), Val(N)))::NTuple{N,Int}

    # Workgroup shared memory (`@localmem T dims`). The KA extension overlays
    # `SharedMemory(T, Val(dims), Val(id))` onto this marker; the walker emits a
    # workgroup-address-space `memref.alloca` of shape `dims` and routes
    # `shared[…]` accesses to it. Returns an `Array{T,N}` so indexing lowers via
    # the same memoryref path as a normal array arg. `Base.donotdelete` makes the
    # call `!effect_free` so it survives DCE and isn't CSE-merged across distinct
    # `@localmem` declarations (each must be its own buffer).
    @noinline function shared_alloc(::Type{T}, ::Val{Dims}) where {T, Dims}
        Base.donotdelete(T, Dims)
        return compilerbarrier(:type,
            Array{T, length(Dims)}(undef, Dims))::Array{T, length(Dims)}
    end

    # Per-thread private memory (`@private T dims`). KA overlays
    # `Scratchpad(ctx, T, Val(dims))` onto this. Same shape as `shared_alloc`,
    # but the walker emits a DEFAULT-address-space `memref.alloca` — per-thread
    # storage (each lane its own copy), no sharing, no barrier.
    @noinline function private_alloc(::Type{T}, ::Val{Dims}) where {T, Dims}
        Base.donotdelete(T, Dims)
        return compilerbarrier(:type,
            Array{T, length(Dims)}(undef, Dims))::Array{T, length(Dims)}
    end
end

# An intrinsic is a function defined in our Intrinsics module. (We don't
# currently need NoCallInfo because default opt params already respect
# `@noinline`; kept for future use.)
isintrinsic(@nospecialize(f)) = isa(f, Function) && parentmodule(f) === Intrinsics

# ----------------------------------------------------------------------------
# Overlay method table — frontends (KA, …) register intrinsic mappings here.
# ----------------------------------------------------------------------------

Base.Experimental.@MethodTable METHOD_TABLE

# ----------------------------------------------------------------------------
# Interpreter
# ----------------------------------------------------------------------------

struct FrontendInterpreter <: CC.AbstractInterpreter
    world::UInt
    method_table::CC.CachedMethodTable{CC.OverlayMethodTable}
    inf_cache::Vector{CC.InferenceResult}
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end

function FrontendInterpreter(world::UInt=Base.get_world_counter())
    mt = CC.CachedMethodTable(CC.OverlayMethodTable(world, METHOD_TABLE))
    # DEFAULT OptimizationParams — crucially NOT inline_cost_threshold=typemax,
    # so `@noinline` on our intrinsics is honoured and the marker calls survive.
    return FrontendInterpreter(world, mt, CC.InferenceResult[],
                               CC.InferenceParams(), CC.OptimizationParams())
end

CC.InferenceParams(i::FrontendInterpreter)     = i.inf_params
CC.OptimizationParams(i::FrontendInterpreter)  = i.opt_params
CC.get_inference_cache(i::FrontendInterpreter) = i.inf_cache
CC.method_table(i::FrontendInterpreter)        = i.method_table
# A custom (non-`nothing`) cache owner is REQUIRED for overlays to apply to
# Base/stdlib callees. With `nothing`, the interpreter reuses Julia's native
# (precompiled) CodeInstances — e.g. Base's range machinery already resolved
# `steprange_last` to the un-lowerable default, so our `@overlay` was bypassed.
# A private owner forces re-inference of reachable methods through our overlay
# method table.
CC.cache_owner(::FrontendInterpreter)          = :MLIRKernelsFrontend
@static if isdefined(CC, :get_inference_world)
    CC.get_inference_world(i::FrontendInterpreter) = i.world
else
    CC.get_world_counter(i::FrontendInterpreter) = i.world
end

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

"""
    structured(f, argtypes::Tuple) -> (sci::StructuredIRCode, rettype)

Infer `f(argtypes...)` under the Frontend interpreter (our overlays + own
intrinsics, default opt params) and structurize the resulting IRCode into a
`StructuredIRCode`.
"""
function structured(@nospecialize(f), @nospecialize(argtypes::Type))
    tt = Tuple(argtypes.parameters)
    interp = FrontendInterpreter()
    results = Base.code_ircode(f, tt; interp)
    isempty(results) && error("Frontend.structured: inference produced no results for $f$tt")
    ir, rettype = results[1]
    sci = StructuredIRCode(ir)
    return sci, CC.widenconst(rettype)
end

end # module Frontend
