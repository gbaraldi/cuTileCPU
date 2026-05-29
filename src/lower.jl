# StructuredIRCode → MLIR module.
#
# Pipeline:
#   Julia kernel + argtypes
#     → optimized StructuredIRCode
#         runs canonicalize → constprop → FMA fusion → CSE → alias
#         → token order → RNG lowering → LICM → divisibility/bounds
#         → no-wrap → DCE; we receive the optimized SCI.
#     → lower_to_mlir(sci, argtypes)                    [this file]
#         emits scf/arith/memref/vector/func dialect MLIR with alignment +
#         strided-layout info from ArraySpec.
#
# Argument model:
#   Each array arg becomes ONE `memref<?x...xT, strided<[…]>>` arg.
#   A flat (ptr, sizes…, strides…) destructuring is *not* done here — the
#   memref descriptor ABI carries that information.
#   `Constant{T,V}` args contribute no func parameter; values inline.
#
# Grid:
#   The body is wrapped in one `scf.parallel` over runtime grid dims passed
#   as trailing `index` args to the func. `@cuda blocks=N kernel(args)` →
#   one ccall per launch, MLIR's OpenMP lowering parallelises.

const _func   = Dialects.func
const _arith  = Dialects.arith
const _memref = Dialects.memref
const _scf    = Dialects.scf
const _vector = Dialects.vector
const _math   = Dialects.math
const _gpu    = Dialects.gpu

# ----------------------------------------------------------------------------
# Lowering context
# ----------------------------------------------------------------------------

# An "offset view": a source memref + per-lane index vector, produced when a
# varying (lane-vector) index reaches `memoryrefnew`. Tracked-only — consumed by
# the SPMD `memoryrefget`/`memoryrefset!` gather/scatter path. `base` is the
# source memref, `indices` the per-lane index vector, `elem_type` the source
# element type, and `idx_shape` the index tile shape (Julia col-major).
struct OffsetInfo
    base::IR.Value
    indices::IR.Value
    elem_type::Type
    idx_shape::Vector{Int}
end

mutable struct LowerCtx
    ctx::IR.Context
    mod::IR.Module
    # SSA / argument → MLIR Value
    ssa_vals::Dict{Int, IR.Value}
    arg_vals::Dict{Int, IR.Value}
    arg_const::Dict{Int, Any}
    # IRStructurizer BlockArguments (id → MLIR Value, e.g. iv / iter_args).
    block_args::Dict{Int, IR.Value}
    # Multi-result control-flow ops (IfOp / ForOp results): SSA → [Value...].
    ssa_multi::Dict{Int, Vector{IR.Value}}
    # Tracked, no IR emitted:
    offsets::Dict{Int, OffsetInfo}             # SSA → gather/scatter pointer-tile
    field_refs::Dict{Int, Tuple{Int, Symbol}}  # SSA → (arg id, fieldname)
    tuples::Dict{Int, Vector{Any}}             # SSA → component refs
    # SSAs that are sentinels (e.g. boundscheck) — extract to a marker.
    sentinels::Dict{Int, Symbol}
    # Per-arg element types — sci.argtypes[slot] for array slots.
    arg_elem_types::Dict{Int, Type}
    # Grid bid Values, one per grid dim, x/y/z order.
    bids::Vector{IR.Value}
    # ----- SPMD ("ISPC-style" scalar-typed kernels) mode -----
    # When true, the walker accepts plain Julia scalar code (Vector{T} args
    # + scalar lane index) and lifts each scalar op to a `lane_width`-wide
    # vector. See `lower_to_mlir_spmd` and the SPMD-aware dispatches in
    # `walk_call!`.
    spmd::Bool
    lane_width::Int
    # Arg slot of the lane index (e.g. the `i::Int` last param). The value
    # in `arg_vals[lane_arg]` is the per-iteration lane vector
    # `vector<lane_width × iX>` of values `[bid*W+1, bid*W+2, ..., (bid+1)*W]`.
    lane_arg::Int
    # MLIR i-type used for the lane index vector (matches the kernel's lane
    # arg Julia type — typically Int64).
    lane_idx_type::Type
    # ----- N-D KA grid (multi-dimensional workgroup/ndrange) -----
    # Workgroup dims and ndrange dims (Julia/column-major order). The workgroup
    # is still flattened to a single `vector<prod(wg_dims)>` lane and the grid
    # to a 1-D `scf.parallel`; these let the N-D `@index(…,NTuple/Cartesian)`
    # markers reconstruct per-dim coords by column-major unflatten. Empty / a
    # 1-tuple for the plain 1-D path.
    wg_dims::Vector{Int}
    nd_dims::Vector{Int}
    # ----- @localmem / SharedMemory -----
    # SSA id → (buffer memref, element type, static Julia dims). Keyed by both
    # the `shared_alloc`/`private_alloc` marker result and the
    # `getfield(buf, :ref)` SSA, so memoryrefnew/get/set/@atomic on a
    # `@localmem`/`@private` buffer route to it. The dims (Julia/column-major
    # order) let `size(buf, d)` resolve to the static constant `dims[d]`.
    local_memrefs::Dict{Int, Tuple{IR.Value, Type, Vector{Int}}}
    # The gpu.module body block (GPU path only). `@localmem` buffers are emitted
    # as workgroup-space `memref.global`s here (siblings of the gpu.func), which
    # gpu-to-nvvm lowers to real `.shared` — unlike `memref.alloca(workgroup)`,
    # which mis-lowers to a per-thread local depot. `nothing` on the CPU path.
    gpu_module_block::Union{Nothing, IR.Block}
    # ----- closure / functor kernel args -----
    # A closure arg (e.g. the `f`/`op` passed to map/reduce) is flattened: each
    # captured field that's an array or scalar becomes its own func param at a
    # synthetic (negative) arg slot, bound in arg_vals/arg_elem_types; singleton
    # captures (the user function) inline away. `captured` maps
    # (closure_slot, fieldname) → (synthetic_slot, :memref|:scalar). `array_ssa`
    # maps the SSA of a `getfield(closure, :array_field)` to its synthetic slot,
    # so the result reuses the array-arg indexing path (its :ref/:size resolve
    # to the synthetic slot's memref).
    # `captured` keys a flattened LEAF by (arg_slot, field_path) where field_path
    # is a Tuple of fieldnames/tuple-indices from the arg down to the leaf
    # (e.g. (:parent,) or (:indices, 1, :stop) for a SubArray). `flattened_slots`
    # is the set of arg slots that were flattened; `field_paths` maps the SSA of a
    # `getfield` reaching a NON-leaf (intermediate struct) to its (slot, path) so
    # the next getfield extends the path.
    captured::Dict{Tuple{Int, Tuple}, Tuple{Int, Symbol}}
    flattened_slots::Set{Int}
    field_paths::Dict{Int, Tuple{Int, Tuple}}
    array_ssa::Dict{Int, Int}
    # `Expr(:new, T, fields...)` struct construction (e.g. KA's `@Const` +
    # `inbounds=true` wraps a read-only array in `Base.Experimental.Const`):
    # SSA → (struct type, field operands). `getfield(new_ssa, :field)` extracts.
    new_structs::Dict{Int, Tuple{Any, Vector{Any}}}
    # SSA → an Argument/SSAValue it transparently aliases (e.g. the array a
    # `getfield(Const, :a)` yields). Resolved at the top of getfield/resolve.
    aliases::Dict{Int, Any}
end

LowerCtx(ctx, mod) = LowerCtx(
    ctx, mod,
    Dict{Int, IR.Value}(), Dict{Int, IR.Value}(), Dict{Int, Any}(),
    Dict{Int, IR.Value}(), Dict{Int, Vector{IR.Value}}(),
    Dict{Int, OffsetInfo}(),
    Dict{Int, Tuple{Int, Symbol}}(), Dict{Int, Vector{Any}}(),
    Dict{Int, Symbol}(),
    Dict{Int, Type}(),
    IR.Value[],
    false, 16, 0, Int64,
    Int[], Int[],
    Dict{Int, Tuple{IR.Value, Type, Vector{Int}}}(), nothing,
    Dict{Tuple{Int, Tuple}, Tuple{Int, Symbol}}(), Set{Int}(),
    Dict{Int, Tuple{Int, Tuple}}(), Dict{Int, Int}(),
    Dict{Int, Tuple{Any, Vector{Any}}}(), Dict{Int, Any}())

# ----------------------------------------------------------------------------
# Type translation: Julia → MLIR
# ----------------------------------------------------------------------------

function mlir_elem_type(T::Type)
    T === Float32  && return IR.Type(Float32)
    T === Float64  && return IR.Type(Float64)
    T === Float16  && return IR.Type(Float16)
    T === BFloat16 && return IR.Type(BFloat16)
    T === Int8     && return IR.Type(Int8)
    T === Int16    && return IR.Type(Int16)
    T === Int32    && return IR.Type(Int32)
    T === Int64    && return IR.Type(Int64)
    # MLIR uses signless integer types: unsigned ints map to the same iN as
    # their signed counterparts (signedness is a per-op attribute).
    T === UInt8    && return IR.Type(Int8)
    T === UInt16   && return IR.Type(Int16)
    T === UInt32   && return IR.Type(Int32)
    T === UInt64   && return IR.Type(Int64)
    T === Bool     && return IR.Type(Bool)
    error("MLIRKernels: unsupported element type $T")
end

# Build a splat DenseElements attribute for a vector type. The MLIR wrapper
# exposes `Base.fill(::T, shaped_type)` overloads only for {Bool, Int8/32/64,
# UInt8/32/64, Float32, Float64}. For Float16 / BFloat16 we go through the
# generic `Base.fill(attr::Attribute, shaped_type)` path which calls
# `mlirDenseElementsAttrSplatGet` under the hood.
function _splat_attr(value::T, vec_t) where {T<:Number}
    if T <: Union{Float16, BFloat16}
        return Base.fill(IR.Attribute(value), vec_t)
    end
    return Base.fill(value, vec_t)
end

# Emit a `vector<N × elem_T>` of values `[0, 1, ..., N-1]`. On MLIR ≥ 19
# we use the native `vector.step` op + `arith.index_cast` to the target
# integer type. On MLIR 18 (no `vector.step`) we fall back to
# `arith.constant dense<[0..N-1]>` parsed via text — the typed
# `DenseElementsAttribute(::Vector{Int64})` overload builds a TensorType
# shape that can't be retargeted at a VectorType, and the generic
# `DenseElementsAttribute(shaped, ::AbstractArray)` overload reinterprets
# the array as raw `MlirAttribute*` (→ segfault).
function _emit_step_vec(vec_t, elem_T::Type, N::Int)
    if MLIR.MLIR_VERSION[] ≥ v"19"
        idx_t = IR.IndexType()
        idx_vec_t = IR.VectorType(1, Int[N], idx_t)
        step_v = IR.result(_vector.step(; result=idx_vec_t))
        return idx_vec_t == vec_t ? step_v :
               IR.result(_arith.index_cast(step_v; out=vec_t))
    end
    elem_str = elem_T === Int32  ? "i32" :
               elem_T === Int64  ? "i64" :
               elem_T === UInt32 ? "i32" :
               elem_T === UInt64 ? "i64" :
               error("_emit_step_vec: unsupported element type $elem_T")
    arr_str = "[" * join(0:(N-1), ", ") * "]"
    attr = parse(IR.Attribute, "dense<$arr_str> : vector<$(N)x$(elem_str)>")
    return IR.result(_arith.constant(; value=attr, result=vec_t))
end

# MLIR vector type from a Julia (col-major) tile shape + Julia element
# type. If the shape is empty (`()` — a 0-D / scalar tile), returns the scalar
# elem MLIR type instead of `vector<f32>` (which isn't a valid MLIR type).
function mlir_tile_type(shape::Tuple, T::Type)
    isempty(shape) && return mlir_elem_type(T)
    elem = mlir_elem_type(T)
    return IR.VectorType(length(shape), reverse(collect(Int, shape)), elem)
end
mlir_tile_type(shape::AbstractVector{<:Integer}, T::Type) =
    mlir_tile_type(Tuple(shape), T)


# ----------------------------------------------------------------------------
# Divisibility annotations on array kernel args
# ----------------------------------------------------------------------------
#
# At func entry we already emit `memref.assume_alignment %arg, N` (covers the
# base pointer / leading dim). The non-leading dims have dynamic (`?`) strides
# in the memref's `strided<…>` layout — MLIR's strided-layout attribute can't
# carry divisibility info, so the alignment proof for vector loads / stores
# along those dims can't come from the type. We supply it by emitting
# `llvm.intr.assume((stride % n) == 0)` on each stride, which the
# `MemorySSA`-aware passes downstream of `mlir-translate --mlir-to-llvmir`
# fold into the vectorizer alignment fact.
#
# The divisibility info comes from the spec-only kernel-arg chain (an upper
# bound on what any consumer would derive); the dataflow results are queried
# at consumer sites, not at entry.



# ----------------------------------------------------------------------------
# Top-level entry: lower one SCI to an MLIR module
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# SPMD mode: lower a scalar-typed Julia kernel to vector MLIR
# ----------------------------------------------------------------------------
#
# Compiles a Julia function of shape
#
#     function k(arr1::Vector{T1}, ..., arrK::Vector{TK}, i::Int)
#         @inbounds arr1[i] = arr2[i] + arr3[i]   # plain Julia, scalar code
#         return
#     end
#
# into vector MLIR, where each grid block processes `lane_width` consecutive
# values of `i` simultaneously. ISPC-style SPMD-on-SIMD: the trailing `i::Int`
# arg is treated as *varying* — at codegen time it's a `vector<lane_width × iX>`
# of values `[bid*W+1, ..., (bid+1)*W]`. Plain Julia ops on varying values
# become vector ops; uniform scalars are broadcast to `lane_width` lanes when
# combined with varying ones.
#
# Implementation:
#   • Vector{T} args lower to plain `memref<?xT>` (no strided<[1]> layout,
#     no ArraySpec). The user passes plain `Vector{T}` host buffers — no
#     alignment requirement beyond what Julia gives by default.
#   • The lane arg's MLIR Value is rebound at each `scf.parallel` iteration
#     to `splat(bid * W) + (1, 2, ..., W)`.
#   • Bounds-check IfOps (`if boundscheck …`) are dropped — we assume the
#     user wrote `@inbounds` (or are okay with elision).
#   • `Base.memoryrefnew(memref_field, i, bc)` builds an OffsetInfo when `i`
#     is the lane vector; `Base.memoryrefget/set!` then lower to
#     `vector.gather`/`vector.scatter`.
#   • Vary/uniform: a value is "varying" iff its MLIR type is a vector.
#     Scalar/uniform-only ops stay scalar (cheaper); when any operand is
#     varying, scalars are broadcast to the lane vector type before the op.
#
# MVP limitation: requires `n % lane_width == 0`. Boundary handling via mask
# is left as a TODO — the kernel signature `(arr..., i::Int)` doesn't carry
# `n`, so any masking has to come from a separate kernel signature with a
# length scalar.
function lower_to_mlir_spmd(sci::StructuredIRCode, argtypes::Type;
                            kernel_name::String, lane_width::Int=16,
                            alignment::Int=16)
    ctx = fresh_context()
    mod_ref = Ref{IR.Module}()
    param_julia_types = Type[]
    param_kinds = Symbol[]
    n_grid_dims = 1

    @with_context ctx begin
        IR.load_all_available_dialects()
        mod = IR.Module(IR.Location())
        mod_ref[] = mod

        @with_module mod begin
            lc = LowerCtx(ctx, mod)
            lc.spmd = true
            lc.lane_width = lane_width

            idx_t = IR.IndexType()
            param_mlir_types = IR.Type[]
            param_arg_slots = Int[]

            # Identify the lane arg: the trailing Integer arg (after dropping
            # Const-seeded slots). Everything else is array (Vector{T}) or
            # uniform scalar.
            lane_slot = 0
            for (i, AT) in enumerate(sci.argtypes)
                i == 1 && continue
                AT_wide = widenconst(AT)
                if AT_wide <: Integer
                    lane_slot = i
                end
            end
            lane_slot == 0 && error(
                "lower_to_mlir_spmd: kernel signature must end with a scalar " *
                "lane index (e.g. `i::Int`); no Integer arg found in $sci.argtypes")
            lc.lane_arg = lane_slot

            for (i, AT) in enumerate(sci.argtypes)
                i == 1 && continue
                if AT isa Core.Const
                    lc.arg_const[i] = AT.val
                    continue
                end
                AT_wide = widenconst(AT)
                if AT_wide <: AbstractArray
                    eT = eltype(AT_wide)
                    N  = ndims(AT_wide)
                    elem = mlir_elem_type(eT)
                    shape = fill(Int(IR.dynsize()), N)
                    # Layout: with alignment > 16 we also encode a
                    # `strided<[1, ?, …]>` layout (unit stride on the
                    # fastest dim). This + `memref.assume_alignment` is
                    # what gets aligned vector loads at DRAM scale.
                    layout = if alignment > 16
                        strides_mlir = Vector{String}(undef, N)
                        for k in 1:N
                            mlir_dim = N - k + 1
                            strides_mlir[mlir_dim] = (k == 1) ? "1" : "?"
                        end
                        parse(IR.Attribute, "strided<[" * join(strides_mlir, ", ") * "]>")
                    else
                        IR.Attribute()
                    end
                    mr = IR.MemRefType(elem, shape, layout, IR.Attribute())
                    push!(param_mlir_types, mr)
                    push!(param_arg_slots, i)
                    push!(param_julia_types, AT_wide)
                    push!(param_kinds, :memref)
                    lc.arg_elem_types[i] = eT
                elseif AT_wide <: Number
                    if i == lane_slot
                        # Lane arg: don't materialise it as a func parameter
                        # here. It's bound to a per-iteration vector inside
                        # the scf.parallel body below.
                        lc.lane_idx_type = AT_wide
                    else
                        # Uniform scalar arg (e.g. a length `n`).
                        push!(param_mlir_types, mlir_elem_type(AT_wide))
                        push!(param_arg_slots, i)
                        push!(param_julia_types, AT_wide)
                        push!(param_kinds, :scalar)
                    end
                else
                    error("MLIRKernels SPMD: unsupported arg type $AT_wide at slot $i")
                end
            end

            # Grid dim (a single `index` arg: nblocks).
            grid_param_offset = length(param_mlir_types)
            push!(param_mlir_types, idx_t)

            ftype = IR.FunctionType(param_mlir_types, IR.Type[])
            arg_locs = [IR.Location() for _ in 1:length(param_mlir_types)]
            entry = IR.Block(param_mlir_types, arg_locs)
            body_region = IR.Region()
            push!(body_region, entry)

            funcop = _func.func_(;
                sym_name      = IR.Attribute(kernel_name),
                function_type = IR.Attribute(ftype),
                body          = body_region,
            )
            IR.setattr!(funcop, "llvm.emit_c_interface", IR.UnitAttribute())
            push!(IR.body(mod), funcop)

            # Raw entry-block arg Values, pre-assume.
            raw_arg_vals = Dict{Int, IR.Value}()
            for (k, slot) in enumerate(param_arg_slots)
                raw_arg_vals[slot] = IR.argument(entry, k)
                lc.arg_vals[slot] = raw_arg_vals[slot]
            end
            grid_val = IR.argument(entry, grid_param_offset + 1)

            @with_block entry begin
                c0 = IR.result(_arith.constant(; value=IR.Attribute(Int(0), idx_t)))
                c1 = IR.result(_arith.constant(; value=IR.Attribute(Int(1), idx_t)))

                # Alignment hints: when alignment > 16, emit a
                # `memref.assume_alignment %arg, N` per memref arg. Lowers
                # to `llvm.assume` on the base pointer so LLVM emits aligned
                # vector load/stores. Skipped at alignment=16 (Julia GC's
                # default) to keep the IR clean. MLIR 18 form is result-
                # less; we keep the raw arg Value as the downstream binding.
                if alignment > 16
                    align_attr = IR.Attribute(alignment, IR.Type(Int32))
                    for (k, slot) in enumerate(param_arg_slots)
                        param_kinds[k] === :memref || continue
                        _memref.assume_alignment(
                            raw_arg_vals[slot]; alignment=align_attr,
                        )
                    end
                end

                par_region = IR.Region()
                par_block = IR.Block([idx_t], [IR.Location()])
                push!(par_region, par_block)
                _scf.parallel(
                    IR.Value[c0],
                    IR.Value[grid_val],
                    IR.Value[c1],
                    IR.Value[];
                    results=IR.Type[], region=par_region,
                )

                @with_block par_block begin
                    bid = IR.argument(par_block, 1)
                    push!(lc.bids, bid)
                    # Build the lane vector for this grid step:
                    #   lane_base = (bid * lane_width)              (index)
                    #   lane_idx  = splat(lane_base) + (1, 2, ..., W)  -- as iX
                    # Julia's user-facing `i` is 1-based; the kernel uses
                    # `c[i]` so we add 1 to get the 1-based lane indices.
                    W_const = IR.result(_arith.constant(;
                        value=IR.Attribute(Int(lane_width), idx_t)))
                    lane_base_idx = IR.result(_arith.muli(bid, W_const; result=idx_t))
                    lane_t = mlir_elem_type(lc.lane_idx_type)
                    # Build (0, 1, ..., W-1) as `vector<W × index>` then cast
                    # to the lane idx integer type, then add splat(lane_base+1).
                    # `vector.step` arrived in MLIR 19; on MLIR 18 we emit the
                    # equivalent `arith.constant dense<0..W-1>` directly in the
                    # target integer type, skipping the index→iN cast.
                    int_vec_t = IR.VectorType(1, Int[lane_width], lane_t)
                    step_int = _emit_step_vec(int_vec_t, lc.lane_idx_type, lane_width)
                    # base + 1 as scalar (1-based)
                    one_idx = IR.result(_arith.constant(;
                        value=IR.Attribute(Int(1), idx_t)))
                    base_p1_idx = IR.result(_arith.addi(lane_base_idx, one_idx;
                                                       result=idx_t))
                    base_p1_int = IR.result(_arith.index_cast(base_p1_idx; out=lane_t))
                    base_splat = IR.result(_vector.broadcast(base_p1_int;
                                                              vector=int_vec_t))
                    lane_vec = IR.result(_arith.addi(base_splat, step_int;
                                                    result=int_vec_t))
                    lc.arg_vals[lane_slot] = lane_vec

                    walk_block!(lc, sci.entry)
                    _scf.reduce(IR.Value[]; reductions=IR.Region[])
                end

                _func.return_(IR.Value[])
            end
        end
    end
    return (mod_ref[], param_julia_types, ctx, param_kinds)
end

# ----------------------------------------------------------------------------
# KernelAbstractions-style entrypoint
# ----------------------------------------------------------------------------
#
# Variant of `lower_to_mlir_spmd` for the KernelAbstractions kernel shape:
#
#     function gpu_foo(__ctx__::CompilerMetadata, A, B, C)
#         @active_lane = KA.__validindex(__ctx__)              # overlay → true
#         if @active_lane
#             i = KA.__index_Global_Linear(__ctx__)            # overlay →
#                                                              # global_index()
#             @inbounds C[i] = A[i] + B[i]
#         end
#     end
#
# The first arg is the KA `__ctx__` (a `CompilerMetadata{…}` struct). With the
# overlay set up in `ext/KernelAbstractionsExt.jl`, the body never reads any
# field of `__ctx__` — every reference to it has been folded to either a
# constant or a call to the sentinel function `global_index()`.
# We therefore don't materialise the ctx as an MLIR parameter at all; we just
# use its arg slot as the SPMD `lane_arg` so the existing walker
# infrastructure (the lane vector, lane-index sentinel clause) lights
# up unchanged.
function lower_to_mlir_ka(sci::StructuredIRCode, argtypes::Type;
                          kernel_name::String, lane_width::Int=16,
                          alignment::Int=16, lane_idx_type::Type=Int64,
                          wg_dims::Vector{Int}=Int[lane_width],
                          nd_dims::Vector{Int}=Int[lane_width])
    ctx = fresh_context()
    mod_ref = Ref{IR.Module}()
    param_julia_types = Type[]
    param_kinds = Symbol[]

    @with_context ctx begin
        IR.load_all_available_dialects()
        mod = IR.Module(IR.Location())
        mod_ref[] = mod

        @with_module mod begin
            lc = LowerCtx(ctx, mod)
            lc.spmd = true
            lc.lane_width = lane_width
            lc.lane_idx_type = lane_idx_type
            lc.wg_dims = wg_dims
            lc.nd_dims = nd_dims

            idx_t = IR.IndexType()
            param_mlir_types = IR.Type[]
            param_arg_slots = Int[]

            # Lane slot = first non-function, non-Const arg = the KA ctx.
            lane_slot = 0
            for (i, AT) in enumerate(sci.argtypes)
                i == 1 && continue
                AT isa Core.Const && continue
                lane_slot = i
                break
            end
            lane_slot == 0 && error(
                "lower_to_mlir_ka: kernel has no non-Const args; expected a " *
                "KernelAbstractions.CompilerMetadata ctx as the first arg")
            lc.lane_arg = lane_slot

            for (i, AT) in enumerate(sci.argtypes)
                i == 1 && continue
                i == lane_slot && continue
                if AT isa Core.Const
                    lc.arg_const[i] = AT.val
                    continue
                end
                AT_wide = widenconst(AT)
                if AT_wide <: AbstractArray
                    eT = eltype(AT_wide)
                    N  = ndims(AT_wide)
                    elem = mlir_elem_type(eT)
                    shape = fill(Int(IR.dynsize()), N)
                    layout = if alignment > 16
                        strides_mlir = Vector{String}(undef, N)
                        for k in 1:N
                            mlir_dim = N - k + 1
                            strides_mlir[mlir_dim] = (k == 1) ? "1" : "?"
                        end
                        parse(IR.Attribute, "strided<[" * join(strides_mlir, ", ") * "]>")
                    else
                        IR.Attribute()
                    end
                    mr = IR.MemRefType(elem, shape, layout, IR.Attribute())
                    push!(param_mlir_types, mr)
                    push!(param_arg_slots, i)
                    push!(param_julia_types, AT_wide)
                    push!(param_kinds, :memref)
                    lc.arg_elem_types[i] = eT
                elseif AT_wide <: Number
                    push!(param_mlir_types, mlir_elem_type(AT_wide))
                    push!(param_arg_slots, i)
                    push!(param_julia_types, AT_wide)
                    push!(param_kinds, :scalar)
                else
                    error("MLIRKernels KA: unsupported arg type $AT_wide at slot $i " *
                          "(only AbstractArray + Number args are wired up; the " *
                          "first non-Const arg is consumed as the KA ctx)")
                end
            end

            # Grid dim (a single `index` arg: nblocks).
            grid_param_offset = length(param_mlir_types)
            push!(param_mlir_types, idx_t)

            ftype = IR.FunctionType(param_mlir_types, IR.Type[])
            arg_locs = [IR.Location() for _ in 1:length(param_mlir_types)]
            entry = IR.Block(param_mlir_types, arg_locs)
            body_region = IR.Region()
            push!(body_region, entry)

            funcop = _func.func_(;
                sym_name      = IR.Attribute(kernel_name),
                function_type = IR.Attribute(ftype),
                body          = body_region,
            )
            IR.setattr!(funcop, "llvm.emit_c_interface", IR.UnitAttribute())
            push!(IR.body(mod), funcop)

            raw_arg_vals = Dict{Int, IR.Value}()
            for (k, slot) in enumerate(param_arg_slots)
                raw_arg_vals[slot] = IR.argument(entry, k)
                lc.arg_vals[slot] = raw_arg_vals[slot]
            end
            grid_val = IR.argument(entry, grid_param_offset + 1)

            @with_block entry begin
                c0 = IR.result(_arith.constant(; value=IR.Attribute(Int(0), idx_t)))
                c1 = IR.result(_arith.constant(; value=IR.Attribute(Int(1), idx_t)))

                if alignment > 16
                    align_attr = IR.Attribute(alignment, IR.Type(Int32))
                    for (k, slot) in enumerate(param_arg_slots)
                        param_kinds[k] === :memref || continue
                        _memref.assume_alignment(
                            raw_arg_vals[slot]; alignment=align_attr,
                        )
                    end
                end

                par_region = IR.Region()
                par_block = IR.Block([idx_t], [IR.Location()])
                push!(par_region, par_block)
                _scf.parallel(
                    IR.Value[c0],
                    IR.Value[grid_val],
                    IR.Value[c1],
                    IR.Value[];
                    results=IR.Type[], region=par_region,
                )

                @with_block par_block begin
                    bid = IR.argument(par_block, 1)
                    push!(lc.bids, bid)
                    W_const = IR.result(_arith.constant(;
                        value=IR.Attribute(Int(lane_width), idx_t)))
                    lane_base_idx = IR.result(_arith.muli(bid, W_const; result=idx_t))
                    lane_t = mlir_elem_type(lc.lane_idx_type)
                    int_vec_t = IR.VectorType(1, Int[lane_width], lane_t)
                    step_int = _emit_step_vec(int_vec_t, lc.lane_idx_type, lane_width)
                    one_idx = IR.result(_arith.constant(;
                        value=IR.Attribute(Int(1), idx_t)))
                    base_p1_idx = IR.result(_arith.addi(lane_base_idx, one_idx;
                                                       result=idx_t))
                    base_p1_int = IR.result(_arith.index_cast(base_p1_idx; out=lane_t))
                    base_splat = IR.result(_vector.broadcast(base_p1_int;
                                                              vector=int_vec_t))
                    lane_vec = IR.result(_arith.addi(base_splat, step_int;
                                                    result=int_vec_t))
                    lc.arg_vals[lane_slot] = lane_vec

                    walk_block!(lc, sci.entry)
                    _scf.reduce(IR.Value[]; reductions=IR.Region[])
                end

                _func.return_(IR.Value[])
            end
        end
    end
    return (mod_ref[], param_julia_types, ctx, param_kinds)
end

# ----------------------------------------------------------------------------
# GPU SIMT entrypoint (gpu dialect → NVVM/ROCDL)
# ----------------------------------------------------------------------------
#
# Emits a `gpu.module { gpu.func @k kernel { … } }` for a SIMT kernel of
# shape
#
#     function k(arr1, ..., arrK, n::Integer, gid::Integer)
#         if gid <= n
#             @inbounds arr1[gid] = arr2[gid] + arr3[gid]
#         end
#         return
#     end
#
# where the trailing `gid` is the *global thread index* (1-based, Julia
# semantics). Unlike the CPU SPMD path which makes the lane a
# `vector<W × iX>` and wraps the body in `scf.parallel`, here:
#
#   • The kernel body IS the per-thread body — no scf.parallel, no grid
#     loop. The host launches `block × grid` threads via gpu.launch_func
#     (or, in the experiments, cudacall after lowering to PTX).
#   • `gid` is bound to a *scalar* index:
#         gid = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x + 1
#     (the +1 makes it 1-based to match Julia indexing; memoryrefnew
#     subtracts 1 to get the 0-based memref index).
#   • Array args become `memref<?xT, #gpu.address_space<global>>` so the
#     pointers are global-qualified and no `cvta.to.global` is emitted
#     (see experiment 04).
#   • Because `gid` is scalar, the SHARED SPMD walker clauses
#     (`emit_spmd_memoryrefnew/get/set!`) take their `isempty(idx_shape)`
#     scalar branches → `memref.load`/`memref.store`. One thread = one
#     element. `_spmd_harmonise` is a no-op on all-scalar operands.
#   • The `if gid <= n` guard is a real `arith.cmpi` (not the `:boundscheck`
#     sentinel), so `emit_if!` emits a normal `scf.if` — kept, as GPU
#     needs it for the tail block.
#
# `gid` is still tracked via `lc.lane_arg`, and the lane-index sentinel
# clause returns `lc.arg_vals[lane_arg]` unchanged — so a KA
# `__index_Global_Linear` overlay routed through this
# entrypoint also works (with `__validindex` providing the `gid <= n`
# guard or `true` for exact-multiple launches).
# `ctx_arg`: when set, the lane comes from a *non*-trailing arg — the slot
# is treated like the KA `__ctx__` (skipped as a func param, its value is
# the synthesized global thread index via the lane-index
# sentinel). This is how a KernelAbstractions `gpu_*` body lowers: the
# `__index_Global_Linear(ctx)` overlay rewrites to the sentinel, and the
# ctx itself is never referenced as data. When `ctx_arg === nothing` we
# use the plain-Julia shape (trailing Integer `gid`).
# Recursively flatten a struct/closure/wrapped-array arg into GPU func params:
# a DenseArray field → memref param, a Number field → scalar param, a singleton
# field (e.g. a captured user function) → skipped, a nested struct/tuple field →
# recurse. Each leaf gets a synthetic (negative) slot, bound in
# arg_vals/arg_elem_types. Only the arg's DIRECT fields (`is_top`) are recorded
# in `lc.captured`, so the walker can resolve `getfield(arg, :field)`; nested
# leaves get params (for marshalling consistency) but aren't directly addressed.
# Returns the updated synthetic-slot counter.
function _flatten_struct_arg!(lc::LowerCtx, @nospecialize(T), arg_slot::Int,
                              path::Tuple, syn::Int, param_mlir_types,
                              param_arg_slots, param_julia_types, param_kinds,
                              global_attr)
    push!(lc.flattened_slots, arg_slot)
    for fn in fieldnames(T)
        FT = fieldtype(T, fn)
        Base.issingletontype(FT) && continue
        leaf = (path..., fn)
        if FT <: DenseArray
            syn -= 1
            mrc = IR.MemRefType(mlir_elem_type(eltype(FT)),
                                fill(Int(IR.dynsize()), ndims(FT)),
                                IR.Attribute(), global_attr)
            push!(param_mlir_types, mrc); push!(param_arg_slots, syn)
            push!(param_julia_types, FT); push!(param_kinds, :memref)
            lc.arg_elem_types[syn] = eltype(FT)
            lc.captured[(arg_slot, leaf)] = (syn, :memref)
        elseif FT <: Number
            syn -= 1
            push!(param_mlir_types, mlir_elem_type(FT)); push!(param_arg_slots, syn)
            push!(param_julia_types, FT); push!(param_kinds, :scalar)
            lc.captured[(arg_slot, leaf)] = (syn, :scalar)
        elseif isstructtype(FT) && !isempty(fieldnames(FT))
            syn = _flatten_struct_arg!(lc, FT, arg_slot, leaf, syn,
                       param_mlir_types, param_arg_slots, param_julia_types,
                       param_kinds, global_attr)
        else
            error("MLIRKernels GPU: capture field $fn::$FT (slot $arg_slot) unsupported")
        end
    end
    return syn
end

function lower_to_mlir_gpu(sci::StructuredIRCode, argtypes::Type;
                           kernel_name::String, module_name::String="kernels",
                           lane_idx_type::Type=Int32, nd_dims::Vector{Int}=Int[],
                           ctx_arg::Union{Nothing,Int}=nothing)
    ctx = fresh_context()
    mod_ref = Ref{IR.Module}()
    param_julia_types = Type[]
    param_kinds = Symbol[]

    @with_context ctx begin
        IR.load_all_available_dialects()
        mod = IR.Module(IR.Location())
        mod_ref[] = mod
        # `gpu.container_module` marks the top-level module as holding
        # gpu.modules — required by the gpu→nvvm pipeline.
        IR.setattr!(IR.Operation(mod), "gpu.container_module", IR.UnitAttribute())

        @with_module mod begin
            lc = LowerCtx(ctx, mod)
            lc.spmd = true                 # reuse the scalar-index SPMD clauses
            lc.lane_width = 1
            lc.lane_idx_type = lane_idx_type
            lc.nd_dims = nd_dims           # ndrange sizes, for valid_index masking

            idx_t = IR.IndexType()
            global_attr = parse(IR.Attribute, "#gpu.address_space<global>")
            lane_t = mlir_elem_type(lane_idx_type)

            # Identify the lane arg. KA mode: the caller-supplied `ctx_arg`
            # slot (the `__ctx__`). Plain mode: the trailing Integer (`gid`).
            lane_slot = 0
            if ctx_arg !== nothing
                lane_slot = ctx_arg
            else
                for (i, AT) in enumerate(sci.argtypes)
                    i == 1 && continue
                    widenconst(AT) <: Integer && (lane_slot = i)
                end
            end
            lane_slot == 0 && error(
                "lower_to_mlir_gpu: kernel must end with a scalar global-index " *
                "arg (e.g. `gid::Int32`), or pass `ctx_arg` for the KA shape")
            lc.lane_arg = lane_slot

            param_mlir_types = IR.Type[]
            param_arg_slots = Int[]
            syn_counter = 0    # synthetic (negative) slots for closure captures
            for (i, AT) in enumerate(sci.argtypes)
                i == 1 && continue
                # KA mode: the ctx slot is the lane and carries no data —
                # skip it as a param. (In plain mode the lane is a trailing
                # Integer, handled in the `Number` branch below.)
                if ctx_arg !== nothing && i == lane_slot
                    continue
                end
                if AT isa Core.Const
                    lc.arg_const[i] = AT.val
                    continue
                end
                AT_wide = widenconst(AT)
                # `DenseArray` (Array, CuArray, …) is memory-backed → memref.
                # Lazy AbstractArrays (ranges like `OneTo`/`UnitRange`, the
                # `eachindex(A)` a kernel iterates) are NOT DenseArrays — they
                # fall to the struct-flatten branch below, where `length`/`[i]`
                # resolve from their captured fields (e.g. OneTo's `.stop`).
                if AT_wide <: DenseArray
                    eT = eltype(AT_wide)
                    N  = ndims(AT_wide)
                    shape = fill(Int(IR.dynsize()), N)
                    mr = IR.MemRefType(mlir_elem_type(eT), shape,
                                       IR.Attribute(), global_attr)
                    push!(param_mlir_types, mr)
                    push!(param_arg_slots, i)
                    push!(param_julia_types, AT_wide)
                    push!(param_kinds, :memref)
                    lc.arg_elem_types[i] = eT
                elseif AT_wide <: Number
                    if i == lane_slot
                        lc.lane_idx_type = AT_wide
                    else
                        push!(param_mlir_types, mlir_elem_type(AT_wide))
                        push!(param_arg_slots, i)
                        push!(param_julia_types, AT_wide)
                        push!(param_kinds, :scalar)
                    end
                else
                    # Closure / functor / wrapped array (SubArray, range, …):
                    # recursively flatten captured fields into their own params.
                    (isstructtype(AT_wide) && !isempty(fieldnames(AT_wide))) ||
                        error("MLIRKernels GPU: unsupported arg type $AT_wide at slot $i")
                    syn_counter = _flatten_struct_arg!(lc, AT_wide, i, (),
                        syn_counter, param_mlir_types, param_arg_slots,
                        param_julia_types, param_kinds, global_attr)
                end
            end

            ftype = IR.FunctionType(param_mlir_types, IR.Type[])

            # gpu.module @<module_name> { ... }
            gmod_block = IR.Block(IR.Type[], IR.Location[])
            gmod_region = IR.Region()
            push!(gmod_region, gmod_block)
            gmodop = _gpu.module_(; sym_name=IR.Attribute(module_name),
                                  bodyRegion=gmod_region)
            push!(IR.body(mod), gmodop)
            lc.gpu_module_block = gmod_block   # @localmem globals go here

            @with_block gmod_block begin
                # gpu.func @<kernel_name>(...) kernel { ... }
                arg_locs = [IR.Location() for _ in 1:length(param_mlir_types)]
                entry = IR.Block(param_mlir_types, arg_locs)
                fbody = IR.Region()
                push!(fbody, entry)
                # `_gpu.func` auto-appends to the active block (gmod_block).
                funcop = _gpu.func(; function_type=IR.Attribute(ftype), body=fbody)
                IR.setattr!(funcop, "sym_name", IR.Attribute(kernel_name))
                IR.setattr!(funcop, "gpu.kernel", IR.UnitAttribute())

                # Bind func params.
                for (k, slot) in enumerate(param_arg_slots)
                    lc.arg_vals[slot] = IR.argument(entry, k)
                end

                @with_block entry begin
                    # gid = thread_id.x + block_id.x * block_dim.x + 1
                    dimx = parse(IR.Attribute, "#gpu<dim x>")
                    tid  = IR.result(_gpu.thread_id(; result_0=idx_t, dimension=dimx))
                    bid  = IR.result(_gpu.block_id(; result_0=idx_t, dimension=dimx))
                    bdim = IR.result(_gpu.block_dim(; result_0=idx_t, dimension=dimx))
                    off  = IR.result(_arith.muli(bid, bdim; result=idx_t))
                    gid_idx = IR.result(_arith.addi(off, tid; result=idx_t))
                    one_idx = IR.result(_arith.constant(; value=IR.Attribute(Int(1), idx_t)))
                    gid1_idx = IR.result(_arith.addi(gid_idx, one_idx; result=idx_t))
                    # Cast to the lane integer type (e.g. i32) — matches the
                    # kernel's `gid::Int32` so the `gid <= n` cmpi is well-typed.
                    gid_int = IR.result(_arith.index_cast(gid1_idx; out=lane_t))
                    lc.arg_vals[lane_slot] = gid_int
                    push!(lc.bids, bid)

                    walk_block!(lc, sci.entry)
                    _gpu.return_(IR.Value[])
                end
            end
        end
    end
    return (mod_ref[], param_julia_types, ctx, param_kinds)
end

# ----------------------------------------------------------------------------
# Block / statement walking
# ----------------------------------------------------------------------------

# Walk a block's body. `kind` selects how the block's terminator is
# materialized:
#   :entry — outer entry block under scf.parallel. Terminator is dropped here;
#            the caller emits `scf.reduce` after we return.
#   :if    — `scf.if` then/else region. ReturnNode becomes `scf.yield` (no
#            operands) — see notes in the design spec. YieldOp becomes
#            `scf.yield <operands>`. Returning early stops emission.
#   :for   — `scf.for` body. ContinueOp becomes `scf.yield <new-iter-args>`.
# Returns the terminator's MLIR Value-list (or empty Vec) for callers that
# care (currently unused; we emit the terminator op inline).
function walk_block!(lc::LowerCtx, block::Block; kind::Symbol=:entry)
    for (idx, entry) in block
        stmt = entry.stmt
        typ  = entry.type
        if stmt isa Core.ReturnNode
            # Some structurization shapes still place ReturnNode in the body
            # rather than terminator (very rare). Treat as early return.
            if kind === :if
                _scf.yield(IR.Value[])
                return
            end
            continue
        end
        v = walk_stmt!(lc, idx, stmt, typ)
        v === nothing || (lc.ssa_vals[idx] = v)
    end
    # Terminator
    term = block.terminator
    if kind === :entry
        # caller will emit scf.reduce + func.return
        return
    elseif kind === :if
        if term isa Core.ReturnNode
            # Early return from the kernel via this branch. Inside scf.if we
            # can only emit scf.yield. Use no operands (the `scf.if` was
            # constructed with empty result types in this case).
            _scf.yield(IR.Value[])
            return
        elseif term isa YieldOp
            vals = IR.Value[]
            for v in term.values
                resolved = resolve_value_or_const(lc, v)
                resolved === nothing &&
                    error("scf.yield: cannot resolve operand $v")
                push!(vals, resolved)
            end
            _scf.yield(vals)
            return
        elseif term === nothing
            _scf.yield(IR.Value[])
            return
        else
            error("walk_block!(:if): unexpected terminator $(typeof(term))")
        end
    elseif kind === :for
        if term isa ContinueOp
            vals = IR.Value[]
            for v in term.values
                resolved = resolve_value_or_const(lc, v)
                resolved === nothing &&
                    error("scf.for continue: cannot resolve operand $v")
                push!(vals, resolved)
            end
            _scf.yield(vals)
            return
        elseif term === nothing
            _scf.yield(IR.Value[])
            return
        else
            error("walk_block!(:for): unexpected terminator $(typeof(term))")
        end
    end
end

function walk_stmt!(lc::LowerCtx, idx::Int, @nospecialize(stmt), @nospecialize(typ))
    if stmt isa Expr
        return walk_expr!(lc, idx, stmt, typ)
    elseif stmt isa Core.ReturnNode
        return nothing
    elseif stmt isa IfOp
        emit_if!(lc, idx, stmt, typ)
        return nothing
    elseif stmt isa ForOp
        emit_for!(lc, idx, stmt, typ)
        return nothing
    elseif stmt === :boundscheck || stmt === Symbol("boundscheck")
        # Synthetic bounds-check sentinel. We assume in-bounds; tag the SSA
        # so a subsequent Base.getfield(_, _, %14) call can recognise it.
        lc.sentinels[idx] = :boundscheck
        return nothing
    elseif stmt isa SSAValue
        # Naked SSA-to-SSA alias (`%a = %b`). Just forward.
        return get(lc.ssa_vals, stmt.id, nothing)
    elseif stmt === nothing
        # Literal `nothing` SSA — produced when an overlay'd void-returning
        # function (e.g. KernelAbstractions.__synchronize → nothing) is
        # inlined and its return value materialises as an SSA. Drop.
        return nothing
    elseif stmt isa Core.Compiler.PiNode || (isdefined(Core, :PiNode) && stmt isa Core.PiNode)
        # PiNode: type-refinement wrapper around a Value. Forward the value.
        inner = stmt.val
        if inner isa SSAValue
            return get(lc.ssa_vals, inner.id, nothing)
        elseif inner isa Argument
            return get(lc.arg_vals, inner.n, nothing)
        end
        return resolve_value_or_const(lc, inner)
    elseif stmt isa Number || stmt isa Bool
        # A bare literal bound to an SSA (e.g. `%6 = 4`) — arises from loop
        # transformations like `@simd`/`@unroll`. Materialise the constant so
        # downstream uses resolve.
        return resolve_value_or_const(lc, stmt)
    end
    error("MLIRKernels.walk_stmt!: unhandled stmt $stmt at %$idx (typ=$typ)")
end

function walk_expr!(lc::LowerCtx, idx::Int, e::Expr, @nospecialize(typ))
    if e.head === :call
        return walk_call!(lc, idx, e.args[1], e.args[2:end], typ)
    elseif e.head === :invoke
        return walk_call!(lc, idx, e.args[2], e.args[3:end], typ)
    elseif e.head === :boundscheck
        # Synthetic bool used as the 3rd `getfield` arg (`@inbounds` hint).
        # Tag the SSA as a sentinel; subsequent getfield call recognises it.
        lc.sentinels[idx] = :boundscheck
        return nothing
    elseif e.head === :aliasscope || e.head === :popaliasscope
        # `@Const`/`@inbounds` (KA's `constify`) wrap loads in alias-scope
        # markers (`Expr(:aliasscope)` ... `Expr(:popaliasscope)`). They carry
        # noalias metadata for LLVM but have no MLIR analogue here — the memref
        # ABI already encodes the (non-aliasing) argument buffers. Drop them.
        return nothing
    elseif e.head === :loopinfo
        # `@simd` / KA's `@unroll` (KernelAbstractions.Extras) annotate the loop
        # body with an `Expr(:loopinfo, "julia.simdloop"/"julia.unroll", …)`
        # hint. It's a pure optimisation directive (the loop is semantically a
        # plain scf.for); drop it and let LLVM/ptxas unroll/vectorise.
        return nothing
    elseif e.head === :new
        # Struct construction. No IR is emitted; we record the field operands so
        # a later `getfield(new_ssa, :field)` extracts them. The case that arises
        # in practice is `Base.Experimental.Const(arr)` (KA's `@Const` +
        # `inbounds=true`) — a transparent read-only-array wrapper.
        T = e.args[1] isa Type ? e.args[1] :
            something(resolve_const(lc, e.args[1]), nothing)
        lc.new_structs[idx] = (T, collect(Any, e.args[2:end]))
        return nothing
    elseif e.head === :throw_undef_if_not
        # `throw_undef_if_not(:name, cond)`: throw `UndefVarError` unless `cond`.
        # Tail-block masking leaves values computed under `if __validindex` undef
        # on the masked path, but a masked thread never reaches the use (its work
        # is guarded) and a GPU throw is just a trap. Elide, like bounds checks.
        return nothing
    end
    error("MLIRKernels.walk_expr!: unhandled Expr head :$(e.head) at %$idx ($e)")
end

# Resolve an operand to an MLIR Value (or `nothing` if it's tracked-only).
function resolve_value(lc::LowerCtx, @nospecialize(op))
    op isa SSAValue && return get(lc.ssa_vals, op.id, nothing)
    op isa Argument && return get(lc.arg_vals, op.n, nothing)
    op isa BlockArgument && return get(lc.block_args, op.id, nothing)
    return nothing
end

# Resolve a compile-time constant if available (Const-seeded arg, literal,
# or — TODO — a tracked ConstantOp).
function resolve_const(lc::LowerCtx, @nospecialize(op))
    op isa Argument && return get(lc.arg_const, op.n, nothing)
    op isa Number   && return op
    op isa Bool     && return op
    if op isa GlobalRef
        v = try
            getfield(op.mod, op.name)
        catch
            return nothing
        end
        return v
    end
    if op isa QuoteNode
        # Symbol/Type literal embedded via a QuoteNode.
        return op.value
    end
    return nothing
end

# Materialise a scalar Julia Number/Bool as an `arith.constant`, using the
# *signless* MLIR integer type (i8/i16/i32/i64) for any unsigned Julia
# integer. MLIR's `arith.constant` rejects signed/unsigned-tagged integer
# return types — every integer there is signless and signedness is per-op.
function _const_value(v::Bool)
    return IR.result(_arith.constant(;
        value=IR.Attribute(v ? 1 : 0, IR.Type(Bool)), result=IR.Type(Bool)))
end
function _const_value(v::Number)
    if v isa Unsigned
        T_signless = v isa UInt8  ? Int8  :
                     v isa UInt16 ? Int16 :
                     v isa UInt32 ? Int32 :
                     v isa UInt64 ? Int64 : Int64
        signed_v = reinterpret(T_signless, v)
        out_t = mlir_elem_type(T_signless)
        return IR.result(_arith.constant(;
            value=IR.Attribute(signed_v, out_t), result=out_t))
    end
    return IR.result(_arith.constant(; value=IR.Attribute(v)))
end

# Resolve to a Value, materialising a Julia number/bool literal as an
# `arith.constant` of the corresponding scalar MLIR type. Returns `nothing`
# if `op` is tracked-only (tuple/view/etc.).
function resolve_value_or_const(lc::LowerCtx, @nospecialize(op))
    v = resolve_value(lc, op)
    v === nothing || return v
    if op isa Bool
        return _const_value(op)
    elseif op isa Number
        return _const_value(op)
    elseif op isa Argument
        c = get(lc.arg_const, op.n, nothing)
        c === nothing && return nothing
        return materialise_const(lc, c)
    elseif op isa Undef
        # IRStructurizer inserts `Undef(T)` on dead branches of a structured
        # `if`. The runtime never observes the value (branch guards), so any
        # value of type T works. Use a zero-valued arith.constant.
        return undef_value(lc, op.type)
    elseif op isa QuoteNode
        # A Symbol or Type literal — handled by callers that special-case it.
        return nothing
    elseif op isa GlobalRef
        # Module-level binding (e.g. a global constant). If it resolves to a
        # `Number`, materialise as an `arith.constant`; otherwise leave for
        # the caller to special-case (Types are handled via resolve_const).
        v = try
            getfield(op.mod, op.name)
        catch
            return nothing
        end
        (v isa Bool || v isa Number) && return _const_value(v)
        return nothing
    end
    return nothing
end

function undef_value(lc::LowerCtx, @nospecialize(T))
    if T isa Type && T <: Number
        return materialise_zero_scalar(T)
    end
    error("undef_value: unsupported type $T")
end

function materialise_zero_scalar(T::Type)
    if T === Bool
        return IR.result(_arith.constant(;
            value=IR.Attribute(0, IR.Type(Bool)), result=IR.Type(Bool)))
    elseif T <: Integer
        return IR.result(_arith.constant(;
            value=IR.Attribute(zero(T), mlir_elem_type(T))))
    elseif T <: AbstractFloat
        return IR.result(_arith.constant(; value=IR.Attribute(zero(T))))
    end
    error("materialise_zero_scalar: unsupported $T")
end

# Materialise a Julia constant value as an MLIR Value.
function materialise_const(lc::LowerCtx, @nospecialize(c))
    (c isa Bool || c isa Number) && return _const_value(c)
    error("materialise_const: cannot materialise $c::$(typeof(c))")
end


function walk_call!(lc::LowerCtx, idx::Int, @nospecialize(callee),
                    args::Vector{Any}, @nospecialize(typ))
    fname = callee_name(callee)

    if fname === :tuple
        lc.tuples[idx] = collect(args)
        return nothing

    elseif fname === :getfield
        return emit_getfield!(lc, idx, args, typ)

    end

    # ----- SPMD-mode dispatch for plain Julia callees -----
    if lc.spmd
        if fname === :memoryrefnew
            return emit_spmd_memoryrefnew!(lc, idx, args, typ)
        elseif fname === :memoryrefget
            return emit_spmd_memoryrefget!(lc, args, typ)
        elseif fname === :memoryrefset!
            return emit_spmd_memoryrefset!(lc, args, typ)
        elseif fname === :atomic_index!
            # KA.@atomic / Atomix.@atomic — KA's *portable* atomic. The KA
            # extension overlays `Atomix.modify!(IndexableRef, op, x, ord)` onto
            # `Frontend.Intrinsics.atomic_index!(arr, op, val, idx)`, so by the
            # time the walker sees it the call is a clean marker (not the raw
            # pointer-arithmetic + `atomicrmw` llvmcall it would otherwise inline
            # to). Adapt to the modifyindex emitter, which expects
            # (mem, order, op, val, idx): `arr` (an Argument) is accepted
            # directly as `mem` and `order` is unused, so we pad position 2.
            return emit_spmd_atomic_modifyindex!(lc,
                (args[1], args[1], args[2], args[3], args[4]), typ)
        elseif fname === :modifyindex_atomic!
            # Bare `Base.@atomic arr[i] op= x` (CPU array) → Base.modifyindex_atomic!.
            # Not the KA-portable form (that's `KA.@atomic`/`:atomic_index!`
            # above) but Julia-native array atomics work on this path too.
            return emit_spmd_atomic_modifyindex!(lc, args, typ)
        elseif fname === :throw_methoderror && length(args) >= 1 &&
               callee_name(args[1]) === :modifyindex_atomic!
            # In a KA kernel, `@atomic arr[bucket] += x` lowers to
            # `Base.modifyindex_atomic!(arr, order, op, val, i)` — but that
            # method wants a GenericMemory first arg, not the Vector, so under
            # the Frontend interpreter inference can't resolve it and wraps the
            # intended call as `Core.throw_methoderror(modifyindex_atomic!, arr,
            # order, op, val, i)` (kernel rettype becomes Union{}). The atomic
            # is fully recoverable from args[2:end]; we emit it directly and
            # the bogus "throw" never happens (we replace it with the RMW).
            return emit_spmd_atomic_modifyindex!(lc, args[2:end], typ)
        elseif fname === :throw || fname === :throw_complex_domainerror ||
               fname === :throw_inexacterror || fname === :throw_overflowerror
            # Dead inside elided bounds-check IfOps; if we hit it here the
            # walker is processing a live throw, which the SPMD MVP doesn't
            # support. `throw_complex_domainerror` is the guard `Base.sqrt`
            # emits for negative inputs (`sqrt(x<0)`); on the lane-vector path
            # the compare is varying so the throw can't be hoisted out — and
            # the kernel author has opted into the math.sqrt (NaN-on-negative)
            # semantics anyway. Drop it (LLVM end gets a no-op).
            return nothing
        end
    end

    # Raw Core.Intrinsics / builtins from the Frontend — see
    # `emit_raw_core_intrinsic!`. These are the plain-Julia op names that
    # weren't rewritten to the named Intrinsics handled above.
    if fname in _RAW_CORE_INTRINSICS
        return emit_raw_core_intrinsic!(lc, fname, args, typ)
    end

    # Lane-index sentinel. `Frontend.Intrinsics.global_index` marks the global
    # thread index; the walker binds it to the lane value synthesized per grid
    # step — a `vector<W×iX>` on the CPU/SPMD path, or a scalar `gpu.thread_id +
    # block_id*block_dim` on the GPU SIMT path.
    if fname === :global_index && lc.spmd && haskey(lc.arg_vals, lc.lane_arg)
        return lc.arg_vals[lc.lane_arg]
    end

    # Local linear index (1-based): CPU SPMD = vector splat(1)+step(0..W-1);
    # GPU SIMT (lane_width==1) = gpu.thread_id + 1 (scalar).
    if fname === :local_index && lc.spmd
        lane_t = mlir_elem_type(lc.lane_idx_type)
        if lc.lane_width == 1                      # GPU SIMT
            idx_t = IR.IndexType()
            dimx = parse(IR.Attribute, "#gpu<dim x>")
            tid  = IR.result(_gpu.thread_id(; result_0=idx_t, dimension=dimx))
            one  = IR.result(_arith.constant(; value=IR.Attribute(Int(1), idx_t)))
            l1   = IR.result(_arith.addi(tid, one; result=idx_t))
            return IR.result(_arith.index_cast(l1; out=lane_t))
        else                                        # CPU SPMD: splat(1) + step
            vt   = IR.VectorType(1, Int[lc.lane_width], lane_t)
            step = _emit_step_vec(vt, lc.lane_idx_type, lc.lane_width)
            one  = IR.result(_arith.constant(; value=IR.Attribute(Int(1), lane_t)))
            os   = IR.result(_vector.broadcast(one; vector=vt))
            return IR.result(_arith.addi(os, step; result=vt))
        end
    end

    # Group linear index (1-based, uniform scalar): CPU = bid+1; GPU = block_id+1.
    # Scalar result; `_spmd_harmonise`/`_broadcast_to_match` lift it on use.
    if fname === :group_index && lc.spmd
        idx_t = IR.IndexType(); lane_t = mlir_elem_type(lc.lane_idx_type)
        bid = lc.lane_width == 1 ?
            IR.result(_gpu.block_id(; result_0=idx_t,
                       dimension=parse(IR.Attribute, "#gpu<dim x>"))) :
            lc.bids[1]
        one = IR.result(_arith.constant(; value=IR.Attribute(Int(1), idx_t)))
        g1  = IR.result(_arith.addi(bid, one; result=idx_t))
        return IR.result(_arith.index_cast(g1; out=lane_t))
    end

    # Group size (uniform count): CPU = lane_width const; GPU = block_dim.
    if fname === :group_size && lc.spmd
        lane_t = mlir_elem_type(lc.lane_idx_type)
        if lc.lane_width == 1                      # GPU
            idx_t = IR.IndexType()
            bd = IR.result(_gpu.block_dim(; result_0=idx_t,
                       dimension=parse(IR.Attribute, "#gpu<dim x>")))
            return IR.result(_arith.index_cast(bd; out=lane_t))
        else                                        # CPU compile-time const
            return IR.result(_arith.constant(;
                value=IR.Attribute(Int(lc.lane_width), lane_t)))
        end
    end

    # N-D `@index(Global/Local/Group, NTuple)` → per-dim 1-based index values,
    # registered as the marker result's tuple components (`ssa_multi`). The
    # kernel's `i, j = @index(…, NTuple)` destructure becomes `getfield(result,
    # d)`, which returns component `d` directly.
    if (fname === :global_ntuple || fname === :local_ntuple ||
        fname === :group_ntuple) && lc.spmd
        kind = fname === :global_ntuple ? :global :
               fname === :local_ntuple  ? :local : :group
        # Grid rank N is carried by the marker's `Val{N}` arg (the only source
        # on the GPU path, where lc.wg_dims is empty).
        a1 = args[1]
        N = a1 isa Val ? Int(typeof(a1).parameters[1]) :
            (a1 isa Type && a1 <: Val) ? Int(a1.parameters[1]) :
            error("ntuple @index: expected Val{N} marker arg, got $a1 :: $(typeof(a1))")
        lc.ssa_multi[idx] = _emit_nd_index!(lc, kind, N)
        return nothing
    end

    # `@localmem` (shared_alloc) / `@private` (private_alloc) → a workgroup or
    # per-thread buffer, tracked so `buf[…]` accesses route to it.
    if fname === :shared_alloc && lc.spmd
        return emit_local_buffer!(lc, idx, args, true)
    end
    if fname === :private_alloc && lc.spmd
        return emit_local_buffer!(lc, idx, args, false)
    end

    # Workgroup barrier marker (`Frontend.Intrinsics.barrier`, from KA
    # `@synchronize`). On the GPU SIMT path it's a real `gpu.barrier` (threads
    # are hardware lanes, so it actually synchronizes shared-memory writes/reads).
    # On the CPU SIMD path the W lanes are SIMD lanes of one thread executing in
    # lockstep, so a barrier is a no-op (the scatter→gather data dependency
    # already orders shared writes before reads — see project_simt_over_cpu_simd).
    if fname === :barrier && lc.spmd
        lc.lane_width == 1 && _gpu.barrier()
        return nothing
    end

    # Tail-block masking. GPU SIMT: valid iff in range on every dim,
    # `∧_d (thread_id.d + block_id.d*block_dim.d < ndrange[d])`. CPU SPMD: always
    # true.
    if fname === :valid_index && lc.spmd
        if lc.gpu_module_block !== nothing && !isempty(lc.nd_dims)
            idx_t = IR.IndexType()
            dimnames = ("x", "y", "z")
            slt = IR.Attribute(2, IR.Type(Int64))   # arith.cmpi signed-less-than
            valid = nothing
            for d in 1:length(lc.nd_dims)
                da = parse(IR.Attribute, "#gpu<dim $(dimnames[d])>")
                tid  = IR.result(_gpu.thread_id(; result_0=idx_t, dimension=da))
                bid  = IR.result(_gpu.block_id(; result_0=idx_t, dimension=da))
                bdim = IR.result(_gpu.block_dim(; result_0=idx_t, dimension=da))
                off  = IR.result(_arith.muli(bid, bdim; result=idx_t))
                g    = IR.result(_arith.addi(off, tid; result=idx_t))
                ndc  = IR.result(_arith.constant(;
                            value=IR.Attribute(lc.nd_dims[d], idx_t)))
                cmp  = IR.result(_arith.cmpi(g, ndc; predicate=slt))
                valid = valid === nothing ? cmp :
                        IR.result(_arith.andi(valid, cmp))
            end
            return valid
        end
        # CPU SPMD path (or no ndrange info): every lane valid.
        return IR.result(_arith.constant(;
            value=IR.Attribute(true, IR.Type(Bool)), result=IR.Type(Bool)))
    end

    error("MLIRKernels.walk_call!: unhandled callee $fname " *
          "(callee=$callee, args=$args)")
end

# Resolve the two operands of a binary arith op (lifting Julia literals /
# Const-args to arith.constant if needed) and emit `op_fn(a, b)`.
function emit_binop_value!(lc::LowerCtx, args, op_fn)
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("$(op_fn): unresolved operands ($(args[1]), $(args[2]))")
    a, b = _spmd_harmonise(lc, a, b)
    return IR.result(op_fn(a, b))
end

# ----------------------------------------------------------------------------
# Raw Core.Intrinsics / builtins (the Frontend path)
# ----------------------------------------------------------------------------
#
# Some passes canonicalize raw `Core.Intrinsics.add_float` to the named
# `Intrinsics.addf` etc. The standalone `Frontend.structured` path
# (src/frontend.jl) does NOT run those rewrites — it hands the walker the
# *raw* Julia intrinsic names. So the walker recognises both: the named
# Intrinsics (above) and the raw names here, routing both to the same
# `_arith`/`_math` emitters.
#
# This set covers the float/int arithmetic intrinsics PLUS the integer
# width-conversions (`sext_int`/`zext_int`/`trunc_int`) needed for plain-Julia
# `Int32` index widening (`a[gid::Int32]`). All emitters are vector-aware so
# they work on SPMD lane vectors.
const _RAW_CORE_INTRINSICS = Set{Symbol}([
    :add_float, :sub_float, :mul_float, :div_float, :neg_float,
    :add_int, :sub_int, :mul_int, :neg_int,
    :and_int, :or_int, :xor_int, :not_int,
    :slt_int, :sle_int, :ult_int, :ule_int, :eq_int, :ne_int,
    :lt_float, :le_float, :eq_float, :ne_float,
    :sext_int, :zext_int, :trunc_int, :bitcast,
    :sitofp, :fptosi,
    :ifelse, Symbol("==="),
    # Unary float math intrinsics (raw Julia names; the named-intrinsic path
    # uses :absf/:sqrt/etc, but the Frontend path hands them raw). One operand,
    # operand-typed result — vector-aware via emit_unary_math!.
    :abs_float, :sqrt_llvm, :sqrt_llvm_fast,
    # Integer div/rem (incl. the overflow/zero-"checked" variants — the check is
    # a div-by-zero guard the kernel is responsible for; we emit the unchecked
    # arith op). Needed e.g. by the steprange_last overlay's unsigned rem.
    :udiv_int, :sdiv_int, :urem_int, :srem_int,
    :checked_udiv_int, :checked_sdiv_int, :checked_urem_int, :checked_srem_int,
])

# A constant matching `v`'s MLIR type (scalar or vector<N×elem>) holding the
# given Julia value. Used for neg_int (0 - x) and not_int (x ^ all-ones).
function _const_like(v::IR.Value, jval)
    t = IR.type(v)
    if IR.isvector(t)
        return IR.result(_arith.constant(; value=_splat_attr(jval, t), result=t))
    else
        return IR.result(_arith.constant(; value=IR.Attribute(jval, t), result=t))
    end
end

# Output MLIR type for a width conversion of `v` to element type `target_T`:
# vector<N×target> if `v` is a vector, else scalar target.
function _conv_out_type(v::IR.Value, target_T::Type)
    t = IR.type(v)
    et = mlir_elem_type(target_T)
    return IR.isvector(t) ? IR.VectorType(1, Int[Int(size(t, 1))], et) : et
end

# All-ones Julia value of the element type behind MLIR type `t` (for not_int).
function _allones_of_elem(t)
    et = IR.isvector(t) ? eltype(t) : t
    et == IR.Type(Int32)  && return Int32(-1)
    et == IR.Type(Int64)  && return Int64(-1)
    et == IR.Type(Bool)   && return true
    et == IR.Type(UInt32) && return Int32(-1)
    et == IR.Type(UInt64) && return Int64(-1)
    return Int64(-1)
end

function emit_raw_core_intrinsic!(lc::LowerCtx, name::Symbol, args, @nospecialize(typ))
    CP = ComparisonPredicate
    SG = Signedness
    # Float / int arithmetic — operand-inferred result, vector-harmonised.
    name === :add_float && return emit_binop_value!(lc, args, _arith.addf)
    name === :sub_float && return emit_binop_value!(lc, args, _arith.subf)
    name === :mul_float && return emit_binop_value!(lc, args, _arith.mulf)
    name === :div_float && return emit_binop_value!(lc, args, _arith.divf)
    name === :add_int   && return emit_binop_value!(lc, args, _arith.addi)
    name === :sub_int   && return emit_binop_value!(lc, args, _arith.subi)
    name === :mul_int   && return emit_binop_value!(lc, args, _arith.muli)
    name === :and_int   && return emit_binop_value!(lc, args, _arith.andi)
    name === :or_int    && return emit_binop_value!(lc, args, _arith.ori)
    name === :xor_int   && return emit_binop_value!(lc, args, _arith.xori)
    (name === :udiv_int || name === :checked_udiv_int) && return emit_binop_value!(lc, args, _arith.divui)
    (name === :sdiv_int || name === :checked_sdiv_int) && return emit_binop_value!(lc, args, _arith.divsi)
    (name === :urem_int || name === :checked_urem_int) && return emit_binop_value!(lc, args, _arith.remui)
    (name === :srem_int || name === :checked_srem_int) && return emit_binop_value!(lc, args, _arith.remsi)
    if name === :neg_float
        v = resolve_value_or_const(lc, args[1])
        v === nothing && error("neg_float: unresolved operand")
        return IR.result(_arith.negf(v))
    elseif name === :neg_int
        v = resolve_value_or_const(lc, args[1])
        v === nothing && error("neg_int: unresolved operand")
        return IR.result(_arith.subi(_const_like(v, 0), v))
    elseif name === :not_int
        v = resolve_value_or_const(lc, args[1])
        v === nothing && error("not_int: unresolved operand")
        return IR.result(_arith.xori(v, _const_like(v, _allones_of_elem(IR.type(v)))))
    end
    # Unary float math. `Base.abs(::AbstractFloat)` → :abs_float (single op);
    # `Base.sqrt(::Float32/64)` → :sqrt_llvm. The named-intrinsic path uses
    # :absf/:sqrt; the Frontend hands them raw. Same op as the named-intrinsic
    # dispatch, vector-aware via emit_unary_math!.
    name === :abs_float      && return emit_unary_math!(lc, args, _math.absf, typ)
    name === :sqrt_llvm      && return emit_unary_math!(lc, args, _math.sqrt, typ)
    name === :sqrt_llvm_fast && return emit_unary_math!(lc, args, _math.sqrt, typ)
    # Integer comparisons → emit_cmpi! with synthesized (pred, signedness).
    name === :slt_int && return emit_cmpi!(lc, Any[args[1], args[2], CP.LessThan, SG.Signed], typ)
    name === :sle_int && return emit_cmpi!(lc, Any[args[1], args[2], CP.LessThanOrEqual, SG.Signed], typ)
    name === :ult_int && return emit_cmpi!(lc, Any[args[1], args[2], CP.LessThan, SG.Unsigned], typ)
    name === :ule_int && return emit_cmpi!(lc, Any[args[1], args[2], CP.LessThanOrEqual, SG.Unsigned], typ)
    name === :eq_int  && return emit_cmpi!(lc, Any[args[1], args[2], CP.Equal, SG.Signed], typ)
    name === :ne_int  && return emit_cmpi!(lc, Any[args[1], args[2], CP.NotEqual, SG.Signed], typ)
    name === Symbol("===") && return emit_cmpi!(lc, Any[args[1], args[2], CP.Equal, SG.Signed], typ)
    # Float comparisons → emit_cmpf!.
    name === :lt_float && return emit_cmpf!(lc, Any[args[1], args[2], CP.LessThan], typ)
    name === :le_float && return emit_cmpf!(lc, Any[args[1], args[2], CP.LessThanOrEqual], typ)
    name === :eq_float && return emit_cmpf!(lc, Any[args[1], args[2], CP.Equal], typ)
    name === :ne_float && return emit_cmpf!(lc, Any[args[1], args[2], CP.NotEqual, ComparisonOrdering.Unordered], typ)
    # Width / type conversions. Raw arg order is (to_type, value).
    if name === :sext_int || name === :zext_int || name === :trunc_int ||
       name === :sitofp || name === :fptosi
        target_T = something(resolve_const(lc, args[1]), args[1])
        target_T isa Type || error("$name: target type must be a Type, got $(args[1])")
        v = resolve_value_or_const(lc, args[2])
        v === nothing && error("$name: unresolved operand $(args[2])")
        out_t = _conv_out_type(v, target_T)
        # Same-width int conversions are no-ops, and `arith.{extsi,extui,trunci}`
        # reject equal operand/result types. This fires on the SPMD lane index:
        # `global_index()` is typed `Int32` in Julia, but the lane vector is
        # materialised directly at `lane_idx_type` (Int64) width, so a later
        # `Core.sext_int(Int64, i)` would emit an illegal `extsi i64→i64`. Pass
        # the operand through unchanged.
        if (name === :sext_int || name === :zext_int || name === :trunc_int) &&
           IR.type(v) == out_t
            return v
        end
        name === :sext_int  && return IR.result(_arith.extsi(v; out=out_t))
        name === :zext_int  && return IR.result(_arith.extui(v; out=out_t))
        name === :trunc_int && return IR.result(_arith.trunci(v; out=out_t))
        name === :sitofp    && return IR.result(_arith.sitofp(v; out=out_t))
        name === :fptosi    && return IR.result(_arith.fptosi(v; out=out_t))
    elseif name === :bitcast
        # Raw `bitcast(T, x)`; emit_bitcast! expects (value, type) → swap.
        return emit_bitcast!(lc, Any[args[2], args[1]], typ)
    elseif name === :ifelse
        # Core.ifelse(c, x, y) → arith.select. Harmonise x/y, broadcast a
        # scalar condition to the vector width if needed.
        c = resolve_value_or_const(lc, args[1])
        x = resolve_value_or_const(lc, args[2])
        y = resolve_value_or_const(lc, args[3])
        (c === nothing || x === nothing || y === nothing) &&
            error("ifelse: unresolved operands")
        x, y = _spmd_harmonise(lc, x, y)
        if IR.isvector(IR.type(x)) && !IR.isvector(IR.type(c))
            n = Int(size(IR.type(x), 1))
            c = _broadcast_to_match(c, IR.VectorType(1, Int[n], IR.Type(Bool)))
        end
        return IR.result(_arith.select(c, x, y))
    end
    error("emit_raw_core_intrinsic!: name $name in raw set but unhandled")
end

# In SPMD mode, if one operand is a vector (varying) and the other is scalar
# (uniform), broadcast the scalar to match the vector type. No-op outside
# SPMD mode or when shapes already match. Used by all binop / cmp emitters.
function _spmd_harmonise(lc::LowerCtx, a::IR.Value, b::IR.Value)
    lc.spmd || return a, b
    ta = IR.type(a); tb = IR.type(b)
    if IR.isvector(ta) && !IR.isvector(tb)
        b = _broadcast_to_match(b, ta)
    elseif IR.isvector(tb) && !IR.isvector(ta)
        a = _broadcast_to_match(a, tb)
    end
    return a, b
end

function _spmd_harmonise(lc::LowerCtx, a::IR.Value, b::IR.Value, c::IR.Value)
    lc.spmd || return a, b, c
    # Find the vector type (if any) among the operands; broadcast scalars to it.
    vec_t = nothing
    for v in (a, b, c)
        if IR.isvector(IR.type(v))
            vec_t = IR.type(v); break
        end
    end
    if vec_t !== nothing
        IR.isvector(IR.type(a)) || (a = _broadcast_to_match(a, vec_t))
        IR.isvector(IR.type(b)) || (b = _broadcast_to_match(b, vec_t))
        IR.isvector(IR.type(c)) || (c = _broadcast_to_match(c, vec_t))
    end
    return a, b, c
end

# Broadcast a scalar `v` to a vector of element type matching `vec_t`. If `v`'s
# scalar type differs from the vector element type (common for SPMD where a
# user literal `1` is `i64` but the lane index is the same — but cmpi on
# `vector<W × i64>` needs the scalar broadcast at i64), insert an int-cast.
function _broadcast_to_match(v::IR.Value, vec_t)
    elem_t = eltype(vec_t)
    v_t = IR.type(v)
    if v_t != elem_t
        # Scalar element-type mismatch. Common case: index → integer cast.
        if v_t == IR.IndexType()
            v = IR.result(_arith.index_cast(v; out=elem_t))
        elseif elem_t == IR.IndexType()
            v = IR.result(_arith.index_cast(v; out=elem_t))
        else
            # Integer width mismatch — extend or truncate. We only need this
            # in narrow cases; fall back to extsi for now.
            v = IR.result(_arith.extsi(v; out=elem_t))
        end
    end
    return IR.result(_vector.broadcast(v; vector=vec_t))
end

# Intrinsics.mulhii(a, b) — high half of the unsigned-widened product.
# Implemented as: widen both operands by `extui` to 2W bits, multiply with
# nuw, shift right by W, then truncate back to W. On vector operands the
# arith ops broadcast naturally.

# Intrinsics.shri(a, b, signedness) — arithmetic / logical right shift.

# Intrinsics.trunci(x, T) — narrow integer cast to type T.

# Intrinsics.itof(x, F, signedness) — int → float convert.

# Intrinsics.negf(x) → arith.negf. Result type matches operand.

# Intrinsics.cat((lhs, rhs), axis::Int) — concatenate two tiles along
# `axis` (0-indexed Julia col-major). For 1-D tiles, lowers to `vector.shuffle`
# with the identity-then-shifted lane permutation. For N-D tiles, lowers via
# two `vector.insert_strided_slice` ops into a zero-initialised result.

# Intrinsics.extract(tile, index, shape) — extract a non-overlapping
# subtile at slice (index) of size (shape). Both `index` and `shape` are in
# Julia col-major order; `index` is 0-indexed (frontend already converted from
# 1-based). Lowers to `vector.extract_strided_slice`, with axes reversed for
# MLIR's row-major convention.

# Build an `ArrayAttr` of i64 attributes — the format `vector.{insert,extract}_strided_slice`
# expect for their offsets/sizes/strides attributes (a `DenseArrayAttr` is silently
# dropped by the create-op state).

# Intrinsics.get_num_tile_blocks(axis) — grid extent along the given
# 0-indexed axis. For axes within the runtime grid, return `index_cast` of
# the grid Value (held by the outer wrapper) cast to i32. For axes beyond
# the grid (`rng_key` always reads axes 0..1 even on 1-D launches),
# return constant Int32(1).

# Unary math op (math.exp, math.log, …). Result type comes from the operand
# (math ops are elementwise; the result type matches the operand type).
function emit_unary_math!(lc::LowerCtx, args, op_fn, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    a === nothing && error("$(op_fn): unresolved operand $(args[1])")
    return IR.result(op_fn(a; result=IR.type(a)))
end

# Intrinsics.fma(x, y, z) → math.fma. Same vector shape across all
# three operands.

# Intrinsics.maxi(a, b, signedness)/mini(a, b, signedness) → arith
# max{s,u}i / min{s,u}i. Signedness is the 3rd arg.

# cmpi(lhs, rhs, predicate::ComparisonPredicate.T, sign::Signedness.T)
# → arith.cmpi with the matching i64 predicate attribute.
function emit_cmpi!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("cmpi: unresolved operands ($(args[1]), $(args[2]))")
    pred = something(resolve_const(lc, args[3]), args[3])
    signed = length(args) >= 4 ?
             something(resolve_const(lc, args[4]), args[4]) :
             Signedness.Signed
    pred isa ComparisonPredicate.T ||
        error("cmpi: predicate must be ComparisonPredicate.T, got $pred")
    signed isa Signedness.T ||
        error("cmpi: signedness must be Signedness.T, got $signed")
    mlir_pred = cmpi_predicate_code(pred, signed)
    pred_attr = IR.Attribute(mlir_pred, IR.Type(Int64))
    # SPMD: broadcast a uniform scalar operand to the lane vector width when
    # the other side is varying (e.g. `gid < n` masks, or a lane-vector vs
    # scalar-literal compare).
    a, b = _spmd_harmonise(lc, a, b)
    # arith.cmpi result is i1; for tile element type Bool yields i1 (or
    # vector<...xi1>). Result type is inferred by the op builder since we
    # don't pass it.
    return IR.result(_arith.cmpi(a, b; predicate=pred_attr))
end

function cmpi_predicate_code(pred::ComparisonPredicate.T, signed::Signedness.T)
    pred === ComparisonPredicate.Equal && return 0
    pred === ComparisonPredicate.NotEqual && return 1
    is_signed = signed === Signedness.Signed
    pred === ComparisonPredicate.LessThan        && return is_signed ? 2 : 6
    pred === ComparisonPredicate.LessThanOrEqual && return is_signed ? 3 : 7
    pred === ComparisonPredicate.GreaterThan     && return is_signed ? 4 : 8
    pred === ComparisonPredicate.GreaterThanOrEqual && return is_signed ? 5 : 9
    error("cmpi: unsupported predicate $pred")
end

# cmpf(lhs, rhs, predicate::ComparisonPredicate.T, [ordering])
function emit_cmpf!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("cmpf: unresolved operands ($(args[1]), $(args[2]))")
    pred = something(resolve_const(lc, args[3]), args[3])
    ord  = length(args) >= 4 ?
           something(resolve_const(lc, args[4]), args[4]) :
           ComparisonOrdering.Ordered
    pred isa ComparisonPredicate.T ||
        error("cmpf: predicate must be ComparisonPredicate.T, got $pred")
    code = cmpf_predicate_code(pred, ord)
    pred_attr = IR.Attribute(code, IR.Type(Int64))
    # SPMD: a varying-vs-uniform compare (e.g. the `x < 0` guard `Base.sqrt`
    # emits, comparing the lane vector against a scalar 0.0) needs the scalar
    # broadcast to the lane vector width so both operands share a type.
    a, b = _spmd_harmonise(lc, a, b)
    return IR.result(_arith.cmpf(a, b; predicate=pred_attr))
end

function cmpf_predicate_code(pred::ComparisonPredicate.T, ord::ComparisonOrdering.T)
    is_ord = ord === ComparisonOrdering.Ordered
    pred === ComparisonPredicate.Equal           && return is_ord ? 1  : 8
    pred === ComparisonPredicate.GreaterThan     && return is_ord ? 2  : 9
    pred === ComparisonPredicate.GreaterThanOrEqual && return is_ord ? 3 : 10
    pred === ComparisonPredicate.LessThan        && return is_ord ? 4  : 11
    pred === ComparisonPredicate.LessThanOrEqual && return is_ord ? 5  : 12
    pred === ComparisonPredicate.NotEqual        && return is_ord ? 6  : 13
    error("cmpf: unsupported predicate $pred")
end

# exti(x, target_jl_type, sign) → arith.extsi / arith.extui

# Intrinsics.cldi(lhs, rhs, sign) → arith.ceildivsi / arith.ceildivui.

# Intrinsics.remi(lhs, rhs, sign) → arith.remsi / arith.remui.
# Surfaces from `rem(::IntTile, ::Integer)` (and tile/tile variants); the
# atomic histogram path uses this for bucket = v % n_buckets.

# Intrinsics.fldi(lhs, rhs, sign) → arith.floordivsi / arith.divui.
# `fldi` is signed floor-division (rounding toward -∞) for signed args; on
# the unsigned side it coincides with truncated division (`arith.divui`).

# Intrinsics.mma(lhs, rhs, acc) — matrix-multiply-accumulate in
# TileIR-row-major form. The frontend's `muladd(a, b, acc)` for Julia 2-D
# tiles becomes `Intrinsics.mma(b, a, acc)`; the batched (≥3-D × ≥3-D) path
# flattens trailing batch dims to a single leading "batch" in TileIR
# row-major and then calls `Intrinsics.mma(b, a, acc)` with operands of
# TileIR shape (B, …).
#
# 2-D case. In TileIR / MLIR row-major coordinates:
#   lhs (the "%50 = b_julia" operand) has shape (N_iter, K_iter)
#   rhs (the "%40 = a_julia" operand) has shape (K_iter, M_iter)
#   acc / result                       has shape (N_iter, M_iter)
# Indexing maps using contraction iterators (m_iter, n_iter, k_iter):
#   lhs: (m, n, k) -> (n, k)
#   rhs: (m, n, k) -> (k, m)
#   acc: (m, n, k) -> (n, m)
# Iterator types: [parallel(m), parallel(n), reduction(k)].
#
# 3-D batched case. The batched-mma `_muladd` reshapes
# operands so that, in Julia col-major, lhs is `(K, N, B_flat)`, rhs is
# `(M, K, B_flat)`, and acc is `(M, N, B_flat)`. After Julia→TileIR axis
# reversal (`mlir_tile_type`), in MLIR row-major coordinates:
#   lhs has shape (B, N_iter, K_iter)
#   rhs has shape (B, K_iter, M_iter)
#   acc has shape (B, N_iter, M_iter)
# Iterators (b, m, n, k) with `b` parallel in all three operands:
#   lhs: (b, m, n, k) -> (b, n, k)
#   rhs: (b, m, n, k) -> (b, k, m)
#   acc: (b, m, n, k) -> (b, n, m)
# Iterator types: [parallel(b), parallel(m), parallel(n), reduction(k)].
#
# We dispatch on the rank of `acc` (which is also the rank of lhs/rhs in the
# canonical batched form): 2 → plain matmul, 3 → batched matmul. Higher ranks
# don't occur because batch dims are pre-flattened to a single axis.

# Intrinsics.broadcast(src, target_shape_tuple) → vector.broadcast.
# `src` can be a scalar literal, a Const-arg scalar, a 0-D / smaller tile.

# Intrinsics.reshape(tile, target_shape_tuple) → vector.shape_cast.

# Intrinsics.permute(tile, perm) → vector.transpose.
#
# `perm` is a 0-indexed Julia (col-major) permutation. The frontend already
# lowered `permutedims(tile, (2, 1))` to `Intrinsics.permute(tile, (1, 0))`,
# i.e. 1-indexed Julia → 0-indexed Julia. We need an MLIR (row-major) perm.
#
# Mapping: given Julia 0-indexed perm `julia_perm`, the row-major (MLIR)
# permutation is
#   mlir_perm[i] = n - 1 - julia_perm[n - 1 - i]   for i in 0:n-1.
# The result MLIR vector type is the input MLIR vector type with its dims
# reordered by `mlir_perm`.

# Intrinsics.constant(shape::Tuple, value, T) → arith.constant of a
# splat dense<value> : vector<...xT> when `value` is a compile-time literal.
# When `value` is a runtime SSA (the `fill(scalar, dims)` overlay uses
# this form to broadcast a scalar to a tile), we instead lower to
# `vector.broadcast` of the scalar Value.

# Intrinsics.reduce((tile,), axis::Int, combiner, (identities,)) →
# Tuple{Tile{..., reduced_shape_with_1_in_axis}}. Lowers to
# vector.multi_reduction + vector.shape_cast (to re-add the size-1 dim that
# reduce semantics preserves).

# Map a combiner function + element type to a vector.kind name.

# Handle Base.getfield in both Argument-rooted (an array view's
# ptr/sizes/strides fields) and SSA-rooted (extract from a multi-result
# control-flow op or a tracked tuple) forms.
# Resolve a `getfield` on a flattened struct arg at (slot, path): a captured
# leaf is a memref (route the result SSA through the array-arg indexing path) or
# a scalar (return its Value); a non-leaf intermediate is recorded so the next
# getfield extends the path.
function _resolve_captured!(lc::LowerCtx, idx::Int, slot::Int, path::Tuple)
    if haskey(lc.captured, (slot, path))
        (syn, knd) = lc.captured[(slot, path)]
        if knd === :memref
            lc.array_ssa[idx] = syn
            return nothing
        else
            return lc.arg_vals[syn]
        end
    end
    lc.field_paths[idx] = (slot, path)
    return nothing
end

function emit_getfield!(lc::LowerCtx, idx::Int, args, @nospecialize(typ))
    obj = args[1]
    # Resolve through transparent aliases (e.g. the array that `getfield(Const,
    # :a)` yielded) so `getfield(alias, :field)` reaches the real Argument/SSA.
    while obj isa SSAValue && haskey(lc.aliases, obj.id)
        obj = lc.aliases[obj.id]
    end
    # `getfield(new_struct, :field)`: extract the recorded field operand. If it
    # is an Argument/SSAValue, alias this result to it (so further getfield/
    # indexing routes through); otherwise resolve it to a Value.
    if obj isa SSAValue && haskey(lc.new_structs, obj.id)
        (T, ops) = lc.new_structs[obj.id]
        fld = args[2] isa QuoteNode ? args[2].value : args[2]
        fi = fld isa Symbol ? Base.fieldindex(T, fld) : Int(fld)
        operand = ops[fi]
        if operand isa Argument || operand isa SSAValue
            lc.aliases[idx] = operand
            return nothing
        end
        return resolve_value_or_const(lc, operand)
    end
    if obj isa Argument
        field = args[2]
        fld_sym = field isa QuoteNode ? field.value :
                  field isa Symbol    ? field :
                  error("getfield: field must be Symbol/QuoteNode, got $field")
        # Flattened struct/closure/wrapped-array arg: resolve `getfield(arg, fld)`
        # against the captured field-path tree.
        if obj.n in lc.flattened_slots
            return _resolve_captured!(lc, idx, obj.n, (fld_sym,))
        end
        lc.field_refs[idx] = (obj.n, fld_sym)
        return nothing
    elseif obj isa SSAValue
        # A getfield reaching a non-leaf intermediate of a flattened arg
        # (e.g. `subarray.indices` then `[1]` then `.stop`): extend the path.
        if haskey(lc.field_paths, obj.id)
            (slot, ppath) = lc.field_paths[obj.id]
            f2 = args[2]
            fld = f2 isa QuoteNode ? f2.value :
                  (f2 isa Symbol || f2 isa Integer) ? f2 :
                  something(resolve_const(lc, f2), f2)
            return _resolve_captured!(lc, idx, slot, (ppath..., fld))
        end
        # A captured-array field (the SSA of `getfield(closure, :arr)`): index it
        # exactly like the array arg at its synthetic slot — `:ref`/`:size` get
        # recorded in field_refs so memoryrefnew + memref.dim route to that slot.
        if haskey(lc.array_ssa, obj.id)
            fld = args[2] isa QuoteNode ? args[2].value : args[2]
            lc.field_refs[idx] = (lc.array_ssa[obj.id], fld)
            return nothing
        end
        # `@localmem`/`@private`-rooted getfield on a local-buffer marker:
        #   :ref  → propagate the buffer mapping (memoryrefnew routes to it).
        #   :size → the buffer's STATIC dims as a tracked constant tuple, so
        #           `size(buf, d)` (used in the column-major linearisation of a
        #           multi-dim `buf[i,j]`) resolves to the constant `dims[d]`.
        if haskey(lc.local_memrefs, obj.id)
            fld = args[2] isa QuoteNode ? args[2].value : args[2]
            if fld === :size
                (_, _, dims) = lc.local_memrefs[obj.id]
                lc.tuples[idx] = collect(Any, dims)
            else
                lc.local_memrefs[idx] = lc.local_memrefs[obj.id]
            end
            return nothing
        end
        k = something(resolve_const(lc, args[2]), args[2])
        k isa Integer || error("getfield: SSA-rooted index must be const int, got $(args[2])")
        ki = Int(k)
        # Tuple-tracked: extract the component, then resolve to a Value.
        if haskey(lc.tuples, obj.id)
            comp = lc.tuples[obj.id][ki]
            v = resolve_value_or_const(lc, comp)
            v === nothing && error("getfield: tuple component not a Value: $comp")
            return v
        end
        # Multi-result CF op (IfOp / ForOp / reduce 1-tuple).
        if haskey(lc.ssa_multi, obj.id)
            return lc.ssa_multi[obj.id][ki]
        end
        # Field-ref-rooted: getfield(getfield(_arg, :sizes), k, [bc]) — the
        # parent SSA was tracked as `(arg_id, :sizes|:strides|:ptr)`. Extract
        # the k-th tuple component:
        #   :sizes  → memref.dim(arg, mlir_dim) cast to the element type
        #   :strides → emit memref.extract_strided_metadata + extract; simpler:
        #              use `memref.extract_strided_metadata`. (Not needed for
        #              matmul; we'll error if requested.)
        if haskey(lc.field_refs, obj.id)
            (arg_id, fld) = lc.field_refs[obj.id]
            memref_v = lc.arg_vals[arg_id]
            if fld === :sizes || fld === :size
                # `:sizes` is an array view's dims field; `:size` is a plain
                # Julia `Array`'s dims tuple (`getfield(A, :size)[k]`, the form
                # `size(A, k)` and N-D `A[i,j]` linearisation expand to). Both
                # resolve a per-dim extent the same way.
                # Convert Julia dim ki (col-major, 1-indexed) to MLIR dim
                # (row-major, 0-indexed). MemRefType rank reverses: Julia
                # dim k ↔ MLIR dim N - k.
                AT = nothing
                N = ndims(IR.type(memref_v))
                mlir_dim = N - ki
                idx_t = IR.IndexType()
                idx_const = IR.result(_arith.constant(;
                    value=IR.Attribute(mlir_dim, idx_t)))
                dim_v = IR.result(_memref.dim(memref_v, idx_const; result=idx_t))
                # Cast to the requested result type (Int32 normally).
                if typ isa Type && typ <: Integer
                    out_t = mlir_elem_type(typ)
                    return IR.result(_arith.index_cast(dim_v; out=out_t))
                else
                    return dim_v
                end
            else
                error("getfield: unsupported field-rooted access :$fld at %$(obj.id)")
            end
        end
        error("getfield: cannot resolve SSA-rooted obj %$(obj.id) at field $ki")
    end
    # Constant-tuple obj: `getfield((c1, c2, …), k)` with a const index. Arises
    # from a literal-index `KA.@atomic out[1] op x` — the IndexableRef's index
    # tuple `(1,)` is a compile-time constant the Frontend interpreter (default
    # opt params) doesn't fold through `ref.indices[1]`, so it reaches the walker
    # as a getfield on the literal tuple. Extract and materialise the element.
    if obj isa Tuple
        k = something(resolve_const(lc, args[2]), args[2])
        k isa Integer || error("getfield(tuple): index must be const int, got $(args[2])")
        v = resolve_value_or_const(lc, obj[Int(k)])
        v === nothing && error("getfield(tuple): element $(obj[Int(k)]) not resolvable")
        return v
    end
    error("getfield: unsupported obj $obj")
end

# ----------------------------------------------------------------------------
# Control flow: scf.if / scf.for
# ----------------------------------------------------------------------------

# A branch region is a "dead throw arm" of a Base-math domain/overflow guard
# (e.g. `Base.sqrt`'s `if x<0; throw_complex_domainerror; end`) when, after
# dropping the throw, it carries no observable effect: every body statement is
# either a `throw`/`throw_*(...)` call (typed `Union{}`) or trivial
# (nothing/constant/QuoteNode). Such arms `return` (early kernel exit). The
# *live* arm — which also ends in `return` at the kernel tail — does real work
# (loads/stores/arith), so it is NOT classified dead. We must inspect the body,
# not just the terminator: structurization gives BOTH arms a `ReturnNode`.
function _is_dead_throw_branch(block::Block)
    # A live continuation yields a value rather than returning — never dead.
    block.terminator isa Core.ReturnNode || return false
    saw_throw = false
    for (_, entry) in block.body
        stmt = entry.stmt
        if stmt isa Expr && stmt.head === :invoke
            nm = callee_name(stmt.args[2])
            if nm === :throw || startswith(String(nm), "throw")
                saw_throw = true
                continue
            end
            return false  # a non-throw invoke = real work
        elseif stmt isa Expr && stmt.head === :call
            nm = callee_name(stmt.args[1])
            if nm === :throw || startswith(String(nm), "throw")
                saw_throw = true
                continue
            end
            return false
        elseif stmt === nothing || stmt isa QuoteNode ||
               stmt isa Core.ReturnNode || stmt isa GlobalRef ||
               stmt isa Core.Const || !(stmt isa Expr)
            continue  # trivial / value-only statement
        else
            return false  # any other Expr (getfield, foreigncall, …) = real work
        end
    end
    return saw_throw
end

function emit_if!(lc::LowerCtx, idx::Int, op::IfOp, @nospecialize(typ))
    # SPMD mode: `if boundscheck …` IfOps gate dead bounds-check code.
    # Their condition resolves to the `:boundscheck` sentinel (no Value).
    # We assume `@inbounds` and skip the entire IfOp — its then-branch is
    # the actual bounds check (compare + throw), its else-branch is empty.
    if lc.spmd && op.condition isa SSAValue &&
       get(lc.sentinels, op.condition.id, nothing) === :boundscheck
        return nothing
    end
    cond_v = resolve_value_or_const(lc, op.condition)
    cond_v === nothing && error("scf.if: cannot resolve condition $(op.condition)")
    # SPMD mode: a varying (lane-vector) condition can't drive an `scf.if`
    # (which requires a scalar i1). The varying guards we produce are the
    # domain/overflow checks Base math helpers emit, e.g. `Base.sqrt`:
    #     %g = (x < 0)            (varying, vector<W×i1>)
    #     if %g  then: throw_complex_domainerror; return
    #            else: sqrt_llvm(x); ...store...; return
    # i.e. one branch is a dead early-return (its throw is already dropped in
    # `walk_call!`); the other carries the live computation. We inline that
    # live branch's body into the current block — no scf.if. A varying guard
    # whose BOTH branches do live work would need per-lane masking (the MVP
    # doesn't support it), so we only take this path when one branch is a
    # throw/return-only dead branch.
    if lc.spmd && IR.isvector(IR.type(cond_v))
        then_dead = _is_dead_throw_branch(op.then_region)
        else_dead = _is_dead_throw_branch(op.else_region)
        if then_dead ⊻ else_dead
            live = then_dead ? op.else_region : op.then_region
            walk_block!(lc, live; kind=:entry)
            # If the IfOp yields values (typ is a Tuple), bind them from the
            # live branch's YieldOp so downstream SSAs resolve.
            if typ !== Nothing && live.terminator isa YieldOp
                vals = IR.Value[]
                for v in live.terminator.values
                    rv = resolve_value_or_const(lc, v)
                    rv === nothing &&
                        error("scf.if (inlined live branch): cannot resolve yield $v")
                    push!(vals, rv)
                end
                lc.ssa_multi[idx] = vals
            end
            return nothing
        end
    end
    # Result types — flatten Tuple{...} into a vector of MLIR types.
    result_types, jl_yield_types = if_result_types(typ)

    then_region = IR.Region()
    else_region = IR.Region()
    then_block = IR.Block(IR.Type[], IR.Location[])
    else_block = IR.Block(IR.Type[], IR.Location[])
    push!(then_region, then_block)
    push!(else_region, else_block)

    @with_block then_block begin
        walk_block!(lc, op.then_region; kind=:if)
    end
    @with_block else_block begin
        walk_block!(lc, op.else_region; kind=:if)
    end

    ifop = _scf.if_(cond_v; results=result_types,
                    thenRegion=then_region, elseRegion=else_region)
    if !isempty(result_types)
        vals = [IR.result(ifop, k) for k in 1:length(result_types)]
        lc.ssa_multi[idx] = vals
    end
    return nothing
end

# Decide the scf.if result types from a Tuple{...} or Nothing.
function if_result_types(@nospecialize(typ))
    typ === Nothing && return IR.Type[], Type[]
    typ isa DataType && typ <: Tuple ||
        error("scf.if: expected Tuple{...} or Nothing result type, got $typ")
    rts = IR.Type[]
    jls = Type[]
    for p in typ.parameters
        push!(jls, p)
        if p isa Type && p <: Number
            push!(rts, mlir_elem_type(p))
        else
            error("scf.if: unsupported yield type $p")
        end
    end
    return rts, jls
end

function emit_for!(lc::LowerCtx, idx::Int, op::ForOp, @nospecialize(typ))
    idx_t = IR.IndexType()
    lower_raw = resolve_value_or_const(lc, op.lower)
    upper_raw = resolve_value_or_const(lc, op.upper)
    step_raw  = resolve_value_or_const(lc, op.step)
    (lower_raw === nothing || upper_raw === nothing || step_raw === nothing) &&
        error("scf.for: cannot resolve bounds (lower=$(op.lower), upper=$(op.upper), step=$(op.step))")
    # Cast bounds to `index`. The IRStructurizer normalises range iteration
    # so that the recorded `upper` is already the half-open end
    # (i.e. the SCI emits `%upper = addi(%n, 1)` for an inclusive `1:n`).
    # We therefore do NOT add another 1 here.
    lower_v = cast_to_index(lower_raw)
    upper_v = cast_to_index(upper_raw)
    step_v  = cast_to_index(step_raw)

    # Init args: resolve each to a Value.
    init_vals = IR.Value[]
    iter_types = IR.Type[]
    for iv_ref in op.init_values
        v = resolve_value_or_const(lc, iv_ref)
        v === nothing && error("scf.for: cannot resolve init $iv_ref")
        push!(init_vals, v)
        push!(iter_types, IR.type(v))
    end

    # The MLIR scf.for block args are [index_iv, iter_args...]. The
    # iv_arg type is Int32/Int64 — cast back inside the body.
    block_arg_types = IR.Type[idx_t; iter_types]
    block_arg_locs  = [IR.Location() for _ in 1:length(block_arg_types)]
    body_region = IR.Region()
    body_block = IR.Block(block_arg_types, block_arg_locs)
    push!(body_region, body_block)

    @with_block body_block begin
        # IV: cast index → kernel IV type.
        iv_index = IR.argument(body_block, 1)
        iv_jltype = op.iv_arg.type
        iv_mlir_t = if iv_jltype isa Type && iv_jltype <: Number
            mlir_elem_type(iv_jltype)
        else
            mlir_elem_type(Int32)  # default
        end
        iv_val = if iv_mlir_t == idx_t
            iv_index
        else
            IR.result(_arith.index_cast(iv_index; out=iv_mlir_t))
        end
        lc.block_args[op.iv_arg.id] = iv_val
        # Iter args: bind the loop body's BlockArguments to the MLIR args.
        for (k, ba) in enumerate(op.body.args)
            lc.block_args[ba.id] = IR.argument(body_block, k + 1)
        end
        walk_block!(lc, op.body; kind=:for)
    end

    forop = _scf.for_(lower_v, upper_v, step_v, init_vals;
                      results=iter_types, region=body_region)
    if !isempty(iter_types)
        vals = [IR.result(forop, k) for k in 1:length(iter_types)]
        lc.ssa_multi[idx] = vals
    end
    return nothing
end

# Cast an integer-typed Value to MLIR index. No-op if already index.
function cast_to_index(v::IR.Value)
    t = IR.type(v)
    if t == IR.IndexType()
        return v
    end
    return IR.result(_arith.index_cast(v; out=IR.IndexType()))
end

function callee_name(@nospecialize(callee))
    callee isa GlobalRef && return callee.name
    callee isa Function && return Symbol(nameof(callee))
    callee isa Type && return Symbol(nameof(callee))
    return Symbol(string(callee))
end

# ----------------------------------------------------------------------------
# Gather/scatter (irregular index load/store)
# ----------------------------------------------------------------------------

# Intrinsics.iota((N,), T) → an `IntTile{(N,), T}` with values 0..N-1.
# We materialise this as `vector.step` (which produces vector<Nxindex>),
# then `arith.index_cast` to vector<NxiK>. Indices are Int32 by default;
# the cast is mandatory because downstream cmpi/addi/bitcast all
# expect the iK element type rather than `index`.

# Intrinsics.bitcast(src, target_T) — a signless reinterpret. For tile
# operands whose source MLIR type already matches the target MLIR element
# type (typical for Int32↔UInt32 / Int64↔UInt64 — MLIR is signless), this is
# a no-op. Otherwise we emit `arith.bitcast` (scalar) or `vector.bitcast`
# (vector).
function emit_bitcast!(lc::LowerCtx, args, @nospecialize(typ))
    v = resolve_value_or_const(lc, args[1])
    v === nothing && error("bitcast: cannot resolve operand $(args[1])")
    target_T = something(resolve_const(lc, args[2]), args[2])
    target_T isa Type ||
        error("bitcast: target type must be a Type, got $target_T")
    # In SPMD mode the inferred `typ` may be a 0-D tile (scalar) but the
    # actual operand is a lane-wide vector — derive the output type from the
    # operand's shape, not from `typ`.
    src_t = IR.type(v)
    elem_target = mlir_elem_type(target_T)
    out_t = if lc.spmd && IR.isvector(src_t)
        n = Int(size(src_t, 1))
        IR.VectorType(1, Int[n], elem_target)
    else
        elem_target
    end
    # If the MLIR type already matches, this is an identity (signless types).
    src_t == out_t && return v
    if IR.isvector(out_t)
        return IR.result(_vector.bitcast(v; result=out_t))
    else
        return IR.result(_arith.bitcast(v; out=out_t))
    end
end

# Resolve `mask_ref` to an i1-vector Value matching `idx_shape`. If it
# resolves to `nothing` ("no mask"), construct an all-true splat.
function _resolve_or_alltrue_mask(lc::LowerCtx, mask_ref, idx_shape)
    mask_v = resolve_value_or_const(lc, mask_ref)
    if mask_v === nothing
        true_scalar = IR.result(_arith.constant(;
            value=IR.Attribute(1, IR.Type(Bool)), result=IR.Type(Bool)))
        bool_vec_t = mlir_tile_type(Tuple(idx_shape), Bool)
        return IR.result(_vector.broadcast(true_scalar; vector=bool_vec_t))
    end
    return mask_v
end

# Resolve `pad_ref` to a vector<NxT> pass-thru value. If `nothing`, return
# a zero-of-T splat broadcast to `idx_shape`.
function _resolve_or_zero_passthru(lc::LowerCtx, pad_ref, elem_T::Type,
                                   idx_shape)
    pad_v = resolve_value_or_const(lc, pad_ref)
    if pad_v === nothing
        z = materialise_zero_scalar(elem_T)
        vec_t = mlir_tile_type(Tuple(idx_shape), elem_T)
        return IR.result(_vector.broadcast(z; vector=vec_t))
    end
    return pad_v
end

# ----------------------------------------------------------------------------
# Atomic RMW (atomic_add / atomic_max / atomic_min / atomic_and / atomic_or
# / atomic_xor / atomic_xchg / atomic_cas)
# ----------------------------------------------------------------------------
#
# An atomic RMW takes a pointer/index, a value, and optional mask/order/scope.
# We map the RMW directly to `memref.atomic_rmw <kind> %val, %base[%idx]`. The
# scalar (0-D) form emits one atomic op on `%base[%idx]`. For tile (N-D) forms
# we unroll a small loop over each lane (acceptable for the small atomic tiles
# involved; vectorising would require `vector.scatter` with an atomic ordering,
# which upstream MLIR doesn't expose yet).
#
# `atomic_cas(arr, idx, expected, desired; …)` has no `assign`-style
# single-keyword form on `memref.atomic_rmw`; we lower it via
# `memref.generic_atomic_rmw` with a region body that compares the loaded
# value against `expected` and yields either `desired` (on match) or the
# loaded value (no-op). Returns the prior value.
#
# `atomic_xchg` reuses `memref.atomic_rmw` with kind `assign` — the verifier
# accepts an unconditional store-and-return-old at any element type.
#
# Memory order/scope arguments (Acquire / Release / AcqRel / Relaxed; Block /
# Device / System) are dropped on the CPU MVP path — `memref.atomic_rmw`
# lowers to `llvm.atomicrmw <op> … acq_rel`, which is the strongest ordering
# the verifier emits for this op. Cross-thread synchronisation beyond what
# libomp + acq_rel give isn't part of this target.

# `memref.atomic_rmw`'s `kind` attribute is an integer-encoded enum, BUT
# the enum was reordered between MLIR 18 and MLIR 20 (and `xori` removed in
# MLIR 20). Codes verified by parsing each kind keyword and reading back
# the attribute's int value. Version-conditional table chosen at MLIR call
# time via `MLIR.MLIR_VERSION[]`.
const _ATOMIC_RMW_KIND_CODES_V18 = Dict{Symbol, Int}(
    :addf => 0, :addi => 1, :andi => 2, :assign => 3,
    :maximumf => 4, :maxnumf => 5, :maxs => 6, :maxu => 7,
    :minimumf => 8, :minnumf => 9, :mins => 10, :minu => 11,
    :mulf => 12, :muli => 13, :ori => 14, :xori => 15,
)
const _ATOMIC_RMW_KIND_CODES_V20 = Dict{Symbol, Int}(
    :addf => 0, :addi => 1, :assign => 2,
    :maximumf => 3, :maxs => 4, :maxu => 5,
    :minimumf => 6, :mins => 7, :minu => 8,
    :mulf => 9, :muli => 10, :ori => 11, :andi => 12,
    :maxnumf => 13, :minnumf => 14,
    # NOTE: no :xori in MLIR 20 — emit_atomic_rmw_generic! handles that case.
)
function _atomic_rmw_kind_codes()
    return MLIR.MLIR_VERSION[] < v"19" ? _ATOMIC_RMW_KIND_CODES_V18 :
                                         _ATOMIC_RMW_KIND_CODES_V20
end

# Pick the `memref.atomic_rmw` kind keyword for a reduction op symbol
# (`:add` / `:max` / `:min` / `:and` / `:or` / `:xor` / `:xchg`) at a given
# Julia element type. `atomic_add` on AbstractFloat →
# `addf`; on integer → `addi`. `atomic_max` / `atomic_min` are signed-int /
# float (`maxnumf` / `minnumf`). Bitwise ops are integer-only.
# `atomic_xchg` (unconditional store-and-return-old) maps to the `assign`
# kind, which is valid on any element type (including floats) — the RMW
# verifier accepts it as a typed exchange.
function _atomic_rmw_kind(op::Symbol, elem_T::Type)
    is_float = elem_T <: AbstractFloat
    is_signed = elem_T <: Signed || elem_T === Bool || !(elem_T <: Unsigned)
    if op === :add
        return is_float ? :addf : :addi
    elseif op === :max
        is_float && return :maxnumf
        return is_signed ? :maxs : :maxu
    elseif op === :min
        is_float && return :minnumf
        return is_signed ? :mins : :minu
    elseif op === :and
        is_float && error("atomic_rmw: atomic_and not supported on float type $elem_T")
        return :andi
    elseif op === :or
        is_float && error("atomic_rmw: atomic_or not supported on float type $elem_T")
        return :ori
    elseif op === :xchg
        return :assign
    end
    # `:xor` deliberately not handled here — upstream `memref.atomic_rmw`
    # has no `xori` enum value; callers must route through
    # `emit_atomic_rmw_generic!` with an `arith.xori` body instead.
    error("atomic_rmw: unsupported op $op")
end

# `#vector.kind<…>` keyword for a `vector.reduction` that collapses the W SPMD
# lanes of one block with the atomic's reduction op (`:add`/`:max`/`:min`/`:and`/
# `:or`). Used when many lanes target the *same* slot: the lanes are SIMD lanes
# of one block (no intra-block race), so we reduce them in-register first and do
# a single atomic per block instead of W atomics. Note the float-add reduction
# uses the `add` kind (which is `fadd` on float operands).
function _atomic_reduce_kind(op::Symbol, elem_T::Type)
    is_float = elem_T <: AbstractFloat
    is_signed = elem_T <: Signed || elem_T === Bool || !(elem_T <: Unsigned)
    op === :add && return :add
    op === :max && return is_float ? :maxnumf : (is_signed ? :maxsi : :maxui)
    op === :min && return is_float ? :minnumf : (is_signed ? :minsi : :minui)
    op === :and && return :and
    op === :or  && return :or
    error("atomic reduce: unsupported op $op")
end

# Materialise an MLIR `kind` attribute for `memref.atomic_rmw`.
function _atomic_rmw_kind_attr(kind_kw::Symbol)
    codes = _atomic_rmw_kind_codes()
    code = get(codes, kind_kw, nothing)
    code === nothing && error("atomic_rmw: kind $kind_kw not available in MLIR " *
                              "$(MLIR.MLIR_VERSION[])")
    return IR.Attribute(code, IR.Type(Int64))
end

# Emit one `memref.atomic_rmw` at scalar index `idx_v` (any integer / index
# Value). `val_v` carries the (matched) scalar element. Returns the result
# Value (the old value at the slot, per atomic_rmw semantics).
function _emit_one_atomic_rmw!(base_memref::IR.Value, idx_v::IR.Value,
                                val_v::IR.Value, kind_kw::Symbol,
                                elem_T::Type)
    # NOTE float min/max: `memref.atomic_rmw maxnumf`/`minnumf` is the correct
    # high-level form (it should lower to `llvm.atomicrmw fmax`/`fmin`), but
    # MLIR ≤ 20's `finalize-memref-to-llvm` does NOT lower these kinds — the op
    # survives with dangling `unrealized_conversion_cast` operands and LLVM
    # translation returns null. (`addf/addi` and integer `maxs/mins/maxu/minu`
    # all lower fine.) Routing through `generic_atomic_rmw` doesn't help either:
    # MLIR lowers that to `llvm.cmpxchg`, which rejects float operands (it needs
    # a bitcast-to-int CAS loop we can't express without dropping to the LLVM
    # dialect by hand). The fix is MLIR ≥ 21, where the native lowering exists.
    # We keep emitting the high-level op and surface a clear error on old MLIR.
    if (kind_kw === :maxnumf || kind_kw === :minnumf) && MLIR.MLIR_VERSION[] < v"21"
        error("atomic_rmw $kind_kw (float min/max) is not lowered by MLIR " *
              "$(MLIR.MLIR_VERSION[])'s memref→llvm pass; requires MLIR ≥ 21. " *
              "Use an integer element type, atomic add, or a newer MLIR_jll.")
    end
    elem_mlir = mlir_elem_type(elem_T)
    kind_attr = _atomic_rmw_kind_attr(kind_kw)
    idx_index = cast_to_index(idx_v)
    return IR.result(_memref.atomic_rmw(val_v, base_memref, IR.Value[idx_index];
                                        result=elem_mlir, kind=kind_attr))
end

# Top-level dispatch for `Intrinsics.atomic_{add,max,min}`.
# Args: (ptr_tile_ssa, val, mask, memory_order, memory_scope, token)
# For the 0-D (scalar-index) form we emit a single atomic_rmw. For N-D tiles
# we unroll one atomic per lane, optionally guarded by the mask.

# Emit one `memref.generic_atomic_rmw` performing compare-and-swap:
#   if loaded == expected: yield desired
#   else:                  yield loaded
# Returns the prior (loaded) value.

# Top-level dispatch for `Intrinsics.atomic_cas`.
# Args: (ptr_tile_ssa, expected, desired, mask, memory_order, memory_scope)
# For the 0-D scalar-index form we emit a single generic_atomic_rmw. For the
# N-D tile form we unroll one CAS per lane (optionally masked).

# Emit one `memref.generic_atomic_rmw` performing a custom binary RMW (used
# for ops the upstream `memref.atomic_rmw` enum doesn't cover — currently
# `atomic_xor`). `binop` is one of the `_arith` dialect binary-op functions
# (e.g. `_arith.xori`). The region body computes `new = binop(loaded, val_v)`
# and yields it. Returns the prior (loaded) value.

# Top-level dispatch for RMW ops the upstream `memref.atomic_rmw` enum
# doesn't cover (currently `:xor`). Same args shape as
# `Intrinsics.atomic_{or,and,xor}` — (ptr_tile, val, mask, order, scope).

# ----------------------------------------------------------------------------
# Tile load / store
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# SPMD-mode plain-Julia op emitters
# ----------------------------------------------------------------------------
#
# These handle the SCI ops that arise from Julia's `Vector{T}[i]` /
# `Vector{T}[i] = v` lowering when the kernel is written with plain Julia
# arrays. The IRStructurizer pipeline accepts plain Julia
# `Vector{T}` args and decomposes indexing into:
#
#   %ref = Base.getfield(_arr, :ref)                   # MemoryRef{T}
#   %mr  = Base.memoryrefnew(%ref, %i, false)          # GenericMemoryRef{T}
#   %v   = Base.memoryrefget(%mr, :not_atomic, false)  # T   -- load
#   Base.memoryrefset!(%mr, %v, :not_atomic, false)    # store
#
# When `%i` is the (varying) lane vector, the resulting memoryrefnew is an
# OffsetInfo with that vector as the per-lane index; `memoryrefget`/`set!`
# then lower to `vector.gather`/`vector.scatter` on the array's memref.
# Indices arriving here are 1-based (Julia semantics); on plain Julia code
# we get the raw 1-based index — we subtract 1 below.

# Recover the scalar `lane_base` (`bid * lane_width`) for a contiguous-lane
# transfer_read/write. Encoded once when the lane vector is built; here we
# rebuild it on demand from the current `bid`.
function _spmd_lane_base(lc::LowerCtx)
    idx_t = IR.IndexType()
    bid = lc.bids[1]
    W_const = IR.result(_arith.constant(;
        value=IR.Attribute(Int(lc.lane_width), idx_t)))
    return IR.result(_arith.muli(bid, W_const; result=idx_t))
end

# `Base.modifyindex_atomic!(mem, order, op, val, i)` — the CPU-array target of
# KA `@atomic arr[i] op= x`. On the SPMD path the index `i` is the per-lane
# bucket (a lane vector, e.g. `idx[lane]`), so this is a SCATTER of W atomic
# `@localmem T dims` → a `memref.alloca` of static shape `dims`, in the
# workgroup address space on the GPU SIMT path (real shared memory; `gpu.barrier`
# synchronises it). The buffer is accessed only via the kernel's own
# (column-major) linearisation, so its physical layout is irrelevant — but
# `size(shared,1)` must equal `dims[1]`, which (with the Julia↔MLIR dim reversal
# of `:size`/`memref.dim`) means the static shape is `reverse(dims)`. We register
# the alloca in `lc.local_memrefs` so the marker result (and `getfield(_,:ref)`
# on it) route memoryrefnew/get/set to this buffer instead of an arg memref.
# `@localmem`/`@private` buffer. `shared=true` (@localmem) → per-BLOCK workgroup
# memory (GPU: a `.shared` memref.global; CPU: a per-block alloca).
# `shared=false` (@private) → per-THREAD storage (a default-space alloca, i.e.
# `.local` on GPU — each lane its own copy).
function emit_local_buffer!(lc::LowerCtx, idx::Int, args, shared::Bool)
    what = shared ? "shared_alloc" : "private_alloc"
    length(args) >= 2 || error("$what: expected (T, Val{Dims}), got $args")
    T = args[1] isa Type ? args[1] : something(resolve_const(lc, args[1]), nothing)
    T isa Type || error("$what: element type must be a Type, got $(args[1])")
    vd = args[2]
    Dims = vd isa Val ? typeof(vd).parameters[1] :
           (vd isa Type && vd <: Val) ? vd.parameters[1] :
           error("$what: dims must be Val{Dims}, got $vd")
    dims = Tuple(Dims)
    elem = mlir_elem_type(T)
    shape = Int[reverse(dims)...]
    if shared && lc.lane_width == 1
        # GPU @localmem: a workgroup-space `memref.global` sibling of the
        # gpu.func + `memref.get_global`. gpu-to-nvvm lowers this to real
        # `.shared`. (`memref.alloca(workgroup)` mis-lowers to a local depot.)
        lc.gpu_module_block !== nothing ||
            error("$what: gpu.module block unavailable")
        ws = parse(IR.Attribute, "#gpu.address_space<workgroup>")
        memref_t = IR.MemRefType(elem, shape, IR.Attribute(), ws)
        sym = "__shmem_$(idx)"   # unique per @localmem site (marker SSA id)
        @with_block lc.gpu_module_block begin
            _memref.global_(; sym_name=IR.Attribute(sym),
                            sym_visibility=IR.Attribute("private"),
                            type=IR.Attribute(memref_t))
        end
        buf = IR.result(_memref.get_global(; result=memref_t,
                        name=parse(IR.Attribute, "@" * sym)))
        lc.local_memrefs[idx] = (buf, T, collect(Int, dims))
    else
        # Default-space alloca: per-block @localmem on CPU, OR per-thread
        # @private on either path (each lane its own copy; `.local` on GPU).
        memref_t = IR.MemRefType(elem, shape, IR.Attribute(), IR.Attribute())
        alloca = IR.result(_memref.alloca(IR.Value[], IR.Value[]; memref=memref_t))
        lc.local_memrefs[idx] = (alloca, T, collect(Int, dims))
    end
    return nothing
end

# N-D workgroup index reconstruction for `@index(Global/Local/Group, NTuple)`.
# The workgroup is flattened to a single `vector<W>` lane (W = prod(wg_dims))
# and the grid to a 1-D `scf.parallel`; we recover each Julia (column-major)
# per-dim coordinate by unflattening the per-lane step (0..W-1) over the
# workgroup dims and the block id over the grid dims:
#
#   local[d]  = (step ÷ ∏wg[1:d-1]) % wg[d]            (per-lane vector, 0-based)
#   block[d]  = (bid  ÷ ∏gridsz[1:d-1]) % gridsz[d]    (uniform scalar, 0-based)
#   global[d] = block[d]·wg[d] + local[d] + 1          (per-lane vector, 1-based)
#
# Returns the N per-dim 1-based index Values (vectors for :global/:local; uniform
# scalars for :group, which `_broadcast_to_match` lifts at the use site). The
# enumeration order of work-items is irrelevant — each computes its own output —
# so this need only be a bijection onto the ndrange, which per-dim divisibility
# (checked in the launcher) guarantees.
function _emit_nd_index!(lc::LowerCtx, kind::Symbol, N::Int)
    # The `@index(…, NTuple)` markers return `NTuple{N,Int}` (Julia `Int` =
    # Int64), so the per-dim components must be Int64 to match the IR's
    # linearisation arithmetic — NOT `lc.lane_idx_type` (which is Int32 on the
    # GPU path for the `gid ≤ n` linear-index compare). On the CPU path Int ==
    # lane_idx_type, so this is unchanged there.
    idx_jl = Int
    lane_t = mlir_elem_type(idx_jl)
    idx_t = IR.IndexType()

    # ---- GPU SIMT (lane_width == 1): each thread reads its own N-D coordinate
    # directly from the gpu intrinsics (Julia dim d ↔ gpu dim x/y/z). No flat-
    # lane unflatten; everything is a scalar per thread. 1-based. ----
    if lc.lane_width == 1
        N <= 3 || error("N-D GPU @index: rank $N > 3 (no w dim)")
        dimnames = ("x", "y", "z")
        one_idx() = IR.result(_arith.constant(; value=IR.Attribute(1, idx_t)))
        tolane(v) = IR.result(_arith.index_cast(v; out=lane_t))
        out = IR.Value[]
        for d in 1:N
            da = parse(IR.Attribute, "#gpu<dim $(dimnames[d])>")
            if kind === :local
                tid = IR.result(_gpu.thread_id(; result_0=idx_t, dimension=da))
                push!(out, tolane(IR.result(_arith.addi(tid, one_idx(); result=idx_t))))
            elseif kind === :group
                bid = IR.result(_gpu.block_id(; result_0=idx_t, dimension=da))
                push!(out, tolane(IR.result(_arith.addi(bid, one_idx(); result=idx_t))))
            else
                kind === :global || error("_emit_nd_index!: bad kind $kind")
                tid  = IR.result(_gpu.thread_id(; result_0=idx_t, dimension=da))
                bid  = IR.result(_gpu.block_id(; result_0=idx_t, dimension=da))
                bdim = IR.result(_gpu.block_dim(; result_0=idx_t, dimension=da))
                off  = IR.result(_arith.muli(bid, bdim; result=idx_t))
                g0   = IR.result(_arith.addi(off, tid; result=idx_t))
                push!(out, tolane(IR.result(_arith.addi(g0, one_idx(); result=idx_t))))
            end
        end
        return out
    end

    # ---- CPU SPMD: column-major unflatten of the flat `vector<W>` lane. ----
    wg, nd = lc.wg_dims, lc.nd_dims
    length(wg) == N == length(nd) ||
        error("_emit_nd_index!: rank mismatch (marker N=$N, wg=$(wg), nd=$(nd))")
    W = prod(wg)
    vec_t = IR.VectorType(1, Int[W], lane_t)
    scalar(c) = IR.result(_arith.constant(; value=IR.Attribute(Int(c), lane_t)))
    splat(c)  = IR.result(_vector.broadcast(scalar(c); vector=vec_t))
    step  = _emit_step_vec(vec_t, idx_jl, W)                      # 0..W-1 vector
    bid_i = IR.result(_arith.index_cast(lc.bids[1]; out=lane_t))  # 0-based scalar
    gridsz = Int[nd[d] ÷ wg[d] for d in 1:N]

    out = IR.Value[]
    for d in 1:N
        wstride = prod(@view wg[1:(d - 1)])      # 1 for d == 1
        gstride = prod(@view gridsz[1:(d - 1)])
        # local[d] (vector, 0-based)
        local_d = step
        wstride != 1 && (local_d = IR.result(_arith.divui(local_d, splat(wstride); result=vec_t)))
        local_d = IR.result(_arith.remui(local_d, splat(wg[d]); result=vec_t))
        if kind === :local
            push!(out, IR.result(_arith.addi(local_d, splat(1); result=vec_t)))
            continue
        end
        # block[d] (scalar, 0-based)
        block_d = bid_i
        gstride != 1 && (block_d = IR.result(_arith.divui(block_d, scalar(gstride); result=lane_t)))
        block_d = IR.result(_arith.remui(block_d, scalar(gridsz[d]); result=lane_t))
        if kind === :group
            push!(out, IR.result(_arith.addi(block_d, scalar(1); result=lane_t)))
            continue
        end
        # global[d] (vector) = block[d]·wg[d] + local[d] + 1
        kind === :global || error("_emit_nd_index!: bad kind $kind")
        bc  = IR.result(_arith.muli(block_d, scalar(wg[d]); result=lane_t))
        bc1 = IR.result(_arith.addi(bc, scalar(1); result=lane_t))
        bc1v = IR.result(_vector.broadcast(bc1; vector=vec_t))
        push!(out, IR.result(_arith.addi(bc1v, local_d; result=vec_t)))
    end
    return out
end

# RMWs — one per lane into arr[bucket_lane] — one atomic per lane.
# (A uniform/scalar index does a single atomic.)
# We recover the base memref from the array arg (Argument or getfield(arr,:ref),
# tracked in lc.field_refs); the order/scope and the old-value return are
# dropped (the RMW effect is all that's modelled).
#
# args = (mem, order::Symbol, op, val, i). `op` is Main.+/max/min/... mapped via
# _atomic_rmw_kind. `i` is 1-based Julia → 0-based for the memref.
function emit_spmd_atomic_modifyindex!(lc::LowerCtx, args, @nospecialize(typ))
    length(args) >= 5 ||
        error("modifyindex_atomic!: expected (mem, order, op, val, i), got $args")
    mem_ref, _order, op_ref, val_ref, idx_ref = args[1], args[2], args[3], args[4], args[5]

    # Base memref + element type. An `@atomic` on a `@localmem` buffer roots at
    # a shared_alloc (in lc.local_memrefs); otherwise at an array arg (an
    # Argument or `getfield(arr,:ref)` tracked in lc.field_refs).
    base, elem_T =
        if mem_ref isa SSAValue && haskey(lc.local_memrefs, mem_ref.id)
            (buf, et, _) = lc.local_memrefs[mem_ref.id]
            (_flatten_memref(buf), et)
        else
            arg_id =
                if mem_ref isa SSAValue && haskey(lc.field_refs, mem_ref.id)
                    (aid, fld) = lc.field_refs[mem_ref.id]
                    fld === :ref || error("modifyindex_atomic!: mem must be getfield(arr,:ref), got :$fld")
                    aid
                elseif mem_ref isa Argument
                    mem_ref.n
                else
                    error("modifyindex_atomic!: cannot trace mem operand $mem_ref to an array arg")
                end
            haskey(lc.arg_vals, arg_id) ||
                error("modifyindex_atomic!: no bound memref for arg $arg_id")
            (lc.arg_vals[arg_id], lc.arg_elem_types[arg_id])
        end

    # Reduction op → memref.atomic_rmw kind keyword.
    op_sym = callee_name(op_ref)
    op_kw = op_sym === :+   ? :add :
            op_sym === :max ? :max :
            op_sym === :min ? :min :
            op_sym === :&   ? :and :
            op_sym === :|   ? :or  :
            error("modifyindex_atomic!: unsupported reduction op $op_sym")
    kind_kw = _atomic_rmw_kind(op_kw, elem_T)
    elem_mlir = mlir_elem_type(elem_T)

    val_v = resolve_value_or_const(lc, val_ref)
    val_v === nothing && error("modifyindex_atomic!: cannot resolve val $val_ref")
    idx_v = resolve_value_or_const(lc, idx_ref)
    idx_v === nothing && error("modifyindex_atomic!: cannot resolve index $idx_ref")

    it = IR.type(idx_v)
    if IR.isvector(it)
        # Per-lane scatter: extract each lane's bucket (1-based) → 0-based →
        # atomic_rmw. `val` is a scalar literal shared by all lanes; if it's a
        # vector, extract per lane too.
        n = Int(size(it, 1))
        idx_elem_t = eltype(it)
        one_attr = _splat_attr(_one_of_elem(idx_elem_t), it)
        one_v = IR.result(_arith.constant(; value=one_attr, result=it))
        idx0 = IR.result(_arith.subi(idx_v, one_v; result=it))   # 0-based vector
        val_is_vec = IR.isvector(IR.type(val_v))
        for lane in 0:(n - 1)
            pos = IR.DenseArrayAttribute(Int64[lane])
            b = IR.result(_vector.extract(idx0, IR.Value[];
                    result=idx_elem_t, static_position=pos))
            v = val_is_vec ?
                IR.result(_vector.extract(val_v, IR.Value[];
                    result=elem_mlir, static_position=pos)) : val_v
            _emit_one_atomic_rmw!(base, b, v, kind_kw, elem_T)
        end
        return nothing
    else
        # Uniform/scalar index. 1-based → 0-based.
        one_v = IR.result(_arith.constant(; value=IR.Attribute(1, it), result=it))
        idx0 = IR.result(_arith.subi(idx_v, one_v; result=it))
        if lc.lane_width > 1
            # Uniform slot: ALL W SPMD lanes run the statement, every one
            # targeting this same slot. Each lane's contribution is a lane of a
            # vector<W> — either already per-lane (`@atomic out[1] op x[i]`) or a
            # uniform scalar broadcast across the W lanes (`@atomic out[1] += c`).
            # Collapse with a single in-register `vector.reduction <op>` then do
            # ONE atomic per block (the only race is *across* blocks).
            #
            # Crucially the scalar case must broadcast FIRST: a uniform `+= c`
            # means each of the W lanes adds `c`, so the reduction must SUM to
            # `W*c` — emitting a single `c` atomic (the obvious-looking shortcut)
            # silently undercounts a thread-counter by a factor of W. For the
            # idempotent ops (max/min/and/or) the broadcast+reduce yields `c`, so
            # they're correct either way; doing it uniformly keeps one code path.
            vt = IR.VectorType(1, Int[lc.lane_width], elem_mlir)
            val_vec = IR.isvector(IR.type(val_v)) ? val_v :
                      IR.result(_vector.broadcast(val_v; vector=vt))
            rkind = _atomic_reduce_kind(op_kw, elem_T)
            kind_attr_red = parse(IR.Attribute, "#vector.kind<$rkind>")
            reduced = IR.result(_vector.reduction(val_vec;
                        dest=elem_mlir, kind=kind_attr_red))
            return _emit_one_atomic_rmw!(base, idx0, reduced, kind_kw, elem_T)
        end
        # GPU SIMT (lane_width==1): one lane per thread, one atomic per thread.
        return _emit_one_atomic_rmw!(base, idx0, val_v, kind_kw, elem_T)
    end
end

# `Base.memoryrefnew(ref_ssa, idx, bc)`
# Flatten a rank-N memref to a contiguous rank-1 view for linear indexing.
# Julia's N-D `A[i,j]` linearises to a SINGLE linear index in the IR (via
# `getfield(A,:size)` + arith), so accesses bottom out in `memoryrefnew(ref,
# linear)`; but the array arg is bound as a rank-N memref (so `memref.dim` can
# still serve per-dim `size(A,d)`). A 1-index gather/load into a rank-N memref
# is invalid ("requires N indices"), so we `reinterpret_cast` to `memref<?×T>`
# of `prod(dims)` elements, stride 1 — valid because Julia arrays are
# contiguous column-major. Rank ≤ 1 is returned unchanged.
function _flatten_memref(base::IR.Value)
    bt = IR.type(base)
    n = ndims(bt)
    n <= 1 && return base
    idx_t = IR.IndexType()
    dimc(d) = IR.result(_memref.dim(base,
        IR.result(_arith.constant(; value=IR.Attribute(d, idx_t))); result=idx_t))
    total = dimc(0)
    for d in 1:(n - 1)
        total = IR.result(_arith.muli(total, dimc(d); result=idx_t))
    end
    elem = eltype(bt)
    # Preserve the source memref's memory space (e.g. `#gpu.address_space<global>`
    # on the GPU path) — reinterpret_cast requires matching spaces.
    memspace = IR.Attribute(MLIR.API.mlirMemRefTypeGetMemorySpace(bt))
    res_t = IR.MemRefType(elem, Int[Int(IR.dynsize())], IR.Attribute(), memspace)
    return IR.result(_memref.reinterpret_cast(
        base, IR.Value[], IR.Value[total], IR.Value[];
        result=res_t,
        static_offsets=IR.DenseArrayAttribute(Int64[0]),
        static_sizes=IR.DenseArrayAttribute(Int64[Int(IR.dynsize())]),
        static_strides=IR.DenseArrayAttribute(Int64[1])))
end

function emit_spmd_memoryrefnew!(lc::LowerCtx, idx::Int, args, @nospecialize(typ))
    ref_ref = args[1]
    # `@localmem` buffer: ref roots at a shared_alloc (workgroup memref), not an
    # arg. Flatten the rank-N alloca to rank-1 for the linear index, same as args.
    if ref_ref isa SSAValue && haskey(lc.local_memrefs, ref_ref.id)
        (alloca, elem_T, _dims) = lc.local_memrefs[ref_ref.id]
        base_memref = _flatten_memref(alloca)
        return _emit_spmd_offset!(lc, idx, args, base_memref, elem_T)
    end
    ref_ref isa SSAValue && haskey(lc.field_refs, ref_ref.id) ||
        error("SPMD memoryrefnew: ref operand must be `getfield(arr, :ref)`, got $ref_ref")
    (arg_id, fld) = lc.field_refs[ref_ref.id]
    fld === :ref || error(
        "SPMD memoryrefnew: ref operand must be `getfield(arr, :ref)` (got :$fld)")
    # N-D array args are bound as rank-N memrefs (for `size`); N-D indexing
    # linearises to one index, so flatten to rank-1 for the gather/scatter/load.
    base_memref = _flatten_memref(lc.arg_vals[arg_id])
    elem_T = lc.arg_elem_types[arg_id]
    return _emit_spmd_offset!(lc, idx, args, base_memref, elem_T)
end

# Shared tail of emit_spmd_memoryrefnew!: record the OffsetInfo (1-based Julia
# index → 0-based) for a resolved (base_memref, elem_T).
function _emit_spmd_offset!(lc::LowerCtx, idx::Int, args, base_memref, elem_T)

    # Index: typically the lane Argument (varying vector). Materialise it; if
    # it's a scalar (uniform load), we'll bypass gather and use a scalar
    # `memref.load`/`memref.store` at the consumer. We still record the
    # OffsetInfo with the (possibly scalar) Value.
    idx_v = resolve_value_or_const(lc, args[2])
    idx_v === nothing &&
        error("SPMD memoryrefnew: cannot resolve index operand $(args[2])")

    # SPMD memoryrefnew indices are 1-based Julia (raw `i`). Convert to 0-based
    # for downstream `memref` ops by subtracting 1 splat'd into the vector.
    t = IR.type(idx_v)
    if IR.isvector(t)
        elem_t = eltype(t)
        n = Int(size(t, 1))
        one_attr = _splat_attr(_one_of_elem(elem_t), t)
        one_v = IR.result(_arith.constant(; value=one_attr, result=t))
        idx_v = IR.result(_arith.subi(idx_v, one_v; result=t))
        sh = Int[n]
    else
        one_v = IR.result(_arith.constant(; value=IR.Attribute(1, t), result=t))
        idx_v = IR.result(_arith.subi(idx_v, one_v; result=t))
        sh = Int[]
    end
    lc.offsets[idx] = OffsetInfo(base_memref, idx_v, elem_T, sh)
    return nothing
end

# Return a 1 of the given MLIR element type (as a Julia value usable in a
# DenseElements splat — we map i32/i64/index to plain Int, and floats to 1.0
# of the matching width).
function _one_of_elem(elem_t)
    elem_t == IR.Type(Int32) && return Int32(1)
    elem_t == IR.Type(Int64) && return Int64(1)
    elem_t == IR.IndexType() && return Int(1)
    elem_t == IR.Type(Float32) && return 1f0
    elem_t == IR.Type(Float64) && return 1.0
    elem_t == IR.Type(Bool) && return true
    return Int(1)
end

# `Base.memoryrefget(mr, :not_atomic, bc)` → vector.gather / vector.transfer_read
function emit_spmd_memoryrefget!(lc::LowerCtx, args, @nospecialize(typ))
    mr_ref = args[1]
    mr_ref isa SSAValue && haskey(lc.offsets, mr_ref.id) ||
        error("SPMD memoryrefget: mr operand must be a tracked memoryrefnew SSA, got $mr_ref")
    off = lc.offsets[mr_ref.id]
    if isempty(off.idx_shape)
        # Scalar load — `memref.load %base[%idx]`.
        idx_index = cast_to_index(off.indices)
        return IR.result(_memref.load(off.base, IR.Value[idx_index];
                                       result=mlir_elem_type(off.elem_type)))
    end

    n = off.idx_shape[1]
    result_t = IR.VectorType(1, Int[n], mlir_elem_type(off.elem_type))

    if _is_contiguous_lane_index_from_offset(lc, off)
        # Fast path: lane is contiguous (`i = bid*W + 1..W`). Use
        # `vector.transfer_read` for a wide contiguous load.
        base_off = _spmd_lane_base(lc)
        pad = IR.result(_arith.constant(;
            value=IR.Attribute(zero(off.elem_type))))
        perm = IR.Attribute(IR.IdentityAffineMap(1))
        inb  = IR.Attribute(IR.Attribute[IR.Attribute(true)])
        return IR.result(_vector.transfer_read(
            off.base, IR.Value[base_off], pad;
            vector=result_t, permutation_map=perm, in_bounds=inb,
        ))
    end

    # Gather path. mask = all-true; pass_thru = zeros.
    idx_t = IR.IndexType()
    c0 = IR.result(_arith.constant(; value=IR.Attribute(Int(0), idx_t)))
    mask_v = _resolve_or_alltrue_mask(lc, nothing, off.idx_shape)
    pad_v = _resolve_or_zero_passthru(lc, nothing, off.elem_type, off.idx_shape)
    return IR.result(_vector.gather(off.base, IR.Value[c0], off.indices,
                                     mask_v, pad_v; result=result_t))
end

# `Base.memoryrefset!(mr, value, :not_atomic, bc)`
function emit_spmd_memoryrefset!(lc::LowerCtx, args, @nospecialize(typ))
    mr_ref = args[1]
    mr_ref isa SSAValue && haskey(lc.offsets, mr_ref.id) ||
        error("SPMD memoryrefset!: mr operand must be a tracked memoryrefnew SSA, got $mr_ref")
    off = lc.offsets[mr_ref.id]
    val_v = resolve_value_or_const(lc, args[2])
    val_v === nothing &&
        error("SPMD memoryrefset!: cannot resolve value operand $(args[2])")

    if isempty(off.idx_shape)
        # Scalar store.
        idx_index = cast_to_index(off.indices)
        _memref.store(val_v, off.base, IR.Value[idx_index])
        return nothing
    end

    # If the value is scalar but the index is varying, broadcast.
    if !IR.isvector(IR.type(val_v))
        n = off.idx_shape[1]
        vec_t = IR.VectorType(1, Int[n], mlir_elem_type(off.elem_type))
        val_v = _broadcast_to_match(val_v, vec_t)
    end

    if _is_contiguous_lane_index_from_offset(lc, off)
        # Fast path: vector.transfer_write at `bid * W`.
        base_off = _spmd_lane_base(lc)
        perm = IR.Attribute(IR.IdentityAffineMap(1))
        inb  = IR.Attribute(IR.Attribute[IR.Attribute(true)])
        _vector.transfer_write(val_v, off.base, IR.Value[base_off];
                                permutation_map=perm, in_bounds=inb)
        return nothing
    end

    # Scatter path.
    idx_t = IR.IndexType()
    c0 = IR.result(_arith.constant(; value=IR.Attribute(Int(0), idx_t)))
    mask_v = _resolve_or_alltrue_mask(lc, nothing, off.idx_shape)
    _vector.scatter(off.base, IR.Value[c0], off.indices, mask_v, val_v)
    return nothing
end

# Detect the contiguous-lane pattern *from an OffsetInfo*. Because
# `emit_spmd_memoryrefnew!` subtracted 1 from the raw lane index to convert
# 1-based → 0-based, the offset's `indices` Value won't `==` the original
# lane vector. Instead we compare against the 0-based lane vector. Simplest
# robust heuristic: see if the indices was produced by an `arith.subi` whose
# LHS is the lane arg. For the MVP we use a structural check via Reactant's
# IR API.
function _is_contiguous_lane_index_from_offset(lc::LowerCtx, off::OffsetInfo)
    lc.spmd || return false
    haskey(lc.arg_vals, lc.lane_arg) || return false
    # Heuristic: walk back from off.indices through one `arith.subi` op to
    # see if its LHS matches the lane arg vector. Reactant's IR exposes the
    # defining op via `IR.owner` on a Value (when the value is an op result).
    try
        owner = IR.owner(off.indices)
        owner === nothing && return false
        # The owner op should be arith.subi with operand 0 equal to the lane
        # arg vector.
        name = String(IR.name(owner))
        name == "arith.subi" || return false
        n = IR.noperands(owner)
        n >= 1 || return false
        op0 = IR.operand(owner, 1)
        return op0 == lc.arg_vals[lc.lane_arg]
    catch
        return false
    end
end
