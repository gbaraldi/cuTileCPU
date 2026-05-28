# StructuredIRCode → MLIR module.
#
# Pipeline:
#   Julia kernel + argtypes
#     → ct.code_structured(f, argtypes; optimize=true)  [cuTile: target-agnostic]
#         runs canonicalize → constprop → FMA fusion → CSE → alias
#         → token order → RNG lowering → LICM → divisibility/bounds
#         → no-wrap → DCE; we receive the optimized SCI.
#     → lower_to_mlir(sci, argtypes)                    [this file]
#         emits scf/arith/memref/vector/func dialect MLIR with alignment +
#         strided-layout info from ArraySpec.
#
# Argument model:
#   Each TileArray<T,N,Spec> becomes ONE `memref<?x...xT, strided<[…]>>` arg.
#   The flat (ptr, sizes…, strides…) destructuring the bytecode target uses is
#   *not* done here — the memref descriptor ABI carries that information.
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

# A PartitionView in cuTile = a memref + per-block tile shape (+ elem type).
# When a load happens at index (%bid_1, …, %bid_N), the memref offset along
# axis k is `%bid_k * tile_shape[k]`. We carry the source memref + tile shape
# + element type so load/store sites can synthesize offsets + correct vector
# types.
struct PartitionInfo
    base::IR.Value
    tile_shape::Vector{Int}   # Julia (col-major) order
    elem_type::Type
end

# A TensorView in cuTile = (ptr, sizes, strides) from a TileArray, plus the
# element type. Tracked-only (no IR emitted) — the downstream partition_view
# reads `.base` (memref Value) and `.elem_type` to build PartitionInfo.
struct TensorViewInfo
    base::IR.Value
    elem_type::Type
end

# An "offset view" produced by `Intrinsics.offset(ptr_tile, indices_tile)`.
# Tracked-only — consumed by `Intrinsics.load_ptr_tko` (gather) and
# `Intrinsics.store_ptr_tko` (scatter). `base` is the source memref (resolved
# via the ptr field-ref), `indices` is the per-lane i32 (or iK) index vector,
# `elem_type` is the source element type, and `idx_shape` is the tile shape
# of the indices (Julia col-major).
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
    tensor_views::Dict{Int, TensorViewInfo}
    partitions::Dict{Int, PartitionInfo}
    offsets::Dict{Int, OffsetInfo}             # SSA → gather/scatter pointer-tile
    field_refs::Dict{Int, Tuple{Int, Symbol}}  # SSA → (arg id, fieldname)
    tuples::Dict{Int, Vector{Any}}             # SSA → component refs
    # SSAs that are sentinels (e.g. boundscheck) — extract to a marker.
    sentinels::Dict{Int, Symbol}
    # Per-arg element types — sci.argtypes[slot] for TileArray slots.
    arg_elem_types::Dict{Int, Type}
    # Grid bid Values, one per grid dim, x/y/z order.
    bids::Vector{IR.Value}
    # Grid extents, one per grid dim. Each is an i32 Value (already
    # `index_cast` from the `index`-typed grid param).
    grid_extents_i32::Vector{IR.Value}
    # SSAs that resolve to `KernelState` (from `Intrinsics.kernel_state()`).
    # `getfield(_, :seed)` on these returns `seed_param` below.
    kernel_state_ssas::Set{Int}
    # MLIR Value of the i32 `KernelState.seed` parameter, bound at func entry.
    seed_param::Union{Nothing, IR.Value}
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
end

LowerCtx(ctx, mod) = LowerCtx(
    ctx, mod,
    Dict{Int, IR.Value}(), Dict{Int, IR.Value}(), Dict{Int, Any}(),
    Dict{Int, IR.Value}(), Dict{Int, Vector{IR.Value}}(),
    Dict{Int, TensorViewInfo}(), Dict{Int, PartitionInfo}(),
    Dict{Int, OffsetInfo}(),
    Dict{Int, Tuple{Int, Symbol}}(), Dict{Int, Vector{Any}}(),
    Dict{Int, Symbol}(),
    Dict{Int, Type}(),
    IR.Value[], IR.Value[],
    Set{Int}(), nothing,
    false, 16, 0, Int64)

# ----------------------------------------------------------------------------
# Type translation: Julia → MLIR
# ----------------------------------------------------------------------------

function mlir_elem_type(T::Type)
    T === Float32  && return IR.Type(Float32)
    T === Float64  && return IR.Type(Float64)
    T === Float16  && return IR.Type(Float16)
    T === BFloat16 && return IR.Type(BFloat16)
    T === Int32    && return IR.Type(Int32)
    T === Int64    && return IR.Type(Int64)
    # MLIR uses signless integer types: UInt32/UInt64 map to the same i32/i64
    # as their signed counterparts (signedness is a per-op attribute).
    T === UInt32   && return IR.Type(Int32)
    T === UInt64   && return IR.Type(Int64)
    T === Bool     && return IR.Type(Bool)
    error("cuTileCPU: unsupported element type $T")
end

# Build a splat DenseElements attribute for a vector type. Reactant exposes
# `Base.fill(::T, shaped_type)` overloads only for {Bool, Int8/32/64,
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

# MLIR vector type from a cuTile Julia (col-major) tile shape + Julia element
# type. If the shape is empty (`()` — a 0-D / scalar tile), returns the scalar
# elem MLIR type instead of `vector<f32>` (which isn't a valid MLIR type).
function mlir_tile_type(shape::Tuple, T::Type)
    isempty(shape) && return mlir_elem_type(T)
    elem = mlir_elem_type(T)
    return IR.VectorType(length(shape), reverse(collect(Int, shape)), elem)
end
mlir_tile_type(shape::AbstractVector{<:Integer}, T::Type) =
    mlir_tile_type(Tuple(shape), T)

# Julia tile-element type for a cuTile-style tile Julia type (Tile{T,Shape},
# IntTile{Shape,T}, FloatTile{Shape,T}, Tile{Bool,Shape}). Returns `nothing` if
# `T` is not a tile.
function tile_eltype(@nospecialize(T))
    T isa DataType || return nothing
    # cuTile.Tile{T,Shape}, FloatTile, IntTile
    if T <: ct.Tile
        return T.parameters[1]
    end
    return nothing
end

# Julia tile-shape tuple for a cuTile tile type. Returns `()` for scalar tiles.
function tile_shape(@nospecialize(T))
    T isa DataType || return nothing
    if T <: ct.Tile
        S = T.parameters[2]
        S isa DataType && S <: Tuple || return nothing
        return S.parameters
    end
    return nothing
end

# MLIR vector type for a cuTile Tile Julia type. For 0-D tiles returns the
# scalar element type.
function mlir_type_for_tile(@nospecialize(T))
    eT = tile_eltype(T)
    eT === nothing && return nothing
    shape = tile_shape(T)
    shape === nothing && return nothing
    return mlir_tile_type(Tuple(shape), eT)
end

# Map any cuTile / Julia type to an MLIR type (scalar or vector).
function mlir_type_for(@nospecialize(T))
    if T isa DataType && T <: ct.Tile
        v = mlir_type_for_tile(T)
        v === nothing || return v
    end
    T isa Type && T <: Number && return mlir_elem_type(T)
    error("cuTileCPU: cannot map type $T to MLIR")
end

# TileArray{T,N,Spec()} → memref<?x?xT, strided<[…]>>. When `spec.contiguous`,
# encode unit stride on the innermost (col-major) dim — gives LLVM a
# compile-time proof of contiguous access.
function mlir_memref_for_tilearray(TA::Type)
    @assert TA <: ct.TileArray "expected TileArray, got $TA"
    eT = ct.eltype(TA)
    N  = ndims(TA)
    spec = TA.parameters[3]  # ArraySpec singleton instance
    elem = mlir_elem_type(eT)
    shape = fill(Int(IR.dynsize()), N)
    layout = if spec.contiguous
        # Julia col-major: dim 1 fastest. MLIR memref row-major: last dim fastest.
        # So Julia dim 1 maps to MLIR dim N. Unit stride goes there; rest dynamic.
        strides_mlir = Vector{String}(undef, N)
        for k in 1:N
            mlir_dim = N - k + 1
            strides_mlir[mlir_dim] = (k == 1) ? "1" : "?"
        end
        parse(IR.Attribute, "strided<[" * join(strides_mlir, ", ") * "]>")
    else
        IR.Attribute()
    end
    return IR.MemRefType(elem, shape, layout, IR.Attribute()), spec
end

spec_alignment(spec) = Int(spec.alignment)

# ----------------------------------------------------------------------------
# Divisibility annotations on TileArray kernel args
# ----------------------------------------------------------------------------
#
# At func entry we already emit `memref.assume_alignment %arg, N` (covers the
# base pointer / leading dim). The non-leading dims have dynamic (`?`) strides
# in the memref's `strided<…>` layout — MLIR's strided-layout attribute can't
# carry divisibility info, so the alignment proof for vector loads / stores
# along those dims can't come from the type. The bytecode path solves this
# with `AssumeOp(DivBy(n))` predicates derived from cuTile's
# divisibility/bounds analyses; here we mirror that idea by emitting
# `llvm.intr.assume((stride % n) == 0)` on each stride, which the
# `MemorySSA`-aware passes downstream of `mlir-translate --mlir-to-llvmir`
# fold into the same vectorizer alignment fact.
#
# Input chain comes from `ct.arg_chain(argT, [3, i])` — the spec-only
# kernel-arg chain (an upper bound on what any consumer would derive); the
# dataflow results are queried at consumer sites, not at entry. This matches
# what cuTile's `apply_arg_assume_predicates!` does for the bytecode target.

"""
    emit_llvm_intr_assume!(cond)

Emit `llvm.intr.assume %cond` (no result, side-effecting). Reactant's
`Dialects.llvm` binding doesn't yet wrap the intrinsic, so we go through
`create_operation` directly.
"""
function emit_llvm_intr_assume!(cond::IR.Value)
    # llvm.intr.assume takes the condition followed by zero or more op-bundle
    # operands. The empty-bundle case still requires the `op_bundle_sizes`
    # DenseI32 array attribute and an `operandSegmentSizes` of [1, 0]
    # (cond, no bundle operands).
    obs = parse(IR.Attribute, "array<i32>")
    opseg = Dialects.operandsegmentsizes([1, 0])
    IR.create_operation(
        "llvm.intr.assume",
        IR.Location();
        operands  = IR.Value[cond],
        owned_regions = IR.Region[],
        successors = IR.Block[],
        attributes = IR.NamedAttribute[
            IR.NamedAttribute("op_bundle_sizes", obs),
            opseg,
        ],
        results   = IR.Type[],
        result_inference = false,
    )
    return nothing
end

"""
    emit_stride_divby_assumes!(memref_val, argT, N) -> Int

Emit `llvm.intr.assume((stride % n) == 0)` for each non-leading stride dim
of a TileArray-typed memref operand whose `arg_chain(argT, [3, i])` yields
a `DivBy(n)` with `n > 1`. `N` is the memref's rank.

Returns the number of stride dims annotated (for reporting / test
introspection). Quiet when no spec divisibility info applies.

Note: arg path `[3, i]` indexes strides in Julia order (dim 1 = fastest,
contiguous). MLIR memref strides are returned in row-major order
(dim 0 = slowest), so Julia stride dim `i` is MLIR stride dim `N - i + 1`.
"""
function emit_stride_divby_assumes!(memref_val::IR.Value, argT::Type, N::Int)
    argT <: ct.TileArray || return 0

    # Determine which (Julia-dim) stride paths carry a non-trivial DivBy.
    # `arg_chain` returns Vector{AssumePredicate}; we project to the DivBy
    # divisor (n > 1) and drop chains that only carry Bounded predicates
    # (Bounded on a stride doesn't help the vectorizer).
    divisors = Int[]
    julia_dims = Int[]
    for i in 1:N
        chain = ct.arg_chain(argT, Int[3, i])
        d = 1
        for p in chain
            if p isa ct.DivBy
                d = max(d, p.divisor)
            end
        end
        if d > 1
            push!(divisors, d)
            push!(julia_dims, i)
        end
    end
    isempty(divisors) && return 0

    # Result types for memref.extract_strided_metadata. base_buffer is a
    # rank-0 memref<elemT>; offset / sizes / strides are `index`.
    idx_t = IR.IndexType()
    elem_t = mlir_elem_type(ct.eltype(argT))
    base_t = IR.MemRefType(elem_t, Int[], IR.Attribute(), IR.Attribute())
    sizes_t = IR.Type[idx_t for _ in 1:N]
    strides_t = IR.Type[idx_t for _ in 1:N]

    md_op = _memref.extract_strided_metadata(
        memref_val;
        base_buffer = base_t,
        offset = idx_t,
        sizes = sizes_t,
        strides = strides_t,
    )
    # Result layout: [base_buffer, offset, sizes..., strides...]
    base_off = 0
    offset_off = 1
    sizes_off = 2
    strides_off = 2 + N

    # arith.cmpi predicate `eq` is i64 code 0.
    eq_attr = IR.Attribute(0, IR.Type(Int64))

    count = 0
    for (k, n) in zip(julia_dims, divisors)
        # Julia dim k → MLIR dim (N - k + 1), 1-based; → 0-based result idx.
        mlir_dim = N - k + 1
        stride_val = IR.result(md_op, strides_off + mlir_dim)
        c_n  = IR.result(_arith.constant(; value = IR.Attribute(Int(n), idx_t)))
        c_0  = IR.result(_arith.constant(; value = IR.Attribute(Int(0), idx_t)))
        rem_ = IR.result(_arith.remui(stride_val, c_n))
        cond = IR.result(_arith.cmpi(rem_, c_0; predicate = eq_attr))
        emit_llvm_intr_assume!(cond)
        count += 1
    end
    return count
end

# ----------------------------------------------------------------------------
# Top-level entry: lower one SCI to an MLIR module
# ----------------------------------------------------------------------------

"""
    lower_to_mlir(sci, argtypes; kernel_name, n_grid_dims=1)
        -> (IR.Module, param_julia_types::Vector{Type}, IR.Context)

Walks `sci` and produces an MLIR module containing
`func.func @<kernel_name>` ready for the standard CPU lowering pipeline.

`argtypes` should be the tuple type originally handed to `ct.code_structured`.
`Const`-seeded slots in `sci.argtypes` contribute no func parameter — their
value is inlined wherever referenced.

Returns the module, the Julia types of the corresponding func parameters
(for host descriptor packing), and the MLIR context (kept alive so the module
remains valid).
"""
function lower_to_mlir(sci::StructuredIRCode, argtypes::Type;
                       kernel_name::String, n_grid_dims::Int=1,
                       divby_info=nothing, bounds_info=nothing)
    ctx = fresh_context()
    mod_ref = Ref{IR.Module}()
    param_julia_types = Type[]
    param_kinds = Symbol[]   # parallel to param_julia_types: :memref or :scalar

    @with_context ctx begin
        IR.load_all_available_dialects()
        mod = IR.Module(IR.Location())
        mod_ref[] = mod

        @with_module mod begin
            lc = LowerCtx(ctx, mod)

            idx_t = IR.IndexType()
            param_mlir_types = IR.Type[]
            # Each entry: (slot, kind, [spec])
            param_arg_slots = Int[]
            param_specs = Any[]     # parallel; only meaningful for :memref slots

            for (i, AT) in enumerate(sci.argtypes)
                i == 1 && continue
                if AT isa Core.Const
                    lc.arg_const[i] = AT.val
                    continue
                end
                AT_wide = widenconst(AT)
                if AT_wide <: ct.TileArray
                    mr, spec = mlir_memref_for_tilearray(AT_wide)
                    push!(param_mlir_types, mr)
                    push!(param_arg_slots, i)
                    push!(param_julia_types, AT_wide)
                    push!(param_kinds, :memref)
                    push!(param_specs, spec)
                    lc.arg_elem_types[i] = ct.eltype(AT_wide)
                elseif AT_wide <: Number
                    push!(param_mlir_types, mlir_elem_type(AT_wide))
                    push!(param_arg_slots, i)
                    push!(param_julia_types, AT_wide)
                    push!(param_kinds, :scalar)
                    push!(param_specs, nothing)
                else
                    error("cuTileCPU: unsupported arg type $AT_wide at slot $i")
                end
            end

            # Trailing implicit `KernelState.seed` parameter (i32). Always
            # emitted, regardless of whether the kernel actually uses RNG —
            # mirrors cuTile's bytecode codegen, which puts the `KernelState`
            # flat params right after the user args. LLVM's IPO drops it for
            # kernels that don't reference `Intrinsics.kernel_state()`. The
            # host-side launch always passes a fresh `rand(UInt32)` here, so
            # consecutive launches see distinct seeds.
            seed_param_idx = length(param_mlir_types) + 1
            push!(param_mlir_types, mlir_elem_type(UInt32))

            grid_param_offset = length(param_mlir_types)
            for _ in 1:n_grid_dims
                push!(param_mlir_types, idx_t)
            end

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
            end
            # Bind the implicit `KernelState.seed` param. `Intrinsics.kernel_state()`
            # and the subsequent `getfield(_, :seed)` chain resolves to this Value.
            lc.seed_param = IR.argument(entry, seed_param_idx)
            grid_vals = IR.Value[IR.argument(entry, grid_param_offset + d)
                                 for d in 1:n_grid_dims]

            @with_block entry begin
                c0 = IR.result(_arith.constant(; value=IR.Attribute(Int(0), idx_t)))
                c1 = IR.result(_arith.constant(; value=IR.Attribute(Int(1), idx_t)))

                # Memref args: propagate ArraySpec.alignment via
                # memref.assume_alignment. Scalar args: bind directly.
                for (k, slot) in enumerate(param_arg_slots)
                    if param_kinds[k] === :memref
                        align = spec_alignment(param_specs[k])
                        align_attr = IR.Attribute(align, IR.Type(Int32))
                        # `memref.assume_alignment` is a side-effecting op
                        # with no result on MLIR 18 (the result-returning
                        # form was added in MLIR 21). Emit the assume and
                        # keep using the raw arg Value.
                        _memref.assume_alignment(
                            raw_arg_vals[slot]; alignment=align_attr,
                        )
                        lc.arg_vals[slot] = raw_arg_vals[slot]
                        # Per-stride divisibility annotations. Plays the
                        # role of cuTile's `apply_arg_assume_predicates!`
                        # for the bytecode target: gives the vectorizer
                        # an alignment proof for non-leading dims (the
                        # leading dim is already covered by the strided
                        # `<[1, …]>` layout). `divby_info` / `bounds_info`
                        # carry the consumer-side dataflow facts; they're
                        # accepted by `lower_to_mlir` for future use but
                        # the entry-time chain is spec-only via
                        # `arg_chain`.
                        argT = widenconst(sci.argtypes[slot])
                        N = ndims(argT)
                        emit_stride_divby_assumes!(raw_arg_vals[slot], argT, N)
                    else
                        lc.arg_vals[slot] = raw_arg_vals[slot]
                    end
                end

                # Pre-cast grid extents to i32 for use by
                # `Intrinsics.get_num_tile_blocks` (cuTile types these as Int32).
                for gv in grid_vals
                    push!(lc.grid_extents_i32,
                          IR.result(_arith.index_cast(gv; out=IR.Type(Int32))))
                end

                par_region = IR.Region()
                par_block = IR.Block([idx_t for _ in 1:n_grid_dims],
                                     [IR.Location() for _ in 1:n_grid_dims])
                push!(par_region, par_block)
                _scf.parallel(
                    IR.Value[c0 for _ in 1:n_grid_dims],
                    grid_vals,
                    IR.Value[c1 for _ in 1:n_grid_dims],
                    IR.Value[];
                    results=IR.Type[], region=par_region,
                )

                @with_block par_block begin
                    for d in 1:n_grid_dims
                        push!(lc.bids, IR.argument(par_block, d))
                    end
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
# SPMD mode: lower a scalar-typed Julia kernel to vector MLIR
# ----------------------------------------------------------------------------
#
# Compiles a Julia function of shape
#
#     function k(arr1::Vector{T1}, ..., arrK::Vector{TK}, i::Int)
#         @inbounds arr1[i] = arr2[i] + arr3[i]   # plain Julia, no Tile/ct.*
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
                    # what the cuTile path emits to get aligned vector
                    # loads at DRAM scale.
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
                    error("cuTileCPU SPMD: unsupported arg type $AT_wide at slot $i")
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
#                                                              # __cutilecpu_spmd_lane_id()
#             @inbounds C[i] = A[i] + B[i]
#         end
#     end
#
# The first arg is the KA `__ctx__` (a `CompilerMetadata{…}` struct). With the
# overlay set up in `ext/KernelAbstractionsExt.jl`, the body never reads any
# field of `__ctx__` — every reference to it has been folded to either a
# constant or a call to the sentinel function `__cutilecpu_spmd_lane_id()`.
# We therefore don't materialise the ctx as an MLIR parameter at all; we just
# use its arg slot as the SPMD `lane_arg` so the existing walker
# infrastructure (the lane vector, `__cutilecpu_spmd_lane_id` clause) lights
# up unchanged.
function lower_to_mlir_ka(sci::StructuredIRCode, argtypes::Type;
                          kernel_name::String, lane_width::Int=16,
                          alignment::Int=16, lane_idx_type::Type=Int64)
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
                    error("cuTileCPU KA: unsupported arg type $AT_wide at slot $i " *
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
# `gid` is still tracked via `lc.lane_arg`, and the
# `__cutilecpu_spmd_lane_id` sentinel clause returns `lc.arg_vals[lane_arg]`
# unchanged — so a KA `__index_Global_Linear` overlay routed through this
# entrypoint also works (with `__validindex` providing the `gid <= n`
# guard or `true` for exact-multiple launches).
# `ctx_arg`: when set, the lane comes from a *non*-trailing arg — the slot
# is treated like the KA `__ctx__` (skipped as a func param, its value is
# the synthesized global thread index via the `__cutilecpu_spmd_lane_id`
# sentinel). This is how a KernelAbstractions `gpu_*` body lowers: the
# `__index_Global_Linear(ctx)` overlay rewrites to the sentinel, and the
# ctx itself is never referenced as data. When `ctx_arg === nothing` we
# use the plain-Julia shape (trailing Integer `gid`).
function lower_to_mlir_gpu(sci::StructuredIRCode, argtypes::Type;
                           kernel_name::String, module_name::String="kernels",
                           lane_idx_type::Type=Int32,
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
                if AT_wide <: AbstractArray
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
                    error("cuTileCPU GPU: unsupported arg type $AT_wide at slot $i")
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
    elseif stmt isa cuTile.MakeTokenNode
        # No memory ordering on CPU MVP; tokens are dropped. Consumers ignore
        # the token operand.
        return nothing
    elseif stmt isa cuTile.TokenResultNode
        # Memory-op result token; no SSA Value needed (the CPU path doesn't
        # thread tokens through). Drop.
        return nothing
    elseif stmt isa cuTile.JoinTokensNode
        # Token merge for ordering; not modelled on the CPU path. Drop.
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
    end
    error("cuTileCPU.walk_stmt!: unhandled stmt $stmt at %$idx (typ=$typ)")
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
    end
    error("cuTileCPU.walk_expr!: unhandled Expr head :$(e.head) at %$idx ($e)")
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
        # Module-level binding (e.g. `cuTile.PHILOX_W`). If it resolves to a
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
    if T isa DataType && T <: ct.Tile
        eT = tile_eltype(T)
        sh = Tuple(tile_shape(T))
        if isempty(sh)
            return materialise_zero_scalar(eT)
        end
        vec_t = mlir_tile_type(sh, eT)
        splat = _splat_attr(zero(eT), vec_t)
        return IR.result(_arith.constant(; value=splat, result=vec_t))
    elseif T isa Type && T <: Number
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

# Resolve a load_/store_partition_view index-tuple argument. Inference may
# leave this either as
#   • an SSAValue referring to a `tuple(...)` call we tracked in `lc.tuples`, or
#   • a literal Julia Tuple{...} (when every index folded to a constant — e.g.
#     `ct.load(K; index=(1, 1), ...)` becomes `(0, 0)` after the 1-based-to-
#     0-based normalisation).
# Returns a Vector{Any} ready to feed `emit_tile_load!`/`emit_tile_store!`.
function _resolve_index_tuple(lc::LowerCtx, idx_arg, errmsg::String)
    if idx_arg isa SSAValue
        haskey(lc.tuples, idx_arg.id) || error(errmsg)
        return lc.tuples[idx_arg.id]
    elseif idx_arg isa Tuple
        return collect(Any, idx_arg)
    end
    error(errmsg)
end

function walk_call!(lc::LowerCtx, idx::Int, @nospecialize(callee),
                    args::Vector{Any}, @nospecialize(typ))
    fname = callee_name(callee)

    if fname === :kernel_state
        # `Intrinsics.kernel_state()` returns the host-supplied `KernelState`
        # struct (single `seed::UInt32` field). cuTile's bytecode codegen
        # binds this to the lazy arg-ref for the implicit trailing arg slot;
        # we instead tag the SSA so the immediate `getfield(_, :seed)` resolves
        # to the seed param. No IR is emitted for the intrinsic itself.
        push!(lc.kernel_state_ssas, idx)
        return nothing

    elseif fname === :get_tile_block_id
        axis = something(resolve_const(lc, args[1]), args[1])
        axis isa Integer ||
            error("get_tile_block_id: axis must be const, got $(args[1])")
        ax = Int(axis)
        i32_t = IR.Type(Int32)
        # cuTile's `rng_key` reads axes 0..2 unconditionally even on a 1-D
        # grid — beyond the actual grid the device-side semantics yield 0.
        # Return constant Int32(0) for axes beyond our runtime grid rank.
        if ax + 1 > length(lc.bids)
            return IR.result(_arith.constant(; value=IR.Attribute(Int32(0))))
        end
        # bid is `index` in MLIR but typed as Int32 in cuTile. Cast so
        # subsequent arithmetic (addi/cmpi/exti) sees a matching i32 type.
        # Tile-load index ops re-cast back via `cast_to_index`.
        bid_idx = lc.bids[ax + 1]
        return IR.result(_arith.index_cast(bid_idx; out=i32_t))

    elseif fname === :make_tensor_view
        # args = [TileArrayType, ptr_getfield, sizes_getfield, strides_getfield]
        # The source arg is recovered from the ptr getfield's tracked refs.
        ptr_arg = args[2]
        ptr_arg isa SSAValue && haskey(lc.field_refs, ptr_arg.id) ||
            error("make_tensor_view: cannot resolve source from ptr operand $ptr_arg")
        src_arg = lc.field_refs[ptr_arg.id][1]
        memref_val = lc.arg_vals[src_arg]
        elem_type = lc.arg_elem_types[src_arg]
        lc.tensor_views[idx] = TensorViewInfo(memref_val, elem_type)
        return nothing

    elseif fname === :make_partition_view
        # args = [TensorView, tile_shape_tuple, padding_mode, ...]
        src_view = args[1]
        src_view isa SSAValue ||
            error("make_partition_view: expected SSAValue source, got $src_view")
        tv = lc.tensor_views[src_view.id]
        tile_shape = something(resolve_const(lc, args[2]), args[2])
        tile_shape isa Tuple ||
            error("make_partition_view: tile shape must be const tuple, got $tile_shape")
        lc.partitions[idx] = PartitionInfo(tv.base, Int[tile_shape...], tv.elem_type)
        return nothing

    elseif fname === :load_partition_view
        part_ssa = args[1]
        part_ssa isa SSAValue || error("load_partition_view: bad partition operand")
        part = lc.partitions[part_ssa.id]
        components = _resolve_index_tuple(lc, args[4],
                                          "load_partition_view: bad index-tuple operand")
        return emit_tile_load!(lc, part, components)

    elseif fname === :store_partition_view
        part_ssa = args[1]
        part_ssa isa SSAValue || error("store_partition_view: bad partition operand")
        part = lc.partitions[part_ssa.id]
        val_v = resolve_value(lc, args[2])
        val_v !== nothing ||
            error("store_partition_view: cannot resolve value to store")
        components = _resolve_index_tuple(lc, args[5],
                                          "store_partition_view: bad index-tuple operand")
        emit_tile_store!(lc, part, val_v, components)
        return nothing

    elseif fname === :addf
        return emit_binop_value!(lc, args, _arith.addf)

    elseif fname === :subf
        return emit_binop_value!(lc, args, _arith.subf)

    elseif fname === :mulf
        return emit_binop_value!(lc, args, _arith.mulf)

    elseif fname === :divf
        return emit_binop_value!(lc, args, _arith.divf)

    elseif fname === :exp
        return emit_unary_math!(lc, args, _math.exp, typ)

    elseif fname === :rsqrt
        return emit_unary_math!(lc, args, _math.rsqrt, typ)

    elseif fname === :sqrt
        return emit_unary_math!(lc, args, _math.sqrt, typ)

    elseif fname === :exp2
        return emit_unary_math!(lc, args, _math.exp2, typ)

    elseif fname === :log
        return emit_unary_math!(lc, args, _math.log, typ)

    elseif fname === :log2
        return emit_unary_math!(lc, args, _math.log2, typ)

    elseif fname === :sin
        return emit_unary_math!(lc, args, _math.sin, typ)

    elseif fname === :cos
        return emit_unary_math!(lc, args, _math.cos, typ)

    elseif fname === :tan
        return emit_unary_math!(lc, args, _math.tan, typ)

    elseif fname === :tanh
        return emit_unary_math!(lc, args, _math.tanh, typ)

    elseif fname === :sinh
        return emit_unary_math!(lc, args, _math.sinh, typ)

    elseif fname === :cosh
        return emit_unary_math!(lc, args, _math.cosh, typ)

    elseif fname === :floor
        return emit_unary_math!(lc, args, _math.floor, typ)

    elseif fname === :ceil
        return emit_unary_math!(lc, args, _math.ceil, typ)

    elseif fname === :absf
        return emit_unary_math!(lc, args, _math.absf, typ)

    elseif fname === :absi
        return emit_unary_math!(lc, args, _math.absi, typ)

    elseif fname === :maxf
        # Intrinsics.maxf(a, b) — element-wise floating-point max. Same vector
        # shape on both operands. Lowers to arith.maxnumf (NaN-quieting form,
        # matches Base.max semantics).
        return emit_binop_value!(lc, args, _arith.maxnumf)

    elseif fname === :minf
        return emit_binop_value!(lc, args, _arith.minnumf)

    elseif fname === :maxi
        # Intrinsics.maxi(a, b, signedness) — signed/unsigned int max.
        return emit_maxmin_int!(lc, args, _arith.maxsi, _arith.maxui)

    elseif fname === :mini
        return emit_maxmin_int!(lc, args, _arith.minsi, _arith.minui)

    elseif fname === :fma
        # Intrinsics.fma(x, y, z) — fused multiply-add x*y + z. Same shape on
        # all three operands.
        return emit_fma!(lc, args, typ)

    elseif fname === :addi
        return emit_binop_value!(lc, args, _arith.addi)

    elseif fname === :subi
        return emit_binop_value!(lc, args, _arith.subi)

    elseif fname === :muli
        return emit_binop_value!(lc, args, _arith.muli)

    elseif fname === :xori
        return emit_binop_value!(lc, args, _arith.xori)

    elseif fname === :ori
        return emit_binop_value!(lc, args, _arith.ori)

    elseif fname === :andi
        return emit_binop_value!(lc, args, _arith.andi)

    elseif fname === :mulhii
        # cuTile Intrinsics.mulhii(a, b) — high-32 bits of the unsigned-widened
        # product. Use `arith.muli` with overflow=nuw on the extui-widened
        # operands and shift right by the operand bit width. (MLIR has no
        # direct mulhii op for vectors; this lowers to LLVM's `llvm.mulhi`
        # equivalent in practice.)
        return emit_mulhii!(lc, args, typ)

    elseif fname === :shli
        # cuTile Intrinsics.shli(a, b) — logical left shift.
        return emit_binop_value!(lc, args, _arith.shli)

    elseif fname === :shri
        # cuTile Intrinsics.shri(a, b, signedness) — logical/arithmetic right
        # shift. Signedness picks shrui vs shrsi.
        return emit_shri!(lc, args, typ)

    elseif fname === :trunci
        # cuTile Intrinsics.trunci(x, T) — truncate to narrower integer T.
        return emit_trunci!(lc, args, typ)

    elseif fname === :itof
        # cuTile Intrinsics.itof(x, F, signedness) — int → float convert.
        return emit_itof!(lc, args, typ)

    elseif fname === :negf
        # cuTile Intrinsics.negf(x) — floating-point negate. Lowers to
        # `arith.negf`.
        return emit_negf!(lc, args, typ)

    elseif fname === :cat
        # cuTile Intrinsics.cat((lhs, rhs), axis::Int) — concatenate two tiles
        # along `axis`. Lowers to `vector.shuffle` (1-D) or
        # `vector.insert_strided_slice` (N-D).
        return emit_cat!(lc, args, typ)

    elseif fname === :extract
        # cuTile Intrinsics.extract(tile, index, shape) — extract a non-
        # overlapping subtile. `index` is 0-indexed (1-based ct.extract was
        # decremented in the frontend); `shape` is the output tile shape, both
        # in Julia col-major order. Lowers to `vector.extract_strided_slice`.
        return emit_extract!(lc, args, typ)

    elseif fname === :get_num_tile_blocks
        # cuTile Intrinsics.get_num_tile_blocks(axis) — grid extent along axis.
        # `axis` is 0-indexed. For grid dims we have, return the i32 cast of
        # the runtime grid-extent arg. For axes beyond `n_grid_dims` (e.g. the
        # RNG-key code unconditionally reads axes 0..1), return 1.
        return emit_num_tile_blocks!(lc, args, typ)

    elseif fname === :cmpi
        return emit_cmpi!(lc, args, typ)

    elseif fname === :cmpf
        return emit_cmpf!(lc, args, typ)

    elseif fname === :exti
        return emit_exti!(lc, args, typ)

    elseif fname === :cldi
        return emit_cldi!(lc, args, typ)

    elseif fname === :remi
        return emit_remi!(lc, args, typ)

    elseif fname === :fldi
        return emit_fldi!(lc, args, typ)

    elseif fname === :mma
        return emit_mma!(lc, args, typ)

    elseif fname === :broadcast
        return emit_broadcast!(lc, args, typ)

    elseif fname === :reshape
        return emit_reshape!(lc, args, typ)

    elseif fname === :permute
        return emit_permute!(lc, args, typ)

    elseif fname === :constant
        return emit_intr_constant!(lc, args, typ)

    elseif fname === :reduce
        return emit_reduce!(lc, idx, args, typ)

    elseif fname === :iota
        return emit_iota!(lc, args, typ)

    elseif fname === :bitcast
        return emit_bitcast!(lc, args, typ)

    elseif fname === :offset
        emit_offset!(lc, idx, args, typ)
        return nothing

    elseif fname === :load_ptr_tko
        return emit_gather!(lc, args, typ)

    elseif fname === :store_ptr_tko
        emit_scatter!(lc, args, typ)
        return nothing

    elseif fname === :tuple
        lc.tuples[idx] = collect(args)
        return nothing

    elseif fname === :getfield
        return emit_getfield!(lc, idx, args, typ)

    elseif fname === :atomic_add
        return emit_atomic_rmw!(lc, args, typ, :add)

    elseif fname === :atomic_max
        return emit_atomic_rmw!(lc, args, typ, :max)

    elseif fname === :atomic_min
        return emit_atomic_rmw!(lc, args, typ, :min)

    elseif fname === :atomic_and
        return emit_atomic_rmw!(lc, args, typ, :and)

    elseif fname === :atomic_or
        return emit_atomic_rmw!(lc, args, typ, :or)

    elseif fname === :atomic_xor
        # Upstream `memref.atomic_rmw` enum has no `xori` kind — route through
        # the generic-RMW path with an `arith.xori` body.
        return emit_atomic_rmw_generic!(lc, args, typ, :xor)

    elseif fname === :atomic_xchg
        return emit_atomic_rmw!(lc, args, typ, :xchg)

    elseif fname === :atomic_cas
        return emit_atomic_cas!(lc, args, typ)
    end

    # ----- SPMD-mode dispatch for plain Julia callees -----
    if lc.spmd
        if fname === :memoryrefnew
            return emit_spmd_memoryrefnew!(lc, idx, args, typ)
        elseif fname === :memoryrefget
            return emit_spmd_memoryrefget!(lc, args, typ)
        elseif fname === :memoryrefset!
            return emit_spmd_memoryrefset!(lc, args, typ)
        elseif fname === :throw
            # Dead inside elided bounds-check IfOps; if we hit it here the
            # walker is processing a live throw, which the SPMD MVP doesn't
            # support. Drop it (the LLVM-IR end will get a no-op).
            return nothing
        end
    end

    # Sentinel function emitted by KernelAbstractions overlays (see
    # `ext/KernelAbstractionsExt.jl`): `KA.__index_Global_Linear(ctx)` is
    # overlaid to `__cutilecpu_spmd_lane_id()`, which inference inlines as
    # a call to a function we never define. The walker recognises the call
    # in SPMD/KA mode and returns the lane vector synthesized at the top of
    # the scf.parallel body.
    if (fname === :__cutilecpu_spmd_lane_id || fname === :__ka_lane_id) &&
       lc.spmd && haskey(lc.arg_vals, lc.lane_arg)
        return lc.arg_vals[lc.lane_arg]
    end

    error("cuTileCPU.walk_call!: unhandled callee $fname " *
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

# cuTile Intrinsics.mulhii(a, b) — high half of the unsigned-widened product.
# Implemented as: widen both operands by `extui` to 2W bits, multiply with
# nuw, shift right by W, then truncate back to W. On vector operands the
# arith ops broadcast naturally.
function emit_mulhii!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("mulhii: unresolved operands ($(args[1]), $(args[2]))")
    t_in = IR.type(a)
    # Determine the input element bit width.
    elem_t = IR.isvector(t_in) ? eltype(t_in) : t_in
    elem_bits = if elem_t == IR.Type(Int32)
        32
    elseif elem_t == IR.Type(Int64)
        64
    elseif elem_t == IR.Type(Int16)
        16
    elseif elem_t == IR.Type(Int8)
        8
    else
        error("mulhii: unsupported element type $elem_t")
    end
    # Widened types.
    wide_elem = if elem_bits == 32
        IR.Type(Int64)
    elseif elem_bits == 16
        IR.Type(Int32)
    elseif elem_bits == 8
        IR.Type(Int16)
    else
        error("mulhii: unsupported elem bits $elem_bits")
    end
    wide_t = if IR.isvector(t_in)
        IR.VectorType(ndims(t_in), [Int(size(t_in, i)) for i in 1:ndims(t_in)], wide_elem)
    else
        wide_elem
    end
    a_wide = IR.result(_arith.extui(a; out=wide_t))
    b_wide = IR.result(_arith.extui(b; out=wide_t))
    prod = IR.result(_arith.muli(a_wide, b_wide))
    shift_const_val = if IR.isvector(t_in)
        n = prod_shape = ndims(t_in)
        # Splat shift constant.
        sh = wide_elem == IR.Type(Int64) ? Int64(elem_bits) :
             wide_elem == IR.Type(Int32) ? Int32(elem_bits) :
             Int16(elem_bits)
        IR.result(_arith.constant(; value=_splat_attr(sh, wide_t), result=wide_t))
    else
        sh = wide_elem == IR.Type(Int64) ? Int64(elem_bits) :
             wide_elem == IR.Type(Int32) ? Int32(elem_bits) :
             Int16(elem_bits)
        IR.result(_arith.constant(; value=IR.Attribute(sh)))
    end
    shifted = IR.result(_arith.shrui(prod, shift_const_val))
    return IR.result(_arith.trunci(shifted; out=t_in))
end

# cuTile Intrinsics.shri(a, b, signedness) — arithmetic / logical right shift.
function emit_shri!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("shri: unresolved operands ($(args[1]), $(args[2]))")
    signed = length(args) >= 3 ?
             something(resolve_const(lc, args[3]), args[3]) :
             ct.Signedness.Signed
    if signed === ct.Signedness.Unsigned
        return IR.result(_arith.shrui(a, b))
    else
        return IR.result(_arith.shrsi(a, b))
    end
end

# cuTile Intrinsics.trunci(x, T) — narrow integer cast to type T.
function emit_trunci!(lc::LowerCtx, args, @nospecialize(typ))
    v = resolve_value_or_const(lc, args[1])
    v === nothing && error("trunci: unresolved operand $(args[1])")
    target_T = something(resolve_const(lc, args[2]), args[2])
    target_T isa Type || error("trunci: target type must be a Type, got $target_T")
    out_t = if typ isa DataType && typ <: ct.Tile
        mlir_type_for_tile(typ)
    else
        mlir_elem_type(target_T)
    end
    # Identity if MLIR types match (signless).
    IR.type(v) == out_t && return v
    return IR.result(_arith.trunci(v; out=out_t))
end

# cuTile Intrinsics.itof(x, F, signedness) — int → float convert.
function emit_itof!(lc::LowerCtx, args, @nospecialize(typ))
    v = resolve_value_or_const(lc, args[1])
    v === nothing && error("itof: unresolved operand $(args[1])")
    target_T = something(resolve_const(lc, args[2]), args[2])
    target_T isa Type || error("itof: target type must be a Type, got $target_T")
    signed = length(args) >= 3 ?
             something(resolve_const(lc, args[3]), args[3]) :
             ct.Signedness.Signed
    out_t = if typ isa DataType && typ <: ct.Tile
        mlir_type_for_tile(typ)
    else
        mlir_elem_type(target_T)
    end
    if signed === ct.Signedness.Unsigned
        return IR.result(_arith.uitofp(v; out=out_t))
    else
        return IR.result(_arith.sitofp(v; out=out_t))
    end
end

# cuTile Intrinsics.negf(x) → arith.negf. Result type matches operand.
function emit_negf!(lc::LowerCtx, args, @nospecialize(typ))
    v = resolve_value_or_const(lc, args[1])
    v === nothing && error("negf: unresolved operand $(args[1])")
    return IR.result(_arith.negf(v))
end

# cuTile Intrinsics.cat((lhs, rhs), axis::Int) — concatenate two tiles along
# `axis` (0-indexed Julia col-major). For 1-D tiles, lowers to `vector.shuffle`
# with the identity-then-shifted lane permutation. For N-D tiles, lowers via
# two `vector.insert_strided_slice` ops into a zero-initialised result.
function emit_cat!(lc::LowerCtx, args, @nospecialize(typ))
    tiles_ref = args[1]
    tile_refs = if tiles_ref isa SSAValue
        haskey(lc.tuples, tiles_ref.id) ||
            error("cat: tiles tuple must be a tuple SSA, got $tiles_ref")
        lc.tuples[tiles_ref.id]
    elseif tiles_ref isa Tuple
        collect(tiles_ref)
    else
        error("cat: tiles operand must be a tuple, got $tiles_ref")
    end
    length(tile_refs) == 2 || error("cat: only 2-tile cat supported, got $(length(tile_refs))")
    lhs = resolve_value_or_const(lc, tile_refs[1])
    rhs = resolve_value_or_const(lc, tile_refs[2])
    (lhs === nothing || rhs === nothing) &&
        error("cat: unresolved operand tiles")
    axis = something(resolve_const(lc, args[2]), args[2])
    axis isa Integer || error("cat: axis must be const int, got $axis")
    julia_axis = Int(axis)  # 0-indexed Julia

    # Output type.
    out_t = mlir_type_for(typ)
    @assert IR.isvector(out_t) "cat: result must be a vector type"
    out_rank = ndims(out_t)
    # MLIR (row-major) axis: Julia col-major axis k → MLIR axis N-1-k.
    mlir_axis = out_rank - 1 - julia_axis

    if out_rank == 1
        # vector.shuffle: permutation 0..n_lhs-1 then n_lhs..n_lhs+n_rhs-1.
        lhs_t = IR.type(lhs)
        rhs_t = IR.type(rhs)
        n_lhs = Int(size(lhs_t, 1))
        n_rhs = Int(size(rhs_t, 1))
        mask = Int64[i for i in 0:(n_lhs + n_rhs - 1)]
        return IR.result(_vector.shuffle(lhs, rhs;
            vector=out_t,
            mask=IR.DenseArrayAttribute(mask)))
    end

    # N-D path: insert lhs at offset 0 along `mlir_axis`, then rhs at offset
    # `size(lhs, mlir_axis)`.
    lhs_t = IR.type(lhs)
    rhs_t = IR.type(rhs)
    elem_t = eltype(out_t)
    # Start with a zero-splat result and use two insert_strided_slice ops.
    out_shape = [Int(size(out_t, i)) for i in 1:out_rank]
    zero_attr = _splat_attr(zero(tile_eltype(typ)), out_t)
    base = IR.result(_arith.constant(; value=zero_attr, result=out_t))

    lhs_offsets = Int64[i == mlir_axis ? 0 : 0 for i in 0:(out_rank - 1)]
    lhs_strides = Int64[1 for _ in 0:(ndims(lhs_t) - 1)]
    base1 = IR.result(_vector.insert_strided_slice(lhs, base;
        res=out_t,
        offsets=_i64_array_attr(lhs_offsets),
        strides=_i64_array_attr(lhs_strides)))

    rhs_offsets = Int64[i == mlir_axis ? Int(size(lhs_t, mlir_axis + 1)) : 0
                        for i in 0:(out_rank - 1)]
    rhs_strides = Int64[1 for _ in 0:(ndims(rhs_t) - 1)]
    return IR.result(_vector.insert_strided_slice(rhs, base1;
        res=out_t,
        offsets=_i64_array_attr(rhs_offsets),
        strides=_i64_array_attr(rhs_strides)))
end

# cuTile Intrinsics.extract(tile, index, shape) — extract a non-overlapping
# subtile at slice (index) of size (shape). Both `index` and `shape` are in
# Julia col-major order; `index` is 0-indexed (frontend already converted from
# 1-based). Lowers to `vector.extract_strided_slice`, with axes reversed for
# MLIR's row-major convention.
function emit_extract!(lc::LowerCtx, args, @nospecialize(typ))
    src_v = resolve_value_or_const(lc, args[1])
    src_v === nothing && error("extract: cannot resolve source operand $(args[1])")
    index_t = something(resolve_const(lc, args[2]), args[2])
    if index_t isa SSAValue
        haskey(lc.tuples, index_t.id) ||
            error("extract: cannot resolve index tuple SSA $(args[2])")
        index_t = Tuple(lc.tuples[index_t.id])
    end
    index_t isa Tuple ||
        error("extract: index must be a tuple, got $index_t")
    shape_t = something(resolve_const(lc, args[3]), args[3])
    if shape_t isa SSAValue
        haskey(lc.tuples, shape_t.id) ||
            error("extract: cannot resolve shape tuple SSA $(args[3])")
        shape_t = Tuple(lc.tuples[shape_t.id])
    end
    shape_t isa Tuple ||
        error("extract: shape must be a tuple, got $shape_t")
    length(index_t) == length(shape_t) ||
        error("extract: index and shape lengths differ ($index_t vs $shape_t)")

    n = length(shape_t)
    julia_idx = Int[Int(i) for i in index_t]
    julia_shape = Int[Int(s) for s in shape_t]

    # Result type from `typ` (the cuTile Tile{T, Tuple{shape...}} return).
    eT = tile_eltype(typ)
    eT === nothing &&
        error("extract: cannot determine element type from $typ")
    out_t = mlir_tile_type(Tuple(julia_shape), eT)

    # Map Julia col-major axes to MLIR row-major axes: axis k_jl → k_mlir = n-1-k_jl.
    # Offset along each axis is index * shape (slice-mode semantics).
    # NOTE: `vector.extract_strided_slice` (and `vector.insert_strided_slice`)
    # consume the offsets/sizes/strides as `ArrayAttr` of i64 attributes — not
    # `DenseArrayAttr`. Passing the latter silently drops the attribute.
    mlir_offsets = Int64[julia_idx[n - i] * julia_shape[n - i] for i in 0:(n - 1)]
    mlir_sizes   = Int64[julia_shape[n - i] for i in 0:(n - 1)]
    mlir_strides = Int64[1 for _ in 0:(n - 1)]

    return IR.result(_vector.extract_strided_slice(src_v;
        result_0=out_t,
        offsets=_i64_array_attr(mlir_offsets),
        sizes=_i64_array_attr(mlir_sizes),
        strides=_i64_array_attr(mlir_strides)))
end

# Build an `ArrayAttr` of i64 attributes — the format `vector.{insert,extract}_strided_slice`
# expect for their offsets/sizes/strides attributes (a `DenseArrayAttr` is silently
# dropped by the create-op state).
function _i64_array_attr(vs::AbstractVector{<:Integer})
    return IR.Attribute(IR.Attribute[IR.Attribute(Int64(v)) for v in vs])
end

# cuTile Intrinsics.get_num_tile_blocks(axis) — grid extent along the given
# 0-indexed axis. For axes within the runtime grid, return `index_cast` of
# the grid Value (held by the outer wrapper) cast to i32. For axes beyond
# the grid (cuTile's `rng_key` always reads axes 0..1 even on 1-D launches),
# return constant Int32(1).
function emit_num_tile_blocks!(lc::LowerCtx, args, @nospecialize(typ))
    axis = something(resolve_const(lc, args[1]), args[1])
    axis isa Integer ||
        error("get_num_tile_blocks: axis must be const int, got $(args[1])")
    ax = Int(axis)
    # `axis` is 0-indexed. Return the corresponding i32 grid extent. For axes
    # beyond `n_grid_dims` (cuTile's `rng_key` unconditionally reads axes
    # 0..1 even on 1-D launches), return constant Int32(1) — matching the
    # device-side semantics where unused grid dims are 1.
    if ax + 1 ≤ length(lc.grid_extents_i32)
        return lc.grid_extents_i32[ax + 1]
    end
    return IR.result(_arith.constant(; value=IR.Attribute(Int32(1))))
end

# Unary math op (math.exp, math.log, …). Result type comes from the operand
# (math ops are elementwise; the result type matches the operand type).
function emit_unary_math!(lc::LowerCtx, args, op_fn, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    a === nothing && error("$(op_fn): unresolved operand $(args[1])")
    return IR.result(op_fn(a; result=IR.type(a)))
end

# cuTile Intrinsics.fma(x, y, z) → math.fma. Same vector shape across all
# three operands.
function emit_fma!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    c = resolve_value_or_const(lc, args[3])
    (a === nothing || b === nothing || c === nothing) &&
        error("fma: unresolved operands ($(args[1]), $(args[2]), $(args[3]))")
    return IR.result(_math.fma(a, b, c; result=IR.type(a)))
end

# cuTile Intrinsics.maxi(a, b, signedness)/mini(a, b, signedness) → arith
# max{s,u}i / min{s,u}i. Signedness is the 3rd arg.
function emit_maxmin_int!(lc::LowerCtx, args, sop_fn, uop_fn)
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("maxi/mini: unresolved operands ($(args[1]), $(args[2]))")
    signed = length(args) >= 3 ?
             something(resolve_const(lc, args[3]), args[3]) :
             ct.Signedness.Signed
    if signed === ct.Signedness.Unsigned
        return IR.result(uop_fn(a, b))
    else
        return IR.result(sop_fn(a, b))
    end
end

# cuTile cmpi(lhs, rhs, predicate::ComparisonPredicate.T, sign::Signedness.T)
# → arith.cmpi with the matching i64 predicate attribute.
function emit_cmpi!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("cmpi: unresolved operands ($(args[1]), $(args[2]))")
    pred = something(resolve_const(lc, args[3]), args[3])
    signed = length(args) >= 4 ?
             something(resolve_const(lc, args[4]), args[4]) :
             ct.Signedness.Signed
    pred isa ct.ComparisonPredicate.T ||
        error("cmpi: predicate must be ComparisonPredicate.T, got $pred")
    signed isa ct.Signedness.T ||
        error("cmpi: signedness must be Signedness.T, got $signed")
    mlir_pred = cmpi_predicate_code(pred, signed)
    pred_attr = IR.Attribute(mlir_pred, IR.Type(Int64))
    # arith.cmpi result is i1; for tile element type Bool yields i1 (or
    # vector<...xi1>). Result type is inferred by Reactant wrapper since we
    # don't pass it.
    return IR.result(_arith.cmpi(a, b; predicate=pred_attr))
end

function cmpi_predicate_code(pred::ct.ComparisonPredicate.T, signed::ct.Signedness.T)
    pred === ct.ComparisonPredicate.Equal && return 0
    pred === ct.ComparisonPredicate.NotEqual && return 1
    is_signed = signed === ct.Signedness.Signed
    pred === ct.ComparisonPredicate.LessThan        && return is_signed ? 2 : 6
    pred === ct.ComparisonPredicate.LessThanOrEqual && return is_signed ? 3 : 7
    pred === ct.ComparisonPredicate.GreaterThan     && return is_signed ? 4 : 8
    pred === ct.ComparisonPredicate.GreaterThanOrEqual && return is_signed ? 5 : 9
    error("cmpi: unsupported predicate $pred")
end

# cuTile cmpf(lhs, rhs, predicate::ComparisonPredicate.T, [ordering])
function emit_cmpf!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("cmpf: unresolved operands ($(args[1]), $(args[2]))")
    pred = something(resolve_const(lc, args[3]), args[3])
    ord  = length(args) >= 4 ?
           something(resolve_const(lc, args[4]), args[4]) :
           ct.ComparisonOrdering.Ordered
    pred isa ct.ComparisonPredicate.T ||
        error("cmpf: predicate must be ComparisonPredicate.T, got $pred")
    code = cmpf_predicate_code(pred, ord)
    pred_attr = IR.Attribute(code, IR.Type(Int64))
    return IR.result(_arith.cmpf(a, b; predicate=pred_attr))
end

function cmpf_predicate_code(pred::ct.ComparisonPredicate.T, ord::ct.ComparisonOrdering.T)
    is_ord = ord === ct.ComparisonOrdering.Ordered
    pred === ct.ComparisonPredicate.Equal           && return is_ord ? 1  : 8
    pred === ct.ComparisonPredicate.GreaterThan     && return is_ord ? 2  : 9
    pred === ct.ComparisonPredicate.GreaterThanOrEqual && return is_ord ? 3 : 10
    pred === ct.ComparisonPredicate.LessThan        && return is_ord ? 4  : 11
    pred === ct.ComparisonPredicate.LessThanOrEqual && return is_ord ? 5  : 12
    pred === ct.ComparisonPredicate.NotEqual        && return is_ord ? 6  : 13
    error("cmpf: unsupported predicate $pred")
end

# cuTile exti(x, target_jl_type, sign) → arith.extsi / arith.extui
function emit_exti!(lc::LowerCtx, args, @nospecialize(typ))
    v = resolve_value_or_const(lc, args[1])
    v === nothing && error("exti: unresolved operand $(args[1])")
    target_T = something(resolve_const(lc, args[2]), args[2])
    target_T isa Type ||
        error("exti: target type must be a Type, got $target_T")
    signed = length(args) >= 3 ?
             something(resolve_const(lc, args[3]), args[3]) :
             ct.Signedness.Signed
    # Determine result MLIR type: if `typ` is a tile, use its mlir vector
    # type; else scalar of target_T.
    out_t = if typ isa DataType && typ <: ct.Tile
        mlir_type_for_tile(typ)
    else
        mlir_elem_type(target_T)
    end
    if signed === ct.Signedness.Unsigned
        return IR.result(_arith.extui(v; out=out_t))
    else
        return IR.result(_arith.extsi(v; out=out_t))
    end
end

# cuTile Intrinsics.cldi(lhs, rhs, sign) → arith.ceildivsi / arith.ceildivui.
function emit_cldi!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("cldi: unresolved operands ($(args[1]), $(args[2]))")
    signed = length(args) >= 3 ?
             something(resolve_const(lc, args[3]), args[3]) :
             ct.Signedness.Signed
    if signed === ct.Signedness.Unsigned
        return IR.result(_arith.ceildivui(a, b))
    else
        return IR.result(_arith.ceildivsi(a, b))
    end
end

# cuTile Intrinsics.remi(lhs, rhs, sign) → arith.remsi / arith.remui.
# Surfaces from `rem(::IntTile, ::Integer)` (and tile/tile variants); the
# atomic histogram path uses this for bucket = v % n_buckets.
function emit_remi!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("remi: unresolved operands ($(args[1]), $(args[2]))")
    signed = length(args) >= 3 ?
             something(resolve_const(lc, args[3]), args[3]) :
             ct.Signedness.Signed
    if signed === ct.Signedness.Unsigned
        return IR.result(_arith.remui(a, b))
    else
        return IR.result(_arith.remsi(a, b))
    end
end

# cuTile Intrinsics.fldi(lhs, rhs, sign) → arith.floordivsi / arith.divui.
# `fldi` is signed floor-division (rounding toward -∞) for signed args; on
# the unsigned side it coincides with truncated division (`arith.divui`).
function emit_fldi!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    (a === nothing || b === nothing) &&
        error("fldi: unresolved operands ($(args[1]), $(args[2]))")
    signed = length(args) >= 3 ?
             something(resolve_const(lc, args[3]), args[3]) :
             ct.Signedness.Signed
    if signed === ct.Signedness.Unsigned
        return IR.result(_arith.divui(a, b))
    else
        return IR.result(_arith.floordivsi(a, b))
    end
end

# cuTile Intrinsics.mma(lhs, rhs, acc) — matrix-multiply-accumulate in
# TileIR-row-major form. The cuTile frontend's `muladd(a, b, acc)` for
# Julia 2-D tiles becomes `Intrinsics.mma(b, a, acc)` (see operations.jl);
# the batched (≥3-D × ≥3-D) path flattens trailing batch dims to a single
# leading "batch" in TileIR row-major and then calls `Intrinsics.mma(b, a, acc)`
# with operands of TileIR shape (B, …).
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
# 3-D batched case. cuTile's batched-mma `_muladd` (operations.jl:1195) reshapes
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
# We dispatch on the rank of `acc` (which is also the rank of lhs/rhs in cuTile's
# canonical batched form): 2 → plain matmul, 3 → batched matmul. Higher ranks
# don't occur because cuTile pre-flattens batch dims to a single axis.
function emit_mma!(lc::LowerCtx, args, @nospecialize(typ))
    a = resolve_value_or_const(lc, args[1])
    b = resolve_value_or_const(lc, args[2])
    acc = resolve_value_or_const(lc, args[3])
    (a === nothing || b === nothing || acc === nothing) &&
        error("mma: unresolved operands ($(args[1]), $(args[2]), $(args[3])))")
    result_t = IR.type(acc)

    acc_t = IR.type(acc)
    rank = IR.isvector(acc_t) ? ndims(acc_t) : 0
    if rank == 2
        iter_attr = IR.Attribute(IR.Attribute[
            parse(IR.Attribute, "#vector.iterator_type<parallel>"),
            parse(IR.Attribute, "#vector.iterator_type<parallel>"),
            parse(IR.Attribute, "#vector.iterator_type<reduction>"),
        ])
        lhs_map = parse(IR.Attribute, "affine_map<(m, n, k) -> (n, k)>")
        rhs_map = parse(IR.Attribute, "affine_map<(m, n, k) -> (k, m)>")
        acc_map = parse(IR.Attribute, "affine_map<(m, n, k) -> (n, m)>")
    elseif rank == 3
        iter_attr = IR.Attribute(IR.Attribute[
            parse(IR.Attribute, "#vector.iterator_type<parallel>"),   # b
            parse(IR.Attribute, "#vector.iterator_type<parallel>"),   # m
            parse(IR.Attribute, "#vector.iterator_type<parallel>"),   # n
            parse(IR.Attribute, "#vector.iterator_type<reduction>"),  # k
        ])
        lhs_map = parse(IR.Attribute, "affine_map<(b, m, n, k) -> (b, n, k)>")
        rhs_map = parse(IR.Attribute, "affine_map<(b, m, n, k) -> (b, k, m)>")
        acc_map = parse(IR.Attribute, "affine_map<(b, m, n, k) -> (b, n, m)>")
    else
        error("mma: unsupported acc rank $rank (expected 2 or 3)")
    end
    maps_attr = IR.Attribute(IR.Attribute[lhs_map, rhs_map, acc_map])

    kind_attr = parse(IR.Attribute, "#vector.kind<add>")
    return IR.result(_vector.contract(a, b, acc;
        result_0=result_t,
        indexing_maps=maps_attr,
        iterator_types=iter_attr,
        kind=kind_attr))
end

# cuTile Intrinsics.broadcast(src, target_shape_tuple) → vector.broadcast.
# `src` can be a scalar literal, a Const-arg scalar, a 0-D / smaller tile.
function emit_broadcast!(lc::LowerCtx, args, @nospecialize(typ))
    src = args[1]
    target_shape = something(resolve_const(lc, args[2]), args[2])
    if target_shape isa SSAValue
        # Tuple-tracked target shape.
        haskey(lc.tuples, target_shape.id) ||
            error("broadcast: cannot resolve target-shape tuple SSA")
        target_shape = Tuple(lc.tuples[target_shape.id])
    end
    target_shape isa Tuple ||
        error("broadcast: target shape must be a tuple, got $target_shape")
    # Materialise src as a Value.
    src_v = resolve_value_or_const(lc, src)
    src_v === nothing && error("broadcast: cannot resolve source operand $src")
    # Element type comes from `typ`.
    eT = tile_eltype(typ)
    eT === nothing && (eT = typeof(src isa Number ? src : 0f0))
    vec_t = mlir_tile_type(target_shape, eT)
    return IR.result(_vector.broadcast(src_v; vector=vec_t))
end

# cuTile Intrinsics.reshape(tile, target_shape_tuple) → vector.shape_cast.
function emit_reshape!(lc::LowerCtx, args, @nospecialize(typ))
    src_v = resolve_value_or_const(lc, args[1])
    src_v === nothing && error("reshape: cannot resolve source")
    target_shape = something(resolve_const(lc, args[2]), args[2])
    if target_shape isa SSAValue
        haskey(lc.tuples, target_shape.id) ||
            error("reshape: cannot resolve target-shape tuple SSA")
        target_shape = Tuple(lc.tuples[target_shape.id])
    end
    target_shape isa Tuple ||
        error("reshape: target shape must be a tuple, got $target_shape")
    eT = tile_eltype(typ)
    eT === nothing &&
        error("reshape: cannot determine element type from $typ")
    out_t = mlir_tile_type(target_shape, eT)
    # 0-D corner case: target shape `()` means scalar element. shape_cast
    # cannot produce a non-vector result, so fall back to a single
    # `vector.extract` at lane 0 — requires the source to be a 1-element
    # 1-D vector (the only meaningful case here).
    if isempty(target_shape)
        src_t = IR.type(src_v)
        if !IR.isvector(src_t)
            # Already a scalar — identity reshape.
            return src_v
        end
        pos_attr = IR.DenseArrayAttribute(Int64[0])
        return IR.result(_vector.extract(src_v, IR.Value[];
            result=out_t, static_position=pos_attr))
    end
    src_t = IR.type(src_v)
    if !IR.isvector(src_t)
        # Scalar → vector with target_shape: arises when a 0-D tile (e.g. the
        # result of `atomic_add` or `bid + 1`) needs to be reshaped to a
        # single-lane (or broadcast-shape) tile prior to a tile store. The
        # SCI emits `reshape(scalar, (1,))` here; `vector.shape_cast` rejects
        # the scalar operand, so we lift it to a vector via `vector.broadcast`.
        return IR.result(_vector.broadcast(src_v; vector=out_t))
    end
    return IR.result(_vector.shape_cast(src_v; result=out_t))
end

# cuTile Intrinsics.permute(tile, perm) → vector.transpose.
#
# `perm` is a 0-indexed Julia (col-major) permutation. cuTile's frontend
# already lowered `permutedims(tile, (2, 1))` to `Intrinsics.permute(tile, (1, 0))`,
# i.e. 1-indexed Julia → 0-indexed Julia. We need an MLIR (row-major) perm.
#
# Mapping (see cuTile.jl/src/compiler/intrinsics/core.jl `emit_intrinsic!`
# for Intrinsics.permute): given Julia 0-indexed perm `julia_perm`, the
# row-major (MLIR) permutation is
#   mlir_perm[i] = n - 1 - julia_perm[n - 1 - i]   for i in 0:n-1.
# The result MLIR vector type is the input MLIR vector type with its dims
# reordered by `mlir_perm`.
function emit_permute!(lc::LowerCtx, args, @nospecialize(typ))
    src_v = resolve_value_or_const(lc, args[1])
    src_v === nothing && error("permute: cannot resolve source operand $(args[1])")
    perm_const = something(resolve_const(lc, args[2]), args[2])
    if perm_const isa SSAValue
        haskey(lc.tuples, perm_const.id) ||
            error("permute: cannot resolve perm tuple SSA $(args[2])")
        perm_const = Tuple(lc.tuples[perm_const.id])
    end
    perm_const isa Tuple ||
        error("permute: permutation must be a tuple, got $perm_const")
    n = length(perm_const)
    julia_perm = Int[Int(p) for p in perm_const]
    mlir_perm = Int64[n - 1 - julia_perm[n - i] for i in 0:n-1]

    eT = tile_eltype(typ)
    eT === nothing && error("permute: cannot determine element type from $typ")
    out_shape = Tuple(tile_shape(typ))
    out_t = mlir_tile_type(out_shape, eT)
    perm_attr = IR.DenseArrayAttribute(mlir_perm)
    return IR.result(_vector.transpose(src_v; result=out_t, permutation=perm_attr))
end

# cuTile Intrinsics.constant(shape::Tuple, value, T) → arith.constant of a
# splat dense<value> : vector<...xT> when `value` is a compile-time literal.
# When `value` is a runtime SSA (cuTile's `fill(scalar, dims)` overlay uses
# this form to broadcast a scalar to a tile), we instead lower to
# `vector.broadcast` of the scalar Value.
function emit_intr_constant!(lc::LowerCtx, args, @nospecialize(typ))
    shape = something(resolve_const(lc, args[1]), args[1])
    if shape isa SSAValue
        haskey(lc.tuples, shape.id) ||
            error("constant: cannot resolve shape tuple SSA")
        shape = Tuple(lc.tuples[shape.id])
    end
    shape isa Tuple ||
        error("constant: shape must be a tuple, got $shape")
    val_arg = args[2]
    val_const = resolve_const(lc, val_arg)
    T = length(args) >= 3 ?
        something(resolve_const(lc, args[3]), args[3]) :
        (val_const isa Number ? typeof(val_const) : nothing)
    T isa Type || error("constant: type must be a Type, got $T")

    if val_const isa Number
        converted = convert(T, val_const)
        if isempty(shape)
            return _const_value(converted)
        end
        vec_t = mlir_tile_type(shape, T)
        # Use Base.fill(value, shaped_type) — there are overloads for primitive
        # element types (Bool/Int*/Float32/Float64) which produce a splat dense
        # elements attribute. For Float16/BFloat16 we splat via the generic
        # Attribute path (see `_splat_attr`).
        splat = _splat_attr(converted, vec_t)
        return IR.result(_arith.constant(; value=splat, result=vec_t))
    end
    # Runtime value (SSA / Argument): broadcast the scalar Value.
    scalar_v = resolve_value_or_const(lc, val_arg)
    scalar_v === nothing &&
        error("constant: value must be a literal or resolvable Value, got $val_arg")
    if isempty(shape)
        return scalar_v
    end
    vec_t = mlir_tile_type(shape, T)
    return IR.result(_vector.broadcast(scalar_v; vector=vec_t))
end

# cuTile Intrinsics.reduce((tile,), axis::Int, combiner, (identities,)) →
# Tuple{Tile{..., reduced_shape_with_1_in_axis}}. Lowers to
# vector.multi_reduction + vector.shape_cast (to re-add the size-1 dim that
# cuTile's reduce semantics preserves).
function emit_reduce!(lc::LowerCtx, idx::Int, args, @nospecialize(typ))
    tiles_ref = args[1]
    tiles_ref isa SSAValue && haskey(lc.tuples, tiles_ref.id) ||
        error("reduce: input tiles tuple must be a tuple SSA, got $tiles_ref")
    tile_refs = lc.tuples[tiles_ref.id]
    length(tile_refs) == 1 ||
        error("reduce: only single-tile reductions supported, got $(length(tile_refs))")
    src_v = resolve_value_or_const(lc, tile_refs[1])
    src_v === nothing && error("reduce: cannot resolve input tile")
    julia_axis = something(resolve_const(lc, args[2]), args[2])
    julia_axis isa Integer || error("reduce: axis must be const Int, got $julia_axis")
    combiner = something(resolve_const(lc, args[3]), args[3])
    identities_ref = args[4]
    identities_ref isa SSAValue && haskey(lc.tuples, identities_ref.id) ||
        identities_ref isa Tuple ||
        error("reduce: identities must be a tuple, got $identities_ref")
    id_vals = identities_ref isa SSAValue ? lc.tuples[identities_ref.id] : collect(identities_ref)
    length(id_vals) == 1 ||
        error("reduce: only one identity supported, got $(length(id_vals))")

    # Result type from `typ`: Tuple{Tile{eT, shape_with_1_in_axis}}
    typ isa DataType && typ <: Tuple && length(typ.parameters) == 1 ||
        error("reduce: expected Tuple{Tile} result type, got $typ")
    out_tile_T = typ.parameters[1]
    out_shape  = Tuple(tile_shape(out_tile_T))
    eT         = tile_eltype(out_tile_T)

    # Input shape — derive from the tile_refs[1] type if we can.
    in_tile_T = nothing
    if tile_refs[1] isa SSAValue
        # Look up from the sci type? We don't have it directly. Get from the
        # vector type of the source Value.
    end
    # Compute MLIR reduction-dim from julia_axis (0-indexed col-major).
    #   Julia (1-indexed) dim k <=> MLIR (0-indexed row-major) dim N-k.
    #   julia_axis (0-indexed) k0 corresponds to Julia 1-indexed k = k0+1, so
    #   MLIR dim = N - (k0+1) = N - 1 - k0.
    # We need N: get it from the source Value's vector type.
    src_t = IR.type(src_v)
    @assert IR.isvector(src_t) "reduce: source must be a vector"
    N = ndims(src_t)
    mlir_dim = N - 1 - Int(julia_axis)
    mlir_dim ≥ 0 && mlir_dim < N ||
        error("reduce: axis $julia_axis out of range for rank $N")

    # Build acc = broadcast of identity to the multi_reduction *output* shape
    # (input shape with `mlir_dim` removed).
    in_mlir_shape = [Int(size(src_t, i)) for i in 1:N]
    out_mr_shape = Int[in_mlir_shape[i] for i in 1:N if (i - 1) != mlir_dim]
    identity_val = id_vals[1]
    identity_val isa Number ||
        error("reduce: identity must be a literal number, got $identity_val")
    identity_v = IR.result(_arith.constant(;
        value=IR.Attribute(convert(eT, identity_val))))
    acc_v = if isempty(out_mr_shape)
        identity_v
    else
        acc_t = IR.VectorType(length(out_mr_shape), out_mr_shape, mlir_elem_type(eT))
        IR.result(_vector.broadcast(identity_v; vector=acc_t))
    end

    # Kind attribute
    kind_name = reduction_kind_name(combiner, eT)
    kind_attr = parse(IR.Attribute, "#vector.kind<$kind_name>")
    # `reduction_dims` is typed as `I64ArrayAttr` (regular `[N : i64]`) in
    # MLIR 18 and `DenseI64ArrayAttr` (`array<i64: N>`) in MLIR 19+ — pick the
    # right form from the active MLIR version. (Detected via `MLIR.MLIR_VERSION[]`.)
    rd_attr = if MLIR.MLIR_VERSION[] < v"19"
        parse(IR.Attribute, "[$(mlir_dim) : i64]")
    else
        parse(IR.Attribute, "array<i64: $(mlir_dim)>")
    end

    # Output type after multi_reduction = vector or scalar.
    out_t = if isempty(out_mr_shape)
        mlir_elem_type(eT)
    else
        IR.VectorType(length(out_mr_shape), out_mr_shape, mlir_elem_type(eT))
    end
    reduced_v = IR.result(_vector.multi_reduction(src_v, acc_v;
        dest=out_t, kind=kind_attr, reduction_dims=rd_attr))

    # Shape-cast back to the cuTile-preserved shape (size-1 in reduced axis).
    final_t = mlir_tile_type(out_shape, eT)
    final_v = if final_t === out_t
        reduced_v
    elseif out_t == mlir_elem_type(eT)
        # Scalar → vector with leading 1s. Use broadcast.
        IR.result(_vector.broadcast(reduced_v; vector=final_t))
    else
        IR.result(_vector.shape_cast(reduced_v; result=final_t))
    end

    # Store as a 1-tuple multi-result for downstream getfield.
    lc.ssa_multi[idx] = IR.Value[final_v]
    return nothing
end

# Map a cuTile combiner function + element type to a vector.kind name.
function reduction_kind_name(combiner, eT::Type)
    # cuTile's combiner is `cuTile.+` etc. — these are overloaded plus/min/max.
    name = string(combiner)
    if combiner === Base.:+ || endswith(name, ".+ ") || endswith(name, ".+")
        return "add"
    elseif combiner === Base.:*
        return "mul"
    elseif combiner === Base.max
        return eT <: AbstractFloat ? "maxnumf" :
               eT <: Signed        ? "maxsi"   : "maxui"
    elseif combiner === Base.min
        return eT <: AbstractFloat ? "minnumf" :
               eT <: Signed        ? "minsi"   : "minui"
    end
    # Fallback by name
    if occursin("+", name); return "add"; end
    if occursin("max", lowercase(name))
        return eT <: AbstractFloat ? "maxnumf" :
               eT <: Signed        ? "maxsi"   : "maxui"
    end
    if occursin("min", lowercase(name))
        return eT <: AbstractFloat ? "minnumf" :
               eT <: Signed        ? "minsi"   : "minui"
    end
    error("reduce: unsupported combiner $combiner")
end

# Handle Base.getfield in both Argument-rooted (TileArray's ptr/sizes/strides
# fields) and SSA-rooted (extract from a multi-result control-flow op or a
# tracked tuple) forms.
function emit_getfield!(lc::LowerCtx, idx::Int, args, @nospecialize(typ))
    obj = args[1]
    if obj isa Argument
        field = args[2]
        fld_sym = field isa QuoteNode ? field.value :
                  field isa Symbol    ? field :
                  error("getfield: field must be Symbol/QuoteNode, got $field")
        lc.field_refs[idx] = (obj.n, fld_sym)
        return nothing
    elseif obj isa SSAValue
        # KernelState-rooted: `getfield(kernel_state_ssa, :seed)` → seed param.
        if obj.id in lc.kernel_state_ssas
            fld_sym = if args[2] isa QuoteNode
                args[2].value
            elseif args[2] isa Symbol
                args[2]
            else
                error("getfield(KernelState, ...): field must be Symbol/QuoteNode, got $(args[2])")
            end
            fld_sym === :seed ||
                error("getfield(KernelState, ...): only :seed is supported, got :$fld_sym")
            lc.seed_param === nothing &&
                error("getfield(KernelState, :seed): seed parameter not bound")
            return lc.seed_param
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
            if fld === :sizes
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
                if typ isa DataType && typ <: ct.Tile
                    eT = tile_eltype(typ)
                    out_t = mlir_elem_type(eT)
                    return IR.result(_arith.index_cast(dim_v; out=out_t))
                elseif typ isa Type && typ <: Integer
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
    error("getfield: unsupported obj $obj")
end

# ----------------------------------------------------------------------------
# Control flow: scf.if / scf.for
# ----------------------------------------------------------------------------

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
        if p isa DataType && p <: ct.Tile
            push!(rts, mlir_type_for_tile(p))
        elseif p isa Type && p <: Number
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
    # Cast bounds to `index`. cuTile's IRStructurizer normalises range
    # iteration so that the recorded `upper` is already the half-open end
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

    # The MLIR scf.for block args are [index_iv, iter_args...]. The cuTile
    # iv_arg type is Int32/Int64 — cast back inside the body.
    block_arg_types = IR.Type[idx_t; iter_types]
    block_arg_locs  = [IR.Location() for _ in 1:length(block_arg_types)]
    body_region = IR.Region()
    body_block = IR.Block(block_arg_types, block_arg_locs)
    push!(body_region, body_block)

    @with_block body_block begin
        # IV: cast index → cuTile IV type.
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

# cuTile Intrinsics.iota((N,), T) → an `IntTile{(N,), T}` with values 0..N-1.
# We materialise this as `vector.step` (which produces vector<Nxindex>),
# then `arith.index_cast` to vector<NxiK>. cuTile produces Int32 indices by
# default; the cast is mandatory because downstream cmpi/addi/bitcast all
# expect the iK element type rather than `index`.
function emit_iota!(lc::LowerCtx, args, @nospecialize(typ))
    shape = something(resolve_const(lc, args[1]), args[1])
    if shape isa SSAValue
        haskey(lc.tuples, shape.id) ||
            error("iota: cannot resolve shape tuple SSA")
        shape = Tuple(lc.tuples[shape.id])
    end
    shape isa Tuple ||
        error("iota: shape must be a tuple, got $shape")
    T = something(resolve_const(lc, args[2]), args[2])
    T isa Type || error("iota: dtype must be a Type, got $T")
    length(shape) == 1 ||
        error("iota: only 1-D iota is supported, got shape $shape")
    N = Int(shape[1])
    # `vector.step` is MLIR 19+. Emit a constant `[0, 1, ..., N-1]` directly
    # in the iota dtype — same result, one fewer op.
    out_t = mlir_tile_type(shape, T)
    return _emit_step_vec(out_t, T, N)
end

# cuTile Intrinsics.bitcast(src, target_T) — a signless reinterpret. For tile
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
    # In SPMD mode the cuTile-inferred `typ` may be a 0-D tile (scalar) but the
    # actual operand is a lane-wide vector — derive the output type from the
    # operand's shape, not from `typ`.
    src_t = IR.type(v)
    elem_target = mlir_elem_type(target_T)
    out_t = if lc.spmd && IR.isvector(src_t)
        n = Int(size(src_t, 1))
        IR.VectorType(1, Int[n], elem_target)
    elseif typ isa DataType && typ <: ct.Tile
        mlir_type_for_tile(typ)
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

# cuTile Intrinsics.offset(ptr_tile, idx_tile) — builds a pointer tile.
# Tracked-only: we record the source memref (from the ptr operand's
# field-ref) plus the index Value. The downstream gather/scatter consumes
# this OffsetInfo.
function emit_offset!(lc::LowerCtx, idx::Int, args, @nospecialize(typ))
    ptr_ref = args[1]
    ptr_ref isa SSAValue ||
        error("Intrinsics.offset: ptr operand must be SSAValue, got $ptr_ref")
    # Chase ptr → field_ref OR ptr → another OffsetInfo (chained offset).
    base_memref = nothing
    elem_T = nothing
    if haskey(lc.field_refs, ptr_ref.id)
        (arg_id, fld) = lc.field_refs[ptr_ref.id]
        fld === :ptr ||
            error("Intrinsics.offset: ptr operand must be a :ptr field, " *
                  "got :$fld")
        base_memref = lc.arg_vals[arg_id]
        elem_T = lc.arg_elem_types[arg_id]
    elseif haskey(lc.offsets, ptr_ref.id)
        prev = lc.offsets[ptr_ref.id]
        base_memref = prev.base
        elem_T = prev.elem_type
        # Chained offset: combine indices. Not needed for current tests.
        error("Intrinsics.offset: chained offset not yet supported")
    else
        error("Intrinsics.offset: cannot resolve ptr operand %$(ptr_ref.id)")
    end
    idx_v = resolve_value_or_const(lc, args[2])
    idx_v === nothing &&
        error("Intrinsics.offset: cannot resolve index operand $(args[2])")
    # Recover the index shape from the result tile type. 0-D pointer tiles
    # (`Tile{Ptr{T}, Tuple{}}`) appear for scalar-index atomics — the
    # downstream atomic_rmw uses the scalar index directly. Track the
    # 0-D case with an empty `idx_shape`; gather/scatter consumers reject
    # empty shapes (they require an i32-vector index), atomic consumers
    # accept either form.
    sh = typ isa DataType && typ <: ct.Tile ? Tuple(tile_shape(typ)) : ()
    lc.offsets[idx] = OffsetInfo(base_memref, idx_v, elem_T, Int[sh...])
    return nothing
end

# cuTile Intrinsics.load_ptr_tko(offset, latency, mask, padding, token) →
# `vector.gather %base[%c0], %indices, %mask, %pass_thru : memref<...>,
#  vector<Nxi32>, vector<Nxi1>, vector<NxT> into vector<NxT>`
# args = [offset_ssa, latency, mask, padding, token]
function emit_gather!(lc::LowerCtx, args, @nospecialize(typ))
    off_ref = args[1]
    off_ref isa SSAValue && haskey(lc.offsets, off_ref.id) ||
        error("load_ptr_tko: ptr-tile operand must be an Intrinsics.offset " *
              "SSA, got $off_ref")
    off = lc.offsets[off_ref.id]
    isempty(off.idx_shape) &&
        error("load_ptr_tko: scalar-index offset (0-D) not supported by gather")
    # Resolve mask. cuTile passes the bounds mask as args[3]; if `nothing`,
    # synthesise an all-true mask.
    mask_v = _resolve_or_alltrue_mask(lc, args[3], off.idx_shape)
    # Resolve pass-through (padding). Some cuTile paths pass `nothing`; build
    # a zero-of-elem-type splat in that case.
    pad_v = _resolve_or_zero_passthru(lc, args[4], off.elem_type, off.idx_shape)
    # Build the result type: vector<Nx{elem_T}>.
    result_t = mlir_tile_type(Tuple(off.idx_shape), off.elem_type)
    idx_t = IR.IndexType()
    c0 = IR.result(_arith.constant(; value=IR.Attribute(Int(0), idx_t)))
    return IR.result(_vector.gather(off.base, IR.Value[c0], off.indices,
                                     mask_v, pad_v; result=result_t))
end

# cuTile Intrinsics.store_ptr_tko(offset, values, latency, mask, token) →
# `vector.scatter %base[%c0], %indices, %mask, %values : memref<...>,
#  vector<Nxi32>, vector<Nxi1>, vector<NxT>`
function emit_scatter!(lc::LowerCtx, args, @nospecialize(typ))
    off_ref = args[1]
    off_ref isa SSAValue && haskey(lc.offsets, off_ref.id) ||
        error("store_ptr_tko: ptr-tile operand must be an Intrinsics.offset " *
              "SSA, got $off_ref")
    off = lc.offsets[off_ref.id]
    isempty(off.idx_shape) &&
        error("store_ptr_tko: scalar-index offset (0-D) not supported by scatter")
    val_v = resolve_value_or_const(lc, args[2])
    val_v === nothing &&
        error("store_ptr_tko: cannot resolve value-to-store $(args[2])")
    mask_v = _resolve_or_alltrue_mask(lc, args[4], off.idx_shape)
    _vector.scatter(off.base, IR.Value[
        IR.result(_arith.constant(; value=IR.Attribute(Int(0),
                                                       IR.IndexType())))],
        off.indices, mask_v, val_v)
    return nothing
end

# Resolve `mask_ref` to an i1-vector Value matching `idx_shape`. If it
# resolves to `nothing` (cuTile passed `nothing` for "no mask"), construct
# an all-true splat.
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
# cuTile lowers `ct.atomic_add(arr, idx, val; …)` to
#   %ptr   = Intrinsics.offset(arg.ptr, idx_0)        (Tile{Ptr{T},S})
#   %rmw   = Intrinsics.atomic_{add,max,min,and,or,xor,xchg}
#               (%ptr, val, mask, order, scope, token)
# We map the RMW directly to `memref.atomic_rmw <kind> %val, %base[%idx]`. The
# scalar (0-D) form is what `ct.atomic_*(arr, scalar_idx, val)` produces — we
# emit one atomic op on `%base[%idx]`. For tile (N-D) forms we unroll a small
# loop over each lane (acceptable for the small atomic tiles cuTile produces;
# vectorising would require `vector.scatter` with an atomic ordering, which
# upstream MLIR doesn't expose yet).
#
# `ct.atomic_cas(arr, idx, expected, desired; …)` has no `assign`-style
# single-keyword form on `memref.atomic_rmw`; we lower it via
# `memref.generic_atomic_rmw` with a region body that compares the loaded
# value against `expected` and yields either `desired` (on match) or the
# loaded value (no-op). Returns the prior value.
#
# `atomic_xchg` reuses `memref.atomic_rmw` with kind `assign` — the verifier
# accepts an unconditional store-and-return-old at any element type.
#
# Memory order/scope arguments from cuTile (Acquire / Release / AcqRel /
# Relaxed; Block / Device / System) are dropped on the CPU MVP path —
# `memref.atomic_rmw` lowers to `llvm.atomicrmw <op> … acq_rel`, which is the
# strongest ordering the verifier emits for this op and matches cuTile's
# default. Cross-thread synchronisation beyond what libomp + acq_rel give
# isn't part of this target.

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

# Pick the `memref.atomic_rmw` kind keyword for a given cuTile op symbol
# (`:add` / `:max` / `:min` / `:and` / `:or` / `:xor` / `:xchg`) at a given
# Julia element type. cuTile's `atomic_add(addf-mode)` on AbstractFloat →
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
function emit_atomic_rmw!(lc::LowerCtx, args, @nospecialize(typ), op::Symbol)
    off_ref = args[1]
    off_ref isa SSAValue && haskey(lc.offsets, off_ref.id) ||
        error("atomic_$op: ptr-tile operand must be an Intrinsics.offset SSA, " *
              "got $off_ref")
    off = lc.offsets[off_ref.id]
    elem_T = off.elem_type
    kind_kw = _atomic_rmw_kind(op, elem_T)

    val_v = resolve_value_or_const(lc, args[2])
    val_v === nothing && error("atomic_$op: cannot resolve val operand $(args[2])")

    if isempty(off.idx_shape)
        # 0-D form: off.indices is the scalar (i32) index. Single atomic.
        return _emit_one_atomic_rmw!(off.base, off.indices, val_v, kind_kw,
                                     elem_T)
    end

    # N-D form: iterate lanes (1-D only for now), masked when a mask is
    # provided. cuTile lowers the histogram path with a (N,) tile shape and
    # an Int32 lane index — we extract each lane statically via
    # `vector.extract`. Multi-D tiles unroll into row-major linear lanes.
    mask_ref = length(args) >= 3 ? args[3] : nothing
    mask_v = resolve_value_or_const(lc, mask_ref)

    n_lanes = prod(off.idx_shape)
    elem_mlir = mlir_elem_type(elem_T)
    idx_elem_t = eltype(IR.type(off.indices))

    last_old = nothing
    for lane in 0:(n_lanes - 1)
        pos_attr = IR.DenseArrayAttribute(Int64[lane])
        idx_scalar = IR.result(_vector.extract(off.indices, IR.Value[];
            result=idx_elem_t, static_position=pos_attr))
        val_scalar = IR.result(_vector.extract(val_v, IR.Value[];
            result=elem_mlir, static_position=pos_attr))

        if mask_v === nothing
            last_old = _emit_one_atomic_rmw!(off.base, idx_scalar, val_scalar,
                                             kind_kw, elem_T)
        else
            mask_bit = IR.result(_vector.extract(mask_v, IR.Value[];
                result=IR.Type(Bool), static_position=pos_attr))
            # `scf.if mask` containing the atomic; result is discarded (we
            # don't model the per-lane old-value tile on the masked path).
            then_region = IR.Region()
            else_region = IR.Region()
            then_block = IR.Block(IR.Type[], IR.Location[])
            else_block = IR.Block(IR.Type[], IR.Location[])
            push!(then_region, then_block)
            push!(else_region, else_block)
            @with_block then_block begin
                _emit_one_atomic_rmw!(off.base, idx_scalar, val_scalar,
                                      kind_kw, elem_T)
                _scf.yield(IR.Value[])
            end
            @with_block else_block begin
                _scf.yield(IR.Value[])
            end
            _scf.if_(mask_bit; results=IR.Type[],
                     thenRegion=then_region, elseRegion=else_region)
        end
    end
    # cuTile's RMW intrinsics return a tile of old values; we don't model
    # that result for the N-D path (no consumer of the SSA in the counter
    # tests). Return the last scalar's result for the 0-D-shape-1 case where
    # callers may forward it; an unused multi-lane result is dropped.
    return last_old
end

# Emit one `memref.generic_atomic_rmw` performing compare-and-swap:
#   if loaded == expected: yield desired
#   else:                  yield loaded
# Returns the prior (loaded) value, matching cuTile semantics.
function _emit_one_atomic_cas!(base_memref::IR.Value, idx_v::IR.Value,
                                expected_v::IR.Value, desired_v::IR.Value,
                                elem_T::Type)
    elem_mlir = mlir_elem_type(elem_T)
    idx_index = cast_to_index(idx_v)

    body_region = IR.Region()
    body_block = IR.Block(IR.Type[elem_mlir], IR.Location[IR.Location()])
    push!(body_region, body_block)

    @with_block body_block begin
        loaded = IR.argument(body_block, 1)
        # Equality predicate for arith.cmpi is code 0 (`eq`). Works for any
        # integer type; for floats we'd want arith.cmpf, but cuTile's
        # `atomic_cas` is typically used on integer locks / counters and
        # MLIR's atomic_rmw is signless integer anyway. If float CAS is
        # added later, branch on `elem_T <: AbstractFloat` here.
        if elem_T <: AbstractFloat
            # ordered equal = code 1 on arith.cmpf
            pred_attr = IR.Attribute(1, IR.Type(Int64))
            eq = IR.result(_arith.cmpf(loaded, expected_v; predicate=pred_attr))
        else
            pred_attr = IR.Attribute(0, IR.Type(Int64))
            eq = IR.result(_arith.cmpi(loaded, expected_v; predicate=pred_attr))
        end
        chosen = IR.result(_arith.select(eq, desired_v, loaded))
        _memref.atomic_yield(chosen)
    end

    return IR.result(_memref.generic_atomic_rmw(base_memref, IR.Value[idx_index];
                                                 result=elem_mlir,
                                                 atomic_body=body_region))
end

# Top-level dispatch for `Intrinsics.atomic_cas`.
# Args: (ptr_tile_ssa, expected, desired, mask, memory_order, memory_scope)
# For the 0-D scalar-index form we emit a single generic_atomic_rmw. For the
# N-D tile form we unroll one CAS per lane (optionally masked).
function emit_atomic_cas!(lc::LowerCtx, args, @nospecialize(typ))
    off_ref = args[1]
    off_ref isa SSAValue && haskey(lc.offsets, off_ref.id) ||
        error("atomic_cas: ptr-tile operand must be an Intrinsics.offset SSA, " *
              "got $off_ref")
    off = lc.offsets[off_ref.id]
    elem_T = off.elem_type

    expected_v = resolve_value_or_const(lc, args[2])
    expected_v === nothing &&
        error("atomic_cas: cannot resolve expected operand $(args[2])")
    desired_v = resolve_value_or_const(lc, args[3])
    desired_v === nothing &&
        error("atomic_cas: cannot resolve desired operand $(args[3])")

    if isempty(off.idx_shape)
        # 0-D form: scalar (i32) index. Single CAS.
        return _emit_one_atomic_cas!(off.base, off.indices,
                                      expected_v, desired_v, elem_T)
    end

    # N-D form: unroll one CAS per lane. cuTile broadcasts both expected and
    # desired to the index tile shape, so both arrive as vectors. Mask is
    # optional (bounds-check tile).
    mask_ref = length(args) >= 4 ? args[4] : nothing
    mask_v = resolve_value_or_const(lc, mask_ref)

    n_lanes = prod(off.idx_shape)
    elem_mlir = mlir_elem_type(elem_T)
    idx_elem_t = eltype(IR.type(off.indices))

    last_old = nothing
    for lane in 0:(n_lanes - 1)
        pos_attr = IR.DenseArrayAttribute(Int64[lane])
        idx_scalar = IR.result(_vector.extract(off.indices, IR.Value[];
            result=idx_elem_t, static_position=pos_attr))
        exp_scalar = IR.result(_vector.extract(expected_v, IR.Value[];
            result=elem_mlir, static_position=pos_attr))
        des_scalar = IR.result(_vector.extract(desired_v, IR.Value[];
            result=elem_mlir, static_position=pos_attr))

        if mask_v === nothing
            last_old = _emit_one_atomic_cas!(off.base, idx_scalar,
                                              exp_scalar, des_scalar, elem_T)
        else
            mask_bit = IR.result(_vector.extract(mask_v, IR.Value[];
                result=IR.Type(Bool), static_position=pos_attr))
            then_region = IR.Region()
            else_region = IR.Region()
            then_block = IR.Block(IR.Type[], IR.Location[])
            else_block = IR.Block(IR.Type[], IR.Location[])
            push!(then_region, then_block)
            push!(else_region, else_block)
            @with_block then_block begin
                _emit_one_atomic_cas!(off.base, idx_scalar,
                                       exp_scalar, des_scalar, elem_T)
                _scf.yield(IR.Value[])
            end
            @with_block else_block begin
                _scf.yield(IR.Value[])
            end
            _scf.if_(mask_bit; results=IR.Type[],
                     thenRegion=then_region, elseRegion=else_region)
        end
    end
    return last_old
end

# Emit one `memref.generic_atomic_rmw` performing a custom binary RMW (used
# for ops the upstream `memref.atomic_rmw` enum doesn't cover — currently
# `atomic_xor`). `binop` is one of the `_arith` dialect binary-op functions
# (e.g. `_arith.xori`). The region body computes `new = binop(loaded, val_v)`
# and yields it. Returns the prior (loaded) value.
function _emit_one_atomic_rmw_generic!(base_memref::IR.Value, idx_v::IR.Value,
                                         val_v::IR.Value, binop,
                                         elem_T::Type)
    elem_mlir = mlir_elem_type(elem_T)
    idx_index = cast_to_index(idx_v)

    body_region = IR.Region()
    body_block = IR.Block(IR.Type[elem_mlir], IR.Location[IR.Location()])
    push!(body_region, body_block)

    @with_block body_block begin
        loaded = IR.argument(body_block, 1)
        newval = IR.result(binop(loaded, val_v))
        _memref.atomic_yield(newval)
    end

    return IR.result(_memref.generic_atomic_rmw(base_memref, IR.Value[idx_index];
                                                  result=elem_mlir,
                                                  atomic_body=body_region))
end

# Top-level dispatch for RMW ops the upstream `memref.atomic_rmw` enum
# doesn't cover (currently `:xor`). Same args shape as
# `Intrinsics.atomic_{or,and,xor}` — (ptr_tile, val, mask, order, scope).
function emit_atomic_rmw_generic!(lc::LowerCtx, args, @nospecialize(typ),
                                    op::Symbol)
    op === :xor || error("atomic_rmw_generic: unsupported op $op")
    off_ref = args[1]
    off_ref isa SSAValue && haskey(lc.offsets, off_ref.id) ||
        error("atomic_$op: ptr-tile operand must be an Intrinsics.offset SSA, " *
              "got $off_ref")
    off = lc.offsets[off_ref.id]
    elem_T = off.elem_type
    elem_T <: AbstractFloat && error("atomic_xor: float element type not supported")
    binop = _arith.xori

    val_v = resolve_value_or_const(lc, args[2])
    val_v === nothing && error("atomic_$op: cannot resolve val operand $(args[2])")

    if isempty(off.idx_shape)
        return _emit_one_atomic_rmw_generic!(off.base, off.indices, val_v,
                                               binop, elem_T)
    end

    mask_ref = length(args) >= 3 ? args[3] : nothing
    mask_v = resolve_value_or_const(lc, mask_ref)

    n_lanes = prod(off.idx_shape)
    elem_mlir = mlir_elem_type(elem_T)
    idx_elem_t = eltype(IR.type(off.indices))

    last_old = nothing
    for lane in 0:(n_lanes - 1)
        pos_attr = IR.DenseArrayAttribute(Int64[lane])
        idx_scalar = IR.result(_vector.extract(off.indices, IR.Value[];
            result=idx_elem_t, static_position=pos_attr))
        val_scalar = IR.result(_vector.extract(val_v, IR.Value[];
            result=elem_mlir, static_position=pos_attr))

        if mask_v === nothing
            last_old = _emit_one_atomic_rmw_generic!(off.base, idx_scalar,
                                                       val_scalar, binop, elem_T)
        else
            mask_bit = IR.result(_vector.extract(mask_v, IR.Value[];
                result=IR.Type(Bool), static_position=pos_attr))
            then_region = IR.Region()
            else_region = IR.Region()
            then_block = IR.Block(IR.Type[], IR.Location[])
            else_block = IR.Block(IR.Type[], IR.Location[])
            push!(then_region, then_block)
            push!(else_region, else_block)
            @with_block then_block begin
                _emit_one_atomic_rmw_generic!(off.base, idx_scalar, val_scalar,
                                                binop, elem_T)
                _scf.yield(IR.Value[])
            end
            @with_block else_block begin
                _scf.yield(IR.Value[])
            end
            _scf.if_(mask_bit; results=IR.Type[],
                     thenRegion=then_region, elseRegion=else_region)
        end
    end
    return last_old
end

# ----------------------------------------------------------------------------
# Tile load / store
# ----------------------------------------------------------------------------

function emit_tile_load!(lc::LowerCtx, part::PartitionInfo,
                         index_components::Vector{Any})
    idx_t = IR.IndexType()
    rank = length(part.tile_shape)
    length(index_components) == rank ||
        error("tile load: rank mismatch (partition rank=$rank, " *
              "got $(length(index_components)) indices)")

    # cuTile gives indices and tile_shape in Julia col-major order; MLIR
    # transfer_read takes indices in row-major (slowest dim first). Build
    # the offsets in Julia order then reverse for MLIR.
    offs_julia = IR.Value[]
    for k in 1:rank
        bid_v = resolve_value_or_const(lc, index_components[k])
        bid_v === nothing && error("tile load: cannot resolve index component $k")
        bid_idx = cast_to_index(bid_v)
        tile_c = IR.result(_arith.constant(;
            value=IR.Attribute(part.tile_shape[k], idx_t)))
        push!(offs_julia, IR.result(_arith.muli(bid_idx, tile_c; result=idx_t)))
    end
    offs = rank == 1 ? offs_julia : reverse(offs_julia)

    elem_t = mlir_elem_type(part.elem_type)
    # MLIR vector shape is row-major (slowest dim first).
    vec_t = IR.VectorType(rank, reverse(part.tile_shape), elem_t)
    padding = IR.result(_arith.constant(;
        value=IR.Attribute(zero(part.elem_type))))
    perm = IR.Attribute(IR.IdentityAffineMap(rank))
    inb  = IR.Attribute(IR.Attribute[IR.Attribute(true) for _ in 1:rank])
    return IR.result(_vector.transfer_read(
        part.base, offs, padding;
        vector=vec_t, permutation_map=perm, in_bounds=inb,
    ))
end

function emit_tile_store!(lc::LowerCtx, part::PartitionInfo,
                          value::IR.Value, index_components::Vector{Any})
    idx_t = IR.IndexType()
    rank = length(part.tile_shape)
    offs_julia = IR.Value[]
    for k in 1:rank
        bid_v = resolve_value_or_const(lc, index_components[k])
        bid_v === nothing && error("tile store: cannot resolve index component $k")
        bid_idx = cast_to_index(bid_v)
        tile_c = IR.result(_arith.constant(;
            value=IR.Attribute(part.tile_shape[k], idx_t)))
        push!(offs_julia, IR.result(_arith.muli(bid_idx, tile_c; result=idx_t)))
    end
    offs = rank == 1 ? offs_julia : reverse(offs_julia)
    perm = IR.Attribute(IR.IdentityAffineMap(rank))
    inb  = IR.Attribute(IR.Attribute[IR.Attribute(true) for _ in 1:rank])
    _vector.transfer_write(value, part.base, offs;
                            permutation_map=perm, in_bounds=inb)
    return nothing
end

# ----------------------------------------------------------------------------
# SPMD-mode plain-Julia op emitters
# ----------------------------------------------------------------------------
#
# These handle the SCI ops that arise from Julia's `Vector{T}[i]` /
# `Vector{T}[i] = v` lowering when the kernel is written without cuTile
# types. The IRStructurizer / cuTile pipeline accepts plain Julia
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
# Indices arriving here are 1-based (Julia semantics); cuTile lowers them
# to 0-based in `memoryrefnew` itself, but on plain Julia code we get the
# raw 1-based index — we subtract 1 below.

# Detect contiguous lane indices. The SPMD lane vector is constructed as
# `splat(bid * W) + step(0..W-1)` — an affine, contiguous pattern. The
# fast path: convert to a `vector.transfer_read` / `vector.transfer_write`
# with the (scalar) base offset; falls back to gather/scatter otherwise.
# For the MVP we detect the contiguous case purely by checking whether the
# index vector equals `lc.arg_vals[lane_arg]` — i.e. the user's `%i` is the
# lane arg directly (the common `a[i]` case). Anything else (computed
# offsets, gather-style indirection) falls back to gather/scatter.
function _is_contiguous_lane_index(lc::LowerCtx, indices::IR.Value)
    lc.spmd || return false
    haskey(lc.arg_vals, lc.lane_arg) || return false
    return indices == lc.arg_vals[lc.lane_arg]
end

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

# `Base.memoryrefnew(ref_ssa, idx, bc)`
function emit_spmd_memoryrefnew!(lc::LowerCtx, idx::Int, args, @nospecialize(typ))
    ref_ref = args[1]
    ref_ref isa SSAValue && haskey(lc.field_refs, ref_ref.id) ||
        error("SPMD memoryrefnew: ref operand must be `getfield(arr, :ref)`, got $ref_ref")
    (arg_id, fld) = lc.field_refs[ref_ref.id]
    fld === :ref || error(
        "SPMD memoryrefnew: ref operand must be `getfield(arr, :ref)` (got :$fld)")
    base_memref = lc.arg_vals[arg_id]
    elem_T = lc.arg_elem_types[arg_id]

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
