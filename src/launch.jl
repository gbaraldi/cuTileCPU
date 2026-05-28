# Host-side launch: cpu_function (mirrors cuTile.cufunction) + the
# CPUKernel callable that ccalls into the JIT'd entry point.

# MLIR memref C-interface descriptor for rank-N memrefs:
#   struct MemRefDesc{T, N} {
#     T* allocated;
#     T* aligned;
#     intptr_t offset;
#     intptr_t sizes[N];
#     intptr_t strides[N];
#   }
# This is the layout `_mlir_ciface_*` expects. We construct it as a flat
# `Vector{Int}` (well-typed at the binding point) since the field count
# varies with rank — easier than generating per-rank structs.

"""
    pack_memref_descriptor(a::AbstractArray{T,N}) -> Vector{Int}

Build the byte-image of the MLIR memref descriptor for `a` (rank `N`, eltype
`T`). The descriptor is a flat sequence:
`[allocated_ptr, aligned_ptr, offset, sizes..., strides...]`, each entry the
size of a host pointer / `intptr_t`. Used by [`launch_so!`](@ref).

Strides are in MLIR (row-major) order — the last entry is the fastest-moving
dim. For a Julia column-major `Array`, that means `strides(a)` reversed.
"""
function pack_memref_descriptor(a::AbstractArray{T,N}) where {T,N}
    p = pointer(a)
    desc = Vector{Int}(undef, 3 + 2N)
    desc[1] = Int(UInt(p))           # allocated (same as aligned for unsafe_wrap'd arrays)
    desc[2] = Int(UInt(p))           # aligned
    desc[3] = 0                      # offset
    sz = size(a)
    st = strides(a)
    # MLIR uses row-major: dim 0 is slowest. Julia col-major: dim 1 is fastest.
    # So MLIR dim k ↔ Julia dim N-k+1. Reverse both shape and strides.
    for k in 1:N
        jk = N - k + 1
        desc[3 + k]     = sz[jk]
        desc[3 + N + k] = st[jk]
    end
    return desc
end

# Pre-compute and cache compiled kernels per (f, argtypes, n_grid_dims).
struct CPUKernel{F, TT}
    f::F
    so_path::String
    so_handle::Ptr{Cvoid}
    fn::Ptr{Cvoid}
    param_types::Vector{Type}       # Julia types of each func param
    param_kinds::Vector{Symbol}     # :memref or :scalar, parallel to param_types
    n_grid_dims::Int
    # Per-memref-arg required alignment (parallel to memref entries of
    # param_types). Empty for scalar params.
    param_alignments::Vector{Int}
    # :cuTile (default — kernels with TileArray args + KernelState seed)
    # or :spmd  (scalar-typed Julia kernels lifted to vector lanes; no
    #            alignment check, no seed param, see `spmd_function`).
    kind::Symbol
end

# Compilation cache: (f, tt, n_grid_dims, serial) → CPUKernel. The `serial`
# bool keys the OpenMP-or-not variant separately so users can keep both
# compiled.
const _kernel_cache = Dict{Tuple{Any, Type, Int, Bool}, CPUKernel}()

# Run cuTile's full pipeline and capture the dataflow analyses' results.
# Mirrors `ct.code_structured(f, argtypes; optimize=true)` but keeps the
# (divby_info, bounds_info) tuple `run_passes!` returns — `code_structured`
# discards them. The walker uses them to emit per-stride `llvm.intr.assume`
# annotations on TileArray kernel args at func entry, which is the alignment
# proof LLVM's vectorizer needs to widen non-leading-dim accesses.
function _structured_with_analyses(@nospecialize(f), @nospecialize(argtypes))
    stripped, const_argtypes = ct.process_const_argtypes(f, argtypes)
    mi = ct.lookup_method_instance(f, stripped)
    cache = ct.CacheView{ct.CuTileResults}(:cuTile, Base.get_world_counter())
    ir, rettype = ct.emit_julia(cache, mi; const_argtypes)
    sci, rettype, _ = ct.emit_structured(ir, rettype)
    sci = copy(sci)
    divby_info, bounds_info = ct.run_passes!(sci)
    return sci, rettype, divby_info, bounds_info
end

"""
    cpu_function(f, argtypes::Tuple{...}; n_grid_dims=1) -> CPUKernel

Compile `f` for the CPU backend. `argtypes` is the tuple of *runtime* argument
types — exactly what `ct.cufunction` expects after `cuTileconvert`. For host
adoption convenience this overload also accepts a tuple of *values*:

    cpu_function(vadd, (a, b, c, ct.Constant(16)))

which derives the types via `Core.Typeof` after `cuTileconvert`.

Compilation is cached on `(f, argtypes, n_grid_dims)`.
"""
function cpu_function(@nospecialize(f), argtypes::Type;
                      n_grid_dims::Int=1, kernel_name=string(nameof(f)),
                      serial::Bool=false)
    # `serial` toggles the lowering pipeline: false (default) emits OpenMP
    # parallel-for over the grid; true compiles a single-threaded .so with
    # `scf.parallel` lowered directly to `scf.for`. Each variant lives in its
    # own cache entry so users can flip between them without recompilation
    # blowing away the other.
    key = (f, argtypes, n_grid_dims, serial)
    haskey(_kernel_cache, key) && return _kernel_cache[key]::CPUKernel

    sci, rettype, divby_info, bounds_info = _structured_with_analyses(f, argtypes)
    rettype === Nothing ||
        error("cpu_function: kernel must return Nothing, got $rettype")

    mod, param_julia_types, mlir_ctx, param_kinds =
        lower_to_mlir(sci, argtypes; kernel_name, n_grid_dims,
                      divby_info, bounds_info)

    passes = serial ? SERIAL_PASSES : DEFAULT_PASSES
    so_path = compile_module_to_so(mod, mlir_ctx; kernel_name, passes)
    h = Libdl.dlopen(so_path)
    fn = Libdl.dlsym(h, Symbol("_mlir_ciface_" * kernel_name))

    alignments = Int[]
    for (i, T) in enumerate(param_julia_types)
        if param_kinds[i] === :memref
            push!(alignments, Int(T.parameters[3].alignment))
        end
    end
    k = CPUKernel{typeof(f), argtypes}(f, so_path, h, fn,
                                       param_julia_types, param_kinds,
                                       n_grid_dims, alignments, :cuTile)
    _kernel_cache[key] = k
    return k
end

# Tuple-of-values overload: derives argtypes via cuTileconvert.
function cpu_function(@nospecialize(f), args::Tuple; kwargs...)
    converted = map(_cpu_convert, args)
    tt = Tuple{map(Core.Typeof, converted)...}
    return cpu_function(f, tt; kwargs...)
end

# Mirror cuTile.KernelAdaptor for host buffers: AbstractArray → TileArray.
_cpu_convert(x::AbstractArray) = ct.TileArray(x)
_cpu_convert(t::Type) = ct.Constant(t)
_cpu_convert(x) = x

"""
    (k::CPUKernel)(args...; blocks)

Launch `k` with `args`. `blocks` is the grid shape (`Int` for 1-D,
`NTuple{N,Int}` for N-D). Memref-typed args must be `AbstractArray`s aligned
to the kernel's `ArraySpec.alignment` — use [`aligned_array`](@ref) to obtain
buffers that meet the requirement.
"""
function (k::CPUKernel)(args...; blocks)
    grid = blocks isa Integer ? (Int(blocks),) : Tuple(Int.(blocks))
    length(grid) == k.n_grid_dims ||
        error("CPUKernel: kernel was compiled for $(k.n_grid_dims) grid dim(s), " *
              "launch passed $(length(grid))")

    is_spmd = k.kind === :spmd
    is_ka   = k.kind === :ka

    # Walk runtime args, matching them against the kernel's expected param
    # kinds. AbstractArray → memref descriptor. Constants are dropped. Scalar
    # Numbers → typed scalar arg.
    #
    # SPMD-mode special case: the trailing `i::Int` lane index is *not* a
    # runtime arg — it's synthesised inside the func per grid step. The
    # user's invocation still passes it (matching the source-level kernel
    # signature) but we silently drop it from the host-side ccall.
    descs = Vector{Vector{Int}}()
    scalar_vals = Any[]
    scalar_types = Type[]
    pin_targets = Any[]
    param_idx = 1
    align_idx = 1
    last_arg_idx = lastindex(args)
    for (ai, a) in enumerate(args)
        a isa ct.Constant && continue
        # SPMD: drop the trailing scalar (the lane index `i::Int`).
        if is_spmd && a isa Integer && ai == last_arg_idx
            continue
        end
        if a isa AbstractArray
            param_idx ≤ length(k.param_kinds) ||
                error("CPUKernel: too many runtime args")
            k.param_kinds[param_idx] === :memref ||
                error("CPUKernel: param #$param_idx expected scalar, got array")
            # Alignment check: cuTile-mode always checks (per-arg from
            # ArraySpec); SPMD-mode checks only when the user requested an
            # alignment > Julia's default 16 (param_alignments populated).
            if !(is_spmd || is_ka) || align_idx ≤ length(k.param_alignments)
                align = k.param_alignments[align_idx]
                pointer_aligned(a, align) || error(
                    "CPUKernel: array arg #$(align_idx) (pointer $(repr(pointer(a)))) is not " *
                    "aligned to $(align) bytes" *
                    (is_spmd ? " required by spmd_function(...; alignment=$(align))" :
                     is_ka   ? " required by ka_function(...; alignment=$(align))" :
                               " required by ArraySpec") *
                    ". Allocate with `cuTileCPU.aligned_array(...; alignment=$(align))`.")
            end
            push!(descs, pack_memref_descriptor(a))
            push!(pin_targets, a)
            align_idx += 1
            param_idx += 1
        elseif a isa Number
            param_idx ≤ length(k.param_kinds) ||
                error("CPUKernel: too many runtime args")
            k.param_kinds[param_idx] === :scalar ||
                error("CPUKernel: param #$param_idx expected memref, got scalar")
            T = k.param_types[param_idx]
            push!(scalar_vals, convert(T, a))
            push!(scalar_types, T)
            param_idx += 1
        else
            error("CPUKernel: unsupported arg of type $(typeof(a))")
        end
    end
    param_idx - 1 == length(k.param_kinds) ||
        error("CPUKernel: not enough runtime args (expected $(length(k.param_kinds)), got $(param_idx - 1))")

    Nm = length(descs)
    Ng = k.n_grid_dims
    desc_ptrs = ntuple(i -> pointer(descs[i]), Nm)

    # Small-grid escape hatch. OpenMP's fork/join floor is ~30–60 μs on this
    # box; when the grid has fewer blocks than the OMP worker pool, the idle
    # workers still incur fork/join cost. Setting nthreads to
    # min(grid, MAX_THREADS) for tiny launches collapses idle workers.
    #
    # `omp_set_num_threads` takes effect for the *next* parallel region only.
    # We restore to `MAX_THREADS` (the value captured ONCE at module init —
    # NOT `omp_get_max_threads()`, which reflects the *current* nthreads
    # setting and would drift to whatever the last small call left).
    n_blocks_total = prod(grid)
    desired = Cint(min(MAX_THREADS, max(1, n_blocks_total)))
    rescale = desired != MAX_THREADS
    rescale && ccall((:omp_set_num_threads, LIBOMP), Cvoid, (Cint,), desired)
    # Implicit trailing KernelState.seed: a fresh `rand(UInt32)` per launch so
    # consecutive launches see distinct seeds. Mirrors cuTile's bytecode
    # launch (cuTile.jl/src/launch.jl, `_flatten_static!`'s `state.seed`).
    # Kernels that don't reference `Intrinsics.kernel_state()` get an unused
    # i32 param — LLVM optimizes it away in the function body but the host
    # still has to pass it to match the ABI.
    seed = Base.rand(UInt32)
    GC.@preserve descs pin_targets begin
        if is_spmd || is_ka
            # Same ABI as SPMD: memref descriptors + uniform scalars + grid;
            # no implicit seed slot (kernels written for the KA / SPMD paths
            # don't have access to `Intrinsics.kernel_state()`).
            _ccall_launch_spmd(k.fn, desc_ptrs, Tuple(scalar_vals),
                               Tuple(scalar_types), grid)
        else
            _ccall_launch(k.fn, desc_ptrs, Tuple(scalar_vals),
                          Tuple(scalar_types), seed, grid)
        end
    end
    rescale && ccall((:omp_set_num_threads, LIBOMP), Cvoid, (Cint,), Cint(MAX_THREADS))
    return nothing
end

# Path to libomp for direct OMP runtime calls. Resolved once from the same
# LLVMOpenMP_jll the JIT'd .so is linked against.
const LIBOMP = libomp_path

# OpenMP worker-pool max-threads, captured *once* at module load. We use this
# as the "restore to" value after small-grid launches. Reading
# `omp_get_max_threads` repeatedly is unsafe — its result reflects the
# current `omp_set_num_threads` setting, so it'd drift each time we drop
# nthreads for a small launch.
const MAX_THREADS = Int(ccall((:omp_get_max_threads, libomp_path), Cint, ()))

# Dispatches per (Nm, scalar tuple types, Ng). The generated body emits a
# static-typed ccall. `scalar_vals` is a heterogeneous Tuple{...} of the
# scalar arg values and `scalar_types` is a Tuple{Type, Type, ...} of the
# corresponding ccall types. The two tuples must have equal length.
# =============================================================================
# Explicit parallel-for surface
# =============================================================================

"""
    parallel_for(f, args; blocks) -> nothing

Launch `f(args...)` with an explicit parallel grid. `blocks` is the grid
extent — `Int` for a 1-D grid, `NTuple{N,Int}` for higher-rank grids. Compiles
on first use; subsequent calls with the same `(f, argtypes, n_grid_dims)`
reuse the cached compilation.

The cuTile-CPU analogue of CUDA's `@cuda blocks=N kernel(args...)`, surfaced
as an explicit parallel-for: the grid is the foreach, and one ccall per
launch dispatches all blocks via MLIR's OpenMP lowering. Internally identical
to `cpu_function(f, args)(args...; blocks=blocks)` — provided here so kernels
read closer to "for bid in 1:N, run this kernel" than to a CUDA macro.

# Example
```julia
n = 1024; tile = 16
a = aligned_array(Float32, n; alignment=128)
b = aligned_array(Float32, n; alignment=128)
c = aligned_array(Float32, n; alignment=128)
parallel_for(vadd, (a, b, c, ct.Constant(tile)); blocks = n ÷ tile)
```
"""
function parallel_for(@nospecialize(f), args::Tuple; blocks, serial::Bool=false)
    grid = blocks isa Integer ? (Int(blocks),) : Tuple(Int.(blocks))
    k = cpu_function(f, args; n_grid_dims=length(grid), serial)
    return k(args...; blocks=grid)
end

"""
    @parallel_for blocks=N  f(args...)
    @parallel_for blocks=(Nx, Ny[, Nz])  f(args...)
    @parallel_for blocks=N  serial=true  f(args...)

Macro form of [`parallel_for`](@ref). Trailing argument is the kernel call;
preceding arguments are `kw=val` settings (`blocks=` is required; `serial=`
optional). Equivalent to `parallel_for(f, (args...,); blocks=..., serial=...)`.

```julia
@parallel_for blocks = n ÷ 16              vadd(a, b, c, ct.Constant(16))
@parallel_for blocks = (M, N)              gemm(a, b, c, ct.Constant(32))
@parallel_for blocks = n ÷ 16  serial=true vadd(a, b, c, ct.Constant(16))
```
"""
macro parallel_for(macro_args...)
    isempty(macro_args) &&
        error("@parallel_for: need at least a kernel call expression")
    call_expr = macro_args[end]
    kw_exprs = macro_args[1:end-1]

    Meta.isexpr(call_expr, :call) ||
        error("@parallel_for: last argument must be a kernel call expression")

    blocks_val = nothing
    serial_val = false
    for kw in kw_exprs
        Meta.isexpr(kw, :(=)) && kw.args[1] isa Symbol ||
            error("@parallel_for: leading args must be `kw = val`, got $kw")
        if kw.args[1] === :blocks
            blocks_val = kw.args[2]
        elseif kw.args[1] === :serial
            serial_val = kw.args[2]
        else
            error("@parallel_for: unsupported keyword `$(kw.args[1])`")
        end
    end
    blocks_val === nothing && error("@parallel_for: `blocks = …` is required")

    f = call_expr.args[1]
    args = call_expr.args[2:end]
    return quote
        parallel_for($(esc(f)), ($(map(esc, args)...),);
                     blocks = $(esc(blocks_val)),
                     serial = $(esc(serial_val)))
    end
end

# Dispatches per (Nm, scalar tuple types, Ng). The generated body emits a
# static-typed ccall. `scalar_vals` is a heterogeneous Tuple{...} of the
# scalar arg values and `scalar_types` is a Tuple{Type, Type, ...} of the
# corresponding ccall types. The two tuples must have equal length.
@generated function _ccall_launch(fn::Ptr{Cvoid},
                                  descs::NTuple{Nm, Ptr{Int}},
                                  scalar_vals::Tuple,
                                  scalar_types::Tuple,
                                  seed::UInt32,
                                  grid::NTuple{Ng, Int}) where {Nm, Ng}
    Ns = length(scalar_vals.parameters)
    # Build ccall arg type tuple: Ptr{Int} per descriptor, then scalar Julia
    # types (from scalar_types), then UInt32 for the implicit KernelState
    # seed, then Int per grid dim.
    scalar_t_exprs = [:(scalar_types[$i]) for i in 1:Ns]
    desc_args = [:(descs[$i]) for i in 1:Nm]
    scalar_args = [:(scalar_vals[$i]) for i in 1:Ns]
    grid_args = [:(grid[$i]) for i in 1:Ng]
    # ccall needs a literal tuple of types in its type slot; the actual
    # element type at each scalar position comes from scalar_types — we splice
    # them via $(scalar_types[i]) at *runtime* by lifting to a generated
    # expression. Since scalar_types is itself a Tuple{...}, we can read its
    # parameters here at @generated time:
    scalar_t_lits = [scalar_vals.parameters[i] for i in 1:Ns]
    types_expr = Expr(:tuple,
                      fill(:(Ptr{Int}), Nm)...,
                      scalar_t_lits...,
                      :(UInt32),
                      fill(:(Int), Ng)...)
    return quote
        ccall(fn, Cvoid, $types_expr, $(desc_args...), $(scalar_args...), seed, $(grid_args...))
    end
end

# SPMD ccall variant: no `seed` parameter (SPMD kernels don't take the
# implicit `KernelState.seed`). Otherwise identical layout to `_ccall_launch`.
@generated function _ccall_launch_spmd(fn::Ptr{Cvoid},
                                       descs::NTuple{Nm, Ptr{Int}},
                                       scalar_vals::Tuple,
                                       scalar_types::Tuple,
                                       grid::NTuple{Ng, Int}) where {Nm, Ng}
    Ns = length(scalar_vals.parameters)
    desc_args = [:(descs[$i]) for i in 1:Nm]
    scalar_args = [:(scalar_vals[$i]) for i in 1:Ns]
    grid_args = [:(grid[$i]) for i in 1:Ng]
    scalar_t_lits = [scalar_vals.parameters[i] for i in 1:Ns]
    types_expr = Expr(:tuple,
                      fill(:(Ptr{Int}), Nm)...,
                      scalar_t_lits...,
                      fill(:(Int), Ng)...)
    return quote
        ccall(fn, Cvoid, $types_expr, $(desc_args...), $(scalar_args...), $(grid_args...))
    end
end

# ============================================================================
# SPMD ("ISPC-style" scalar-typed kernels)
# ============================================================================
#
# `spmd_function(f, argtypes; lane_width=16)` is the SPMD analogue of
# `cpu_function`. The kernel writes plain Julia (no `Tile`/`ct.*` types):
#
#     function vadd_spmd(a, b, c, i::Int)
#         @inbounds c[i] = a[i] + b[i]
#         return
#     end
#
# Each grid block processes `lane_width` consecutive lane indices i in
# `bid*W+1 : bid*W+W` simultaneously, by lifting every op on the lane index
# to a `vector<lane_width × eT>`. The host-side launch grid is
# `cld(n, lane_width)` blocks; the kernel signature still names `i::Int` so
# the caller's invocation reads naturally, but the actual lane index value
# is synthesised inside the func and the user-supplied `i` arg is *ignored*
# at launch time (a placeholder; pass any Int).
#
# MVP limitations:
#   • The grid size must be a multiple of `lane_width` (no boundary mask).
#   • Only the contiguous `a[i]` pattern is lowered to `vector.transfer_*`;
#     anything more elaborate falls through to `vector.gather`/scatter.
#
# Alignment hints: SPMD accepts an optional `alignment` kwarg. When >16,
# each array arg gets a `memref.assume_alignment %arg, N` at func entry and
# a `strided<[1, ?, …]>` layout in the memref type — the same alignment-
# proof machinery the cuTile path inherits from `ArraySpec`. At DRAM-scale
# memory-bandwidth-bound workloads this is the difference between
# `vmovaps` and `vmovups` in the inner loop. Users must pass buffers that
# actually meet the alignment (e.g. via `aligned_array(T, n; alignment=N)`);
# the launcher checks at launch time.
const _spmd_kernel_cache = Dict{Tuple{Any, Type, Int, Bool, Int, Int}, CPUKernel}()

"""
    spmd_function(f, argtypes::Type; lane_width=16, alignment=16, serial=false) -> CPUKernel

Compile `f` in SPMD ("ISPC-style") mode: the trailing scalar arg is treated
as a *lane index* and the body is lifted to a `lane_width`-wide vector. See
the module-level docstring for details.

`alignment` (bytes) is an optional hint emitted as `memref.assume_alignment`
on each array arg. The default of 16 matches what Julia's GC guarantees for
`Vector{T}`; pass 64/128 to get aligned vector loads at DRAM scale, but the
caller must then supply buffers that actually meet the alignment (use
[`aligned_array`](@ref)).

Compilation is cached on `(f, argtypes, lane_width, alignment, serial)`.
"""
function spmd_function(@nospecialize(f), argtypes::Type;
                       lane_width::Int=16,
                       alignment::Int=16,
                       kernel_name=string(nameof(f), "_spmd"),
                       serial::Bool=false)
    key = (f, argtypes, 1, serial, lane_width, alignment)
    haskey(_spmd_kernel_cache, key) && return _spmd_kernel_cache[key]::CPUKernel

    # Standalone inference — no cuTile interpreter. Plain-Julia SPMD kernels
    # use no cuTile tile intrinsics, so Frontend.structured (own interpreter,
    # own Intrinsics, default opt params) is all that's needed.
    sci, rettype = Frontend.structured(f, argtypes)
    rettype === Nothing ||
        error("spmd_function: kernel must return Nothing, got $rettype")

    mod, param_julia_types, mlir_ctx, param_kinds =
        lower_to_mlir_spmd(sci, argtypes; kernel_name, lane_width, alignment)

    passes = serial ? SERIAL_PASSES : DEFAULT_PASSES
    so_path = compile_module_to_so(mod, mlir_ctx; kernel_name, passes)
    h = Libdl.dlopen(so_path)
    fn = Libdl.dlsym(h, Symbol("_mlir_ciface_" * kernel_name))

    # Per-arg alignment for runtime check at launch. SPMD has the same
    # alignment for every memref arg (`alignment` kwarg) when > 16; below
    # that we skip the check (Julia's GC guarantees 16 already).
    nmemref = count(==(:memref), param_kinds)
    alignments = alignment > 16 ? fill(alignment, nmemref) : Int[]

    k = CPUKernel{typeof(f), argtypes}(f, so_path, h, fn,
                                       param_julia_types, param_kinds,
                                       1, alignments, :spmd)
    _spmd_kernel_cache[key] = k
    return k
end

# Tuple overload: accepts either a tuple of Julia *types* (e.g.
# `(Vector{Float32}, Int)`) or a tuple of runtime *values*. SPMD doesn't
# apply `cuTileconvert`; args stay as their plain Julia types.
function spmd_function(@nospecialize(f), args::Tuple; kwargs...)
    if all(a -> a isa Type, args)
        tt = Tuple{args...}
    else
        tt = Tuple{map(Core.Typeof, args)...}
    end
    return spmd_function(f, tt; kwargs...)
end

# Cache for `ka_function`. Key shape mirrors `_spmd_kernel_cache`:
# (f, argtypes, n_grid_dims, serial, lane_width, alignment).
const _ka_kernel_cache = Dict{Tuple{Any, Type, Int, Bool, Int, Int}, CPUKernel}()

"""
    ka_function(f, argtypes::Type; lane_width=16, alignment=16, serial=false) -> CPUKernel

Compile a KernelAbstractions-style `gpu_*` kernel body via cuTileCPU's MLIR
pipeline. `argtypes` is `Tuple{CtxType, ArgTypes...}` where the first slot
is a `KernelAbstractions.CompilerMetadata{…}` type — that slot is *consumed*
by the KA-intrinsic overlays (see `ext/KernelAbstractionsExt.jl`) and is
not materialised as an MLIR parameter.

Used internally by `(::Kernel{cuTileBackend})(...)`; users normally don't
call this directly.
"""
function ka_function(@nospecialize(f), argtypes::Type;
                     lane_width::Int=16,
                     alignment::Int=16,
                     kernel_name=string(nameof(f), "_ka"),
                     serial::Bool=false)
    key = (f, argtypes, 1, serial, lane_width, alignment)
    haskey(_ka_kernel_cache, key) && return _ka_kernel_cache[key]::CPUKernel

    # Standalone inference via Frontend (KA overlays live in
    # Frontend.METHOD_TABLE — see ext/KernelAbstractionsExt.jl). No cuTile.
    sci, rettype = Frontend.structured(f, argtypes)
    rettype === Nothing ||
        error("ka_function: kernel must return Nothing, got $rettype")

    mod, param_julia_types, mlir_ctx, param_kinds =
        lower_to_mlir_ka(sci, argtypes; kernel_name, lane_width, alignment)

    passes = serial ? SERIAL_PASSES : DEFAULT_PASSES
    so_path = compile_module_to_so(mod, mlir_ctx; kernel_name, passes)
    h = Libdl.dlopen(so_path)
    fn = Libdl.dlsym(h, Symbol("_mlir_ciface_" * kernel_name))

    nmemref = count(==(:memref), param_kinds)
    alignments = alignment > 16 ? fill(alignment, nmemref) : Int[]

    # We reuse the `:spmd` kind for launch dispatch — the launcher's SPMD
    # path drops a trailing Integer arg from the user-facing call. KA
    # kernels don't have that trailing arg (the lane is synthesized in
    # MLIR), so the user-call signature is `(args...; blocks)` with no
    # placeholder — but cpu_function's launcher loops over args and skips
    # the trailing Integer ONLY for SPMD. We set kind = :ka and add the
    # corresponding branch in the launcher.
    k = CPUKernel{typeof(f), argtypes}(f, so_path, h, fn,
                                       param_julia_types, param_kinds,
                                       1, alignments, :ka)
    _ka_kernel_cache[key] = k
    return k
end
