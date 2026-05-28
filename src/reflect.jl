# Reflection — mirror cuTile's `code_*` for the CPU backend.
#
# These bypass the launch / dlopen path and stop at intermediate IRs, so
# they're useful for debugging codegen without needing libomp + clang on
# every call.

"""
    code_mlir(f, argtypes; kernel_name=nameof(f), n_grid_dims=1) -> String

The MLIR module emitted from `f`'s StructuredIRCode, **before** the lowering
pipeline runs. Use this to inspect what the cuTile→MLIR walker produces.

Like cuTile's `code_tiled`, accepts either a `Tuple{T1, T2, …}` type or a
tuple of runtime values (in which case `cuTileconvert` is applied).
"""
function code_mlir(@nospecialize(f), argtypes::Type;
                   kernel_name::String=string(nameof(f)), n_grid_dims::Int=1,
                   spmd::Bool=false, lane_width::Int=16, alignment::Int=16)
    if spmd
        # SPMD reflection uses the standalone Frontend (no cuTile inference),
        # matching the spmd_function launch path.
        sci, rettype = Frontend.structured(f, argtypes)
        mod, _, mlir_ctx, _ = lower_to_mlir_spmd(sci, argtypes;
                                                  kernel_name, lane_width, alignment)
    else
        sci, rettype, divby_info, bounds_info = _structured_with_analyses(f, argtypes)
        mod, _, mlir_ctx, _ = lower_to_mlir(sci, argtypes; kernel_name, n_grid_dims,
                                            divby_info, bounds_info)
    end
    @with_context mlir_ctx begin
        return sprint(show, mod)
    end
end

function code_mlir(@nospecialize(f), args::Tuple; spmd::Bool=false, kwargs...)
    if spmd
        # Accept either a tuple of Julia *types* or a tuple of values.
        tt = if all(a -> a isa Type, args)
            Tuple{args...}
        else
            Tuple{map(Core.Typeof, args)...}
        end
        return code_mlir(f, tt; spmd=true, kwargs...)
    end
    converted = map(_cpu_convert, args)
    tt = Tuple{map(Core.Typeof, converted)...}
    return code_mlir(f, tt; kwargs...)
end

"""
    code_mlir_lowered(f, argtypes; kwargs...) -> String

The MLIR module **after** the standard CPU lowering pipeline — LLVM-dialect
MLIR, ready for `mlir-translate --mlir-to-llvmir`.
"""
function code_mlir_lowered(@nospecialize(f), argtypes::Type;
                           kernel_name::String=string(nameof(f)),
                           n_grid_dims::Int=1, passes=DEFAULT_PASSES,
                           spmd::Bool=false, lane_width::Int=16,
                           alignment::Int=16)
    mlir = code_mlir(f, argtypes; kernel_name, n_grid_dims,
                     spmd, lane_width, alignment)
    return lower_mlir_text(mlir; passes)
end

function code_mlir_lowered(@nospecialize(f), args::Tuple; spmd::Bool=false, kwargs...)
    if spmd
        tt = all(a -> a isa Type, args) ? Tuple{args...} : Tuple{map(Core.Typeof, args)...}
        return code_mlir_lowered(f, tt; spmd=true, kwargs...)
    end
    converted = map(_cpu_convert, args)
    tt = Tuple{map(Core.Typeof, converted)...}
    return code_mlir_lowered(f, tt; kwargs...)
end

"""
    code_llvm(f, argtypes; kwargs...) -> String

The LLVM IR (textual `.ll`) emitted for `f`. Produced by running the lowering
pipeline through `mlir-translate --mlir-to-llvmir`. Equivalent to
`Base.code_llvm` in spirit, but for the cuTile-CPU pipeline.

Accepts the same SPMD-mode kwargs (`spmd`, `lane_width`, `alignment`) as
`code_mlir`.
"""
function code_llvm(@nospecialize(f), argtypes::Type;
                   kernel_name::String=string(nameof(f)),
                   n_grid_dims::Int=1, passes=DEFAULT_PASSES,
                   spmd::Bool=false, lane_width::Int=16, alignment::Int=16)
    lowered = code_mlir_lowered(f, argtypes; kernel_name, n_grid_dims, passes,
                                spmd, lane_width, alignment)
    return translate_to_llvmir(lowered)
end

function code_llvm(@nospecialize(f), args::Tuple; spmd::Bool=false, kwargs...)
    if spmd
        tt = all(a -> a isa Type, args) ? Tuple{args...} : Tuple{map(Core.Typeof, args)...}
        return code_llvm(f, tt; spmd=true, kwargs...)
    end
    converted = map(_cpu_convert, args)
    tt = Tuple{map(Core.Typeof, converted)...}
    return code_llvm(f, tt; kwargs...)
end
