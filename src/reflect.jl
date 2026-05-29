# Reflection for the CPU SPMD path — the MLIR/LLVM IR emitted from a plain-Julia
# kernel via the standalone Frontend (no cuTile). For the GPU SIMT path see
# `code_gpu` (MLIRCUDAExt). These stop at intermediate IRs, useful for debugging
# codegen without the dlopen/launch round-trip.

"""
    code_mlir(f, argtypes; kernel_name=nameof(f), lane_width=16, alignment=16) -> String

The MLIR module emitted from `f`'s StructuredIRCode (via `Frontend.structured`
+ `lower_to_mlir_spmd`), **before** the lowering pipeline runs. `argtypes` is a
`Tuple{T1, T2, …}` type or a tuple of runtime values/types.
"""
function code_mlir(@nospecialize(f), argtypes::Type;
                   kernel_name::String=string(nameof(f)),
                   lane_width::Int=16, alignment::Int=16)
    sci, rettype = Frontend.structured(f, argtypes)
    mod, _, mlir_ctx, _ = lower_to_mlir_spmd(sci, argtypes;
                                             kernel_name, lane_width, alignment)
    @with_context mlir_ctx begin
        return sprint(show, mod)
    end
end

_argtypes_tuple(args::Tuple) =
    all(a -> a isa Type, args) ? Tuple{args...} : Tuple{map(Core.Typeof, args)...}

code_mlir(@nospecialize(f), args::Tuple; kwargs...) =
    code_mlir(f, _argtypes_tuple(args); kwargs...)

"""
    code_mlir_lowered(f, argtypes; kwargs...) -> String

The MLIR module **after** the standard CPU lowering pipeline — LLVM-dialect
MLIR, ready for `mlir-translate --mlir-to-llvmir`.
"""
function code_mlir_lowered(@nospecialize(f), argtypes::Type;
                           kernel_name::String=string(nameof(f)),
                           passes=DEFAULT_PASSES,
                           lane_width::Int=16, alignment::Int=16)
    mlir = code_mlir(f, argtypes; kernel_name, lane_width, alignment)
    return lower_mlir_text(mlir; passes)
end

code_mlir_lowered(@nospecialize(f), args::Tuple; kwargs...) =
    code_mlir_lowered(f, _argtypes_tuple(args); kwargs...)

"""
    code_llvm(f, argtypes; kwargs...) -> String

The LLVM IR (textual `.ll`) emitted for `f` — the lowering pipeline run through
`mlir-translate --mlir-to-llvmir`. `Base.code_llvm` in spirit, for the SPMD path.
"""
function code_llvm(@nospecialize(f), argtypes::Type;
                   kernel_name::String=string(nameof(f)), passes=DEFAULT_PASSES,
                   lane_width::Int=16, alignment::Int=16)
    lowered = code_mlir_lowered(f, argtypes; kernel_name, passes, lane_width, alignment)
    return translate_to_llvmir(lowered)
end

code_llvm(@nospecialize(f), args::Tuple; kwargs...) =
    code_llvm(f, _argtypes_tuple(args); kwargs...)
