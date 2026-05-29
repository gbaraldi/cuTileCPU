# Reflection for the CPU SPMD path — the MLIR/LLVM IR emitted from a plain-Julia
# kernel via the standalone Frontend. For the GPU SIMT path see `code_gpu`
# (MLIRCUDAExt). These stop at intermediate IRs, useful for debugging codegen
# without the dlopen/launch round-trip.
#
# Like CUDA.jl's `code_llvm`/`code_ptx`, each reflector PRINTS the IR to an `io`
# (default `stdout`) via the object's own printer and returns nothing — so it
# displays as raw multi-line IR in the REPL, not an escaped string. To capture
# the text (e.g. in tests), use `sprint(io -> code_mlir(io, …))`.

_argtypes_tuple(args::Tuple) =
    all(a -> a isa Type, args) ? Tuple{args...} : Tuple{map(Core.Typeof, args)...}

"""
    code_mlir([io=stdout], f, argtypes; kernel_name=nameof(f), lane_width=16, alignment=16)

Print the MLIR module emitted from `f`'s StructuredIRCode (via `Frontend.structured`
+ `lower_to_mlir_spmd`), **before** the lowering pipeline runs, to `io` (MLIR's own
printer). `argtypes` is a `Tuple{T1,…}` type or a tuple of runtime values/types.
"""
function code_mlir(io::IO, @nospecialize(f), argtypes::Type;
                   kernel_name::String=string(nameof(f)),
                   lane_width::Int=16, alignment::Int=16)
    sci, _ = Frontend.structured(f, argtypes)
    mod, _, mlir_ctx, _ = lower_to_mlir_spmd(sci, argtypes;
                                             kernel_name, lane_width, alignment)
    @with_context mlir_ctx show(io, mod)   # MLIR's native printer, straight to io
    return nothing
end

"""
    code_mlir_lowered([io=stdout], f, argtypes; kwargs...)

Print the MLIR module **after** the standard CPU lowering pipeline — LLVM-dialect
MLIR, ready for `mlir-translate --mlir-to-llvmir`.
"""
function code_mlir_lowered(io::IO, @nospecialize(f), argtypes::Type;
                           kernel_name::String=string(nameof(f)),
                           passes=DEFAULT_PASSES,
                           lane_width::Int=16, alignment::Int=16)
    mlir = sprint(io2 -> code_mlir(io2, f, argtypes; kernel_name, lane_width, alignment))
    print(io, lower_mlir_text(mlir; passes))
    return nothing
end

"""
    code_llvm([io=stdout], f, argtypes; kwargs...)

Print the LLVM IR (textual `.ll`) emitted for `f` — the lowering pipeline run
through `mlir-translate --mlir-to-llvmir`. `Base.code_llvm` in spirit, SPMD path.
"""
function code_llvm(io::IO, @nospecialize(f), argtypes::Type;
                   kernel_name::String=string(nameof(f)), passes=DEFAULT_PASSES,
                   lane_width::Int=16, alignment::Int=16)
    lowered = sprint(io2 -> code_mlir_lowered(io2, f, argtypes;
                                              kernel_name, passes, lane_width, alignment))
    print(io, translate_to_llvmir(lowered))
    return nothing
end

# `io`-defaulting + tuple-of-values argtype convenience forms.
for fn in (:code_mlir, :code_mlir_lowered, :code_llvm)
    @eval begin
        $fn(@nospecialize(f), argtypes::Type; kwargs...) = $fn(stdout, f, argtypes; kwargs...)
        $fn(io::IO, @nospecialize(f), args::Tuple; kwargs...) =
            $fn(io, f, _argtypes_tuple(args); kwargs...)
        $fn(@nospecialize(f), args::Tuple; kwargs...) =
            $fn(stdout, f, _argtypes_tuple(args); kwargs...)
    end
end
