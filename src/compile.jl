# MLIR pipeline + LLVM-IR translation run **in-process** via MLIR.jl's
# `PassManager` and `mlirTranslateModuleToLLVMIR`. Only the final LLVM-IR
# → `.so` codegen step shells out to `clang`.
#
# Why in-process now? MLIR.jl is pinned to MLIR_jll v18 (the highest libLLVM
# version that coexists with Julia's bundled libLLVM_jll 18). Our MLIR module
# is therefore emitted in MLIR-18 dialect syntax. Driving `mlir-opt` from
# LLVM_full_jll v21 against that text fails on ops whose syntax changed
# (e.g. `memref.assume_alignment` gained a result type in MLIR 21). Running
# the passes against the *same* libMLIR that built the module eliminates the
# version-skew problem entirely.
#
# What the in-process path replaces:
#   - mlir-opt        → `IR.PassManager` + `parse(opm, pipeline_str)` + run
#   - mlir-translate  → `MLIR.API.mlirTranslateModuleToLLVMIR` +
#                       `LLVMPrintModuleToString`
#   - clang           → still shells out (for `-O2 -shared -fPIC` + libomp link)

# libomp from LLVMOpenMP_jll. `libomp_path` is the JLL's lazy accessor; we
# resolve to a directory + `-lomp` link line for clang. Override via
# `ENV["CUTILECPU_LIBOMP_DIR"]` if you need a custom build.
_libomp_dir() = get(ENV, "CUTILECPU_LIBOMP_DIR", dirname(libomp_path))

# clang from LLVM_full_jll. Used only for the LLVM-IR → .so step. Override
# via `CUTILECPU_CLANG` if you want a different toolchain.
_clang() = get(ENV, "CUTILECPU_CLANG",
               joinpath(LLVM_full_jll.artifact_dir, "tools", "clang"))

# Default lowering pipeline: cuTile-style `scf.parallel` + vector dialect →
# OpenMP → LLVM dialect. Joined into a single textual pipeline string for
# `IR.PassManager.parse`.
const DEFAULT_PASSES = String[
    # Math lowering, two passes in strict order:
    #   1. `convert-math-to-llvm` — handles ops with LLVM intrinsics (`exp`,
    #      `sin`, `cos`, `log`, `sqrt`, `rsqrt` — the last expands to
    #      `1.0 / llvm.intr.sqrt`). Works for scalar AND vector forms.
    #      MUST run before libm: otherwise libm emits a `rsqrtf` call to a
    #      function that doesn't exist in libm, dlopen fails at launch.
    #   2. `convert-math-to-libm` — picks up the remainder (notably
    #      `math.tanh` on MLIR 20+, which has no LLVM intrinsic registered)
    #      and routes them to libm function calls (`tanhf`). On vector
    #      math it scalarizes via `vector.extract` per lane; those
    #      extracts/inserts must be lowered by `convert-vector-to-llvm`
    #      later, so this MUST run before vector-to-llvm.
    "convert-math-to-llvm",
    "convert-math-to-libm",
    # `lower-vector-multi-reduction` decomposes `vector.multi_reduction` ops
    # into `vector.reduction` + assembly that `convert-vector-to-llvm` can
    # match. Without it, `multi_reduction` survives the whole pipeline and
    # the LLVM-IR translator fails with "missing LLVMTranslationDialectInterface".
    # The pass is anchored at `func::FuncOp` in MLIR's TableGen, so the
    # pipeline syntax must nest it under `func.func(...)`; `_wrap_pass` does
    # that automatically.
    "lower-vector-multi-reduction",
    "convert-vector-to-scf",
    # `convert-vector-to-llvm` MUST run before any pass that rewrites the
    # surrounding region's types to LLVM (`convert-openmp-to-llvm` wraps
    # kernel bodies in `omp.parallel` whose region argument types are
    # converted to `!llvm.array<...>` — after which residual vector ops
    # can't be matched). Lowering vector ops first keeps their operand
    # types in the `vector` dialect's expected form.
    "convert-vector-to-llvm",
    "convert-scf-to-openmp",
    # `scf-to-openmp` wraps the kernel body in `memref.alloca_scope`, which
    # requires its region to have at most one basic block. `scf-to-cf` would
    # multiply blocks. We sidestep this by running `convert-openmp-to-llvm`
    # first to rewrite the alloca_scope into LLVM regions that tolerate
    # branches, and only then convert scf.if/for to cf.
    "convert-openmp-to-llvm",
    "convert-scf-to-cf",
    "lower-affine",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "convert-arith-to-llvm",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    # `vector.contract` lowering can introduce `ub.poison` for undef padding;
    # the LLVM-IR translator needs this lowered first.
    "convert-ub-to-llvm",
    "reconcile-unrealized-casts",
]

# Single-threaded lowering pipeline: same shape as DEFAULT_PASSES, but
# `convert-scf-to-cf` runs directly on `scf.parallel` (gets degraded to a
# serial `scf.for` automatically) and we never touch the `omp` dialect. The
# resulting `.so` has no libomp dependency and runs the entire grid on the
# calling thread.
const SERIAL_PASSES = String[
    "convert-math-to-llvm",
    "convert-math-to-libm",
    "lower-vector-multi-reduction",
    "convert-vector-to-scf",
    "convert-vector-to-llvm",
    # NO `convert-scf-to-openmp`. scf.parallel runs serially via scf-to-cf.
    "convert-scf-to-cf",
    "lower-affine",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "convert-arith-to-llvm",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "convert-ub-to-llvm",
    "reconcile-unrealized-casts",
]

# Build the textual pipeline string MLIR's PassManager parser consumes:
# `builtin.module(pass1, pass2, ...)`. Two pieces of MLIR-version-conditional
# logic live here:
#
#   1. Passes that the maintainers anchored at `func::FuncOp` (e.g.
#      `Pass<"lower-vector-multi-reduction", "func::FuncOp">` in their
#      TableGen) must be nested under `func.func(...)` in the pipeline
#      syntax — flat-listed they error with "does not refer to a registered
#      pass." `_wrap_pass` does that automatically.
#
#   2. Some passes are only registered in newer MLIR versions (e.g.
#      `lower-vector-multi-reduction` first appears as a registered pass in
#      MLIR 19; on MLIR 18 the textual parser doesn't recognise it).
#      `_pass_available` filters those out for the current MLIR version.
const _FUNC_LEVEL_PASSES = Set([
    "lower-vector-multi-reduction",  # Pass<"...", "func::FuncOp">
])
function _pass_available(p::String)
    name = split(p, ['{', ' '])[1]
    name == "lower-vector-multi-reduction" && return MLIR.MLIR_VERSION[] ≥ v"19"
    return true
end
function _wrap_pass(p::String)
    name = split(p, ['{', ' '])[1]
    return name in _FUNC_LEVEL_PASSES ? "func.func($p)" : p
end
_pipeline_str(passes) =
    "builtin.module(" * join(_wrap_pass.(filter(_pass_available, passes)), ",") * ")"

"""
    run_pipeline!(mod::MLIR.IR.Module; passes=DEFAULT_PASSES)

Run the given pass pipeline on `mod` **in-place** using the current MLIR
context's `PassManager`. Mutates `mod`; returns it.
"""
function run_pipeline!(mod::IR.Module; passes=DEFAULT_PASSES)
    pm = IR.PassManager()
    parse(IR.OpPassManager(pm), _pipeline_str(passes))
    status = MLIR.API.mlirPassManagerRunOnOp(pm, IR.Operation(mod))
    # MlirLogicalResult is a struct holding `.value::Int32` — nonzero is
    # success, zero is failure. (The `mlirLogicalResultIs*` helper symbols
    # aren't exported from libMLIR-C, so we inspect the field directly.)
    status.value == 0 && error("cuTileCPU pass pipeline failed")
    return mod
end

"""
    translate_to_llvmir(mod::MLIR.IR.Module) -> String

Translate LLVM-dialect MLIR to textual LLVM IR via libMLIR's
`mlirTranslateModuleToLLVMIR`. Returns the `.ll` text.

`LLVMContextCreate`/`LLVMContextDispose`/`LLVMPrintModuleToString` aren't
re-exported from MLIR_jll v18's libMLIR-C — they live in Julia's bundled
libLLVM. We `ccall` them directly. (MLIR 21+ also exposes them through
libMLIR-C, but we're pinned to v18 to coexist with Julia 1.12's libLLVM 18.)
"""
# Text-in convenience wrapper: parse lowered MLIR text into a fresh context,
# then translate to textual LLVM IR. Kept for `reflect.jl`'s `code_llvm`.
function translate_to_llvmir(mlir_text::String)
    ctx = fresh_context()
    return @with_context ctx begin
        mod = parse(IR.Module, mlir_text)
        translate_to_llvmir(mod)
    end
end

function translate_to_llvmir(mod::IR.Module)
    llvm_ctx = ccall((:LLVMContextCreate, "libLLVM"), Ptr{Cvoid}, ())
    try
        llvm_mod = MLIR.API.mlirTranslateModuleToLLVMIR(IR.Operation(mod), llvm_ctx)
        llvm_mod == C_NULL && error("mlirTranslateModuleToLLVMIR returned null")
        try
            cstr = ccall((:LLVMPrintModuleToString, "libLLVM"), Cstring,
                         (Ptr{Cvoid},), llvm_mod)
            cstr == C_NULL && error("LLVMPrintModuleToString returned null")
            try
                return unsafe_string(cstr)
            finally
                ccall((:LLVMDisposeMessage, "libLLVM"), Cvoid, (Cstring,), cstr)
            end
        finally
            ccall((:LLVMDisposeModule, "libLLVM"), Cvoid, (Ptr{Cvoid},), llvm_mod)
        end
    finally
        ccall((:LLVMContextDispose, "libLLVM"), Cvoid, (Ptr{Cvoid},), llvm_ctx)
    end
end

# Textual-input wrapper, kept for `reflect.jl`. Parses the text into a fresh
# module (using `fresh_context`, which registers the conversion passes), runs
# the pipeline, and prints the result.
"""
    lower_mlir_text(mlir_text::String; passes=DEFAULT_PASSES) -> String

Run the standard CPU lowering pipeline on a textual MLIR module. Returns
LLVM-dialect MLIR (still textual).
"""
function lower_mlir_text(mlir_text::String; passes=DEFAULT_PASSES)
    ctx = fresh_context()
    @with_context ctx begin
        mod = parse(IR.Module, mlir_text)
        run_pipeline!(mod; passes)
        return sprint(show, mod)
    end
end

"""
    compile_module_to_so(mod::IR.Module, mlir_ctx::IR.Context; kernel_name,
                         passes=DEFAULT_PASSES) -> so_path

End-to-end: in-process MLIR pass pipeline + LLVM-IR translation, then
`clang -O2 -shared` for the final `.so`. Takes the `MLIR.IR.Module`
directly (not text) so we never round-trip through MLIR's textual printer,
which is the easiest way to dodge printer/parser asymmetries on MLIR 18
(e.g. `vector.multi_reduction` is printed in a form the v18 parser can't
read back).

The .so is rpath-linked against libomp from `LLVMOpenMP_jll` (or
`CUTILECPU_LIBOMP_DIR`) when the pipeline mentions `openmp`.
"""
function compile_module_to_so(mod::IR.Module, mlir_ctx::IR.Context;
                              kernel_name::String, opt_level::Int=2,
                              passes=DEFAULT_PASSES, clang::String=_clang())
    workdir = mktempdir(; prefix="cuTileCPU_$(kernel_name)_")
    llvm_path = joinpath(workdir, "$(kernel_name).ll")
    so_path   = joinpath(workdir, "$(kernel_name).so")

    llvm_text = @with_context mlir_ctx begin
        run_pipeline!(mod; passes)
        translate_to_llvmir(mod)
    end
    write(llvm_path, llvm_text)

    needs_libomp = any(p -> occursin("openmp", p), passes)
    if needs_libomp
        libomp_dir = _libomp_dir()
        run(`$clang -O$opt_level -shared -fPIC $llvm_path
             -L$libomp_dir -Wl,-rpath,$libomp_dir -lomp
             -o $so_path`)
    else
        run(`$clang -O$opt_level -shared -fPIC $llvm_path -o $so_path`)
    end
    return so_path
end

"""
    compile_to_so(mlir_text::String; kernel_name, passes=DEFAULT_PASSES) -> so_path

Text-in convenience wrapper around `compile_module_to_so`. Parses the
textual MLIR into a fresh context, then defers to the module-based path.
Note: textual MLIR may not round-trip cleanly through MLIR 18's
printer/parser for every op (see `compile_module_to_so` docstring). The
launch path uses `compile_module_to_so` directly to avoid that.
"""
function compile_to_so(mlir_text::String; kernel_name::String,
                       opt_level::Int=2, passes=DEFAULT_PASSES,
                       clang::String=_clang())
    ctx = fresh_context()
    mod = @with_context ctx parse(IR.Module, mlir_text)
    return compile_module_to_so(mod, ctx; kernel_name, opt_level, passes, clang)
end
