"""
    MLIRKernels

An MLIR-based, multi-target kernel compiler for Julia. It infers a plain-Julia
kernel through a standalone `Frontend` (its own `AbstractInterpreter` +
`IRStructurizer`), lowers the resulting StructuredIRCode to
high-level MLIR (`scf`/`arith`/`memref`/`vector`/`math`/`func`/`gpu`), and
targets two backends:

  - **CPU** — via `mlir-opt`/`mlir-translate` (`MLIR_jll`) → `clang` → a shared
    object. Surfaced as `spmd_function` (ISPC-style scalar→vector lanes) and as
    the KernelAbstractions `MLIRBackend` (`KernelAbstractionsExt`).
  - **GPU SIMT** — via the `gpu` dialect → NVVM → PTX (LLVM.jl) → `cudacall`.
    Surfaced as the KernelAbstractions `MLIRCUDABackend` (`MLIRCUDAExt`).

# Quick start (KernelAbstractions)

```julia
using KernelAbstractions, CUDA, MLIRKernels
const MLIRArray = Base.get_extension(MLIRKernels, :MLIRCUDAExt).MLIRArray

@kernel function vadd!(c, @Const(a), @Const(b))
    i = @index(Global, Linear)
    @inbounds c[i] = a[i] + b[i]
end

n = 1024
# Inputs are MLIRArrays, so the backend is inferred from the data via get_backend.
a = MLIRArray(CUDA.rand(Float32, n)); b = MLIRArray(CUDA.rand(Float32, n)); c = MLIRArray(CUDA.zeros(Float32, n))
vadd!(get_backend(a), 256)(c, a, b; ndrange=n); CUDA.synchronize()
@assert Array(c) ≈ Array(a) .+ Array(b)
```

# Reflection

```julia
# CPU SPMD path:
println(code_mlir(spmd_kernel, (Vector{Float32}, Vector{Float32}, Int)))
# GPU SIMT path — IR at any level (:sci, :mlir, :lowered, :llvm, :ptx):
println(code_gpu(vadd!(get_backend(a), 256), c, a, b; ndrange=n, level=:ptx))
```
"""
module MLIRKernels

using BFloat16s: BFloat16

# Comparison/signedness enums the scalar (SPMD/KA/GPU) lowering uses. Member
# values are the standard signless encodings the predicate-code lookups expect.
using EnumX
@enumx Signedness Unsigned=0 Signed=1
@enumx ComparisonPredicate Equal=0 NotEqual=1 LessThan=2 LessThanOrEqual=3 GreaterThan=4 GreaterThanOrEqual=5
@enumx ComparisonOrdering Unordered=0 Ordered=1

using MLIR
using MLIR.IR
const IR = MLIR.IR
const Dialects = MLIR.Dialects

# `@with_context` / `@with_module` / `@with_block` push an IR handle onto
# MLIR.jl's per-task TLS stack while executing a block. They wrap the primitive
# `IR.activate(x)` / `IR.deactivate(x)` in a try/finally; the three names are
# kept separate purely for call-site readability.
for macro_name in (:with_context, :with_module, :with_block)
    @eval macro $macro_name(x, body)
        quote
            $IR.activate($(esc(x)))
            try
                $(esc(body))
            finally
                $IR.deactivate($(esc(x)))
            end
        end
    end
end

using IRStructurizer: Block, BlockArgument, YieldOp, ContinueOp, BreakOp,
                      ConditionOp, IfOp, ForOp, WhileOp, LoopOp,
                      StructuredIRCode, Undef
import IRStructurizer
using Core: SSAValue, Argument, ReturnNode
using Core.Compiler: widenconst

using Libdl
import LLVM_full_jll
using LLVMOpenMP_jll: libomp_path

# libMLIR (from MLIR_jll) registers all upstream conversion passes via
# `mlirRegisterAllPasses` and exposes `mlirTranslateModuleToLLVMIR`. We call
# `mlirRegisterAllPasses` lazily (once per session) and
# `mlirRegisterAllLLVMTranslations` per context.
const _passes_registered = Ref(false)
function _ensure_passes_registered()
    _passes_registered[] && return
    MLIR.API.mlirRegisterAllPasses()
    _passes_registered[] = true
    return
end

"Create a fresh MLIR Context with all upstream dialects/passes registered and loaded."
function fresh_context()
    _ensure_passes_registered()
    # `IR.Context()` defaults to an empty `DialectRegistry`. We populate the
    # registry with every upstream dialect FIRST so the context exposes them
    # to `load_all_available_dialects` below — without this, walker
    # operations like `math.exp` fail with "operation was not registered".
    registry = IR.DialectRegistry()
    MLIR.API.mlirRegisterAllDialects(registry)
    ctx = IR.Context(registry)
    MLIR.API.mlirRegisterAllLLVMTranslations(ctx)
    @with_context ctx IR.load_all_available_dialects()
    return ctx
end

include("allocator.jl")
include("frontend.jl")
include("lower.jl")
include("compile.jl")
include("launch.jl")
include("reflect.jl")

"""
    code_gpu(kernel_or_f, args...; level=:ptx, ndrange, workgroupsize, sm, feat) -> String

Reflection for the KA → GPU SIMT path. Returns the kernel's intermediate
representation at one stage of the codegen pipeline:

  - `:sci`     — the StructuredIRCode (post-inference, post-overlay Julia IR)
  - `:mlir`    — the high-level `gpu`-dialect MLIR module (pre-pipeline)
  - `:lowered` — the LLVM/NVVM-dialect MLIR (after gpu→nvvm + memref/scf/arith
                 lowering, before serialisation to a `gpu.binary`)
  - `:llvm`    — the LLVM IR (textual `.ll`) carried in the `gpu.binary`
  - `:ptx`     — the final PTX assembly (what the driver JITs to SASS)

Two call forms (the method lives in `MLIRCUDAExt`; CUDA + LLVM + KA must be
loaded):

    code_gpu(kernel(MLIRCUDABackend(), wg), args...; ndrange, level=:ptx)
    code_gpu(gpu_kernel_body, Tuple{CtxT, ArgTs...}; level=:ptx)
"""
function code_gpu end

export aligned_array, spmd_function, ka_function,
       code_mlir, code_mlir_lowered, code_llvm, code_gpu

end # module
