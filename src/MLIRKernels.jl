"""
    MLIRKernels

CPU backend for cuTile.jl. Takes a cuTile kernel + argtypes, runs cuTile's
existing inference + structurization pipeline, then lowers the resulting
StructuredIRCode to MLIR (high-level dialects: `scf`, `arith`, `memref`,
`vector`, `math`, `func`), runs the standard CPU lowering pipeline via
`mlir-opt` and `mlir-translate` from `MLIR_jll`, JIT-compiles the result via
`clang` into a shared object, and exposes a launch entry point.

# Quick start

```julia
using cuTile, MLIRKernels
const ct = cuTile

function vadd(a, b, c, tile_size::Int)
    pid = ct.bid(1)
    ta = ct.load(a; index=pid, shape=(tile_size,))
    tb = ct.load(b; index=pid, shape=(tile_size,))
    ct.store(c; index=pid, tile=ta + tb)
    return
end

# Aligned host buffers (the kernel's TileArray ArraySpec demands alignment
# > 16 bytes, which plain `Vector{Float32}` doesn't guarantee).
n = 1024
a = MLIRKernels.aligned_array(Float32, n)
b = MLIRKernels.aligned_array(Float32, n)
c = MLIRKernels.aligned_array(Float32, n)
copyto!(a, 1:n); copyto!(b, 1:n); fill!(c, 0)

k = MLIRKernels.cpu_function(vadd, (a, b, c, ct.Constant(16)))
k(a, b, c, ct.Constant(16); blocks=(n ÷ 16,))
@assert c == a .+ b
```

# Reflection

```julia
println(MLIRKernels.code_mlir(vadd, (a, b, c, ct.Constant(16))))
println(MLIRKernels.code_llvm(vadd, (a, b, c, ct.Constant(16))))
```
"""
module MLIRKernels

using cuTile
const ct = cuTile
using cuTile: BFloat16

using MLIR
using MLIR.IR
const IR = MLIR.IR
const Dialects = MLIR.Dialects

# Reactant exposed a `@with_context` / `@with_module` / `@with_block` macro
# trio that pushed an IR handle onto a TLS stack while executing a block.
# MLIR.jl exposes the primitive `IR.activate(x)` / `IR.deactivate(x)`
# (per-task TLS stack); each of the three macros below is just a try/finally
# around activate/deactivate, kept as separate names purely for call-site
# readability.
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

# Vanilla libMLIR (from MLIR_jll) registers ALL upstream conversion passes
# via `mlirRegisterAllPasses` and exposes `mlirTranslateModuleToLLVMIR` —
# both things Reactant's libReactantExtra deliberately omitted. We call
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

export aligned_array, cpu_function, parallel_for, @parallel_for,
       spmd_function, ka_function,
       code_mlir, code_mlir_lowered, code_llvm, code_gpu

end # module
