# Drive the hand-written gpu-dialect vadd kernel through MLIR.jl's
# in-process PassManager + mlirTranslateModuleToLLVMIR → produce PTX.
#
# This is the pipeline-validation step before any Julia frontend work.
# Once this compiles + lands a runnable PTX kernel, we know the
# MLIR→NVVM→PTX path works end-to-end via MLIR.jl on this MLIR_jll.

using cuTileCPU
using MLIR
const IR = MLIR.IR

const HERE = @__DIR__

# ---------------------------------------------------------------------------
# 1. Read the hand-written MLIR
# ---------------------------------------------------------------------------

mlir_text = read(joinpath(HERE, "01_handwritten_vadd_gpu.mlir"), String)

# ---------------------------------------------------------------------------
# 2. Run the GPU-dialect lowering pipeline
# ---------------------------------------------------------------------------
#
# Two things mirror the CPU pipeline we already use:
#   - In-process via MLIR.jl PassManager (no shell-out to mlir-opt).
#   - `_pipeline_str` from cuTileCPU automatically nests `convert-gpu-to-nvvm`
#     under `gpu.module(...)` because it's a `gpu::GPUModuleOp`-anchored pass
#     (same `_FUNC_LEVEL_PASSES`-style nesting trick we used for
#     `lower-vector-multi-reduction` under func.func).

const GPU_PASSES = String[
    # Attach NVPTX target attrs (compute capability, ptx version, ...) to
    # the gpu.module so downstream passes know what to emit.
    "nvvm-attach-target{chip=sm_90 features=+ptx80}",
    "gpu-kernel-outlining",
    # `convert-gpu-to-nvvm` is the only pass that has to be nested under
    # gpu.module (it walks gpu.func bodies and rewrites gpu.* → nvvm.*).
    # The rest of the conversion passes walk all ops recursively when run
    # at the builtin.module level — they pick up ops inside gpu.module
    # without explicit nesting.
    "gpu.module(convert-gpu-to-nvvm)",
    "convert-scf-to-cf",
    "convert-cf-to-llvm",
    "convert-arith-to-llvm",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "convert-nvvm-to-llvm",
    "reconcile-unrealized-casts",
    # Emit final PTX text inline into gpu.binary attributes.
    # `format=isa` means PTX assembly (text); `=bin` would be cubin.
    "gpu-module-to-binary{format=isa}",
]

# Have to override cuTileCPU's `_FUNC_LEVEL_PASSES` for our gpu.module
# nesting — for the POC we just build the pipeline string by hand here
# (the nested ones are already `gpu.module(...)` wrapped in the array above).
_pipeline_str(passes) = "builtin.module(" * join(passes, ",") * ")"

ctx = cuTileCPU.fresh_context()
IR.activate(ctx)

mod = parse(IR.Module, mlir_text)
println("=== Parsed module ===")
println(mod)

println("\n=== Running gpu lowering pipeline ===")
pm = IR.PassManager()
parse(IR.OpPassManager(pm), _pipeline_str(GPU_PASSES))
status = MLIR.API.mlirPassManagerRunOnOp(pm, IR.Operation(mod))
status.value == 0 && error("GPU pipeline failed")

println("=== Lowered module (after pipeline) ===")
println(mod)

# ---------------------------------------------------------------------------
# 3. Extract the PTX text from the gpu.binary attribute
# ---------------------------------------------------------------------------
#
# After `gpu-module-to-binary`, the gpu.module has been replaced by a
# `gpu.binary @kernels [#gpu.object<...>]` op holding the PTX bytes in a
# DenseArrayAttribute. We pretty-print the module and grep out the PTX —
# crude, but enough for a pipeline validation.

mod_text = sprint(show, mod)
println("\n=== Searching for PTX in the binary attr... ===")
# Look for the gpu.binary op
m = match(r"gpu\.binary\s+@\w+\s+\[(.+?)\]"s, mod_text)
if m === nothing
    error("no gpu.binary found in lowered module — pipeline didn't reach gpu-module-to-binary")
end
println("Found gpu.binary attr (truncated to 200 chars):")
println(first(m.captures[1], 200))

# A more robust PTX extraction would use the C API to walk the
# gpu.binary's `objects` ArrayAttr and pull the DenseI8ArrayAttr bytes.
# For the POC we just confirm the binary slot is populated and stop here.

println("\n✓ Pipeline reached gpu-module-to-binary on MLIR_jll v$(MLIR.MLIR_VERSION[])")
println("  Next: extract the PTX bytes via the C API + load through CUDA.jl")
println("  (gpu.binary.objects[].object holds the PTX DenseI8ArrayAttr)")
