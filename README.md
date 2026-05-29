# MLIRKernels.jl

An MLIR-based kernel compiler for Julia. It infers a plain-Julia kernel through
a standalone frontend (its own `AbstractInterpreter` +
[IRStructurizer](https://github.com/maleadt/IRStructurizer.jl), no external
kernel DSL), lowers the result to high-level MLIR
(`scf`/`arith`/`memref`/`vector`/`math`/`func`/`gpu`) **in-process via
[MLIR.jl](https://github.com/JuliaLLVM/MLIR.jl)**, and emits code for two
targets:

- **GPU (NVPTX)** — the `gpu` dialect → NVVM → PTX (LLVM.jl) → `cudacall`.
  Scalar-per-thread SIMT; one GPU thread per work-item.
- **CPU** — high-level MLIR → LLVM IR → `clang -O2 -shared` → `dlopen`, with the
  grid dispatched over OpenMP threads. ISPC-style: the workgroup is lifted to
  `vector<W>` lanes.

The primary frontend is [**KernelAbstractions.jl**](https://github.com/JuliaGPU/KernelAbstractions.jl):
an unmodified `@kernel` runs on either target by choosing a backend. A lower-level
`spmd_function` surface (plain scalar Julia, trailing lane index) is also
available for the CPU path.

## Quick start (GPU)

```julia
using KernelAbstractions, CUDA, MLIRKernels
const MLIRArray = Base.get_extension(MLIRKernels, :MLIRCUDAExt).MLIRArray

@kernel function vadd!(c, @Const(a), @Const(b))
    i = @index(Global, Linear)
    @inbounds c[i] = a[i] + b[i]
end

n = 1 << 20
# Wrap device arrays in MLIRArray; the backend is then inferred from the data,
# so no backend type is named at the call site.
a = MLIRArray(CUDA.rand(Float32, n))
b = MLIRArray(CUDA.rand(Float32, n))
c = MLIRArray(CUDA.zeros(Float32, n))
vadd!(get_backend(a), 256)(c, a, b; ndrange=n)     # backend from data, workgroupsize 256
CUDA.synchronize()
@assert Array(c) ≈ Array(a) .+ Array(b)
```

`MLIRArray` is the backend's array type — a thin wrapper around a `CuArray` whose
`get_backend` returns `MLIRCUDABackend`. Because the backend is derived from the
data (not hardcoded), backend-agnostic KA code dispatches here automatically:
AcceleratedKernels' `map!`/`reduce` and similar run unmodified on `MLIRArray`
inputs. `KernelAbstractions.allocate`/`zeros`/`ones` on the backend return
`MLIRArray`s too.

The GPU backend supports N-D `@index(Global/Local/Group, Linear|NTuple|Cartesian)`,
N-D array indexing, `@localmem`/`@private`/`@synchronize`, `@Const`, `@groupsize`,
`@atomic` (via Atomix), `@simd`/`@unroll`, scalar reduction loops, and tail-block
masking — `ndrange` need not be a multiple of the workgroupsize (the grid is
padded and out-of-range threads are masked).

### CPU (SPMD)

For kernels you'd rather write as plain scalar Julia, `spmd_function` lifts a
function whose trailing arg is a lane index to `lane_width`-wide vector MLIR:

```julia
function vadd_spmd(a::Vector{Float32}, b::Vector{Float32}, c::Vector{Float32}, i::Int)
    @inbounds c[i] = a[i] + b[i]
    return
end

k = spmd_function(vadd_spmd,
        (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
        lane_width=16, alignment=128)
k(a, b, c, 0; blocks = cld(n, 16))             # the lane-index arg is ignored at launch
```

Contiguous `a[i]` lowers to `vector.transfer_read/write`; indirect `a[idx[i]]`
falls back to `vector.gather/scatter`. KernelAbstractions `@kernel`s also run on
CPU through the `MLIRBackend` (in `KernelAbstractionsExt`).

## Reflection

Inspect the emitted IR at any level without launching:

```julia
# The code_* reflectors PRINT IR to stdout (or a given io), CUDA.jl-style.
# Capture as a String with: sprint(io -> code_gpu(io, …))

# GPU SIMT path — :sci, :mlir, :lowered, :llvm_unopt, :llvm, :ptx
code_gpu(vadd!(get_backend(a), 256), c, a, b; ndrange=n, level=:ptx)

# CPU SPMD path
code_mlir(vadd_spmd, (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int))
code_llvm(vadd_spmd, (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int))
```

`examples/reflection_and_perf.jl` prints all five GPU levels for vadd and
benchmarks the MLIR backend against CUDA.jl's native KA backend.

To dump kernels as they compile (without editing code — handy when a kernel is
launched deep inside a library like AcceleratedKernels), set `MLIRKERNELS_DUMP`
to a comma-separated subset of `sci,mlir,lowered,llvm,ptx` (or `all`); each
requested level is printed to stderr per kernel. `MLIRKERNELS_DUMP_FILTER=<substr>`
restricts it to kernels whose name contains `<substr>`. E.g.
`MLIRKERNELS_DUMP=sci MLIRKERNELS_DUMP_FILTER=vadd julia …`.

## Architecture

```
@kernel  ──KernelAbstractionsExt / MLIRCUDAExt──┐
spmd_function ──────────────────────────────────┤
                                                 ▼
                       Frontend.structured  (own AbstractInterpreter, default
                       opt params so @noinline markers survive; overlay method
                       table for the KA intrinsics) + IRStructurizer
                                                 │
                                                 ▼
                       lower_to_mlir_{spmd,ka,gpu}  (the walker, src/lower.jl)
                       high-level MLIR: scf / arith / memref / vector / math /
                       func / gpu — never the `llvm` dialect directly
                          │                                   │
                CPU       ▼                          GPU      ▼
   in-process PassManager → LLVM IR          nvvm-attach-target → gpu-kernel-
   → clang -O2 -shared → dlopen → ccall       outlining → convert-gpu-to-nvvm →
   (grid over OpenMP threads)                 … → gpu-module-to-binary → LLVM
                                              bitcode → LLVM.jl NVPTX → PTX →
                                              CuModule + cudacall
```

Lowering goes via **high-level MLIR dialects only**, so the CPU and GPU paths
share the same emitted IR and diverge only in the lowering pipeline. Every value
on the GPU SIMT path is scalar-per-thread, which sidesteps the uniform/varying
harmonization the CPU `vector<W>` path needs.

### GPU pass pipeline (in-process, via MLIR.jl + LLVM.jl)

```
nvvm-attach-target{chip=sm_90 features=+ptx80}
gpu-kernel-outlining
gpu.module(convert-gpu-to-nvvm)
convert-scf-to-cf, convert-cf-to-llvm, convert-arith-to-llvm
expand-strided-metadata, finalize-memref-to-llvm
convert-nvvm-to-llvm, reconcile-unrealized-casts
gpu-module-to-binary{format=llvm}        → LLVM bitcode → LLVM.jl NVPTX → PTX
```

### CPU pass pipeline (in-process)

```
convert-math-to-llvm        ← before convert-math-to-libm (rsqrt has no libm fn)
convert-math-to-libm        ← tanh etc.
func.func(lower-vector-multi-reduction)   ← MLIR 19+
convert-vector-to-scf, convert-vector-to-llvm   ← after libm scalarization
convert-scf-to-openmp, convert-openmp-to-llvm
convert-scf-to-cf, lower-affine, expand-strided-metadata
finalize-memref-to-llvm, convert-{arith,func,cf,ub}-to-llvm
reconcile-unrealized-casts
```

## Performance

H100 (SXM), `min` over 100 launches. The same `@kernel` runs on both the MLIR
backend and CUDA.jl's native KA backend:

| Kernel | MLIR backend | CUDA.jl KA | CUBLAS |
|---|---|---|---|
| vadd, n = 2²⁴ (GB/s) | 2785 | 2828 | — |
| matmul naive, n = 1024 (GFLOP/s) | 4430 | 3238 | — |
| matmul tiled (16×16 `@localmem`), n = 1024 (GFLOP/s) | 6847 | 6959 | 25921 |
| matmul tiled, n = 2048 (GFLOP/s) | 7308 | 7372 | 40442 |

vadd is bandwidth-bound and at parity. The tiled (one-element-per-thread,
shared-memory) matmul beats the naive kernel ~1.5× and matches CUDA.jl's native
KA running the identical algorithm; register blocking is the next lever toward
CUBLAS-class throughput. (`examples/tiled_matmul.jl` reproduces this.)

The CPU SPMD vadd path produces the AVX-512 inner loop `vmovups → vaddps →
vmovups`; at DRAM scale it is competitive with hand-threaded Julia.

## Public API

```julia
# Allocator
aligned_array(T, dims...; alignment=64)              → Array{T,N}

# CPU compile + launch (SPMD / KA)
spmd_function(f, argtypes::Type; lane_width=16, alignment=16, serial=false) → CPUKernel
(k::CPUKernel)(args...; blocks)                       → nothing

# Reflection
# PRINT IR to io (default stdout) via each object's own printer; return nothing
# (CUDA.jl-style). Capture with `sprint(io -> code_*(io, …))`.
code_mlir([io], f, argtypes; lane_width=16, alignment=16)   # CPU, pre-pipeline
code_mlir_lowered([io], f, argtypes; …)                     # CPU, post-pipeline
code_llvm([io], f, argtypes; …)                             # CPU LLVM IR
code_gpu([io], kernel, args…; ndrange, level=:ptx)          # GPU (sci/mlir/lowered/llvm_unopt/llvm/ptx)

# GPU launch is the KernelAbstractions surface; the backend comes from the data:
#   kernel(get_backend(mlirarray), workgroupsize)(args…; ndrange)
```

## Module layout

```
MLIRKernels/
├── src/
│   ├── MLIRKernels.jl        ← module entry, context macros, code_gpu stub
│   ├── frontend.jl           ← standalone AbstractInterpreter + intrinsics + overlay table
│   ├── lower.jl              ← StructuredIRCode → MLIR walker (SPMD / KA / GPU)
│   ├── compile.jl            ← in-process CPU pass pipeline + clang
│   ├── launch.jl             ← spmd_function / ka_function, CPUKernel
│   ├── allocator.jl          ← aligned_array
│   └── reflect.jl            ← code_mlir / code_mlir_lowered / code_llvm
├── ext/
│   ├── KernelAbstractionsExt.jl  ← MLIRBackend (CPU) + KA intrinsic overlays
│   └── MLIRCUDAExt.jl            ← MLIRCUDABackend (GPU) + launch + code_gpu
├── examples/                 ← reflection_and_perf.jl, tiled_matmul.jl
└── test/                     ← ParallelTestRunner driver + test_{spmd,ka_cpu,ka_upstream,downstream_ka,gpu_simt}.jl
```

## Tests

Run with [ParallelTestRunner.jl](https://github.com/JuliaTesting/ParallelTestRunner.jl)
— each `test/*.jl` runs isolated and concurrent:

```julia
] test          # or: julia --project=test test/runtests.jl
```

`test_ka_upstream.jl` runs KernelAbstractions' own canonical testsuite kernels
(localmem, private, index, copyto!, unroll) verbatim on the GPU backend, and
`test_downstream_ka.jl` runs AcceleratedKernels' `map!`/`reduce` on `MLIRArray`
inputs — both via the `get_backend` auto-dispatch path, with no backend named.
The GPU/KA testsets self-skip when CUDA isn't functional.

## Dependencies

- **MLIR.jl** + **MLIR_jll** — the MLIR builder bindings + libMLIR-C (builder
  and in-process PassManager).
- **IRStructurizer** — structurizes inferred IR into the `StructuredIRCode` the
  walker consumes.
- **LLVM** + **LLVM_full_jll** — NVPTX backend (GPU PTX emission) and `clang`
  (CPU `.so`).
- **LLVMOpenMP_jll** — libomp, linked into the JIT'd CPU `.so`.
- **BFloat16s**, **EnumX** — small leaf deps (the `BFloat16` element type and the
  comparison/signedness enums the walker uses).
- Weak deps: **KernelAbstractions** + **Atomix** (KA frontend), **CUDA** + **LLVM**
  (GPU backend) — load them to activate the extensions.
