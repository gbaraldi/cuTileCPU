# Matmul perf research

Investigation into the ~27% of OpenBLAS gap on F32 matmul. Three hand-written
MLIR variants of the same 1024×1024×1024 matmul, lowered to `.so` and timed
against OpenBLAS as the reference.

## Files

- `contract.mlir` — the kernel shape our walker emits today: `scf.parallel`
  grid over 16×16 blocks, `scf.for` over K-tiles, `vector.contract` on
  64×64×64 inputs with `outerproduct` lowering. ~37 K lines of LLVM-dialect
  MLIR, ~4 K unrolled FMA instructions.
- `linalg.mlir` — same outer kernel shape but the inner matmul is
  `linalg.matmul` on tensors, with a minimal transform schedule:
  `tile_using_for [16, 16, 8]` + `vectorize vector_sizes [16, 16, 8]`.
  Compact MLIR (~8 K lines).
- `linalg_rich.mlir` — same as `linalg.mlir` but with a richer transform
  schedule modeled after `llvm-project/mlir/test/Examples/transform/ChH/full.mlir`
  (the canonical ~80%-of-MKL example). Bisected: the MKL-class cleanups
  (`hoist_redundant_vector_transfers`, `apply_licm`) **break correctness**
  in our outer-`scf.for`-K-loop kernel shape — they hoist `vector.transfer`s
  out of the K-loop incorrectly.
- `run_bench.jl` — host harness. Compiles each `.mlir` via `mlir-opt` +
  `mlir-translate` + `clang -O2`, dlopens the `.so`, ccalls it, verifies
  correctness against `mul!` (OpenBLAS) output, then times 10 cache-flushed
  samples per variant.

## Reproduce

```bash
julia -t auto bench/perf_research/run_bench.jl
```

Needs no Julia env setup beyond what cuTileCPU already requires (LLVM_full_jll
for `mlir-opt`/`mlir-translate`/`clang`, LLVMOpenMP_jll for libomp).

## Results (M=N=K=1024, F32, 64 threads, cache-flushed)

```
contract:         8019.1 μs    267.8 GFLOPS  (19% of BLAS)
linalg minimal:   5713.9 μs    375.8 GFLOPS  (27% of BLAS)
linalg rich:      5762.8 μs    372.6 GFLOPS  (26% of BLAS)
OpenBLAS:         1514.8 μs   1417.7 GFLOPS  (reference)
```

`rich` is the stripped-back version with the unsafe cleanups disabled. With
them enabled, perf becomes irrelevant — output is numerically wrong by ~5e4.

## Bisection summary

The MKL-class transform schedule's per-kernel wins **don't apply to our
outer-scf.for-K-loop kernel shape**. Specifically:

- `transform.structured.hoist_redundant_vector_transfers` — hoists a
  `vector.transfer_read` from inside the outer K-loop out of the loop,
  destroying the per-K-iter data flow. Likely confused by the
  `vector.transfer_write → tensor → linalg.matmul → tensor → vector.transfer_read`
  round-trip the schedule uses for value↔tensor conversion.
- `transform.apply_licm` to `LoopLikeInterface` matches all loops including
  the outer K-loop, then hoists ops that depend on `%k`.
- `tile_reduction_using_for [0, 0, 1]` — also fails. Adds another K-tiling
  inside an existing K-loop and the combiner step doesn't compose correctly.

The MKL-class transforms assume **`linalg.matmul` owns the entire matmul,
not just a per-tile slice inside a hand-written K-loop**. To use them we'd
need to emit one `linalg.matmul` for the WHOLE M×N×K (with `memref.subview`
inputs over the per-block partition) and let the transform schedule handle
all tiling — a different walker emission strategy.

## How Triton-CPU handles it

Researched `triton-lang/triton-cpu`'s pipeline. They have **four** matmul
lowering strategies, picked at compile time:

| Strategy | What it emits | Perf class |
|---|---|---|
| `ConvertDotToUkernels` | `cpu.brgemm_create` + `cpu.brgemm_execute` calling into **libxsmm or oneDNN BRGEMM** | BLAS-class |
| `ConvertDotToAMX` | Intel AMX tile instructions (Sapphire Rapids+) | BLAS-class on AMX HW |
| `ConvertDotToFMA` | Hand-written row-extract + `vector.fma` + `memref.prefetch` + explicit `keepAccOnRegs` register tracking | ~50–70% of BLAS |
| `ConvertDotGeneric` | Falls through to generic `vector.contract`-equivalent | ~25–35% of BLAS (matches our current perf) |

**Key finding: Triton-CPU also punts to oneDNN/libxsmm for BLAS-class matmul
perf.** They don't try to beat hand-tuned microkernels with MLIR transform
schedules — they *call* them. Our current path is at the same perf level as
their `ConvertDotGeneric` fallback. The "MLIR transforms alone can reach
MKL" framing in the ChH tutorial is misleading for matmul specifically;
the conv example it demos works because conv doesn't have a hand-tuned
microkernel library as the benchmark.

## Realistic paths to close the gap

In rough priority order:

1. **libxsmm/oneDNN BRGEMM integration** — the only path to true BLAS-class
   matmul perf, mirroring Triton-CPU's `ConvertDotToUkernels`. Detect the
   `mma`-in-a-K-loop pattern in our walker, emit a libxsmm `brgemm` call
   instead of `vector.contract`. Substantial engineering but well-defined.

2. **A dedicated cuTile-CPU mid-level dialect** (analogous to Triton's
   `TTCIR`). Sits between the cuTile SCI and standard MLIR dialects. Gives
   us a layer to do CPU-specific transforms (dot-shape detection, mask
   opt, FP decomposition) cleanly. Without one, we mix cuTile-specific
   recognition with downstream MLIR lowering in the walker.

3. **`memref.prefetch` insertion** at strategic points (e.g. before each
   K-iter `transfer_read`). Easy add, marginal but real win.

4. **`math.*` via vector-math libs** (libmvec/SVML) instead of LLVM
   intrinsics (`--convert-math-to-vec-lib`/`--convert-math-to-libm`).
   Triton-CPU does this.

## What's NOT a path forward

- Better MLIR transform schedules. We tried the canonical MKL-class
  pipeline (LICM + hoist transfers + fold unit dims + tile reduction +
  parallelarith contract lowering + alloc-to-alloca + buffer-loop-hoist).
  Either correctness breaks (our K-loop shape isn't compatible) or perf is
  unchanged. The Triton-CPU evidence corroborates: even their generic
  fallback path lands at 25–35% of BLAS, same as ours.

- "Just bigger tiles." Tried BM=BN=BK=128 and 256. 128 compiles in ~minute,
  modest perf gain. 256 hangs clang -O2 for hours (16M FMAs unrolled).
  Unsustainable.

## What this means for cuTileCPU at the package level

Our matmul at **~27% of BLAS with no hand-tuning is the right place to sit
for the auto-codegen path** — comparable to Triton-CPU's generic. If a user
genuinely needs MKL-class matmul, the realistic options are:

- Compose their kernel from `cuTileCPU.aligned_array` + a direct `mul!`
  call (uses OpenBLAS — already there)
- Hand-write multi-level blocking in the cuTile kernel (works today; the
  user owns the cache hierarchy explicitly)
- Wait for / contribute libxsmm BRGEMM integration

For non-matmul kernels (softmax, layernorm, attention, FFT, MoE), the gap
between auto-codegen and hand-tuned is much smaller because there's no
BLAS-equivalent canonical kernel to beat — generic MLIR + vector dialect
is the right level, and our perf there is competitive.
