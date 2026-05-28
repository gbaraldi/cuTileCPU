// Hand-written gpu-dialect vadd kernel — the smallest MLIR module that
// exercises the gpu → nvvm → llvm → ptx pipeline.
//
// Pipeline (driven from 01_drive.jl):
//   1. nvvm-attach-target{chip=sm_90}        — attach NVPTX target attrs
//   2. gpu-kernel-outlining                  — already outlined (kernel is
//                                              in gpu.module), no-op here
//   3. gpu.module(convert-gpu-to-nvvm)       — gpu.thread_id etc. -> nvvm.*
//   4. convert-nvvm-to-llvm                  — nvvm.* -> llvm dialect
//   5. convert-arith-to-llvm                 — arith.muli, addi etc.
//   6. finalize-memref-to-llvm               — memref descriptors -> llvm
//   7. reconcile-unrealized-casts
//   8. gpu-module-to-binary{format=isa}      — emit PTX text inline as a
//                                              gpu.binary
//
// Then we extract the PTX string from the gpu.binary attribute and hand
// it to CUDA.jl's `CuModule(...)` to load + launch.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vadd(%a: memref<?xf32>, %b: memref<?xf32>, %c: memref<?xf32>, %n: index) kernel {
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %i = arith.muli %bid, %bdim : index
      %gid = arith.addi %i, %tid : index
      %in_bounds = arith.cmpi ult, %gid, %n : index
      scf.if %in_bounds {
        %av = memref.load %a[%gid] : memref<?xf32>
        %bv = memref.load %b[%gid] : memref<?xf32>
        %sum = arith.addf %av, %bv : f32
        memref.store %sum, %c[%gid] : memref<?xf32>
      }
      gpu.return
    }
  }
}
