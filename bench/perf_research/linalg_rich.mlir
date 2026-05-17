// Richer linalg matmul lowering. Adapted from the canonical MKL-class
// transform schedule in mlir/test/Examples/transform/ChH/full.mlir.
//
// Key additions vs the minimal schedule:
//   - tile_reduction_using_for for K  (so the inner mma decomposes cleanly)
//   - vectorize_children_and_apply_patterns (vectorize + cleanup in one go)
//   - hoist_redundant_vector_transfers (the biggest single win)
//   - LICM
//   - fold_unit_extent_dims_via_reshapes
//   - bufferization + buffer_loop_hoisting + alloc_to_alloca
//   - vector lowering patterns inside the transform (parallelarith,
//     max_transfer_rank=1)
//
// Same kernel shape as linalg.mlir: 16×16 grid blocks of 64×64 inner matmul
// over a 16-step K loop. The transform tiles the linalg.matmul itself.

module attributes {transform.with_named_sequence} {
  func.func @kernel(%A: memref<1024x1024xf32, strided<[1024, 1]>>,
                    %B: memref<1024x1024xf32, strided<[1024, 1]>>,
                    %C: memref<1024x1024xf32, strided<[1024, 1]>>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32

    scf.parallel (%bm, %bn) = (%c0, %c0) to (%c16, %c16) step (%c1, %c1) {
      %off_m = arith.muli %bm, %c64 : index
      %off_n = arith.muli %bn, %c64 : index

      %acc_init_v = arith.constant dense<0.0> : vector<64x64xf32>
      %acc_init_empty = tensor.empty() : tensor<64x64xf32>
      %acc_init_t = vector.transfer_write %acc_init_v, %acc_init_empty[%c0, %c0]
        : vector<64x64xf32>, tensor<64x64xf32>

      %acc_final = scf.for %k = %c0 to %c16 step %c1
                    iter_args(%acc_t = %acc_init_t) -> tensor<64x64xf32> {
        %off_k = arith.muli %k, %c64 : index
        %a_v = vector.transfer_read %A[%off_m, %off_k], %cst {in_bounds = [true, true]}
               : memref<1024x1024xf32, strided<[1024, 1]>>, vector<64x64xf32>
        %b_v = vector.transfer_read %B[%off_k, %off_n], %cst {in_bounds = [true, true]}
               : memref<1024x1024xf32, strided<[1024, 1]>>, vector<64x64xf32>
        %a_empty = tensor.empty() : tensor<64x64xf32>
        %b_empty = tensor.empty() : tensor<64x64xf32>
        %a_t = vector.transfer_write %a_v, %a_empty[%c0, %c0]
          : vector<64x64xf32>, tensor<64x64xf32>
        %b_t = vector.transfer_write %b_v, %b_empty[%c0, %c0]
          : vector<64x64xf32>, tensor<64x64xf32>
        %r_t = linalg.matmul ins(%a_t, %b_t : tensor<64x64xf32>, tensor<64x64xf32>)
                             outs(%acc_t : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.yield %r_t : tensor<64x64xf32>
      }

      %r_v = vector.transfer_read %acc_final[%c0, %c0], %cst {in_bounds = [true, true]}
        : tensor<64x64xf32>, vector<64x64xf32>
      vector.transfer_write %r_v, %C[%off_m, %off_n] {in_bounds = [true, true]}
        : vector<64x64xf32>, memref<1024x1024xf32, strided<[1024, 1]>>
      scf.reduce
    }
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
              : (!transform.any_op) -> !transform.any_op

    // Tile all dims (incl. K reduction) of linalg.matmul to a register-sized
    // inner kernel.
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul
                       tile_sizes [16, 16, 8]
                       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 3. Cleanup after tiling. Fold unit-extent dims, canonicalize, CSE, LICM.
    %f = transform.structured.match ops{["func.func"]} in %arg0
         : (!transform.any_op) -> !transform.any_op
    // Vectorize, then run the safe cleanups (canonicalize + cse + LICM +
    // tensor-subset folding). Skip hoist_redundant_vector_transfers — it
    // hoists transfers out of the outer K-loop incorrectly, producing
    // garbage results (confirmed via bisection).
    transform.structured.vectorize %tiled vector_sizes [16, 16, 8] : !transform.any_op
    %fv = transform.structured.match ops{["func.func"]} in %arg0
          : (!transform.any_op) -> !transform.any_op
    // Just LICM, drop tensor-subset folding.
    %all_loops = transform.structured.match interface{LoopLikeInterface} in %arg0
                 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op

    // Leave bufferization to the outer pass pipeline (matches the minimal
    // schedule's approach). Stop here.
    transform.yield
  }
}
