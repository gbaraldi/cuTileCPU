// Linalg path: same kernel shape but the inner matmul is linalg.matmul on
// tensors. A transform schedule tiles + vectorizes the linalg.matmul.
//
// Compiled with the linalg pipeline (transform-interpreter + bufferize + …).

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

  // Transform schedule: tile the linalg.matmul to a register-sized inner
  // kernel, then vectorize. tile_sizes [16, 16, 8] is roughly AVX-512-tuned
  // (16-wide N, 16 acc registers via M, K=8 inner reduction).
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
              : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [16, 16, 8]
                       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled vector_sizes [16, 16, 8] : !transform.any_op
    transform.yield
  }
}
