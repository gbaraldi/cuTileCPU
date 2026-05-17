// Contract path: matches what cuTileCPU's walker emits for matmul today.
// Outer scf.parallel over grid (16×16 = 256 blocks for 1024÷64=16), inner
// scf.for over K-tiles, vector.contract on 64×64×64 inputs.
//
// Compiled with: vector-contract-lowering=outerproduct.

module {
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
      %acc_init = arith.constant dense<0.0> : vector<64x64xf32>

      %acc_final = scf.for %k = %c0 to %c16 step %c1
                    iter_args(%acc = %acc_init) -> vector<64x64xf32> {
        %off_k = arith.muli %k, %c64 : index
        %a = vector.transfer_read %A[%off_m, %off_k], %cst {in_bounds = [true, true]}
             : memref<1024x1024xf32, strided<[1024, 1]>>, vector<64x64xf32>
        %b = vector.transfer_read %B[%off_k, %off_n], %cst {in_bounds = [true, true]}
             : memref<1024x1024xf32, strided<[1024, 1]>>, vector<64x64xf32>
        %r = vector.contract {
            indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                              affine_map<(m, n, k) -> (k, n)>,
                              affine_map<(m, n, k) -> (m, n)>],
            iterator_types = ["parallel", "parallel", "reduction"],
            kind = #vector.kind<add>
        } %a, %b, %acc : vector<64x64xf32>, vector<64x64xf32> into vector<64x64xf32>
        scf.yield %r : vector<64x64xf32>
      }

      vector.transfer_write %acc_final, %C[%off_m, %off_n] {in_bounds = [true, true]}
        : vector<64x64xf32>, memref<1024x1024xf32, strided<[1024, 1]>>
      scf.reduce
    }
    return
  }
}
