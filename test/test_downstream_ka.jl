# Downstream coverage: a real AcceleratedKernels.jl primitive running on the
# MLIRCUDABackend via the MLIRArray wrapper (get_backend auto-dispatch) — no
# explicit backend, exactly how a user would call it. Exercises the
# closure-kernel-arg flattening (AK's map! closure captures dst/src/user-fn) and
# the lazy-range arg (`eachindex(src)` is a OneTo, flattened to its `.stop`).
using CUDA, LLVM, KernelAbstractions, Atomix, AcceleratedKernels
const AK = AcceleratedKernels
const MLIRArray = Base.get_extension(MLIRKernels, :MLIRCUDAExt).MLIRArray

@testset "AcceleratedKernels on MLIRCUDABackend" begin
    if !CUDA.functional()
        @info "CUDA not functional — skipping AcceleratedKernels test"
        @test true
    else
        n = 1024
        src = MLIRArray(CUDA.CuArray(collect(Float32, 1:n)))
        dst = MLIRArray(CUDA.zeros(Float32, n))
        AK.map!(x -> 2f0 * x, dst, src)        # dispatches to MLIRCUDABackend
        CUDA.synchronize()
        @test Array(dst) ≈ 2f0 .* (1:n)        # AK.map! end-to-end

        # reduce: @Const(@view src[1:end]) wrapped array + @localmem block
        # reduction + @synchronize. Exercises the recursive struct-flatten
        # (SubArray.parent/offset/stride + .indices), the Const/:new wrapper, and
        # the block-reduction helper.
        rsrc = MLIRArray(CUDA.CuArray(collect(Float32, 1:n)))
        @test AK.reduce(+, rsrc; init=0.0f0) ≈ sum(1:n)   # AK.reduce end-to-end

        # mapreduce / sum / count / any / all — reductions over a map or
        # predicate, all on the @localmem block-reduction path.
        @test AK.sum(rsrc) ≈ sum(1:n)
        @test AK.mapreduce(x -> x, +, rsrc; init=0f0) ≈ sum(1:n)
        @test AK.count(x -> x > 0f0, rsrc) == n           # predicate count (bit-shift ops)
        @test AK.any(x -> x > Float32(n - 1), rsrc)
        @test AK.all(x -> x > 0f0, rsrc)

        # accumulate! / cumsum — prefix scan (scf.while loop + bit-shift block
        # arithmetic in the decoupled-lookback kernel).
        adst = MLIRArray(CUDA.zeros(Float32, n))
        AK.accumulate!(+, adst, rsrc; init=0f0); CUDA.synchronize()
        @test Array(adst) ≈ cumsum(1:n)                   # AK.accumulate! end-to-end
        @test Array(AK.cumsum(rsrc)) ≈ cumsum(1:n)        # AK.cumsum end-to-end

        # foreachindex: grid over indices, user loop-body closure (unsafe_indices
        # kernel + indirect `f(indices[i])` that inlines via the capture remap).
        # The loop-body closure must live in a function so it captures dst/src as
        # typed fields (a top-level closure over globals is type-unstable — AK
        # rejects that on GPU); the captured MLIRArrays flatten + `_host_argtype`-
        # remap to host Array, so `f(indices[i])` inlines.
        foreach_double!(dst, src) = AK.foreachindex(i -> (@inbounds dst[i] = 2f0 * src[i]), src)
        fdst = MLIRArray(CUDA.zeros(Float32, n))
        foreach_double!(fdst, rsrc); CUDA.synchronize()
        @test Array(fdst) ≈ 2f0 .* (1:n)                  # AK.foreachindex end-to-end

        # A contiguous view of an MLIRArray materialises (GPUArrays `derive`) to
        # an MLIRArray sharing storage — copy to host via the wrapped CuArray.
        @test Array(@view rsrc[3:7]) == Float32[3, 4, 5, 6, 7]
        # A strided (non-contiguous) view stays a SubArray; its host copy must go
        # through the wrapped CuArray (the generic path scalar-indexes).
        @test Vector(@view rsrc[3:2:9]) == Float32[3, 5, 7, 9]

        # sort! — the merge-sort kernel mixes Int32/Int64 indices, exercising the
        # binop/cmp/select/while width coercion (the merge step's `addi`/`cmpi`
        # over an `Int32` index and an `Int64` if-result must agree on a type).
        s = MLIRArray(CUDA.CuArray(collect(Float32, n:-1:1)))
        AK.sort!(s); CUDA.synchronize()
        @test issorted(Array(s))                          # AK.sort! end-to-end
        si = MLIRArray(CUDA.CuArray(rand(Int32(1):Int32(1000), n)))
        sv = copy(Array(si)); AK.sort!(si); CUDA.synchronize()
        @test Array(si) == sort(sv)

        # accumulate! exclusive — `if inclusive || iblock != 0 { shift }`, the
        # short-circuit-`||`-guarded body that needs the IRStructurizer edge
        # multiplexer (otherwise the shift runs unconditionally → inclusive scan).
        ex = MLIRArray(CUDA.CuArray(ones(Int32, 16)))
        AK.accumulate!(+, ex; init=Int32(0), inclusive=false,
                       block_size=8, alg=AK.ScanPrefixes()); CUDA.synchronize()
        @test Array(ex) == collect(Int32, 0:15)           # exclusive prefix sum
    end
end
