# Downstream coverage: a real AcceleratedKernels.jl primitive running on the
# MLIRCUDABackend via the MLIRArray wrapper (get_backend auto-dispatch) — no
# explicit backend, exactly how a user would call it. Exercises the
# closure-kernel-arg flattening (AK's map! closure captures dst/src/user-fn) and
# the lazy-range arg (`eachindex(src)` is a OneTo, flattened to its `.stop`).
using CUDA, LLVM, KernelAbstractions, Atomix, AcceleratedKernels
const AK = AcceleratedKernels
const MLIRArray = Base.get_extension(MLIRKernels, :MLIRCUDAExt).MLIRArray

# foreachindex's loop-body closure must be defined inside a function so it
# captures `dst`/`src` as typed fields (a top-level closure capturing globals is
# type-unstable — AK rejects that on GPU). The captured MLIRArrays flatten +
# `_host_argtype`-remap to host Array, so the `f(indices[i])` call inlines.
function _ak_foreach_double!(dst, src)
    AK.foreachindex(src) do i
        @inbounds dst[i] = 2f0 * src[i]
    end
end

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
        fdst = MLIRArray(CUDA.zeros(Float32, n))
        _ak_foreach_double!(fdst, rsrc); CUDA.synchronize()
        @test Array(fdst) ≈ 2f0 .* (1:n)                  # AK.foreachindex end-to-end
    end
end
