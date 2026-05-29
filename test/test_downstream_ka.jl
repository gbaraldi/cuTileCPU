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
    end
end
