# test_ka_upstream.jl — KernelAbstractions' OWN canonical testsuite kernels,
# copied verbatim from KA's test/ (localmem.jl, private.jl, test.jl, unroll.jl,
# copyto.jl) and driven through our MLIRCUDABackend (the MLIR→PTX SIMT path).
# This is real upstream coverage, not hand-written approximations.
#
# We do NOT call Testsuite.testsuite(...) because it fans out into backend-
# protocol methods we deliberately don't implement (get_backend(::CuArray),
# device!, partition/mkcontext) and kernels we don't lower (@index Cartesian,
# @print, non-multiple ndrange tail masking); instead we copy the directly-
# runnable kernels, exactly as KA's own backend testsuites are structured.
# Kernels are defined at top level; only execution is guarded on CUDA.functional().
using CUDA, LLVM, KernelAbstractions
using KernelAbstractions: @localmem, @synchronize, @uniform, @groupsize, @private
using KernelAbstractions.Extras: @unroll

const KAU_GPUB = Base.get_extension(MLIRKernels, :MLIRCUDAExt).MLIRCUDABackend

# KA test/test.jl — index_linear_global (multiple-of-wg case)
@kernel function _ka_index_linear_global(A)
    I = @index(Global, Linear)
    @inbounds A[I] = I
end
# KA test/localmem.jl — localmem (reverse within each workgroup)
@kernel function _ka_localmem(A)
    N  = @uniform prod(@groupsize())
    N2 = @uniform prod(@groupsize())
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Int (N,)
    @inbounds begin
        lmem[i] = i
        @synchronize
        A[I] = lmem[N2 - i + 1]
    end
end
# KA test/private.jl — private (per-thread scratch)
@kernel function _ka_private(A)
    N = @uniform prod(@groupsize())
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    priv = @private Int (1,)
    @inbounds begin
        priv[1] = N - i + 1
        @synchronize
        A[I] = priv[1]
    end
end
# KA test/unroll.jl — kernel_unroll! (static ndrange from kernel obj)
@kernel function _ka_kernel_unroll!(a)
    @unroll for i in 1:5
        @inbounds a[i] = i
    end
end

@testset "KA upstream testsuite kernels (MLIRCUDABackend)" begin
    if !CUDA.functional()
        @info "CUDA not functional — skipping KA upstream testsuite"
        @test true
    else
        b = KAU_GPUB()
        AT = CUDA.CuArray

        # (1) index: A[I] = I over a length-256 array, wg=8 (256 = 32*8)
        A = KernelAbstractions.allocate(b, Int, 16, 16)
        _ka_index_linear_global(b, 8)(A; ndrange=length(A))
        KernelAbstractions.synchronize(b)
        @test Array(A) == collect(LinearIndices(A))

        # (2) localmem: each 16-wide group reversed; ndrange 64 = 4*16
        A = AT{Int}(undef, 64)
        _ka_localmem(b, 16)(A; ndrange=size(A))
        KernelAbstractions.synchronize(b)
        Bv = Array(A)
        @test all(Bv[g*16+1:g*16+16] == collect(16:-1:1) for g in 0:3)

        # (3) private: per-thread scratch, same reversal pattern
        A = AT{Int}(undef, 64)
        _ka_private(b, 16)(A; ndrange=size(A))
        KernelAbstractions.synchronize(b)
        @test Array(A)[1:16] == collect(16:-1:1)

        # (4) copyto!: host↔device round-trip via KA.copyto!
        M = 1024
        Ad = AT(rand(Float32, M)); Bd = AT(rand(Float32, M)); h = Array{Float32}(undef, M)
        KernelAbstractions.copyto!(b, h, Bd)
        KernelAbstractions.copyto!(b, Ad, h)
        KernelAbstractions.synchronize(b)
        @test isapprox(h, Array(Ad)) && isapprox(h, Array(Bd))

        # (5) @unroll: static-ndrange kernel `kernel(b, 1, 1)(a)` (no ndrange kwarg)
        a = AT(zeros(Float32, 5))
        _ka_kernel_unroll!(b, 1, 1)(a)
        KernelAbstractions.synchronize(b)
        @test Array(a) == Float32[1, 2, 3, 4, 5]
    end
end
