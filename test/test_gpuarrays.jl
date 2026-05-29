# Downstream coverage: GPUArrays' generic array operations running on the
# MLIRCUDABackend. Making `MLIRArray <: AbstractGPUArray` (plus a broadcast
# style + `similar`/`derive`) routes GPUArrays' generic `broadcast`/`map!`/
# `fill!` here — they are KA kernels launched via `get_backend`, so they compile
# through MLIRKernels. Math (`sqrt`/`abs`) additionally exercises the libdevice
# link in the PTX step.
using CUDA, LLVM, KernelAbstractions, GPUArrays
const MLIRArray = Base.get_extension(MLIRKernels, :MLIRCUDAExt).MLIRArray

mk(v) = MLIRArray(CUDA.CuArray(v))

@testset "GPUArrays generic ops on MLIRCUDABackend" begin
    if !CUDA.functional()
        @info "CUDA not functional — skipping GPUArrays test"
        @test true
    else
        @test MLIRArray <: GPUArrays.AbstractGPUArray

        n = 1024
        a = mk(rand(Float32, n)); b = mk(rand(Float32, n))
        A = Array

        # fill! — GPUArrays' fill_kernel! (single array + scalar arg).
        f = mk(zeros(Float32, 8)); fill!(f, 3.0f0); CUDA.synchronize()
        @test all(A(f) .== 3.0f0)

        # broadcast — needs the MLIRArrayStyle + `similar(::Broadcasted)`. The
        # kernel takes a `Broadcasted` whose nested arrays flatten to memrefs.
        @test A(a .+ b) ≈ A(a) .+ A(b)
        @test A(2.0f0 .* a) ≈ 2 .* A(a)
        @test A(@. a + b * a) ≈ A(a) .+ A(b) .* A(a)          # fused
        @test A(a .> 0.5f0) == (A(a) .> 0.5f0)                # comparison → Bool
        @test A(ifelse.(a .> 0.5f0, a, 0.0f0)) == ifelse.(A(a) .> 0.5f0, A(a), 0.0f0)

        # broadcast! (the `.=` in-place form) and Base.map!.
        d = mk(zeros(Float32, n)); d .= a .+ b; CUDA.synchronize()
        @test A(d) ≈ A(a) .+ A(b)
        m = mk(zeros(Float32, n)); map!(x -> x^2, m, a); CUDA.synchronize()
        @test A(m) ≈ A(a) .^ 2

        # 2-D broadcast — exercises the N-D default workgroupsize (GPUArrays
        # launches an N-D ndrange with no workgroupsize).
        a2 = mk(rand(Float32, 32, 16)); b2 = mk(rand(Float32, 32, 16))
        @test A(a2 .+ b2) ≈ A(a2) .+ A(b2)

        # Integer broadcast (no libdevice; signless int arith).
        ia = mk(rand(Int32, n)); ib = mk(rand(Int32, n))
        @test A(ia .+ ib) == A(ia) .+ A(ib)

        # Math via libdevice (`__nv_sqrtf`/`__nv_fabsf`), linked into the PTX.
        p = mk(rand(Float32, n) .+ 0.5f0)
        @test A(sqrt.(abs.(p))) ≈ sqrt.(abs.(A(p)))

        # device↔device copyto! through the backend.
        c = mk(zeros(Float32, n)); copyto!(c, a); CUDA.synchronize()
        @test A(c) == A(a)
    end
end
