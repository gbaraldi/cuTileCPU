# KernelAbstractions CPU backend (MLIRBackend <: KA.GPU): KA @kernel →
# Frontend.structured (own interpreter/intrinsics) → MLIR → clang → libomp.
# No GPU needed; KA/Atomix are test deps, so kernels are defined at top level
# and the CPU path always runs.
using KernelAbstractions, Atomix
using Atomix: @atomic
const KA = KernelAbstractions
const MLIRBackend = Base.get_extension(MLIRKernels, :KernelAbstractionsExt).MLIRBackend

@kernel function _ka_vadd!(C, A, B)
    i = @index(Global, Linear)
    @inbounds C[i] = A[i] + B[i]
end
# `@atomic` here is Atomix's portable atomic (imported above), which our KA
# extension overlays onto the Frontend `atomic_index!` marker → memref.atomic_rmw.
@kernel function _ka_hist!(bins, @Const(idx))
    i = @index(Global, Linear)
    @inbounds @atomic bins[idx[i]] += 1f0
end
@kernel function _ka_amax!(out, @Const(x))
    i = @index(Global, Linear)
    @inbounds @atomic out[1] max x[i]
end
@kernel function _ka_amin!(out, @Const(x))
    i = @index(Global, Linear)
    @inbounds @atomic out[1] min x[i]
end
@kernel function _ka_ctr!(out)
    i = @index(Global, Linear)
    @inbounds @atomic out[1] += 1f0
end
@kernel function _ka_transpose!(a, @Const(b))
    i, j = @index(Global, NTuple)
    @inbounds a[i, j] = b[j, i]
end

@testset "KA: vadd via MLIRBackend (CPU, decoupled)" begin
    N = 4096; W = 16
    A = MLIRKernels.aligned_array(Float32, N; alignment=128); copyto!(A, rand(Float32, N))
    B = MLIRKernels.aligned_array(Float32, N; alignment=128); copyto!(B, rand(Float32, N))
    C = MLIRKernels.aligned_array(Float32, N; alignment=128); fill!(C, 0f0)
    _ka_vadd!(MLIRBackend(), W)(C, A, B; ndrange=N)
    @test C ≈ A .+ B
    # The @noinline global_index marker must survive inference under the
    # Frontend interpreter (default opt params) — appear as a call in the SCI,
    # not be inlined/folded away.
    ctxT = let
        ndr = KA.NDIteration.StaticSize{(N,)}
        wg  = KA.NDIteration.StaticSize{(W,)}
        grp = KA.NDIteration.StaticSize{(N ÷ W,)}
        ndro = KA.NDIteration.NDRange{1, grp, wg, Nothing, Nothing}
        KA.CompilerMetadata{ndr, KA.NDIteration.NoDynamicCheck, Nothing, Nothing, ndro}
    end
    sci, rt = MLIRKernels.Frontend.structured(gpu__ka_vadd!,
        Tuple{ctxT, Vector{Float32}, Vector{Float32}, Vector{Float32}})
    @test rt === Nothing
    @test occursin("global_index", sprint(show, sci))
end

# KA.@atomic — the *portable* atomic (= Atomix.@atomic). Covers: varying-index
# float add (per-lane scatter), uniform-slot float add + integer max/min
# (lane-reduction → single atomic per block), cross-block atomicity, and the
# float-min/max MLIR-version gate.
@testset "KA: @atomic via MLIRBackend (Atomix portable path)" begin
    W = 16

    # (a) Histogram: varying per-lane index → per-lane atomic scatter.
    N = 4096; NB = 8
    idx  = Int32[(j % NB) + 1 for j in 0:N-1]
    bins = MLIRKernels.aligned_array(Float32, NB; alignment=128); fill!(bins, 0f0)
    _ka_hist!(MLIRBackend(), W)(bins, idx; ndrange=N)
    @test all(==(Float32(N ÷ NB)), bins)

    # (b) Atomicity: every lane → one slot; no lost updates across blocks.
    M = 65536
    ones_idx = ones(Int32, M)
    acc = MLIRKernels.aligned_array(Float32, 1; alignment=128); fill!(acc, 0f0)
    _ka_hist!(MLIRBackend(), W)(acc, ones_idx; ndrange=M)
    @test acc[1] == Float32(M)

    # (c) Integer max/min into a uniform slot → vector.reduction + one atomic
    #     per block (maxs/mins lower on every supported MLIR).
    Ni = 256
    xi = Int32.(collect(1:Ni)); xi[100] = Int32(9999)
    omax = MLIRKernels.aligned_array(Int32, 1; alignment=128); omax[1] = typemin(Int32)
    omin = MLIRKernels.aligned_array(Int32, 1; alignment=128); omin[1] = typemax(Int32)
    _ka_amax!(MLIRBackend(), W)(omax, xi; ndrange=Ni)
    _ka_amin!(MLIRBackend(), W)(omin, xi; ndrange=Ni)
    @test omax[1] == 9999
    @test omin[1] == 1

    # (d) The Atomix overlay → atomic_index! marker must survive inference
    #     (DCE would otherwise delete the unused-result call).
    ctxT = let
        ndr = KA.NDIteration.StaticSize{(N,)}; wg = KA.NDIteration.StaticSize{(W,)}
        grp = KA.NDIteration.StaticSize{(N ÷ W,)}
        ndro = KA.NDIteration.NDRange{1, grp, wg, Nothing, Nothing}
        KA.CompilerMetadata{ndr, KA.NDIteration.NoDynamicCheck, Nothing, Nothing, ndro}
    end
    sci, rt = MLIRKernels.Frontend.structured(gpu__ka_hist!,
        Tuple{ctxT, Vector{Float32}, Vector{Int32}})
    @test occursin("atomic_index!", sprint(show, sci))

    # (e) Float min/max atomics need MLIR ≥ 21 (LLVM 20's memref→llvm doesn't
    #     lower maxnumf/minnumf). On older MLIR the walker raises a clear error;
    #     on MLIR ≥ 21 it just works.
    xf = Float32.(collect(1:Ni)); xf[100] = 9999f0
    of = MLIRKernels.aligned_array(Float32, 1; alignment=128); of[1] = -Inf32
    if MLIRKernels.MLIR.MLIR_VERSION[] < v"21"
        @test_throws Exception _ka_amax!(MLIRBackend(), W)(of, xf; ndrange=Ni)
    else
        _ka_amax!(MLIRBackend(), W)(of, xf; ndrange=Ni)
        @test of[1] == 9999f0
    end

    # (f) Counter idiom `@atomic out[1] += c` with a UNIFORM scalar value. Each
    #     of the W lanes runs the statement, so the slot must gain W*c per block
    #     (== ndrange total) — the value is broadcast to W lanes then reduced.
    ctr = MLIRKernels.aligned_array(Float32, 1; alignment=128); ctr[1] = 0f0
    _ka_ctr!(MLIRBackend(), W)(ctr; ndrange=256)
    @test ctr[1] == 256f0   # NOT 256/W

    # (g) The backend lowers a 1-D grid/workgroup; multi-dimensional ndrange or a
    #     launch-time workgroupsize conflicting with a static one must error.
    @test_throws Exception _ka_ctr!(MLIRBackend(), W)(ctr; ndrange=(8, 4))
    @test_throws Exception _ka_ctr!(MLIRBackend(), W)(ctr; ndrange=256, workgroupsize=2W)
end

# Multi-dimensional support: N-D `@index(Global, NTuple)` + N-D `A[i,j]`. The
# workgroup is flattened to a 1-D lane vector, per-dim coords reconstructed by
# column-major unflatten, and `A[i,j]` linearised to a gather/scatter over a
# flattened rank-1 view. 2-D transpose (KA's naive_transpose) is the gate.
@testset "KA: multi-dim @index(Global, NTuple) + A[i,j]" begin
    M = 8
    b = MLIRKernels.aligned_array(Float32, (M, M); alignment=128)
    copyto!(b, reshape(collect(Float32, 1:(M * M)), M, M))
    a = MLIRKernels.aligned_array(Float32, (M, M); alignment=128)
    fill!(a, 0f0)
    _ka_transpose!(MLIRBackend(), (4, 4))(a, b; ndrange=(M, M))
    @test a == permutedims(b)            # full N-D index + linearised A[i,j]
end
