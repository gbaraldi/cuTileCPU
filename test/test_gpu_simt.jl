# GPU SIMT path (MLIRCUDABackend): KA @kernels compiled through the MLIR gpu
# dialect → PTX and run on the device. Scalar-per-thread, so N-D @index +
# A[i,j] + reduction accumulators work with no uniform/varying harmonization.
# Kernels are defined at top level; only execution is guarded on a functional
# CUDA device (skips otherwise).
using CUDA, LLVM, KernelAbstractions, Atomix
using KernelAbstractions: @localmem, @synchronize, @uniform, @groupsize, @private, get_backend
using KernelAbstractions.Extras: @unroll

const GPUB = Base.get_extension(MLIRKernels, :MLIRCUDAExt).MLIRCUDABackend
const MLIRArray = Base.get_extension(MLIRKernels, :MLIRCUDAExt).MLIRArray

@kernel function _g_vadd!(c, @Const(a), @Const(b))
    i = @index(Global, Linear)
    @inbounds c[i] = a[i] + b[i]
end
@kernel function _g_transpose!(a, @Const(b))
    i, j = @index(Global, NTuple)
    @inbounds a[i, j] = b[j, i]
end
@kernel function _g_matmul!(out, @Const(a), @Const(b))
    i, j = @index(Global, NTuple)
    tmp = zero(eltype(out))
    for k in 1:size(a, 2)
        @inbounds tmp += a[i, k] * b[k, j]
    end
    @inbounds out[i, j] = tmp
end
# cross-lane reverse within each block through shared memory
@kernel function _g_shrev!(out, @Const(inp))
    gid = @index(Global, Linear); lid = @index(Local, Linear)
    s = @localmem Float32 (256,)
    @inbounds s[lid] = inp[gid]
    @synchronize
    @inbounds out[gid] = s[256 - lid + 1]
end
# atomic-on-shared per-block reduction
@kernel function _g_blocksum!(out, @Const(inp))
    gid = @index(Global, Linear); gi = @index(Group, Linear)
    acc = @localmem Float32 (1,)
    @inbounds acc[1] = 0f0
    @synchronize
    Atomix.@atomic acc[1] += inp[gid]
    @synchronize
    @inbounds out[gi] = acc[1]
end
# The full KA histogram example (verbatim): @localmem + two-level shared→global
# @atomic + @synchronize + @groupsize + a 1:gs:N step-range loop + divergent ifs.
@kernel unsafe_indices=true function _g_histogram!(histogram_output, input)
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)
    @uniform gs = prod(@groupsize())
    tid = (gid - 1) * gs + lid
    @uniform Nh = length(histogram_output)
    shared_histogram = @localmem eltype(input) (gs)
    for min_element in 1:gs:Nh
        @inbounds shared_histogram[lid] = 0
        @synchronize()
        max_element = min_element + gs
        if max_element > Nh
            max_element = Nh + 1
        end
        bin = tid <= length(input) ? input[tid] : 0
        if bin >= min_element && bin < max_element
            bin -= min_element - 1
            Atomix.@atomic shared_histogram[bin] += 1
        end
        @synchronize()
        if ((lid + min_element - 1) <= Nh)
            Atomix.@atomic histogram_output[lid + min_element - 1] += shared_histogram[lid]
        end
    end
end
# @private: per-thread storage (default-space alloca). Scalar + array forms;
# the array kernel takes a compile-time `::Val{M}` (not a runtime param).
@kernel function _g_privrev!(A)
    @uniform Np = prod(@groupsize())
    I = @index(Global, Linear); il = @index(Local, Linear)
    pp = @private Int (1,)
    @inbounds pp[1] = Np - il + 1
    @inbounds A[I] = pp[1]
end
@kernel function _g_privarr!(out, @Const(A), ::Val{M}) where {M}
    I = @index(Global, Linear)
    pp = @private Int (M,)
    s = 0
    @inbounds for j in 1:M; pp[j] = A[I] * j; end
    @inbounds for j in 1:M; s += pp[j]; end
    @inbounds out[I] = s
end
# 2-D @localmem tile: size(tile,d) resolves to the static dims. Copy +
# cross-lane transpose through a 16x16 shared tile.
@kernel function _g_tilecopy!(o, @Const(a))
    I, J = @index(Global, NTuple); ii, jj = @index(Local, NTuple)
    t = @localmem Float32 (16, 16)
    @inbounds t[ii, jj] = a[I, J]
    @synchronize
    @inbounds o[I, J] = t[ii, jj]
end
@kernel function _g_tiletr!(o, @Const(a))
    I, J = @index(Global, NTuple); ii, jj = @index(Local, NTuple)
    t = @localmem Float32 (16, 16)
    @inbounds t[ii, jj] = a[I, J]
    @synchronize
    @inbounds o[I, J] = t[jj, ii]
end
# @simd / @unroll loops: the loopinfo hint is dropped (plain scf.for); LLVM/
# ptxas unroll. Both forms over a Val{M} bound.
@kernel function _g_simdsum!(o, @Const(a), ::Val{M}) where {M}
    Ix = @index(Global, Linear); ac = zero(eltype(o))
    @simd for k in 1:M; @inbounds ac += a[Ix] * k; end
    @inbounds o[Ix] = ac
end
@kernel function _g_unrollsum!(o, @Const(a), ::Val{M}) where {M}
    Ix = @index(Global, Linear); ac = zero(eltype(o))
    @unroll for k in 1:M
        @inbounds ac += a[Ix] * k
    end
    @inbounds o[Ix] = ac
end
# Tiled shared-memory matmul: the capstone — 2-D @localmem tiles, a k-tile loop
# with two @synchronize, and a register accumulator. The inner-product
# accumulator `out` is LOCAL to each k-tile (folded into `acc` after the inner
# loop) — one accumulator carried through BOTH loop levels trips the structurizer.
@kernel function _g_mmtiled!(C, @Const(A), @Const(B))
    I, J = @index(Global, NTuple); ii, jj = @index(Local, NTuple)
    tA = @localmem Float32 (16, 16)
    tB = @localmem Float32 (16, 16)
    acc = zero(Float32); nkt = size(A, 2) ÷ 16
    for kt in 1:nkt
        kb = (kt - 1) * 16
        @inbounds tA[ii, jj] = A[I, kb + jj]
        @inbounds tB[ii, jj] = B[kb + ii, J]
        @synchronize
        out = zero(Float32)
        @inbounds for kk in 1:16
            out += tA[ii, kk] * tB[kk, jj]
        end
        acc += out
        @synchronize
    end
    @inbounds C[I, J] = acc
end
# Closure/functor kernel arg: the kernel calls a closure `f` that captures
# device arrays (the map/reduce/broadcast pattern). The closure is flattened
# into its captured array params; the call inlines.
@kernel function _g_applyclos!(out, f)
    i = @index(Global, Linear)
    @inbounds out[i] = f(i)
end
@kernel function _g_mapclos!(n, f)
    i = @index(Global, Linear)
    if i <= n
        f(i)
    end
end
# @index(Global, Cartesian): a CartesianIndex{N}. Full-index `A[I]` + component
# `I[k]` access (the GPUArrays broadcast/copy/transpose pattern).
@kernel function _g_cartdbl!(A)
    I = @index(Global, Cartesian)
    @inbounds A[I] = A[I] * 2f0
end
@kernel function _g_carttr!(B, @Const(A))
    I = @index(Global, Cartesian)
    @inbounds B[I[2], I[1]] = A[I[1], I[2]]
end

@testset "GPU: KA @kernel on MLIRCUDABackend (SIMT)" begin
    if !CUDA.functional()
        @info "CUDA not functional in this env — skipping GPU backend test"
        @test true
    else
        # Every input is an MLIRArray, so KernelAbstractions.get_backend infers
        # MLIRCUDABackend from the data — no backend type is named at the call
        # sites (the path GPUArrays / AcceleratedKernels take).
        # 1-D vadd
        N = 4096
        a1 = MLIRArray(CUDA.rand(Float32, N)); b1 = MLIRArray(CUDA.rand(Float32, N))
        c1 = MLIRArray(CUDA.zeros(Float32, N))
        backend = get_backend(a1)
        @test backend isa GPUB                            # auto-dispatch → MLIRCUDABackend
        _g_vadd!(backend, 256)(c1, a1, b1; ndrange=N); CUDA.synchronize()
        @test Array(c1) == Array(a1) .+ Array(b1)
        # 2-D transpose (non-square) — catches descriptor dim-order
        bh = reshape(collect(Float32, 1:32), 8, 4)
        bt = MLIRArray(CUDA.CuArray(bh)); at = MLIRArray(CUDA.zeros(Float32, 4, 8))
        _g_transpose!(backend, (4, 4))(at, bt; ndrange=(4, 8)); CUDA.synchronize()
        @test Array(at) == permutedims(bh)
        # 2-D matmul (non-square) — scalar accumulator over a for-loop
        ah = rand(Float32, 8, 4); bbh = rand(Float32, 4, 6)
        am = MLIRArray(CUDA.CuArray(ah)); bm = MLIRArray(CUDA.CuArray(bbh)); om = MLIRArray(CUDA.zeros(Float32, 8, 6))
        _g_matmul!(backend, (4, 2))(om, am, bm; ndrange=(8, 6)); CUDA.synchronize()
        @test maximum(abs.(Array(om) .- ah * bbh)) < 1f-3

        # @localmem cross-lane reverse + atomic-on-shared block reduction
        Nl = 1024; Wl = 256; NBl = Nl ÷ Wl
        inl = MLIRArray(CUDA.CuArray(rand(Float32, Nl))); ihl = Array(inl)
        orl = MLIRArray(CUDA.zeros(Float32, Nl))
        _g_shrev!(backend, Wl)(orl, inl; ndrange=Nl); CUDA.synchronize()
        refrev = similar(ihl)
        for b in 0:(NBl-1), k in 1:Wl; refrev[b*Wl+k] = ihl[b*Wl + (Wl-k+1)]; end
        @test Array(orl) == refrev                       # cross-lane shared + barrier
        osum = MLIRArray(CUDA.zeros(Float32, NBl))
        _g_blocksum!(backend, Wl)(osum, inl; ndrange=Nl); CUDA.synchronize()
        refsum = [sum(ihl[(b*Wl+1):((b+1)*Wl)]) for b in 0:(NBl-1)]
        @test isapprox(Array(osum), refsum; rtol=1f-4)   # atomic-on-shared

        # full KA histogram
        Lh = 4096; NBINS = 256
        hin = rand(1:NBINS, Lh)
        dhin = MLIRArray(CUDA.CuArray(hin)); dhout = MLIRArray(CUDA.zeros(Int, NBINS))
        _g_histogram!(backend, (256,))(dhout, dhin; ndrange=Lh); CUDA.synchronize()
        hist_ref = zeros(Int, NBINS); for v in hin; hist_ref[v] += 1; end
        @test Array(dhout) == hist_ref                   # full KA histogram

        # @private scalar + array (with Val arg)
        Np = 64; Wp = 16
        ap = MLIRArray(CUDA.zeros(Int, Np))
        _g_privrev!(backend, Wp)(ap; ndrange=Np); CUDA.synchronize()
        @test Array(ap) == repeat(collect(Wp:-1:1), Np ÷ Wp)   # per-thread scalar
        Mp = 4; inpp = MLIRArray(CUDA.CuArray(collect(1:Np))); op = MLIRArray(CUDA.zeros(Int, Np))
        _g_privarr!(backend, Wp)(op, inpp, Val(Mp); ndrange=Np); CUDA.synchronize()
        @test Array(op) == [i * sum(1:Mp) for i in 1:Np]       # per-thread array + Val arg

        # 2-D @localmem tile copy + cross-lane transpose
        Mt = 32
        int = MLIRArray(CUDA.CuArray(reshape(collect(Float32, 1:(Mt*Mt)), Mt, Mt))); iht = Array(int)
        otc = MLIRArray(CUDA.zeros(Float32, Mt, Mt))
        _g_tilecopy!(backend, (16,16))(otc, int; ndrange=(Mt,Mt)); CUDA.synchronize()
        @test Array(otc) == iht                                # 2-D tile copy
        ott = MLIRArray(CUDA.zeros(Float32, Mt, Mt))
        _g_tiletr!(backend, (16,16))(ott, int; ndrange=(Mt,Mt)); CUDA.synchronize()
        reft = copy(iht)
        for bi in 0:1, bj in 0:1, ai in 1:16, aj in 1:16
            reft[bi*16+ai, bj*16+aj] = iht[bi*16+aj, bj*16+ai]
        end
        @test Array(ott) == reft                               # 2-D tile cross-lane transpose

        # @simd / @unroll reduction loops over a Val{M} bound
        Ns = 64; Ws = 16; Ms = 4
        as = MLIRArray(CUDA.CuArray(collect(1:Ns))); refs = [Array(as)[k]*sum(1:Ms) for k in 1:Ns]
        os1 = MLIRArray(CUDA.zeros(Int, Ns))
        _g_simdsum!(backend, Ws)(os1, as, Val(Ms); ndrange=Ns); CUDA.synchronize()
        @test Array(os1) == refs                               # @simd
        os2 = MLIRArray(CUDA.zeros(Int, Ns))
        _g_unrollsum!(backend, Ws)(os2, as, Val(Ms); ndrange=Ns); CUDA.synchronize()
        @test Array(os2) == refs                               # @unroll

        # code_gpu reflection: every codegen level emits its expected IR.
        ar = MLIRArray(CUDA.rand(Float32, 256)); br = MLIRArray(CUDA.rand(Float32, 256))
        cr = MLIRArray(CUDA.zeros(Float32, 256))
        kr = _g_vadd!(backend, 256)
        @test occursin("gpu.func",        code_gpu(kr, cr, ar, br; ndrange=256, level=:mlir))
        @test occursin("llvm.",           code_gpu(kr, cr, ar, br; ndrange=256, level=:lowered))
        @test occursin("ptx_kernel",      code_gpu(kr, cr, ar, br; ndrange=256, level=:llvm))
        @test occursin(".visible .entry", code_gpu(kr, cr, ar, br; ndrange=256, level=:ptx))

        # tiled matmul
        for nm in (256, 512)
            Am = MLIRArray(CUDA.rand(Float32, nm, nm)); Bm = MLIRArray(CUDA.rand(Float32, nm, nm))
            Cm = MLIRArray(CUDA.zeros(Float32, nm, nm))
            _g_mmtiled!(backend, (16, 16))(Cm, Am, Bm; ndrange=(nm, nm)); CUDA.synchronize()
            @test isapprox(Array(Cm), Array(Am) * Array(Bm); rtol=1f-2)  # tiled matmul
        end

        # Closure kernel args (the map/reduce pattern): a closure capturing 1 or
        # 2 device arrays, flattened into memref params; the call inlines.
        csrc = MLIRArray(CUDA.CuArray(collect(Float32, 1:1024)))
        cg = let s = csrc; i -> 2f0 * s[i]; end
        cout = MLIRArray(CUDA.zeros(Float32, 1024))
        _g_applyclos!(backend, 256)(cout, cg; ndrange=1024); CUDA.synchronize()
        @test Array(cout) ≈ 2f0 .* (1:1024)                    # 1-capture closure
        cdst = MLIRArray(CUDA.zeros(Float32, 1024)); csrc2 = MLIRArray(CUDA.CuArray(collect(Float32, 1:1024)))
        ch = let d = cdst, s = csrc2; i -> (@inbounds d[i] = 3f0 * s[i]); end
        _g_mapclos!(backend, 256)(1024, ch; ndrange=1024); CUDA.synchronize()
        @test Array(cdst) ≈ 3f0 .* (1:1024)                    # 2-capture closure + write

        # @index(Global, Cartesian): full-index A[I] (2-D + 1-D) + component I[k].
        cda = MLIRArray(CUDA.CuArray(rand(Float32, 16, 16))); cda0 = Array(cda)
        _g_cartdbl!(backend, (4, 4))(cda; ndrange=size(cda)); CUDA.synchronize()
        @test Array(cda) ≈ 2f0 .* cda0                         # 2-D Cartesian A[I]
        cv = MLIRArray(CUDA.CuArray(rand(Float32, 1024))); cv0 = Array(cv)
        _g_cartdbl!(backend, 256)(cv; ndrange=length(cv)); CUDA.synchronize()
        @test Array(cv) ≈ 2f0 .* cv0                           # 1-D Cartesian
        cta = MLIRArray(CUDA.CuArray(rand(Float32, 8, 12))); ctb = MLIRArray(CUDA.zeros(Float32, 12, 8))
        _g_carttr!(backend, (4, 4))(ctb, cta; ndrange=size(cta)); CUDA.synchronize()
        @test Array(ctb) == permutedims(Array(cta))            # Cartesian I[k] transpose

        # Tail-block masking: ndrange NOT a multiple of the workgroup. The grid
        # is padded (cld) and __validindex masks the out-of-range tail threads
        # (a tail thread that wrote would index out of bounds).
        for Nt in (1000, 257)
            ta = MLIRArray(CUDA.CuArray(rand(Float32, Nt))); tb = MLIRArray(CUDA.CuArray(rand(Float32, Nt)))
            tc = MLIRArray(CUDA.zeros(Float32, Nt))
            _g_vadd!(backend, 256)(tc, ta, tb; ndrange=Nt); CUDA.synchronize()
            @test Array(tc) == Array(ta) .+ Array(tb)          # 1-D masked tail
        end
        mta = MLIRArray(CUDA.CuArray(rand(Float32, 100, 70))); mta0 = Array(mta)
        _g_cartdbl!(backend, (16, 16))(mta; ndrange=size(mta)); CUDA.synchronize()
        @test Array(mta) ≈ 2f0 .* mta0                         # 2-D masked tail
    end
end
