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
# Numeric-union scf.if result: `flag ? Int32 : Int64` → Union{Int32,Int64},
# promoted to a common type; the i64 result is stored back into the Int32 array.
@kernel function _g_unionsel!(out, @Const(a), flag::Bool)
    i = @index(Global, Linear)
    v = flag ? a[i] : Int64(7)
    @inbounds out[i] = v % Int32
end

# Runtime dimension extent: `size(a, d)` with a runtime `d` reads
# `getfield(a.size::Tuple, d)` with a non-const index → a select-chain over the
# per-dim `memref.dim`s.
@kernel function _g_dimsz!(out, @Const(a), d)
    i = @index(Global, Linear)
    @inbounds out[i] = size(a, d)
end

# An infinite loop with `break` (a structurized LoopOp that doesn't promote to
# for/while) → scf.while carrying a `done` sentinel.
@kernel function _g_breakloop!(out, @Const(ns))
    i = @index(Global, Linear)
    n = @inbounds ns[i]; s = 0; k = 1
    while true
        s += k; k += 1
        k > n && break
    end
    @inbounds out[i] = s
end

# A runtime tuple index `t[d]` (non-const `d`) → select-chain over components.
@kernel function _g_tupidx!(out, @Const(a), @Const(ds))
    i = @index(Global, Linear)
    x = @inbounds a[i]; t = (x, 2x, 3x)
    @inbounds out[i] = t[@inbounds ds[i]]
end

# `@noinline` keeps this a real `:invoke` (Julia's inliner can't fold it), so the
# walker emits an OUTLINED `func.call` to a `func.func` lowered from its IR and
# MLIR `-inline` splices it back. Exercises the outlined-call worklist.
@noinline _g_poly(x::Float32) = x * x + 2f0 * x + 1f0
@kernel function _g_outline!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = _g_poly(a[i])
end

# Explicit device throw: a negative element raises a `DomainError` (typed `Union{}`
# → `emit_exception!` signals CUDA's per-context exception flag, which the host's
# `check_exceptions()` turns into a `KernelException`). A non-negative input runs
# clean — no false exception. `@inbounds` so only the explicit throw is in play.
@kernel function _g_throw!(out, @Const(a))
    i = @index(Global, Linear)
    x = @inbounds a[i]
    if x < 0f0
        throw(DomainError(x, "negative"))
    end
    @inbounds out[i] = x + 1f0
end

# A value-returning @noinline helper whose `return` is inside a conditional branch
# (here one arm throws) is unsupported — the live `return` can't be a func.return
# inside an scf.if region. It must error CLEANLY at compile, NOT emit a duplicate
# `@__mlirkernels_exc` global (the exception-global dedup must be shared across the
# kernel + outlined-func lowering contexts) nor silently drop the return value via
# a poison func.return. Regression for both.
@noinline _g_thrhelper(x::Float32) = x < 0f0 ? throw(DomainError(x, "neg")) : x * x
@kernel function _g_outthrow!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = _g_thrhelper(a[i])
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
        @test occursin("gpu.func",        _ir(code_gpu, kr, cr, ar, br; ndrange=256, level=:mlir))
        @test occursin("llvm.",           _ir(code_gpu, kr, cr, ar, br; ndrange=256, level=:lowered))
        @test occursin("ptx_kernel",      _ir(code_gpu, kr, cr, ar, br; ndrange=256, level=:llvm))
        @test occursin(".visible .entry", _ir(code_gpu, kr, cr, ar, br; ndrange=256, level=:ptx))

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

        # Numeric union scf.if result (Union{Int32,Int64}) promoted to a common
        # type, then stored back into the Int32 array (value coerced at the store).
        ua = MLIRArray(CUDA.CuArray(collect(Int32, 1:64))); uo = MLIRArray(CUDA.zeros(Int32, 64))
        _g_unionsel!(backend, 16)(uo, ua, true; ndrange=64); CUDA.synchronize()
        @test Array(uo) == collect(Int32, 1:64)                # union branch: a[i]
        uo2 = MLIRArray(CUDA.zeros(Int32, 64))
        _g_unionsel!(backend, 16)(uo2, ua, false; ndrange=64); CUDA.synchronize()
        @test all(Array(uo2) .== Int32(7))                     # union branch: Int64(7)%Int32

        # `size(a, d)` with a RUNTIME `d` — getfield(a.size, d) at a non-const
        # index → select-chain over memref.dims.
        dm = MLIRArray(CUDA.CuArray(rand(Float32, 8, 5))); dz = MLIRArray(CUDA.zeros(Int64, 4))
        _g_dimsz!(backend, 4)(dz, dm, 1; ndrange=4); CUDA.synchronize()
        @test all(Array(dz) .== 8)
        _g_dimsz!(backend, 4)(dz, dm, 2; ndrange=4); CUDA.synchronize()
        @test all(Array(dz) .== 5)

        # LoopOp: `while true … break` → scf.while + done sentinel.
        bn = MLIRArray(CUDA.CuArray(Int64[3, 5, 10, 1])); bo = MLIRArray(CUDA.zeros(Int64, 4))
        _g_breakloop!(backend, 4)(bo, bn; ndrange=4); CUDA.synchronize()
        @test Array(bo) == [sum(1:n) for n in (3, 5, 10, 1)]

        # Runtime tuple index → select-chain.
        ta = MLIRArray(CUDA.CuArray(Int64[5, 5, 5])); td = MLIRArray(CUDA.CuArray(Int64[1, 2, 3]))
        to = MLIRArray(CUDA.zeros(Int64, 3))
        _g_tupidx!(backend, 4)(to, ta, td; ndrange=3); CUDA.synchronize()
        @test Array(to) == [5, 10, 15]

        # Outlined call: a `@noinline` callee → func.call to an emitted func.func,
        # spliced back by MLIR `-inline`.
        ox = rand(Float32, 64); oa = MLIRArray(CUDA.CuArray(ox)); oo = MLIRArray(CUDA.zeros(Float32, 64))
        _g_outline!(backend, 64)(oo, oa; ndrange=64); CUDA.synchronize()
        @test Array(oo) ≈ ox .^ 2 .+ 2 .* ox .+ 1

        # Device exceptions: an explicit `throw` reaches the host as a
        # `KernelException` (via CUDA's per-context exception flag), while valid
        # input never raises a false exception.
        let N = 256
            good = MLIRArray(CUDA.CuArray(rand(Float32, N) .+ 1f0))   # all > 0
            tgo = MLIRArray(CUDA.zeros(Float32, N))
            _g_throw!(backend, 64)(tgo, good; ndrange=N); CUDA.synchronize()
            @test Array(tgo) ≈ Array(good) .+ 1f0                    # no false throw
            badv = rand(Float32, N) .+ 1f0; badv[123] = -5f0          # one negative
            bad = MLIRArray(CUDA.CuArray(badv)); tbo = MLIRArray(CUDA.zeros(Float32, N))
            @test_throws CUDA.CUDACore.KernelException begin
                _g_throw!(backend, 64)(tbo, bad; ndrange=N); CUDA.synchronize()
            end
        end

        # A value-returning @noinline helper that throws → clean COMPILE error
        # (value `return` inside a conditional branch), not a duplicate-global
        # crash or a silent miscompile. Lowering to :mlir is enough to trigger it.
        let oa = MLIRArray(CUDA.zeros(Float32, 64))
            @test_throws "conditional branch" code_gpu(devnull, _g_outthrow!(backend, 64),
                MLIRArray(CUDA.zeros(Float32, 64)), oa; ndrange=64, level=:mlir)
        end
    end
end

@testset "SCI optimization (DCE/CSE/LICM) affects KA codegen" begin
    if !CUDA.functional()
        @info "CUDA not functional — skipping SCI-optimization codegen test"
        @test true
    else
        N = 256
        a = MLIRArray(CUDA.rand(Float32, N)); b = MLIRArray(CUDA.rand(Float32, N))
        c = MLIRArray(CUDA.zeros(Float32, N))
        bk = get_backend(a)
        k = _g_vadd!(bk, 256)
        # `optimize_sci!` (on by default) runs in the KA compile path; the
        # `optimize` toggle lets us compare against the un-optimized lowering.
        nlines(s) = count('\n', s)
        nops(s)   = count(r"= [a-z_]+\.[a-z_]+", s)   # MLIR op-result lines
        raw_sci = _ir(code_gpu, k, c, a, b; ndrange=N, level=:sci,  optimize=false)
        opt_sci = _ir(code_gpu, k, c, a, b; ndrange=N, level=:sci,  optimize=true)
        raw_ir  = _ir(code_gpu, k, c, a, b; ndrange=N, level=:mlir, optimize=false)
        opt_ir  = _ir(code_gpu, k, c, a, b; ndrange=N, level=:mlir, optimize=true)
        @test nlines(opt_sci) < nlines(raw_sci)   # passes transform the structured IR
        @test nops(opt_ir)   <  nops(raw_ir)      # ...and that reaches the emitted MLIR
        # The optimized kernel (the default path) still computes the right result.
        k(c, a, b; ndrange=N); CUDA.synchronize()
        @test Array(c) == Array(a) .+ Array(b)
    end
end
