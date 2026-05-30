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

        # Transcendentals (Base.sin etc. → math.sin → __nv_sinf via libdevice) and
        # rounding / fma intrinsics (floor_llvm→math.floor, muladd_float→math.fma).
        for f in (sin, cos, exp, log, tanh, cbrt, floor)
            @test A(f.(p)) ≈ f.(A(p))
        end
        @test A(p .^ 3.0f0) ≈ A(p) .^ 3.0f0                       # ^ → math.powf
        @test A(muladd.(p, p, p)) ≈ muladd.(A(p), A(p), A(p))     # → math.fma
        @test A(copysign.(p, .-p)) ≈ copysign.(A(p), .-A(p))      # → math.copysign
        # hypot: an elided `sqrt` domain-error throw makes one `scf.if` branch
        # return early; it must yield poison matching the result arity.
        @test A(hypot.(p, p)) ≈ hypot.(A(p), A(p))

        # device↔device copyto! through the backend.
        c = mk(zeros(Float32, n)); copyto!(c, a); CUDA.synchronize()
        @test A(c) == A(a)

        # Array manipulation whose copy kernels positionally `getfield` a range /
        # wrapper arg (`getfield(arg, 1)`), not by name.
        seq = mk(collect(Float32, 1:n))
        @test A(seq[10:20]) == collect(Float32, 10:20)            # getindex range
        @test A(vcat(a, b)) == vcat(A(a), A(b))                   # vcat
        let m = mk(rand(Float32, 16, 4)); @test A(hcat(m, m)) == hcat(A(m), A(m)); end
        # repeat — its copy kernel uses a generic break-loop (LoopOp).
        @test A(repeat(seq, 2)) == repeat(A(seq), 2)
        let m = mk(rand(Float32, 4, 3)); @test A(repeat(m, 2, 1)) == repeat(A(m), 2, 1); end

        # Base reductions → GPUArrays.mapreducedim! → AcceleratedKernels' reduce.
        # `count`/`any`/`all` exercise the Bool→Int width coercion + the explicit
        # result-type neutral; max/min the new maxnumf/minnumf.
        @test sum(a) ≈ sum(A(a))
        @test maximum(a) == maximum(A(a))
        @test minimum(a) == minimum(A(a))
        @test sum(abs2, a) ≈ sum(abs2, A(a))
        @test prod(p) ≈ prod(A(p))
        @test count(>(0.5f0), a) == count(>(0.5f0), A(a))
        @test any(>(0.5f0), a) == any(>(0.5f0), A(a))
        @test all(>(0.0f0), p)
        @test mapreduce(abs2, +, a) ≈ mapreduce(abs2, +, A(a))

        # Dimensional reductions → GPUArrays.mapreducedim! (dims) → AK's
        # `mapreduce_nd_by_block` kernel. That kernel has two sequential loops
        # where the second references a `Union{Int32,Int64}` accumulator carried
        # out of the first via a `PiNode`; exercises the structurizer's PiNode
        # exit-value threading + the numeric-union undef coercion.
        m2 = mk(rand(Float32, 32, 17)); M2 = A(m2)
        @test A(sum(m2; dims=1)) ≈ sum(M2; dims=1)
        @test A(sum(m2; dims=2)) ≈ sum(M2; dims=2)
        @test A(sum(abs2, m2; dims=1)) ≈ sum(abs2, M2; dims=1)
        @test A(maximum(m2; dims=2)) == maximum(M2; dims=2)
        @test A(minimum(m2; dims=1)) == minimum(M2; dims=1)
        m3 = mk(rand(Float32, 8, 4, 3)); M3 = A(m3)
        for d in 1:3
            @test A(sum(m3; dims=d)) ≈ sum(M3; dims=d)
        end

        # `extrema` reduces over a `Tuple{T,T}` (min,max) accumulator — the init
        # is a struct-typed kernel ARG and the per-step result a `Core.tuple`,
        # both reconstructed into `vector<2×T>`; the carried accumulator is a
        # struct-typed loop block arg (getfield→extractelement).
        ev = mk(rand(Float32, 3000)); EV = A(ev)
        @test extrema(ev) == extrema(EV)
        @test A(extrema(m2; dims=1)) == extrema(M2; dims=1)
        @test A(extrema(m2; dims=2)) == extrema(M2; dims=2)
        ei = mk(rand(Int32(1):Int32(99), 1500))
        @test extrema(ei) == extrema(A(ei))

        # findmax/argmax/findfirst reduce with a HETEROGENEOUS `(value,index)`
        # accumulator — `Tuple{Float32,Int64}` (or `Tuple{Bool,Int64}`). That can't
        # be a `vector<N×T>`; it lowers via the MLIR `llvm` dialect (`!llvm.struct`
        # register values + `!llvm.ptr`/getelementptr/load/store for the AoS array,
        # since `memref` can't hold an aggregate). `fpiseq` backs the isequal tie-break.
        fx = mk(rand(Float32, 2000)); FX = A(fx)
        @test findmax(fx) == findmax(FX)
        @test findmin(fx) == findmin(FX)
        @test argmax(fx) == argmax(FX)
        @test argmin(fx) == argmin(FX)
        @test findfirst(>(0.99f0), fx) == findfirst(>(0.99f0), FX)
        fi = mk(rand(Int32(1):Int32(999), 1500))
        @test argmax(fi) == argmax(A(fi))            # heterogeneous Tuple{Int32,Int64}
        fd = mk(rand(Float64, 777))
        @test findmin(fd) == findmin(A(fd))          # Tuple{Float64,Int64}
    end
end

# Homogeneous bits-structs (all fields one scalar type) lower as `vector<N×T>`,
# matching array-of-structs layout: load→destructure, getfield→extractelement,
# `:new`→from_elements, store→vector. Covers Complex + a custom struct.
struct P2; x::Float32; y::Float32; end
Base.:+(a::P2, b::P2) = P2(a.x + b.x, a.y + b.y)
Base.zero(::Type{P2}) = P2(0.0f0, 0.0f0)

@testset "struct/Complex element types on MLIRCUDABackend" begin
    if !CUDA.functional()
        @test true
    else
        n = 512; A = Array
        a = mk(rand(ComplexF32, n)); b = mk(rand(ComplexF32, n))
        @test A(a .+ b) ≈ A(a) .+ A(b)
        @test A(a .* b) ≈ A(a) .* A(b)           # complex mul: re/im cross terms
        @test A(abs.(a)) ≈ abs.(A(a))
        @test A(real.(a)) ≈ real.(A(a))
        @test A(conj.(a)) ≈ conj.(A(a))
        @test sum(a) ≈ sum(A(a))                 # reduction over a struct element
        @test A(mk(rand(ComplexF64, n)) .+ mk(rand(ComplexF64, n))) isa Vector{ComplexF64}

        # custom homogeneous struct (P2 = 2×Float32) — not `<: Number`.
        ps = mk([P2(Float32(i), Float32(2i)) for i in 1:n])
        qs = mk([P2(1.0f0, 1.0f0) for _ in 1:n])
        r = A(ps .+ qs)
        @test r[3] == P2(4.0f0, 7.0f0)
        d = mk([P2(0.0f0, 0.0f0) for _ in 1:n])
        map!(p -> P2(2p.x, 2p.y), d, ps); CUDA.synchronize()
        @test A(d)[5] == P2(10.0f0, 20.0f0)
    end
end
