# test_spmd.jl — the SPMD (ISPC-style scalar→vector) path. Plain-Julia kernels.
# Exercises spmd_function / lower_to_mlir_spmd.

# Lane kernel: plain Julia (no Tile/ct.*); the trailing `i::Int` is the lane index.
function vadd_spmd(a::Vector{Float32}, b::Vector{Float32},
                   c::Vector{Float32}, i::Int)
    @inbounds c[i] = a[i] + b[i]
    return
end

@testset "SPMD: vadd" begin
    # ISPC-style SPMD mode: the kernel is plain Julia (no Tile/ct.* types,
    # no `ct.bid()`). The trailing `i::Int` arg is the lane index and the
    # walker lifts the body to `LANE_WIDTH`-wide vectors. Each grid block
    # processes 16 consecutive `i`-values in parallel.
    n = 1024
    lane_width = 16
    # Plain Vector{Float32} — SPMD mode doesn't impose an ArraySpec
    # alignment requirement.
    a = rand(Float32, n)
    b = rand(Float32, n)
    c = zeros(Float32, n)

    k = MLIRKernels.spmd_function(vadd_spmd,
        (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
        lane_width)
    # The `0` in the launch is a placeholder — the lane index is
    # synthesised inside the kernel, not read from this arg.
    k(a, b, c, 0; blocks = cld(n, lane_width))
    @test c ≈ a .+ b

    # Reflection: the emitted MLIR should contain the vector arith op and
    # one of the vector memory ops (transfer_read / gather) at the lane
    # width.
    mlir = _ir(MLIRKernels.code_mlir, vadd_spmd,
        (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
        lane_width)
    @test occursin("vector<16xf32>", mlir)
    @test occursin("arith.addf", mlir)
    @test occursin("vector.transfer_read", mlir) ||
          occursin("vector.gather", mlir)
    @test occursin("vector.transfer_write", mlir) ||
          occursin("vector.scatter", mlir)
    @test occursin("scf.parallel", mlir)
    # No Tile types / ArraySpec strided layout in SPMD memrefs.
    @test !occursin("strided<", mlir)
end

@testset "SPMD: vadd with alignment=128" begin
    # Same kernel as the basic SPMD vadd test, but with the explicit
    # `alignment=128` kwarg. The walker emits memref.assume_alignment
    # and a strided<[1]> layout, which lets the vectorizer assume unit stride.
    # Closes the DRAM-scale perf gap between
    # SPMD and tile (see bench/bench_spmd.jl). Caller is responsible
    # for supplying aligned buffers (aligned_array).
    n = 1024
    lane_width = 16
    a = MLIRKernels.aligned_array(Float32, n; alignment=128)
    b = MLIRKernels.aligned_array(Float32, n; alignment=128)
    c = MLIRKernels.aligned_array(Float32, n; alignment=128)
    copyto!(a, rand(Float32, n))
    copyto!(b, rand(Float32, n))

    k = MLIRKernels.spmd_function(vadd_spmd,
        (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
        lane_width, alignment=128)
    k(a, b, c, 0; blocks = cld(n, lane_width))
    @test c ≈ a .+ b

    # MLIR should now contain the alignment hint AND a strided<[1]>
    # layout on the memref args.
    mlir = _ir(MLIRKernels.code_mlir, vadd_spmd,
        (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
        lane_width, alignment=128)
    @test occursin("memref.assume_alignment", mlir)
    @test occursin("strided<[1]>", mlir)

    # And the launcher must reject a misaligned buffer.
    bad = Vector{Float32}(undef, n)   # 16-byte aligned, not 128
    if UInt(pointer(bad)) % 128 != 0
        @test_throws ErrorException k(bad, b, c, 0;
                                      blocks = cld(n, lane_width))
    end
end

