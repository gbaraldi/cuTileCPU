# Standalone tests for the vendored generic SCI optimization passes (SCIOpt).
#
# These are PURE SCI transforms — no GPU / CUDA / MLIR needed. We build SCIs
# from small Julia functions via `Base.code_ircode(f, types)[1][1] |>
# StructuredIRCode`, run `optimize_sci!`, and assert:
#   (a) the result still validates (`validate_ssa_defs`),
#   (b) the transform actually happened (DCE removed dead stmts, CSE merged
#       duplicate `add_int`s, LICM hoisted invariant code),
#   (c) semantics are preserved via the IRStructurizer roundtrip-execute
#       technique (StructuredIRCode → IRCode → OpaqueClosure), reused from
#       IRStructurizer's own `test/runtests.jl`.
#
# Run with:  julia --project=. test/test_transform.jl
# (the worktree's top-level Project resolves IRStructurizer via the dev pin).

using Test

# Load the standalone optimizer module directly (no MLIRKernels package needed).
include(joinpath(@__DIR__, "..", "src", "transform", "optimize.jl"))
using .SCIOpt: optimize_sci!, dce_pass!, cse_pass!, licm_pass!, canonicalize_pass!

using IRStructurizer: StructuredIRCode, validate_ssa_defs, instructions,
                      eachblock, resolve_call, insert_before!, Instruction
using Core: SSAValue
const CC = Core.Compiler

#=============================================================================
 Helpers
=============================================================================#

build_sci(@nospecialize(f), @nospecialize(types::Tuple)) =
    StructuredIRCode(Base.code_ircode(f, types)[1][1])

# Roundtrip-execute: StructuredIRCode → IRCode → OpaqueClosure → call.
# Mirrors IRStructurizer's test/runtests.jl `execute`.
function execute(sci::StructuredIRCode, args...)
    ir = CC.copy(CC.IRCode(sci))
    ir.argtypes[1] = Tuple{}
    @static if VERSION >= v"1.12-"
        ir.debuginfo.def = Symbol("unstructurized")
    end
    oc = Core.OpaqueClosure(ir)
    return oc(args...)
end

# Count instructions (optionally matching a resolved callee) across all blocks.
function count_calls(sci::StructuredIRCode, @nospecialize(target)=nothing)
    n = 0
    for block in eachblock(sci)
        for inst in instructions(block)
            if target === nothing
                n += 1
            else
                c = resolve_call(block, inst)
                c === nothing && continue
                c[1] === target && (n += 1)
            end
        end
    end
    return n
end

total_stmts(sci::StructuredIRCode) = count_calls(sci, nothing)

#=============================================================================
 Tests
=============================================================================#

@testset "SCIOpt vendored passes" verbose=true begin

    @testset "CSE merges duplicate add_int" begin
        # a = x+y; b = x+y; return a+b  → the two x+y collapse to one.
        f(x, y) = (x + y) + (x + y)
        sci = build_sci(f, (Int, Int))
        before = count_calls(sci, Base.add_int)
        @test before == 3                      # two duplicate adds + the outer add
        cse_pass!(sci)
        after = count_calls(sci, Base.add_int)
        @test after == 2                       # duplicate merged
        @test validate_ssa_defs(sci)
        @test execute(sci, 3, 4) == f(3, 4)    # semantics preserved
    end

    @testset "DCE removes an injected dead statement" begin
        # Build a clean SCI, then inject a provably-dead, effect-free op
        # (add_int with no uses) and confirm DCE removes exactly it.
        g(x, y) = x + y
        sci = build_sci(g, (Int, Int))
        before = total_stmts(sci)

        # Insert `add_int(_2, _3)` before the first instruction; never used.
        entry = sci.entry
        first_inst = first(instructions(entry))
        dead = insert_before!(entry, first_inst,
                              Expr(:call, Base.add_int, Core.Argument(2), Core.Argument(3)),
                              Int; flag=CC.IR_FLAG_EFFECT_FREE)
        @test total_stmts(sci) == before + 1
        @test validate_ssa_defs(sci)

        dce_pass!(sci)
        @test total_stmts(sci) == before       # dead stmt removed
        @test validate_ssa_defs(sci)
        @test execute(sci, 7, 8) == g(7, 8)
    end

    @testset "DCE removes a dead loop carry / dead value" begin
        # `t` is computed each iteration but never observed after the loop.
        function f(x::Int, n::Int)
            s = 0
            for i in 1:n
                t = x * x        # dead: never escapes
                s += x
            end
            return s
        end
        sci = build_sci(f, (Int, Int))
        optimize_sci!(sci)
        @test validate_ssa_defs(sci)
        # The dead per-iteration multiply must be gone.
        @test count_calls(sci, Base.mul_int) == 0
        @test execute(sci, 3, 5) == f(3, 5)
    end

    @testset "LICM hoists loop-invariant code" begin
        function f(x::Int, y::Int, n::Int)
            s = 0
            for i in 1:n
                s += x + y * y    # y*y is loop-invariant
            end
            return s
        end
        sci = build_sci(f, (Int, Int, Int))

        # Locate the for-loop body and confirm the invariant mul_int starts inside it.
        function muls_in_loop_bodies(sci)
            n = 0
            for block in eachblock(sci)
                for inst in instructions(block)
                    s = inst[:stmt]
                    if s isa SCIOpt.ForOp
                        for inner in instructions(s.body)
                            c = resolve_call(s.body, inner)
                            c !== nothing && c[1] === Base.mul_int && (n += 1)
                        end
                    end
                end
            end
            n
        end

        @test muls_in_loop_bodies(sci) >= 1
        optimize_sci!(sci)
        @test muls_in_loop_bodies(sci) == 0     # hoisted out of every loop body
        @test count_calls(sci, Base.mul_int) >= 1  # still present, just outside
        @test validate_ssa_defs(sci)
        @test execute(sci, 3, 4, 5) == f(3, 4, 5)
    end

    @testset "optimize_sci! preserves semantics (roundtrip)" begin
        cases = Any[
            (((x, y) -> x + y * y),                 (Int, Int),        (6, 7)),
            (((x, y) -> (x + y) + (x + y) + x),     (Int, Int),        (3, 9)),
            ((x -> x > 0 ? x + 1 : x - 1),          (Int,),            (5,)),
            ((x -> x > 0 ? x + 1 : x - 1),          (Int,),            (-5,)),
            (function (x::Int, n::Int)
                 s = 0
                 for i in 1:n
                     s += x * 2 + i
                 end
                 s
             end,                                    (Int, Int),        (4, 6)),
            (function (a::Float64, b::Float64)
                 c = a * b
                 d = a * b          # CSE target
                 c + d + a
             end,                                    (Float64, Float64), (1.5, 2.5)),
        ]
        for (f, types, args) in cases
            sci = build_sci(f, types)
            optimize_sci!(sci)
            @test validate_ssa_defs(sci)
            @test execute(sci, args...) == f(args...)
        end
    end

    @testset "CSE skips memory loads (different load values not merged)" begin
        # Two loads of the same array element straddling a store must NOT be
        # CSE'd into one (the load value can change). We just assert the pass
        # runs, validates, and preserves semantics — the store sits between.
        function f(a::Vector{Int}, i::Int)
            x = a[i]
            a[i] = x + 1
            y = a[i]
            return x + y
        end
        sci = build_sci(f, (Vector{Int}, Int))
        optimize_sci!(sci)
        @test validate_ssa_defs(sci)
        let a = [10, 20, 30]
            @test execute(sci, copy(a), 2) == f(copy(a), 2)
        end
    end

    @testset "canonicalize: power-of-2 strength reduction" begin
        # Julia keeps `mul_int(x, 8)` in the IR (it doesn't strength-reduce), so
        # this rule fires on real inferred code: x*8 → x<<3.
        sr(x::Int) = x * 8
        sci = build_sci(sr, (Int,))
        @test count_calls(sci, Base.mul_int) == 1
        @test count_calls(sci, Base.shl_int) == 0
        canonicalize_pass!(sci)
        @test count_calls(sci, Base.mul_int) == 0   # mul → shl
        @test count_calls(sci, Base.shl_int) == 1
        @test validate_ssa_defs(sci)
        sci2 = build_sci(sr, (Int,)); optimize_sci!(sci2)
        @test execute(sci2, 7) == sr(7)             # 7*8 == 7<<3 == 56
    end

end
