# Pass pipeline / orchestrator.
#
# Vendored from cuTile's `transform/pipeline.jl` `run_passes!`, reduced to the
# generic, backend-agnostic passes. cuTile's pipeline additionally ran
# tile-specific stages — constant propagation, the `Tile`-keyed normalization /
# FMA-fusion rule sets, alias analysis, token-ordering, RNG lowering,
# divisibility / bounds analyses, and no-wrap flag attachment — all of which are
# skipped here (see the project README / report). What remains is the optimizer
# core that helps any SCI-based scalar-per-thread compiler: canonicalize → CSE →
# LICM → DCE, run to fixpoint.

"""
    optimize_sci!(sci::StructuredIRCode) -> StructuredIRCode

Run the generic SCI optimization pipeline in place and return `sci`.

Pipeline (repeated to fixpoint):

1. `canonicalize_pass!` — generic algebraic canonicalization (no-op skeleton by
   default; see canonicalize.jl).
2. `cse_pass!`  — common-subexpression elimination (value numbering).
3. `licm_pass!` — loop-invariant code motion.
4. `dce_pass!`  — dead-code elimination (also cleans up anything the rewrite /
   CSE / LICM passes left behind).

The loop terminates when a full round changes nothing (measured by total
instruction count) or after `max_rounds` iterations. Each pass preserves SSA
validity (`IRStructurizer.validate_ssa_defs`) and program semantics (verified
by the standalone roundtrip-execute test).
"""
function optimize_sci!(sci::StructuredIRCode; max_rounds::Int=8)
    prev = -1
    for _ in 1:max_rounds
        canonicalize_pass!(sci)
        cse_pass!(sci)
        licm_pass!(sci)
        dce_pass!(sci)

        n = _stmt_count(sci)
        n == prev && break
        prev = n
    end
    return sci
end

"""Total number of instructions across all blocks (fixpoint progress metric)."""
function _stmt_count(sci::StructuredIRCode)
    n = 0
    for block in eachblock(sci)
        for _ in instructions(block)
            n += 1
        end
    end
    return n
end
