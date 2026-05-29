# Canonicalization (generic rewrite hook).
#
# cuTile's `transform/canonicalize.jl` is entirely tile-specific: it lowers
# Julia `Core.Intrinsics` (`add_int`, `mul_float`, `slt_int`, …) into cuTile
# `Intrinsics` (`addi`, `mulf`, `cmpi`, …) and runs `scalar_elim_pass!` to
# rewrite `to_scalar`/`from_scalar` and promote scalar `Number`s to 0-D
# `Tile`s. Likewise cuTile's algebraic / identity / comparison / power rule
# sets (defined in `transform/pipeline.jl`) are all keyed on the `Tile` type
# and the cuTile `Intrinsics` module. NONE of that applies to MLIRKernels,
# whose SCI is raw Julia IR lowered directly by `src/lower.jl`'s walker — there
# is no `Tile` type and no `Intrinsics` module.
#
# So what is vendored here is the GENERIC SKELETON only: a (currently empty)
# canonicalization rule set plus a pass that runs it through the vendored
# `rewrite_patterns!` driver. This is the extension point for backend-agnostic
# algebraic identities that operate on raw Julia ops (e.g. `add_int(x, 0) → x`,
# `mul_int(x, 1) → x`) — left empty by default because such rewrites are most
# safely expressed with a constant analysis (not vendored) to recognise
# literal-vs-SSA constants, and because MLIRKernels currently relies on Julia
# inference + the MLIR lowering for these.

"""
Generic canonicalization rules. Empty by default — see the file header. Add
`@rewrite`/`@rewriter` rules keyed on raw Julia callees (`Base.add_int`, …)
here to enable algebraic canonicalization before lowering.
"""
const CANONICALIZE_RULES = RewriteRule[]

"""
    canonicalize_pass!(sci::StructuredIRCode)

Run the generic canonicalization rule set to fixpoint via the pattern-rewrite
driver. A no-op while `CANONICALIZE_RULES` is empty.
"""
function canonicalize_pass!(sci::StructuredIRCode)
    isempty(CANONICALIZE_RULES) && return sci
    rewrite_patterns!(sci, CANONICALIZE_RULES)
    return sci
end
