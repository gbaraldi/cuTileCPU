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

# Guards inspect operands bound by `~name`: a *literal* operand binds to its
# value (so `c isa Number` holds), while an SSA operand binds to an `SSAValue`
# (failing the test) — i.e. these fire only on literal constants, without the
# (unvendored) constant analysis. `iszero`/`isone` are type-generic, so they
# match the typed zero/one Julia emits next to a same-typed operand.
_lit_is0(m, _) = (c = get(m.bindings, :c, nothing); c isa Number && iszero(c))
_lit_is1(m, _) = (c = get(m.bindings, :c, nothing); c isa Number && isone(c))

# `x * 2^n → x << n`. Power-of-two strength reduction. The guard injects the
# (operand-typed) shift count `:n` so the declarative RHS can reference it.
function _pow2_mul(m, _)
    c = get(m.bindings, :c, nothing)
    (c isa Integer && c > 0 && ispow2(c)) || return false
    m.bindings[:n] = oftype(c, trailing_zeros(c))
    return true
end

"""
Generic, backend-agnostic canonicalization rules over raw Julia integer ops.

Most algebraic *identities* (`x+0`, `x*1`, …) are already folded by Julia's
optimizer before structurization, so they're defensive here — they catch only
cases that survive (or that earlier SCI passes expose). Power-of-two strength
reduction DOES fire on real IR (Julia keeps `mul_int(x, 8)`); ptxas would also
strength-reduce, so the payoff is a cleaner MLIR, not new SASS.
"""
const CANONICALIZE_RULES = RewriteRule[
    @rewrite(Base.add_int(~x, ~c) => ~x, _lit_is0),   # x + 0 → x
    @rewrite(Base.add_int(~c, ~x) => ~x, _lit_is0),   # 0 + x → x
    @rewrite(Base.sub_int(~x, ~c) => ~x, _lit_is0),   # x - 0 → x
    @rewrite(Base.mul_int(~x, ~c) => ~x, _lit_is1),   # x * 1 → x
    @rewrite(Base.mul_int(~c, ~x) => ~x, _lit_is1),   # 1 * x → x
    @rewrite(Base.or_int(~x, ~c)  => ~x, _lit_is0),   # x | 0 → x
    @rewrite(Base.xor_int(~x, ~c) => ~x, _lit_is0),   # x ⊻ 0 → x
    @rewrite(Base.shl_int(~x, ~c)  => ~x, _lit_is0),  # x << 0 → x
    @rewrite(Base.lshr_int(~x, ~c) => ~x, _lit_is0),  # x >>> 0 → x
    @rewrite(Base.ashr_int(~x, ~c) => ~x, _lit_is0),  # x >> 0 → x
    @rewrite(Base.mul_int(~x, ~c) => Base.shl_int(~x, ~n), _pow2_mul),  # x*2ⁿ → x<<n
]

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
