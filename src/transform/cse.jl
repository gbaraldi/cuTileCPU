# Common Subexpression Elimination
#
# Lightweight value-numbering on `StructuredIRCode`. Vendored from cuTile's
# `transform/cse.jl`. Mirrors LLVM's `EarlyCSE.cpp` and MLIR's
# `Transforms/CSE.cpp`: a recursive walk over the structured-control-flow tree
# maintains a per-scope hash table mapping `(func, type, operands...)` to a
# canonical `SSAValue`. When an instruction's signature matches one already
# defined in the enclosing scope, all uses are redirected to the canonical SSA
# and the redundant instruction is erased.
#
# Dominance is implicit in the SCI shape: an instruction in block `B` dominates
# `B`'s remaining instructions and every instruction in `B`'s nested control-
# flow regions (recursively). The pass copies the parent's table on entering a
# sub-block — children see parent definitions, but additions inside one branch
# don't leak to siblings (e.g. `then` vs `else` of an `IfOp`).
#
# Purity is decided by two checks (see effects.jl):
#   1. The Julia-inferred `IR_FLAG_EFFECT_FREE` bit on the instruction (the
#      authoritative side-effect signal — stores/atomics/markers have it off).
#   2. `classify_memory_op == MEM_NONE` — a load (`memoryrefget`/`arrayref`/…)
#      carries `IR_FLAG_EFFECT_FREE` (it writes nothing) yet its value depends
#      on mutable memory state, so CSE must still skip it.
#
# Single-pass: we don't iterate to fixpoint. Once a definition is
# canonicalised, every later use of an equivalent expression in program order
# resolves through it (via `replace_uses!`), so a single forward walk suffices.

"""
    cse_pass!(sci::StructuredIRCode)

Run common-subexpression elimination over `sci`. Replaces redundant pure-op
instructions with their canonical predecessor and erases the redundancies.
"""
function cse_pass!(sci::StructuredIRCode)
    cse_block!(sci.entry, Dict{Tuple, SSAValue}())
    return sci
end

# Recursive walk. `parent_table` is the value-numbering table visible from the
# enclosing scope; this block extends it locally so additions don't leak to
# sibling branches.
function cse_block!(block::Block, parent_table::Dict{Tuple, SSAValue})
    table = copy(parent_table)
    snapshot = collect(instructions(block))
    for inst in snapshot
        s = inst[:stmt]
        if s isa ControlFlowOp
            for sub in blocks(s)
                cse_block!(sub, table)
            end
            continue
        end
        cse_one!(block, inst, table)
    end
end

# Try to dedup a single instruction. On a hit, redirect all uses to the cached
# canonical SSA and delete this instruction. On a miss, add the signature.
#
# The signature includes the SCI return-type annotation: two ops with the same
# operands but different result types are not equivalent. Mirrors LLVM
# `EarlyCSE`'s `isIdenticalToWhenDefined`.
function cse_one!(block::Block, inst::Instruction, table::Dict{Tuple, SSAValue})
    s = inst[:stmt]
    s isa Expr || return
    call = resolve_call(block, inst)
    call === nothing && return
    func, ops = call
    is_pure_for_cse(inst, func) || return
    # `ops` is a @view into stmt.args; materialise the operands into the key.
    sig = (func, inst[:type], collect(ops)...)
    canonical = get(table, sig, nothing)
    if canonical === nothing
        table[sig] = SSAValue(inst)
    else
        # The redundant SSA is defined in `block`, so by SSA dominance its uses
        # are confined to `block` and its nested CF regions; `replace_uses!`
        # walks exactly that subtree.
        replace_uses!(block, SSAValue(inst), canonical)
        delete!(block, inst)
    end
    return
end

#=============================================================================
 Purity classification
=============================================================================#

"""
    is_pure_for_cse(inst::Instruction, func) -> Bool

Decide whether `inst` is safe to CSE. Pure for CSE iff it has no observable
side effect beyond producing its result AND that result is a function of its
operands alone (so loads, which depend on memory state, are excluded).
"""
function is_pure_for_cse(inst::Instruction, @nospecialize(func))
    stmt_effect_free(inst) || return false
    classify_memory_op(func) == MEM_NONE || return false
    return true
end
