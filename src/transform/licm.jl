# Loop-Invariant Code Motion (LICM)
#
# Hoists loop-invariant operations out of loops. Vendored from cuTile's
# `transform/licm.jl`.
#
# Hoist gate (two parts):
#   1. The per-stmt `IR_FLAG_EFFECT_FREE` bit (Julia inference). The analogue
#      of LLVM LICM's `!I.mayHaveSideEffects()` / MLIR's `isMemoryEffectFree`:
#      stores, atomics, asserts, and `donotdelete`-pinned markers are pinned.
#   2. `classify_memory_op == MEM_NONE`. cuTile pinned loads via token operands
#      installed by `token_order_pass!`; MLIRKernels has no token threading, so
#      LICM must reject loads (`memoryrefget`/`arrayref`/…) directly — their
#      value can change across iterations if the loop mutates the memory.
#
# All operands must be `is_defined_outside` the loop. Uses IRStructurizer's
# `is_defined_outside`, `move_before!`, and `operands`. Processes innermost
# loops first and repeats until fixpoint.

"""
    licm_pass!(sci::StructuredIRCode)

Hoist loop-invariant operations out of loops.
"""
function licm_pass!(sci::StructuredIRCode)
    for (loop_inst, loop_op) in collect_loops(sci.entry)
        hoist_from_loop!(loop_inst, loop_op)
    end
    return sci
end

# Collect (instruction, loop_op) pairs in post-order (innermost first).
function collect_loops(root::Block)
    result = Tuple{Instruction, Union{ForOp, LoopOp, WhileOp}}[]
    collect_loops!(result, root)
    return result
end

function collect_loops!(result, block::Block)
    for inst in instructions(block)
        s = inst[:stmt]
        if s isa ForOp || s isa LoopOp
            collect_loops!(result, s.body)
            push!(result, (inst, s))
        elseif s isa WhileOp
            collect_loops!(result, s.before)
            collect_loops!(result, s.after)
            push!(result, (inst, s))
        elseif s isa ControlFlowOp
            for b in blocks(s)
                collect_loops!(result, b)
            end
        end
    end
end

# Whether a (resolvable) instruction is safe to hoist: effect-free, and not a
# memory load whose value depends on loop-mutated state.
function is_hoistable(body::Block, inst::Instruction)
    stmt_effect_free(inst) || return false
    call = resolve_call(body, inst)
    if call !== nothing
        func, _ = call
        classify_memory_op(func) == MEM_NONE || return false
    end
    return true
end

function hoist_from_loop!(loop_inst::Instruction, loop_op)
    changed = true
    while changed
        changed = false
        for body in blocks(loop_op)
            for inst in collect(instructions(body))
                inst[:stmt] isa ControlFlowOp && continue
                is_hoistable(body, inst) || continue
                all(v -> is_defined_outside(v, loop_op), operands(body, inst)) || continue
                move_before!(inst, loop_inst)
                changed = true
            end
        end
    end
end
