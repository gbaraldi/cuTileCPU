# Dead Code Elimination for Structured IR
#
# General-purpose DCE using a dependency graph + BFS reachability. Vendored
# from cuTile's `transform/dce.jl`, with the cuTile token-threading machinery
# removed (MLIRKernels' IRStructurizer has no `JoinTokensNode` /
# `TokenResultNode` / `MakeTokenNode` / token types) and the effect/must-keep
# decision rebased on the generic `stmt_effect_free` + `classify_memory_op`
# (effects.jl) instead of cuTile's `intrinsic_effects`.
#
# Algorithm:
# 1. Build a dependency graph: each value → list of values it depends on
# 2. Seed live set from side-effectful operations (stores, atomics, returns)
# 3. BFS backward through dependencies to find all live values
# 4. Prune: remove dead instructions and dead loop carries / IfOp results
#
# Handles cycles naturally: dead carries that form
# body_arg → ContinueOp → body_arg are never reachable from any root, so they
# remain dead.

#=============================================================================
 CF pseudo-nodes
=============================================================================#

# Each ControlFlowOp gets a unique sentinel key in the dependency graph,
# matching cuTile's "$cf.<N>" naming.
struct CFNode
    id::UInt64
end

Base.hash(n::CFNode, h::UInt) = hash(n.id, hash(:CFNode, h))
Base.:(==)(a::CFNode, b::CFNode) = a.id == b.id

cf_node(op) = CFNode(objectid(op))

#=============================================================================
 Build dependency graph
=============================================================================#

"""
    is_trackable_value(x) -> Bool

Check if `x` is a trackable value in the dependency graph.
"""
is_trackable_value(@nospecialize(x)) = x isa SSAValue || x isa BlockArgument || x isa Argument

"""
    get_stmt_operands(s) -> Vector{Any}

Extract trackable operand values from a statement. Only `Expr`-shaped calls
carry SSA/argument operands in MLIRKernels' raw-Julia SCI; everything else
contributes nothing.
"""
function get_stmt_operands(@nospecialize(s))
    result = Any[]
    if s isa Expr
        start = s.head === :invoke ? 3 : 2
        for i in start:length(s.args)
            is_trackable_value(s.args[i]) && push!(result, s.args[i])
        end
    elseif s isa PiNode
        is_trackable_value(s.val) && push!(result, s.val)
    elseif is_trackable_value(s)
        # Alias/forwarding statement: the stmt itself IS a value.
        push!(result, s)
    end
    return result
end

"""
    must_keep(block, s) -> Bool

Check if a statement is side-effectful and must be kept as a DCE root.

Uses the per-statement `IR_FLAG_EFFECT_FREE` bit (Julia inference): stores,
atomics, and `donotdelete`-pinned markers are NOT effect-free, so they are
kept. `ReturnNode`s are always kept. Statements that can't be resolved to a
call but are also effect-free (e.g. `getfield`, `tuple`, casts) are droppable.
Mirrors cuTile's `_must_keep` and Julia's `stmt_effect_free`.
"""
function must_keep(block::Block, @nospecialize(s))
    s isa ReturnNode && return true
    # Non-Expr, non-value statements with no SSA def (GotoNode etc. don't occur
    # in structured IR) — keep conservatively.
    if !(s isa Expr) && !(s isa PiNode) && !is_trackable_value(s)
        return true
    end
    # PiNode / alias forwards are pure value carriers.
    (s isa PiNode || is_trackable_value(s)) && return false
    # Expr: keep unless it's a resolvable, effect-free, non-loading call.
    # An `:boundscheck` / `:foreigncall` / unresolved call is kept conservatively.
    call = resolve_call(block, s)
    if call !== nothing
        return false  # purity is decided at the instruction level via the flag
    end
    return true
end

"""
    inst_must_keep(block, inst) -> Bool

Instruction-level must-keep: a resolvable call is kept iff it is not
effect-free (stores/atomics/markers). Falls back to the statement-level
`must_keep` for non-call statements.
"""
function inst_must_keep(block::Block, inst::Instruction)
    s = inst[:stmt]
    s isa ReturnNode && return true
    if s isa Expr
        call = resolve_call(block, s)
        if call !== nothing
            # A call is droppable only when inference proved it effect-free.
            return !stmt_effect_free(inst)
        end
        # Non-call Exprs (:boundscheck, :foreigncall, :new with mutable type…):
        # keep unless effect-free.
        return !stmt_effect_free(inst)
    end
    return must_keep(block, s)
end

"""
    _add_dep!(graph, key, dep)

Add a dependency edge: `key` depends on `dep`.
"""
function _add_dep!(graph::Dict{Any, Vector{Any}}, @nospecialize(key), @nospecialize(dep))
    deps = get!(Vector{Any}, graph, key)
    push!(deps, dep)
end

"""
    _build_dataflow_graph!(graph, roots, op_to_cf, block, ...)

Recursively build the dependency graph for a block and all nested blocks.
"""
function _build_dataflow_graph!(graph::Dict{Any, Vector{Any}},
                                  roots::Set{Any},
                                  op_to_cf::Dict{UInt64, CFNode},
                                  block::Block,
                                  innermost_loop_op,
                                  innermost_loop_cf::Union{CFNode, Nothing},
                                  innermost_cf::Union{CFNode, Nothing})
    for inst in instructions(block)
        s = inst[:stmt]
        val = SSAValue(inst.ssa_idx)

        if s isa ForOp
            cf = cf_node(s)
            op_to_cf[objectid(s)] = cf
            graph[cf] = Any[]

            # CF_COND: ForOp depends on its bounds
            is_trackable_value(s.lower) && _add_dep!(graph, cf, s.lower)
            is_trackable_value(s.upper) && _add_dep!(graph, cf, s.upper)
            is_trackable_value(s.step)  && _add_dep!(graph, cf, s.step)

            # CF_NESTED
            innermost_cf !== nothing && _add_dep!(graph, cf, innermost_cf)

            # CF_DEFINED_VARS: body_args depend on init values + CF node
            for i in 1:length(s.init_values)
                ba = s.body.args[i]
                graph[ba] = Any[s.init_values[i], cf]
            end

            _build_dataflow_graph!(graph, roots, op_to_cf, s.body, s, cf, cf)
            _build_loop_result_deps!(graph, block, inst, s, cf)

        elseif s isa LoopOp
            cf = cf_node(s)
            op_to_cf[objectid(s)] = cf
            graph[cf] = Any[]

            innermost_cf !== nothing && _add_dep!(graph, cf, innermost_cf)

            for i in 1:length(s.init_values)
                ba = s.body.args[i]
                graph[ba] = Any[s.init_values[i], cf]
            end

            _build_dataflow_graph!(graph, roots, op_to_cf, s.body, s, cf, cf)
            _build_loop_result_deps!(graph, block, inst, s, cf)

        elseif s isa WhileOp
            cf = cf_node(s)
            op_to_cf[objectid(s)] = cf
            graph[cf] = Any[]

            innermost_cf !== nothing && _add_dep!(graph, cf, innermost_cf)

            for i in 1:length(s.init_values)
                ba = s.before.args[i]
                graph[ba] = Any[s.init_values[i], cf]
                if i <= length(s.after.args)
                    graph[s.after.args[i]] = Any[ba, cf]
                end
            end

            _build_dataflow_graph!(graph, roots, op_to_cf, s.before, s, cf, cf)
            _build_dataflow_graph!(graph, roots, op_to_cf, s.after,  s, cf, cf)
            _build_loop_result_deps!(graph, block, inst, s, cf)

        elseif s isa IfOp
            cf = cf_node(s)
            op_to_cf[objectid(s)] = cf

            deps = Any[]
            is_trackable_value(s.condition) && push!(deps, s.condition)
            innermost_cf !== nothing && push!(deps, innermost_cf)
            graph[cf] = deps

            _build_dataflow_graph!(graph, roots, op_to_cf, s.then_region,
                                   innermost_loop_op, innermost_loop_cf, cf)
            _build_dataflow_graph!(graph, roots, op_to_cf, s.else_region,
                                   innermost_loop_op, innermost_loop_cf, cf)

            _build_if_result_deps!(graph, block, inst, s, cf)

        else
            # Regular instruction — skip if already handled by a CF result builder
            if !haskey(graph, val)
                operands_ = get_stmt_operands(s)
                deps = copy(operands_)
                innermost_cf !== nothing && push!(deps, innermost_cf)
                graph[val] = deps
            end

            if inst_must_keep(block, inst)
                push!(roots, val)
                for op in get_stmt_operands(s)
                    push!(roots, op)
                end
            end
        end
    end

    term = terminator(block)
    _build_terminator_deps!(graph, roots, term, block,
                            innermost_loop_op, innermost_loop_cf, innermost_cf)
end

"""
    _build_terminator_deps!(graph, roots, term, ...)

Add dependency edges for terminators (ContinueOp, BreakOp, YieldOp, ConditionOp, ReturnNode).
"""
function _build_terminator_deps!(graph, roots, term, block,
                                   innermost_loop_op, innermost_loop_cf, innermost_cf)
    term === nothing && return

    if term isa ContinueOp && innermost_loop_op !== nothing
        innermost_cf !== nothing && _add_dep!(graph, innermost_loop_cf, innermost_cf)
        body = innermost_loop_op isa WhileOp ? innermost_loop_op.before : innermost_loop_op.body
        n_carries = length(innermost_loop_op.init_values)
        for i in 1:min(n_carries, length(operands(term)))
            _add_dep!(graph, body.args[i], operands(term)[i])
        end

    elseif term isa BreakOp && innermost_loop_op !== nothing
        innermost_cf !== nothing && _add_dep!(graph, innermost_loop_cf, innermost_cf)
        # Break values flow to loop result getfields; handled by
        # _build_loop_result_deps! scanning reachable terminators.

    elseif term isa ConditionOp && innermost_loop_op !== nothing
        if innermost_loop_cf !== nothing && is_trackable_value(term.condition)
            _add_dep!(graph, innermost_loop_cf, term.condition)
        end
        body = innermost_loop_op isa WhileOp ? innermost_loop_op.before : innermost_loop_op.body
        n_carries = length(innermost_loop_op.init_values)
        for i in 1:min(n_carries, length(operands(term)))
            _add_dep!(graph, body.args[i], operands(term)[i])
        end

    elseif term isa YieldOp
        if innermost_loop_op isa WhileOp
            body = innermost_loop_op.before
            n_carries = length(innermost_loop_op.init_values)
            for i in 1:min(n_carries, length(operands(term)))
                v = operands(term)[i]
                is_trackable_value(v) && _add_dep!(graph, body.args[i], v)
            end
        end

    elseif term isa ReturnNode
        if isdefined(term, :val) && is_trackable_value(term.val)
            push!(roots, term.val)
        end
    end
end

"""
    _build_loop_result_deps!(graph, parent_block, loop_inst, op, cf)

Add dependency edges for a loop's result extractions (getfield calls in parent).
"""
function _build_loop_result_deps!(graph, parent_block::Block, loop_inst::Instruction,
                                    op, cf::CFNode)
    loop_ssa = SSAValue(loop_inst.ssa_idx)
    n_carries = length(op.init_values)

    for inst in instructions(parent_block)
        is_getfield_of(inst[:stmt], loop_ssa) || continue
        field_idx = inst[:stmt].args[3]
        field_idx isa Int || continue
        gf_val = SSAValue(inst.ssa_idx)

        deps = Any[cf]
        if op isa ForOp && field_idx <= n_carries
            push!(deps, op.init_values[field_idx])
        end
        body = op isa WhileOp ? op.before : op.body
        for term in reachable_terminators(body)
            ops = operands(term)
            if field_idx <= length(ops) && is_trackable_value(ops[field_idx])
                push!(deps, ops[field_idx])
            end
        end
        if op isa WhileOp
            for term in reachable_terminators(op.after)
                ops = operands(term)
                if field_idx <= length(ops) && is_trackable_value(ops[field_idx])
                    push!(deps, ops[field_idx])
                end
            end
        end
        graph[gf_val] = deps
    end
end

"""
    _build_if_result_deps!(graph, parent_block, if_inst, op, cf)

Add dependency edges for IfOp result extractions.
"""
function _build_if_result_deps!(graph, parent_block::Block, if_inst::Instruction,
                                  op::IfOp, cf::CFNode)
    if_ssa = SSAValue(if_inst.ssa_idx)

    for inst in instructions(parent_block)
        is_getfield_of(inst[:stmt], if_ssa) || continue
        field_idx = inst[:stmt].args[3]
        field_idx isa Int || continue
        gf_val = SSAValue(inst.ssa_idx)

        deps = Any[cf]
        then_term = terminator(op.then_region)
        if then_term isa YieldOp && field_idx <= length(operands(then_term))
            v = operands(then_term)[field_idx]
            is_trackable_value(v) && push!(deps, v)
        end
        else_term = terminator(op.else_region)
        if else_term isa YieldOp && field_idx <= length(operands(else_term))
            v = operands(else_term)[field_idx]
            is_trackable_value(v) && push!(deps, v)
        end
        graph[gf_val] = deps
    end
end

"""
    is_getfield_of(s, ref::SSAValue) -> Bool

Check if `s` is a `getfield(ref, idx::Int)` expression.
"""
function is_getfield_of(@nospecialize(s), ref::SSAValue)
    s isa Expr || return false
    s.head === :call || return false
    length(s.args) >= 3 || return false
    func = s.args[1]
    is_gf = if func isa GlobalRef
        getfield(func.mod, func.name) === getfield
    else
        func === getfield
    end
    is_gf || return false
    s.args[2] == ref || return false
    s.args[3] isa Int || return false
    return true
end

#=============================================================================
 BFS liveness propagation
=============================================================================#

function _find_live_values!(graph::Dict{Any, Vector{Any}}, live::Set{Any})
    worklist = collect(live)
    while !isempty(worklist)
        val = pop!(worklist)
        for dep in get(graph, val, Any[])
            if dep ∉ live
                push!(live, dep)
                push!(worklist, dep)
            end
        end
    end
end

#=============================================================================
 Pruning
=============================================================================#

function _prune_block!(block::Block, live::Set{Any}, op_to_cf::Dict{UInt64, CFNode},
                         yield_mask)
    changed = false
    to_delete = Instruction[]

    for inst in instructions(block)
        s = inst[:stmt]
        val = SSAValue(inst.ssa_idx)

        if s isa ForOp || s isa LoopOp || s isa WhileOp
            cf = get(op_to_cf, objectid(s), nothing)
            if cf !== nothing && cf ∉ live
                push!(to_delete, inst)
                changed = true
            else
                changed |= _prune_loop!(block, inst, s, live, op_to_cf)
            end

        elseif s isa IfOp
            cf = get(op_to_cf, objectid(s), nothing)
            if cf !== nothing && cf ∉ live
                push!(to_delete, inst)
                changed = true
            else
                changed |= _prune_if!(block, inst, s, live, op_to_cf)
            end

        else
            if val ∉ live && !inst_must_keep(block, inst)
                push!(to_delete, inst)
                changed = true
            end
        end
    end

    for inst in to_delete
        delete!(block, inst)
    end

    changed |= _prune_terminator!(block, live, yield_mask)
    return changed
end

function _prune_loop!(parent_block::Block, inst::Instruction,
                        op::Union{ForOp, LoopOp, WhileOp},
                        live::Set{Any}, op_to_cf::Dict{UInt64, CFNode})
    changed = false
    n_carries = length(op.init_values)
    body = op isa WhileOp ? op.before : op.body

    carry_live = BitVector(false for _ in 1:n_carries)
    for i in 1:n_carries
        carry_live[i] = body.args[i] ∈ live
    end
    loop_ssa = SSAValue(inst.ssa_idx)
    for pinst in instructions(parent_block)
        is_getfield_of(pinst[:stmt], loop_ssa) || continue
        field_idx = pinst[:stmt].args[3]
        field_idx isa Int || continue
        if field_idx <= n_carries && SSAValue(pinst.ssa_idx) ∈ live
            carry_live[field_idx] = true
        end
    end

    if !all(carry_live)
        lc = carries(op)
        old_to_new = filter!(lc) do cr
            carry_live[cr.index]
        end
        _renumber_getfields!(parent_block, loop_ssa, old_to_new)
        _update_cf_result_type!(parent_block, inst, body)
        changed = true
    end

    if op isa WhileOp
        changed |= _prune_block!(op.before, live, op_to_cf, nothing)
        changed |= _prune_block!(op.after, live, op_to_cf, nothing)
    else
        changed |= _prune_block!(op.body, live, op_to_cf, nothing)
    end

    return changed
end

function _prune_if!(parent_block::Block, inst::Instruction, op::IfOp,
                      live::Set{Any}, op_to_cf::Dict{UInt64, CFNode})
    changed = false

    if_ssa = SSAValue(inst.ssa_idx)
    result_type = value_type(parent_block, if_ssa)
    n_results = if result_type === Nothing || result_type === nothing
        0
    elseif result_type <: Tuple
        length(result_type.parameters)
    else
        1
    end

    if n_results > 0
        result_live = BitVector(false for _ in 1:n_results)
        for pinst in instructions(parent_block)
            is_getfield_of(pinst[:stmt], if_ssa) || continue
            field_idx = pinst[:stmt].args[3]
            field_idx isa Int || continue
            if field_idx <= n_results && SSAValue(pinst.ssa_idx) ∈ live
                result_live[field_idx] = true
            end
        end

        if !all(result_live)
            old_to_new = Dict{Int, Int}()
            new_idx = 0
            for i in 1:n_results
                if result_live[i]
                    new_idx += 1
                    old_to_new[i] = new_idx
                end
            end

            _renumber_getfields!(parent_block, if_ssa, old_to_new)

            kept_types = Type[]
            if result_type <: Tuple
                for i in 1:n_results
                    result_live[i] && push!(kept_types, result_type.parameters[i])
                end
            elseif result_live[1]
                push!(kept_types, result_type)
            end
            new_type = isempty(kept_types) ? Nothing : Tuple{kept_types...}
            inst[:type] = new_type

            yield_mask = result_live
            changed = true
        else
            yield_mask = nothing
        end
    else
        yield_mask = nothing
    end

    changed |= _prune_block!(op.then_region, live, op_to_cf, yield_mask)
    changed |= _prune_block!(op.else_region, live, op_to_cf, yield_mask)

    return changed
end

"""
    _prune_terminator!(block, live, yield_mask) -> Bool

Filter dead values from IfOp YieldOp terminators only. Loop terminators are
handled by `filter!(carries(op))` in `_prune_loop!` and must NOT be modified
here to avoid double-removal.
"""
function _prune_terminator!(block::Block, live::Set{Any}, yield_mask)
    term = terminator(block)
    term === nothing && return false

    if term isa YieldOp && yield_mask !== nothing
        ops = operands(term)
        n = min(length(ops), length(yield_mask))
        changed = false
        for i in n:-1:1
            if !yield_mask[i]
                deleteat!(ops, i)
                changed = true
            end
        end
        return changed
    end

    return false
end

#=============================================================================
 Getfield renumbering
=============================================================================#

function _renumber_getfields!(block::Block, cf_ssa::SSAValue, old_to_new::Dict{Int, Int})
    to_delete = Instruction[]
    for inst in instructions(block)
        is_getfield_of(inst[:stmt], cf_ssa) || continue
        field_idx = inst[:stmt].args[3]::Int
        if haskey(old_to_new, field_idx)
            inst[:stmt].args[3] = old_to_new[field_idx]
        else
            push!(to_delete, inst)
        end
    end
    for inst in to_delete
        delete!(block, inst)
    end
end

"""
    _update_cf_result_type!(block, inst, body_block)

Recompute a CF op's result type from its remaining body block args.
"""
function _update_cf_result_type!(block::Block, inst::Instruction, body_block::Block)
    types = Type[arg.type for arg in body_block.args]
    new_type = isempty(types) ? Nothing : Tuple{types...}
    inst[:type] = new_type
end

#=============================================================================
 Top-level API
=============================================================================#

"""
    dce_pass!(sci::StructuredIRCode)

Dead code elimination for structured IR. Removes dead instructions, dead loop
carries, and dead IfOp results via dependency-graph reachability.
"""
function dce_pass!(sci::StructuredIRCode)
    graph = Dict{Any, Vector{Any}}()
    roots = Set{Any}()
    op_to_cf = Dict{UInt64, CFNode}()
    _build_dataflow_graph!(graph, roots, op_to_cf, sci.entry, nothing, nothing, nothing)

    live = copy(roots)
    _find_live_values!(graph, live)

    _prune_block!(sci.entry, live, op_to_cf, nothing)
    return sci
end
