# Declarative IR Rewrite Pattern Framework
#
# Worklist-based fixpoint driver inspired by MLIR's GreedyPatternRewriteDriver.
# Vendored from cuTile's `transform/rewrite.jl`. Patterns compile into
# pattern/rewrite node trees; the driver processes a LIFO worklist until
# fixpoint, re-adding affected instructions when a rewrite fires. Dead code is
# cleaned up by `dce_pass!`.
#
# Adaptations from cuTile (0.6.2 → main, and tile → raw-Julia):
#   • cuTile threaded a `ConstantInfo` (tile constant analysis) through the
#     driver for `$(literal)` matching and constant guards. That analysis is
#     tile-specific and not vendored, so the driver's `constants` is `Any`
#     (typically `nothing`) and `const_value` here is the literal-only form
#     (cuTile's `const_value(::Nothing, op)`). Wire a constant analysis back in
#     by populating `constants` and overriding `const_value` if needed.
#   • Inserted / opcode-changed ops get `CC.IR_FLAG_NULL` (cleared flag),
#     mirroring IRStructurizer's "fresh instruction, then opt-in copyIRFlags"
#     convention. cuTile recomputed flags from per-`Intrinsics` effect
#     overrides (`inferred_flags`); there is no analogue for raw Julia callees,
#     so a cleared flag is the conservative choice (downstream CSE/LICM simply
#     won't treat the rewritten op as pure until re-inference, which is safe).
#
# NOTE: MLIRKernels vendors the *infrastructure* only — cuTile's actual rule
# sets (ALGEBRA/IDENTITY/FMA/…) are all keyed on the `Tile` type / `Intrinsics`
# and are NOT generic, so none are wired by default (see canonicalize.jl).

#=============================================================================
 Generic constant query (literal-only fallback)
=============================================================================#

"""
    const_value(info, op) -> Number | nothing

Literal-only constant resolution: raw numeric operands and numeric QuoteNodes
resolve to their value; everything else (including SSAValues) is treated as
non-constant. `info` is unused in this generic build (no constant analysis is
vendored); kept for signature compatibility with cuTile rule guards.
"""
function const_value(@nospecialize(info), @nospecialize(op))
    op isa Number && return op
    op isa QuoteNode && op.value isa Number && return op.value
    return nothing
end

# Flag for freshly inserted or opcode-changed instructions: cleared.
fresh_flag(@nospecialize(func)) = CC.IR_FLAG_NULL

#=============================================================================
 Pattern & Rewrite Nodes
=============================================================================#

abstract type PatternNode end
struct PCall <: PatternNode; func::Any; operands::Vector{PatternNode}; end
struct PBind <: PatternNode; name::Symbol; end
struct PTypedBind <: PatternNode; name::Symbol; type::Type; end
struct POneUse <: PatternNode; inner::PatternNode; end
struct PLiteral <: PatternNode; val::Any; end
struct PSplat <: PatternNode; name::Symbol; end  # ~x... — captures remaining operands

abstract type RewriteNode end
struct RCall <: RewriteNode; func::Any; operands::Vector{RewriteNode}; end
struct RBind <: RewriteNode; name::Symbol; end
struct RConst <: RewriteNode; val::Any; end
struct RSplat <: RewriteNode; name::Symbol; end  # ~x... — expands splat binding

"""
    RFunc(func)

Imperative rewrite node. The function is called with
`(sci, block, inst, match, driver)` and returns `true` if applied, `false` to
skip this rule and try the next one.
"""
struct RFunc <: RewriteNode; func::Function; end

struct RewriteRule
    lhs::PCall
    rhs::RewriteNode
    guard::Union{Function, Nothing}  # (match, driver) -> Bool, or nothing
    inplace::Bool                    # true = modify matched ops in-place
end
RewriteRule(lhs::PCall, rhs::RewriteNode) = RewriteRule(lhs, rhs, nothing, false)
RewriteRule(lhs::PCall, rhs::RewriteNode, guard) = RewriteRule(lhs, rhs, guard, false)

root_func(rule::RewriteRule) = rule.lhs.func

#=============================================================================
 @rewrite / @rewriter Macros
=============================================================================#

"""
    @rewrite lhs => rhs
    @rewrite(lhs => rhs, guard)
    @rewrite(inplace=true, lhs => rhs)
    @rewrite(inplace=true, lhs => rhs, guard)

Compile a declarative rewrite rule. LHS: `func(args...)` matches calls, `~x`
binds (repeated names require equality), `~x::T` binds with type constraint,
`one_use(pat)` requires single use, `\$(expr)` matches literal values. RHS:
`func(args...)` emits calls, `~x` references bindings, `\$(expr)` injects a
literal constant. Optional `guard` is `(match, driver) -> Bool`.
"""
macro rewrite(args...)
    inplace = false
    positional = Any[]
    for arg in args
        if arg isa Expr && arg.head === :(=) && arg.args[1] === :inplace
            inplace = arg.args[2]::Bool
        else
            push!(positional, arg)
        end
    end
    length(positional) >= 1 || error("@rewrite expects: lhs => rhs")
    ex = positional[1]
    guard = length(positional) >= 2 ? positional[2] : nothing

    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewrite expects: lhs => rhs")
    g = guard === nothing ? :nothing : guard
    esc(:(RewriteRule($(compile_lhs(ex.args[2])), $(compile_rhs(ex.args[3])), $g, $inplace)))
end

"""
    @rewriter lhs => func

Declarative pattern with imperative rewrite. RHS is `(sci, block, inst, match,
driver) -> Bool`.
"""
macro rewriter(ex)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewriter expects: lhs => func")
    esc(:(RewriteRule($(compile_lhs(ex.args[2])), RFunc($(ex.args[3])))))
end

function compile_lhs(ex)
    if ex isa Expr && ex.head === :$
        return :(PLiteral($(ex.args[1])))
    end
    if ex isa Expr && ex.head === :... && length(ex.args) == 1
        inner = ex.args[1]
        if inner isa Expr && inner.head === :call && inner.args[1] === :~ && length(inner.args) == 2
            name = inner.args[2]
            return :(PSplat($(QuoteNode(name))))
        end
    end
    ex isa Expr && ex.head === :call || error("@rewrite LHS: expected call, got $ex")
    f = ex.args[1]
    if f === :~
        inner = ex.args[2]
        if inner isa Expr && inner.head === :(::)
            return :(PTypedBind($(QuoteNode(inner.args[1])), $(inner.args[2])))
        end
        return :(PBind($(QuoteNode(inner))))
    end
    f === :one_use && return :(POneUse($(compile_lhs(ex.args[2]))))
    :(PCall($f, PatternNode[$(compile_lhs.(ex.args[2:end])...)]))
end

function compile_rhs(ex)
    if ex isa Expr && ex.head === :$
        return :(RConst($(ex.args[1])))
    end
    if ex isa Expr && ex.head === :... && length(ex.args) == 1
        inner = ex.args[1]
        if inner isa Expr && inner.head === :call && inner.args[1] === :~ && length(inner.args) == 2
            name = inner.args[2]
            return :(RSplat($(QuoteNode(name))))
        end
    end
    ex isa Expr && ex.head === :call || error("@rewrite RHS: expected call or \$const, got $ex")
    f = ex.args[1]
    f === :~ && return :(RBind($(QuoteNode(ex.args[2]))))
    :(RCall($f, RewriteNode[$(compile_rhs.(ex.args[2:end])...)]))
end

#=============================================================================
 Worklist
=============================================================================#

mutable struct Worklist
    list::Vector{SSAValue}            # entries (SSAValue(-1) = removed sentinel)
    member::Dict{SSAValue, Int}       # val -> position in list
end

const SENTINEL = SSAValue(-1)

Worklist() = Worklist(SSAValue[], Dict{SSAValue, Int}())

function Base.push!(wl::Worklist, val::SSAValue)
    haskey(wl.member, val) && return
    push!(wl.list, val)
    wl.member[val] = length(wl.list)
end

function Base.pop!(wl::Worklist)
    while !isempty(wl.list)
        val = pop!(wl.list)
        val == SENTINEL && continue
        delete!(wl.member, val)
        return val
    end
    return nothing
end

function remove!(wl::Worklist, val::SSAValue)
    pos = get(wl.member, val, 0)
    pos == 0 && return
    wl.list[pos] = SENTINEL
    delete!(wl.member, val)
end

Base.isempty(wl::Worklist) = isempty(wl.member)

#=============================================================================
 Driver State
=============================================================================#

struct DefEntry
    block::Block
    val::SSAValue
    func::Any
end

"""Operands of a DefEntry, read from the live IR."""
function def_operands(entry::DefEntry)
    haskey(entry.block, entry.val.id) || return Any[]
    call = resolve_call(entry.block, entry.block[entry.val.id][:stmt])
    call === nothing && return Any[]
    _, ops = call
    return ops
end

mutable struct RewriteDriver
    sci::StructuredIRCode
    defs::Dict{SSAValue, DefEntry}
    dispatch::Dict{Any, Vector{RewriteRule}}
    worklist::Worklist
    constants::Any                   # optional constant-analysis result (nothing here)
    modified::Set{SSAValue}          # instructions whose operands were modified
    max_rewrites::Int
end

"""Compute fresh use count for an SSA value."""
use_count(driver::RewriteDriver, val::SSAValue) =
    length(uses(driver.sci.entry, val))

#=============================================================================
 Notifications
=============================================================================#

function add_operands_to_worklist!(driver::RewriteDriver, entry::DefEntry)
    for op in def_operands(entry)
        op isa SSAValue || continue
        haskey(driver.defs, op) && push!(driver.worklist, op)
    end
end

function add_users_to_worklist!(driver::RewriteDriver, val::SSAValue)
    for inst in users(driver.sci.entry, val)
        push!(driver.worklist, SSAValue(inst))
    end
end

function erase_op!(driver::RewriteDriver, entry::DefEntry)
    add_operands_to_worklist!(driver, entry)
    if haskey(entry.block, entry.val.id)
        delete!(entry.block, entry.val.id)
    end
    delete!(driver.defs, entry.val)
    remove!(driver.worklist, entry.val)
end

function notify_insert!(driver::RewriteDriver, block::Block, inst::Instruction)
    val = SSAValue(inst)
    call = resolve_call(block, inst)
    call === nothing && return
    func, _ = call
    driver.defs[val] = DefEntry(block, val, func)
    push!(driver.worklist, val)
end

#=============================================================================
 Matching
=============================================================================#

struct MatchResult
    bindings::Dict{Symbol, Any}
    matched_ssas::Vector{SSAValue}
end

function merge_bindings!(dest::Dict{Symbol,Any}, src::Dict{Symbol,Any})
    for (k, v) in src
        if haskey(dest, k)
            dest[k] === v || return false
        else
            dest[k] = v
        end
    end
    return true
end

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PCall,
                       block::Block=driver.sci.entry)
    val isa SSAValue || return nothing
    entry = get(driver.defs, val, nothing)
    entry === nothing && return nothing

    if entry.func === pat.func
        ops = def_operands(entry)
        has_splat = !isempty(pat.operands) && last(pat.operands) isa PSplat
        n_fixed = has_splat ? length(pat.operands) - 1 : length(pat.operands)

        if has_splat ? length(ops) >= n_fixed : length(ops) == n_fixed
            result = MatchResult(Dict{Symbol,Any}(), SSAValue[val])
            for i in 1:n_fixed
                m = pattern_match(driver, ops[i], pat.operands[i], entry.block)
                m === nothing && return nothing
                merge_bindings!(result.bindings, m.bindings) || return nothing
                append!(result.matched_ssas, m.matched_ssas)
            end
            if has_splat
                splat_name = pat.operands[end]::PSplat
                result.bindings[splat_name.name] = collect(ops[n_fixed+1:end])
            end
            return result
        end
    end

    return nothing
end

pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PBind, block::Block=driver.sci.entry) =
    MatchResult(Dict{Symbol,Any}(pat.name => val), SSAValue[])

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PTypedBind,
                       block::Block=driver.sci.entry)
    T = value_type(block, val)
    T === nothing && return nothing
    CC.widenconst(T) <: pat.type || return nothing
    MatchResult(Dict{Symbol,Any}(pat.name => val), SSAValue[])
end

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::POneUse,
                       block::Block=driver.sci.entry)
    val isa SSAValue && use_count(driver, val) == 1 || return nothing
    pattern_match(driver, val, pat.inner, block)
end

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PLiteral,
                       block::Block=driver.sci.entry)
    val === pat.val && return MatchResult(Dict{Symbol,Any}(), SSAValue[])
    if val isa SSAValue
        c = const_value(driver.constants, val)
        c !== nothing && c == pat.val &&
            return MatchResult(Dict{Symbol,Any}(), SSAValue[])
    end
    return nothing
end

#=============================================================================
 Rewrite Application
=============================================================================#

resolve_rhs(driver, block, ref, op::RBind, bindings, root_typ) = bindings[op.name]
resolve_rhs(driver, block, ref, op::RConst, bindings, root_typ) = op.val
function resolve_rhs(driver::RewriteDriver, block, ref, op::RCall, bindings, root_typ)
    operands_ = Any[]
    for sub in op.operands
        if sub isa RSplat
            append!(operands_, bindings[sub.name])
        else
            push!(operands_, resolve_rhs(driver, block, ref, sub, bindings, root_typ))
        end
    end
    typ = root_typ
    for o in operands_
        o isa SSAValue || continue
        t = value_type(block, o)
        t === nothing && continue
        typ = CC.widenconst(t)
        break
    end
    inst = insert_before!(block, ref, Expr(:call, op.func, operands_...), typ;
                          flag=fresh_flag(op.func))
    notify_insert!(driver, block, inst)
    SSAValue(inst)
end

function apply_inplace_rewrite!(driver::RewriteDriver, block, val::SSAValue, rule, match)
    haskey(block, val.id) || return false

    new_operands = Any[resolve_inplace_rhs(driver, match.bindings, op, lhs_op)
                       for (op, lhs_op) in zip(rule.rhs.operands, rule.lhs.operands)]
    new_stmt = Expr(:call, rule.rhs.func, new_operands...)
    if rule.rhs.func === driver.defs[val].func
        block[val.id] = (stmt=new_stmt,)
    else
        block[val.id] = (stmt=new_stmt, flag=fresh_flag(rule.rhs.func))
    end
    driver.defs[val] = DefEntry(block, val, rule.rhs.func)
    push!(driver.worklist, val)
    add_users_to_worklist!(driver, val)
    return true
end

resolve_inplace_rhs(driver, bindings, op::RBind, @nospecialize(lhs_op)) = bindings[op.name]
resolve_inplace_rhs(driver, bindings, op::RConst, @nospecialize(lhs_op)) = op.val

function resolve_inplace_rhs(driver, bindings, op::RCall, lhs_op::PCall)
    op.func === lhs_op.func && length(op.operands) == length(lhs_op.operands) ||
        error("inplace rewrite: RHS sub-call $(op.func) doesn't match LHS structure")
    matched_ssa = @something find_matched_ssa(driver, lhs_op, bindings) error(
        "inplace rewrite: could not find matched SSA for $(lhs_op.func)")
    entry = @something get(driver.defs, matched_ssa, nothing) error(
        "inplace rewrite: no def entry for $matched_ssa")
    haskey(entry.block, matched_ssa.id) ||
        error("inplace rewrite: $matched_ssa not found in block")
    new_ops = Any[resolve_inplace_rhs(driver, bindings, sub_rhs, sub_lhs)
                  for (sub_rhs, sub_lhs) in zip(op.operands, lhs_op.operands)]
    entry.block[matched_ssa.id] = (stmt=Expr(:call, op.func, new_ops...),)
    push!(driver.worklist, matched_ssa)
    return matched_ssa
end

function resolve_inplace_rhs(driver, bindings, op::RCall, @nospecialize(lhs_op))
    error("inplace rewrite: RHS has RCall but LHS has $(typeof(lhs_op)) at same position")
end

function find_matched_ssa(driver, pat::PCall, bindings)
    entry = driver.sci.entry
    for sub in pat.operands
        if sub isa PBind
            bound = get(bindings, sub.name, nothing)
            bound isa SSAValue || continue
            for inst in users(entry, bound)
                call = resolve_call(entry, inst)
                call === nothing && continue
                func, _ = call
                func === pat.func && return SSAValue(inst)
            end
        elseif sub isa PCall
            inner_ssa = find_matched_ssa(driver, sub, bindings)
            if inner_ssa !== nothing
                for inst in users(entry, inner_ssa)
                    call = resolve_call(entry, inst)
                    call === nothing && continue
                    func, _ = call
                    func === pat.func && return SSAValue(inst)
                end
            end
        end
    end
    return nothing
end

function apply_rewrite!(driver::RewriteDriver, block, val::SSAValue, rule, match)
    if rule.inplace
        return apply_inplace_rewrite!(driver, block, val, rule, match)
    end

    entry = driver.defs[val]
    if rule.rhs isa RFunc
        haskey(block, val.id) || return false
        inst = block[val.id]
        rule.rhs.func(driver.sci, block, inst, match, driver) || return false
        return true
    elseif rule.rhs isa RBind
        for inst in users(driver.sci.entry, val)
            push!(driver.modified, SSAValue(inst))
        end
        add_users_to_worklist!(driver, val)
        replace_uses!(driver.sci.entry, val, match.bindings[rule.rhs.name])
        erase_op!(driver, entry)
    else
        for dead_val in match.matched_ssas
            dead_val == val && continue
            dead_entry = get(driver.defs, dead_val, nothing)
            dead_entry === nothing && continue
            use_count(driver, dead_val) == 0 || continue
            erase_op!(driver, dead_entry)
        end
        typ = block[val.id][:type]
        operands_ = Any[]
        for op in rule.rhs.operands
            if op isa RSplat
                append!(operands_, match.bindings[op.name])
            else
                push!(operands_, resolve_rhs(driver, block, val, op, match.bindings, typ))
            end
        end
        new_stmt = Expr(:call, rule.rhs.func, operands_...)
        if rule.rhs.func === driver.defs[val].func
            block[val.id] = (stmt=new_stmt,)
        else
            block[val.id] = (stmt=new_stmt, flag=fresh_flag(rule.rhs.func))
        end
        driver.defs[val] = DefEntry(block, val, rule.rhs.func)
        push!(driver.worklist, val)
        add_users_to_worklist!(driver, val)
    end
end

#=============================================================================
 Driver
=============================================================================#

"""
    rewrite_patterns!(sci, rules; max_rewrites=10_000, constants=nothing)

Apply rewrite rules to the structured IR using a worklist-based fixpoint
driver. Rules are tried until no more matches fire or `max_rewrites` is
reached. Dead code left behind is cleaned up by `dce_pass!`.
"""
function rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule};
                           max_rewrites::Int=10_000,
                           constants=nothing)
    dispatch = Dict{Any, Vector{RewriteRule}}()
    for rule in rules
        push!(get!(dispatch, root_func(rule), RewriteRule[]), rule)
    end

    defs = Dict{SSAValue, DefEntry}()
    for block in eachblock(sci)
        for inst in instructions(block)
            call = resolve_call(block, inst)
            call === nothing && continue
            func, _ = call
            val = SSAValue(inst)
            defs[val] = DefEntry(block, val, func)
        end
    end

    wl = Worklist()
    for block in eachblock(sci)
        for inst in instructions(block)
            val = SSAValue(inst)
            haskey(defs, val) && push!(wl, val)
        end
    end

    driver = RewriteDriver(sci, defs, dispatch, wl, constants, Set{SSAValue}(), max_rewrites)

    num_rewrites = 0
    while !isempty(driver.worklist) && num_rewrites < driver.max_rewrites
        val = pop!(driver.worklist)::SSAValue
        entry = get(driver.defs, val, nothing)
        entry === nothing && continue

        haskey(entry.block, val.id) || begin
            delete!(driver.defs, val)
            continue
        end

        # Trivial dead-op elimination: keeps use counts accurate for `one_use`.
        if use_count(driver, val) == 0
            if !inst_must_keep(entry.block, entry.block[val.id])
                erase_op!(driver, entry)
                continue
            end
        end

        applicable = get(driver.dispatch, entry.func, nothing)
        matched = false
        if applicable !== nothing
            for rule in applicable
                m = pattern_match(driver, val, rule.lhs)
                m === nothing && continue
                rule.guard !== nothing && !rule.guard(m, driver) && continue
                if apply_rewrite!(driver, entry.block, val, rule, m) === false
                    continue
                end
                num_rewrites += 1
                matched = true
                break
            end
        end

        if !matched && val in driver.modified
            delete!(driver.modified, val)
            for inst in users(driver.sci.entry, val)
                uv = SSAValue(inst)
                push!(driver.modified, uv)
                haskey(driver.defs, uv) && push!(driver.worklist, uv)
            end
        end
    end
    return sci
end
