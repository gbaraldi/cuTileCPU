# Memory-effect & purity classification (generic, raw-Julia-IR).
#
# Adapted from cuTile's `analysis/effects.jl`. cuTile keyed every decision on
# its `Intrinsics` module (`load_ptr_tko`, `store_ptr_tko`, `efunc` effect
# overrides, atomic intrinsics). MLIRKernels' SCI is *raw* Julia IR — there is
# no `Intrinsics` module and no `Tile` type — so the model here is driven by:
#
#   1. The Julia-inferred per-statement `IR_FLAG_EFFECT_FREE` bit (carried
#      through the SCI from `IRCode.stmts.flag`). This is the authoritative
#      "no observable side effect beyond the result" signal: stores
#      (`memoryrefset!`), mutating builtins (`setfield!`), and the frontend's
#      `Base.donotdelete`-pinned markers all have the bit OFF; pure arithmetic,
#      `getfield` of immutables, `memoryrefnew`, casts, etc. have it ON.
#
#   2. A small memory-op classifier (`classify_memory_op`) that recognises the
#      *loads* — ops that carry `IR_FLAG_EFFECT_FREE` (the load itself writes
#      nothing) but whose *value* depends on mutable memory state, so CSE/LICM
#      must still treat them as impure. Mirrors cuTile's MEM_LOAD split and
#      LLVM's `mayReadFromMemory`.
#
# Mirrors LLVM's `Instruction::mayReadOrWriteMemory` / MLIR's
# `MemoryEffectOpInterface`: one source of truth for "what does this op do to
# memory", consumed by DCE / CSE / LICM.

@enum MemoryEffect MEM_NONE MEM_LOAD MEM_STORE

# Raw Julia builtins / intrinsics that READ mutable memory. They are inferred
# `effect_free` (the read writes nothing) yet their result is a function of
# memory state, not just operands — so they are NOT valid CSE/LICM candidates.
const _LOAD_NAMES = Set{Symbol}((
    :memoryrefget,     # Base.memoryrefget(ref, order, boundscheck)  — array load
    :arrayref,         # legacy array load (pre-Memory)
    :pointerref,       # unsafe_load / Ptr load
    :atomic_pointerref,
    :getglobal,        # reads a (possibly mutable) global binding
))

# Raw Julia builtins / intrinsics that WRITE / mutate memory (or are otherwise
# observably side-effecting). These already carry `IR_FLAG_EFFECT_FREE` = false
# from inference, so DCE/LICM/CSE handle them via the flag alone — this set is
# only used to give a precise `MEM_STORE` answer when a caller wants one.
const _STORE_NAMES = Set{Symbol}((
    :memoryrefset!,
    :arrayset,
    :setfield!,
    :replacefield!, :modifyfield!, :swapfield!,
    :setglobal!,
    :pointerset, :atomic_pointerset, :atomic_pointermodify,
    :atomic_pointerswap, :atomic_pointerreplace,
    :modifyindex_atomic!,                # Base array @atomic lowering
))

"""
    callee_symbol(func) -> Union{Symbol, Nothing}

Best-effort name of a resolved callee. Handles plain `Function`s,
`Core.Builtin`s, and `Core.IntrinsicFunction`s (whose `nameof` is reliable on
1.11+). Returns `nothing` if no stable name is available.
"""
function callee_symbol(@nospecialize(func))
    func isa Symbol && return func
    try
        return nameof(func)
    catch
        return nothing
    end
end

"""
    classify_memory_op(resolved_func) -> MemoryEffect

Return the memory effect of a resolved call's callee. `MEM_LOAD` for reads of
mutable memory, `MEM_STORE` for writes, `MEM_NONE` otherwise (pure w.r.t.
memory). Keyed on the callee's name so it works for builtins and intrinsics
alike (which don't have stable identities across worlds the way module
functions do).
"""
function classify_memory_op(@nospecialize(resolved_func))
    name = callee_symbol(resolved_func)
    name === nothing && return MEM_NONE
    name in _LOAD_NAMES  && return MEM_LOAD
    name in _STORE_NAMES && return MEM_STORE
    return MEM_NONE
end

"""
    stmt_effect_free(inst::Instruction) -> Bool

Whether `inst` is free of observable side effects, per Julia inference's
`IR_FLAG_EFFECT_FREE` bit. The analogue of LLVM's `!mayHaveSideEffects()`.
This is the gate DCE uses to drop dead ops and LICM/CSE use to reject
side-effecting ops.
"""
stmt_effect_free(inst::Instruction) =
    CC.has_flag(inst[:flag], CC.IR_FLAG_EFFECT_FREE)

stmt_effect_free(flag::UInt32) = CC.has_flag(flag, CC.IR_FLAG_EFFECT_FREE)
