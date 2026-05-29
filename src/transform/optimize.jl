# Generic StructuredIRCode (SCI) optimization passes.
#
# Vendored from cuTile.jl's `src/compiler/transform/` + `analysis/`, adapted to
# MLIRKernels' IRStructurizer (main, ~9 commits ahead of v0.6.2) and to raw
# Julia IR (MLIRKernels' SCI is plain Julia statements — there is no cuTile
# `Tile` type, `Intrinsics` module, or token-threading; see effects.jl).
#
# Self-contained: depends ONLY on `IRStructurizer` + `Core.Compiler`, so it
# loads standalone (no dependency on the rest of MLIRKernels). The single entry
# point is `optimize_sci!(sci) -> sci`, which runs the vendored passes to
# fixpoint.
#
# Passes vendored (generic, backend-agnostic):
#   • DCE   — dead-code elimination via dependency-graph reachability (dce.jl)
#   • CSE   — common-subexpression elimination / value numbering (cse.jl)
#   • LICM  — loop-invariant code motion (licm.jl)
#   • the declarative pattern-rewrite driver infra (rewrite.jl) + a generic
#     canonicalization rule set hook (canonicalize.jl)
#
# Passes SKIPPED (cuTile-tile-specific — see report):
#   tile intrinsics / tilearray, FMA & broadcast fusion, divisibility/bounds,
#   token_keys/token_order, no_wrap, random, plus cuTile's scalar-elim and
#   Julia-intrinsic→Tile-intrinsic lowering (those are the bulk of cuTile's
#   `canonicalize.jl`, all keyed on the `Tile` type).

module SCIOpt

const CC = Core.Compiler

using IRStructurizer
using IRStructurizer: Block, ControlFlowOp, BlockArgument, Instruction,
                      YieldOp, ContinueOp, BreakOp, ConditionOp,
                      IfOp, ForOp, WhileOp, LoopOp, Undef,
                      StructuredIRCode,
                      # utilities used by the passes
                      instructions, blocks, eachblock, terminator,
                      resolve_call, value_type, operands,
                      uses, users, replace_uses!,
                      insert_before!, insert_after!, move_before!,
                      is_defined_outside, carries, reachable_terminators,
                      validate_ssa_defs

using Core: SSAValue, Argument, SlotNumber, ReturnNode, PiNode, QuoteNode,
            GlobalRef

# Analysis / shared infrastructure
include("effects.jl")        # generic purity / memory-effect classification

# Transforms
include("dce.jl")            # defines must_keep / inst_must_keep used by rewrite
include("rewrite.jl")        # declarative pattern-rewrite driver (infra only)
include("cse.jl")
include("licm.jl")
include("canonicalize.jl")
include("pipeline.jl")       # optimize_sci! orchestrator

export optimize_sci!

end # module SCIOpt
