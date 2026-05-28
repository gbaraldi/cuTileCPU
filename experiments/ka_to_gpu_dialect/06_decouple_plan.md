# Decouple + KA coverage plan (design workflow wf8s72lx1)

## PLAN
The `cd` failed silently and the first command ran in `cuTile.jl`. But I got the key data: the SPMD test (`runtests.jl:2051`) asserts `arith.addf` appears — confirming the walker must handle the add. The reflection path (`code_mlir(...; spmd=true)`) is tested in lockstep (R6 confirmed). Let me verify the current SPMD path: does `spmd_function` today go through `run_passes!` (which rewrites `add_float`→`addf`)? Yes — `_structured_with_analyses` calls `run_passes!` at `launch.jl:80-82`. So today the walker sees `:addf`. Under `Frontend.structured` (no passes) it would see `:add_float`. **R1 is confirmed and decisive.**

Now I have all the facts I need. The change-set is correct except it missed R1/R2/R3 (the intrinsic-name rewrite is load-bearing, not optimization). My plan must insert that as a mandatory step. I have enough to write the authoritative plan without further tool calls.

---

# Implementation Plan: Decouple SPMD/KA/GPU from cuTile, then Broaden KA Coverage

## Reconciliation note (read first)

The three analyses agree on the mechanics but **conflict on sufficiency**. The CHANGE-SET claims Edits 1-4 are enough; the REGRESSION AUDIT (R1/R2/R3) proves they are not. I verified the audit directly:

- `callee_name` (`lower.jl:2852-2857`) returns `nameof(f)` for a `Function`, so `Core.Intrinsics.add_float` → `:add_float`.
- `walk_call!` has cases **only** for the cuTile-rewritten names `:addf`/`:addi`/`:cmpi`/`:bitcast`/`:select` (`lower.jl:1513,1594,1662,1701`), **none** for raw `:add_float`/`:sub_int`/`:ult_int`/`:===`/`:ifelse`.
- The current SPMD path sees `:addf` **only because** `_structured_with_analyses` runs `run_passes!` → `canonicalize!` → `lower_intr_pass!`, whose `INTRINSIC_RULES` (`canonicalize.jl:17-68`) rewrite `Core.Intrinsics.add_float → addf` etc.
- The SPMD test (`runtests.jl:2051`) hard-asserts `@test occursin("arith.addf", mlir)`.

**Therefore: dropping `run_passes!` for SPMD without porting `INTRINSIC_RULES` makes lowering throw `unhandled callee :add_float` on the very first arithmetic op.** This is the single riskiest fact and reshapes Part A: the intrinsic-name rewrite is a **mandatory prerequisite step**, sequenced *before* any path is switched.

The CHANGE-SET author tested only the KA/GPU `global_index` sentinel path (which has no arithmetic in the trivial vadd-on-GPU case routing through the sentinel), not the SPMD arithmetic body. That is the gap.

---

# PART A — DECOUPLE

Each step leaves the tree green and is independently verifiable. The Frontend module already exists (`frontend.jl`) and is correct per all three analyses — no change there.

**Baseline before touching anything.** Note: the REGRESSION AUDIT flagged that `LLVM_full_jll [whYTI]` fails to load (`Zstd_jll` missing), blocking precompile. Step 0 is to confirm the baseline actually builds/tests in *this* environment, because if it doesn't, no verification command below is trustworthy.

### Step 0 — Establish a green baseline
- **File(s):** none (read-only).
- **Change:** none.
- **Verify:** `julia --project=. -e 'using cuTileCPU'` then `julia --project=. test/runtests.jl`. Capture which testsets pass — specifically `"SPMD: vadd"` (`runtests.jl:2051`) and `"SPMD: vadd with alignment=128"` (`:2089`).
- **If precompile fails** (the `Zstd_jll`/`LLVM_full_jll` issue): fix the environment first — `julia --project=. -e 'using Pkg; Pkg.add("Zstd_jll"); Pkg.resolve()'` or pin a working `LLVM_full_jll`. **Do not proceed** until `using cuTileCPU` works and the two SPMD testsets are green. Every step below depends on this.
- **Rollback:** n/a.

### Step 1 — Port the intrinsic-name rewrite into the Frontend path (THE RISKIEST STEP — do it first, standalone, before any switch)
This is R1/R2/R3 from the audit. It must land and be verified while SPMD/KA still run on the old `_structured_with_analyses` path, so it can be tested in isolation.
- **File(s):** `src/frontend.jl` (add the pass), wired in `Frontend.structured` (`frontend.jl:114-122`).
- **Change:** Add a pre-structurize (or post-structurize, pre-walk) intrinsic-lowering pass that reproduces cuTile's `INTRINSIC_RULES` (`canonicalize.jl:17-68`) **without** depending on cuTile. Two implementation options — **choose (b)**:
  - **(a) SCI rewrite pass** mirroring `lower_intr_pass!`. Requires reimplementing `rewrite_patterns!`/`RewriteRule` machinery → heavy, drags in IRStructurizer rewrite internals.
  - **(b) Add walker name-cases (CHOSEN — minimal, no new pass infra).** In `walk_call!` (`lower.jl:1438+`), add raw-intrinsic aliases that route to the existing emitters. These fire only when the name isn't already the cuTile form, so the cuTile tile path is unaffected:
    - `:add_float`→`emit_binop_value!(lc,args,_arith.addf)`; `:sub_float`→`subf`; `:mul_float`→`mulf`; `:div_float`→`divf`; `:neg_float`→negate.
    - `:add_int`→`emit_binop_value!(lc,args,_arith.addi)`; `:sub_int`→`subi`; `:mul_int`→`muli`; `:neg_int`.
    - `:slt_int`/`:sle_int`→`cmpi(LessThan/LessThanOrEqual, Signed)`; `:ult_int`→`cmpi(LessThan, Unsigned)`; `:eq_int`/`:===`→`cmpi(Equal, Signed)`.
    - `:and_int`/`:or_int`/`:xor_int`→`andi`/`ori`/`xori`; `:not_int`→`xori` with type-dependent all-ones const (mirror `canonicalize.jl:38-43`).
    - `:lt_float`/`:le_float`/`:eq_float`/`:ne_float`→`cmpf(...)`.
    - **`:bitcast` (R2 — arg-order):** raw `Core.Intrinsics.bitcast(T, x)` has args `(T, x)`; `emit_bitcast!` (`lower.jl:2893`) expects `(value, type)`. Add a `:bitcast` case that detects arg order — if `args[1]` resolves to a `Type`, swap to `emit_bitcast!(lc, [args[2], args[1]], typ)`. (The cuTile-rewritten form already arrives swapped, so guard on which arg is the `Type`.)
    - **`:ifelse` (R3):** `Core.ifelse`→ existing `:select` emitter.
  - Gate none of these on `lc.spmd` — they are pure scalar/vector arith and are valid on any path; the cuTile path simply never produces raw names so the new cases are dead there.
- **Verify (while still on old path):** `julia --project=. test/runtests.jl` — the 2 SPMD testsets stay green (new cases are dead until Step 3). Then a standalone check that the new cases compile: a temporary `@test` lowering a Frontend SCI of `vadd_spmd` (build via `cuTileCPU.Frontend.structured(vadd_spmd, Tuple{Vector{Float32},Vector{Float32},Vector{Float32},Int})` then walk) — must produce `arith.addf` + `vector<16xf32>`, **not** throw `unhandled callee :add_float`. Add a branchy kernel (`c[i] = a[i] > 0f0 ? a[i] : b[i]`) to exercise `:cmpf`/`:ifelse`/`:select` (R3 has no existing coverage).
- **Rollback:** revert the added `walk_call!` cases. Old path unaffected (it never hit them).
- **Why riskiest:** if this is wrong or incomplete, Steps 3-4 produce a hard throw on real kernels, and the failure mode (`unhandled callee :X`) only surfaces for the *specific* ops a kernel uses — so incomplete coverage ships latent. Porting `INTRINSIC_RULES` in full (every row of `canonicalize.jl:17-68`) up front is the mitigation.

### Step 2 — Walker sentinel clause: add `:global_index` and `:barrier`
This is CHANGE-SET Edit 1. Independent of the path switch; safe to land while still on the old path (the old cuTile-overlay names stay).
- **File(s):** `src/lower.jl:1765-1774`.
- **Change:** Replace the sentinel clause to recognize `:global_index` (new Frontend name) **alongside** the legacy `:__cutilecpu_spmd_lane_id`/`:__ka_lane_id`, returning `lc.arg_vals[lc.lane_arg]` when `lc.spmd`. Add a separate `:barrier` clause returning `nothing` (no-op) when `lc.spmd`. **Do not** add `:block_index`/`:block_dim` (no codegen yet — let them fall through to the `error` at `:1776`).
- **Verify:** `julia --project=. test/runtests.jl` — SPMD testsets green (clause is additive; legacy names still matched, so nothing regresses).
- **Rollback:** restore the original 10-line clause.

### Step 3 — Switch `spmd_function` to `Frontend.structured` + reflection in lockstep (R6)
CHANGE-SET Edit 2, **plus** the mandatory R6 lockstep the change-set omitted from its ordered sequence.
- **File(s):** `src/launch.jl:472` AND `src/reflect.jl:19` (the SPMD branch of `code_mlir`).
- **Change:** In both, replace `sci, rettype, _, _ = _structured_with_analyses(f, argtypes)` with `sci, rettype = Frontend.structured(f, argtypes)`. **Both must change together** — `runtests.jl:2073-2086,2115-2116` assert on `code_mlir(...; spmd=true)`; if only `spmd_function` switches, the reflection tests validate a path the launcher no longer uses (false green, R6).
- **Verify:** `julia --project=. test/runtests.jl` — both SPMD testsets must pass end-to-end: numeric `c ≈ a .+ b`, **and** reflection asserts `arith.addf` + `vector<16xf32>` + `scf.parallel` + (alignment variant) `memref.assume_alignment`/`strided<[1]>`. This is the proof Step 1's intrinsic rewrite works on the real Frontend SCI. Confirm the lane is still the trailing `Int` (`lower.jl:636`), not the sentinel.
- **Rollback:** revert both lines to `_structured_with_analyses`. (Step 1's walker cases are harmless if left in.)

### Step 4 — KA extension: re-home overlays to `Frontend.METHOD_TABLE`, drop cuTile, delete `__init__`
CHANGE-SET Edit 4 + REGRESSION R4. Must be one atomic change (overlay table + inference call + intrinsic module move together, per R4 — piecemeal makes the lane sentinel resolve wrong).
- **File(s):** `ext/KernelAbstractionsExt.jl` (delete `using cuTile`/`const ct` at `:22-23`; replace overlay+`__init__` block `:28-93`) AND `src/launch.jl:533` (`ka_function` → `Frontend.structured`).
- **Change:**
  - Delete `using cuTile`/`const ct` (`:22-23`).
  - `const Frontend = cuTileCPU.Frontend`.
  - Overlays into `Frontend.METHOD_TABLE`: `KA.__index_Global_Linear(ctx) = Frontend.Intrinsics.global_index()`; `KA.__validindex(ctx) = true`; `KA.__synchronize() = Frontend.Intrinsics.barrier()`; `KA.SharedMemory(...)`/`KA.Scratchpad(...)` → `error(...)` stubs.
  - Delete the `__init__` `Core.eval`-into-`cuTile.Intrinsics` hack — overlays now register at precompile against our own table (legal: adding methods for a foreign generic into *our* table).
  - `launch.jl:533`: `sci, rettype = Frontend.structured(f, argtypes)`.
- **Verify:** Precompile cuTileCPU, then `using KernelAbstractions` to load the ext (must precompile with **no** `__init__` eval). Run the KA regression test added in Step 4a below.
- **Rollback:** restore the old `ext/KernelAbstractionsExt.jl` block + `launch.jl:533`. **Critical:** if rolled back, also revert any `Frontend.Intrinsics.barrier`/`global_index` references that only the ext used.

### Step 4a — Add the missing KA regression test BEFORE relying on Step 4 (R5)
R5: there are **zero** KA tests today, so Step 4 is otherwise unverifiable. Land the test as part of Step 4.
- **File(s):** `test/runtests.jl` (new `@testset "KA: vadd"`).
- **Change:** A KA `@kernel function vadd_ka(a,b,c) i=@index(Global,Linear); c[i]=a[i]+b[i] end`, launched on `cuTileBackend()`, asserting `c ≈ a .+ b`; plus `code_mlir` reflection asserting `arith.addf` and that inference left a `:global_index` call (not an inlined body — proves Frontend's default opt params keep the `@noinline` intrinsic alive, and Step 2's `:global_index` clause fires). Add a `@synchronize` no-op kernel asserting it compiles (Step 2 `:barrier` clause).
- **Verify:** `julia --project=. test/runtests.jl`.
- **Rollback:** remove the testset.

### Step 5 — Zero-cuTile audit + experiments
CHANGE-SET Edit 5. Confirmation only.
- **File(s):** `ext/KernelAbstractionsExt.jl`, `experiments/ka_to_gpu_dialect/` (the GPU experiments; `07_*` referenced in the change-set lives here).
- **Change:** `grep -n 'cuTile\|\bct\b' ext/KernelAbstractionsExt.jl` must return only comments. Repoint the GPU experiments' SCI construction to `cuTileCPU.Frontend.structured` (experiment-local, outside the package). Do **not** remove `_structured_with_analyses` — `cpu_function` (`launch.jl:109`) and `reflect.jl` tile path still need it.
- **Verify:** run the GPU experiment in `experiments/ka_to_gpu_dialect/` (the highest-numbered driver) — it must lower via Frontend and emit `gpu.thread_id`/`gpu.block_id`. `julia --project=. test/runtests.jl` full pass.
- **Rollback:** experiments are non-package; revert individually.

**Riskiest step flagged: Step 1.** It is the one the CHANGE-SET missed entirely, it is a hard-throw (not silent) failure mode, and its coverage must be *complete* (every `INTRINSIC_RULES` row) or kernels using an unported op throw at lowering time. Land and verify it standalone (Step 1) before any path switch (Steps 3-4).

---

# PART B — BROADEN KA COVERAGE

Foundation from Part A: overlays live in `Frontend.METHOD_TABLE`, markers in `Frontend.Intrinsics` (`frontend.jl:43-59`), walker clauses gated on `lc.spmd` in `walk_call!` (`lower.jl:1438+`), CPU via `lower_to_mlir_ka`/`_spmd` (single grid dim, `lower.jl:818-974`), GPU via `lower_to_mlir_gpu` (x-dim only, `lower.jl:1024-1156`).

Each feature is: **KA fn → `Frontend.Intrinsics` marker → overlay → walker clause → CPU op → GPU op → validation kernel.**

### Tier 1 — 1-D index kernels (THIS SESSION cut starts here)

**B1. `@index(Global,Linear)` — DONE in Part A Step 4.** Baseline (`vadd`). No further work.

**B2. `memcopy` / SAXPY / elementwise-unary.** Same intrinsic set as B1; validates Frontend path on 3+ array args and a `@uniform` scalar coefficient (item #9 — uniform scalars flow through `_spmd_harmonise`, `lower.jl:1794`, already). 
- **Marker/clause:** none new.
- **CPU/GPU:** existing `memoryref` transfer clauses (`lower.jl:3531-3660`).
- **Validate:** KA `memcopy`, `saxpy(y,a,x)`. **Trivial.**

**B3. `@index(Local,Linear)`, `@index(Group,Linear)`, `@groupsize()`.** First new-intrinsic milestone — pure index lowering, no memory model.
- **Markers (`frontend.jl:43-59`):** add `local_index()::Int32`, `group_index()::Int32`; reuse `block_dim(dim)`.
- **Overlays:** `KA.__index_Local_Linear→local_index()`, `KA.__index_Group_Linear→group_index()`, `KA.groupsize→block_dim(Int32(0))`.
- **Walker clauses (`lower.jl:1438+`, `lc.spmd`):** `:local_index`, `:group_index`, `:block_dim`.
  - CPU `:local_index` = lane vector **without** the `bid*W` base = `_emit_step_vec`+1 (split the base-add out of `lower.jl:951-963` so the pre-base step vector is reusable).
  - CPU `:group_index` = uniform scalar `bid+1` (`lc.bids[1]` index-cast; first lane-derived **uniform**, composes via `_spmd_harmonise`).
  - CPU `:block_dim` = compile-time `arith.constant lc.lane_width`.
  - GPU: `_gpu.thread_id+1`, `_gpu.block_id+1`, `_gpu.block_dim` (x; `lower.jl:1136-1138`).
- **Validate:** `y[i] = @index(Local,Linear) + @index(Group,Linear)*@groupsize()`. **Moderate.**

**B4. KA atomics (`Atomix.@atomic` / histogram).** High ROI — RMW emitters already exist.
- **Marker:** none (match the lowered `Atomix.modify!`/`memoryref`-rooted atomic form directly).
- **Walker clause:** new SPMD dispatch routing an Atomix atomic on the `OffsetInfo` (already built from `memoryrefnew` at `lower.jl:3531-3564`) into the existing `_emit_one_atomic_rmw!` (`lower.jl:3139`). Identify the exact post-inference `Atomix.modify!`/`Core.Intrinsics.atomic_*` call shape first.
- **CPU:** `memref.atomic_rmw <kind>`. **GPU:** same → PTX `atom.*` via gpu→nvvm.
- **Validate:** histogram `@atomic bins[bucket]+=1`. **Moderate.**

### ——— CUT LINE: ship B1–B4 this session ———
B1-B4 are low-risk, need no new memory model, and broaden coverage to **all 1-D index kernels + atomics**. They build only on Part A's decoupled foundation plus per-axis-free index lowering. Stop here for the session.

### Tier 2 — FUTURE (own milestones)

**B5. Multi-dim `@index` (Cartesian/NTuple; N-D Linear).** Highest-leverage hard task; scope as its own milestone *before* localmem.
- **Markers:** extend `global_index`/`local_index`/`group_index` to take `dim::Int32` (mirror `block_index(dim)`).
- **Walker:** dim-parameterised clauses + model `CartesianIndex`/`Tuple`/`getindex` chains (no walker support today).
- **CPU:** generalise `lower_to_mlir_ka` beyond hardcoded `n_grid_dims=1` (`lower.jl:818`); per-axis lane vector (one axis varies, others uniform per block) + multiple `index` block args.
- **GPU:** y/z dims (`#gpu<dim y/z>`; `lower.jl:1135` is x-only); multi-dim `gpu.launch`.
- **Validate:** 2-D `memcopy`, then `naive_transpose` (exercises gather/scatter fallback `lower.jl:3611-3658`, `_is_contiguous_lane_index_from_offset` `lower.jl:3669`). **Hard.**

**B6. `@synchronize` (meaningful) + `gpu.barrier`.** CPU no-op already lands in Part A Step 2; here add `_gpu.barrier()` (GPU dialect imported as `_gpu`, `lower.jl:30`, but `barrier` not yet called). Only *useful* paired with B7.

**B7. `@localmem` / `SharedMemory`.** Hard — needs a memory model the single-block-vector SPMD design lacks.
- **Marker:** `shared_array(T,dims)` returning a **tracked handle** (like `TensorViewInfo`, `lower.jl:41-66`), not a scalar.
- **Walker:** `:shared_array` (alloc+track); teach `emit_spmd_memoryrefnew!/get!/set!` (`lower.jl:3531-3660`) to target a shared buffer, not just Argument-rooted memrefs.
- **CPU:** `memref.alloca` outside lane vectorisation; lane access = scatter/gather into it. **GPU:** `memref.alloca` in `#gpu.address_space<workgroup>`.
- **Validate:** tiled matmul (shared-mem variant). **Hardest.**

**B8. `@private` / `Scratchpad`.** Moderate-hard; shares tracked-handle + `Val{Dims}` unwrapping with B7 but **no** cross-lane/barrier semantics (simpler). CPU = per-lane `vector<W×…>` or private alloca; GPU = per-thread alloca. Validate: per-workitem scratch kernel.

**B9. Block reduction (`@localmem` + tree reduction across `@synchronize`).** Depends on B5+B6+B7. CPU "lanes = SIMD vector" model maps a tree reduction to `vector.multi_reduction` (if recognised) or scatter→barrier-noop→gather. **Research-level on CPU.**

**B10. `@print`/`__print`.** Low value; defer indefinitely. `vector.print` (CPU) / vprintf (GPU).

**Recommended next-milestone order after this session:** B5 (multi-dim grid — unblocks all 2-D) → B6 → B7/B8 → B9.

---

## Key file:line anchors
- Walker dispatch / `unhandled callee` error: `src/lower.jl:1438`, `:1776`; `callee_name`: `:2852-2857`.
- Existing arith cases to alias from raw names (Step 1): `:addf` `lower.jl:1513`, `:addi` `:1594`, `:cmpi` `:1662`, `:bitcast` `:1701`/`emit_bitcast!` `:2893`.
- cuTile rules to port (Step 1 source of truth): `cuTile.jl/src/compiler/transform/canonicalize.jl:17-68`.
- Sentinel clause (Step 2): `src/lower.jl:1765-1774`.
- Path switches: `spmd_function` `src/launch.jl:472`; `ka_function` `:533`; reflection `src/reflect.jl:19`.
- KA ext (Step 4): `ext/KernelAbstractionsExt.jl:22-23`, `:28-93`.
- SPMD lane detection: `src/lower.jl:636-637`. KA entrypoint (1-D): `:818-974`. GPU entrypoint (x-only): `:1024-1156`. SPMD memoryref/gather/scatter: `:3531-3689`. Atomic emitters: `:3139-3422`.
- SPMD testsets (regression surface): `test/runtests.jl:2051`, `:2089`.

## CHANGE-SET ANALYSIS
Confirmed: `StructuredIRCode.argtypes = copy(ir.argtypes)`, and `ir.argtypes[1]` is the function type (standard Julia IRCode convention) — identical to what cuTile's `emit_structured` produces. So slot-1-is-function and `Core.Const` entries are preserved identically. The lowering functions need no change for argtype handling.

I now have everything required. Here is the complete change-set.

---

# CHANGE-SET: Wire `Frontend.structured` into the SPMD / KA / GPU paths

## Findings that drive the plan

- **`Frontend.structured(f, argtypes::Type)`** (`frontend.jl:114-122`) returns `(sci, rettype)` where `sci = StructuredIRCode(ir)` and `ir = Base.code_ircode(f, tt; interp)[1]`. Critically, `StructuredIRCode.argtypes = copy(ir.argtypes)` (`IRStructurizer/src/ir.jl:429`), so `sci.argtypes[1]` is the function type and `Core.Const` slots are preserved — **identical convention** to cuTile's `emit_structured`. Every `lower_to_mlir_*` function loops `for (i, AT) in enumerate(sci.argtypes); i == 1 && continue; AT isa Core.Const && …` (e.g. `lower.jl:434-438`, `:645-650`, `:843-859`, `:1069-1080`). No argtype-shape change is needed anywhere.

- **The plain-Julia paths never consume `divby_info`/`bounds_info`.** Only `cpu_function` (`launch.jl:109,114-115`) and `reflect.jl:19` pass them to `lower_to_mlir`. `spmd_function`/`ka_function` already discard them: `sci, rettype, _, _ = _structured_with_analyses(...)` (`launch.jl:472,533`). So dropping those two return values for SPMD/KA is free.

- **The walker (`walk_block!`) for SPMD/KA/GPU does not depend on any cuTile dataflow pass.** It only depends on `lc.spmd`, `lc.lane_arg`, the `arg_vals`/`arg_const` it sets up, and the callee `fname` dispatch. The only cuTile-frontend-specific recognition is the sentinel clause at `lower.jl:1771`.

- **The sentinel today.** Inference (cuTile + KA overlay) inlines `KA.__index_Global_Linear(ctx)` → `cuTile.Intrinsics.__cutilecpu_spmd_lane_id()`, and `callee_name` (`lower.jl:2852-2857`) returns `:__cutilecpu_spmd_lane_id`. Under `Frontend`, the KA overlay will instead resolve to `Frontend.Intrinsics.global_index()`, so `callee_name` returns `:global_index`.

---

## Edit 1 — Walker sentinel clause (`lower.jl:1765-1774`)

Make the clause recognize the new Frontend intrinsic names. Keep the two legacy names so the cuTile-overlay path and the experiments (`07_ka_kernel_on_gpu.jl`) still work during transition.

`global_index` is the only one any current path emits. `block_index`/`block_dim`/`barrier` have **no walker support yet** and would need real codegen (e.g. `barrier` → `gpu.barrier`/no-op, `block_index` → `lc.bids[dim+1]` cast). **Defer them** — but add an explicit, friendly error for `barrier` (KA `__synchronize` will route to it, see Edit 4) so a no-op is emitted on the CPU path rather than the generic "unhandled callee" crash.

Replace `lower.jl:1765-1774`:

```julia
    # Sentinel function emitted by KernelAbstractions overlays (see
    # `ext/KernelAbstractionsExt.jl`): `KA.__index_Global_Linear(ctx)` is
    # overlaid to `__cutilecpu_spmd_lane_id()`, which inference inlines as
    # a call to a function we never define. The walker recognises the call
    # in SPMD/KA mode and returns the lane vector synthesized at the top of
    # the scf.parallel body.
    if (fname === :__cutilecpu_spmd_lane_id || fname === :__ka_lane_id) &&
       lc.spmd && haskey(lc.arg_vals, lc.lane_arg)
        return lc.arg_vals[lc.lane_arg]
    end
```

with:

```julia
    # Global-index sentinel. Emitted by the standalone `Frontend.Intrinsics`
    # (`Frontend.Intrinsics.global_index()` → callee_name `:global_index`)
    # and, for the legacy cuTile-overlay path, by
    # `cuTile.Intrinsics.__cutilecpu_spmd_lane_id()`. In SPMD/KA mode it
    # returns the lane vector synthesized at the top of the scf.parallel
    # body; in GPU mode it returns the scalar global thread index.
    if (fname === :global_index ||
        fname === :__cutilecpu_spmd_lane_id ||
        fname === :__ka_lane_id) &&
       lc.spmd && haskey(lc.arg_vals, lc.lane_arg)
        return lc.arg_vals[lc.lane_arg]
    end

    # Workgroup barrier sentinel (`Frontend.Intrinsics.barrier()`, callee_name
    # `:barrier`). KA's `__synchronize` overlays to this. On the CPU SPMD/KA
    # path there is no cross-lane barrier to emit (one block == one SIMD
    # vector), so it is a no-op. (GPU `gpu.barrier` lowering is a TODO.)
    if fname === :barrier && lc.spmd
        return nothing
    end
```

Note: `block_index` / `block_dim` are deliberately **not** added. They are unused by any current path and have no codegen; adding them as silent stubs would mask bugs. They fall through to the existing `error(...)` at `lower.jl:1776`. Wire them when a kernel actually needs them.

---

## Edit 2 — `spmd_function` (`launch.jl:464-495`): switch to `Frontend.structured`

**Determination:** SPMD's lane is a real *trailing `Int` arg* (`lower_to_mlir_spmd` finds it via `widenconst(AT) <: Integer`, `lower.jl:636-637`), **not** the `global_index` sentinel. The SPMD body is plain Julia `Vector` indexing → `memoryrefnew/get/set!` (handled by `emit_spmd_*` at `lower.jl:1750-1763`). It uses **no** cuTile tile intrinsic and **no** `divby_info`/`bounds_info`.

But it *does* currently rely on cuTile in one harmful way: `_structured_with_analyses` runs cuTile's interpreter with `inline_cost_threshold = typemax(Int)` (per the `frontend.jl:11-14` comment), which is exactly the inlining behavior `Frontend` was built to avoid. SPMD has no marker to protect today, but routing it through `Frontend` (a) removes the cuTile dependency and (b) gives it default opt params. **Yes, `spmd_function` should switch.** The change is minimal — only the inference call line.

Replace `launch.jl:472`:

```julia
    sci, rettype, _, _ = _structured_with_analyses(f, argtypes)
```

with:

```julia
    sci, rettype = Frontend.structured(f, argtypes)
```

Everything downstream (`lower_to_mlir_spmd`, the `rettype === Nothing` check at `:473`, alignment computation) is unchanged.

---

## Edit 3 — `ka_function` (`launch.jl:533`) + GPU path: switch to `Frontend.structured`

The KA `gpu_*` body references `KA.__index_Global_Linear` / `KA.__validindex` / `KA.__synchronize`, which Edit 4 reroutes onto `Frontend.METHOD_TABLE`. `Frontend.structured` runs the `FrontendInterpreter`, whose `OverlayMethodTable(world, METHOD_TABLE)` (`frontend.jl:85`) applies those overlays, and whose default opt params keep `Frontend.Intrinsics.global_index()`/`barrier()` un-inlined for the walker.

Replace `launch.jl:533`:

```julia
    sci, rettype, _, _ = _structured_with_analyses(f, argtypes)
```

with:

```julia
    sci, rettype = Frontend.structured(f, argtypes)
```

The rest of `ka_function` (`lower_to_mlir_ka` call at `:537-538`, kind `:ka` at `:555-557`) is unchanged. `lower_to_mlir_ka` already takes the first non-Const arg as the lane (`lower.jl:842-852`) — works identically on the Frontend SCI.

**GPU path (`lower_to_mlir_gpu`):** the function itself needs no change — the sentinel it depends on is now handled by Edit 1 (`:global_index`). Its only callers are the experiments (`05/06/07_*.jl`), which call `lower_to_mlir_gpu` directly with an SCI they built themselves. To make them cuTile-free, those experiments should build the SCI via `cuTileCPU.Frontend.structured` instead of whatever cuTile path they use now — but that is experiment-local and outside the package edit. If a packaged `gpu_function` entry point is desired, add it mirroring `ka_function` (same `Frontend.structured` call, then `lower_to_mlir_gpu(sci, argtypes; kernel_name, ctx_arg=…)`). Not required for the stated goal.

**Note on `Frontend` visibility from `launch.jl`:** `frontend.jl` is `include`d at `cuTileCPU.jl:115`, defining submodule `cuTileCPU.Frontend`. `launch.jl` is included at `:118` into the same `cuTileCPU` module scope, so bare `Frontend.structured` resolves. No import line needed.

---

## Edit 4 — KA extension rewrite (`ext/KernelAbstractionsExt.jl`)

Replace the `cuTile.cuTileMethodTable` overlays + the `__init__` `cuTile.Intrinsics` eval hack with overlays into `Frontend.METHOD_TABLE`. All of it works at precompile because:
- We overlay into **our own** module's method table (`cuTileCPU.Frontend.METHOD_TABLE`) — defining an `@overlay` method for a foreign generic (`KA.__index_Global_Linear`) in our table is legal at precompile (it adds a method to *our* table, not KA's).
- The intrinsics referenced (`Frontend.Intrinsics.global_index`, `Frontend.Intrinsics.barrier`) already exist in our package — no cross-package `eval`, no `__init__`.

The `cuTile`/`ct` import is no longer needed for the intrinsic plumbing. It is still needed only for the backend `<: KA.GPU` machinery? No — `cuTileBackend <: KA.GPU` (`:32`) uses only KA. `cuTileCPU.aligned_array` (`:104`) uses cuTileCPU. **The `using cuTile` / `const ct` lines become dead** and should be removed (Edit 5 goal: zero cuTile).

New file shape for `ext/KernelAbstractionsExt.jl` — change the header/overlay region (lines 19-93). Concretely:

Replace `ext/KernelAbstractionsExt.jl:22-23`:

```julia
using cuTile
const ct = cuTile
```

with: *(delete both lines)*

Replace the entire overlay + `__init__` block, `ext/KernelAbstractionsExt.jl:28-93`, with:

```julia
# ----------------------------------------------------------------------------
# 1. Backend
# ----------------------------------------------------------------------------

struct cuTileBackend <: KA.GPU end

# ----------------------------------------------------------------------------
# 2. Overlays into cuTileCPU's Frontend method table
# ----------------------------------------------------------------------------
#
# The KA `@kernel` macro emits a `gpu_*` body that calls a handful of KA
# intrinsics on the `__ctx__`. We redirect those onto cuTileCPU's standalone
# Frontend (cuTileCPU.Frontend), whose interpreter applies METHOD_TABLE and
# uses DEFAULT optimization params (so our `@noinline` marker intrinsics
# survive inference for the walker to intercept):
#
#   • __index_Global_Linear(ctx) → Frontend.Intrinsics.global_index()
#       The walker's `:global_index` clause binds this to the SPMD lane
#       vector (CPU) or the scalar global thread index (GPU).
#   • __validindex(ctx)          → true
#       POC assumes ndrange is a multiple of the workgroup size, so every
#       lane is valid. Masked launches are a TODO.
#   • __synchronize()            → Frontend.Intrinsics.barrier()
#       The walker's `:barrier` clause emits a no-op on the CPU SIMD path
#       (one block == one SIMD vector; no cross-lane barrier).
#
# All of this is legal at PRECOMPILE: we add methods to cuTileCPU's OWN
# overlay table and reference cuTileCPU's OWN intrinsics — no cross-package
# binding mutation, hence no `__init__` eval hack.

const Frontend = cuTileCPU.Frontend

Base.Experimental.@overlay Frontend.METHOD_TABLE KA.__index_Global_Linear(ctx) =
    Frontend.Intrinsics.global_index()

Base.Experimental.@overlay Frontend.METHOD_TABLE KA.__validindex(ctx) = true

Base.Experimental.@overlay Frontend.METHOD_TABLE KA.__synchronize() =
    Frontend.Intrinsics.barrier()

# `SharedMemory` / `Scratchpad` — not yet wired up.
Base.Experimental.@overlay Frontend.METHOD_TABLE KA.SharedMemory(::Type{T}, ::Val, ::Val) where {T} =
    error("cuTileBackend: @localmem / SharedMemory not yet implemented")
Base.Experimental.@overlay Frontend.METHOD_TABLE KA.Scratchpad(ctx, ::Type, ::Val) =
    error("cuTileBackend: @private / Scratchpad not yet implemented")
```

Notes:
- The `import Base.Experimental: @overlay` line (`:26`) can stay; I fully-qualified `Base.Experimental.@overlay` above for clarity but `@overlay Frontend.METHOD_TABLE …` with the existing import works equally. Keep one form consistently.
- `global_index()` returns `Int32` (`frontend.jl:49`). The KA `gpu_*` body uses `i = __index_Global_Linear(ctx)` then indexes `C[i]`. `lower_to_mlir_ka` defaults `lane_idx_type::Type=Int64` (`lower.jl:820`) and builds the lane vector at that type (`:951-962`). With the Frontend intrinsic typed `Int32`, the inferred index is `Int32`; `lower_to_mlir_ka`'s `lane_t` is still `Int64` from the default. This is a **pre-existing** width concern that also existed with the old `tfunc(…)= Int32` override (`KernelAbstractionsExt.jl:68`), so behavior is unchanged. If a type mismatch surfaces, pass `lane_idx_type=Int32` from `ka_function`→`lower_to_mlir_ka`; flag but do not change blindly.
- `_resolve_wgsize`/`launch_config`/`mkcontext` (`:124-178`) are untouched — pure KA, no cuTile.

---

## Edit 5 — Residual cuTile in the plain-Julia paths (goal: zero)

After Edits 1-4, audit:

1. **`launch.jl`** — `spmd_function`/`ka_function` no longer call `_structured_with_analyses` (the only cuTile touchpoint they had). The shared `(k::CPUKernel)(...)` launcher (`:155-261`) and `_ccall_launch_spmd` (`:395-412`) use no cuTile for `:spmd`/`:ka` kinds. The `seed = Base.rand(UInt32)` at `:246` is computed unconditionally but only *used* in the `else` (cuTile) branch (`:255`); harmless, not a dependency. **Clean.** (Do **not** remove `_structured_with_analyses` itself — `cpu_function` at `:109` and `reflect.jl:19` still need it for the cuTile tile path.)

2. **`lower.jl`** — `lower_to_mlir_spmd`/`_ka`/`_gpu` reference `ct.` nowhere (they use `mlir_elem_type`, `widenconst`, `IR.*`). The `:cuTile`-only walker clauses (`make_tensor_view`, `load_partition_view`, etc., `lower.jl:1469-1746`) are simply never reached for SPMD/KA bodies. **Clean.** `widenconst` is `Core.Compiler.widenconst` (`cuTileCPU.jl:80`), not cuTile.

3. **`ext/KernelAbstractionsExt.jl`** — after removing `using cuTile`/`const ct` and the `__init__`, the only non-KA deps are `cuTileCPU` (`aligned_array`, `Frontend`, `ka_function`). **Clean.** Verify no remaining `ct.`/`cuTile.` token survives (the old `KA.SharedMemory`/`Scratchpad` overlays referenced neither; `aligned_array` uses cuTileCPU). 

4. **`frontend.jl`** — already cuTile-free by construction (`frontend.jl:30-31` import only `Core.Compiler` and `IRStructurizer`). **Clean.**

**Net residual cuTile for SPMD/KA/GPU paths after the edits: zero.** cuTile remains a dependency of the *package* solely for the tile path (`cpu_function`/`_structured_with_analyses`/`reflect.jl` and the `mlir_memref_for_tilearray`/`ct.TileArray`/`ct.Constant` handling in `lower_to_mlir` + `_cpu_convert`).

---

## Ordered apply sequence (safe build order)

1. **`frontend.jl`** — no change (already correct; verified standalone).
2. **`lower.jl:1765-1774`** — Edit 1 (sentinel clause: add `:global_index`, add `:barrier` no-op).
3. **`launch.jl:472`** — Edit 2 (`spmd_function` → `Frontend.structured`).
4. **`launch.jl:533`** — Edit 3 (`ka_function` → `Frontend.structured`).
5. **`ext/KernelAbstractionsExt.jl:22-23` + `:28-93`** — Edit 4 (drop `using cuTile`/`const ct`; overlays into `Frontend.METHOD_TABLE`; delete `__init__`).
6. Rebuild/precompile cuTileCPU, then load the KA extension. The `__init__` removal eliminates the runtime `Core.eval` into `cuTile.Intrinsics`; overlays now register at precompile.

## Verification checklist (after applying)
- `spmd_function(vadd, (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int))` compiles and the lane is still the trailing `Int` (`lower.jl:636`), not the sentinel — confirms Edit 2 didn't perturb SPMD lane detection.
- A KA `@kernel` over `cuTileBackend()` lowers: inference must leave a call whose `callee_name` is `:global_index` (not an inlined body) — confirms `Frontend`'s default opt params + the `@noinline` intrinsic survive, and Edit 1's clause fires.
- A KA kernel containing `@synchronize` compiles to a no-op (Edit 1 `:barrier` clause) instead of erroring.
- `grep -n "cuTile\|\bct\b" ext/KernelAbstractionsExt.jl` returns only comment/docstring mentions (or nothing) — confirms Edit 5.

## REGRESSION AUDIT
I have all the information needed. This is a regression audit subagent task — my job is to return the risk register as my final text output, not to mutate task state. Here is the complete analysis.

---

# REGRESSION + HIDDEN-COUPLING AUDIT: `_structured_with_analyses` → `Frontend.structured` for SPMD/KA/GPU

## Verified facts (file:line)

**`_structured_with_analyses` returns 4-tuple; `run_passes!` is a separate SCI→SCI pass.**
`/home/gbaraldi/gpufun/cuTileCPU/src/launch.jl:74-83`: builds the SCI via `ct.emit_structured(ir, rettype)` then *separately* calls `divby_info, bounds_info = ct.run_passes!(sci)`. The SCI is constructed BEFORE passes run — confirmed in cuTile at `/home/gbaraldi/gpufun/cuTile.jl/src/compiler/driver.jl:150-155` (`emit_structured` = `process_meta!` + `StructuredIRCode(ir)`, no passes).

**`StructuredIRCode(ir)` expects raw optimized IRCode, NOT post-pass IR.** `/home/gbaraldi/.julia/packages/IRStructurizer/xRTJx/src/ir/types.jl:657-706` — the constructor takes `ir::IRCode` straight from inference, copies argtypes/sptypes/stmts/types/flags and structurizes the CFG. IRStructurizer's own canonical usage (`interface.jl:55-58`) is `code_ircode(...)` → `StructuredIRCode(ir)` — *exactly* what `Frontend.structured` does (`frontend.jl:117-121`). So `Frontend.structured`'s SCI-construction step is shape-identical to cuTile's; there is no "passes-must-have-run" precondition on the constructor.

**`divby_info`/`bounds_info` are NOT consumed by SPMD/KA/GPU.** Grep across the package: they flow only into `lower_to_mlir` (cuTile tile path, `lower.jl:412-414`) and `reflect.jl:19-25`. The SPMD/KA/GPU lowering signatures take neither: `lower_to_mlir_spmd` (`lower.jl:606-608`), `lower_to_mlir_ka` (`lower.jl:818-820`), `lower_to_mlir_gpu` (`lower.jl:1024-1027`). `spmd_function`/`ka_function` explicitly discard them: `sci, rettype, _, _ = _structured_with_analyses(...)` (`launch.jl:472, 533`). Even on the cuTile path, `lower_to_mlir` only *accepts* them "for future use" — the entry-time alignment chain is spec-only via `arg_chain` / `emit_stride_divby_assumes!` (`lower.jl:517-529`). They are dead on the SPMD/KA/GPU paths today.

**The dominant hidden coupling: `run_passes!`'s FIRST pass (`canonicalize!` → `lower_intr_pass!`) rewrites Julia Core intrinsics into cuTile intrinsics, and the cuTileCPU walker only knows the cuTile-intrinsic names.**
- `/home/gbaraldi/gpufun/cuTile.jl/src/compiler/transform/pipeline.jl:274-275`: `run_passes!` begins with `canonicalize!(sci)`.
- `/home/gbaraldi/gpufun/cuTile.jl/src/compiler/transform/canonicalize.jl:17-68` (`INTRINSIC_RULES`): `Core.Intrinsics.add_float → addf`, `add_int → addi`, `sub_int → subi`, `mul_int → muli`, `slt_int/sle_int/ult_int → cmpi(...)`, `and_int/or_int/xor_int → andi/ori/xori`, `not_int → xori`, `bitcast(T,x) → bitcast(x,T)` (arg order swapped), builtins `===` → `cmpi`, `Core.ifelse → select`.
- The cuTileCPU walker dispatches by name (`callee_name` at `lower.jl:2852-2857`) and has cases ONLY for the cuTile names — `:addf` (`lower.jl:1513`), `:subi` (1597), `:cmpi` (1662), `:bitcast` (1701) — and NONE for `:add_float`/`:sub_int`/`:ult_int`/raw `Core.Intrinsics.*`.

I confirmed empirically what the default interpreter (= Frontend) produces for the exact SPMD test kernel `vadd_spmd` (`@inbounds c[i]=a[i]+b[i]`): optimized IRCode contains `Base.add_float(%17,%35)`, `Base.sub_int`, `Base.ult_int`, `Base.bitcast`, plus the full bounds-check machinery (`getfield(_,:size)`, `throw_boundserror`, `unreachable`). `callee_name(Core.Intrinsics.add_float)` = `:add_float` (it's `isa Function`, so `nameof` is used). The walker would hit `error("cuTileCPU.walk_call!: unhandled callee :add_float ...")` at `lower.jl:1776`.

Note: bounds-check *shape* is identical in both paths (both interpreters use default `InferenceParams()` — `interpreter.jl:29` and `frontend.jl:89`), and bounds elision is handled by IRStructurizer + the walker's `:boundscheck`-sentinel IfOp elision (`lower.jl:2723-2728`), which is shared. So bounds checks are NOT a new risk. The `memoryrefnew/get/set!` + `getfield(arr,:ref/:size)` chains (`lower.jl:3531-3585`) are plain Julia builtins that canonicalize never touches — identical in both paths. The lane `subi -1` is emitted by the walker itself.

---

## Direct answers

**1. Correctness / performance of losing `run_passes!` for memory-bound vadd-class SPMD/KA/GPU kernels.**
- **Correctness: BROKEN, not preserved** — but not for a subtle numeric reason. `run_passes!`'s `canonicalize!`/`lower_intr_pass!` is *load-bearing for the walker to function at all*: it is what currently translates `Base.add_float`/`sub_int`/`ult_int`/etc. into the `:addf`/`:subi`/`:cmpi` names the walker matches. Without it, lowering throws on the first arithmetic op. This is a hard failure, not a perf regression.
- **Performance (assuming the name gap is fixed):** The genuinely *optimization*-flavored passes — `FMA_RULES` (fusion), `cse_pass!`, `licm_pass!`, `no_wrap_pass!`, `dce_pass!` (`pipeline.jl:277-316`) — are largely redundant for memory-bound vadd-class kernels: the downstream MLIR pipeline + `mlir-translate`→LLVM already does CSE/LICM/DCE, and the kernels are DRAM-bandwidth-bound, not FLOP-bound. FMA fusion matters for matmul-class (`mma`) kernels, which are the *tile* path (untouched). For the current SPMD/KA/GPU scope (vadd/elementwise), dropping the optimization passes is performance-neutral. The alignment/vectorization win comes from `memref.assume_alignment` + strided layout emitted in `lower_to_mlir_spmd` (`lower.jl:662`), independent of `run_passes!`.

**2. Are `divby_info`/`bounds_info` used on SPMD/KA/GPU?** No. They are never threaded into `lower_to_mlir_spmd`/`_ka`/`_gpu`. Dropping them on these paths is free.

**3. Does the cuTile tile path stay 100% intact?** Yes. `cpu_function` keeps calling `_structured_with_analyses` (`launch.jl:109`) and passes `divby_info/bounds_info` to `lower_to_mlir` (`launch.jl:114-115`). No shared mutable state: caches are separate (`_kernel_cache` vs `_spmd_kernel_cache` vs `_ka_kernel_cache`, `launch.jl:66/447/511`); each lowering builds its own `fresh_context()` + `LowerCtx`; `_structured_with_analyses` does `sci = copy(sci)` before mutating (`launch.jl:80`), and Frontend builds a brand-new SCI. The two interpreters are distinct types with distinct method tables (`cuTile.cuTileMethodTable`/`get_method_table_view` vs `Frontend.METHOD_TABLE`). One caveat (Risk R6): `reflect.jl` `code_mlir(...; spmd=true)` still routes through `_structured_with_analyses` — if SPMD switches to Frontend but reflection doesn't, the two will diverge and the SPMD reflection tests will exercise a different SCI than the launch path.

**4. IRCode-shape mismatch between `code_ircode` output and the walker?** The container type is identical (both produce `IRStructurizer.StructuredIRCode`, same package version 0.6.2 / `xRTJx`). The mismatch is in *statement content*, concentrated entirely in call-target names: raw `Core.Intrinsics.*` and builtins (`===`, `Core.ifelse`) vs cuTile-rewritten names. Secondary: `Core.Intrinsics.bitcast` arg order is `(T, x)` raw vs `(x, T)` after canonicalize (`lower.jl:1701` `emit_bitcast!` expects `args[1]=value, args[2]=type`). Everything else (`memoryrefnew/get/set!`, `getfield`, `tuple`, `:boundscheck` sentinels, PiNodes, ReturnNode, IfOp/ForOp from structurization) is shared and unaffected.

**5. Test-suite regression surface.** Only **two** testsets touch `spmd_function`: `"SPMD: vadd"` (`runtests.jl:2051`) and `"SPMD: vadd with alignment=128"` (`runtests.jl:2089`). Both also call `code_mlir(...; spmd=true)` for reflection assertions. There are **zero** KA testsets and **zero** GPU testsets in `runtests.jl`; `lower_to_mlir_gpu` has no caller anywhere in src/test (dead/experimental). So the SPMD switch's regression surface is exactly those 2 testsets; KA/GPU have no test coverage to regress (which is itself a risk — R5).

---

## RISK REGISTER

| # | Risk | Likelihood | Impact | Verify / Mitigate |
|---|------|-----------|--------|-------------------|
| **R1** | **Walker can't name-match raw Julia intrinsics.** Frontend SCI keeps `Base.add_float`/`sub_int`/`mul_int`/`ult_int`/`and_int`/`not_int`/`neg_float`/etc.; walker only handles `:addf`/`:subi`/`:cmpi`/... → hard `error` at `lower.jl:1776` on the first arithmetic op. This is the SAME work `canonicalize.jl:17-68` does today. | **Certain** | **Critical** (lowering throws; SPMD/KA produce nothing) | Port `INTRINSIC_RULES` into the Frontend path. Two options: (a) run an equivalent intrinsic-lowering pass on the Frontend SCI before walking; (b) add `:add_float`→addf, `:sub_int`→subi, `:slt_int/sle_int/ult_int`→cmpi(pred,sign), `:and_int/or_int/xor_int`, `:not_int`, `:===`, `:Core.ifelse` cases to `walk_call!`. Verify with `code_mlir(vadd_spmd, ...; spmd=true)` — must still contain `arith.addf` + `vector<16xf32>`. |
| **R2** | **`Core.Intrinsics.bitcast` arg-order mismatch.** Raw form is `bitcast(T, x)`; walker's `emit_bitcast!` (`lower.jl:1701,2893`) expects cuTile's swapped `bitcast(x, T)`. The `UInt` index conversions in array indexing (`%4 = bitcast(UInt, %3)`) hit this. | High (any indexed access) | High (wrong operand or throw) | When porting the bitcast rule, swap operand order (mirror `canonicalize.jl:62`). Verify indexed SPMD kernel lowers + numerically matches. |
| **R3** | **Builtins `===` and `Core.ifelse` survive un-lowered.** Walker has `:select` but `Frontend` SCI emits `Core.ifelse`/`===`. Any branchy SPMD/KA kernel (e.g. `ifelse`, integer `!=`) throws. | Medium (vadd doesn't hit it; richer kernels do) | High | Add `:(===)`→cmpi(Equal,Signed) and `:ifelse`→select to the Frontend lowering. Add a branchy SPMD regression test (none exists today). |
| **R4** | **KA overlay re-homing.** KA path overlays into `cuTile.cuTileMethodTable` + relies on `cuTile.Intrinsics` membership for `isintrinsic`/no-inline (`KernelAbstractionsExt.jl:8-60`, the `__init__` eval hack). Switching to Frontend requires re-registering the `__index_Global_Linear`/`__cutilecpu_spmd_lane_id` overlays into `Frontend.METHOD_TABLE` + `Frontend.Intrinsics`. If only the inference call is swapped but overlays stay in cuTile's table, the KA lane sentinel won't be produced → lane resolves wrong / call inlined away. | High (if done piecemeal) | Critical (KA path silently wrong or throws) | Move the KA overlays to register against `Frontend.METHOD_TABLE`; define the lane sentinel in `Frontend.Intrinsics`; update the walker's `:__cutilecpu_spmd_lane_id`/`:__ka_lane_id` clause (`lower.jl:1771`) — it already matches by name so it's fine once the overlay targets Frontend. **There is NO KA test** — add one before/with the switch. |
| **R5** | **No KA/GPU test coverage.** Zero KA/GPU testsets; `lower_to_mlir_gpu` has no caller. A regression here is invisible to CI. | Certain (coverage gap) | Medium-High (latent breakage ships) | Add at least one KA `@kernel` vadd regression test and one `lower_to_mlir_gpu` smoke/`code_mlir` test BEFORE switching, so the switch is guarded. |
| **R6** | **Reflection divergence.** `reflect.jl:19` `code_mlir(...; spmd=true)` still uses `_structured_with_analyses`. If `spmd_function` switches to Frontend but `code_mlir` doesn't, the two SCIs differ; the SPMD reflection `@test occursin(...)` assertions (`runtests.jl:2078-2086,2115-2116`) validate a path the launch no longer uses. | High (easy to miss) | Medium (false-green tests; debugging confusion) | Route `code_mlir`/`code_mlir_lowered`/`code_llvm` SPMD branches through `Frontend.structured` in lockstep with `spmd_function`. |
| **R7** | **Optimization-pass loss (FMA/CSE/LICM/DCE/no-wrap).** Frontend skips `run_passes!`'s optimizers. Numerically negligible for elementwise; FMA fusion + integer no-wrap flags absent. | High (passes do drop) | Low for current vadd-class scope; Medium if SPMD/KA later targets matmul/`mma` | Accept for current scope (memory-bound, MLIR/LLVM re-optimizes). If matmul-class SPMD/KA lands, add a minimal FMA-fusion pass to Frontend. Verify perf parity with `bench/bench_spmd.jl`. |
| **R8** | **`@noinline`/intrinsic-survival now correct, but subprogram inlining differs.** cuTile's `inline_cost_threshold=typemax` inlines *everything*; Frontend uses default params (the whole point). A previously-inlined helper now stays as an `:invoke` the walker may not handle (walker handles `:invoke` at `lower.jl:1287`, but only if the callee resolves to a known name). | Medium | Medium (unhandled-callee throw on helper-using kernels) | Confirm via `Frontend.structured` on a kernel that calls a small helper; if helpers must inline, either mark them `@inline` in user code or set a higher `inline_cost_threshold` in `FrontendInterpreter` (without going to typemax, to keep `@noinline` markers alive). |
| **R9** | **`scalar_elim_pass!` loss.** canonicalize also runs `scalar_elim_pass!` (`canonicalize.jl:87`) removing `to_scalar`/`from_scalar` and promoting scalar→0D-tile types. These are cuTile-Tile constructs; plain-Julia SPMD/KA IR has no `to_scalar`/`from_scalar`, so this pass is a no-op for them. | Low | Low | None needed for SPMD/KA (no Tile values). Confirm Frontend SPMD SCI contains no `to_scalar`/`from_scalar` (it won't — those only arise from `Tile` ops). |

---

## Bottom line

- **Dropping `divby_info`/`bounds_info` from SPMD/KA/GPU is free** (never consumed).
- **The cuTile tile path is fully isolated** (separate cache, separate context, `copy(sci)`, distinct interpreter) — confirmed safe.
- **The one mandatory port is canonicalize's `INTRINSIC_RULES`** (R1/R2/R3): the SPMD/KA/GPU paths depend on `run_passes!` *not* for optimization but for the `Core.Intrinsics.* → cuTile-Intrinsics.*` name rewrite the walker keys on. `Frontend.structured` must reproduce that lowering (as a pre-walk pass or as added `walk_call!` name cases) or lowering throws immediately.
- **KA path needs its overlays re-homed to `Frontend.METHOD_TABLE`/`Frontend.Intrinsics` in the same change** (R4), and **KA/GPU have zero tests** (R5) — add coverage before the switch.
- **Regression surface for the SPMD switch is exactly 2 testsets** (`runtests.jl:2051, 2089`), both of which also assert on `code_mlir(...; spmd=true)`, so reflection must switch in lockstep (R6).

Key files: `/home/gbaraldi/gpufun/cuTileCPU/src/launch.jl`, `/home/gbaraldi/gpufun/cuTileCPU/src/lower.jl`, `/home/gbaraldi/gpufun/cuTileCPU/src/frontend.jl`, `/home/gbaraldi/gpufun/cuTileCPU/src/reflect.jl`, `/home/gbaraldi/gpufun/cuTileCPU/ext/KernelAbstractionsExt.jl`, `/home/gbaraldi/gpufun/cuTile.jl/src/compiler/transform/canonicalize.jl`, `/home/gbaraldi/gpufun/cuTile.jl/src/compiler/transform/pipeline.jl`, `/home/gbaraldi/gpufun/cuTileCPU/test/runtests.jl`.

Note: I could not execute a live SCI comparison — the environment's `LLVM_full_jll [whYTI]` fails to load (`Zstd_jll` missing from its manifest deps), blocking `cuTileCPU` precompile. All findings above are from static source reading plus a minimal standalone Julia check of `code_ircode`/`callee_name` behavior, which is sufficient and decisive for the central R1/R2 finding.

## KA COVERAGE MAP
I have a complete and exhaustive picture. Confirmed key facts:
- `lower_to_mlir_ka` and `lower_to_mlir_spmd` are single-grid-dim only (`n_grid_dims = 1`, one `index` block arg, lane vector built from `bid * W`).
- The GPU path emits `gpu.thread_id`/`gpu.block_id`/`gpu.block_dim` only on the **x** dimension. No `gpu.barrier`, no `memref.alloca`, no workgroup address space anywhere.
- All atomics, gather/scatter, mma, reduce clauses are keyed on cuTile `Intrinsics` names — they fire on the *tile* path, not the plain-Julia KA/SPMD path (which only goes through `memoryrefnew/get/set!`).

Now I'll write up the coverage map.

---

# KA Coverage Map for cuTileCPU (CPU SPMD + GPU SIMT)

## Ground truth from the code

**KA version in use:** `KernelAbstractions 0.9.41` at `~/.julia/packages/KernelAbstractions/X5fk1/`.

**How the `gpu_*` body is shaped** (`X5fk1/src/macros.jl:83-120`, `transform_gpu!`): the body becomes
```
__active_lane__ = __validindex(__ctx__)   # macros.jl:99
<body, with @index calls hoisted/aggregated by `split(emit_gpu, …)` macros.jl:105>
return nothing
```
On the GPU path `@synchronize` is **not** used to split the body (that is the CPU path's `transform_cpu!`/`WorkgroupLoop`, macros.jl:129-167); on the GPU path `@synchronize` survives inline as a call to `__synchronize()`. cuTileCPU's KA path picks the `gpu_*` body because `cuTileBackend <: KA.GPU` (`ext/KernelAbstractionsExt.jl:32`).

**`@index` expansion** (`macros.jl:422-441`): `@index(locale, kind, I...)` → `KernelAbstractions.__index_{locale}_{kind}(__ctx__, I...)`. `locale ∈ {Global,Local,Group}`, `kind ∈ {Linear,Cartesian,NTuple}` (default `Linear`). Declarations at `macros.jl:448-458`. NTuple variants are `Tuple(__index_*_Cartesian(ctx, I...))`.

**Canonical GPU semantics** (CUDA.jl reference, `CUDA/1WEyo/CUDACore/src/CUDAKernels.jl:161-211`):
- `__index_Local_Linear(ctx) = threadIdx().x`
- `__index_Group_Linear(ctx) = blockIdx().x`
- `__index_Global_Linear(ctx)` = `LinearIndices(ndrange)[expand(iterspace, blockIdx().x, threadIdx().x)]`
- `__index_*_Cartesian` = `expand`/`workitems`/`blocks` of the iterspace (CartesianIndex)
- `SharedMemory(T, Val{Dims}, Val{Id}) = CuStaticSharedArray(T, Dims)` (`@localmem`, macros.jl:237-244)
- `Scratchpad(ctx, T, Val{Dims}) = MArray{__size(Dims),T}(undef)` (`@private`, macros.jl:258-265)
- `__synchronize() = sync_threads()`
- `@uniform`/`@private expr` are pure host-side passthroughs (`macros.jl:283-285`, `273-275`) — no intrinsic.

**Current cuTileCPU handling** (`ext/KernelAbstractionsExt.jl`): only `__index_Global_Linear`→sentinel `__cutilecpu_spmd_lane_id` (lines 61-72), `__validindex`→`true` (line 81), `__synchronize`→`nothing` (line 87). `SharedMemory`/`Scratchpad` overlays throw (lines 90-93). The walker clause that consumes the sentinel is `lower.jl:1771-1774` (returns `lc.arg_vals[lc.lane_arg]`). Everything else in `walk_call!` (atomics, gather/scatter, mma, reduce) keys on **cuTile** `Intrinsics` names and only fires on the tile path.

**Frontend target intrinsics** (`src/frontend.jl:43-59`): `global_index()`, `block_index(dim::Int32)`, `block_dim(dim::Int32)`, `barrier()` — all `@noinline`+`compilerbarrier`, `Int32` (or `nothing`) returns. The overlay table is `Frontend.METHOD_TABLE` (frontend.jl:70). So the migration is: route KA overlays into `Frontend.METHOD_TABLE`, mapping each `__index_*`/`__synchronize`/`SharedMemory`/`Scratchpad` to a `Frontend.Intrinsics.*` marker, and add walker clauses keyed on the new marker names.

---

## Coverage matrix

Difficulty key: **trivial** = overlay + reuse existing clause; **moderate** = new Frontend intrinsic + new walker clause + both lowerings; **hard** = needs grid/iterspace plumbing or a memory model the current single-block-vector SPMD design doesn't have.

### 1. `@index(Global, Linear)` — WORKS today
- **KA fn:** `__index_Global_Linear(ctx)::Int64` (macros.jl:450)
- **Frontend intrinsic:** `global_index()` (exists). Currently routed through cuTile's `__cutilecpu_spmd_lane_id`; migrate the overlay to `Frontend.Intrinsics.global_index`.
- **Walker clause:** exists — `lower.jl:1771-1774` (rename target from `:__cutilecpu_spmd_lane_id` to `:global_index`).
- **CPU SPMD:** the per-iteration lane vector `splat(bid*W+1)+step(0..W-1)` built at `lower.jl:945-963`; bound to `lc.arg_vals[lane_arg]`.
- **GPU:** scalar `gpu.thread_id.x + gpu.block_id.x*gpu.block_dim.x + 1`, `lower.jl:1134-1146`.
- **Difficulty:** trivial (re-pointing the overlay; the only real KA kernel proven end-to-end — `examples/kernel_abstractions_poc.jl` vadd).

### 2. `@index(Local, Linear)` — NEW
- **KA fn:** `__index_Local_Linear(ctx)` = `threadIdx().x` (1-based thread-in-block).
- **Frontend intrinsic:** NEW `local_index()::Int32` (or reuse `block_dim`-style; semantically it is `thread_id.x + 1`).
- **Walker clause:** NEW `:local_index`.
- **CPU SPMD:** the local linear index within a workgroup *is the lane position*: `step(1..W)` = `vector<W × iX>` of `[1,2,…,W]` (i.e. the lane vector **without** the `bid*W` base). Build from `_emit_step_vec` + add 1 (subset of what `lower.jl:951-963` already computes — split the base-add so the pre-base step vector is reusable).
- **GPU:** `gpu.thread_id.x + 1` (`_gpu.thread_id`, already imported `lower.jl:1136`).
- **Difficulty:** moderate (CPU: the no-base lane vector; GPU: trivial scalar). Independent of grid plumbing.

### 3. `@index(Group, Linear)` — NEW
- **KA fn:** `__index_Group_Linear(ctx)` = `blockIdx().x` (1-based block index).
- **Frontend intrinsic:** NEW `group_index()::Int32` (block index, 1-based) — analogous to existing `block_index(dim)` but 1-based and KA-shaped.
- **Walker clause:** NEW `:group_index`.
- **CPU SPMD:** *uniform scalar* `bid + 1` (`lc.bids[1]` index-cast to iX, `lower.jl:954`-style). Note: this is the first KA intrinsic producing a **uniform** (scalar) value in SPMD mode — `_spmd_harmonise` (lower.jl:1794) already broadcasts it when combined with a varying value, so it composes.
- **GPU:** `gpu.block_id.x + 1` (`_gpu.block_id`, `lower.jl:1137`).
- **Difficulty:** moderate (mostly trivial; the only subtlety is making sure a uniform-scalar lane-derived value flows through `_spmd_harmonise` correctly, which it already does).

### 4. `@groupsize()` / block dimension — NEW (needed by reductions & localmem indexing)
- **KA fn:** `groupsize(ctx)` (macros.jl:204,214) — workgroup size tuple. Not an `__index_*` but kernels read it.
- **Frontend intrinsic:** `block_dim(dim::Int32)` (exists, frontend.jl:55).
- **Walker clause:** NEW `:block_dim`.
- **CPU SPMD:** compile-time constant `lc.lane_width` → `arith.constant` (uniform). Trivial.
- **GPU:** `gpu.block_dim.x` (`_gpu.block_dim`, `lower.jl:1138`).
- **Difficulty:** trivial. (Worth doing early — `@localmem`/reduction kernels index by group size.)

### 5. Multi-dim `@index` (Cartesian / NTuple, and Linear on N-D ndrange) — NEW, HARD
- **KA fns:** `__index_Global_Cartesian`, `__index_Local_Cartesian`, `__index_Group_Cartesian` (macros.jl:452-454) return `CartesianIndex{N}`; `*_NTuple` wrap them in `Tuple` (macros.jl:456-458). Multi-dim `Linear` also needs `LinearIndices(ndrange)[…]`.
- **Frontend intrinsic:** NEW per-axis `global_index(dim)`, `local_index(dim)`, `group_index(dim)` (extend the existing single-axis markers to take a `dim::Int32`, mirroring `block_index(dim)`/`block_dim(dim)` which already take a dim).
- **Walker clause:** NEW dim-parameterised clauses; **plus** CartesianIndex construction/destructuring must survive inference (the `CartesianIndex`/`Tuple`/`getindex` chain currently has no walker support — it would surface as `getfield`/`tuple` on a struct the walker doesn't model).
- **CPU SPMD:** the present design lowers exactly **one** grid dim (`n_grid_dims=1` hardcoded in `lower_to_mlir_ka`, `lower.jl:818-974`; single `index` block arg `lower.jl:935`). Multi-dim requires generalising the KA entrypoint to `n_grid_dims` grid args and constructing a **per-axis lane vector** (only one axis varies across lanes; the others are uniform per block). Significant.
- **GPU:** `gpu.thread_id/block_id/block_dim` on `y`/`z` dims (`#gpu<dim y>`, `#gpu<dim z>` — `lower.jl:1135` only emits `x`). Need a 3-element `lc.bids`-style structure and a multi-dim `gpu.launch`.
- **Difficulty:** hard (CPU: multi-grid SPMD redesign + CartesianIndex modelling; GPU: y/z dims + multi-dim launch). Biggest single gap.

### 6. `@synchronize` / `__synchronize()` — partial today
- **KA fn:** `__synchronize()` (macros.jl:805); `@synchronize`/`@synchronize(cond)` both lower to it (macros.jl:297-323).
- **Frontend intrinsic:** `barrier()` (exists, frontend.jl:58, returns nothing).
- **Walker clause:** NEW `:barrier`.
- **CPU SPMD:** currently overlaid to `nothing` (`KernelAbstractionsExt.jl:87`) — **correct only for kernels without cross-lane communication**. With the lane = SIMD-vector model, all W lanes execute in lockstep within one `scf.parallel` iteration, so a barrier *within a block* is genuinely a no-op (emit nothing). This is sound as long as `@localmem` is also modelled as block-local SSA (see #7). So: keep no-op, but it only becomes *meaningful* once localmem exists.
- **GPU:** `gpu.barrier` (NEW — `Dialects.gpu` is imported as `_gpu`, `lower.jl:30`, but `barrier` is not yet called anywhere; add `_gpu.barrier()`).
- **Difficulty:** trivial (no-op CPU + `gpu.barrier`), but only *useful* paired with #7.

### 7. `@localmem` / `SharedMemory(T, Val{Dims}, Val{Id})` — NEW, HARD
- **KA fn:** `SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id})` (macros.jl:237-244 → `KernelAbstractions.jl:803`). Returns an `MtlDeviceArray`/`CuDeviceArray`-like workgroup-shared array; the kernel then indexes it with `@index(Local,…)`.
- **Frontend intrinsic:** NEW `shared_array(::Type{T}, dims)::SharedArrayMarker` — but it must return a *tracked handle* (like `TensorViewInfo`), not a scalar marker, because subsequent `getindex`/`setindex!` must lower to loads/stores against it.
- **Walker clause:** NEW `:shared_array` (allocate + track), plus the `memoryrefnew/get/set!` SPMD clauses (`lower.jl:3531-3660`) must learn to target a shared buffer in addition to func-arg memrefs (currently they only resolve `getfield(arr,:ref)` rooted at an Argument, `lower.jl:3533-3539`).
- **CPU SPMD:** `memref.alloca` of `vector`/`memref<Dims×T>` **outside** the lane vectorisation, shared by all W lanes in the block. But the whole CPU model is "one block = one `scf.parallel` iteration processing W contiguous elements as a SIMD vector" — a `@localmem` array indexed by `@index(Local,Linear)` (lane id) maps to vector lanes, while a barrier+reread implies cross-lane shuffle. Modelling this faithfully requires either (a) treating localmem as an `scf.alloca`'d `memref` and lane access as scatter/gather into it (correct, slower), or (b) recognising staged "scatter→barrier→gather" reduction idioms. Hard.
- **GPU:** `memref.alloca` in `#gpu.address_space<workgroup>` (NEW — no alloca/workgroup-addrspace anywhere in lower.jl, confirmed by grep). The gpu→nvvm pipeline lowers workgroup allocas to `.shared`.
- **Difficulty:** hard (both targets need a memory model the SPMD path lacks; needs `Dims`/`Id` `Val` unwrapping + a tracked-handle infrastructure).

### 8. `@private` / `Scratchpad(ctx, T, Val{Dims})` — NEW, moderate-to-hard
- **KA fn:** `Scratchpad(ctx, ::Type{T}, ::Val{Dims})` (macros.jl:258-265 → `KernelAbstractions.jl:802`). Per-workitem `MArray`.
- **Frontend intrinsic:** NEW `local_array(::Type{T}, dims)` (private scratch handle), tracked like #7.
- **Walker clause:** NEW `:local_array` + the same `memoryrefnew/get/set!` retargeting as #7.
- **CPU SPMD:** per-lane private storage = a `vector<W × …>` per scratch slot, or a `memref.alloca<W×Dims×T>` (private dim added, matching KA's CPU "additional implicit dimensions" note, macros.jl:250-251). Moderate (no cross-lane semantics, unlike localmem, so simpler than #7).
- **GPU:** `memref.alloca` in default (private/local) address space, one per thread. Moderate.
- **Difficulty:** moderate (no barrier interaction) but shares the tracked-handle + `Val{Dims}` unwrapping work with #7.

### 9. `@uniform expr` — trivial / mostly free
- **KA fn:** none — `@uniform` is a pure passthrough macro (`macros.jl:283-285`); on the GPU body it just evaluates `expr` in workitem scope.
- **Frontend intrinsic:** none.
- **Walker clause:** none specific — a `@uniform` value is whatever expression produced it. In SPMD mode it stays a *scalar* (uniform) MLIR Value; `_spmd_harmonise` (`lower.jl:1794-1820`) broadcasts it on demand when combined with a varying lane vector.
- **CPU/GPU:** falls out of existing scalar-op lowering.
- **Difficulty:** trivial (likely already works for arithmetic uniforms; only fails if the uniform expression itself uses an unsupported op).

### 10. KA atomics (`@atomic`, `Atomix`) — NEW, moderate
- **KA fn:** KA exposes atomics via `Atomix.@atomic` / `Atomix.@atomicswap` / `Atomix.@atomicreplace` on array references (KA depends on `Atomix`, per Manifest deps line 224). These lower (through Atomix → the backend) to `Atomix.modify!`/`Core.Intrinsics.atomic_*` on a `memoryref`, **not** to cuTile's `Intrinsics.atomic_*`.
- **Frontend intrinsic:** none needed if the walker matches the lowered form directly.
- **Walker clause:** NEW SPMD-mode clauses for `Atomix.modify!`/`Base.modifyfield!`/`memoryref`-rooted atomic intrinsics. The existing `emit_atomic_rmw!`/`emit_atomic_cas!`/`emit_atomic_rmw_generic!` machinery (`lower.jl:3139-3422`) already emits `memref.atomic_rmw` / `memref.generic_atomic_rmw` — but it is keyed on cuTile `Intrinsics.offset` + `Intrinsics.atomic_*` (`lower.jl:1722-1746`, consuming `OffsetInfo`). For KA, the index path is `memoryrefnew` → `OffsetInfo` (the SPMD path at `lower.jl:3531-3564` already builds an `OffsetInfo` from `memoryrefnew`!), so a new clause can route an Atomix-atomic on that `OffsetInfo` straight into `_emit_one_atomic_rmw!` (`lower.jl:3139`). The reusable lowering already exists; only the *front* dispatch is missing.
- **CPU SPMD:** `memref.atomic_rmw <kind>` (scalar per lane, or unroll lanes like `lower.jl:3183-3215`).
- **GPU:** same `memref.atomic_rmw`; the gpu→nvvm pipeline lowers to PTX `atom.*`. (Per-thread scalar = single atomic.)
- **Difficulty:** moderate (wire Atomix's lowered call shape to the existing RMW emitters; main cost is identifying the exact `Atomix.modify!`/intrinsic call the KA frontend produces after inference and matching it).

### 11. `@print` / `__print` — NEW, low priority
- **KA fn:** `__print(args...)` (KernelAbstractions.jl:809, `@generated`).
- **Frontend/walker:** would map to `vector.print` (CPU) or a `gpu.printf`/`llvm` vprintf (GPU). 
- **Difficulty:** moderate, low value for kernels — defer.

---

## Summary of net-new work to land the matrix

**New `Frontend.Intrinsics` markers** (add to `frontend.jl:43-59`): `local_index([dim])`, `group_index([dim])`, plus reuse of `block_dim(dim)`/`barrier()`; tracked-handle markers `shared_array(T,dims)` and `local_array(T,dims)`. Extend `global_index`/`local_index`/`group_index` to optionally take a `dim::Int32`.

**New overlays** (move all of `ext/KernelAbstractionsExt.jl:61-93` from `cuTile.cuTileMethodTable` to `Frontend.METHOD_TABLE`): `__index_Global_Linear`→`global_index()`, `__index_Local_Linear`→`local_index()`, `__index_Group_Linear`→`group_index()`, `__synchronize`→`barrier()`, `SharedMemory`→`shared_array`, `Scratchpad`→`local_array`, the Cartesian/NTuple variants → dim-parameterised forms. `__validindex` stays `true` for exact-multiple launches (tighter masking is a separate item, noted at `KernelAbstractionsExt.jl:76-80`).

**New walker clauses** in `walk_call!` (`lower.jl:1438+`): `:local_index`, `:group_index`, `:block_dim`, `:barrier`, `:shared_array`, `:local_array`, and an Atomix-atomic dispatch — all gated on `lc.spmd`. GPU needs `_gpu.barrier()` and y/z-dim `gpu.thread_id`/`block_id`/`block_dim` emission (extend `lower.jl:1134-1146`), plus workgroup-addrspace `memref.alloca`.

**New lowering infra:** multi-grid-dim SPMD KA entrypoint (generalise `lower_to_mlir_ka`, `lower.jl:818-974`, beyond `n_grid_dims=1`); a tracked shared/private buffer handle (analogous to `TensorViewInfo`/`PartitionInfo` at `lower.jl:41-66`) that `emit_spmd_memoryrefnew!`/`get!`/`set!` (`lower.jl:3531-3660`) can target.

---

## Ordered list of KA kernels to target next

Easiest → hardest, with the intrinsics each unlocks. (Reference kernels are the standard KA examples: `examples/memcopy.jl`, `examples/matmul.jl`, `examples/naive_transpose.jl`, reduction examples in the KA test suite.)

1. **`vadd` — DONE** (`examples/kernel_abstractions_poc.jl`). Needs only `@index(Global,Linear)` + `__validindex`→true. Baseline.

2. **`memcopy` / SAXPY / elementwise unary (`y[i] = f(x[i])`)** — same intrinsic set as vadd (`@index(Global,Linear)`). Validates the migrated `Frontend.structured` path + existing `memoryref` gather/transfer clauses on more arg shapes (3+ arrays, a scalar `@uniform` coefficient → exercises #9). **Trivial after migration.**

3. **`mul_by_local_id` / index-arithmetic kernel** using `@index(Local,Linear)` and `@index(Group,Linear)` (e.g. `y[i] = @index(Local,Linear) + @index(Group,Linear)*@groupsize()`). Forces items #2, #3, #4 (local/group index + block_dim). No memory model needed; pure index lowering. **Moderate — first new-intrinsic milestone.**

4. **2-D `memcopy` / elementwise on a matrix** using `@index(Global, Cartesian)` (or 2-D `Linear`). Forces multi-dim item #5 (multi-grid SPMD + y/z GPU dims + CartesianIndex modelling). **Hard — the multi-dim grid milestone; unblocks everything 2-D.**

5. **`naive_transpose`** (`B[i,j] = A[j,i]`, no localmem). Needs multi-dim `@index` (#5) and 2-D non-contiguous indexing → exercises the **gather/scatter** fallback in `emit_spmd_memoryrefget!/set!` (`lower.jl:3611-3658`) since the transposed access is not the contiguous-lane pattern (`_is_contiguous_lane_index_from_offset`, `lower.jl:3669`). **Hard, but only multi-dim + existing scatter — no localmem/barrier.**

6. **Histogram / atomic scatter** (`@atomic bins[bucket] += 1`). Needs KA atomics (#10) on top of `@index(Global,Linear)`. Reuses the existing `memref.atomic_rmw` emitters; only new front dispatch. **Moderate; good ROI because the RMW lowering already exists.**

7. **Tiled / `@localmem` matmul** (`examples/matmul.jl` shared-memory variant) — needs `@localmem` (#7), `@synchronize` (#6, now meaningful), `@index(Local/Group,…)` (#3,#4), multi-dim (#5). This is the convergence of nearly every hard item. **Hardest; do last.**

8. **Block reduction** (`@localmem` + tree reduction across `@synchronize`) — same dependency set as #7 plus correct cross-lane semantics through localmem. The CPU SPMD "lanes = SIMD vector" model makes a within-block tree reduction either a `vector.reduction`/`vector.multi_reduction` (if recognised as an idiom) or scatter-to-alloca + barrier-noop + gather. **Hardest / research-level on the CPU path.**

**Recommended near-term cut:** items 2–3 (migrate overlays to `Frontend.METHOD_TABLE` + land `local_index`/`group_index`/`block_dim` + `gpu.barrier` no-op) are low-risk and immediately broaden coverage to all 1-D index kernels and atomics (#6). The multi-dim grid generalisation (#4/#5) is the single highest-leverage hard task and should be scoped as its own milestone before localmem (#7/#8).

---

**Key file:line references**
- KA overlays to migrate: `/home/gbaraldi/gpufun/cuTileCPU/ext/KernelAbstractionsExt.jl:61-93`
- Frontend intrinsics + overlay table: `/home/gbaraldi/gpufun/cuTileCPU/src/frontend.jl:43-59`, `:70`, `:114-122`
- Sentinel walker clause (Global,Linear): `/home/gbaraldi/gpufun/cuTileCPU/src/lower.jl:1771-1774`
- KA SPMD entrypoint (single grid dim): `/home/gbaraldi/gpufun/cuTileCPU/src/lower.jl:818-974`
- GPU SIMT entrypoint (x-dim only, no barrier/alloca): `/home/gbaraldi/gpufun/cuTileCPU/src/lower.jl:1024-1156`
- SPMD memoryref load/store/gather/scatter: `/home/gbaraldi/gpufun/cuTileCPU/src/lower.jl:3531-3689`
- Atomic RMW/CAS emitters (reusable for KA atomics): `/home/gbaraldi/gpufun/cuTileCPU/src/lower.jl:3139-3422`
- KA `@index`/`@localmem`/`@private`/`@uniform`/`@synchronize` macros: `~/.julia/packages/KernelAbstractions/X5fk1/src/KernelAbstractions.jl:237-323`, `macros.jl:422-458`
- GPU transform (validindex guard, body split): `~/.julia/packages/KernelAbstractions/X5fk1/src/macros.jl:83-120`
- Canonical GPU intrinsic semantics: `~/.julia/packages/CUDA/1WEyo/CUDACore/src/CUDAKernels.jl:161-211`