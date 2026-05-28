# Compile the hand-written gpu-dialect kernel to PTX **via CUDA.jl's
# shared LLVM NVPTX backend**, then launch it on the H100.
#
# Architectural note (responding to "can we from nvvm do whatever cuda is
# doing instead of going via ptx?"): PTX is unavoidable as the format the
# CUDA driver accepts. What we *can* skip is MLIR's *own* NVPTX backend
# (`gpu-module-to-binary{format=isa}`) and instead translate at the LLVM
# IR boundary, handing the LLVM module to **the same NVPTX target machine
# CUDA.jl uses for its own kernels** (via `LLVM.jl` directly, or via
# `GPUCompiler.llvm_machine` if going through a full job). Same backend,
# same opts, same LLVM version — so the PTX we emit here and the PTX
# CUDA.jl emits for an equivalent kernel come from one place.
#
# Pipeline used here:
#   MLIR `gpu.module` (high-level)
#     ↓  gpu-kernel-outlining
#     ↓  gpu.module(convert-gpu-to-nvvm)
#     ↓  convert-{scf,cf,arith,memref,nvvm}-to-llvm + reconcile-casts
#     ↓  gpu-module-to-binary{format=llvm}    -- emits LLVM IR (not PTX!)
#                                                inline as a gpu.binary attr
#     [extract LLVM IR bytes via MLIR C API]
#     ↓  LLVM.parse_ir(...)                    -- get an LLVM.Module
#     ↓  LLVM.NVPTXTargetMachine{sm_90}        -- the same shape CUDA.jl
#                                                 sets up internally
#     ↓  LLVM.emit(tm, mod, AssemblyFile)      -- PTX text
#     ↓  CUDA.CuModule(ptx)                    -- driver-loaded
#     ↓  cudacall(...)                         -- launched

using cuTileCPU
using MLIR
const IR = MLIR.IR
using LLVM
using CUDA

const HERE = @__DIR__

# ---------------------------------------------------------------------------
# 1. MLIR pipeline (stops at LLVM IR — no MLIR-side NVPTX backend run)
# ---------------------------------------------------------------------------

const GPU_PASSES = String[
    "nvvm-attach-target{chip=sm_90 features=+ptx80}",
    "gpu-kernel-outlining",
    "gpu.module(convert-gpu-to-nvvm)",
    "convert-scf-to-cf",
    "convert-cf-to-llvm",
    "convert-arith-to-llvm",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "convert-nvvm-to-llvm",
    "reconcile-unrealized-casts",
    "gpu-module-to-binary{format=llvm}",
]
_pipeline_str(passes) = "builtin.module(" * join(passes, ",") * ")"

ctx = cuTileCPU.fresh_context()
IR.activate(ctx)

mlir_text = read(joinpath(HERE, "01_handwritten_vadd_gpu.mlir"), String)
mod = parse(IR.Module, mlir_text)

pm = IR.PassManager()
parse(IR.OpPassManager(pm), _pipeline_str(GPU_PASSES))
status = MLIR.API.mlirPassManagerRunOnOp(pm, IR.Operation(mod))
status.value == 0 && error("GPU pipeline failed")

# ---------------------------------------------------------------------------
# 2. Walk the lowered module to find gpu.binary, pull out the LLVM IR bytes
# ---------------------------------------------------------------------------

# `format=llvm` emits LLVM **bitcode** (binary), not textual IR. Pull
# bytes (not a String) — LLVM.jl's `parse(Module, ::Vector{UInt8})`
# accepts bitcode directly.
function _stringref_to_bytes(sr::MLIR.API.MlirStringRef)
    return unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(sr.data), sr.length; own=false) |> copy
end

function extract_object_bytes(mod::IR.Module)
    body = IR.body(mod)
    for op in body
        IR.name(op) == "gpu.binary" || continue
        objects_attr = IR.getattr(op, "objects")
        @assert MLIR.IR.isarray(objects_attr)
        nobj = MLIR.API.mlirArrayAttrGetNumElements(objects_attr)
        nobj > 0 || error("gpu.binary has no objects")
        first_obj = MLIR.IR.Attribute(MLIR.API.mlirArrayAttrGetElement(objects_attr, 0))
        @assert MLIR.API.mlirAttributeIsAGPUObjectAttr(first_obj)
        sr = MLIR.API.mlirGPUObjectAttrGetObject(first_obj)
        return _stringref_to_bytes(sr)
    end
    error("no gpu.binary op found")
end

bc_bytes = extract_object_bytes(mod)
println("=== Extracted LLVM bitcode (magic: $(repr(bc_bytes[1:4])), $(length(bc_bytes)) bytes) ===")
@assert bc_bytes[1:4] == [0x42, 0x43, 0xc0, 0xde]  "not LLVM bitcode (magic BC C0 DE missing)"

# ---------------------------------------------------------------------------
# 3. Parse the LLVM IR + emit PTX through LLVM.jl's NVPTX TargetMachine
# ---------------------------------------------------------------------------
#
# This is the same backend CUDA.jl uses — same `libLLVM`, same NVPTX
# codegen. We set up a TargetMachine matching the H100 (sm_90).

println("=== Parsing LLVM bitcode + emitting PTX through LLVM.jl ===")

llvm_ctx = LLVM.Context()
llvm_mod = LLVM.context!(llvm_ctx) do
    parse(LLVM.Module, bc_bytes)
end
println("LLVM module parsed: ", llvm_mod)
for f in LLVM.functions(llvm_mod)
    println("  function: ", LLVM.name(f))
end

# Triple + target machine matching what CUDA.jl uses on an H100 driver.
triple = "nvptx64-nvidia-cuda"
LLVM.triple!(llvm_mod, triple)
target = LLVM.Target(; triple)
tm = LLVM.TargetMachine(target, triple, "sm_90", "+ptx80")
LLVM.asm_verbosity!(tm, true)

ptx = String(LLVM.emit(tm, llvm_mod, LLVM.API.LLVMAssemblyFile))
println("\n=== Emitted PTX (first 400 chars) ===")
println(first(ptx, 400))
println("...")
println("=== Total PTX size: $(length(ptx)) bytes ===\n")

@assert occursin(".entry vadd", ptx) || occursin(".entry _", ptx)  "PTX missing entry"

# Find the entry name (MLIR may have mangled it).
entry_match = match(r"\.entry\s+(\w+)\s*\(", ptx)
entry_name = entry_match === nothing ? "vadd" : entry_match.captures[1]
println("=== Entry symbol: $entry_name ===\n")

# ---------------------------------------------------------------------------
# 4. Load PTX through CUDA.jl + launch on the H100
# ---------------------------------------------------------------------------

CUDA.functional() || error("no CUDA device available")
println("=== Loading PTX as a CuModule ===")
cumod = CuModule(ptx)
kernel = CuFunction(cumod, String(entry_name))
println("✓ CuFunction handle: ", kernel.handle, "\n")

const N = 1024 * 1024
println("=== Allocating + initialising device buffers (N = $N) ===")
A_host = rand(Float32, N)
B_host = rand(Float32, N)
A = CuArray(A_host)
B = CuArray(B_host)
C = CUDA.zeros(Float32, N)

# Memref<?xf32> descriptor after finalize-memref-to-llvm is a flat struct
# of 5 × u64: (allocated_ptr, aligned_ptr, offset, size_0, stride_0).
# Our PTX entry has 16 params = 3 memrefs × 5 + 1 `n: index` arg.
function memref_descriptor(arr::CuArray{T,1}) where {T}
    p = UInt64(UInt(pointer(arr)))
    return (p, p, UInt64(0), UInt64(length(arr)), UInt64(1))
end

a_desc = memref_descriptor(A)
b_desc = memref_descriptor(B)
c_desc = memref_descriptor(C)
n_arg  = UInt64(N)

# `cudacall(f, Tuple{T1,...,Tn}, a1, ..., an)` — types as a Tuple{} type,
# args spliced individually. We have 16 Culonglong args (3 memrefs × 5 + 1 n).
const SIG = Tuple{ntuple(_->Culonglong, 16)...}
args_list = (
    a_desc[1], a_desc[2], a_desc[3], a_desc[4], a_desc[5],
    b_desc[1], b_desc[2], b_desc[3], b_desc[4], b_desc[5],
    c_desc[1], c_desc[2], c_desc[3], c_desc[4], c_desc[5],
    n_arg,
)

const BLOCK = 256
const GRID  = cld(N, BLOCK)

println("=== Launching: grid = $GRID, block = $BLOCK ===")
cudacall(kernel, SIG, args_list...; threads=BLOCK, blocks=GRID)
CUDA.synchronize()

println("\n=== Verifying ===")
C_host = Array(C)
expected = A_host .+ B_host
err = maximum(abs.(C_host .- expected))
println("max abs diff = $err")
@assert err == 0  "vadd mismatch"
println("✓ all $N elements match — MLIR-emitted PTX runs identically to ground truth")

# ---------------------------------------------------------------------------
# 5. Quick timing
# ---------------------------------------------------------------------------

println("\n=== Quick timing (min of 10 launches, post-warmup) ===")
function bench(kernel, sig, args, block, grid)
    for _ in 1:3
        cudacall(kernel, sig, args...; threads=block, blocks=grid)
    end
    CUDA.synchronize()
    best = typemax(Float64)
    for _ in 1:10
        t = CUDA.@elapsed begin
            cudacall(kernel, sig, args...; threads=block, blocks=grid)
        end
        best = min(best, t)
    end
    return best
end
t_mlir = bench(kernel, SIG, args_list, BLOCK, GRID)
gb = 3 * N * sizeof(Float32) / 1e9
println("  MLIR→LLVM→PTX→driver:  $(round(t_mlir*1e6, digits=1)) μs  =  $(round(gb/t_mlir, digits=1)) GB/s")

# ---------------------------------------------------------------------------
# 6. Reference: same vadd via CUDA.jl's broadcast (uses GPUCompiler)
# ---------------------------------------------------------------------------

println("\n=== Reference: CUDA.jl broadcast `C .= A .+ B` (GPUCompiler path) ===")
function bench_broadcast(A, B, C)
    for _ in 1:3; C .= A .+ B; end
    CUDA.synchronize()
    best = typemax(Float64)
    for _ in 1:10
        t = CUDA.@elapsed (C .= A .+ B)
        best = min(best, t)
    end
    return best
end
t_bcast = bench_broadcast(A, B, C)
println("  CUDA.jl broadcast:    $(round(t_bcast*1e6, digits=1)) μs  =  $(round(gb/t_bcast, digits=1)) GB/s")

println("\n=== Ratio: MLIR / CUDA.jl = $(round(t_mlir/t_bcast, digits=2))x ===")
