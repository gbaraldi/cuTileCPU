# Shared setup for the ParallelTestRunner test files. Evaluated (via
# runtests.jl `init_code`) into each test file's sandbox module before the file
# runs. This is intentionally minimal —
# CUDA/LLVM/KernelAbstractions/Atomix are conditionally loaded inside the
# guarded GPU/KA test files themselves.
using MLIRKernels

# The `code_*` reflectors print IR to an `io` (CUDA.jl-style) and return nothing;
# capture the text as a String for pattern checks. `_ir(code_gpu, k, args...; kw)`.
_ir(reflector, args...; kwargs...) = sprint(io -> reflector(io, args...; kwargs...))
