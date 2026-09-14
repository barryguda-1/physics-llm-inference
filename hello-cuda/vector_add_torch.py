# vector_add_torch.py — the same kernel, callable from Python via
# torch.utils.cpp_extension.load_inline.
#
#   requires: a CUDA GPU, PyTorch with a matching CUDA toolkit (nvcc on PATH)
#   run:      python vector_add_torch.py
#
# What happens under the hood: on first run, PyTorch writes the sources to
# ~/.cache/torch_extensions/<python-cuda-hash>/, compiles them with nvcc into
# a C++ extension, loads it, and returns a normal Python module. Subsequent
# runs load the cache (near-instant). The C++ wrapper (cpp_sources) handles
# torch::Tensor objects; the CUDA source holds the __global__ kernel — the
# exact one from the chapter.

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;   // grid-stride: any grid size works
    for (int i = idx; i < n; i += stride)
        c[i] = a[i] + b[i];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    int blocks = (n + 255) / 256;
    add_kernel<<<blocks, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}
"""

cpp_source = "torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);"

module = load_inline(
    name="custom_add",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["add_cuda"],
)

a = torch.randn(1_000_000, device="cuda")
b = torch.randn(1_000_000, device="cuda")
c = module.add_cuda(a, b)          # first call may also trigger the JIT build

torch.cuda.synchronize()           # launches are async — sync before timing
assert torch.allclose(c, a + b), "kernel result does not match torch's a + b"
print("correct: module.add_cuda(a, b) == a + b")

# Timing note from the chapter: the kernel is memory-bound (12 bytes moved
# per add -> AI = 1/12 FLOP/byte). Compare against torch's own add:
torch.cuda.synchronize()
start, end = torch.cuda.Event(True), torch.cuda.Event(True)

start.record()
for _ in range(100):
    c = module.add_cuda(a, b)
end.record()
torch.cuda.synchronize()
print(f"custom kernel : {start.elapsed_time(end) / 100:.4f} ms")

start.record()
for _ in range(100):
    c = a + b
end.record()
torch.cuda.synchronize()
print(f"torch a + b   : {start.elapsed_time(end) / 100:.4f} ms")
# Expect the same order of magnitude: both are pinned by HBM bandwidth,
# which is the chapter's point — the win is never "my add is faster",
# it's fusing the ops around it.
