# Hello CUDA — Vector Addition, the GPU "Hello World"

A newbie-friendly companion to the chapter. The whole concept fits in one
picture: **[hello-cuda.png](hello-cuda.png)** (regenerate with
`python hello_cuda.py`; layout is verified by `python check_layout.py`).

## The 60-second version

- The **CPU (host)** cannot do arithmetic on GPU memory. It prepares data,
  copies it over, and **launches a kernel**: a function marked `__global__`
  that runs *on the GPU (device)*.
- The launch `vector_add<<<num_blocks, block_size>>>(...)` starts
  **1,000,192 lightweight threads**, all executing the *same* code at once.
- Each thread's entire job: work out **which element is mine?**
  (`idx = blockIdx.x * blockDim.x + threadIdx.x`), check it's in bounds,
  and do **one addition**.
- The GPU's memory is separate from the CPU's, so every byte crosses PCIe
  explicitly — and for this tiny problem, **the copies cost ~130× more than
  the kernel**.
- From Python you can do all of this with
  `torch.utils.cpp_extension.load_inline` — but you almost never should:
  libraries (cuBLAS, FlashAttention) win almost always.

## Vocabulary (the six words that unlock every CUDA tutorial)

| Term | Meaning | Analogy |
|---|---|---|
| **kernel** | A function that runs on the GPU, marked `__global__` | a recipe pinned to the kitchen wall |
| **host / device** | The CPU side / the GPU side | two offices connected by a (slow) courier |
| **thread** | One execution of the kernel; holds a `threadIdx` | one worker |
| **block** | A group of threads (max 1024) that shares shared memory | one team of workers |
| **grid** | All blocks launched by one `<<<...>>>` call | the whole shift |
| **warp** | The 32 threads the hardware actually executes in lockstep | the smallest team the boss will schedule |

Rule of thumb: choose block sizes that are **multiples of 32** (256 is the
common default), and let the grid have as many blocks as needed.

## 1. The kernel, line by line

```cuda
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

- `__global__` — "runs on the GPU, callable from the CPU". This is the whole
  magic of a kernel; otherwise it's ordinary C.
- `int idx = blockIdx.x * blockDim.x + threadIdx.x;` — "which element is
  mine?". Without this line all 1,000,192 threads would write to `c[0]`.
- `if (idx < n)` — the bounds guard (panel C of the picture). Cheap
  insurance; skipping it corrupts memory.
- `c[idx] = a[idx] + b[idx];` — the actual work: one load + one load + one
  add + one store per thread.

## 2. Thread indexing — worked examples

With `blockDim.x = 256` (threads per block):

| Thread | Computation | Global index |
|---|---|---|
| thread 0 of block 0 | 0 × 256 + 0 | 0 |
| thread 255 of block 0 | 0 × 256 + 255 | 255 |
| thread 0 of block 1 | 1 × 256 + 0 | 256 |
| thread 42 of block 17 | 17 × 256 + 42 | 4,394 |

`threadIdx.x` **restarts at 0 inside every block** — that is exactly why the
block index has to be multiplied by the block size before adding it.

## 3. Why `if (idx < n)` — the tail block

`n = 1,000,000` is not a multiple of 256, so we round the block count **up**:

```
num_blocks = (n + block_size - 1) / block_size   // integer division
           = (1,000,000 + 255) / 256 = 3907      // plain / would floor to 3906
threads launched = 3907 × 256 = 1,000,192        // 192 threads too many
```

The last block covers indices 999,936 … 1,000,191. Only the first **64** of
its threads find `idx < n` and write; the other **192** hit the guard and
return without touching memory. The guard is one integer compare — the
cheapest line in the kernel and the only one standing between you and
out-of-bounds writes.

## 4. The grid-stride loop

Instead of one thread per element, each thread walks the array with a step
equal to the **total number of threads**:

```cuda
int idx    = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;     // all threads in the grid
for (int i = idx; i < n; i += stride)
    c[i] = a[i] + b[i];
```

For `n = 20` and 8 threads (stride 8): thread 0 takes 0, 8, 16; thread 1
takes 1, 9, 17; …; threads 4–7 get one element each.

Why bother? The launch no longer needs one thread per element — **any**
grid size is correct, fewer blocks mean less scheduling overhead, and for
large arrays this is often *faster*. It's the same trick as splitting a
`for`-loop across CPU workers, except an H100 offers 270,336 resident
threads.

## 5. Memory: two separate worlds

The GPU cannot see host RAM (and vice versa), so the full ceremony is:

```cuda
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, n * sizeof(float));              // 1. allocate on the GPU
cudaMalloc(&d_b, n * sizeof(float));
cudaMalloc(&d_c, n * sizeof(float));
cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);   // 2. up
cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
vector_add<<<num_blocks, block_size>>>(d_a, d_b, d_c, n);          // 3. compute
cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);   // 4. down
```

For `n = 1,000,000` floats (4 MB per array), on an H100-class GPU:

| Step | Bytes | Where | Time |
|---|---|---|---|
| copy a, b up (H2D) | 8 MB | PCIe ≈ 25 GB/s | ~320 µs |
| kernel | 12 MB touched | HBM 3.35 TB/s | ~3.6 µs |
| copy c down (D2H) | 4 MB | PCIe ≈ 25 GB/s | ~160 µs |

**~480 µs of copies vs ~3.6 µs of compute.** This is why inference engines
keep every tensor resident on the GPU and never bounce through the host.

It's also why the chapter calls this kernel **memory-bound**: each element
moves 12 bytes (2 loads + 1 store of `float`) to do 1 add, so arithmetic
intensity = 1/12 ≈ 0.08 FLOP/byte, while the H100's ridge point is ~295.
The cores idle waiting for bytes — real optimizations raise that ratio
(fuse ops, reuse data), they don't shave FLOPs.

## 6. Compile and run

```
nvcc -o vector_add vector_add.cu
./vector_add
```

A complete, error-checked, timing-instrumented version is in
[vector_add.cu](vector_add.cu) — the chapter's snippets omit the host
boilerplate; that file has all of it.

## 7. Calling your kernel from Python

`torch.utils.cpp_extension.load_inline` JIT-compiles a CUDA string on first
import (cached under `~/.cache/torch_extensions`), wraps it with pybind11,
and hands you a normal Python function:

```python
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name='custom_add',
    cpp_sources=cpp_source,        # C++ wrapper: torch::Tensor in/out
    cuda_sources=cuda_source,      # the __global__ kernel
    functions=['add_cuda'],
)
c = module.add_cuda(a, b)          # your kernel, from Python
```

A complete runnable version (with correctness check against `a + b`) is in
[vector_add_torch.py](vector_add_torch.py). For production code, package the
extension with `setup(..., ext_modules=[CUDAExtension(...)], cmdclass={'build_ext': BuildExtension})`.

## 8. When should you write a custom kernel? Almost never.

| Your op | Use |
|---|---|
| Linear layers | cuBLAS, via `torch.nn.Linear` |
| Attention | FlashAttention / FlashInfer |
| Normalization | PyTorch defaults |
| Fusions no library covers | **your custom kernel — the one real niche** |

Write custom CUDA only when: **(1)** you need to fuse operations that aren't
fused by default, **(2)** you have a novel algorithm no library covers,
**(3)** you profiled and found a specific kernel is the bottleneck, **(4)**
the performance gain is worth the maintenance cost. Bugs are hard to find,
tuning is time-consuming — default to libraries, and drop down only with
evidence.

## Gotchas checklist (things that bite every beginner once)

- Block size must be ≤ 1024 and a multiple of 32 (the warp size).
- Forget `if (idx < n)` → works on your test sizes, corrupts memory later.
- `cudaMemcpy`'s direction argument is *destination, source* — H2D and D2H
  are mirror images; mixing them up gives garbage or a fault.
- Dereferencing a device pointer (`d_a[i]`) in host code is an error — the
  CPU cannot see GPU memory.
- Check the return of every CUDA call (see the `CHECK` macro in
  `vector_add.cu`); errors surface late and far from their cause.
- The first `load_inline` call compiles (30–60 s) — that's normal, later
  runs load the cache.
- Timings: use `torch.cuda.synchronize()` / CUDA events; kernel launches
  are asynchronous.

## Files in this folder

| File | What it is |
|---|---|
| [hello-cuda.png](hello-cuda.png) | the six-panel visualization |
| [hello_cuda.py](hello_cuda.py) | generates it; config numbers at the top |
| [check_layout.py](check_layout.py) | text-overlap/clipping gate for the figure |
| [vector_add.cu](vector_add.cu) | complete runnable CUDA C++ version |
| [vector_add_torch.py](vector_add_torch.py) | complete runnable `load_inline` version |
