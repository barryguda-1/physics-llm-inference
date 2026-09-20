import sys
import time
import torch

PEAK_TFLOPS = 65.0
PEAK_TBS = 0.32

def benchmark_gemv(m, k, iters=1000, warmup=50):
    """Time `iters` GEMV launches; return (achieved TFLOPS, achieved GB/s)."""
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    x = torch.randn(k, dtype=torch.float16, device="cuda")

    # Warmup: the first calls pay one-time costs (allocator growth, kernel
    # selection). Never let them inside the timed region.

    for _ in range(warmup):
        y = torch.matmul(a, x)
    torch.cuda.synchronize()      # drain the queue before starting the clock

    start = time.perf_counter()

    for _ in range(iters):
        y = torch.matmul(a, x)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * m * k * iters
    gbs = 8 * m * k * iters / elapsed / 1e9
    tflops = flops / 1e12 / elapsed
    return tflops, gbs

def benchmark_gemm(m, k, n, iters=100, warmup=50):
    """Bonus: batch b vectors through the same matrix -> GEMM.
    Same weight bytes, b times the FLOPs: intensity should scale with b."""
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")

    for _ in range(warmup):
        y = torch.matmul(a, b.T)
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        y = torch.matmul(a, b.T)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * m * n * k * iters
    gbs = 8 * m * n * k * iters / elapsed / 1e9

    tflops = flops / 1e12 / elapsed
    return tflops, gbs

def sanity_check():
    m, k = 4, 5
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(k, dtype=torch.float16, device="cuda")
    y = torch.matmul(a, b)
    assert y.shape == (m,)
    manual = (a[2].float() * b.float()).sum()
    assert torch.allclose(y[2].float(), manual, atol=1e-2)
    print("Sanity check: y is (m,) and y[2] == dot(A[2], x) OK\n")

def main():
    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible -- run this on the GPU box.")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"peaks : {PEAK_TFLOPS:.0f} TFLOPS fp16 | {PEAK_TBS} TB/s"
          "   (edit constants if not on an H100 SXM)\n")
    sanity_check()
    sizes = [(512, 512), (1024, 1024), (2048, 2048),
             (4096, 4096), (8192, 8192), (11008, 4096)]

    # Filter arguments to verify if we were passed a custom shape manually, ignoring Jupyter flags
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if len(args) == 2:
        try:
            sizes = [(int(args[0]), int(args[1]))]
        except ValueError:
            pass

    hdr = f"{'shape':>14} {'TFLOPS':>8} {'GB/s':>7} {'%BW':>6} {'%FLOPS':>8} {'FLOP/byte':>10}"
    print(hdr)
    print("-" * len(hdr))
    for m, k in sizes:
        tflops, gbps = benchmark_gemv(m, k)
        pct_bw    = gbps / (PEAK_TBS * 1000) * 100
        pct_flops = tflops / PEAK_TFLOPS * 100
        print(f"{f'{m}x{k}':>14} {tflops:8.2f} {gbps:7.0f} {pct_bw:5.0f}%"
              f"{pct_flops:7.2f}% {tflops / gbps:10.2f}")
    print("\nHow to read it:")
    print("  %BW is high (50-80%) while %FLOPS is ~0 -> memory-bound, as the theory says")
    print(" FLOP/byte ~ 1 at EVERY size    -> bigger matrices don't help")
    print(f"the H100 needs {PEAK_TFLOPS / PEAK_TBS:.0f} FLOP/byte to saturate compute; GEMV delivers ~1")

    print("\nbonus: same 4096x4096 matrix, batched (GEMM) -- batching is the fix:")
    hdr2 = f"{'batch':>14} {'TFLOPS':>8} {'GB/s':>7} {'FLOP/byte':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    for b in (1, 8, 32):
        tflops, gbps = benchmark_gemm(4096, 4096, b)
        print(f"{b:>14} {tflops:8.2f} {gbps:7.0f} {tflops / gbps:10.2f}")
    print("\nsame bytes, b times the FLOPs -> intensity and TFLOPS scale with batch.")


if __name__ == "__main__":
    main()