# Profiling with Nsight Systems — See Where the Time Actually Goes

A companion to the chapter. The whole concept fits in one picture:
**[nsys-profiling.png](nsys-profiling.png)** (regenerate with
`python nsys_profiling.py`; layout is verified by `python check_layout.py`).

## The 60-second version

- **Profile before optimizing.** Where time actually goes is rarely where you
  think it goes — that's the entire point of the tool.
- `nsys profile -o report python inference.py` shadows your run and records
  every CUDA API call, kernel, memcpy, and NVTX range, timestamped.
- Read it two ways: **`nsys stats report.nsys-rep`** (a text table that names
  the culprit — panel B) and the **GUI timeline** (four lanes that show you
  why — panel C).
- Three classic sick patterns: **gaps between kernels** (the GPU waits on
  launch overhead — panel D), **blocking `cudaMemcpy`** (everything stalls —
  panel E), and **independent kernels serialized on one stream** (panel E).
- Mark your regions with **NVTX ranges** so code and timeline correlate
  (panel F).

## Command reference

```bash
# capture (basic)
nsys profile -o profile_output python inference.py

# capture only between cudaProfilerStart/Stop calls, repeatedly
nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat \
     -o /mnt/profiles/trace --force-overwrite=true -- <your command>

# see kernels INSIDE CUDA graphs (graph is the default; node shows each kernel,
# with runtime overhead)
nsys profile --cuda-graph-trace=node ...

# skip warmup / cap length
nsys profile --delay 60 --duration 30 ...

# summaries, and SQL access to every table
nsys stats profile_output.nsys-rep
nsys export --type sqlite profile_output.nsys-rep
```

Flags verified against the Nsight Systems CLI docs, Sep 2026 (Nsight Systems
2026.5.1 current). On Blackwell+ `--trace=cuda` uses hardware tracing by
default; `--trace=cuda-sw` forces software tracing for MPS/MIG/vGPU.

## Reading the output — pointers into the picture

| You see | It means | Panel |
|---|---|---|
| One kernel family eats 45%+ of GPU time | attack the top of the table, not what's easy | B |
| Instance counts equal across ops (1024 = 1024) | per-layer ops — the model's fingerprint | B |
| White gaps between short kernels | launch-bound: CPU dispatch, fix = CUDA graphs | D |
| One long copy, lanes empty around it | sync memcpy stalls everything — async it, or don't copy | E |
| Independent kernels end-to-end on one stream | overlap them on a second stream | E |
| Kernels vanished after adopting CUDA graphs | default `graph` trace hides them — `--cuda-graph-trace=node` | D |
| SMs underused / DRAM nowhere near ceiling | occupancy & throughput questions | F |

---

# Profiling a predictor pod deployed by KServe

The chapter profiles a script you launch yourself. A deployed predictor is a
**long-running server inside Kubernetes** — you can't attach nsys to a running
process (CUDA tracing is injected at process start). The recipes below were
verified against KServe master and vLLM source in Sep 2026.

## Step 0 — identify what you're running

```bash
kubectl get isvc
kubectl get isvc <name> -o yaml          # spec.predictor: model? containers?
kubectl get deploy -l serving.kserve.io/inferenceservice=<name> -o yaml

# the vLLM version decides the syntax later — check it:
kubectl exec deploy/<name>-predictor-xxxxx -c kserve-container -- \
  python -c "import vllm; print(vllm.__version__)"
```

KServe's `kserve-vllmserver` runtime runs
`python -m vllm.entrypoints.openai.api_server --port=8080 --served-model-name=<name> --model=/mnt/models`
on container `kserve-container`, unprivileged, port 8080.

## Route 2 — vLLM predictor: built-in endpoints (no image change)

vLLM exposes `POST /start_profile` and `POST /stop_profile` on the API server
(root paths, no `/v1` prefix). **How they're enabled depends on the version** —
verified from source at each tag (Sep 2026):

| vLLM | Endpoint gate | What to set |
|---|---|---|
| ≤ 0.11 | env `VLLM_TORCH_PROFILER_DIR` mounts the routes | env var |
| 0.12 | env `VLLM_TORCH_PROFILER_DIR` **or** `VLLM_TORCH_CUDA_PROFILE` | env var |
| 0.13 – 0.15 | `--profiler-config` CLI; env vars still honored as fallbacks | either |
| **0.16 – 0.29.0** | `--profiler-config` **only** — env vars removed | CLI JSON |

**Confirmed at v0.29.0 specifically:** the routes live in
`vllm/entrypoints/serve/profile/api_router.py`, are attached to the server via
`register_vllm_serve_api_routers`, and mount only when
`profiler_config.profiler` is set to one of `torch` / `cuda` / `proton`
(vLLM 0.29.0's `ProfilerKind`). `--profiler-config` is the CLI arg
(`vllm/engine/arg_utils.py`); `VLLM_TORCH_PROFILER_DIR` no longer exists
anywhere in the tree. KServe master's `huggingfaceserver` image builds vLLM
from source with `VLLM_VERSION=0.24.0` as the default pin — also in the
config-only era, so on any recent KServe + vLLM deployment the CLI path below
is *the* path; the env recipe only applies to legacy clusters on ≤ 0.15.

**Config-only versions (0.16+, incl. 0.29.0)** — pass via `predictor.model.args`:

```yaml
      args:
        - --profiler-config
        - '{"profiler":"torch","torch_profiler_dir":"/mnt/traces","delay_iterations":20,"max_iterations":10}'
```

`ProfilerConfig` fields at 0.29.0 worth knowing: `delay_iterations` /
`max_iterations` (skip warmup, cap length — the config-era equivalents of
nsys `--delay/--duration`), `warmup_iterations` / `active_iterations` (torch
profiler schedule), `capture_torch_profiler` (profile the CUDA-graph capture
itself), and `torch_profiler_dir` accepts `s3://` / `gs://` URIs — skip the
volume and ship traces straight to a bucket.

**Legacy versions (≤ 0.11)** — routes mount only if env
`VLLM_TORCH_PROFILER_DIR` is set. Patch a *debug copy* of the InferenceService:

```yaml
spec:
  predictor:
    minReplicas: 1                     # keep it warm; avoid cold-start noise
    model:
      modelFormat: vLLM
      runtime: kserve-vllmserver
      # ...your existing fields (storageUri, resources, GPU limit)...
      env:
        - name: VLLM_TORCH_PROFILER_DIR
          value: /mnt/traces           # must exist and be writable
```

(Quick look, no volume needed: point it at `/dev/shm` — the runtime already
mounts a Memory-backed emptyDir there — or `/tmp`.)

Caveat: ISVC-provided `args` may **replace** the runtime's args rather than
append — after applying, verify the resolved result and repeat
`--port=8080 --served-model-name=<name> --model=/mnt/models` if they vanished:

```bash
kubectl get deploy -l serving.kserve.io/inferenceservice=<name> \
  -o jsonpath='{.items[0].spec.template.spec.containers[0].args}'
```

**Capture a window:**

```bash
POD=$(kubectl get pod -l serving.kserve.io/inferenceservice=<name> -o name | head -1)
kubectl port-forward $POD 8080:8080

curl -X POST http://localhost:8080/start_profile
#   ...send real traffic now (through the gateway for full-path timing,
#      or straight to :8080/v1/completions if you only care about the engine)
curl -X POST http://localhost:8080/stop_profile
```

**Collect and read:**

```bash
kubectl cp $POD:/mnt/traces/ ./traces -c kserve-container
```

You get timestamped `.json.gz` Chrome traces (one per TP/DP worker rank).
Drag one into <https://ui.perfetto.dev> — CPU ops, CUDA kernels, and gaps,
which is everything the sheet's panels diagnose. For aggregate stats:
`pip install hta` (Holistic Trace Analysis) gives the table-equivalent view.

**The kernel-level upgrade path:** the `"profiler":"cuda"` mode (confirmed in
v0.29.0's `vllm/profiler/wrapper.py`; env-triggered as
`VLLM_TORCH_CUDA_PROFILE` back on v0.12) calls
`torch.cuda.profiler.start()/stop()` behind those same HTTP endpoints — i.e.
remote-controlled `cudaProfilerStart/Stop`. Combine with Route 1's nsys
wrap + `--capture-range=cudaProfilerApi` and you can open an nsys capture
window on a live server over HTTP. vLLM even ships a helper for the analysis
side: `tools/profiler/nsys_profile_tools/gputrc2graph.py` (in the repo, incl.
v0.29.0) turns a `.nsys-rep` captured with `-t cuda` into kernel-level CSV/HTML
summaries of GPU vs non-GPU time.

## Route 1 — any predictor: wrap the entrypoint with nsys

**1. Image** — add the Nsight Systems CLI (NVIDIA apt repo or tarball):

```dockerfile
FROM your-predictor:latest
USER root
RUN apt-get update && apt-get install -y nsight-systems-cli && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/nvidia/nsight-systems-cli/bin:$PATH
COPY profile-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/profile-entrypoint.sh"]
```

```bash
#!/bin/bash
# profile-entrypoint.sh — unchanged unless PROFILE=nsys
if [ "$PROFILE" = "nsys" ]; then
  exec nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat \
       --cuda-graph-trace=node -t cuda,nvtx -o /mnt/profiles/trace \
       --force-overwrite=true -- "$@"
else
  exec "$@"
fi
```

**2. InferenceService** — KServe's predictor accepts standard PodSpec fields
(container `kserve-container`): set `minReplicas: 1`, mount a volume at
`/mnt/profiles`, set `PROFILE=nsys` — on a dedicated debug ISVC, not prod.

**3. Trigger from your handler** (guarded, so only chosen requests capture):

```python
if self.profiling:                          # env var or request header
    torch.cuda.cudart().cudaProfilerStart()
out = self.model(inputs)
if self.profiling:
    torch.cuda.cudart().cudaProfilerStop()
```

**4. Collect** — send one or two requests, then:

```bash
kubectl cp <ns>/<pod>:/mnt/profiles/trace.nsys-rep . -c kserve-container
```

and open it in the GUI / `nsys stats` exactly like the chapter. Keep windows
to a few requests — capture files grow fast.

## Route 0 — coarse signals first

nsys/torch-profiler traces only see the **predictor process**. Before
kernel-level work, check the cheap signals: DCGM exporter metrics for
per-pod GPU utilization, KServe's request metrics for queue/gateway latency
(the parts nsys will never show), and only reach for a profiler when those
say the GPU itself is the problem.

## Gotchas checklist

- Never patch prod: vLLM logs the profiler "should ONLY be used for local
  development" — run it on a debug InferenceService copy.
- Traces in emptyDir die with the pod — `kubectl cp` them out **before** any
  redeploy, and set `minReplicas: 1` so Knative doesn't scale to zero
  mid-session.
- ISVC `args` may replace (not append to) the runtime's args — always check
  the resolved Deployment after applying.
- CUPTI errors in the pod log usually mean seccomp is blocking it — lift
  securityContext on the debug pod only.
- Multi-GPU pods write one trace per rank — several files is expected.
- Profile with one pod per GPU; time-slicing neighbors' kernels won't appear
  in your trace.
- nsys plain CUDA/NVTX tracing works unprivileged; GPU metrics sampling and
  perf counters need `privileged: true` / `CAP_SYS_ADMIN` / `CAP_PERFMON`.
- Keep capture windows short (a few requests / iterations) — .nsys-rep files
  reach gigabytes quickly on long windows.

## Sources (checked Sep 2026)

- Nsight Systems CLI reference & product page —
  <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>,
  <https://developer.nvidia.com/nsight-systems> (2026.5.1 current)
- KServe vLLM ServingRuntime —
  `config/runtimes/kserve-vllmserver.yaml` in `kserve/kserve` (master)
- KServe InferenceService CRD — `predictor.model.{args,env,command}` and
  full PodSpec container fields confirmed in
  `config/crd/full/serving.kserve.io_inferenceservices.yaml`
- vLLM profiler endpoints — version matrix verified from source at tags
  v0.10.0, v0.11.0, v0.12.0, v0.13.0, v0.14.0, v0.15.1, v0.16.0, and
  **v0.29.0**: routes at `vllm/entrypoints/serve/profile/api_router.py`
  (attached via `register_vllm_serve_api_routers`), gated by
  `profiler_config.profiler ∈ {torch, cuda, proton}` from 0.13 on, env-var
  fallbacks (`VLLM_TORCH_PROFILER_DIR`, `VLLM_TORCH_CUDA_PROFILE`) removed
  at 0.16; `"cuda"` mode in `vllm/profiler/wrapper.py` calls
  `torch.cuda.profiler.start()/stop()`; `--profiler-config` registered in
  `vllm/engine/arg_utils.py`; nsys helper `tools/profiler/nsys_profile_tools/`
- KServe vLLM image — `python/huggingface_server.Dockerfile` (master) builds
  vLLM from source, default `VLLM_VERSION=0.24.0`
- PyTorch NVTX — <https://docs.pytorch.org/docs/2.14/cuda.html>:
  `torch.cuda.nvtx.range` / `range_push` / `range_pop` current, nothing
  deprecated

## Files in this folder

| File | What it is |
|---|---|
| [nsys-profiling.png](nsys-profiling.png) | the six-panel visualization |
| [nsys_profiling.py](nsys_profiling.py) | generates it; config numbers at the top |
| [check_layout.py](check_layout.py) | text-overlap/clipping gate for the figure |
