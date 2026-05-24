# ADR 026 — Run YOLO on the GPU (supersedes the CPU-torch part of ADR-024)

**Status:** Accepted
**Date:** 2026-05-24

## Context

ADR-024 deliberately shipped **CPU-only torch** to keep the image small and
because GPU headroom on the shared workstation was dictated by host apps. It
noted the detector "still pegs ~8-9 CPU cores" and named a `DETECT_EVERY_N`
frame-skip as the only further lever, left unadopted.

In practice the detector ran at **~1069% CPU (~11 of 24 cores)** for per-frame
YOLO across 2 RTSP streams (one `pipeline.run` thread per camera, CPU-only
torch), at ~16 fps. The operator asked why resources were so high.

## Decision

Run YOLO on the **RTX 3090**. The container already had GPU passthrough (the
compose `nvidia` reservation works — `nvidia-smi` sees the card); the only
blocker was the CPU torch build. So:

- **Dockerfile**: swap the torch wheel from the CPU index to **cu124** (matches
  host driver 595.79; the wheels bundle CUDA/cuDNN, so the `python:3.11-slim`
  base needs no system CUDA — only the host driver + nvidia-container-runtime,
  both already present). No `nvidia/cuda` base image required.
- **No detection code change** — ultralytics auto-selects CUDA when available.
- **`_setup_gpu()`** (pipeline.py): logs the active device and caps YOLO at
  `GPU_MEM_FRACTION` (default 0.15 ≈ 3.6 GB) via
  `set_per_process_memory_fraction`, so it can't starve the on-demand ollama
  VLM (~6 GB qwen2.5vl) or the host's LM Studio on the shared 24 GB GPU.

### Measured result

| Metric | CPU-only (ADR-024) | GPU (this ADR) |
|---|---|---|
| Detector CPU | ~1069% (~11 cores) | **~170% (~1.7 cores)** |
| FPS (per cam) | 16 | **21** |
| GPU VRAM added | — | **~450 MB** (yolov8n is tiny) |

FPS *rose* because CPU inference had been the throughput bottleneck; on GPU the
pipeline runs at the cameras' real rate, so the live bbox is also smoother. The
residual ~170% CPU is now HEVC decode + MOG2, not inference.

### Rejected first: CPU thread / OpenMP tuning

Before the GPU move, thread tuning was tried and **measured to be a dead end** —
there is no config that gives both low CPU and 16 fps on CPU-only torch:

| Config | CPU | FPS |
|---|---|---|
| Baseline (no cap, default spin) | ~1069% | 16 |
| `torch.set_num_threads(4)`, active spin | ~1400% | 16 |
| cap 4, `OMP_WAIT_POLICY=passive` | ~485% | 10 |
| cap 8, passive | ~520% | 10 |
| cap 8, `GOMP_SPINCOUNT=10000` | ~1130% | 14 |

The ~1069% baseline was mostly OpenMP **busy-spin** at sync barriers between the
many short per-frame inference regions (`/proc/loadavg` showed ~1829 threads).
Killing the spin (`passive`) more than halved CPU but the wakeup latency dropped
throughput to 10 fps regardless of thread count — it's latency, not
parallelism. All of this was reverted; GPU made it moot.

## Consequences

**Positive**
- CPU cut ~6× (~9 cores freed for the operator's desktop / other stacks).
- Higher, smoother frame rate (16 → 21 fps) as a side effect.
- The hi-res localizer model (ADR-025) also runs on GPU.

**Negative / watch out**
- **Reverses ADR-024's small-image stance**: the cu124 wheels grow the image
  ~3-4 GB. Accepted — the CPU saving outweighs image size here.
- **Shared-GPU VRAM is the live risk, not the detector.** At cutover only
  ~6.8 GB of 24 GB was free (LM Studio ~13 GB). YOLO adds only ~450 MB, but a
  VLM event loads ollama (~6 GB) onto the same card; if LM Studio grows,
  something can OOM. The `GPU_MEM_FRACTION` cap protects ollama *from YOLO*, not
  from LM Studio. This risk pre-existed (ollama was already GPU-resident).
- **GPU/driver coupling**: the image now depends on the host NVIDIA driver
  supporting cu124. A machine without a CUDA GPU needs the Dockerfile reverted
  to the CPU index (torch falls back to CPU automatically if the driver is
  present but `_setup_gpu` logs `Inference device: CPU`).

## Principle reinforced

**Measure the lever before committing to it.** The thread-tuning matrix above
falsified the intuitive "cap threads to stop oversubscription" fix in minutes
and pointed at the real bottleneck. The earlier ADR-024 leanness call was right
for its constraints; this one updates it now that freeing CPU mattered more than
image size — same method (measure on the real workload), different answer.
