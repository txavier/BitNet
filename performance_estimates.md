# Performance Estimates: 10-Minute Video Generation (720P@24fps)

## Assumptions

- **Target output**: 10 minutes of video at 720P resolution, 24fps
- **Total frames**: 14,400
- **Generation batches**: ~176 clips (81 frames / ~3.4 sec each)
- **Denoising steps**: 50 per batch
- **VAE decode**: Wan2.2-VAE (float32) after each batch
- **Wan 2.2**: Wan2.2-TI2V-5B (5B dense, bfloat16) — latest available version (Wan 2.7 does not exist)
- **BitNet-Video**: Proposed 1.58-bit ternary model distilled from Wan 2.2

## Generation Time Comparison

| System | Model | Per-Clip Time | 10-Min Video Total | Notes |
|:---|:---|---:|---:|:---|
| **CPU-only, 32GB / 15 threads** | Wan 2.2 FP (bfloat16) | ~167 min | **~20.4 days** | Extremely slow; model barely fits RAM with offloading |
| **CPU-only, 32GB / 15 threads** | BitNet-Video (1.58-bit) | ~42 min | **~5.1 days** | ~4x faster than FP via add/sub kernels; model fits easily in RAM |
| **RTX 5070 Ti + CPU** | Wan 2.2 FP (bfloat16) | ~12 min | **~35 hours** | 16GB VRAM needs `--offload_model --t5_cpu`; ~1.3x slower than 4090 |
| **RTX 5070 Ti + CPU** | BitNet-Video (1.58-bit) | ~8 min | **~23 hours** | GPU accelerates VAE decode + teacher-free; BitNet DiT still benefits from GPU tensor cores |
| **K8s cluster (no GPU): 8 nodes, 52 cores, 80GB** | BitNet-Video (1.58-bit) | ~15 min | **~1.8 days** | ~2.7x faster than single-node 15t; limited by inter-node latency on denoising steps |
| **K8s cluster to match RTX 5070 Ti** *(see spec below)* | BitNet-Video (1.58-bit) | ~12 min (throughput-matched) | **~35 hours** | Runs 4 pods in parallel, each processing different clips |

## K8s Cluster Spec to Match RTX 5070 Ti

To match the RTX 5070 Ti's ~35-hour total wall time for a 10-minute video:

| Resource | Requirement |
|:---|:---|
| **Nodes** | 4 (minimum) — 8 (recommended for redundancy) |
| **Total cores** | 64 |
| **Cores per node** | 16 (on 4 nodes) or 8 (on 8 nodes) |
| **Total threads** | 128 (with SMT/hyperthreading) |
| **Total RAM** | 96 GB (24 GB per worker pod) |
| **RAM per node** | 24 GB (4 nodes) or 12 GB (8 nodes) |
| **Network** | 10 Gbps inter-node minimum |
| **Storage** | 200 GB shared PV (model weights + output) |
| **Strategy** | 4 parallel pods, each generating different 3.4-sec clips simultaneously |

**How it works:** Instead of trying to make a single clip faster (which requires low-latency shared memory that K8s can't provide), run 4 BitNet-Video pods concurrently, each generating a different clip from the 176-clip sequence. Wall time = (176 ÷ 4) × 42 min ≈ 31 hours, matching the GPU's ~35 hours.

## Why This Works for BitNet but Not Wan FP

| Factor | Wan 2.2 FP on CPU | BitNet-Video on CPU |
|:---|:---|:---|
| Model memory per pod | ~20 GB (bfloat16) | ~7 GB (1.58-bit + VAE + T5) |
| Weight math | FP multiply (expensive) | Add/subtract (cheap) |
| Memory bandwidth | Bottlenecked (10 GB model) | 5–10x less data moved |
| Pods at 24 GB/node | 1 pod (tight) | 3 pods possible |
| Parallel clip throughput | Impractical (RAM-starved) | 4 pods easily |

The 1.58-bit model's small footprint (~1.2 GB for the DiT) is what makes CPU K8s viable — you can run many pods in parallel where a full-precision model would be memory-bound to a single pod per node.
