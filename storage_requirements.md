# Storage Requirements: BitNet Video Generation

## Model Downloads (Phase 0)

| Model | Size |
|:---|---:|
| Wan2.2-TI2V-5B (DiT + T5 + VAE) | ~20 GB |
| Wan2.2-T2V-A14B (MoE teacher, 27B total) | ~55 GB |
| Falcon Perception 0.6B + OCR | ~2.5 GB |
| CLIP image encoder | ~0.5 GB |
| **Subtotal** | **~78 GB** |

## Training Data (Phase 2)

| Component | 500K clips (min) | 2M clips (full) |
|:---|---:|---:|
| Raw video (720P, ~5s avg, compressed) | ~1.5 TB | ~6 TB |
| VAE-encoded latents (Wan2.2-VAE, 64x compression) | ~250 GB | ~1 TB |
| Falcon annotation JSONs (~20 KB/clip) | ~10 GB | ~40 GB |
| Captions | negligible | negligible |
| **Subtotal** | **~1.8 TB** | **~7 TB** |

> **Note:** Raw video is the bulk. Once VAE pre-encoding is complete, raw video can be deleted if storage is tight — the model trains on latents only.

## Training Checkpoints (Phase 3)

| Component | 1.3B PoC | 5B Full |
|:---|---:|---:|
| Model checkpoint (BF16 master weights) | ~2.6 GB | ~10 GB |
| Optimizer state (Adam, 2× model) | ~5.2 GB | ~20 GB |
| Per checkpoint total | ~8 GB | ~30 GB |
| Keep 5 checkpoints | ~40 GB | ~150 GB |
| **Subtotal** | **~40 GB** | **~150 GB** |

## Container Images & Deployment

| Component | Size |
|:---|---:|
| bitnet-base image | ~3 GB |
| wan-reference image | ~8 GB |
| falcon-labeler image | ~4 GB |
| **Subtotal** | **~15 GB** |

## Final Trained Model (Inference)

| Component | Size |
|:---|---:|
| BitNet-Video 5B at 1.58-bit (GGUF) | ~1.2 GB |
| Wan2.2-VAE (float32) | ~0.5 GB |
| T5 encoder (frozen) | ~5 GB |
| **Per inference pod** | **~7 GB** |

## Total Summary

| Scenario | Storage Required |
|:---|---:|
| **1.3B PoC, 500K clips** (recommended start) | **~2 TB** |
| **5B full, 500K clips** | **~2.1 TB** |
| **5B full, 2M clips** | **~7.3 TB** |
| **5B full, 2M clips + keep raw video** | **~7.3 TB** |

## Recommended K8s PVC Layout

| PVC | Size | Purpose |
|:---|---:|:---|
| `model-weights` | 200 GB | All pre-trained models |
| `training-data` | 2–8 TB | Latents + raw video + annotations |
| `checkpoints` | 200 GB | Rolling checkpoint window |
| `chroma-db` | 10 GB | Video-RAG vector store |
| **Total shared storage** | **2.5–8.5 TB** | |

> **Practical minimum:** ~2.5 TB with Ceph or NFS. Starting with the 1.3B PoC on 500K clips and deleting raw video after VAE encoding keeps total storage under 2 TB.
