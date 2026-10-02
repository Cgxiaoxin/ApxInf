# PI0.5 RTX 5090 — FP32 path B (executor)

Date: 2026-10-02  
GPU: NVIDIA GeForce RTX 5090 (sm_120)  
Checkpoint: `pi05_droid_pytorch`

## Scope

Explicit `model_variant=fp32` eager reference executor (cuBLAS F32 GEMM + FP32
ops). `Auto` never selects FP32. PreferGraph/tactics for FP32 are not required
for this smoke; default PreferGraph may fall back to eager.

## Gold ladder (host float32 OpenPI gold, seed 7)

Artifact: `devlocal/pi05-rtx5090/logs/fp32/gold_ladder_pathb/`

| Variant | max_abs | cosine | rel_l2 | soft pass |
|---|---:|---:|---:|:---:|
| bf16 | 0.012084 | 0.999957 | 0.010197 | ✓ |
| **fp32** | **0.007993** | **0.999976** | **0.007173** | ✓ |

Soft gates: max_abs≤0.05, cosine≥0.99, rel_l2≤0.1.

Note: OpenPI `pi05_droid` **compute** dtype remains bfloat16; float32 is the
stored gold / comparison width. FP32 APXInf is an APXInf-side executor, not a
clone of OpenPI’s compute dtype.

## Latency (5090, gold seed7, warmup=3, samples=10)

Artifact: `devlocal/pi05-rtx5090/logs/fp32/latency_compare/`

| Path | P50 |
|---|---:|
| APXInf **fp32** (this smoke) | **91.29 ms** |
| APXInf bf16 PreferGraph (prior) | ~27.33 ms |
| OpenPI JAX (prior) | ~61.24 ms |

FP32 is the correctness / reference executor, not the deploy latency path yet.

## Next

- PreferGraph capture for FP32 (if workspace allows)
- Optional cuBLASLt / tactics for F32 GEMMs
- Keep BF16 as the 5090 ship path until FP32 is competitive where needed
