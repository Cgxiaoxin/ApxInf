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

| Variant | max_abs | cosine | rel_l2 | soft pass | Artifact |
|---|---:|---:|---:|:---:|---|
| bf16 PreferGraph | 0.012084 | 0.999957 | 0.010197 | ✓ | `gold_ladder_cublas_mqa_fixed` |
| **fp32** reference MQA (pre-accel) | **0.007993** | **0.999976** | **0.007173** | ✓ | `gold_ladder_ref_mqa` |
| **fp32** cuBLAS F32 MQA (post-fix) | **0.007994** | **0.999976** | **0.007173** | ✓ | `gold_ladder_cublas_mqa_fixed` |
| fp32 cuBLAS MQA before softmax fix | 0.462445 | 0.917954 | 0.399502 | ✗ | `gold_ladder_exact_mqa` |
| fp32 + TF32 (opt-in, not default) | 0.175326 | 0.990674 | 0.138685 | ✗ | `gold_ladder_tf32` |

Soft gates: max_abs≤0.05, cosine≥0.99, rel_l2≤0.1.

Takeaway: after the softmax race fix, cuBLAS F32 MQA matches reference-MQA FP32
gold to ~1e-6 on max_abs; both beat BF16 on max_abs (~0.008 vs ~0.012) while
staying well inside soft gates.

Note: OpenPI `pi05_droid` **compute** dtype remains bfloat16; float32 is the
stored gold / comparison width. FP32 APXInf is an APXInf-side executor, not a
clone of OpenPI’s compute dtype.

## Latency (5090, gold seed7, warmup=3, samples=10)

Artifact: `devlocal/pi05-rtx5090/logs/fp32/latency_compare/`

| Path | P50 |
|---|---:|
| APXInf **fp32** (reference MQA) | **91.29 ms** |
| APXInf **fp32** (cuBLAS MQA, exact FP32 compute) | **66.83 ms** |
| APXInf bf16 PreferGraph (prior) | ~27.33 ms |
| OpenPI JAX (prior) | ~61.24 ms |

Root cause of the earlier gold failure was an **in-place softmax race** in the
cuBLAS F32 MQA path (`softmax_f32_kernel` multi-block with `input==output`).
Fixed with a one-warp-per-row `softmax_scalar_f32_kernel`. Operator parity and
e2e gold both pass; TF32 remains off by default.

## Next

- PreferGraph capture for FP32 (workspace may need a larger arena)
- Optional TF32 as an explicit opt-in (must re-pass gold before defaulting)
- Keep BF16 as the 5090 ship path until FP32 is competitive where needed
- See `doc/pi05-sft-fp32-deploy.md` for SFT checkpoint usage
