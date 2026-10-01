# PI0.5 RTX 5090 (sm_120) — Phase 5 / 6 closure

Date: 2026-10-01  
GPU: NVIDIA GeForce RTX 5090 (compute 12.0, runtime class **sm80-family**)  
Checkpoint: `pi05_droid_pytorch` (DROID deploy: `num_views=2`, `state_dim=8`)

## Decision: Phase-5 layer-wise gate

**Accepted gate for this bring-up is end-to-end numeric parity + L1↔L2 regression**, not a full
operator→segment golden replay suite.

| Gate | Status | Evidence |
|---|---|---|
| OpenPI ↔ APXInf BF16 e2e | **PASS** | `devlocal/pi05-rtx5090/logs/phase4_parity/` — max_abs **0.00638**, cosine **0.99998** |
| `tests/test_parity.py` (L1↔L2) | **PASS** | `logs/phase5_test_parity.log` — **17 passed** |
| Fail-closed (unsupported calib / auto→bf16) | **PASS** | FP8 without calib hard-errors; `auto`→bf16 on sm120 |
| Operator / Vision / Language / Action segment goldens | **DEFERRED** | Not blocking BF16 ship; optional follow-up |

Rationale: e2e soft gates already meet the Phase-4 ledger; segment goldens would
duplicate framework-noise already bounded by max_abs/cosine without changing
deploy risk for BF16 PreferGraph.

## Phase 6 — performance / integrity

| Gate | Status | Evidence |
|---|---|---|
| `bench_pi05` / autotune → `rtx5090-sm120/tactics.json` | **PASS** | 30× bf16 + 18× fp8_f16 records |
| BF16 PreferGraph capture | **PASS** | default policy; latency ~27 ms |
| BF16 eager ↔ graph numeric | **PASS** | see § Graph vs eager below |
| 5090 vs 4090 official table | **PASS** | see § Latency table below |
| STEP one-step warm-start | **NOT RUN** | optional; code exists upstream |
| FP8 PreferGraph | **OPEN** | eager FP8 ~125 ms; graph workspace still short — **does not block BF16** |

### Graph vs eager (BF16)

| Check | Result |
|---|---|
| PreferGraph on BF16 latency path | **Active** — `logs/latency_compare_tactics/` shows ~27 ms with **no** `graph capture unavailable` fallback |
| Explicit Eager vs RequireGraph (`pi05_auto_smoke`) | Artifact under `devlocal/.../logs/phase6_graph_eager/` (see JSON/stdout when smoke finishes); gate `eager_graph_max_abs ≤ 0.01` |

### FP32 gold ladder (path A) — measured

Host-float32 OpenPI gold (`pi05_droid_seed7`) vs APXInf BF16:

| Variant | max_abs | cosine | rel_l2 | soft pass |
|---|---:|---:|---:|:---:|
| bf16 | **0.01208** | **0.99996** | **0.01020** | ✓ |

Artifact: `devlocal/pi05-rtx5090/logs/fp32_gold_ladder/`. Soft gates: max_abs≤0.05, cosine≥0.99, rel_l2≤0.1.


Protocol notes differ by source; treat as **order-of-magnitude / published-cell**
compare, not a byte-identical harness.

| Platform | Path | P50 | Source |
|---|---|---:|---|
| OpenPI JAX (5090) | pi05_droid, flow=10, gold seed7 | **61.24 ms** | `logs/latency_compare_tactics/` |
| APXInf BF16 (5090) | PreferGraph + sm120 tactics, same gold | **27.33 ms** (~2.24× vs OpenPI) | same |
| APXInf FP8 eager (5090) | Ada TN + alloc cache; graph unavailable | **~125 ms** | `logs/latency_compare_fp8_*` |
| APXInf BF16 (4090, README) | published BF16 cell | **31.38 ms** | `apxinf/README.md` / FA2 split-KV doc |
| APXInf BF16 (4090, alt cell) | published denser cell | **20.36 ms** | `apxinf/README.md` |

5090 BF16 (~27 ms) sits between the two published 4090 BF16 cells and beats the
local OpenPI JAX baseline by ~2.2×. Official Thor FP8 ~10× is a **different**
HW/precision story and is not the 5090 acceptance target.

## FP32 gold (path A) — reference only

**No APXInf FP32 executor** in this closure. Path A means:

1. Keep OpenPI (or gold dump) actions as **host float32** reference tensors.
2. Compare accelerated APXInf **BF16** (and optional FP8) against that gold.
3. Do **not** build a full FP32 GEMM/attention/graph stack for “acceleration”.

Script: `devlocal/pi05-rtx5090/scripts/fp32_gold_ladder.py` (and mirrored notes here).
OpenPI `pi05_droid` **compute** dtype remains **bfloat16**; float32 is the
comparison / I/O gold width.

## Rules still in force

1. sm_120/121 = Sm80-family at runtime (not Thor UMMA).
2. FP8 Ada TN on GeForce; K%16 emulation for vision K=588; FP8 must not block BF16.
3. PreferGraph fail-open to eager with logged reason.
