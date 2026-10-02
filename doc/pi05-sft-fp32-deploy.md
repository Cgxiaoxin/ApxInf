# Deploy SFT π0.5 checkpoints with APXInf FP32 / BF16

Target: RTX 5090 (`sm_120`). Use when you want APXInf to accelerate **your**
finetuned OpenPI π0.5 weights (same layout as
`/mnt/sdb/cgq/projects/openpi/weight/21000`).

## 0. What APXInf can load

APXInf PI0.5 expects an **OpenPI PyTorch export**, not the JAX Orbax tree:

```
<ckpt_dir>/
  model.safetensors          # required
  metadata.pt                # openpi pytorch export (preferred)
  assets/<asset_id>/norm_stats.json
```

Your current dir:

```
/mnt/sdb/cgq/projects/openpi/weight/21000/
  params/                    # JAX / Orbax  ← not directly loadable
  assets/tianji_pi05_16d_direct/norm_stats.json
```

Convert first (OpenPI env, **not** `apxinf-5090`):

```bash
source /mnt/sdb/cgq/projects/env/activate_cgq.sh openpi-ref
cd /mnt/sdb/cgq/projects/openpi
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /mnt/sdb/cgq/projects/openpi/weight/21000 \
  --output_path /mnt/sdb/cgq/projects/APXinf-robo/devlocal/ckpts/tianji_pi05_16d_pytorch
# copy / symlink assets so norm_stats is discoverable
mkdir -p /mnt/sdb/cgq/projects/APXinf-robo/devlocal/ckpts/tianji_pi05_16d_pytorch/assets
cp -a /mnt/sdb/cgq/projects/openpi/weight/21000/assets/. \
  /mnt/sdb/cgq/projects/APXinf-robo/devlocal/ckpts/tianji_pi05_16d_pytorch/assets/
```

Confirm the convert script’s config name matches how this run was trained
(π0.5 + your action dim). `tianji_pi05_16d_direct` implies **action_dim=16**,
not DROID’s 8.

## 1. Smoke latency / numeric (APXInf env)

```bash
source /mnt/sdb/cgq/projects/env/activate_cgq.sh apxinf-5090
export CUDA_VISIBLE_DEVICES=0

python - <<'PY'
from apxinf import Pi05Policy
import numpy as np

CKPT = "/mnt/sdb/cgq/projects/APXinf-robo/devlocal/ckpts/tianji_pi05_16d_pytorch"
# FP32 reference executor (explicit). Deploy latency path today is still BF16.
policy = Pi05Policy.from_pretrained(
    CKPT,
    device="cuda:0",
    model_variant="fp32",   # or "bf16" for ~27ms PreferGraph path on 5090
    discrete_state=True,
    state_key="state",
    image_keys=("base_0_rgb", "left_wrist_0_rgb"),  # adjust to your wire keys
    num_views=2,
    action_dim=16,          # match SFT
    state_dim=16,           # match training proprio width if discrete_state
    num_flow_steps=10,
    asset_id="tianji_pi05_16d_direct",
)
# Build a dummy obs matching your robot wire protocol, then:
# out = policy.infer(obs)
# print(out["actions"].shape, policy.metadata.get("model_variant"))
policy.close()
print("loaded ok", policy.metadata if False else CKPT)
PY
```

Gold-ladder style timing (after you have a gold dump for this SFT):

```bash
python /mnt/sdb/cgq/projects/APXinf-robo/devlocal/pi05-rtx5090/scripts/compare_openpi_apxinf_latency.py \
  --engine apxinf --model-variant fp32 --gpu 0 \
  --apxinf-checkpoint <pytorch_ckpt_dir> \
  --out-dir /tmp/fp32_sft_latency
```

**5090 reference numbers (pi05_droid, not your SFT):**

| Variant | P50 infer |
|---|---:|
| BF16 PreferGraph | ~27 ms |
| FP32 (eager / TF32 path) | was ~91 ms before TF32+cuBLAS MQA; rebench after rebuild |
| OpenPI JAX | ~61 ms |

Use **BF16** for deploy speed; use **FP32** when you need the APXInf FP32
executor as a stronger numeric baseline against your SFT.

## 2. LIBERO (separate env — do not mix with `apxinf-5090`)

LIBERO / MuJoCo pull a different NumPy stack. Keep a dedicated env (suggested
name `libero-eval`) and only call into APXInf via:

- **websocket**: run `apxinf-robo serve` in `apxinf-5090`, eval in `libero-eval`
- or install both carefully in one env (harder; avoid for bring-up)

Smoke (once libero env + assets exist):

```bash
# terminal A — model server (apxinf-5090)
source /mnt/sdb/cgq/projects/env/activate_cgq.sh apxinf-5090
apxinf-robo serve --robot franka_libero \
  --model-dir <pytorch_ckpt_dir> --precision bf16   # or extend for fp32

# terminal B — LIBERO harness (libero-eval)
python apxinf/scripts/eval_libero.py \
  --backend websocket --precision bf16 \
  --suite libero_10 --tasks 0 --trials-per-task 1 \
  --results-jsonl /tmp/libero_r.jsonl --summary-json /tmp/libero_s.json
```

Note: stock `eval_libero.py` precision choices are `bf16|fp8|int8`; FP32
needs a small CLI mapping (`fp32` → `model_variant=fp32`) before in-process
FP32 LIBERO.

## 3. Checklist for a new SFT drop

1. JAX Orbax → PyTorch safetensors (+ copy `assets/`)
2. Confirm `action_dim` / `state_dim` / `num_views` / `asset_id`
3. BF16 smoke latency + optional gold compare vs OpenPI on same obs
4. FP32 smoke if you need the FP32 executor baseline
5. LIBERO smoke in isolated env (`--tasks 0 --trials-per-task 1`) then full
