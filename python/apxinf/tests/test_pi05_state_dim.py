"""state_dim= trims discrete-state normalization like OpenPI's tokenize-before-pad."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apxinf.checkpoints.descriptor import QUANTILE, RESOLVED, TransformSpec
from apxinf.policies.impls import pi05 as pi05_mod
from apxinf.policies.impls.pi05 import Pi05Policy
from apxinf.processors.transforms import OBSERVATION, PROMPT, TOKEN_IDS


class _FakeRunner:
    action_horizon = 10
    action_dim = 32
    num_views = 2
    image_size = 224
    max_token_len = 200

    def reset_sampling(self, seed=None):
        return None

    def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
        return np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)


class _ConstTokenizer:
    discrete_state = True
    max_token_len = 200

    def __call__(self, prompt, state=None):
        # Encode state length into tokens so the test can assert trim width.
        width = 0 if state is None else int(np.asarray(state).shape[-1])
        return np.asarray([2, width, 108], dtype=np.uint32)


def test_normalizer_from_transform_trims_to_state_dim():
    spec = TransformSpec(
        feature_key="state",
        mode=QUANTILE,
        width=32,
        eps=1e-6,
        values={
            "q01": tuple([-1.0] * 32),
            "q99": tuple([1.0] * 32),
        },
        source="test",
        status=RESOLVED,
    )
    normalizer = pi05_mod._normalizer_from_transform(spec, dims=8, dtype="float64")
    assert normalizer is not None
    assert normalizer.width == 8
    out = normalizer(np.zeros(8, dtype=np.float64))
    assert out.shape == (8,)


def test_normalizer_from_transform_rejects_oversized_state_dim():
    spec = TransformSpec(
        feature_key="state",
        mode=QUANTILE,
        width=8,
        eps=1e-6,
        values={"q01": tuple([-1.0] * 8), "q99": tuple([1.0] * 8)},
        source="test",
        status=RESOLVED,
    )
    with pytest.raises(ValueError, match="state_dim=16"):
        pi05_mod._normalizer_from_transform(spec, dims=16)


def test_from_pretrained_state_dim_requires_discrete_state(tmp_path: Path):
    with pytest.raises(ValueError, match="discrete_state=True"):
        Pi05Policy.from_pretrained(
            tmp_path,
            model_runner=_FakeRunner(),
            state_dim=8,
            discrete_state=False,
            image_keys=("observation/image", "observation/wrist_image"),
        )


def test_from_pretrained_state_dim_trims_prompt_state(tmp_path: Path, monkeypatch):
    stats = {
        "norm_stats": {
            "state": {"q01": [-1.0] * 32, "q99": [1.0] * 32},
            "actions": {"q01": [-1.0] * 32, "q99": [1.0] * 32},
        }
    }
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    # Avoid needing a real SentencePiece model on disk.
    monkeypatch.setattr(
        pi05_mod,
        "PromptTokenizer",
        lambda *args, **kwargs: _ConstTokenizer(),
    )
    monkeypatch.setattr(pi05_mod, "_resolve_tokenizer", lambda *args, **kwargs: tmp_path / "tok")

    policy = Pi05Policy.from_pretrained(
        tmp_path,
        model_runner=_FakeRunner(),
        discrete_state=True,
        state_key="state",
        state_dim=8,
        action_dim=8,
        image_keys=("observation/image", "observation/wrist_image"),
        norm_dtype="float64",
    )
    assert policy.metadata["state_dim"] == 8
    assert policy.input_pipeline["tokenize"].state_normalizer.width == 8

    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
        "prompt": "pick up the fork",
    }
    data = policy.input_pipeline({OBSERVATION: obs, PROMPT: obs["prompt"]})
    # _ConstTokenizer encodes state width into token[1]
    assert int(data[TOKEN_IDS][1]) == 8
