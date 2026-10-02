//! Native-FP32 π0.5 transformer-layer computation.
//!
//! This mirrors the BF16 schedule operator for operator but every tensor,
//! GEMM and fused-operator stand-in is FP32. Nothing is ever cast to BF16 or
//! FP16; unfused F32 operators replace the BF16 dual-GeGLU and fused-bias paths.

use crate::pi05::backend::{kernels, Context};
use apxinf_core::{Result, Tensor};
use kernels::{fp32, gemm};

use crate::pi05::{
    Fp32DeviceActionLayer, Fp32DeviceLanguageLayer, Fp32DeviceVisionBlock, Fp32LinearWeights,
    GemmaVariantConfig,
};

pub struct Fp32LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct Fp32ActionLayerOutput {
    pub hidden: Tensor,
    pub next_normalized: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_fp32(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Fp32DeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp32LanguageLayerOutput> {
    let normalized = fp32::rms_f32(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = gemm::f32(ctx, &normalized, &weights.qkv.weight)?;
    let qkv = fp32::split_qkv_apply_f32(
        ctx,
        &qkv,
        weights.qkv.bias.as_ref(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
    )?;
    let tokens = input.shape().dims()[0];
    if !compute_tail {
        return Ok(Fp32LanguageLayerOutput {
            hidden: input.clone(),
            key: qkv.k.reshape(vec![tokens, config.head_dim])?,
            value: qkv.v.reshape(vec![tokens, config.head_dim])?,
        });
    }
    let attention = fp32::mqa_f32(ctx, &qkv.q, &qkv.k, &qkv.v, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = gemm::f32(ctx, &attention, &weights.output.weight)?;
    let fused = fp32::bias_residual_rms_f32(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let gate_up = gemm::f32(ctx, &fused.normalized, &weights.gate_up.weight)?;
    let activated = fp32::geglu_f32(ctx, &gate_up)?;
    let projected = gemm::f32(ctx, &activated, &weights.down.weight)?;
    let hidden =
        fp32::bias_residual_f32(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)?;
    Ok(Fp32LanguageLayerOutput {
        hidden,
        key: qkv.k.reshape(vec![tokens, config.head_dim])?,
        value: qkv.v.reshape(vec![tokens, config.head_dim])?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer_fp32(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Fp32DeviceActionLayer,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_modulation: &Tensor,
    mlp_modulation: &Tensor,
    next_norm_modulation: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp32ActionLayerOutput> {
    let normalized = match attention_normalized {
        Some(value) => value.clone(),
        None => fp32::adaptive_rms_f32(ctx, input, attention_modulation, rms_eps)?,
    };
    let qkv = gemm::f32(ctx, &normalized, &weights.qkv.weight)?;
    let q = fp32::apply_q_write_kv_f32(
        ctx,
        &qkv,
        weights.qkv.bias.as_ref(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
        prefix_k,
        prefix_v,
        position_offset,
    )?;
    let tokens = input.shape().dims()[0];
    let attention = fp32::mqa_f32(ctx, &q, prefix_k, prefix_v, position_offset + tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = gemm::f32(ctx, &attention, &weights.output.weight)?;
    let fused = fp32::adaptive_gate_residual_rms_f32(
        ctx,
        &projected,
        input,
        attention_modulation,
        mlp_modulation,
        rms_eps,
    )?;
    let gate_up = gemm::f32(ctx, &fused.normalized, &weights.gate_up.weight)?;
    let activated = fp32::geglu_f32(ctx, &gate_up)?;
    let projected = gemm::f32(ctx, &activated, &weights.down.weight)?;
    let fused = fp32::adaptive_gate_residual_rms_f32(
        ctx,
        &projected,
        &fused.hidden,
        mlp_modulation,
        next_norm_modulation,
        rms_eps,
    )?;
    Ok(Fp32ActionLayerOutput {
        hidden: fused.hidden,
        next_normalized: fused.normalized,
    })
}

pub fn vision_patch_embed_fp32(
    ctx: &Context,
    weights: &Fp32LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    let projection = gemm::f32(ctx, patches, &weights.weight)?;
    fp32::add_position_f32(
        ctx,
        &projection,
        weights.bias.as_ref(),
        position_embedding,
        patches_per_view,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer_fp32(
    ctx: &Context,
    weights: &Fp32DeviceVisionBlock,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    let normalized = fp32::layer_f32(
        ctx,
        input,
        &weights.norm1.weight,
        &weights.norm1.bias,
        layer_norm_eps,
    )?;
    let qkv = gemm::f32(ctx, &normalized, &weights.qkv.weight)?;
    let qkv = fp32::split_qkv_bias_f32(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = fp32::mha_f32(ctx, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projection = gemm::f32(ctx, &attention, &weights.output.weight)?;
    let fused = fp32::bias_residual_layer_f32(
        ctx,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
    )?;
    let activation = gemm::f32(ctx, &fused.normalized, &weights.fc1.weight)?;
    let activation = fp32::bias_gelu_f32(ctx, &activation, weights.fc1.bias.as_ref())?;
    let projection = gemm::f32(ctx, &activation, &weights.fc2.weight)?;
    fp32::bias_residual_f32(ctx, &projection, weights.fc2.bias.as_ref(), &fused.hidden)
}

// Precision-specific backbone operations share this file with their layers.
pub(in crate::pi05::model) mod backbone {
    use super::*;
    use crate::pi05::backend::{kernels, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
    use crate::pi05::weights::*;
    use crate::pi05::Pi05Config;
    use apxinf_core::{DType, Error, Result, Tensor};
    use kernels::{fp32, gemm};
    use std::sync::Arc;

    pub struct Fp32PrefixKvCache {
        pub keys: Vec<Tensor>,
        pub values: Vec<Tensor>,
        pub tokens: usize,
    }

    pub struct Fp32StepModulation {
        attention: Vec<Tensor>,
        mlp: Vec<Tensor>,
        final_norm: Tensor,
    }

    pub struct Fp32Blocks {
        pub(in crate::pi05::model) backend: Arc<RuntimeBackend>,
        pub(in crate::pi05::model) config: Arc<Pi05Config>,
        pub(in crate::pi05::model) weights: Arc<Fp32Weights>,
    }

    fn require_f32(tensor: &Tensor, label: &str) -> Result<()> {
        if tensor.dtype() != DType::F32 {
            return Err(Error::Other(format!(
                "π0.5 FP32 {label} must be F32, got {}",
                tensor.dtype()
            )));
        }
        Ok(())
    }

    impl Fp32Blocks {
        pub fn new(
            backend: Arc<RuntimeBackend>,
            config: Arc<Pi05Config>,
            weights: Arc<Fp32Weights>,
        ) -> Result<Self> {
            config.validate()?;
            if weights.vision_layers.len() != config.vision_depth
                || weights.language_layers.len() != config.language.depth
                || weights.action_layers.len() != config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 FP32 device weight depth mismatch".into(),
                ));
            }
            Ok(Self {
                backend,
                config,
                weights,
            })
        }

        fn ctx(&self) -> &Context {
            self.backend.context()
        }

        pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
            require_f32(patches, "patches")?;
            let mut hidden = vision_patch_embed_fp32(
                self.ctx(),
                &self.weights.patch_embedding,
                &self.weights.position_embedding,
                patches,
                self.config.patches_per_view(),
            )?;
            for layer in &self.weights.vision_layers {
                hidden = vision_layer_fp32(
                    self.ctx(),
                    layer,
                    &hidden,
                    self.config.patches_per_view(),
                    self.config.vision_heads,
                    self.config.vision_head_dim,
                    self.config.layer_norm_eps,
                )?;
            }
            let hidden = fp32::layer_f32(
                self.ctx(),
                &hidden,
                &self.weights.vision_post_norm.weight,
                &self.weights.vision_post_norm.bias,
                self.config.layer_norm_eps,
            )?;
            let projected = gemm::f32(
                self.ctx(),
                &hidden,
                &self.weights.multimodal_projector.weight,
            )?;
            fp32::bias_f32(
                self.ctx(),
                &projected,
                self.weights.multimodal_projector.bias.as_ref(),
            )
        }

        pub fn embed_prefix(
            &self,
            vision_tokens: &Tensor,
            token_ids: &CudaBuffer,
            token_count: usize,
        ) -> Result<Tensor> {
            if token_count == 0 || token_count > self.config.max_token_len {
                return Err(Error::Other(format!(
                    "π0.5 token count must be in 1..={}, got {token_count}",
                    self.config.max_token_len
                )));
            }
            let language = fp32::embedding_lookup_f32(
                self.ctx(),
                &self.weights.token_embedding,
                token_ids,
                token_count,
            )?;
            fp32::concat_rows_f32(self.ctx(), vision_tokens, &language)
        }

        pub fn prefix_forward(&self, prefix: &Tensor) -> Result<Fp32PrefixKvCache> {
            let mut hidden = prefix.clone();
            let mut keys = Vec::with_capacity(self.config.language.depth);
            let mut values = Vec::with_capacity(self.config.language.depth);
            for (index, layer) in self.weights.language_layers.iter().enumerate() {
                let output = language_layer_fp32(
                    self.ctx(),
                    self.config.language,
                    layer,
                    &hidden,
                    index + 1 < self.config.language.depth,
                    0,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                )?;
                hidden = output.hidden;
                let cache_rows = prefix.shape().dims()[0] + self.config.action_horizon;
                keys.push(fp32::reserve_prefix_f32(
                    self.ctx(),
                    &output.key,
                    cache_rows,
                )?);
                values.push(fp32::reserve_prefix_f32(
                    self.ctx(),
                    &output.value,
                    cache_rows,
                )?);
            }
            Ok(Fp32PrefixKvCache {
                keys,
                values,
                tokens: prefix.shape().dims()[0],
            })
        }

        fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
            require_f32(time_embedding, "time embedding")?;
            let hidden = gemm::f32(self.ctx(), time_embedding, &self.weights.time_mlp_in.weight)?;
            let hidden = fp32::bias_silu_f32(
                self.ctx(),
                &hidden,
                self.weights.time_mlp_in.bias.as_ref(),
            )?;
            let output = gemm::f32(self.ctx(), &hidden, &self.weights.time_mlp_out.weight)?;
            fp32::bias_silu_f32(self.ctx(), &output, self.weights.time_mlp_out.bias.as_ref())
        }

        fn modulation(&self, conditioning: &Tensor, weights: &Fp32LinearWeights) -> Result<Tensor> {
            let projected = gemm::f32(self.ctx(), conditioning, &weights.weight)?;
            let modulation = fp32::bias_f32(self.ctx(), &projected, weights.bias.as_ref())?;
            modulation.reshape(vec![modulation.numel()])
        }

        fn prepare_step_modulation(&self, time_embedding: &Tensor) -> Result<Fp32StepModulation> {
            let conditioning = self.conditioning(time_embedding)?;
            let mut attention = Vec::with_capacity(self.config.action_expert.depth);
            let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
            for layer in &self.weights.action_layers {
                attention.push(self.modulation(&conditioning, &layer.input_modulation)?);
                mlp.push(self.modulation(&conditioning, &layer.post_attention_modulation)?);
            }
            let final_norm =
                self.modulation(&conditioning, &self.weights.action_final_modulation)?;
            Ok(Fp32StepModulation {
                attention,
                mlp,
                final_norm,
            })
        }

        pub(super) fn prepare_all_modulation(
            &self,
            time_embeddings: &[Tensor],
        ) -> Result<Vec<Fp32StepModulation>> {
            if time_embeddings.len() != self.config.num_flow_steps {
                return Err(Error::Other(format!(
                    "π0.5 expected {} timestep embeddings, got {}",
                    self.config.num_flow_steps,
                    time_embeddings.len()
                )));
            }
            time_embeddings
                .iter()
                .map(|embedding| self.prepare_step_modulation(embedding))
                .collect()
        }

        fn denoise_step_with_modulation(
            &self,
            state: &Tensor,
            modulation: &Fp32StepModulation,
            prefix: &Fp32PrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            require_f32(state, "action state")?;
            if prefix.keys.len() != self.config.action_expert.depth
                || prefix.values.len() != self.config.action_expert.depth
                || modulation.attention.len() != self.config.action_expert.depth
                || modulation.mlp.len() != self.config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 FP32 prefix/modulation depth mismatch".into(),
                ));
            }
            let hidden = gemm::f32(self.ctx(), state, &self.weights.action_in.weight)?;
            let mut hidden =
                fp32::bias_f32(self.ctx(), &hidden, self.weights.action_in.bias.as_ref())?;
            let mut attention_normalized = None;
            for index in 0..self.config.action_expert.depth {
                let layer = &self.weights.action_layers[index];
                let next_norm_modulation = if index + 1 < self.config.action_expert.depth {
                    &modulation.attention[index + 1]
                } else {
                    &modulation.final_norm
                };
                let output = action_layer_fp32(
                    self.ctx(),
                    self.config.action_expert,
                    layer,
                    &hidden,
                    attention_normalized.as_ref(),
                    &modulation.attention[index],
                    &modulation.mlp[index],
                    next_norm_modulation,
                    &prefix.keys[index],
                    &prefix.values[index],
                    prefix.tokens,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                )?;
                hidden = output.hidden;
                attention_normalized = Some(output.next_normalized);
            }
            let hidden = attention_normalized.ok_or_else(|| {
                Error::Other("π0.5 action expert must contain at least one layer".into())
            })?;
            let velocity = gemm::f32(self.ctx(), &hidden, &self.weights.action_out.weight)?;
            let velocity = fp32::bias_f32(
                self.ctx(),
                &velocity,
                self.weights.action_out.bias.as_ref(),
            )?;
            fp32::euler_update_f32(self.ctx(), state, &velocity, dt)
        }

        pub fn denoise_step(
            &self,
            state: &Tensor,
            time_embedding: &Tensor,
            prefix: &Fp32PrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            let modulation = self.prepare_step_modulation(time_embedding)?;
            self.denoise_step_with_modulation(state, &modulation, prefix, dt)
        }
    }

    impl super::super::Blocks for Fp32Blocks {
        type Prefix = Fp32PrefixKvCache;
        type StepModulation = Fp32StepModulation;
        fn config(&self) -> &Pi05Config {
            &self.config
        }
        fn vision(&self, patches: &Tensor, native: bool) -> Result<Tensor> {
            let _ = native;
            self.encode_vision(patches)
        }
        fn embed_prefix(&self, vision: &Tensor, ids: &CudaBuffer, count: usize) -> Result<Tensor> {
            self.embed_prefix(vision, ids, count)
        }
        fn prefix(&self, input: &Tensor) -> Result<Self::Prefix> {
            self.prefix_forward(input)
        }
        fn prepare_modulation(&self, embeddings: &[Tensor]) -> Result<Vec<Self::StepModulation>> {
            self.prepare_all_modulation(embeddings)
        }
        fn eager_modulation(
            &self,
            embeddings: &[Tensor],
        ) -> Result<Option<Vec<Self::StepModulation>>> {
            self.prepare_all_modulation(embeddings).map(Some)
        }
        fn step(
            &self,
            state: &Tensor,
            embedding: &Tensor,
            prefix: &Self::Prefix,
            dt: f32,
        ) -> Result<Tensor> {
            self.denoise_step(state, embedding, prefix, dt)
        }
        fn step_with_modulation(
            &self,
            state: &Tensor,
            modulation: &Self::StepModulation,
            prefix: &Self::Prefix,
            dt: f32,
        ) -> Result<Tensor> {
            self.denoise_step_with_modulation(state, modulation, prefix, dt)
        }
    }
}

impl crate::pi05::model::PrepareBlocks for backbone::Fp32Blocks {
    fn backend(&self) -> &std::sync::Arc<crate::pi05::backend::RuntimeBackend> {
        &self.backend
    }
    fn workspace_requirements(
        &self,
        tokens: usize,
    ) -> apxinf_core::Result<crate::pi05::model::WorkspaceRequirements> {
        // FP32 attention is a self-contained kernel (no split-KV scratch), so
        // the doubled BF16 arena bound is the whole requirement. Tactics and
        // autotune are intentionally not wired for FP32 yet.
        Ok(crate::pi05::model::WorkspaceRequirements {
            bytes: self.config.cuda_graph_workspace_bytes_fp32(tokens)?,
            fp8_scratch: None,
        })
    }
    fn raw_patch_dtype(&self) -> apxinf_core::DType {
        apxinf_core::DType::F32
    }
    fn preprocess(
        &self,
        images: &crate::pi05::backend::DeviceBuffer,
        patches: &Tensor,
        layout: crate::pi05::Pi05ImageLayout,
    ) -> Result<()> {
        crate::pi05::backend::kernels::preprocess::rgb_u8_to_patches_f32(
            self.backend.context(),
            images,
            patches,
            self.config.num_views,
            self.config.image_size,
            self.config.patch_size,
            layout,
        )
    }
}
