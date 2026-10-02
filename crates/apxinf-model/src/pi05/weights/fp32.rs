//! Native-FP32 π0.5 device weights.
//!
//! Every tensor is uploaded as `DType::F32`. QKV and gate/up are packed with a
//! plain column concatenation only: there is no dual-GeGLU interleaving, so
//! `[gate | up]` rows feed the unfused `geglu_f32` operator directly.

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use crate::pi05::weights::packing::concat_host_2d;
use crate::pi05::{
    ActionLayerWeights, AdaRmsNormWeights, LanguageLayerWeights, LayerNormWeights, LinearWeights,
    Pi05Weights, VisionBlockWeights,
};

/// Upload any non-FP8 host tensor to the device as FP32.
pub fn fp32_to_device(tensor: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
    if tensor.dtype() == DType::F8E4M3 {
        return Err(Error::Other(
            "cannot convert scale-less E4M3 data to FP32".into(),
        ));
    }
    if tensor.dtype() == DType::F32 {
        return backend.to_device(tensor);
    }
    backend.to_device(&Tensor::from_f32(
        tensor.shape().dims().to_vec(),
        &tensor.to_f32_vec()?,
    )?)
}

fn concat_biases_fp32(tensors: &[&Tensor], backend: &dyn Backend) -> Result<Tensor> {
    let mut values = Vec::new();
    for tensor in tensors {
        if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
            return Err(Error::Other(
                "packed FP32 biases must be non-FP8 vectors".into(),
            ));
        }
        values.extend(tensor.to_f32_vec()?);
    }
    backend.to_device(&Tensor::from_f32(vec![values.len()], &values)?)
}

#[derive(Debug)]
pub struct Fp32LinearWeights {
    /// Physical row-major `[input, output]` matrix.
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Fp32LinearWeights {
    pub fn from_host(linear: &LinearWeights, backend: &dyn Backend) -> Result<Self> {
        Self::from_host_parts(&[linear], backend)
    }

    /// Pack projections along the output dimension (plain concatenation).
    pub fn from_host_parts(linears: &[&LinearWeights], backend: &dyn Backend) -> Result<Self> {
        if linears.is_empty() {
            return Err(Error::Other(
                "cannot pack an empty FP32 linear group".into(),
            ));
        }
        let weight = if let [single] = linears {
            fp32_to_device(&single.weight, backend)?
        } else {
            let packed = concat_host_2d(
                &linears
                    .iter()
                    .map(|linear| &linear.weight)
                    .collect::<Vec<_>>(),
            )?;
            backend.to_device(&packed)?
        };
        let bias = if linears.iter().all(|linear| linear.bias.is_none()) {
            None
        } else if linears.iter().all(|linear| linear.bias.is_some()) {
            Some(concat_biases_fp32(
                &linears
                    .iter()
                    .map(|linear| linear.bias.as_ref().unwrap())
                    .collect::<Vec<_>>(),
                backend,
            )?)
        } else {
            return Err(Error::Other(
                "cannot pack FP32 projections with mixed bias presence".into(),
            ));
        };
        Ok(Self { weight, bias })
    }
}

#[derive(Debug)]
pub struct Fp32DeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Fp32DeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            weight: fp32_to_device(&weights.weight, backend)?,
            bias: fp32_to_device(&weights.bias, backend)?,
        })
    }
}

#[derive(Debug)]
pub struct Fp32DeviceVisionBlock {
    pub norm1: Fp32DeviceLayerNorm,
    pub qkv: Fp32LinearWeights,
    pub output: Fp32LinearWeights,
    pub norm2: Fp32DeviceLayerNorm,
    pub fc1: Fp32LinearWeights,
    pub fc2: Fp32LinearWeights,
}

impl Fp32DeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            norm1: Fp32DeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Fp32LinearWeights::from_host_parts(&[&weights.q, &weights.k, &weights.v], backend)?,
            output: Fp32LinearWeights::from_host(&weights.output, backend)?,
            norm2: Fp32DeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Fp32LinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Fp32LinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

#[derive(Debug)]
pub struct Fp32DeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Fp32LinearWeights,
    pub output: Fp32LinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Fp32LinearWeights,
    pub down: Fp32LinearWeights,
}

impl Fp32DeviceLanguageLayer {
    fn from_host(weights: &LanguageLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_norm_scale: fp32_to_device(&weights.input_norm_scale, backend)?,
            qkv: Fp32LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Fp32LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: fp32_to_device(&weights.post_attention_norm_scale, backend)?,
            gate_up: Fp32LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Fp32LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

#[derive(Debug)]
pub struct Fp32DeviceActionLayer {
    pub input_modulation: Fp32LinearWeights,
    pub qkv: Fp32LinearWeights,
    pub output: Fp32LinearWeights,
    pub post_attention_modulation: Fp32LinearWeights,
    pub gate_up: Fp32LinearWeights,
    pub down: Fp32LinearWeights,
}

impl Fp32DeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_modulation: modulation_to_device(&weights.input_norm, backend)?,
            qkv: Fp32LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Fp32LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_modulation: modulation_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Fp32LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Fp32LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn modulation_to_device(
    weights: &AdaRmsNormWeights,
    backend: &dyn Backend,
) -> Result<Fp32LinearWeights> {
    Fp32LinearWeights::from_host(&weights.modulation, backend)
}

/// Fully materialized native-FP32 π0.5 weights.
#[derive(Debug)]
pub struct Fp32Weights {
    pub patch_embedding: Fp32LinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<Fp32DeviceVisionBlock>,
    pub vision_post_norm: Fp32DeviceLayerNorm,
    pub multimodal_projector: Fp32LinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<Fp32DeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Fp32DeviceActionLayer>,
    pub action_final_modulation: Fp32LinearWeights,
    pub action_in: Fp32LinearWeights,
    pub action_out: Fp32LinearWeights,
    pub time_mlp_in: Fp32LinearWeights,
    pub time_mlp_out: Fp32LinearWeights,
}

impl Fp32Weights {
    pub fn from_host(weights: &Pi05Weights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            patch_embedding: Fp32LinearWeights::from_host(&weights.vision.patch_embedding, backend)?,
            position_embedding: fp32_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| Fp32DeviceVisionBlock::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: Fp32DeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Fp32LinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: fp32_to_device(&weights.vision.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| Fp32DeviceLanguageLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: fp32_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| Fp32DeviceActionLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            action_final_modulation: modulation_to_device(&weights.action_final_norm, backend)?,
            action_in: Fp32LinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Fp32LinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Fp32LinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Fp32LinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::CpuBackend;

    #[test]
    fn packs_qkv_as_native_fp32_plain_concat() {
        let linear = |weight: &[f32], shape: [usize; 2], bias: &[f32]| LinearWeights {
            weight: Tensor::from_f32(shape.to_vec(), weight).unwrap(),
            bias: Some(Tensor::from_f32(vec![bias.len()], bias).unwrap()),
        };
        let q = linear(&[1., 2., 3., 4.], [2, 2], &[1., 2.]);
        let k = linear(&[5., 6.], [2, 1], &[3.]);
        let v = linear(&[7., 8.], [2, 1], &[4.]);
        let packed = Fp32LinearWeights::from_host_parts(&[&q, &k, &v], &CpuBackend).unwrap();
        assert_eq!(packed.weight.shape().dims(), &[2, 4]);
        assert_eq!(packed.weight.dtype(), DType::F32);
        assert_eq!(
            packed.weight.to_f32_vec().unwrap(),
            vec![1., 2., 5., 7., 3., 4., 6., 8.]
        );
        assert_eq!(
            packed.bias.unwrap().to_f32_vec().unwrap(),
            vec![1., 2., 3., 4.]
        );
    }

    #[test]
    fn bf16_host_weights_upload_as_f32_without_dual_geglu_interleave() {
        let to_bf16 = |values: &[f32]| {
            Tensor::from_bf16(
                vec![1, values.len()],
                &values
                    .iter()
                    .map(|v| half::bf16::from_f32(*v))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let gate = LinearWeights {
            weight: to_bf16(&[1., 2.]),
            bias: None,
        };
        let up = LinearWeights {
            weight: to_bf16(&[3., 4.]),
            bias: None,
        };
        let packed = Fp32LinearWeights::from_host_parts(&[&gate, &up], &CpuBackend).unwrap();
        assert_eq!(packed.weight.dtype(), DType::F32);
        assert_eq!(packed.weight.to_f32_vec().unwrap(), vec![1., 2., 3., 4.]);
        assert!(packed.bias.is_none());
        let single = Fp32LinearWeights::from_host(&gate, &CpuBackend).unwrap();
        assert_eq!(single.weight.dtype(), DType::F32);
    }

    #[test]
    fn mixed_bias_presence_and_fp8_are_rejected() {
        let with_bias = LinearWeights {
            weight: Tensor::from_f32(vec![1, 1], &[1.]).unwrap(),
            bias: Some(Tensor::from_f32(vec![1], &[1.]).unwrap()),
        };
        let without = LinearWeights {
            weight: Tensor::from_f32(vec![1, 1], &[1.]).unwrap(),
            bias: None,
        };
        assert!(Fp32LinearWeights::from_host_parts(&[&with_bias, &without], &CpuBackend).is_err());
    }
}
