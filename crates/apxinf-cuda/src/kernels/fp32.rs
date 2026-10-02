//! Native-FP32 operator contracts for the PI0.5 FP32 reference executor.
//!
//! Every function is fail-closed: all tensors must be `DType::F32` on the
//! context device, and nothing is ever cast to BF16/FP16. Outputs come from the
//! bound workspace (so a body using only these operators and `gemm::f32` stays
//! CUDA-graph capturable) and arithmetic mirrors the BF16 static operators
//! without their intermediate rounding. These are reference kernels, not tuned
//! for latency.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::contracts::{check_cuda, gpu_ptr, matrix_shape, optional_ptr};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

pub use super::attention::QkvTensors;

/// Residual stream plus its following normalization.
pub struct ResidualNormTensors {
    pub hidden: Tensor,
    pub normalized: Tensor,
}

fn require_f32(ctx: &CudaContext, operation: &str, tensors: &[&Tensor]) -> Result<()> {
    for tensor in tensors {
        if tensor.dtype() != DType::F32 {
            return Err(Error::Other(format!(
                "FP32 {operation} requires F32 tensors, got {}",
                tensor.dtype()
            )));
        }
        if tensor.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::DeviceMismatch {
                expected: Device::Cuda(ctx.device_id()),
                got: tensor.device(),
            });
        }
        if tensor.shape().dims().contains(&0) {
            return Err(Error::Other(format!(
                "FP32 {operation} dimensions must be non-zero: {:?}",
                tensor.shape().dims()
            )));
        }
    }
    Ok(())
}

fn require_bias(operation: &str, bias: Option<&Tensor>, cols: usize) -> Result<()> {
    if bias.is_some_and(|value| value.shape().dims() != [cols]) {
        return Err(Error::Other(format!(
            "FP32 {operation} bias must be [{cols}]"
        )));
    }
    Ok(())
}

fn f32_buffer(ctx: &CudaContext, shape: &[usize], operation: &str) -> Result<CudaBuffer> {
    let bytes = shape
        .iter()
        .try_fold(DType::F32.size_in_bytes(), |bytes, dim| {
            bytes.checked_mul(*dim)
        })
        .ok_or_else(|| Error::Other(format!("FP32 {operation} output size overflow")))?;
    output_buffer(ctx, bytes)
}

fn f32_tensor(buffer: CudaBuffer, shape: Vec<usize>) -> Tensor {
    buffer.into_tensor(Shape::new(shape), DType::F32)
}

fn dim_i32(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Other(format!("FP32 {what} {value} exceeds i32")))
}

// ── Pointwise ────────────────────────────────────────────────────────────

fn bias_activation(
    ctx: &CudaContext,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: i32,
    operation: &str,
) -> Result<Tensor> {
    let (_, cols) = matrix_shape(input, operation)?;
    require_f32(ctx, operation, &[input])?;
    if let Some(bias) = bias {
        require_f32(ctx, operation, &[bias])?;
    }
    require_bias(operation, bias, cols)?;
    let output = f32_buffer(ctx, input.shape().dims(), operation)?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_bias_activation(
            gpu_ptr(input)?,
            optional_ptr(bias)?,
            output.ptr(),
            input.numel() as i64,
            dim_i32(cols, "columns")?,
            activation,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, input.shape().dims().to_vec()))
}

/// `input + bias` broadcast over rows; `None` returns an FP32 copy.
pub fn bias_f32(ctx: &CudaContext, input: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    bias_activation(ctx, input, bias, 0, "bias")
}

/// `gelu_tanh(input + bias)`.
pub fn bias_gelu_f32(ctx: &CudaContext, input: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    bias_activation(ctx, input, bias, 1, "bias GELU")
}

/// `silu(input + bias)`.
pub fn bias_silu_f32(ctx: &CudaContext, input: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    bias_activation(ctx, input, bias, 2, "bias SiLU")
}

/// GeGLU over an unpermuted `[rows, 2*inner]` = `[gate | up]` matrix:
/// `gelu_tanh(gate) * up`.
pub fn geglu_f32(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    let (rows, twice_inner) = matrix_shape(gate_up, "GeGLU")?;
    require_f32(ctx, "GeGLU", &[gate_up])?;
    if twice_inner % 2 != 0 {
        return Err(Error::Other("FP32 GeGLU expects [rows,2*inner]".into()));
    }
    let inner = twice_inner / 2;
    let output = f32_buffer(ctx, &[rows, inner], "GeGLU")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_geglu(
            gpu_ptr(gate_up)?,
            output.ptr(),
            dim_i32(rows, "rows")?,
            dim_i32(inner, "inner width")?,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, vec![rows, inner]))
}

/// Row concatenation of two `[rows, cols]` matrices with equal widths.
pub fn concat_rows_f32(ctx: &CudaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let (first_rows, cols) = matrix_shape(first, "row concatenation")?;
    let (second_rows, second_cols) = matrix_shape(second, "row concatenation")?;
    require_f32(ctx, "row concatenation", &[first, second])?;
    if cols != second_cols {
        return Err(Error::Other(
            "FP32 row concatenation requires matrices with equal widths".into(),
        ));
    }
    let output = f32_buffer(ctx, &[first_rows + second_rows, cols], "row concatenation")?;
    let first_bytes = first.size_in_bytes();
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            output.ptr(),
            gpu_ptr(first)?,
            first_bytes,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            (output.ptr() as *mut u8).add(first_bytes) as *mut _,
            gpu_ptr(second)?,
            second.size_in_bytes(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(f32_tensor(output, vec![first_rows + second_rows, cols]))
}

/// `state + dt * velocity`.
pub fn euler_update_f32(
    ctx: &CudaContext,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    require_f32(ctx, "Euler update", &[state, velocity])?;
    if state.shape() != velocity.shape() {
        return Err(Error::Other(
            "FP32 Euler update expects matching tensors".into(),
        ));
    }
    let output = f32_buffer(ctx, state.shape().dims(), "Euler update")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_euler_update(
            gpu_ptr(state)?,
            gpu_ptr(velocity)?,
            output.ptr(),
            state.numel() as i64,
            dt,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, state.shape().dims().to_vec()))
}

/// Allocate a `[total_rows, cols]` K/V cache and copy `prefix` into its first rows.
pub fn reserve_prefix_f32(ctx: &CudaContext, prefix: &Tensor, total_rows: usize) -> Result<Tensor> {
    let (prefix_rows, cols) = matrix_shape(prefix, "prefix KV cache")?;
    require_f32(ctx, "prefix KV cache", &[prefix])?;
    if total_rows < prefix_rows {
        return Err(Error::Other(
            "FP32 prefix KV cache total rows are smaller than the prefix".into(),
        ));
    }
    let output = f32_buffer(ctx, &[total_rows, cols], "prefix KV cache")?;
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            output.ptr(),
            gpu_ptr(prefix)?,
            prefix.size_in_bytes(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(f32_tensor(output, vec![total_rows, cols]))
}

/// Gemma-scaled (`* sqrt(width)`) token embedding lookup from an FP32
/// `[vocab, width]` table with device `u32` ids.
pub fn embedding_lookup_f32(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    tokens: usize,
) -> Result<Tensor> {
    require_f32(ctx, "embedding", &[table])?;
    let dims = table.shape().dims();
    if dims.len() != 2 || tokens == 0 || ids.len() < tokens * 4 {
        return Err(Error::Other(
            "FP32 embedding expects [vocab,width] and device u32 ids".into(),
        ));
    }
    let output = f32_buffer(ctx, &[tokens, dims[1]], "embedding")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_embedding_scaled(
            gpu_ptr(table)?,
            ids.ptr(),
            output.ptr(),
            dim_i32(tokens, "tokens")?,
            dim_i32(dims[1], "width")?,
            dim_i32(dims[0], "vocab")?,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, vec![tokens, dims[1]]))
}

/// `projection + bias + position[token % tokens_per_view]`.
pub fn add_position_f32(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "position embedding")?;
    require_f32(ctx, "position embedding", &[projection, position])?;
    if let Some(bias) = bias {
        require_f32(ctx, "position embedding", &[bias])?;
    }
    let position_dims = position.shape().dims();
    if !(position_dims == [tokens_per_view, cols] || position_dims == [1, tokens_per_view, cols])
        || tokens_per_view == 0
        || rows % tokens_per_view != 0
    {
        return Err(Error::Other(
            "FP32 position embedding shape mismatch".into(),
        ));
    }
    require_bias("position embedding", bias, cols)?;
    let output = f32_buffer(ctx, &[rows, cols], "position embedding")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_bias_position(
            gpu_ptr(projection)?,
            optional_ptr(bias)?,
            gpu_ptr(position)?,
            output.ptr(),
            dim_i32(rows, "rows")?,
            dim_i32(cols, "columns")?,
            dim_i32(tokens_per_view, "tokens per view")?,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, vec![rows, cols]))
}

// ── Normalization and residual ───────────────────────────────────────────

fn norm_params(input: &Tensor, operation: &str) -> Result<(usize, usize)> {
    matrix_shape(input, operation)
}

fn positive_eps(eps: f32, operation: &str) -> Result<()> {
    if eps.is_finite() && eps > 0.0 {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "FP32 {operation} epsilon must be positive and finite"
        )))
    }
}

pub fn rms_f32(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let (rows, cols) = norm_params(input, "RMSNorm")?;
    require_f32(ctx, "RMSNorm", &[input, weight])?;
    positive_eps(eps, "RMSNorm")?;
    if weight.shape().dims() != [cols] {
        return Err(Error::Other("FP32 RMSNorm weight shape mismatch".into()));
    }
    let output = f32_buffer(ctx, &[rows, cols], "RMSNorm")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_rms_norm(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            dim_i32(rows, "rows")?,
            dim_i32(cols, "columns")?,
            eps,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, vec![rows, cols]))
}

/// LayerNorm with affine weight and bias, in FP32 (no BF16 rounding).
pub fn layer_f32(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = norm_params(input, "LayerNorm")?;
    require_f32(ctx, "LayerNorm", &[input, weight, bias])?;
    positive_eps(eps, "LayerNorm")?;
    if weight.shape().dims() != [cols] || bias.shape().dims() != [cols] {
        return Err(Error::Other("FP32 LayerNorm parameter shape mismatch".into()));
    }
    let output = f32_buffer(ctx, &[rows, cols], "LayerNorm")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_layer_norm(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            output.ptr(),
            dim_i32(rows, "rows")?,
            dim_i32(cols, "columns")?,
            eps,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, vec![rows, cols]))
}

/// Adaptive RMSNorm: `rms(x) * (1 + style[:cols]) + style[cols:2*cols]`;
/// `style` is `[3*cols]` in `[scale, shift, gate]` order.
pub fn adaptive_rms_f32(
    ctx: &CudaContext,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = norm_params(input, "AdaRMSNorm")?;
    require_f32(ctx, "AdaRMSNorm", &[input, style])?;
    positive_eps(eps, "AdaRMSNorm")?;
    if style.shape().dims() != [3 * cols] {
        return Err(Error::Other("FP32 AdaRMSNorm style must be [3*cols]".into()));
    }
    let output = f32_buffer(ctx, &[rows, cols], "AdaRMSNorm")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_ada_rms_norm(
            gpu_ptr(input)?,
            gpu_ptr(style)?,
            output.ptr(),
            dim_i32(rows, "rows")?,
            dim_i32(cols, "columns")?,
            eps,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, vec![rows, cols]))
}

/// `projection + bias + residual`.
pub fn bias_residual_f32(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let (_, cols) = matrix_shape(projection, "bias residual")?;
    require_f32(ctx, "bias residual", &[projection, residual])?;
    if let Some(bias) = bias {
        require_f32(ctx, "bias residual", &[bias])?;
    }
    if residual.shape() != projection.shape() {
        return Err(Error::Other("FP32 bias residual shape mismatch".into()));
    }
    require_bias("bias residual", bias, cols)?;
    let output = f32_buffer(ctx, projection.shape().dims(), "bias residual")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_bias_residual(
            gpu_ptr(projection)?,
            optional_ptr(bias)?,
            gpu_ptr(residual)?,
            output.ptr(),
            projection.numel() as i64,
            dim_i32(cols, "columns")?,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, projection.shape().dims().to_vec()))
}

/// `hidden = projection + bias + residual; normalized = rms(hidden) * weight`.
pub fn bias_residual_rms_f32(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "residual RMSNorm")?;
    require_f32(ctx, "residual RMSNorm", &[projection, residual, weight])?;
    if let Some(bias) = bias {
        require_f32(ctx, "residual RMSNorm", &[bias])?;
    }
    positive_eps(eps, "residual RMSNorm")?;
    if residual.shape() != projection.shape() || weight.shape().dims() != [cols] {
        return Err(Error::Other("FP32 residual RMSNorm shape mismatch".into()));
    }
    require_bias("residual RMSNorm", bias, cols)?;
    let hidden = f32_buffer(ctx, &[rows, cols], "residual RMSNorm")?;
    let normalized = f32_buffer(ctx, &[rows, cols], "residual RMSNorm")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_bias_residual_rms_norm(
            gpu_ptr(projection)?,
            optional_ptr(bias)?,
            gpu_ptr(residual)?,
            gpu_ptr(weight)?,
            hidden.ptr(),
            normalized.ptr(),
            dim_i32(rows, "rows")?,
            dim_i32(cols, "columns")?,
            eps,
            ctx.stream().handle(),
        )
    })?;
    Ok(ResidualNormTensors {
        hidden: f32_tensor(hidden, vec![rows, cols]),
        normalized: f32_tensor(normalized, vec![rows, cols]),
    })
}

/// `hidden = projection + bias + residual; normalized = layer_norm(hidden)`.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_f32(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "residual LayerNorm")?;
    require_f32(
        ctx,
        "residual LayerNorm",
        &[projection, residual, norm_weight, norm_bias],
    )?;
    if let Some(bias) = projection_bias {
        require_f32(ctx, "residual LayerNorm", &[bias])?;
    }
    positive_eps(eps, "residual LayerNorm")?;
    if residual.shape() != projection.shape()
        || norm_weight.shape().dims() != [cols]
        || norm_bias.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "FP32 residual LayerNorm shape mismatch".into(),
        ));
    }
    require_bias("residual LayerNorm", projection_bias, cols)?;
    let hidden = f32_buffer(ctx, &[rows, cols], "residual LayerNorm")?;
    let normalized = f32_buffer(ctx, &[rows, cols], "residual LayerNorm")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_bias_residual_layer_norm(
            gpu_ptr(projection)?,
            optional_ptr(projection_bias)?,
            gpu_ptr(residual)?,
            gpu_ptr(norm_weight)?,
            gpu_ptr(norm_bias)?,
            hidden.ptr(),
            normalized.ptr(),
            dim_i32(rows, "rows")?,
            dim_i32(cols, "columns")?,
            eps,
            ctx.stream().handle(),
        )
    })?;
    Ok(ResidualNormTensors {
        hidden: f32_tensor(hidden, vec![rows, cols]),
        normalized: f32_tensor(normalized, vec![rows, cols]),
    })
}

/// `hidden = residual + projection * gate_style[2*cols..]`, then
/// `normalized = rms(hidden) * (1 + norm_style[..cols]) + norm_style[cols..2*cols]`.
pub fn adaptive_gate_residual_rms_f32(
    ctx: &CudaContext,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "Ada residual RMSNorm")?;
    require_f32(
        ctx,
        "Ada residual RMSNorm",
        &[projection, residual, gate_style, norm_style],
    )?;
    positive_eps(eps, "Ada residual RMSNorm")?;
    if projection.shape() != residual.shape()
        || gate_style.shape().dims() != [3 * cols]
        || norm_style.shape().dims() != [3 * cols]
    {
        return Err(Error::Other(
            "FP32 Ada residual RMSNorm shape mismatch".into(),
        ));
    }
    let hidden = f32_buffer(ctx, &[rows, cols], "Ada residual RMSNorm")?;
    let normalized = f32_buffer(ctx, &[rows, cols], "Ada residual RMSNorm")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_ada_gate_residual_rms_norm(
            gpu_ptr(projection)?,
            gpu_ptr(residual)?,
            gpu_ptr(gate_style)?,
            gpu_ptr(norm_style)?,
            hidden.ptr(),
            normalized.ptr(),
            dim_i32(rows, "rows")?,
            dim_i32(cols, "columns")?,
            eps,
            ctx.stream().handle(),
        )
    })?;
    Ok(ResidualNormTensors {
        hidden: f32_tensor(hidden, vec![rows, cols]),
        normalized: f32_tensor(normalized, vec![rows, cols]),
    })
}

// ── QKV split and RoPE ───────────────────────────────────────────────────

struct RopeShape {
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

fn validate_rope(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    shape: &RopeShape,
    theta: f32,
) -> Result<()> {
    let expected = (shape.q_heads + 2 * shape.kv_heads) * shape.head_dim;
    require_f32(ctx, "QKV RoPE", &[qkv])?;
    if let Some(bias) = bias {
        require_f32(ctx, "QKV RoPE", &[bias])?;
    }
    if qkv.shape().dims() != [shape.tokens, expected]
        || shape.q_heads == 0
        || shape.kv_heads == 0
        || shape.head_dim == 0
        || shape.head_dim % 2 != 0
        || shape.head_dim / 2 > 1024
        || !(theta.is_finite() && theta > 0.0)
        || bias.is_some_and(|value| value.shape().dims() != [expected])
    {
        return Err(Error::Other("FP32 QKV RoPE shape mismatch".into()));
    }
    Ok(())
}

/// Split fused QKV `[tokens, (q_heads + 2*kv_heads) * head_dim]`, add the
/// optional fused bias, and apply half-split RoPE to Q and K.
/// Returns Q `[tokens, q_heads, head_dim]` and K/V `[tokens, kv_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_apply_f32(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    let tokens = qkv.shape().dims().first().copied().unwrap_or(0);
    let shape = RopeShape {
        tokens,
        q_heads,
        kv_heads,
        head_dim,
    };
    validate_rope(ctx, qkv, bias, &shape, theta)?;
    let q = f32_buffer(ctx, &[tokens, q_heads, head_dim], "QKV RoPE")?;
    let k = f32_buffer(ctx, &[tokens, kv_heads, head_dim], "QKV RoPE")?;
    let v = f32_buffer(ctx, &[tokens, kv_heads, head_dim], "QKV RoPE")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_qkv_rope(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            q.ptr(),
            k.ptr(),
            v.ptr(),
            dim_i32(tokens, "tokens")?,
            dim_i32(q_heads, "query heads")?,
            dim_i32(kv_heads, "KV heads")?,
            dim_i32(head_dim, "head dim")?,
            theta,
            dim_i32(position_offset, "position offset")?,
            0,
            ctx.stream().handle(),
        )
    })?;
    Ok(QkvTensors {
        q: f32_tensor(q, vec![tokens, q_heads, head_dim]),
        k: f32_tensor(k, vec![tokens, kv_heads, head_dim]),
        v: f32_tensor(v, vec![tokens, kv_heads, head_dim]),
    })
}

/// As [`split_qkv_apply_f32`] for the single-KV-head action expert, writing K
/// and V into rows `output_offset..` of the caller's `[rows, head_dim]` caches
/// and returning only the rotated Q.
#[allow(clippy::too_many_arguments)]
pub fn apply_q_write_kv_f32(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    k_cache: &Tensor,
    v_cache: &Tensor,
    output_offset: usize,
) -> Result<Tensor> {
    let tokens = qkv.shape().dims().first().copied().unwrap_or(0);
    let shape = RopeShape {
        tokens,
        q_heads,
        kv_heads,
        head_dim,
    };
    validate_rope(ctx, qkv, bias, &shape, theta)?;
    require_f32(ctx, "cached QKV RoPE", &[k_cache, v_cache])?;
    let cache_shape = k_cache.shape().dims();
    if cache_shape.len() != 2
        || v_cache.shape().dims() != cache_shape
        || cache_shape[1] != head_dim
        || kv_heads != 1
        || output_offset + tokens > cache_shape[0]
    {
        return Err(Error::Other("FP32 cached QKV shape mismatch".into()));
    }
    let q = f32_buffer(ctx, &[tokens, q_heads, head_dim], "cached QKV RoPE")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_qkv_rope(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            q.ptr(),
            gpu_ptr(k_cache)?,
            gpu_ptr(v_cache)?,
            dim_i32(tokens, "tokens")?,
            dim_i32(q_heads, "query heads")?,
            dim_i32(kv_heads, "KV heads")?,
            dim_i32(head_dim, "head dim")?,
            theta,
            dim_i32(position_offset, "position offset")?,
            dim_i32(output_offset, "output offset")?,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(q, vec![tokens, q_heads, head_dim]))
}

/// Vision QKV split with fused bias: `[tokens, 3*heads*head_dim]` to three
/// `[tokens, heads, head_dim]` tensors.
pub fn split_qkv_bias_f32(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "vision QKV split")?;
    require_f32(ctx, "vision QKV split", &[qkv])?;
    if let Some(bias) = bias {
        require_f32(ctx, "vision QKV split", &[bias])?;
    }
    let projection_width = heads * head_dim;
    if width != 3 * projection_width || bias.is_some_and(|value| value.shape().dims() != [width]) {
        return Err(Error::Other("FP32 vision QKV shape mismatch".into()));
    }
    let q = f32_buffer(ctx, &[tokens, heads, head_dim], "vision QKV split")?;
    let k = f32_buffer(ctx, &[tokens, heads, head_dim], "vision QKV split")?;
    let v = f32_buffer(ctx, &[tokens, heads, head_dim], "vision QKV split")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_qkv_split_bias(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            q.ptr(),
            k.ptr(),
            v.ptr(),
            dim_i32(tokens, "tokens")?,
            dim_i32(projection_width, "projection width")?,
            ctx.stream().handle(),
        )
    })?;
    Ok(QkvTensors {
        q: f32_tensor(q, vec![tokens, heads, head_dim]),
        k: f32_tensor(k, vec![tokens, heads, head_dim]),
        v: f32_tensor(v, vec![tokens, heads, head_dim]),
    })
}

// ── Attention ────────────────────────────────────────────────────────────

/// Non-causal multi-query attention in FP32. `q` is `[queries, heads, head_dim]`;
/// `k`/`v` hold at least `key_tokens` rows of `head_dim` (a pre-sized cache is
/// fine; only the first `key_tokens` rows are read). Scale is `1/sqrt(head_dim)`.
pub fn mqa_f32(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    require_f32(ctx, "MQA", &[q, k, v])?;
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q_shape.len() != 3
        || k_shape.len() < 2
        || v.shape() != k.shape()
        || k_shape[k_shape.len() - 1] != q_shape[2]
        || key_tokens == 0
        || key_tokens > k.numel() / q_shape[2]
    {
        return Err(Error::Other("FP32 MQA shape mismatch".into()));
    }
    let output = f32_buffer(ctx, q_shape, "MQA")?;
    // Prefer cuBLAS TF32 GEMM + softmax; fall back to the reference kernel if
    // the vendor path rejects the shape (e.g. key_tokens over the MQA logits cap).
    let cublas_status = unsafe {
        ffi::apxinf_static_cublas_mqa_f32(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            dim_i32(q_shape[0], "query tokens")?,
            dim_i32(key_tokens, "key tokens")?,
            dim_i32(q_shape[1], "heads")?,
            dim_i32(q_shape[2], "head dim")?,
            ctx.stream().handle(),
        )
    };
    if cublas_status == 0 {
        return Ok(f32_tensor(output, q_shape.to_vec()));
    }
    check_cuda(unsafe {
        ffi::apxinf_fp32_mqa(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            dim_i32(q_shape[0], "query tokens")?,
            dim_i32(key_tokens, "key tokens")?,
            dim_i32(q_shape[1], "heads")?,
            dim_i32(q_shape[2], "head dim")?,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, q_shape.to_vec()))
}

/// Non-causal multi-head attention in FP32 over `[tokens, heads, head_dim]`
/// tensors, independently per batch of `tokens_per_batch` consecutive tokens.
pub fn mha_f32(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    require_f32(ctx, "MHA", &[q, k, v])?;
    let shape = q.shape().dims();
    if shape.len() != 3
        || k.shape() != q.shape()
        || v.shape() != q.shape()
        || tokens_per_batch == 0
        || shape[0] % tokens_per_batch != 0
    {
        return Err(Error::Other("FP32 MHA shape mismatch".into()));
    }
    let output = f32_buffer(ctx, shape, "MHA")?;
    check_cuda(unsafe {
        ffi::apxinf_fp32_mha(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            dim_i32(tokens_per_batch, "tokens per batch")?,
            dim_i32(shape[0] / tokens_per_batch, "batches")?,
            dim_i32(shape[1], "heads")?,
            dim_i32(shape[2], "head dim")?,
            ctx.stream().handle(),
        )
    })?;
    Ok(f32_tensor(output, shape.to_vec()))
}
