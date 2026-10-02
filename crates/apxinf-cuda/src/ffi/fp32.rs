//! Raw bindings for the native-FP32 PI0.5 reference operators
//! (`adapters/fp32_adapter.cu`). All pointers are FP32 device pointers.

use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    pub fn apxinf_fp32_bias_activation(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        count: i64,
        cols: i32,
        activation: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_bias_residual(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        count: i64,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_geglu(
        gate_up: *const c_void,
        output: *mut c_void,
        rows: i32,
        inner: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_euler_update(
        state: *const c_void,
        velocity: *const c_void,
        output: *mut c_void,
        count: i64,
        dt: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_bias_position(
        projection: *const c_void,
        bias: *const c_void,
        position: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        tokens_per_view: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_embedding_scaled(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        tokens: i32,
        width: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_rms_norm(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_layer_norm(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_ada_rms_norm(
        input: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_bias_residual_rms_norm(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_bias_residual_layer_norm(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        norm_bias: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_ada_gate_residual_rms_norm(
        projection: *const c_void,
        residual: *const c_void,
        gate_style: *const c_void,
        norm_style: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_qkv_rope(
        qkv: *const c_void,
        bias: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        theta: f32,
        position_offset: i32,
        kv_output_offset: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_qkv_split_bias(
        qkv: *const c_void,
        bias: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        projection_width: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_mqa(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        query_tokens: i32,
        key_tokens: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_fp32_mha(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        tokens_per_batch: i32,
        batches: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
