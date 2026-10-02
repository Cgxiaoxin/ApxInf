// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA kernels for the native-FP32 PI0.5 reference executor.
//
// Every operator reads and writes FP32 only; accumulation is FP32. These are
// deliberately simple reference kernels (one CTA per row / per query-head) whose
// arithmetic mirrors the BF16 static operators minus every BF16 rounding, so an
// FP32 forward is a faithful full-precision twin of the BF16 schedule. They are
// not tuned for latency.

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kFp32Threads = 256;

__device__ __forceinline__ float fp32_gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

__device__ __forceinline__ float fp32_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ float fp32_warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  return value;
}

// `scratch` holds at least 32 floats. Safe to call repeatedly with the same
// scratch: the leading barrier retires the previous call's reads.
__device__ float fp32_block_sum(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = fp32_warp_sum(value);
  __syncthreads();
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  const float total = lane < (blockDim.x >> 5) ? scratch[lane] : 0.0f;
  return fp32_warp_sum(total);
}

__device__ float fp32_block_max(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = fp32_warp_max(value);
  __syncthreads();
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  const float total = lane < (blockDim.x >> 5) ? scratch[lane] : -FLT_MAX;
  return fp32_warp_max(total);
}

int fp32_blocks_for(int64_t count) {
  int64_t blocks = (count + kFp32Threads - 1) / kFp32Threads;
  return static_cast<int>(blocks > 65535 ? 65535 : blocks);
}

// ── Pointwise ────────────────────────────────────────────────────────────

__global__ void bias_activation_f32_kernel(
    const float* input, const float* bias, float* output, int64_t count,
    int cols, int activation) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = input[index];
    if (bias != nullptr) value += bias[index % cols];
    if (activation == 1) value = fp32_gelu_tanh(value);
    if (activation == 2) value = value / (1.0f + expf(-value));
    output[index] = value;
  }
}

__global__ void bias_residual_f32_kernel(
    const float* projection, const float* bias, const float* residual,
    float* output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = projection[index] + residual[index];
    if (bias != nullptr) value += bias[index % cols];
    output[index] = value;
  }
}

__global__ void geglu_f32_kernel(
    const float* gate_up, float* output, int rows, int inner) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int64_t row = index / inner;
    const int col = static_cast<int>(index % inner);
    const float* row_input = gate_up + row * 2 * inner;
    output[index] = fp32_gelu_tanh(row_input[col]) * row_input[inner + col];
  }
}

__global__ void euler_update_f32_kernel(
    const float* state, const float* velocity, float* output, int64_t count,
    float dt) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride)
    output[index] = state[index] + dt * velocity[index];
}

__global__ void bias_position_f32_kernel(
    const float* projection, const float* bias, const float* position,
    float* output, int64_t count, int cols, int tokens_per_view) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    const int token = static_cast<int>((index / cols) % tokens_per_view);
    float value = projection[index] + position[token * cols + col];
    if (bias != nullptr) value += bias[col];
    output[index] = value;
  }
}

// Scaled embedding lookup (Gemma multiplies by sqrt(width)); out-of-range ids
// produce zeros exactly like the BF16 operator.
__global__ void embedding_scaled_f32_kernel(
    const float* table, const uint32_t* ids, float* output, int tokens,
    int width, int vocab_size) {
  const int64_t count = static_cast<int64_t>(tokens) * width;
  const float normalizer = sqrtf(static_cast<float>(width));
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int token = static_cast<int>(index / width);
    const int col = static_cast<int>(index % width);
    const uint32_t id = ids[token];
    output[index] = id < static_cast<uint32_t>(vocab_size)
        ? table[static_cast<int64_t>(id) * width + col] * normalizer
        : 0.0f;
  }
}

// ── Normalization ────────────────────────────────────────────────────────

__global__ void rms_norm_f32_rows_kernel(
    const float* input, const float* weight, float* output, int cols,
    float eps) {
  __shared__ float scratch[32];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = input[base + col];
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(fp32_block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    output[base + col] = input[base + col] * inverse_rms * weight[col];
}

__global__ void layer_norm_f32_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int cols, float eps) {
  __shared__ float scratch[32];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += input[base + col];
  const float mean = fp32_block_sum(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = input[base + col] - mean;
    variance_sum += centered * centered;
  }
  const float inverse_std =
      rsqrtf(fp32_block_sum(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    output[base + col] =
        (input[base + col] - mean) * inverse_std * weight[col] + bias[col];
}

__global__ void ada_rms_norm_f32_kernel(
    const float* input, const float* style, float* output, int cols,
    float eps) {
  __shared__ float scratch[32];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = input[base + col];
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(fp32_block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    output[base + col] =
        input[base + col] * inverse_rms * (1.0f + style[col]) + style[cols + col];
}

__global__ void bias_residual_rms_norm_f32_kernel(
    const float* projection, const float* bias, const float* residual,
    const float* weight, float* hidden, float* normalized, int cols,
    float eps) {
  __shared__ float scratch[32];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = projection[base + col] + residual[base + col];
    if (bias != nullptr) value += bias[col];
    hidden[base + col] = value;
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(fp32_block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    normalized[base + col] = hidden[base + col] * inverse_rms * weight[col];
}

__global__ void bias_residual_layer_norm_f32_kernel(
    const float* projection, const float* bias, const float* residual,
    const float* weight, const float* norm_bias, float* hidden,
    float* normalized, int cols, float eps) {
  __shared__ float scratch[32];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = projection[base + col] + residual[base + col];
    if (bias != nullptr) value += bias[col];
    hidden[base + col] = value;
    sum += value;
  }
  const float mean = fp32_block_sum(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = hidden[base + col] - mean;
    variance_sum += centered * centered;
  }
  const float inverse_std =
      rsqrtf(fp32_block_sum(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    normalized[base + col] =
        (hidden[base + col] - mean) * inverse_std * weight[col] + norm_bias[col];
}

__global__ void ada_gate_residual_rms_norm_f32_kernel(
    const float* projection, const float* residual, const float* gate_style,
    const float* norm_style, float* hidden, float* normalized, int cols,
    float eps) {
  __shared__ float scratch[32];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value =
        residual[base + col] + projection[base + col] * gate_style[2 * cols + col];
    hidden[base + col] = value;
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(fp32_block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    normalized[base + col] = hidden[base + col] * inverse_rms *
                                 (1.0f + norm_style[col]) +
                             norm_style[cols + col];
}

// ── QKV split / RoPE ─────────────────────────────────────────────────────

// Half-split ("rotate half") RoPE on Q and K, optional fused bias, K/V written
// at row `kv_output_offset + token` of a [rows, kv_heads * head_dim] buffer.
__global__ void qkv_rope_f32_kernel(
    const float* qkv, const float* bias, float* q, float* k, float* v,
    int tokens, int q_heads, int kv_heads, int head_dim, float theta,
    int position_offset, int kv_output_offset) {
  const int token = blockIdx.x;
  const int projection_head = blockIdx.y;
  const int half_dim = head_dim / 2;
  const int pair = threadIdx.x;
  if (pair >= half_dim) return;
  const int q_width = q_heads * head_dim;
  const int kv_width = kv_heads * head_dim;
  const int fused_width = q_width + 2 * kv_width;
  const int position = position_offset + token;
  const float frequency = powf(theta, -static_cast<float>(pair) / half_dim);
  float sine, cosine;
  sincosf(position * frequency, &sine, &cosine);

  if (projection_head < q_heads) {
    const int source = token * fused_width + projection_head * head_dim;
    float first = qkv[source + pair];
    float second = qkv[source + half_dim + pair];
    if (bias != nullptr) {
      first += bias[projection_head * head_dim + pair];
      second += bias[projection_head * head_dim + half_dim + pair];
    }
    const int destination = (token * q_heads + projection_head) * head_dim;
    q[destination + pair] = first * cosine - second * sine;
    q[destination + half_dim + pair] = second * cosine + first * sine;
  } else if (projection_head < q_heads + kv_heads) {
    const int head = projection_head - q_heads;
    const int source = token * fused_width + q_width + head * head_dim;
    float first = qkv[source + pair];
    float second = qkv[source + half_dim + pair];
    if (bias != nullptr) {
      first += bias[q_width + head * head_dim + pair];
      second += bias[q_width + head * head_dim + half_dim + pair];
    }
    const int destination = ((kv_output_offset + token) * kv_heads + head) * head_dim;
    k[destination + pair] = first * cosine - second * sine;
    k[destination + half_dim + pair] = second * cosine + first * sine;
  } else {
    const int head = projection_head - q_heads - kv_heads;
    const int source = token * fused_width + q_width + kv_width + head * head_dim;
    const int destination = ((kv_output_offset + token) * kv_heads + head) * head_dim;
    float first = qkv[source + pair];
    float second = qkv[source + half_dim + pair];
    if (bias != nullptr) {
      first += bias[q_width + kv_width + head * head_dim + pair];
      second += bias[q_width + kv_width + head * head_dim + half_dim + pair];
    }
    v[destination + pair] = first;
    v[destination + half_dim + pair] = second;
  }
}

__global__ void qkv_split_bias_f32_kernel(
    const float* qkv, const float* bias, float* q, float* k, float* v,
    int projection_width) {
  const int token = blockIdx.x;
  const int fused_width = 3 * projection_width;
  for (int col = threadIdx.x; col < fused_width; col += blockDim.x) {
    float value = qkv[static_cast<int64_t>(token) * fused_width + col];
    if (bias != nullptr) value += bias[col];
    const int64_t row = static_cast<int64_t>(token) * projection_width;
    if (col < projection_width) {
      q[row + col] = value;
    } else if (col < 2 * projection_width) {
      k[row + col - projection_width] = value;
    } else {
      v[row + col - 2 * projection_width] = value;
    }
  }
}

// ── Attention ────────────────────────────────────────────────────────────

// Non-causal softmax attention. Grid is (queries per batch, heads, batches).
// MQA: k/v token stride = head_dim, head stride = 0, one batch.
// MHA: k/v token stride = heads * head_dim, head stride = head_dim.
// One warp per key for the scores; softmax and the value sum use the CTA.
__global__ void attention_f32_kernel(
    const float* q, const float* k, const float* v, float* output,
    int queries_per_batch, int key_tokens, int heads, int head_dim,
    int64_t kv_token_stride, int64_t kv_head_stride,
    int kv_tokens_per_batch) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* scratch = shared + key_tokens;
  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  const int64_t global_query =
      static_cast<int64_t>(batch) * queries_per_batch + query;
  const float* query_ptr = q + (global_query * heads + head) * head_dim;
  const int64_t kv_base = static_cast<int64_t>(batch) * kv_tokens_per_batch;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  for (int token = warp; token < key_tokens; token += warps) {
    const float* key =
        k + (kv_base + token) * kv_token_stride + head * kv_head_stride;
    float dot = 0.0f;
    for (int d = lane; d < head_dim; d += 32) dot += query_ptr[d] * key[d];
    dot = fp32_warp_sum(dot);
    if (lane == 0) scores[token] = dot * scale;
  }
  __syncthreads();
  float maximum = -FLT_MAX;
  for (int token = tid; token < key_tokens; token += blockDim.x)
    maximum = fmaxf(maximum, scores[token]);
  maximum = fp32_block_max(maximum, scratch);
  float denominator = 0.0f;
  for (int token = tid; token < key_tokens; token += blockDim.x) {
    const float weight = expf(scores[token] - maximum);
    scores[token] = weight;
    denominator += weight;
  }
  denominator = fp32_block_sum(denominator, scratch);
  // fp32_block_sum ends with a barrier-free read of scratch only; scores were
  // written before its internal barriers, so they are visible here.
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float accumulator = 0.0f;
    for (int token = 0; token < key_tokens; ++token) {
      const float* value =
          v + (kv_base + token) * kv_token_stride + head * kv_head_stride;
      accumulator += scores[token] * value[d];
    }
    output[(global_query * heads + head) * head_dim + d] = accumulator / denominator;
  }
}

}  // namespace

#define FP32_REQUIRE(cond) \
  if (!(cond)) return cudaErrorInvalidValue

extern "C" cudaError_t apxinf_fp32_bias_activation(
    const void* input, const void* bias, void* output, int64_t count, int cols,
    int activation, cudaStream_t stream) {
  FP32_REQUIRE(input && output && count > 0 && cols > 0 && activation >= 0 &&
               activation <= 2);
  bias_activation_f32_kernel<<<fp32_blocks_for(count), kFp32Threads, 0, stream>>>(
      static_cast<const float*>(input), static_cast<const float*>(bias),
      static_cast<float*>(output), count, cols, activation);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_bias_residual(
    const void* projection, const void* bias, const void* residual,
    void* output, int64_t count, int cols, cudaStream_t stream) {
  FP32_REQUIRE(projection && residual && output && count > 0 && cols > 0);
  bias_residual_f32_kernel<<<fp32_blocks_for(count), kFp32Threads, 0, stream>>>(
      static_cast<const float*>(projection), static_cast<const float*>(bias),
      static_cast<const float*>(residual), static_cast<float*>(output), count,
      cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_geglu(
    const void* gate_up, void* output, int rows, int inner,
    cudaStream_t stream) {
  FP32_REQUIRE(gate_up && output && rows > 0 && inner > 0);
  geglu_f32_kernel<<<fp32_blocks_for(static_cast<int64_t>(rows) * inner),
                     kFp32Threads, 0, stream>>>(
      static_cast<const float*>(gate_up), static_cast<float*>(output), rows,
      inner);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_euler_update(
    const void* state, const void* velocity, void* output, int64_t count,
    float dt, cudaStream_t stream) {
  FP32_REQUIRE(state && velocity && output && count > 0 && std::isfinite(dt));
  euler_update_f32_kernel<<<fp32_blocks_for(count), kFp32Threads, 0, stream>>>(
      static_cast<const float*>(state), static_cast<const float*>(velocity),
      static_cast<float*>(output), count, dt);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_bias_position(
    const void* projection, const void* bias, const void* position,
    void* output, int rows, int cols, int tokens_per_view,
    cudaStream_t stream) {
  FP32_REQUIRE(projection && position && output && rows > 0 && cols > 0 &&
               tokens_per_view > 0 && rows % tokens_per_view == 0);
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_position_f32_kernel<<<fp32_blocks_for(count), kFp32Threads, 0, stream>>>(
      static_cast<const float*>(projection), static_cast<const float*>(bias),
      static_cast<const float*>(position), static_cast<float*>(output), count,
      cols, tokens_per_view);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_embedding_scaled(
    const void* table, const void* ids, void* output, int tokens, int width,
    int vocab_size, cudaStream_t stream) {
  FP32_REQUIRE(table && ids && output && tokens > 0 && width > 0 &&
               vocab_size > 0);
  embedding_scaled_f32_kernel<<<
      fp32_blocks_for(static_cast<int64_t>(tokens) * width), kFp32Threads, 0,
      stream>>>(static_cast<const float*>(table),
                static_cast<const uint32_t*>(ids), static_cast<float*>(output),
                tokens, width, vocab_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_rms_norm(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, cudaStream_t stream) {
  FP32_REQUIRE(input && weight && output && rows > 0 && cols > 0 &&
               std::isfinite(eps) && eps > 0.0f);
  rms_norm_f32_rows_kernel<<<rows, kFp32Threads, 0, stream>>>(
      static_cast<const float*>(input), static_cast<const float*>(weight),
      static_cast<float*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_layer_norm(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  FP32_REQUIRE(input && weight && bias && output && rows > 0 && cols > 0 &&
               std::isfinite(eps) && eps > 0.0f);
  layer_norm_f32_kernel<<<rows, kFp32Threads, 0, stream>>>(
      static_cast<const float*>(input), static_cast<const float*>(weight),
      static_cast<const float*>(bias), static_cast<float*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_ada_rms_norm(
    const void* input, const void* style, void* output, int rows, int cols,
    float eps, cudaStream_t stream) {
  FP32_REQUIRE(input && style && output && rows > 0 && cols > 0 &&
               std::isfinite(eps) && eps > 0.0f);
  ada_rms_norm_f32_kernel<<<rows, kFp32Threads, 0, stream>>>(
      static_cast<const float*>(input), static_cast<const float*>(style),
      static_cast<float*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_bias_residual_rms_norm(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, int rows, int cols,
    float eps, cudaStream_t stream) {
  FP32_REQUIRE(projection && residual && weight && hidden && normalized &&
               rows > 0 && cols > 0 && std::isfinite(eps) && eps > 0.0f);
  bias_residual_rms_norm_f32_kernel<<<rows, kFp32Threads, 0, stream>>>(
      static_cast<const float*>(projection), static_cast<const float*>(bias),
      static_cast<const float*>(residual), static_cast<const float*>(weight),
      static_cast<float*>(hidden), static_cast<float*>(normalized), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_bias_residual_layer_norm(
    const void* projection, const void* bias, const void* residual,
    const void* weight, const void* norm_bias, void* hidden, void* normalized,
    int rows, int cols, float eps, cudaStream_t stream) {
  FP32_REQUIRE(projection && residual && weight && norm_bias && hidden &&
               normalized && rows > 0 && cols > 0 && std::isfinite(eps) &&
               eps > 0.0f);
  bias_residual_layer_norm_f32_kernel<<<rows, kFp32Threads, 0, stream>>>(
      static_cast<const float*>(projection), static_cast<const float*>(bias),
      static_cast<const float*>(residual), static_cast<const float*>(weight),
      static_cast<const float*>(norm_bias), static_cast<float*>(hidden),
      static_cast<float*>(normalized), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_ada_gate_residual_rms_norm(
    const void* projection, const void* residual, const void* gate_style,
    const void* norm_style, void* hidden, void* normalized, int rows, int cols,
    float eps, cudaStream_t stream) {
  FP32_REQUIRE(projection && residual && gate_style && norm_style && hidden &&
               normalized && rows > 0 && cols > 0 && std::isfinite(eps) &&
               eps > 0.0f);
  ada_gate_residual_rms_norm_f32_kernel<<<rows, kFp32Threads, 0, stream>>>(
      static_cast<const float*>(projection),
      static_cast<const float*>(residual),
      static_cast<const float*>(gate_style),
      static_cast<const float*>(norm_style), static_cast<float*>(hidden),
      static_cast<float*>(normalized), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_qkv_rope(
    const void* qkv, const void* bias, void* q, void* k, void* v, int tokens,
    int q_heads, int kv_heads, int head_dim, float theta, int position_offset,
    int kv_output_offset, cudaStream_t stream) {
  FP32_REQUIRE(qkv && q && k && v && tokens > 0 && q_heads > 0 &&
               kv_heads > 0 && head_dim > 0 && head_dim % 2 == 0 &&
               head_dim / 2 <= 1024 && std::isfinite(theta) && theta > 0.0f &&
               position_offset >= 0 && kv_output_offset >= 0);
  dim3 grid(tokens, q_heads + 2 * kv_heads, 1);
  qkv_rope_f32_kernel<<<grid, head_dim / 2, 0, stream>>>(
      static_cast<const float*>(qkv), static_cast<const float*>(bias),
      static_cast<float*>(q), static_cast<float*>(k), static_cast<float*>(v),
      tokens, q_heads, kv_heads, head_dim, theta, position_offset,
      kv_output_offset);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_qkv_split_bias(
    const void* qkv, const void* bias, void* q, void* k, void* v, int tokens,
    int projection_width, cudaStream_t stream) {
  FP32_REQUIRE(qkv && q && k && v && tokens > 0 && projection_width > 0);
  qkv_split_bias_f32_kernel<<<tokens, kFp32Threads, 0, stream>>>(
      static_cast<const float*>(qkv), static_cast<const float*>(bias),
      static_cast<float*>(q), static_cast<float*>(k), static_cast<float*>(v),
      projection_width);
  return cudaGetLastError();
}

static cudaError_t fp32_launch_attention(
    const void* q, const void* k, const void* v, void* output,
    int queries_per_batch, int batches, int key_tokens, int heads,
    int head_dim, int64_t kv_token_stride, int64_t kv_head_stride,
    int kv_tokens_per_batch, cudaStream_t stream) {
  FP32_REQUIRE(q && k && v && output && queries_per_batch > 0 && batches > 0 &&
               key_tokens > 0 && heads > 0 && head_dim > 0 &&
               heads <= 65535 && batches <= 65535);
  const size_t shared = static_cast<size_t>(key_tokens + 32) * sizeof(float);
  FP32_REQUIRE(shared <= 48 * 1024);
  dim3 grid(queries_per_batch, heads, batches);
  attention_f32_kernel<<<grid, kFp32Threads, shared, stream>>>(
      static_cast<const float*>(q), static_cast<const float*>(k),
      static_cast<const float*>(v), static_cast<float*>(output),
      queries_per_batch, key_tokens, heads, head_dim, kv_token_stride,
      kv_head_stride, kv_tokens_per_batch);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fp32_mqa(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int heads, int head_dim,
    cudaStream_t stream) {
  return fp32_launch_attention(q, k, v, output, query_tokens, 1, key_tokens,
                               heads, head_dim, head_dim, 0, 0, stream);
}

extern "C" cudaError_t apxinf_fp32_mha(
    const void* q, const void* k, const void* v, void* output,
    int tokens_per_batch, int batches, int heads, int head_dim,
    cudaStream_t stream) {
  return fp32_launch_attention(
      q, k, v, output, tokens_per_batch, batches, tokens_per_batch, heads,
      head_dim, static_cast<int64_t>(heads) * head_dim, head_dim,
      tokens_per_batch, stream);
}
