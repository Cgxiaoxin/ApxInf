//! Native-FP32 operator checks against straightforward CPU references.
use apxinf_core::{DType, Tensor};
use apxinf_cuda::{
    kernels::{fp32, gemm},
    transfers, CudaContext,
};

fn up(shape: Vec<usize>, data: &[f32]) -> Tensor {
    transfers::to_cuda(&Tensor::from_f32(shape, data).unwrap(), 0).unwrap()
}
fn down(ctx: &CudaContext, tensor: &Tensor) -> Vec<f32> {
    ctx.synchronize().unwrap();
    transfers::to_cpu(tensor).unwrap().to_f32_vec().unwrap()
}
fn pattern(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101) % 211) as f32 - 105.0) / 53.0)
        .collect()
}
fn close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * (1.0 + e.abs()),
            "{what}[{i}]: got {a}, expected {e}"
        );
    }
}
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

#[test]
#[ignore = "requires a CUDA device"]
fn gemm_f32_matches_cpu_and_fails_closed() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (5, 37, 11);
    let (a, b) = (pattern(m * k, 1), pattern(k * n, 2));
    let out = gemm::f32(&ctx, &up(vec![m, k], &a), &up(vec![k, n], &b)).unwrap();
    assert_eq!(out.dtype(), DType::F32);
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
        }
    }
    close(&down(&ctx, &out), &expected, 1e-5, "gemm");
    let bf = transfers::to_cuda(
        &Tensor::from_bf16(vec![k, n], &vec![half::bf16::ONE; k * n]).unwrap(),
        0,
    )
    .unwrap();
    assert!(gemm::f32(&ctx, &up(vec![m, k], &a), &bf).is_err());
    assert!(gemm::f32(&ctx, &up(vec![m, k], &a), &up(vec![k + 1, n], &pattern((k + 1) * n, 3))).is_err());
}

#[test]
#[ignore = "requires a CUDA device"]
fn norms_residuals_and_activations_match_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    let (rows, cols) = (3usize, 300usize);
    let x = pattern(rows * cols, 4);
    let w = pattern(cols, 5);
    let bias = pattern(cols, 6);
    let res = pattern(rows * cols, 7);
    let eps = 1e-6f32;
    let (xt, wt, bt, rt) = (
        up(vec![rows, cols], &x),
        up(vec![cols], &w),
        up(vec![cols], &bias),
        up(vec![rows, cols], &res),
    );

    // rms
    let mut e = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps).sqrt();
        for c in 0..cols {
            e[r * cols + c] = row[c] * inv * w[c];
        }
    }
    close(&down(&ctx, &fp32::rms_f32(&ctx, &xt, &wt, eps).unwrap()), &e, 1e-5, "rms");

    // layer norm
    let mut e = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            e[r * cols + c] = (row[c] - mean) * inv * w[c] + bias[c];
        }
    }
    close(
        &down(&ctx, &fp32::layer_f32(&ctx, &xt, &wt, &bt, eps).unwrap()),
        &e,
        1e-4,
        "layer",
    );

    // bias + gelu / silu / residual
    let eb: Vec<f32> = (0..rows * cols).map(|i| x[i] + bias[i % cols]).collect();
    close(&down(&ctx, &fp32::bias_f32(&ctx, &xt, Some(&bt)).unwrap()), &eb, 1e-6, "bias");
    let eg: Vec<f32> = eb.iter().map(|v| gelu(*v)).collect();
    close(&down(&ctx, &fp32::bias_gelu_f32(&ctx, &xt, Some(&bt)).unwrap()), &eg, 1e-5, "gelu");
    let es: Vec<f32> = eb.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    close(&down(&ctx, &fp32::bias_silu_f32(&ctx, &xt, Some(&bt)).unwrap()), &es, 1e-5, "silu");
    let er: Vec<f32> = (0..rows * cols).map(|i| eb[i] + res[i]).collect();
    close(
        &down(&ctx, &fp32::bias_residual_f32(&ctx, &xt, Some(&bt), &rt).unwrap()),
        &er,
        1e-6,
        "bias residual",
    );

    // bias residual rms
    let fused = fp32::bias_residual_rms_f32(&ctx, &xt, Some(&bt), &rt, &wt, eps).unwrap();
    close(&down(&ctx, &fused.hidden), &er, 1e-6, "hidden");
    let mut e = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &er[r * cols..(r + 1) * cols];
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps).sqrt();
        for c in 0..cols {
            e[r * cols + c] = row[c] * inv * w[c];
        }
    }
    close(&down(&ctx, &fused.normalized), &e, 1e-5, "residual rms");

    // adaptive gate residual rms
    let style_a = pattern(3 * cols, 8);
    let style_b = pattern(3 * cols, 9);
    let fused = fp32::adaptive_gate_residual_rms_f32(
        &ctx,
        &xt,
        &rt,
        &up(vec![3 * cols], &style_a),
        &up(vec![3 * cols], &style_b),
        eps,
    )
    .unwrap();
    let hidden: Vec<f32> = (0..rows * cols)
        .map(|i| res[i] + x[i] * style_a[2 * cols + i % cols])
        .collect();
    close(&down(&ctx, &fused.hidden), &hidden, 1e-6, "ada hidden");
    let mut e = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &hidden[r * cols..(r + 1) * cols];
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps).sqrt();
        for c in 0..cols {
            e[r * cols + c] = row[c] * inv * (1.0 + style_b[c]) + style_b[cols + c];
        }
    }
    close(&down(&ctx, &fused.normalized), &e, 1e-5, "ada normalized");
    let ada = fp32::adaptive_rms_f32(&ctx, &xt, &up(vec![3 * cols], &style_b), eps).unwrap();
    let mut e = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps).sqrt();
        for c in 0..cols {
            e[r * cols + c] = row[c] * inv * (1.0 + style_b[c]) + style_b[cols + c];
        }
    }
    close(&down(&ctx, &ada), &e, 1e-5, "ada rms");

    // geglu / euler / concat / reserve
    let inner = 20usize;
    let gu = pattern(rows * 2 * inner, 10);
    let out = fp32::geglu_f32(&ctx, &up(vec![rows, 2 * inner], &gu)).unwrap();
    let e: Vec<f32> = (0..rows * inner)
        .map(|i| gelu(gu[(i / inner) * 2 * inner + i % inner]) * gu[(i / inner) * 2 * inner + inner + i % inner])
        .collect();
    close(&down(&ctx, &out), &e, 1e-5, "geglu");
    let euler = fp32::euler_update_f32(&ctx, &xt, &rt, -0.1).unwrap();
    let e: Vec<f32> = (0..rows * cols).map(|i| x[i] - 0.1 * res[i]).collect();
    close(&down(&ctx, &euler), &e, 1e-6, "euler");
    let cat = fp32::concat_rows_f32(&ctx, &xt, &rt).unwrap();
    assert_eq!(cat.shape().dims(), &[2 * rows, cols]);
    let mut e = x.clone();
    e.extend(&res);
    close(&down(&ctx, &cat), &e, 0.0, "concat");
    let reserved = fp32::reserve_prefix_f32(&ctx, &xt, rows + 4).unwrap();
    assert_eq!(reserved.shape().dims(), &[rows + 4, cols]);
    close(&down(&ctx, &reserved)[..rows * cols], &x, 0.0, "reserve");

    // fail-closed on BF16 input
    let bf = transfers::to_cuda(
        &Tensor::from_bf16(vec![rows, cols], &vec![half::bf16::ONE; rows * cols]).unwrap(),
        0,
    )
    .unwrap();
    assert!(fp32::rms_f32(&ctx, &bf, &wt, eps).is_err());
    assert!(fp32::layer_f32(&ctx, &bf, &wt, &bt, eps).is_err());
    assert!(fp32::geglu_f32(&ctx, &bf).is_err());
}

#[test]
#[ignore = "requires a CUDA device"]
fn rope_and_attention_match_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, q_heads, kv_heads, hd) = (6usize, 3usize, 1usize, 8usize);
    let theta = 10000.0f32;
    let width = (q_heads + 2 * kv_heads) * hd;
    let qkv = pattern(tokens * width, 11);
    let bias = pattern(width, 12);
    let offset = 5usize;
    let out = fp32::split_qkv_apply_f32(
        &ctx,
        &up(vec![tokens, width], &qkv),
        Some(&up(vec![width], &bias)),
        q_heads,
        kv_heads,
        hd,
        theta,
        offset,
    )
    .unwrap();
    let rope = |token: usize, base: usize, bias_base: usize| -> Vec<f32> {
        let half = hd / 2;
        let mut o = vec![0.0f32; hd];
        for p in 0..half {
            let f = theta.powf(-(p as f32) / half as f32);
            let (s, c) = ((offset + token) as f32 * f).sin_cos();
            let a = qkv[token * width + base + p] + bias[bias_base + p];
            let b = qkv[token * width + base + half + p] + bias[bias_base + half + p];
            o[p] = a * c - b * s;
            o[half + p] = b * c + a * s;
        }
        o
    };
    let (q, k, v) = (down(&ctx, &out.q), down(&ctx, &out.k), down(&ctx, &out.v));
    let mut e_q = Vec::new();
    let mut e_k = Vec::new();
    let mut e_v = Vec::new();
    for t in 0..tokens {
        for h in 0..q_heads {
            e_q.extend(rope(t, h * hd, h * hd));
        }
        e_k.extend(rope(t, q_heads * hd, q_heads * hd));
        for d in 0..hd {
            e_v.push(qkv[t * width + (q_heads + kv_heads) * hd + d] + bias[(q_heads + kv_heads) * hd + d]);
        }
    }
    close(&q, &e_q, 2e-5, "rope q");
    close(&k, &e_k, 2e-5, "rope k");
    close(&v, &e_v, 1e-6, "v");

    // MQA over q [tokens, heads, hd] and shared k/v with extra cache rows.
    let key_tokens = 6usize;
    let kv_rows = 9usize;
    let kc = pattern(kv_rows * hd, 13);
    let vc = pattern(kv_rows * hd, 14);
    let mqa = fp32::mqa_f32(
        &ctx,
        &up(vec![tokens, q_heads, hd], &e_q),
        &up(vec![kv_rows, hd], &kc),
        &up(vec![kv_rows, hd], &vc),
        key_tokens,
    )
    .unwrap();
    let mut e = Vec::new();
    for t in 0..tokens {
        for h in 0..q_heads {
            let qv = &e_q[(t * q_heads + h) * hd..][..hd];
            let scores: Vec<f32> = (0..key_tokens)
                .map(|j| (0..hd).map(|d| qv[d] * kc[j * hd + d]).sum::<f32>() / (hd as f32).sqrt())
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let ex: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = ex.iter().sum();
            for d in 0..hd {
                e.push((0..key_tokens).map(|j| ex[j] / z * vc[j * hd + d]).sum());
            }
        }
    }
    close(&down(&ctx, &mqa), &e, 1e-5, "mqa");

    // Larger MQA shape closer to π0.5 language/action (head_dim=256, many keys).
    // Exercises the cuBLAS F32 MQA path (key_tokens fits kSoftmaxMaxCols=1024).
    let (q2, h2, d2, k2) = (4usize, 8usize, 256usize, 512usize);
    let q_big = pattern(q2 * h2 * d2, 21);
    let k_big = pattern(k2 * d2, 22);
    let v_big = pattern(k2 * d2, 23);
    let mqa_big = fp32::mqa_f32(
        &ctx,
        &up(vec![q2, h2, d2], &q_big),
        &up(vec![k2, d2], &k_big),
        &up(vec![k2, d2], &v_big),
        k2,
    )
    .unwrap();
    let mut e_big = Vec::with_capacity(q2 * h2 * d2);
    for t in 0..q2 {
        for h in 0..h2 {
            let qv = &q_big[(t * h2 + h) * d2..][..d2];
            let scores: Vec<f32> = (0..k2)
                .map(|j| (0..d2).map(|d| qv[d] * k_big[j * d2 + d]).sum::<f32>() / (d2 as f32).sqrt())
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let ex: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = ex.iter().sum();
            for d in 0..d2 {
                e_big.push((0..k2).map(|j| ex[j] / z * v_big[j * d2 + d]).sum());
            }
        }
    }
    close(&down(&ctx, &mqa_big), &e_big, 2e-4, "mqa_cublas_shape");

    // MHA with two batches of 4 tokens.
    let (heads, per_batch, batches) = (2usize, 4usize, 2usize);
    let n = per_batch * batches * heads * hd;
    let (q, k, v) = (pattern(n, 15), pattern(n, 16), pattern(n, 17));
    let shape = vec![per_batch * batches, heads, hd];
    let mha = fp32::mha_f32(&ctx, &up(shape.clone(), &q), &up(shape.clone(), &k), &up(shape, &v), per_batch).unwrap();
    let mut e = vec![0.0f32; n];
    for b in 0..batches {
        for t in 0..per_batch {
            for h in 0..heads {
                let qi = ((b * per_batch + t) * heads + h) * hd;
                let scores: Vec<f32> = (0..per_batch)
                    .map(|j| {
                        let ki = ((b * per_batch + j) * heads + h) * hd;
                        (0..hd).map(|d| q[qi + d] * k[ki + d]).sum::<f32>() / (hd as f32).sqrt()
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                let ex: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f32 = ex.iter().sum();
                for d in 0..hd {
                    e[qi + d] = (0..per_batch)
                        .map(|j| ex[j] / z * v[((b * per_batch + j) * heads + h) * hd + d])
                        .sum();
                }
            }
        }
    }
    close(&down(&ctx, &mha), &e, 1e-5, "mha");
}
