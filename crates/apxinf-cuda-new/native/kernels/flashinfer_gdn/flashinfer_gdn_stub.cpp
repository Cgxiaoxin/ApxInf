// Copyright 2026 ApxInf contributors.
//
// Stub FlashInfer GDN entry points for architectures without tcgen05
// (e.g. consumer Blackwell sm_120). The real implementation lives in
// flashinfer_gdn_tma.cpp + flashinfer_gdn_launch.cu and is only built for
// SM100-family targets.

#include "flashinfer_gdn.h"

namespace apxinf::cuda_new::flashinfer_gdn {

int prefill(const void* /*q*/, const void* /*k*/, const void* /*v*/, void* /*out*/,
            const void* /*gate_log*/, const void* /*beta*/,
            const void* /*cu_seqlens*/, void* /*state*/,
            void* /*tensor_map_workspace*/, int /*tokens*/, int /*q_heads*/,
            int /*v_heads*/, int /*num_seqs*/, float /*scale*/,
            cudaStream_t /*stream*/) {
  // Unsupported on this compiled architecture.
  return -100;
}

size_t tensor_map_workspace_bytes(int /*v_heads*/, int /*num_seqs*/) {
  return 0;
}

}  // namespace apxinf::cuda_new::flashinfer_gdn
