// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:17:12 on Sun, Nov 12, 2023
//
// Description: block info

#pragma once

#include "cuda_runtime_api.h"

struct DecodingFP8BlockInfo {
    template <typename Params>
    __device__ DecodingFP8BlockInfo(const Params &params, const int bidb, const int bidh)
        : b(bidb),
          h(bidh),
          h_k(h / params.h_h_k_ratio),
          sum_s_q(params.cu_seqlens_q[b]),
          sum_s_k(params.cu_seqlens_k[b]),
          actual_seqlen_q(params.cu_seqlens_q[b + 1] - sum_s_q),
          actual_seqlen_k(params.cu_seqlens_k[b + 1] - sum_s_k),
          row_shift(actual_seqlen_k - actual_seqlen_q),
          h_slope(1.0 / exp2f(8.0 * (h + 1) / params.h)) {}

    inline __device__ size_t q_offset(const size_t row_stride, const size_t head_stride, const size_t dim_idx) const {
        return static_cast<size_t>(sum_s_q) * row_stride + h * head_stride + dim_idx;
    }

    inline __device__ size_t k_offset(const size_t seqlen_k, const size_t row_stride, const size_t head_stride,
                                      const size_t dim_idx) const {
        return static_cast<size_t>(sum_s_k + seqlen_k) * row_stride + h_k * head_stride + dim_idx;
    }

    const int b;
    const int h;
    const int h_k;
    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    const int actual_seqlen_k;
    const int row_shift;
    const float h_slope;
};
