// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:17:12 on Sun, Nov 12, 2023
//
// Description: decoding fp8

#pragma once

#include "common.h"

template <typename fp8_t>
struct DecodingFP8Params {
    // The QKV matrices.
    half *__restrict__ q_ptr;
    half *__restrict__ k_ptr;
    half *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    size_t q_row_stride;
    size_t k_row_stride;
    size_t v_row_stride;
    size_t q_head_stride;
    size_t k_head_stride;
    size_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio;  // precompute h / h_k,

    // Quantization: half -> fp8
    fp8_t *__restrict__ k_fp8_ptr;
    fp8_t *__restrict__ v_fp8_ptr;

    // The O matrix (output).
    half *__restrict__ o_ptr;

    // The stride between rows of O.
    size_t o_row_stride;
    size_t o_head_stride;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;

    // The scaling factors for the kernel.
    float scale_softmax;

    // array of length b+1 holding starting offset of each sequence.
    int *__restrict__ cu_seqlens_q;
    int *__restrict__ cu_seqlens_k;

    cudaStream_t stream;
    cudaDeviceProp *props;

    bool is_alibi;
};

template <size_t HeadDim, typename fp8_t>
void run_mha_decoding_fp8_fwd_(const DecodingFP8Params<fp8_t> &params);
