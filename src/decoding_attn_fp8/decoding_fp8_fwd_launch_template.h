// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:17:12 on Sun, Nov 12, 2023
//
// Description: decoding fp8 fwd launch template

#pragma once

#include "decoding_attn_fp8/decoding_fp8_fwd_kernel.h"
#include "decoding_attn_fp8/static_switch.h"

template <size_t HeadDim, size_t ThreadsPerBlock, size_t GroupSize, typename fp8_t>
void mha_decoding_fp8_fwd(const DecodingFP8Params<fp8_t> &params) {
    constexpr size_t warp_size = 32;
    constexpr size_t static_smem_size = ThreadsPerBlock / warp_size * sizeof(float);
    const size_t dynamic_smem_size = std::max(params.seqlen_k * sizeof(float), params.d * sizeof(float));
    FAI_CHECK_GT(params.props->sharedMemPerBlock, static_smem_size + dynamic_smem_size);

    dim3 block(ThreadsPerBlock);
    dim3 grid(params.b, params.h);

    BOOL_SWITCH(params.is_alibi, IsAlibi, [&] {
        mha_decoding_fp8_fwd_kernel<DecodingFP8KernelTraits<HeadDim, ThreadsPerBlock, GroupSize>, fp8_t, IsAlibi>
            <<<grid, block, dynamic_smem_size, params.stream>>>(params);
        FAI_CHECK_CUDART_ERROR(cudaPeekAtLastError());
    });
}
