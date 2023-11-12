// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:17:12 on Sun, Nov 12, 2023
//
// Description: decoding fp8 fwd hdim64

#include "decoding_attn_fp8/decoding_fp8_fwd_launch_template.h"

#ifdef FAI_ENABLE_FP8

template <>
void run_mha_decoding_fp8_fwd_<64, __nv_fp8_e5m2>(const DecodingFP8Params<__nv_fp8_e5m2> &params) {
    mha_decoding_fp8_fwd<64, 256, 4, __nv_fp8_e5m2>(params);
}

template <>
void run_mha_decoding_fp8_fwd_<64, __nv_fp8_e4m3>(const DecodingFP8Params<__nv_fp8_e4m3> &params) {
    mha_decoding_fp8_fwd<64, 256, 4, __nv_fp8_e4m3>(params);
}

#endif
