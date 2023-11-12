// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:17:12 on Sun, Nov 12, 2023
//
// Description: decoding fp8 fwd hdim128

#include "decoding_attn_fp8/decoding_fp8_fwd_launch_template.h"

#ifdef FAI_ENABLE_FP8

template <>
void run_mha_decoding_fp8_fwd_<128, __nv_fp8_e5m2>(const DecodingFP8Params<__nv_fp8_e5m2> &params) {
    if (params.b <= 4) {
        mha_decoding_fp8_fwd<128, 256, 8, __nv_fp8_e5m2>(params);
    } else {
        mha_decoding_fp8_fwd<128, 128, 16, __nv_fp8_e5m2>(params);
    }
}

template <>
void run_mha_decoding_fp8_fwd_<128, __nv_fp8_e4m3>(const DecodingFP8Params<__nv_fp8_e4m3> &params) {
    if (params.b <= 4) {
        mha_decoding_fp8_fwd<128, 256, 8, __nv_fp8_e4m3>(params);
    } else {
        mha_decoding_fp8_fwd<128, 128, 16, __nv_fp8_e4m3>(params);
    }
}

#endif
