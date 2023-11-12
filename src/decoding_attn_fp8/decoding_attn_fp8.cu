// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:17:12 on Sun, Nov 12, 2023
//
// Description: decoding attn fp8

#include "decoding_attn_fp8/static_switch.h"
#include "decoding_attn_fp8/util.h"

#ifdef FAI_ENABLE_FP8

template <typename fp8_t>
DecodingFP8Params<fp8_t> set_mha_decoding_fp8_fwd_params(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V,
                                                         Tensor<half> *O, Tensor<int> *cu_seq_q, Tensor<int> *cu_seq_k,
                                                         size_t max_seq_q, size_t max_seq_k, cudaStream_t stream,
                                                         cudaDeviceProp *dev_prop, bool is_alibi) {
    size_t head_q = Q->getShape()[1];
    size_t dim = Q->getShape()[2];
    size_t total_k = K->getShape()[0];
    size_t head_k = K->getShape()[1];
    size_t batch = cu_seq_q->getShape()[0] - 1;

    FAI_CHECK_LE(dim, 256);
    FAI_CHECK_EQ(head_q % head_k, 0);

    DecodingFP8Params<fp8_t> params;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = Q->getDevPtr();
    params.k_ptr = K->getDevPtr();
    params.v_ptr = V->getDevPtr();

    params.q_row_stride = head_q * dim;
    params.k_row_stride = head_k * dim;
    params.v_row_stride = head_k * dim;
    params.q_head_stride = dim;
    params.k_head_stride = dim;
    params.v_head_stride = dim;

    params.h = head_q;
    params.h_k = head_k;
    params.h_h_k_ratio = params.h / params.h_k;

    Tensor<fp8_t> *k_fp8 = new Tensor<fp8_t>({total_k, head_k, dim}, "Tensor k_fp8");
    FAI_CHECK(k_fp8);
    params.k_fp8_ptr = k_fp8->getDevPtr();

    Tensor<fp8_t> *v_fp8 = new Tensor<fp8_t>({total_k, head_k, dim}, "Tensor v_fp8");
    FAI_CHECK(v_fp8);
    params.v_fp8_ptr = v_fp8->getDevPtr();

    params.o_ptr = O->getDevPtr();

    params.o_row_stride = head_q * dim;
    params.o_head_stride = dim;

    // Set the dimensions.
    params.b = batch;
    params.seqlen_q = max_seq_q;
    params.seqlen_k = max_seq_k;
    params.d = dim;

    params.scale_softmax = 1.0 / std::sqrt(dim);

    params.cu_seqlens_q = cu_seq_q->getDevPtr();
    params.cu_seqlens_k = cu_seq_k->getDevPtr();

    params.stream = stream;
    params.props = dev_prop;

    params.is_alibi = is_alibi;

    return params;
}

template <typename fp8_t>
void run_mha_decoding_fp8_fwd(const DecodingFP8Params<fp8_t> &params) {
    DECODING_FP8_FWD_HEADDIM_SWITCH(params.d, [&] { run_mha_decoding_fp8_fwd_<HeadDim, fp8_t>(params); });
}

void decoding_attn_fp8e5m2(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O, Tensor<int> *cu_seq_q,
                           Tensor<int> *cu_seq_k, size_t max_seq_q, size_t max_seq_k, bool is_causal, int num_splits,
                           cudaStream_t stream, cudaDeviceProp *dev_prop, bool is_alibi) {
    static DecodingFP8Params<__nv_fp8_e5m2> params = set_mha_decoding_fp8_fwd_params<__nv_fp8_e5m2>(
        Q, K, V, O, cu_seq_q, cu_seq_k, max_seq_q, max_seq_k, stream, dev_prop, is_alibi);
    run_mha_decoding_fp8_fwd<__nv_fp8_e5m2>(params);
    // check_quantization_fp8<__nv_fp8_e5m2>(params, K, V, cu_seq_k);
}

void decoding_attn_fp8e4m3(Tensor<half> *Q, Tensor<half> *K, Tensor<half> *V, Tensor<half> *O, Tensor<int> *cu_seq_q,
                           Tensor<int> *cu_seq_k, size_t max_seq_q, size_t max_seq_k, bool is_causal, int num_splits,
                           cudaStream_t stream, cudaDeviceProp *dev_prop, bool is_alibi) {
    static DecodingFP8Params<__nv_fp8_e4m3> params = set_mha_decoding_fp8_fwd_params<__nv_fp8_e4m3>(
        Q, K, V, O, cu_seq_q, cu_seq_k, max_seq_q, max_seq_k, stream, dev_prop, is_alibi);
    run_mha_decoding_fp8_fwd<__nv_fp8_e4m3>(params);
    // check_quantization_fp8<__nv_fp8_e4m3>(params, K, V, cu_seq_k);
}

#endif
