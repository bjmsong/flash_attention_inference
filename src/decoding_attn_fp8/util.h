// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:17:12 on Sun, Nov 12, 2023
//
// Description: util

#pragma once

#include "decoding_attn_fp8/decoding_fp8.h"
#include "tensor.h"

template <typename fp8_t>
inline void check_quantization_fp8(const DecodingFP8Params<fp8_t> &params, Tensor<half> *K, Tensor<half> *V,
                                   Tensor<int> *cu_seq_k) {
    size_t total_k = K->getShape()[0];
    size_t head_k = K->getShape()[1];
    size_t dim = K->getShape()[2];
    size_t k_elem_num = K->getElemNum();
    half *k_ptr = K->getHostPtr();
    half *v_ptr = V->getHostPtr();
    size_t batch = cu_seq_k->getShape()[0] - 1;
    int *cu_seq_k_ptr = cu_seq_k->getHostPtr();

    fp8_t *k_fp8_ptr = new fp8_t[k_elem_num];
    FAI_CHECK(k_fp8_ptr);
    FAI_CHECK_CUDART_ERROR(cudaMemcpy(k_fp8_ptr, params.k_fp8_ptr, k_elem_num * sizeof(fp8_t), cudaMemcpyDeviceToHost));

    fp8_t *v_fp8_ptr = new fp8_t[k_elem_num];
    FAI_CHECK(v_fp8_ptr);
    FAI_CHECK_CUDART_ERROR(cudaMemcpy(v_fp8_ptr, params.v_fp8_ptr, k_elem_num * sizeof(fp8_t), cudaMemcpyDeviceToHost));

    half *k_dequantization_ptr = new half[k_elem_num];
    FAI_CHECK(k_dequantization_ptr);
    half *v_dequantization_ptr = new half[k_elem_num];
    FAI_CHECK(v_dequantization_ptr);

    double k_max_diff = 0.0;
    double v_max_diff = 0.0;
    double k_avg_diff = 0.0;
    double v_avg_diff = 0.0;
    for (size_t b = 0; b < batch; ++b) {
        size_t sum_seq_k = static_cast<size_t>(cu_seq_k_ptr[b]);
        size_t seq_k = static_cast<size_t>(cu_seq_k_ptr[b + 1]) - sum_seq_k;
        for (size_t h = 0; h < head_k; ++h) {
            for (size_t sk = 0; sk < seq_k; ++sk) {
                for (size_t d = 0; d < dim; ++d) {
                    k_dequantization_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d] =
                        static_cast<half>(k_fp8_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d]);
                    v_dequantization_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d] =
                        static_cast<half>(v_fp8_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d]);

                    double k_diff = static_cast<double>(
                        std::abs(__half2float(k_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d]) -
                                 __half2float(k_dequantization_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d])));
                    double v_diff = static_cast<double>(
                        std::abs(__half2float(v_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d]) -
                                 __half2float(v_dequantization_ptr[(sum_seq_k + sk) * (head_k * dim) + h * dim + d])));

                    k_max_diff = std::max(k_max_diff, k_diff);
                    v_max_diff = std::max(v_max_diff, v_diff);
                    k_avg_diff += k_diff;
                    v_avg_diff += v_diff;
                }
            }
        }
    }

    FLOG("Quantization: k_max_diff: %f, k_avg_diff: %f, v_max_diff: %f, v_avg_diff: %f", k_max_diff,
         k_avg_diff / k_elem_num, v_max_diff, v_avg_diff / k_elem_num);

    if (k_fp8_ptr) {
        delete[] k_fp8_ptr;
        k_fp8_ptr = nullptr;
    }

    if (v_fp8_ptr) {
        delete[] v_fp8_ptr;
        v_fp8_ptr = nullptr;
    }

    if (k_dequantization_ptr) {
        delete[] k_dequantization_ptr;
        k_dequantization_ptr = nullptr;
    }

    if (v_dequantization_ptr) {
        delete[] v_dequantization_ptr;
        v_dequantization_ptr = nullptr;
    }
}
