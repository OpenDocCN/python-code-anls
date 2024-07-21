# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\block_info.h`

```py
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace pytorch_flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    // 构造函数，初始化 BlockInfo 结构体
    __device__ BlockInfo(const Params &params, const int bidb)
        // 初始化 sum_s_q 成员变量，若 Varlen 为 false 或 params.cu_seqlens_q 为 nullptr，则为 -1
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        // 初始化 sum_s_k 成员变量，若 Varlen 为 false 或 params.cu_seqlens_k 为 nullptr 或 !params.is_seqlens_k_cumulative 为 true，则为 -1
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        // 计算 actual_seqlen_q 成员变量，若 Varlen 为 false 或 params.cu_seqlens_q 为 nullptr，则使用 params.seqlen_q
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // 缓存 seqlen_k_cache 成员变量，若 Varlen 为 false 或 params.cu_seqlens_k 为 nullptr，则使用 params.seqlen_k；
        // 若 params.is_seqlens_k_cumulative 为 true，则使用 cu_seqlens_k[bidb + 1] - sum_s_k，否则使用 cu_seqlens_k[bidb]
        , seqlen_k_cache(!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb]))
        // 计算 actual_seqlen_k 成员变量，若 params.seqused_k 不为 nullptr，则使用 params.seqused_k[bidb]，否则使用 seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        {
        }

    template <typename index_t>
    // Q 偏移计算函数，返回 Q 的偏移量
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        // 如果 sum_s_q 等于 -1，则返回 bidb 乘以 batch_stride，否则返回 uint32_t(sum_s_q) 乘以 row_stride
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    // K 偏移计算函数，返回 K 的偏移量
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        // 如果 sum_s_k 等于 -1，则返回 bidb 乘以 batch_stride，否则返回 uint32_t(sum_s_k) 乘以 row_stride
        return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
    }

    const int sum_s_q;       // Q 序列的起始位置或标记
    const int sum_s_k;       // K 序列的起始位置或标记
    const int actual_seqlen_q;   // 实际 Q 序列长度
    const int seqlen_k_cache;    // 缓存的 K 序列长度
    const int actual_seqlen_k;   // 实际 K 序列长度
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace pytorch_flash


这段代码是一个 C++ 的结构体模板 `BlockInfo`，用于表示某些块的信息。它包含了初始化构造函数和两个模板成员函数，用来计算 Q 和 K 序列的偏移量。
```