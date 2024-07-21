# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\flash.h`

```
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack
namespace pytorch_flash {
constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

// 结构体定义：用于存储 QKV 矩阵的指针及其步长信息
struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;   // Q 矩阵的指针
    void *__restrict__ k_ptr;   // K 矩阵的指针
    void *__restrict__ v_ptr;   // V 矩阵的指针

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;     // Q 矩阵批次间的步长
    index_t k_batch_stride;     // K 矩阵批次间的步长
    index_t v_batch_stride;     // V 矩阵批次间的步长
    index_t q_row_stride;       // Q 矩阵行间的步长
    index_t k_row_stride;       // K 矩阵行间的步长
    index_t v_row_stride;       // V 矩阵行间的步长
    index_t q_head_stride;      // Q 矩阵头部间的步长
    index_t k_head_stride;      // K 矩阵头部间的步长
    index_t v_head_stride;      // V 矩阵头部间的步长

    // The number of heads.
    int h, h_k;                 // QKV 头数
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio;            // 多查询和分组查询注意力中，h_k / h 的比率
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 结构体定义：继承自 Qkv_params，包含前向传播的参数
struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;              // 输出矩阵 O 的指针
    void * __restrict__ oaccum_ptr;         // 输出矩阵 O 的累加指针

    // The stride between rows of O.
    index_t o_batch_stride;                 // 输出矩阵 O 批次间的步长
    index_t o_row_stride;                   // 输出矩阵 O 行间的步长
    index_t o_head_stride;                  // 输出矩阵 O 头部间的步长

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;              // P 矩阵的指针

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;    // softmax 和的指针
    void * __restrict__ softmax_lseaccum_ptr; // softmax 累加和的指针

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim; // 各种维度参数

    // The scaling factors for the kernel.
    float scale_softmax;                    // softmax 核的缩放因子
    float scale_softmax_log2;               // softmax 核的缩放因子的对数

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;        // 长度为 b+1 的数组，存储每个 Q 序列的起始偏移量
    int * __restrict__ cu_seqlens_k;        // 长度为 b+1 的数组，存储每个 K 序列的起始偏移量

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;           // 如果提供，每个 K 序列的实际长度

    int *__restrict__ blockmask;            // 块掩码

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;           // K_new 矩阵的指针
    void * __restrict__ vnew_ptr;           // V_new 矩阵的指针

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;              // K_new 矩阵批次间的步长
    index_t vnew_batch_stride;              // V_new 矩阵批次间的步长
    index_t knew_row_stride;                // K_new 矩阵行间的步长
    index_t vnew_row_stride;                // V_new 矩阵行间的步长
    index_t knew_head_stride;               // K_new 矩阵头部间的步长
    index_t vnew_head_stride;               // V_new 矩阵头部间的步长

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;     // 旋转嵌入的余弦矩阵的指针
    void * __restrict__ rotary_sin_ptr;     // 旋转嵌入的正弦矩阵的指针

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;     // 用于索引 KV 缓存的索引

    // Paged KV cache
    // 指向整数数组的指针，使用 __restrict__ 限定符以增加编译器优化的可能性
    int * __restrict__ block_table;
    // 每批次处理的块表的步幅
    index_t block_table_batch_stride;
    // 页面块大小

    int page_block_size;

    // 激活保留的概率（即不进行 dropout 的概率）
    float p_dropout;
    // 激活保留概率的无符号整数表示
    uint8_t p_dropout_in_uint8_t;

    // 概率 1 / (1 - p_dropout) 的比例因子
    float rp_dropout;
    // 缩放 softmax 和 rp_dropout 的比例
    float scale_softmax_rp_dropout;

    // 本地窗口大小
    int window_size_left, window_size_right;

    // 随机状态
    at::PhiloxCudaState philox_args;
    // 额外图偏移量的指针
    int64_t * extragraph_offset;
    // 种子值的指针
    int64_t * seed;

    // 是否使用 bf16 数据类型
    bool is_bf16;
    // 是否因果关系
    bool is_causal;

    // 如果 is_seqlens_k_cumulative 为真，则 seqlen_k 是 cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb]；
    // 否则，它是 cu_seqlens_k[bidb]，即我们使用 cu_seqlens_k 存储 K 的序列长度。
    bool is_seqlens_k_cumulative;

    // 是否使用旋转交错
    bool is_rotary_interleaved;

    // 分裂-KV 版本的分割数
    int num_splits;

    // 指向 Alibi 斜率数据的指针，使用 __restrict__ 限定符以增加编译器优化的可能性
    void * __restrict__ alibi_slopes_ptr;
    // Alibi 斜率批处理步幅
    index_t alibi_slopes_batch_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Flash_bwd_params 继承自 Flash_fwd_params，并添加了额外的成员变量，用于 MHA 反向传播参数设置
struct Flash_bwd_params : public Flash_fwd_params {

    // dO 和 dQKV 矩阵的指针
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // 用于累积 dQ 的指针
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr; // 用于累积 dK 的指针
    void *__restrict__ dv_accum_ptr; // 用于累积 dV 的指针

    // // 用于在 seqlen_q 维度上分割反向传播时累积 dK 和 dV 的指针
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // dO, dQ, dK 和 dV 矩阵行之间的步长
    // TD [2022-04-16]: 我们使用 32 位索引以节省寄存器。
    // 代码可能无法处理大于 2GB 的数组。
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // softmax d 和的指针
    void *__restrict__ dsoftmax_sum;

    // 是否确定性操作的标志
    bool deterministic;

    // dQ 累积的分割步长
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 模板函数声明，用于运行 MHA 前向传播
template<typename T, int Headdim> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

// 模板函数声明，用于根据是否分割 KV 运行 MHA 前向传播
template<typename T, int Headdim> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

// 模板函数声明，用于运行 MHA 反向传播
template<typename T, int Headdim> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);

// 命名空间结束符
} // namespace pytorch_flash
```