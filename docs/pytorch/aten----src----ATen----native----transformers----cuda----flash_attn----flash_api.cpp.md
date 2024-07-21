# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\flash_api.cpp`

```
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <cstdint>
#include <tuple>


#ifdef USE_FLASH_ATTENTION
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/zeros.h>
#endif


#include <cutlass/numeric_types.h>

#include <ATen/native/transformers/cuda/flash_attn/flash.h>
#include <ATen/native/transformers/cuda/flash_attn/flash_api.h>
#include <ATen/native/transformers/cuda/flash_attn/static_switch.h>

#include <c10/util/Exception.h>


namespace pytorch_flash {

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == at::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool seqlenq_ngroups_swapped=false) {

    // Reset the parameters
    params = {};

    // Determine if q has type BFloat16
    params.is_bf16 = q.dtype() == at::kBFloat16;

    // Set the pointers and strides for q, k, and v tensors
    params.q_ptr = q.data_ptr();                      // Pointer to q tensor data
    params.k_ptr = k.data_ptr();                      // Pointer to k tensor data
    params.v_ptr = v.data_ptr();                      // Pointer to v tensor data

    // Set row strides (in elements, not bytes) for q, k, and v tensors
    params.q_row_stride = q.stride(-3);               // Row stride for q tensor
    params.k_row_stride = k.stride(-3);               // Row stride for k tensor
    params.v_row_stride = v.stride(-3);               // Row stride for v tensor

    // Set head stride (in elements) for q tensor
    params.q_head_stride = q.stride(-2);              // Head stride for q tensor


This block of code sets up parameters for a forward pass operation in a custom Flash Attention implementation. It initializes and configures various pointers and strides necessary for efficient computation on CUDA devices.
    // 设置 params 结构中的 k_head_stride 为 k 张量在倒数第二维上的步长
    params.k_head_stride = k.stride(-2);
    // 设置 params 结构中的 v_head_stride 为 v 张量在倒数第二维上的步长
    params.v_head_stride = v.stride(-2);
    // 设置 params 结构中的 o_ptr 为 out 张量的数据指针
    params.o_ptr = out.data_ptr();
    // 设置 params 结构中的 o_row_stride 为 out 张量在倒数第三维上的步长
    params.o_row_stride = out.stride(-3);
    // 设置 params 结构中的 o_head_stride 为 out 张量在倒数第二维上的步长
    params.o_head_stride = out.stride(-2);

    // 如果 cu_seqlens_q_d 为 nullptr，则设置 params 结构中的 q_batch_stride、k_batch_stride、v_batch_stride、o_batch_stride
    if (cu_seqlens_q_d == nullptr) {
        // 设置 q_batch_stride 为 q 张量在第0维上的步长
        params.q_batch_stride = q.stride(0);
        // 设置 k_batch_stride 为 k 张量在第0维上的步长
        params.k_batch_stride = k.stride(0);
        // 设置 v_batch_stride 为 v 张量在第0维上的步长
        params.v_batch_stride = v.stride(0);
        // 设置 o_batch_stride 为 out 张量在第0维上的步长
        params.o_batch_stride = out.stride(0);
        // 如果 seqlenq_ngroups_swapped 为真，则乘以 seqlen_q 对应的值来调整 q_batch_stride 和 o_batch_stride
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    // 将 cu_seqlens_q_d、cu_seqlens_k_d、seqused_k 强制转换为 int 指针，并分别存入 params 结构
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // 设置 params 结构中的 p_ptr 为 p_d 指针，表示 softmax 结果的指针
    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // 设置 params 结构中的 softmax_lse_ptr 为 softmax_lse_d，表示 softmax 的 sum 值的指针
    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // 设置 params 结构中的 b、h、h_k、h_h_k_ratio、seqlen_q、seqlen_k、seqlen_q_rounded、seqlen_k_rounded、d、d_rounded 分别为对应的维度值
    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // 设置 params 结构中的 scale_softmax、scale_softmax_log2 分别为 softmax_scale 和 softmax_scale * M_LOG2E 的值
    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // 设置 params 结构中的 p_dropout、p_dropout_in_uint8_t、rp_dropout、scale_softmax_rp_dropout 分别为对应的概率或比例值
    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // 检查概率 p_dropout 是否小于 1，确保概率范围合理
    TORCH_CHECK(p_dropout < 1.f);

    // 如果 FLASHATTENTION_DISABLE_DROPOUT 宏定义被定义，则进一步检查 p_dropout 是否为 0，如果不支持 dropout 则报错
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // 根据 window_size_left 和 window_size_right 的值设置 params 结构中的 is_causal、window_size_left、window_size_right 的值
    // Causal 是 window_size_right == 0 且 window_size_left < 0 的特殊情况
    // Local 是更一般的情况，其中 window_size_right >= 0 或 window_size_left >= 0
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    // 如果 FLASHATTENTION_DISABLE_LOCAL 宏定义被定义，则检查是否支持本地 attention，如果不支持则报错
    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif
    # 设置参数，表示序列长度 k 是否累积
    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        #ifdef 指令：检查是否禁用不均匀的 k
        #TORCH_CHECK 函数：如果条件不满足，则抛出错误，要求 headdim 必须是 32 的倍数
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif
void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,                            // 批量大小
                      const size_t seqlen_q,                     // 查询序列长度
                      const size_t seqlen_k,                     // 键序列长度
                      const size_t seqlen_q_rounded,             // 舍入后的查询序列长度
                      const size_t seqlen_k_rounded,             // 舍入后的键序列长度
                      const size_t h,                            // 头数
                      const size_t h_k,                          // 键头数
                      const size_t d,                            // 维度
                      const size_t d_rounded,                    // 舍入后的维度
                      // device pointers
                      const at::Tensor q,                        // 查询张量
                      const at::Tensor k,                        // 键张量
                      const at::Tensor v,                        // 值张量
                      const at::Tensor out,                      // 输出张量
                      const at::Tensor dout,                     // 输出梯度张量
                      at::Tensor dq,                             // 查询梯度张量
                      at::Tensor dk,                             // 键梯度张量
                      at::Tensor dv,                             // 值梯度张量
                      void *cu_seqlens_q_d,                      // 查询序列长度CUDA指针
                      void *cu_seqlens_k_d,                      // 键序列长度CUDA指针
                      void *dq_accum_d,                          // 查询梯度累积CUDA指针
                      void *dk_accum_d,                          // 键梯度累积CUDA指针
                      void *dv_accum_d,                          // 值梯度累积CUDA指针
                      void *softmax_lse_d,                       // Softmax的log-sum-exp值CUDA指针
                      void *dsoftmax_sum_d,                      // Softmax梯度的累积CUDA指针
                      float p_dropout,                           // Dropout概率
                      float softmax_scale,                       // Softmax缩放因子
                      int window_size_left,                      // 左侧窗口大小
                      int window_size_right,                     // 右侧窗口大小
                      bool deterministic) {                      // 是否确定性操作

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right);

    // 设置指针和步长。
    params.do_ptr = dout.data_ptr();                         // 输出梯度张量的数据指针
    params.do_row_stride = dout.stride(-3);                  // 输出梯度张量在行方向的步长
    params.do_head_stride = dout.stride(-2);                 // 输出梯度张量在头数方向的步长
    params.dq_ptr = dq.data_ptr();                           // 查询梯度张量的数据指针
    params.dk_ptr = dk.data_ptr();                           // 键梯度张量的数据指针
    params.dv_ptr = dv.data_ptr();                           // 值梯度张量的数据指针
    params.dq_row_stride = dq.stride(-3);                    // 查询梯度张量在行方向的步长
    params.dk_row_stride = dk.stride(-3);                    // 键梯度张量在行方向的步长
    params.dv_row_stride = dv.stride(-3);                    // 值梯度张量在行方向的步长
    params.dq_head_stride = dq.stride(-2);                   // 查询梯度张量在头数方向的步长
    params.dk_head_stride = dk.stride(-2);                   // 键梯度张量在头数方向的步长
    params.dv_head_stride = dv.stride(-2);                   // 值梯度张量在头数方向的步长

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);             // 如果查询序列长度CUDA指针为空，设置输出梯度张量的批量步长
        params.dq_batch_stride = dq.stride(0);               // 设置查询梯度张量的批量步长
        params.dk_batch_stride = dk.stride(0);               // 设置键梯度张量的批量步长
        params.dv_batch_stride = dv.stride(0);               // 设置值梯度张量的批量步长
    }

    params.dq_accum_ptr = dq_accum_d;                        // 设置查询梯度累积CUDA指针
    params.dk_accum_ptr = dk_accum_d;                        // 设置键梯度累积CUDA指针
    params.dv_accum_ptr = dv_accum_d;                        // 设置值梯度累积CUDA指针

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;                    // 设置Softmax梯度的累积CUDA指针

    params.deterministic = deterministic;                    // 设置是否为确定性操作
}
    // 根据 !params.is_bf16 条件开启 FP16_SWITCH 宏定义的代码块
    FP16_SWITCH(!params.is_bf16, [&] {
        // 根据 HEADDIM_SWITCH 宏定义的代码块，处理维度参数 params.d
        HEADDIM_SWITCH(params.d, [&] {
            // 检查是否不需要分裂并且没有强制分裂内核的要求
            if (params.num_splits <= 1 && !force_split_kernel) {
                // 如果 num_splits <= 1 且没有强制分裂内核的要求，则运行非分裂版本的 MHA 前向传播
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
                // 否则，根据参数调用适当的分裂键值操作分发函数
                run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
            }
        });
    });
// Find the number of splits that maximizes the occupancy.
// For example, if we have batch * n_heads = 48 and we have 108 SMs,
// having 2 splits (efficiency = 0.89) is better than having 3 splits (efficiency = 0.67).
// However, we also avoid too many splits to minimize HBM reads/writes.
// Thus, we find the best efficiency and choose the smallest number of splits
// that achieves at least 85% of this efficiency.

inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If batch_nheads_mblocks can almost fill 80% of the SMs, use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }

    // Limit max_splits to the smallest of max_splits, num_SMs, and num_n_blocks
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});

    // Initialize variables
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);

    // Lambda function to calculate ceiling division
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    // Lambda function to check if a split is eligible based on block distribution
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };

    // Calculate efficiencies for each split configuration
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);  // Not eligible, efficiency is 0
        } else {
            // Calculate number of waves and efficiency
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            if (eff > max_efficiency) { max_efficiency = eff; }  // Track maximum efficiency
            efficiency.push_back(eff);
        }
    }

    // Choose the optimal number of splits that meets the efficiency criteria
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            return num_splits;  // Return the number of splits chosen
        }
    }

    return 1;  // Default to 1 split if no suitable split is found
}

void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, cudaDeviceProp *dprops, struct c10::TensorOptions opts) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Number of m blocks based on max_seqlen_q and fixed block size of 64
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;

    // Set the number of splits in the parameters
    params.num_splits = num_splits;
}
    // 如果 p_dropout 等于 0.0f，则表示不使用 dropout，因此不需要进行 SplitKV 的实现
    if (p_dropout == 0.0f) {
        // 如果 num_splits 小于 1，则使用启发式方法估算 num_splits
        if (num_splits < 1) {
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount, num_n_blocks, 128);
        }
        // 如果计算得到的 num_splits 大于 1，则创建 softmax_lse_accum 和 out_accum 的张量
        if (params.num_splits > 1) {
            at::Tensor softmax_lse_accum = at::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            at::Tensor out_accum = at::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            // 将张量的数据指针保存到相应的参数中
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        // 检查 num_splits 是否小于等于 128，否则抛出异常
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }
}

// 设置 FlashAttention 的参数和 ALiBi 斜率（如果可选）
void set_params_alibi(Flash_fwd_params &params, std::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    // 检查是否禁用 ALiBi 斜率选项
    TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
    // 将参数中的 ALiBi 斜率指针设为 nullptr
    params.alibi_slopes_ptr = nullptr;
#else
    // 如果传入了 ALiBi 斜率
    if (alibi_slopes_.has_value()) {
        // 获取 ALiBi 斜率张量
        auto alibi_slopes = alibi_slopes_.value();
        // 检查 ALiBi 斜率张量的数据类型必须为 float32
        TORCH_CHECK(alibi_slopes.dtype() == at::kFloat, "ALiBi slopes must have dtype fp32");
        // 检查 ALiBi 斜率张量的设备与参数的一致性
        CHECK_DEVICE(alibi_slopes);
        // 检查 ALiBi 斜率张量的最后一个维度是否是连续的
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        // 检查 ALiBi 斜率张量的尺寸是否符合预期
        TORCH_CHECK(alibi_slopes.sizes() == at::IntArrayRef({num_heads}) || alibi_slopes.sizes() == at::IntArrayRef({batch_size, num_heads}));
        // 将 FlashAttention 参数中的 ALiBi 斜率指针设为 ALiBi 斜率张量的数据指针
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        // 如果 ALiBi 斜率张量是二维的，则设置批次步长
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        // 如果未传入 ALiBi 斜率，则将参数中的 ALiBi 斜率指针设为 nullptr
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

// 执行多头注意力机制的前向传播，返回多个张量作为结果
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(const at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,       // dropout 概率
        const float softmax_scale,   // softmax 缩放因子
        bool is_causal,              // 是否是因果注意力
        int window_size_left,        // 左侧窗口大小
        int window_size_right,       // 右侧窗口大小
        const bool return_softmax,   // 是否返回 softmax 结果
        std::optional<at::Generator> gen_) {   // 可选的随机数生成器

    auto dprops = at::cuda::getCurrentDeviceProperties();
    // 检查当前设备是否为 Ampere 架构或更新版本
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // 在不久的将来，我们将支持图灵架构
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    auto q_dtype = q.dtype();
    // 检查查询张量的数据类型必须是 half 或者 bfloat16
    TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    // 如果查询张量的数据类型是 bfloat16，则要求设备为 Ampere 架构或更新版本
    if (q_dtype == at::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    // 检查查询张量、键张量、值张量的数据类型必须一致
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    // 检查输入张量的最后一个维度是否是连续的
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // 检查输入张量 k 的最后一个维度是否为 1，要求其为连续的
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // 检查输入张量 v 的最后一个维度是否为 1，要求其为连续的
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    // 获取输入张量 q 的尺寸信息
    const auto sizes = q.sizes();

    // 解构尺寸信息
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    // 检查批次大小是否为正数
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    // 检查头部尺寸是否是 8 的倍数，这通过填充保证
    TORCH_CHECK(head_size_og % 8 == 0, "head_size must be a multiple of 8, this is ensured by padding!");
    // 检查头部尺寸是否不超过 256
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    // 检查在键/值输入中的头部数目是否能整除查询输入中的头部数目
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // 如果窗口左侧大小大于等于键的序列长度，则将窗口左侧大小设置为 -1
    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    // 如果窗口右侧大小大于等于键的序列长度，则将窗口右侧大小设置为 -1
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // 如果查询的序列长度为 1 并且没有斜率调整的值，将 is_causal 设置为 false
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    // 如果 is_causal 为真，则将窗口右侧大小设置为 0
    if (is_causal) { window_size_right = 0; }

    // 在这种情况下，从 (b, 1, (nheads_kv ngroups), d) 转置 q 为 (b, ngroups, nheads_kv, d) 更快
    // 感谢 Daniel Haziza 提供的方法
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    at::Tensor temp_q = q;
    if (seqlenq_ngroups_swapped) {
        // 计算新的 ngroups
        const int ngroups = num_heads / num_heads_k;
        // 重塑并转置查询张量 q
        temp_q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    // 检查 temp_q 的形状是否正确
    CHECK_SHAPE(temp_q, batch_size, seqlen_q, num_heads, head_size_og);
    // 检查 k 的形状是否正确
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    // 检查 v 的形状是否正确
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

    // 创建用于输出的张量 q_padded, k_padded, v_padded
    at::Tensor q_padded, k_padded, v_padded;
    q_padded = temp_q;
    k_padded = k;
    v_padded = v;

    // 创建用于输出的张量 out
    at::Tensor out;
    // 如果已经有输出张量 out_，则使用其值
    if (out_.has_value()) {
        out = out_.value();
        // 检查输出张量的数据类型是否与输入张量 q 的数据类型相同
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        // 检查输出张量的设备类型
        CHECK_DEVICE(out);
        // 检查输出张量的最后一个维度是否为 1，要求其为连续的
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        // 检查输出张量的形状是否正确
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        // 如果头部尺寸不是 8 的倍数，则创建一个与 q_padded 形状相同的空张量作为输出
        if (head_size_og % 8 != 0) { out = at::empty_like(q_padded); }
    } else {
        // 否则，创建一个与 q_padded 形状相同的空张量作为输出
        out = at::empty_like(q_padded);
    }

    // 定义一个函数，用于将 x 取整为 m 的倍数
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // 计算头部尺寸的 8 的倍数
    const int head_size = round_multiple(head_size_og, 8);
    // 计算头部尺寸的 32 的倍数
    const int head_size_rounded = round_multiple(head_size, 32);
    // 计算查询序列长度的 128 的倍数
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    // 计算键的序列长度的 128 的倍数
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // 否则，将从 cuda:0 设备启动内核
    // 将 q.get_device() 的返回值强制转换为 char 类型，以避免编译器在窄化类型转换时发出警告
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // 获取张量的选项
    auto opts = q.options();

    // 创建一个空的张量 softmax_lse，形状为 {batch_size, num_heads, seqlen_q}，使用与 opts 相同的数据类型
    auto softmax_lse = at::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;

    // 如果需要返回 softmax 结果并且设置了 dropout 以减少编译时间
    if (return_softmax) {
        // 检查 p_dropout 必须大于 0.0，否则抛出异常
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        // 创建一个空的张量 p，形状为 {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}，使用 opts 选项
        p = at::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    // 定义 Flash_fwd_params 结构体变量 params
    Flash_fwd_params params;

    // 设置前向传播参数
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right);

    // 设置分割键值对参数
    set_params_splitkv(params, batch_size, num_heads,
                       head_size, seqlen_k, seqlen_q,
                       head_size_rounded, p_dropout, /*num_splits*/0, dprops, opts);

    // 如果需要在反向传播时保存随机数生成器状态，则获取种子和偏移量张量
    // seed_t 和 offset_t 将在后向函数中使用
    at::Tensor seed_t, offset_t;
    // 如果 dropout 概率大于 0.0，则需要生成随机数，以下是相关处理逻辑
    if (p_dropout > 0.0)  {
        // 获取默认的 CUDA 生成器
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
        // 计算每个线程生成随机数的次数，以调整 Philox 计数器在 THC 随机状态中的偏移量
        int64_t counter_offset = params.b * params.h * 32;
        // 获取随机数生成器时需要获取锁
        std::lock_guard<std::mutex> lock(gen->mutex_);
        // 根据偏移量获取 Philox CUDA 状态
        at::PhiloxCudaState philox_state = gen->philox_cuda_state(counter_offset);
        // 如果当前没有捕获 CUDA 流，则使用 Philox 状态中的种子和偏移量初始化 seed_t 和 offset_t
        if (at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None) {
          auto [seed, offset] = at::cuda::philox::unpack(philox_state);
          seed_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(seed)), at::dtype(at::kLong));
          offset_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(offset)), at::dtype(at::kLong));
        } else {
          // 否则，在 CUDA 设备上创建空的 seed_t 和 offset_t 张量，并将指针分配给 params 的对应成员变量
          seed_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
          offset_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
          params.seed = seed_t.data_ptr<int64_t>();
          params.extragraph_offset = offset_t.data_ptr<int64_t>();
        }
        // 将 Philox 状态保存到 params 的 philox_args 中
        params.philox_args = philox_state;
    } else {
        // 如果 dropout 概率为 0，则不需要生成随机数，初始化 seed_t 和 offset_t 张量
        if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
            seed_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
            offset_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
        } else {
            seed_t = at::empty({}, at::dtype(at::kLong));
            offset_t = at::empty({}, at::dtype(at::kLong));
        }
    }

    // 设置模型参数和 alibi_slopes_ 相关参数
    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // 如果 seqlen_k 大于 0，则在当前 CUDA 流上运行多头注意力的前向传播
    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // 如果 seqlen_k == 0，说明输入张量为空，需要将输出置为 0
        out.zero_();
        // 同时将 softmax_lse 填充为正无穷大
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // 如果 seqlenq_ngroups_swapped 为真，则需要对输出 out、q_padded 和 softmax_lse 进行形状调整和转置
    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        q_padded = q_padded.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    // 返回计算结果和相关参数
    return {out, q_padded, k_padded, v_padded, softmax_lse, seed_t, offset_t, p};
    // 定义函数 mha_varlen_fwd，返回一个包含八个张量的元组
    // q: 查询张量，形状为 total_q x num_heads x head_size，total_q 表示所有序列的总长度
    // k: 键张量，形状为 total_k x num_heads_k x head_size，total_k 表示所有序列的总长度
    // v: 值张量，形状同 k
    // out_: 可选张量，形状为 total_q x num_heads x head_size，若提供则为输出张量
    // cu_seqlens_q: 包含当前查询序列长度的张量，长度为 b+1，b 表示批次大小
    // cu_seqlens_k: 包含当前键序列长度的张量，长度为 b+1
    // seqused_k: 可选张量，长度为 b，若提供则限制每批次元素的键使用数量
    // alibi_slopes_: 可选张量，形状为 num_heads 或 b x num_heads，若提供则是斜率参数
    // max_seqlen_q: 当前查询序列的最大长度
    // max_seqlen_k: 当前键序列的最大长度
    // p_dropout: 丢弃率，用于注意力计算中的随机丢弃
    // softmax_scale: Softmax 缩放因子，用于调整注意力分布的稳定性
    // zero_tensors: 是否将输出张量初始化为零
    // is_causal: 是否为因果注意力机制
    // window_size_left: 左侧窗口大小，用于局部注意力机制
    // window_size_right: 右侧窗口大小，用于局部注意力机制
    // return_softmax: 是否返回 Softmax 结果
    // gen_: 可选随机数生成器

    // 获取当前 CUDA 设备属性
    auto dprops = at::cuda::getCurrentDeviceProperties();
    // 检查是否为 Ampere 架构或更新版本的 GPU
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // 暂时不支持 Turing 架构
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    // 检查查询张量的数据类型，仅支持 fp16 和 bf16
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    // 若数据类型为 bf16，则要求设备为 Ampere 架构或更新版本
    if (q_dtype == at::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    // 检查键和值张量的数据类型必须与查询张量相同
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    // 检查 cu_seqlens_q 和 cu_seqlens_k 的数据类型必须为 int32
    TORCH_CHECK(cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must have dtype int32");

    // 检查 q, k, v 张量是否在 GPU 上
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    // 检查 cu_seqlens_q, cu_seqlens_k 张量是否在 GPU 上
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    // 检查 q, k, v 张量最后一维是否连续
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // 检查 cu_seqlens_q, cu_seqlens_k 张量是否连续
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    // 获取查询张量的尺寸信息
    const auto sizes = q.sizes();
    // 计算批次大小
    const int batch_size = cu_seqlens_q.numel() - 1;
    // 获取头数
    int num_heads = sizes[1];
    // 从 sizes 数组中获取第三个元素作为 head_size_og 的初始值
    const int head_size_og = sizes[2];
    // 获取张量 k 的第一维大小作为 total_k
    const int total_k = k.size(0);
    // 获取张量 k 的第二维大小作为 num_heads_k
    const int num_heads_k = k.size(1);

    // 如果 max_seqlen_q 为 1 并且 alibi_slopes_ 没有值，则将 is_causal 置为 false
    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case
    // 如果 is_causal 为 true，则将 window_size_right 置为 0
    if (is_causal) { window_size_right = 0; }

    // 获取 cu_seqlens_q 的数据指针
    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    // 根据给定的条件，判断是否进行 q 张量的维度转置优化
    // 在这种情况下，将 q 从 (b, 1, (nheads_kv ngroups), d) 转置为 (b, ngroups, nheads_kv, d) 更快
    // 提示来自 Daniel Haziza
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    at::Tensor temp_q = q;
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        temp_q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size_og});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q_d = nullptr;
    }

    // 获取转置后的 q 张量的总大小
    const int total_q = temp_q.sizes()[0];

    // 检查 batch_size 必须大于 0
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    // 检查 head_size_og 必须小于等于 256
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    // 检查 num_heads 必须能被 num_heads_k 整除
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    // 检查 head_size_og 必须是 8 的倍数，这由填充确保！

    // 如果 window_size_left 大于等于 max_seqlen_k，则将其置为 -1
    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    // 如果 window_size_right 大于等于 max_seqlen_k，则将其置为 -1
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    // 检查张量的形状是否符合预期
    CHECK_SHAPE(temp_q, total_q, num_heads, head_size_og);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    // 如果 seqused_k 有值，则进一步检查其形状和属性
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == at::kInt, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    // 初始化 q_padded, k_padded, v_padded 张量，并赋予初始值
    at::Tensor q_padded, k_padded, v_padded;
    q_padded = temp_q;
    k_padded = k;
    v_padded = v;

    // 初始化 out 张量，根据是否已有预分配的 out_ 张量进行赋值
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size_og);
        // 如果 head_size_og 不是 8 的倍数，则重新分配一个与 q_padded 相同形状的 out 张量
        if (head_size_og % 8 != 0) { out = at::empty_like(q_padded); }
    } else {
        // 如果没有预分配的 out_ 张量，则创建一个与 q_padded 相同形状的新张量
        out = at::empty_like(q_padded);
    }

    // 定义一个函数 round_multiple，用于将 x 向上取整到 m 的倍数
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // 将 head_size 舍入为 8 的倍数
    const int head_size = round_multiple(head_size_og, 8);
    // 将 head_size_rounded 舍入为 32 的倍数
    const int head_size_rounded = round_multiple(head_size, 32);
    // 将 seqlen_q_rounded 舍入为 128 的倍数
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    // 将 seqlen_k_rounded 舍入为 128 的倍数
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // 设置 CUDA 设备为张量 q 的设备，避免从 cuda:0 设备启动内核
    // 将 q.get_device() 强制转换为 char 类型以避免编译器关于缩小转换的警告
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // 获取张量 q 的选项
    auto opts = q.options();

    // 创建一个形状为 {batch_size, num_heads, max_seqlen_q} 的空张量 softmax_lse
    // 使用 opts 指定的数据类型（默认为 float）
    auto softmax_lse = at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // 如果 return_softmax 为 true，则创建一个形状为 {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded} 的空张量 p
    // 这样可以减少编译时间
    if (return_softmax) {
        // 检查 p_dropout 大于 0.0，因为只有在 dropout 概率大于 0 时才支持返回 softmax
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = at::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    // 如果 zero_tensors 为 true，则将 out 张量置零，将 softmax_lse 张量填充为负无穷
    // 如果 return_softmax 为 true，则也将 p 张量置零
    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }

    // 创建 Flash_fwd_params 结构体对象 params
    Flash_fwd_params params;
    // 设置 params 中的参数，用于前向传播
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k.data_ptr(),
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     seqlenq_ngroups_swapped);
    
    // 如果 seqlenq_ngroups_swapped 为 true，则只在解码时应用 split-k
    // 设置 params 中的分割 kv 参数
    if (seqlenq_ngroups_swapped) {
        set_params_splitkv(params, batch_size, num_heads,
                           head_size, max_seqlen_k, max_seqlen_q,
                           head_size_rounded, p_dropout, /*num_splits*/0, dprops, opts);
    }

    // 检查是否要在反向传播时检查点和保存 RNG 状态（如果使用了 dropout）
    // 获取默认生成器 gen，并返回用于反向传播函数的种子和偏移量
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    at::Tensor seed_t, offset_t;
    // 如果 dropout 概率大于 0，则需要生成随机数以进行数据随机丢弃
    if (p_dropout > 0.0)  {
        // 计算每个线程生成随机数的次数，以便调整 thc 随机状态的 philox 计数器
        int64_t counter_offset = params.b * params.h * 32;
        // 获取随机数生成器的互斥锁，确保线程安全
        std::lock_guard<std::mutex> lock(gen->mutex_);
        // 根据偏移量创建 PhiloxCudaState 状态
        at::PhiloxCudaState philox_state = gen->philox_cuda_state(counter_offset);
        // 检查当前 CUDA 流的捕获状态，如果没有捕获状态，则从 Philox 状态中提取种子和偏移量
        if (at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None) {
          auto [seed, offset] = at::cuda::philox::unpack(philox_state);
          seed_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(seed)), at::dtype(at::kLong));
          offset_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(offset)), at::dtype(at::kLong));
        } else {
          // 如果当前有捕获状态，则创建一个空的 CUDA 张量以保存种子和偏移量
          seed_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
          offset_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
          // 将种子和偏移量的数据指针存储到 params 结构中
          params.seed = seed_t.data_ptr<int64_t>();
          params.extragraph_offset = offset_t.data_ptr<int64_t>();
        }
        // 将 PhiloxCudaState 参数保存到 params 结构中
        params.philox_args = philox_state;
    } else {
        // 如果 dropout 概率为 0，则不需要生成随机数
        if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
            // 如果当前有捕获状态，则创建空的 CUDA 张量以保存种子和偏移量
            seed_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
            offset_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
        } else {
            // 否则，创建空的 CPU 张量以保存种子和偏移量
            seed_t = at::empty({}, at::dtype(at::kLong));
            offset_t = at::empty({}, at::dtype(at::kLong));
        }
    }

    // 设置模型参数的 "假名" 参数，用于解释参数的保护
    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // 如果最大序列长度大于 0，则在当前 CUDA 流上执行多头自注意力前向传播
    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // 如果最大序列长度为 0，则输出张量为空。需要将输出张量和 softmax_lse 张量填充为无穷大
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // 如果交换了序列长度分组，则重塑输出张量、填充的查询张量和 softmax_lse 张量的形状
    if (seqlenq_ngroups_swapped) {
        std::array<int64_t, 4> size_before = {batch_size, max_seqlen_q, num_heads_k, head_size_og};
        std::array<int64_t, 3> size_after = {batch_size, num_heads_k * max_seqlen_q, head_size_og};
        out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
        q_padded = q_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * max_seqlen_q, 1});
    }

    // 返回计算后的张量和参数
    return {out, q_padded, k_padded, v_padded, softmax_lse, seed_t, offset_t, p};
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    // 如果不是 bf16 数据类型，则执行以下逻辑
    FP16_SWITCH(!params.is_bf16, [&] {
        // 根据头部维度选择执行不同的处理逻辑
        HEADDIM_SWITCH(params.d, [&] {
            // 调用具体的多头自注意力反向传播函数
            run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const bool deterministic,
        const at::Tensor philox_seed,
        const at::Tensor philox_offset) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        // 如果编译时禁用了反向传播，则抛出错误信息
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    // 如果是因果注意力，将右侧窗口大小设置为0
    if (is_causal) { window_size_right = 0; }
    // 获取当前 CUDA 设备的属性
    auto dprops = at::cuda::getCurrentDeviceProperties();
    // 检查是否为 Ampere 架构或更新版本的 GPU
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // 检查是否启用了 dropout
    bool is_dropout = p_dropout > 0.0;
    // 获取当前 CUDA 流
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // 检查查询张量的数据类型，只支持 fp16 和 bf16
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    // 如果查询张量是 bf16 类型，要求当前 GPU 架构为 Ampere 或更新版本
    if (q_dtype == at::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    // 检查键和值张量与查询张量的数据类型相同
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    // 检查查询、键、值张量是否位于 CUDA 设备上
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    // 检查输出张量、导数张量和 softmax_lse 是否在正确的设备上
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    // 检查输入张量 q, k, v, out, dout 是否在最后一个维度上是连续的
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    // 获取输入张量 q 的尺寸
    const auto sizes = q.sizes();

    // 获取尺寸中的具体数值
    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = dout.size(3);
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    // 检查批大小是否为正数
    TORCH_CHECK(batch_size > 0, "batch size must be positive");

    // 检查头部大小是否是 8 的倍数
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");

    // 检查头部原始大小是否是 8 的倍数，通过填充保证
    TORCH_CHECK(head_size_og % 8 == 0, "head_size_og should be a multiple of 8, this is ensured by padding!");

    // 检查头部大小是否不超过 256
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");

    // 如果头部大小大于 192 并且符合特定条件，需检查是否满足硬件要求
    if (head_size > 192 && (head_size <= 224 || is_dropout)) {
        TORCH_CHECK(is_sm80 || is_sm90, "FlashAttention backward for head dim 256 with dropout, or head dim 224 with/without dropout requires A100/A800 or H100/H800");
    }

    // 检查 key/value 中的头部数量是否整除查询中的头部数量
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // 定义一个函数，将 x 向上取整到 m 的倍数
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

    // 将头部大小舍入到最接近的 32 的倍数
    const int head_size_rounded = round_multiple(head_size, 32);

    // 将查询长度舍入到最接近的 128 的倍数
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);

    // 将 key 长度舍入到最接近的 128 的倍数
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // 检查头部大小是否等于头部原始大小 head_size_og 舍入到 8 的倍数
    TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    // 如果窗口大小超过了 key 的长度，将其设为 -1
    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // 检查张量的形状是否符合预期：q, k, v, out, dout
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

    // 创建 dq, dk, dv 张量，如果 dq_ 有值则使用其值，否则创建一个与 q 相同的空张量
    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = at::empty_like(q);
    }
    // 如果 dk_ 有值，则将其赋给 dk，否则创建一个与 k 相同形状的空 Tensor
    if (dk_.has_value()) {
        dk = dk_.value();
        // 检查 dk 的数据类型必须与 q 的数据类型相同
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        // 检查 dk 在正确的设备上
        CHECK_DEVICE(dk);
        // 检查 dk 的最后一个维度必须是连续的
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        // 检查 dk 的形状必须符合指定的要求
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        // 如果 dk_ 没有值，则创建一个与 k 形状相同的空 Tensor
        dk = at::empty_like(k);
    }

    // 如果 dv_ 有值，则将其赋给 dv，否则创建一个与 v 相同形状的空 Tensor
    if (dv_.has_value()) {
        dv = dv_.value();
        // 检查 dv 的数据类型必须与 q 的数据类型相同
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        // 检查 dv 在正确的设备上
        CHECK_DEVICE(dv);
        // 检查 dv 的最后一个维度必须是连续的
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        // 检查 dv 的形状必须符合指定的要求
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        // 如果 dv_ 没有值，则创建一个与 v 形状相同的空 Tensor
        dv = at::empty_like(v);
    }

    // 设置循环标志为 true，这是一个临时的简化，后续需要修改
    bool loop = true;

    // 将 q 的设备设置为当前设备，避免在不同设备上启动内核
    // 使用 char 类型进行强制转换，避免编译器关于窄化的警告
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // 创建一个与 q 具有相同选项的空 Tensor 作为 softmax_d
    auto opts = q.options();
    auto softmax_d = at::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));

    // 初始化 dq_accum、dk_accum 和 dv_accum
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    // 如果 loop 为 true
    if (loop) {
        // 如果不是确定性计算，则创建一个空的 dq_accum
        if (!deterministic) {
            dq_accum = at::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            // 否则根据计算能力和批次大小等参数创建 dq_accum
            const int nsplits = (dprops->multiProcessorCount + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = at::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
        // 暂时注释掉的代码段，后续需要处理 dk_accum 和 dv_accum
        // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        // dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    }

    // 初始化 dk_expanded 和 dv_expanded
    at::Tensor dk_expanded, dv_expanded;
    // 如果 num_heads_k 不等于 num_heads，即 MQA 或 GQA
    if (num_heads_k != num_heads) {
        // 则分配新的形状以扩展 dk 和 dv
        dk_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
        dv_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    } else {
        // 否则直接使用已有的 dk 和 dv
        dk_expanded = dk;
        dv_expanded = dv;
    }

    // 初始化 Flash_bwd_params 结构体实例 params
    Flash_bwd_params params;
    // 调用函数设置梯度计算的参数，涵盖了多个输入参数，如批量大小、序列长度等
    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout, dq, dk_expanded, dv_expanded,
                     nullptr,
                     nullptr,
                     loop ? dq_accum.data_ptr() : nullptr,  // 如果循环为真，使用dq_accum的数据指针，否则为nullptr
                     // loop ? dk_accum.data_ptr() : nullptr, // 暂时注释掉的dk_accum数据指针设置
                     // loop ? dv_accum.data_ptr() : nullptr, // 暂时注释掉的dv_accum数据指针设置
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),  // softmax_lse的数据指针
                     softmax_d.data_ptr(),    // softmax_d的数据指针
                     p_dropout,               // dropout概率
                     softmax_scale,           // softmax缩放因子
                     window_size_left,        // 窗口左侧大小
                     window_size_right,       // 窗口右侧大小
                     deterministic);          // 是否确定性操作标志位

    // 根据deterministic条件设置dq_accum_split_stride，用于确定是否需要分割dq_accum
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

    // 设置launch函数指向run_mha_bwd函数
    auto launch = &run_mha_bwd;

    // 定义PhiloxCudaState对象philox_args，并根据is_dropout条件初始化
    at::PhiloxCudaState philox_args;
    if (is_dropout) {
        if (at::cuda::currentStreamCaptureStatus() ==
                at::cuda::CaptureStatus::None)
        {
            // 当前无流捕获时，使用philox_seed和philox_offset初始化philox_args
            philox_args = at::PhiloxCudaState(*philox_seed.data_ptr<int64_t>(), *philox_offset.data_ptr<int64_t>());
        } else { // dropout + capture
            // 否则，使用philox_seed和philox_offset，额外指定0初始化philox_args
            philox_args = at::PhiloxCudaState(
                philox_seed.data_ptr<int64_t>(), philox_offset.data_ptr<int64_t>(), 0);
        }
    }
    params.philox_args = philox_args;  // 将初始化后的philox_args赋给params的philox_args成员

    // 设置alibi参数，针对alibi_slopes_、批量大小和头数
    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // 若seqlen_q大于0，则调用launch函数执行参数和流的反向处理
    if (seqlen_q > 0) {
        launch(params, stream);
    } else {
        // 如果seqlen_q等于0，表示张量为空，将dk_expanded、dv_expanded、softmax_d置零
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // 对于MQA/GQA，需要对组间的dK和dV求和处理
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }
    // 返回结果dq、dk、dv、softmax_d的元组
    return { dq, dk, dv, softmax_d };
    // 定义函数 mha_varlen_bwd，返回四个张量元组，用于执行变长 Multi-Head Attention 的反向传播
    // dout: 梯度张量，维度为 total_q x num_heads x head_size，表示输出梯度
    // q: 查询张量，维度为 total_q x num_heads x head_size，表示查询向量
    // k: 键张量，维度为 total_k x num_heads_k x head_size，表示键向量
    // v: 值张量，维度为 total_k x num_heads_k x head_size，表示值向量
    // out: 输出张量，维度为 total_q x num_heads x head_size，表示 Multi-Head Attention 的输出
    // softmax_lse: softmax logsumexp 张量，维度为 b x h x s，用于 softmax 的 logsumexp 计算
    // dq_, dk_, dv_: 可选的梯度张量，分别对应查询、键、值的梯度
    // cu_seqlens_q, cu_seqlens_k: 序列长度张量，维度为 b+1，表示每个批次的累计序列长度
    // alibi_slopes_: 可选的斜率张量，维度为 num_heads 或 b x num_heads，用于梯度斜率的计算
    // max_seqlen_q, max_seqlen_k: 最大序列长度，用于选择合适的核大小
    // p_dropout: dropout 概率
    // softmax_scale: softmax 缩放因子
    // zero_tensors: 是否将张量清零
    // is_causal: 是否为因果模式
    // window_size_left, window_size_right: 左右窗口大小，用于因果注意力
    // deterministic: 是否确定性运行
    // philox_seed, philox_offset: 随机种子和偏移量张量

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        // 如果定义了 FLASHATTENTION_DISABLE_BACKWARD 宏，则抛出错误，表示不支持反向传播
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif

    // 如果是因果模式，则右窗口大小设为 0
    if (is_causal) { window_size_right = 0; }

    // 获取当前 CUDA 设备的属性
    auto dprops = at::cuda::getCurrentDeviceProperties();

    // 检查当前 GPU 是否为 Ampere 或更新版本，否则抛出错误
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    // 检查是否需要 dropout
    bool is_dropout = p_dropout > 0.0;

    // 获取当前 CUDA 流
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // 检查查询张量的数据类型，只支持 fp16 和 bf16 类型
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    // 如果查询张量是 bf16 类型，则要求当前 GPU 必须是 Ampere 或更新版本
    if (q_dtype == at::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }

    // 检查键和值张量的数据类型必须与查询张量一致
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    // 检查输出的数据类型是否与查询的数据类型相同
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    // 检查导数的数据类型是否与查询的数据类型相同
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    // 检查 cu_seqlens_q 张量是否具有 int32 类型
    TORCH_CHECK(cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must have dtype int32");
    // 检查 cu_seqlens_k 张量是否具有 int32 类型
    TORCH_CHECK(cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must have dtype int32");

    // 检查 q, k, v, out, dout, softmax_lse, cu_seqlens_q, cu_seqlens_k 张量是否在同一个设备上
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    // 检查输入张量的最后一个维度是否是连续的
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    // 检查 cu_seqlens_q, cu_seqlens_k 张量是否是连续的
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    // 获取输入张量 q 的尺寸信息
    const auto sizes = q.sizes();

    // 初始化变量以存储尺寸信息和相关计算结果
    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = dout.size(2);
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);

    // 检查 batch_size 是否大于 0
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    // 检查 head_size 是否是 8 的倍数
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    // 检查 head_size_og 是否是 8 的倍数
    TORCH_CHECK(head_size_og % 8 == 0, "head_size_og should be a multiple of 8, this is ensured by padding!");

    // 检查 head_size 是否不超过 256
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");

    // 如果 head_size 大于 192 并且 (head_size <= 224 或者 is_dropout 为真)，则进一步检查硬件兼容性
    if (head_size > 192 && (head_size <= 224 || is_dropout)) {
        TORCH_CHECK(is_sm80 || is_sm90, "FlashAttention backward for head dim 256 with dropout, or head dim 224 with/without dropout requires A100/A800 or H100/H800");
    }

    // 检查 key/value 的头部数是否能整除查询的头部数
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // 定义一个函数，将 x 向上舍入到最接近 m 的倍数
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

    // 计算头部大小向上舍入到最接近的 32 的倍数后的值
    const int head_size_rounded = round_multiple(head_size, 32);
    // 计算查询序列长度向上舍入到最接近的 128 的倍数后的值
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    // 计算 key 序列长度向上舍入到最接近的 128 的倍数后的值
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // 检查 head_size 是否等于 head_size_og 向上舍入到最接近的 8 的倍数
    TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    // 如果 window_size_left 大于等于 max_seqlen_k，则将其设置为 -1
    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    // 如果 window_size_right 大于等于 max_seqlen_k，则将其设置为 -1
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    // 检查张量的形状是否符合预期
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    // 声明 dq, dk, dv 张量，用于存储梯度信息
    at::Tensor dq, dk, dv;
    // 如果 dq_ 有值，则将其值赋给 dq
    if (dq_.has_value()) {
        dq = dq_.value();
        // 检查 dq 的数据类型必须与 q 相同
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        // 检查 dq 的设备是否正确
        CHECK_DEVICE(dq);
        // 检查 dq 的最后一个维度是否是连续的
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        // 检查 dq 的形状是否符合要求
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        // 如果 dq_ 没有值，则创建一个与 q 相同形状的空张量 dq
        dq = at::empty_like(q);
    }

    // 如果 dk_ 有值，则将其值赋给 dk
    if (dk_.has_value()) {
        dk = dk_.value();
        // 检查 dk 的数据类型必须与 q 相同
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        // 检查 dk 的设备是否正确
        CHECK_DEVICE(dk);
        // 检查 dk 的最后一个维度是否是连续的
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        // 检查 dk 的形状是否符合要求
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        // 如果 dk_ 没有值，则创建一个与 k 相同形状的空张量 dk
        dk = at::empty_like(k);
    }

    // 如果 dv_ 有值，则将其值赋给 dv
    if (dv_.has_value()) {
        dv = dv_.value();
        // 检查 dv 的数据类型必须与 q 相同
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        // 检查 dv 的设备是否正确
        CHECK_DEVICE(dv);
        // 检查 dv 的最后一个维度是否是连续的
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        // 检查 dv 的形状是否符合要求
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
        // 如果 dv_ 没有值，则创建一个与 v 相同形状的空张量 dv
        dv = at::empty_like(v);
    }

    // 设置循环标志为 true，暂时硬编码以后修改
    bool loop = true;

    // 切换到与 q 张量相同的设备上，避免在 cuda:0 设备上启动核心
    // 将设备编号转换为 char 类型以避免编译器关于缩小转换的警告
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // 获取与 q 张量相同选项的张量选项
    auto opts = q.options();

    // 创建一个空张量 softmax_d，用于存储 softmax 结果
    auto softmax_d = at::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));

    // 初始化 dq_accum 张量
    at::Tensor dq_accum;
    if (loop) {
        // 如果需要循环处理，则根据是否确定性来选择不同的初始化方式
        // 避免在非确定性情况下分配过大的 dq_accum
        if (!deterministic) {
            dq_accum = at::empty({total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            // 在确定性情况下，根据多处理器数量和批次大小分配 dq_accum
            const int nsplits = (dprops->multiProcessorCount + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = at::zeros({nsplits, total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
    }

    // 初始化 dk_expanded 和 dv_expanded 张量
    at::Tensor dk_expanded, dv_expanded;
    // 如果 num_heads_k 不等于 num_heads，则执行 MQA / GQA 的逻辑
    if (num_heads_k != num_heads) {  // MQA / GQA
        // 创建空的扩展张量 dk_expanded 和 dv_expanded，形状为 {total_k, num_heads, head_size}，使用给定的 opts 选项
        dk_expanded = at::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = at::empty({total_k, num_heads, head_size}, opts);
    } else {
        // 否则，直接使用传入的 dk 和 dv 张量
        dk_expanded = dk;
        dv_expanded = dv;
    }

    // 如果 zero_tensors 为真，则将 dq、dk_expanded、dv_expanded 和 softmax_d 张量置零
    if( zero_tensors ) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // 创建 Flash_bwd_params 结构体对象 params
    Flash_bwd_params params;

    // 调用 set_params_dgrad 函数设置 params 结构体中的各参数
    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout, dq, dk_expanded, dv_expanded,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     loop ? dq_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     deterministic);

    // 如果非确定性且 loop 为真，则设置 params 结构体中的 dq_accum_split_stride 为 dq_accum 张量的步长，否则为 0
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

    // 设置 launch 指向 run_mha_bwd 函数
    auto launch = &run_mha_bwd;

    // 创建 PhiloxCudaState 对象 philox_args，用于存储 Philox 算法的参数
    at::PhiloxCudaState philox_args;
    // 如果 is_dropout 为真
    if (is_dropout) {
        // 检查当前 CUDA 流的捕获状态
        if (at::cuda::currentStreamCaptureStatus() ==
                at::cuda::CaptureStatus::None)
        {
            // 如果未捕获 CUDA 流，则使用 philox_seed 和 philox_offset 初始化 philox_args
            philox_args = at::PhiloxCudaState(*philox_seed.data_ptr<int64_t>(), *philox_offset.data_ptr<int64_t>());
        } else { // dropout + capture
            // 如果捕获了 CUDA 流，则使用 philox_seed 和 philox_offset 初始化 philox_args，同时设置额外参数为 0
            philox_args = at::PhiloxCudaState(
                philox_seed.data_ptr<int64_t>(), philox_offset.data_ptr<int64_t>(), 0);
        }
    }
    // 将 philox_args 设置到 params 结构体中
    params.philox_args = philox_args;

    // 调用 set_params_alibi 函数设置 params 结构体中的 alibi 相关参数
    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // 如果 max_seqlen_q 大于 0，则调用 launch 函数执行 MHA 的反向传播
    if (max_seqlen_q > 0) {
        launch(params, stream);
    } else {
        // 如果 seqlen_q == 0，则输出张量为空，将 dk_expanded、dv_expanded 和 softmax_d 张量置零
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // 如果 num_heads_k 不等于 num_heads，则执行 MQA/GQA 的求和操作，将结果存入 dk 和 dv 张量中
    // 将 dk_expanded 和 dv_expanded 张量按照指定维度求和后存入 dk 和 dv 张量中
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    }

    // 返回包含 dq、dk、dv 和 softmax_d 四个张量的元组
    return { dq, dk, dv, softmax_d };
    }

    // 定义函数 mha_fwd_kvcache，接收多个参数，返回包含两个 Tensor 的 tuple
    std::tuple<at::Tensor, at::Tensor>
    mha_fwd_kvcache(at::Tensor &q,                 // 输入查询 Tensor，形状为 batch_size x seqlen_q x num_heads x head_size
                    const at::Tensor &kcache,       // 输入键的缓存 Tensor，形状为 batch_size_c x seqlen_k x num_heads_k x head_size 或者 num_blocks x page_block_size x num_heads_k x head_size（如果有 block_table）
                    const at::Tensor &vcache,       // 输入值的缓存 Tensor，形状同 kcache
                    std::optional<const at::Tensor> &k_, // 可选输入的新键 Tensor，形状为 batch_size x seqlen_knew x num_heads_k x head_size
                    std::optional<const at::Tensor> &v_, // 可选输入的新值 Tensor，形状同 k_
                    std::optional<const at::Tensor> &seqlens_k_, // 可选输入的序列长度 Tensor，形状为 batch_size
                    std::optional<const at::Tensor> &rotary_cos_, // 可选输入的旋转余弦 Tensor，形状为 seqlen_ro x (rotary_dim / 2)
                    std::optional<const at::Tensor> &rotary_sin_, // 可选输入的旋转正弦 Tensor，形状同 rotary_cos_
                    std::optional<const at::Tensor> &cache_batch_idx_, // 可选输入的用于索引 KV 缓存的索引 Tensor
                    std::optional<at::Tensor> &block_table_, // 可选输入的块表格 Tensor，形状为 batch_size x max_num_blocks_per_seq
                    std::optional<at::Tensor> &alibi_slopes_, // 可选输入的 alibi slopes Tensor，形状为 num_heads 或者 batch_size x num_heads
                    std::optional<at::Tensor> &out_,             // 可选输出的结果 Tensor，形状为 batch_size x seqlen_q x num_heads x head_size
                    const float softmax_scale,       // 输入的 softmax 缩放因子
                    bool is_causal,                  // 是否是因果注意力
                    int window_size_left,            // 左侧窗口大小
                    int window_size_right,           // 右侧窗口大小
                    bool is_rotary_interleaved,      // 是否交错旋转
                    int num_splits                   // 分割数
                    ) {

        // 获取当前 CUDA 设备的属性
        auto dprops = at::cuda::getCurrentDeviceProperties();
        // 检查是否是 Ampere 架构或更新版本的 GPU
        bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
        bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
        TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

        // 检查查询 Tensor 的数据类型是否为 fp16 或 bf16
        auto q_dtype = q.dtype();
        TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
                    "FlashAttention only support fp16 and bf16 data type");
        // 如果查询 Tensor 的数据类型是 bf16，则要求 GPU 架构为 Ampere 或更新版本
        if (q_dtype == at::kBFloat16) {
            TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
        }

        // 检查键缓存和查询 Tensor 的数据类型是否相同
        TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
        TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

        // 检查输入 Tensor 是否在期望的设备上
        CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

        // 检查查询 Tensor 是否有连续的最后一个维度
        TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
        TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // 检查输入张量的最后一个维度是否连续，必须为1，否则抛出错误信息
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    // 声明一个名为 block_table 的张量
    at::Tensor block_table;
    // 检查是否存在分页的键值缓存
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        // 分页键值缓存不支持 cache_batch_idx，如果存在则抛出错误信息
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        // 将 block_table_.value() 的值赋给 block_table
        block_table = block_table_.value();
        // 检查 block_table 的设备是否正确
        CHECK_DEVICE(block_table);
        // 检查 block_table 的数据类型是否为 torch.int32
        TORCH_CHECK(block_table.dtype() == at::kInt, "block_table must have dtype torch.int32");
        // 检查 block_table 的最后一个维度是否连续，必须为1，否则抛出错误信息
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    // 获取输入张量 q 的尺寸
    const auto sizes = q.sizes();

    // 获取批量大小
    const int batch_size = sizes[0];
    // 获取查询序列长度
    int seqlen_q = sizes[1];
    // 获取查询头的数量
    int num_heads = sizes[2];
    // 获取原始头部大小
    const int head_size_og = sizes[3];

    // 计算每个序列最大的块数，如果没有分页键值缓存则为0
    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    // 计算块的数量，如果没有分页键值缓存则为0
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);
    // 计算页块大小，如果没有分页键值缓存则为1
    const int page_block_size = !paged_KV ? 1 : kcache.size(1);
    // 检查分页键值缓存的页块大小是否可以被256整除，否则抛出错误信息
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    // 计算键 k 的序列长度，如果没有分页键值缓存则为 kcache 的第二维大小
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    // 计算键 k 的头部数量，如果没有分页键值缓存则为 kcache 的第三维大小
    const int num_heads_k = kcache.size(2);
    // 计算调整后的批量大小，如果没有分页键值缓存则为 kcache 的第一维大小
    const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
    // 检查批量大小是否大于0，否则抛出错误信息
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    // 检查原始头部大小是否小于等于256，否则抛出错误信息
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    // 检查查询头的数量是否能整除键 k 的头的数量，否则抛出错误信息
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // 如果查询序列长度为1且 alibi_slopes_ 不存在，则将 is_causal 设为 false
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    // 如果是因果，将右侧窗口大小设为0
    if (is_causal) { window_size_right = 0; }

    // 根据特定条件，将查询张量 q 从 (b, 1, (nheads_kv ngroups), d) 转置为 (b, ngroups, nheads_kv, d)
    // 这种情况下更快，感谢 Daniel Haziza 的建议
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    // 如果左侧窗口大小大于等于键 k 的序列长度，将左侧窗口大小设为-1
    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    // 如果右侧窗口大小大于等于键 k 的序列长度，将右侧窗口大小设为-1
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // 检查查询张量 q 的形状是否符合预期
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    // 如果没有分页键值缓存，检查键 k 和值 v 的形状是否符合预期
    if (!paged_KV) {
        CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    } else {
        // 如果有分页键值缓存，检查键 k、值 v 和 block_table 的形状是否符合预期
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    // 声明张量 q_padded、kcache_padded、vcache_padded
    // 如果原始头大小不是8的倍数，则对q、kcache和vcache进行填充，使其大小成为8的倍数
    if (head_size_og % 8 != 0) {
        q_padded = at::pad(q, {0, 8 - head_size_og % 8});
        kcache_padded = at::pad(kcache, {0, 8 - head_size_og % 8});
        vcache_padded = at::pad(vcache, {0, 8 - head_size_og % 8});
        // 使用PyTorch的pad函数对输入进行填充，保证大小为8的倍数
        // q_padded = at::nn::functional::pad(q, at::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        // kcache_padded = at::nn::functional::pad(kcache, at::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        // vcache_padded = at::nn::functional::pad(vcache, at::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        // 如果原始头大小已经是8的倍数，则直接使用原始的q、kcache和vcache
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
    }

    // 初始化输出张量
    at::Tensor out;
    if (out_.has_value()) {
        // 如果输出已经存在，则使用给定的输出
        out = out_.value();
        // 检查输出张量的数据类型必须与输入q的数据类型相同
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        // 检查输出张量的设备是否一致
        CHECK_DEVICE(out);
        // 检查输出张量最后一个维度是否连续
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        // 检查输出张量的形状是否符合预期
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        // 如果原始头大小不是8的倍数，则重新分配空的张量以匹配填充后的q_padded大小
        if (head_size_og % 8 != 0) { out = at::empty_like(q_padded); }
    } else {
        // 如果输出不存在，则创建一个和q_padded相同大小的空张量作为输出
        out = at::empty_like(q_padded);
    }

    // 定义一个函数，用于向上舍入到最接近的m的倍数
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // 对头大小进行向上舍入到最接近的8的倍数
    const int head_size = round_multiple(head_size_og, 8);
    // 对向上舍入后的头大小再次向上舍入到最接近的32的倍数
    const int head_size_rounded = round_multiple(head_size, 32);
    // 对seqlen_q向上舍入到最接近的128的倍数
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    // 对seqlen_k向上舍入到最接近的128的倍数
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // 设置CUDA设备，避免从cuda:0设备启动内核
    // 将设备ID转换为char类型以避免编译器关于类型缩小的警告
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // 获取输入张量的选项
    auto opts = q.options();

    // 创建一个空的softmax_lse张量，用于存储softmax的log-sum-exp值
    auto softmax_lse = at::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    // 创建Flash_fwd_params对象，并设置其参数
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, kcache_padded, vcache_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right);

    // 初始化k、v、k_padded和v_padded张量
    at::Tensor k, v, k_padded, v_padded;
    // 如果 k_ 包含值
    if (k_.has_value()) {
        // 检查 v_ 也必须包含值，否则抛出异常
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        // 检查 seqlens_k_ 也必须包含值，否则抛出异常
        TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        // 检查查询序列长度 seqlen_q 必须小于等于键序列长度 seqlen_k，否则抛出异常
        TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        // 从 k_ 中获取实际值赋给 k
        k = k_.value();
        // 从 v_ 中获取实际值赋给 v
        v = v_.value();
        // 检查键 k 的数据类型必须与查询 q 的数据类型相同，否则抛出异常
        TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        // 检查值 v 的数据类型必须与查询 q 的数据类型相同，否则抛出异常
        TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        // 检查键 k 的设备和查询 q 的设备必须一致，否则抛出异常
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        // 检查键 k 的最后一个维度必须是连续的，否则抛出异常
        TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        // 检查值 v 的最后一个维度必须是连续的，否则抛出异常
        TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        // 计算新的键序列长度 seqlen_knew
        int seqlen_knew = k.size(1);
        // 检查键 k 的形状必须符合预期的批量大小、新的序列长度、头数和原始头大小
        CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        // 检查值 v 的形状必须符合预期的批量大小、新的序列长度、头数和原始头大小
        CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        
        // 如果原始头大小 head_size_og 不是 8 的倍数，对 k 和 v 进行填充
        if (head_size_og % 8 != 0) {
            k_padded = at::pad(k, {0, 8 - head_size_og % 8});
            v_padded = at::pad(v, {0, 8 - head_size_og % 8});
            // k_padded = at::nn::functional::pad(k, at::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            // v_padded = at::nn::functional::pad(v, at::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            // 否则直接使用原始的 k 和 v
            k_padded = k;
            v_padded = v;
        }
        
        // 将新的键序列长度存储在 params 结构体中
        params.seqlen_knew = seqlen_knew;
        // 将 k_padded 的数据指针存储在 params 结构体中
        params.knew_ptr = k_padded.data_ptr();
        // 将 v_padded 的数据指针存储在 params 结构体中
        params.vnew_ptr = v_padded.data_ptr();
        
        // 将 k_padded 的批量步长存储在 params 结构体中
        params.knew_batch_stride = k_padded.stride(0);
        // 将 v_padded 的批量步长存储在 params 结构体中
        params.vnew_batch_stride = v_padded.stride(0);
        // 将 k_padded 的行步长存储在 params 结构体中
        params.knew_row_stride = k_padded.stride(-3);
        // 将 v_padded 的行步长存储在 params 结构体中
        params.vnew_row_stride = v_padded.stride(-3);
        // 将 k_padded 的头步长存储在 params 结构体中
        params.knew_head_stride = k_padded.stride(-2);
        // 将 v_padded 的头步长存储在 params 结构体中
        params.vnew_head_stride = v_padded.stride(-2);
    }
    
    // 如果 seqlens_k_ 包含值
    if (seqlens_k_.has_value()) {
        // 从 seqlens_k_ 中获取实际值赋给 seqlens_k
        auto seqlens_k = seqlens_k_.value();
        // 检查 seqlens_k 的数据类型必须是 int32，否则抛出异常
        TORCH_CHECK(seqlens_k.dtype() == at::kInt, "seqlens_k must have dtype int32");
        // 检查 seqlens_k 的设备必须与预期一致，否则抛出异常
        CHECK_DEVICE(seqlens_k);
        // 检查 seqlens_k 必须是连续的，否则抛出异常
        CHECK_CONTIGUOUS(seqlens_k);
        // 检查 seqlens_k 的形状必须符合预期的批量大小
        CHECK_SHAPE(seqlens_k, batch_size);
        // 将 seqlens_k 的数据指针转换为 int* 类型并存储在 params 结构体中
        params.cu_seqlens_k = static_cast<int *>(seqlens_k.data_ptr());
    }
    
    // 如果 seqlens_k_ 不包含值，则将 params 结构体中的 is_seqlens_k_cumulative 设置为 true
    params.is_seqlens_k_cumulative = !(seqlens_k_.has_value());
    // 检查是否提供了旋转余弦值
    if (rotary_cos_.has_value()) {
        // 如果提供了旋转余弦/正弦值，必须同时提供新的键/值对以追加到KV缓存中
        TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        // 获取旋转余弦值并检查其设备类型
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos);
        // 设置参数params中的rotary_dim，要求其不超过head_size
        params.rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        // 检查rotary_dim是否能被16整除，当前仅支持这种维度
        TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        // 获取旋转余弦值的序列长度
        const int seqlen_ro = rotary_cos.size(0);
        // 检查cos/sin序列长度是否至少与KV缓存的序列长度一致
        TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        // 检查旋转余弦值的形状，确保其为(seqlen_ro, params.rotary_dim / 2)
        CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        // 检查旋转余弦值是否是连续存储的
        CHECK_CONTIGUOUS(rotary_cos);
        // 检查旋转余弦值的数据类型是否与查询相同
        TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        // 检查是否同时提供了旋转正弦值
        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        // 获取旋转正弦值并检查其设备类型
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin);
        // 检查旋转正弦值的形状，确保其为(seqlen_ro, params.rotary_dim / 2)
        CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        // 检查旋转正弦值是否是连续存储的
        CHECK_CONTIGUOUS(rotary_sin);
        // 检查旋转正弦值的数据类型是否与查询相同
        TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
        // 设置params中的旋转余弦和正弦指针，以及旋转插入标志
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        // 若未提供旋转余弦值，则将params中的rotary_dim设置为0
        params.rotary_dim = 0;
    }

    // 检查是否提供了缓存批次索引
    if (cache_batch_idx_.has_value()) {
        // 获取缓存批次索引并检查其设备类型
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        // 检查缓存批次索引是否是连续存储的
        CHECK_CONTIGUOUS(cache_batch_idx);
        // 检查缓存批次索引的数据类型是否为int32
        TORCH_CHECK(cache_batch_idx.scalar_type() == at::kInt, "cache_batch_idx must have dtype int32");
        // 设置params中的cache_batch_idx为缓存批次索引的数据指针
        params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }

    // 设置MHA的参数，包括拆分KV、batch_size、num_heads等
    set_params_splitkv(params, batch_size, num_heads,
                       head_size, seqlen_k, seqlen_q,
                       head_size_rounded, /*dropout*/0.f, num_splits, dprops, opts);

    // 如果使用分页KV，则设置params中的块表和块表批次步幅
    if (paged_KV) {
        // 设置params中的块表指针和块表批次步幅
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
    }
    // 设置params中的页块大小
    params.page_block_size = page_block_size;

    // 设置params中的alibi参数
    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // 获取当前CUDA流
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // 只有当存在要追加到KV缓存的新键/值对、使用cache_batch_idx索引缓存或使用分页KV缓存时，才强制使用拆分核心
    run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);
    // 如果原始头部大小不是8的倍数，则执行以下操作
    if (head_size_og % 8 != 0) {
        // 使用 narrow 方法在最后一个维度上对 out 进行切片，保留前 head_size_og 个元素
        out = out.narrow(-1, 0, head_size_og);
        // 如果 out_ 中有值，则将 out 的内容复制给 out_
        if (out_.has_value()) { out_.value().copy_(out); }
        // 如果 k_ 中有值，则执行以下操作
        if (k_.has_value()) {
            // 复制 kcache_padded 在最后一个维度上前 head_size_og 个元素的内容到 kcache
            kcache.copy_(kcache_padded.narrow(-1, 0, head_size_og));
            // 复制 vcache_padded 在最后一个维度上前 head_size_og 个元素的内容到 vcache
            vcache.copy_(vcache_padded.narrow(-1, 0, head_size_og));
        }
    }

    // 如果 seqlenq_ngroups_swapped 为真，则执行以下操作
    if (seqlenq_ngroups_swapped) {
        // 对 out 进行转置，交换第1和第2维度，并重新形状为 {batch_size, 1, num_heads_k * seqlen_q, head_size_og}
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        // 将 softmax_lse 重新形状为 {batch_size, num_heads_k * seqlen_q, 1}
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    // 返回一个包含 out 和 softmax_lse 的字典
    return {out, softmax_lse};
}

} // namespace pytorch_fmha

#endif
```