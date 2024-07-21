# `.\pytorch\aten\src\ATen\native\transformers\attention.h`

```py
#pragma once
// 一旦该头文件被包含，保证本头文件内容只被编译一次

#include <ATen/core/Tensor.h>
// 引入 PyTorch 的张量 Tensor 类定义

#include <c10/macros/Export.h>
// 引入 C10 库中的导出宏定义

#include <ATen/native/DispatchStub.h>
// 引入 PyTorch 中用于分发的 Stub 定义

#include <ATen/native/transformers/attention.h>
// 引入 PyTorch 中的注意力机制相关函数定义

#include <c10/util/Optional.h>
// 引入 C10 库中的可选类型的实现

namespace at {
namespace native {

using fused_sdp_choice_fn = int64_t (*)(const Tensor& query_, const Tensor& key, const Tensor& value,
        const std::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal, std::optional<double> scale);
// 定义用于融合注意力机制的函数指针类型，接收查询、键、值张量及注意力掩码、dropout 概率、是否因果、缩放参数

DECLARE_DISPATCH(fused_sdp_choice_fn, _fused_sdp_choice_stub);
// 声明融合注意力机制分发的函数指针变量 _fused_sdp_choice_stub

TORCH_API Tensor bmm_nt(const Tensor& a, const Tensor& b);
// 声明 Torch API 下的矩阵乘法函数 bmm_nt，用于非转置乘法

TORCH_API Tensor masked_softmax(
    Tensor& attn_scores,
    std::optional<Tensor> attn_mask,
    const Tensor& query,
    std::optional<int64_t> mask_type = {});
// 声明 Torch API 下的带掩码的 softmax 函数，接收注意力分数、注意力掩码、查询张量及可选的掩码类型

using transform_bias_rescale_qkv_fn = void(*)(
    at::ScalarType type,
    void* _q_k_v,
    const void* _qkv,
    const void* _qkv_bias,
    int64_t B,
    int64_t T,
    int64_t D,
    int64_t num_head);
// 定义用于变换偏置并重新缩放 QKV 的函数指针类型，接收标量类型、指向 QKV 数据的指针、QKV 数据的常量指针、QKV 偏置的常量指针以及批次数、序列长度、维度数、头数

DECLARE_DISPATCH(transform_bias_rescale_qkv_fn, transform_bias_rescale_qkv_stub);
// 声明变换偏置并重新缩放 QKV 的分发函数指针变量 transform_bias_rescale_qkv_stub

TORCH_API Tensor transform0213_gemm_nt_bias(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& query);
// 声明 Torch API 下的 transform0213_gemm_nt_bias 函数，执行矩阵乘法并添加偏置，接收输入张量 a、b、c 和查询张量

TORCH_API Tensor bmm_nn(Tensor& out, const Tensor& a, const Tensor& b);
// 声明 Torch API 下的矩阵乘法函数 bmm_nn，用于非转置乘法并存储结果于输出张量 out

TORCH_API void debug_assert_shape(int line, const Tensor& t, c10::IntArrayRef shape);
// 声明 Torch API 下的调试函数 debug_assert_shape，用于验证张量 t 的形状是否与给定形状匹配

TORCH_API Tensor qkv_projection(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const Tensor& qkv_weight);
// 声明 Torch API 下的 QKV 投影函数 qkv_projection，用于计算查询、键、值张量的投影结果

using flash_attention_fn = void (*)(
    const Tensor& output, const Tensor& logsumexp,
    const Tensor& query, const Tensor& key, const Tensor& value,
    double dropout_p, bool is_causal,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale);
// 定义用于 Flash 注意力机制的前向函数指针类型，接收输出张量、logsumexp 张量、查询、键、值张量、dropout 概率、是否因果、注意力掩码和缩放参数

using flash_attention_backward_fn = void (*)(
    const Tensor& grad_q, const Tensor& grad_k,
    const Tensor& grad_v, const Tensor& grad_out,
    const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& out, const Tensor& logsumexp,
    double dropout_p, bool is_causal,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale);
// 定义用于 Flash 注意力机制的反向函数指针类型，接收查询、键、值、输出的梯度张量、查询、键、值、输出张量、logsumexp 张量、dropout 概率、是否因果、注意力掩码和缩放参数

DECLARE_DISPATCH(flash_attention_fn, flash_attention_kernel);
// 声明 Flash 注意力机制前向分发函数指针变量 flash_attention_kernel

DECLARE_DISPATCH(flash_attention_backward_fn, flash_attention_backward_kernel);
// 声明 Flash 注意力机制反向分发函数指针变量 flash_attention_backward_kernel

} // namespace native
} // namespace at
```