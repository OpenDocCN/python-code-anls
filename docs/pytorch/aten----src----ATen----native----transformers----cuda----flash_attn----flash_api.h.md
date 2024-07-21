# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\flash_api.h`

```py
#pragma once
#include <cstddef>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

// 命名空间 pytorch_flash

namespace pytorch_flash {

// 声明 TORCH_API，用于导出该函数或类型
TORCH_API
// 定义函数 mha_fwd，返回一个包含八个 at::Tensor 的元组
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(const at::Tensor &q,         // 输入参数 q，形状为 batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // 输入参数 k，形状为 batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // 输入参数 v，形状为 batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &out_,             // 可选的输出参数 out_
        std::optional<at::Tensor> &alibi_slopes_,    // 可选的输出参数 alibi_slopes_
        const float p_dropout,       // 浮点数参数 p_dropout
        const float softmax_scale,   // 浮点数参数 softmax_scale
        bool is_causal,              // 布尔值参数 is_causal
        int window_size_left,        // 整数参数 window_size_left
        int window_size_right,       // 整数参数 window_size_right
        const bool return_softmax,   // 布尔值参数 return_softmax
        std::optional<at::Generator> gen_);   // 可选的生成器参数 gen_

// 声明函数 mha_varlen_fwd，返回一个包含八个 at::Tensor 的元组
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd(const at::Tensor &q,  // 输入参数 q，形状为 total_q x num_heads x head_size
               const at::Tensor &k,  // 输入参数 k，形状为 total_k x num_heads_k x head_size
               const at::Tensor &v,  // 输入参数 v，形状为 total_k x num_heads_k x head_size
               std::optional<at::Tensor> &out_, // 可选的输出参数 out_
               const at::Tensor &cu_seqlens_q,  // 输入参数 cu_seqlens_q，形状为 b+1
               const at::Tensor &cu_seqlens_k,  // 输入参数 cu_seqlens_k，形状为 b+1
               std::optional<at::Tensor> &seqused_k, // 可选的输出参数 seqused_k
               std::optional<at::Tensor> &alibi_slopes_, // 可选的输出参数 alibi_slopes_
               int max_seqlen_q,      // 整数参数 max_seqlen_q
               const int max_seqlen_k, // 整数参数 max_seqlen_k
               const float p_dropout, // 浮点数参数 p_dropout
               const float softmax_scale, // 浮点数参数 softmax_scale
               const bool zero_tensors, // 布尔值参数 zero_tensors
               bool is_causal,         // 布尔值参数 is_causal
               int window_size_left,   // 整数参数 window_size_left
               int window_size_right,  // 整数参数 window_size_right
               const bool return_softmax, // 布尔值参数 return_softmax
               std::optional<at::Generator> gen_); // 可选的生成器参数 gen_

} // namespace pytorch_flash
// 定义函数 mha_bwd，用于计算多头注意力机制的反向传播
mha_bwd(const at::Tensor &dout,  // 输出梯度，大小为 batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &q,     // 查询张量，大小为 batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,     // 键张量，大小为 batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,     // 值张量，大小为 batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // 输出张量，大小为 batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // Softmax logsumexp 张量，大小为 batch_size x num_heads x seqlen_q
        std::optional<at::Tensor> &dq_,   // 查询梯度（可选），大小为 batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // 键梯度（可选），大小为 batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // 值梯度（可选），大小为 batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // 用于调试的斜率（可选），大小为 num_heads 或 batch_size x num_heads
        const float p_dropout,         // 丢弃概率
        const float softmax_scale,     // Softmax 缩放参数
        const bool is_causal,          // 是否使用因果注意力
        int window_size_left,          // 左侧窗口大小
        int window_size_right,         // 右侧窗口大小
        const bool deterministic,      // 是否确定性计算
        const at::Tensor philox_seed,  // Philox 随机数种子张量
        const at::Tensor philox_offset); // Philox 随机数偏移张量

// 定义函数 mha_varlen_bwd，用于变长输入的多头注意力反向传播
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // 输出梯度，大小为 total_q x num_heads x head_size，其中 total_q 是所有序列长度之和
               const at::Tensor &q,     // 查询张量，大小为 total_q x num_heads x head_size
               const at::Tensor &k,     // 键张量，大小为 total_k x num_heads_k x head_size
               const at::Tensor &v,     // 值张量，大小为 total_k x num_heads_k x head_size
               const at::Tensor &out,   // 输出张量，大小为 total_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // Softmax logsumexp 张量，大小为 b x h x s
               std::optional<at::Tensor> &dq_,   // 查询梯度（可选），大小为 total_q x num_heads x head_size
               std::optional<at::Tensor> &dk_,   // 键梯度（可选），大小为 total_k x num_heads_k x head_size
               std::optional<at::Tensor> &dv_,   // 值梯度（可选），大小为 total_k x num_heads_k x head_size
               const at::Tensor &cu_seqlens_q,    // 查询序列长度张量，大小为 b+1
               const at::Tensor &cu_seqlens_k,    // 键序列长度张量，大小为 b+1
               std::optional<at::Tensor> &alibi_slopes_, // 用于调试的斜率（可选），大小为 num_heads 或 b x num_heads
               const int max_seqlen_q,    // 最大查询序列长度
               const int max_seqlen_k,    // 最大键序列长度
               const float p_dropout,     // 丢弃概率
               const float softmax_scale, // Softmax 缩放参数
               const bool zero_tensors,   // 是否将梯度张量置零
               const bool is_causal,      // 是否使用因果注意力
               int window_size_left,      // 左侧窗口大小
               int window_size_right,     // 右侧窗口大小
               const bool deterministic,  // 是否确定性计算
               const at::Tensor philox_seed,  // Philox 随机数种子张量
               const at::Tensor philox_offset); // Philox 随机数偏移张量
```