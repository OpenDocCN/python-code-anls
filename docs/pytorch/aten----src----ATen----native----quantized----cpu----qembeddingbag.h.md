# `.\pytorch\aten\src\ATen\native\quantized\cpu\qembeddingbag.h`

```py
#pragma once
#include <ATen/core/Tensor.h>
#include <cstdint>

namespace at {
namespace native {

// 函数声明：用于计算按字节打包的行偏移量嵌入袋操作，输出到给定的张量
Tensor& embedding_bag_byte_rowwise_offsets_out(
    Tensor& output,  // 输出张量，将嵌入结果写入其中
    const Tensor& weight,  // 嵌入权重张量
    const Tensor& indices,  // 索引张量，指定要嵌入的数据
    const std::optional<Tensor>& offsets_in,  // 可选的偏移量张量，指示每个样本的起始索引
    const bool /* scale_grad_by_freq */,  // 是否按频率缩放梯度（未使用）
    const int64_t /* mode */,  // 模式参数（未使用）
    bool pruned_weights,  // 是否使用修剪过的权重
    const std::optional<Tensor>& per_sample_weights_,  // 可选的每样本权重张量
    const std::optional<Tensor>& compressed_indices_mapping,  // 可选的压缩索引映射张量
    bool include_last_offset);  // 是否包括最后一个偏移量

// 函数声明：用于计算按4位打包的行偏移量嵌入袋操作，输出到给定的张量
Tensor& embedding_bag_4bit_rowwise_offsets_out(
    Tensor& output,  // 输出张量，将嵌入结果写入其中
    const Tensor& weight,  // 嵌入权重张量
    const Tensor& indices,  // 索引张量，指定要嵌入的数据
    const std::optional<Tensor>& offsets_in,  // 可选的偏移量张量，指示每个样本的起始索引
    const bool /* scale_grad_by_freq */,  // 是否按频率缩放梯度（未使用）
    const int64_t /* mode */,  // 模式参数（未使用）
    bool pruned_weights,  // 是否使用修剪过的权重
    const std::optional<Tensor>& per_sample_weights_,  // 可选的每样本权重张量
    const std::optional<Tensor>& compressed_indices_mapping,  // 可选的压缩索引映射张量
    bool include_last_offset);  // 是否包括最后一个偏移量

// 函数声明：用于解包按字节打包的量化嵌入袋操作，输出到给定的张量
Tensor& qembeddingbag_byte_unpack_out(
    Tensor& output,  // 输出张量，将解包结果写入其中
    const Tensor& packed_weight);  // 打包的量化权重张量

} // native
} // at
```