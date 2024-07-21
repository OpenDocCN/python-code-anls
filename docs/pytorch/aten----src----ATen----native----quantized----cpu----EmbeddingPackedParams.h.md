# `.\pytorch\aten\src\ATen\native\quantized\cpu\EmbeddingPackedParams.h`

```py
#pragma once
// 预处理指令，确保此头文件仅被编译一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 Tensor 类的头文件

#include <ATen/core/ivalue.h>
// 包含 ATen 库中的 IValue 类的头文件

struct EmbeddingPackedParamsBase : public torch::jit::CustomClassHolder {
  // 定义 EmbeddingPackedParamsBase 结构体，继承自 torch::jit::CustomClassHolder

  virtual at::Tensor embeddingbag_byte(
    const at::Tensor& indices,
    // embeddingbag_byte 方法接受一个 Tensor 类型的 indices 参数，用于索引
    const std::optional<at::Tensor>& offsets,
    // 可选的 Tensor 类型 offsets 参数，用于偏移量
    bool pruned_weights,
    // 布尔类型参数，指示是否进行权重修剪
    const std::optional<at::Tensor>& per_sample_weights_,
    // 可选的 Tensor 类型 per_sample_weights_ 参数，用于每个样本的权重
    const std::optional<at::Tensor>& compressed_indices_mapping,
    // 可选的 Tensor 类型 compressed_indices_mapping 参数，用于压缩索引映射
    bool include_last_offset,
    // 布尔类型参数，指示是否包括最后一个偏移量
    bool is_embedding_op) = 0;
    // 纯虚函数，表示此方法是抽象的，需要在派生类中实现

  virtual at::Tensor embeddingbag_4bit(
    const at::Tensor& indices,
    // embeddingbag_4bit 方法接受一个 Tensor 类型的 indices 参数，用于索引
    const std::optional<at::Tensor>& offsets,
    // 可选的 Tensor 类型 offsets 参数，用于偏移量
    bool pruned_weights,
    // 布尔类型参数，指示是否进行权重修剪
    const std::optional<at::Tensor>& per_sample_weights_,
    // 可选的 Tensor 类型 per_sample_weights_ 参数，用于每个样本的权重
    const std::optional<at::Tensor>& compressed_indices_mapping,
    // 可选的 Tensor 类型 compressed_indices_mapping 参数，用于压缩索引映射
    bool include_last_offset,
    // 布尔类型参数，指示是否包括最后一个偏移量
    bool is_embedding_op) = 0;
    // 纯虚函数，表示此方法是抽象的，需要在派生类中实现

  virtual at::Tensor unpack() = 0;
  // 纯虚函数，表示 unpack 方法是抽象的，需要在派生类中实现

  virtual int64_t bit_rate() const = 0;
  // 纯虚函数，表示 bit_rate 方法是抽象的，返回一个 int64_t 类型的值，需要在派生类中实现

  virtual int64_t version() const = 0;
  // 纯虚函数，表示 version 方法是抽象的，返回一个 int64_t 类型的值，需要在派生类中实现
};
// 结构体 EmbeddingPackedParamsBase 的定义结束
```