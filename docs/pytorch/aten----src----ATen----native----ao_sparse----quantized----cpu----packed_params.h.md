# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\packed_params.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <cstdint>
// 包含 C++ 标准头文件cstdint，定义了整数类型，如int64_t

#include <ATen/core/ivalue.h>
// 包含 PyTorch ATen 库的头文件ivalue.h

namespace ao {
namespace sparse {

// <Weight, bias, out_features_block_size, in_features_block_size>
// 定义了LinearPackedSerializationType为包含权重、偏置、输出特征块大小和输入特征块大小的元组类型
using LinearPackedSerializationType =
    std::tuple<at::Tensor, std::optional<at::Tensor>, std::vector<int64_t>>;

#define SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION 2
// 定义了稀疏线性层压缩参数序列化版本号为2

// 定义了BCSRSerializationType为包含稀疏矩阵压缩格式相关数据的元组类型
using BCSRSerializationType =
    std::tuple<
        int64_t,                    // 序列化版本号
        std::optional<at::Tensor>,  // 偏置
        int64_t,                    // 输出特征（行）块大小
        int64_t,                    // 输入特征（列）块大小
        at::Tensor,                 // 权重的缩放因子（如果每个张量则是单元素向量）（float）
        at::Tensor,                 // 权重零点的包装器（如果每个张量则是单元素向量）（int8_t）
        bool,                       // 量化方案（true: 每个张量, false: 每个通道）
        at::Tensor,                 // 行块索引的包装器（int8_t, int16_t 或 int32_t）
        at::Tensor,                 // 列块索引的包装器（int8_t, int16_t 或 int32_t）
        at::Tensor,                 // 非零权重值的包装器，每个+128（uint8_t）
        int64_t,                    // 输出通道数
        int64_t                     // 输入通道数
    >;

// 定义了BCSR为包含稀疏矩阵BCSR格式相关数据的元组类型
using BCSR =
    std::tuple<
        std::vector<int8_t>,    // 非零权重值
        std::vector<int32_t>,   // 压缩行块索引
        std::vector<int32_t>    // 列块索引
    >;

// 定义了LinearPackedParamsBase结构体，继承自torch::jit::CustomClassHolder，用于表示线性层压缩参数的基础类
struct LinearPackedParamsBase : public torch::jit::CustomClassHolder {
 public:
  // 构造函数，接受输出特征块大小和输入特征块大小作为参数
  LinearPackedParamsBase(
      const int64_t out_features_block_size,
      const int64_t in_features_block_size)
      : out_features_block_size_(out_features_block_size),
        in_features_block_size_(in_features_block_size) {}

  // 纯虚函数，子类需实现应用线性变换的方法，接受输入张量、输出缩放和输出零点作为参数
  virtual at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;

  // 纯虚函数，子类需实现应用ReLU激活的线性变换的方法，接受输入张量、输出缩放和输出零点作为参数
  virtual at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;

  // 纯虚函数，子类需实现应用动态量化的方法，接受输入张量作为参数
  virtual at::Tensor apply_dynamic(const at::Tensor& input) = 0;

  // 纯虚函数，子类需实现应用动态ReLU激活和动态量化的方法，接受输入张量作为参数
  virtual at::Tensor apply_dynamic_relu(const at::Tensor& input) = 0;

  // 纯虚函数，子类需实现反序列化为LinearPackedSerializationType类型的方法
  virtual LinearPackedSerializationType unpack() = 0;

  // 纯虚函数，子类需实现序列化为BCSRSerializationType类型的方法
  virtual BCSRSerializationType serialize() = 0;

  // 纯虚函数，子类需实现获取偏置的方法
  virtual std::optional<at::Tensor> bias() = 0;

  // 虚函数，设置偏置的方法，如果未实现则抛出运行时错误
  virtual void set_bias(const std::optional<at::Tensor>& bias) {
    throw std::runtime_error(
        "set_bias is not implemented for this packed "
        "parameter type");
  }

 protected:
  const int64_t out_features_block_size_, in_features_block_size_;
  // 输出特征块大小和输入特征块大小
};

}}  // namespace ao::sparse
```