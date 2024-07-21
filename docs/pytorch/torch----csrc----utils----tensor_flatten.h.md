# `.\pytorch\torch\csrc\utils\tensor_flatten.h`

```
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <c10/core/TensorOptions.h>
#include <torch/csrc/Export.h>
#include <utility>

namespace torch::utils {

/// 生成一个组合了张量后端和标量类型的ID，用于对张量进行排序
/// （'like'张量通过提取它们的后端和标量类型来分组，因此此函数将它们结合成一个单一的数字）
inline size_t type_id(const at::Tensor& tensor) {
  // 返回张量的后端乘以标量类型的数量加上标量类型本身的值
  return static_cast<size_t>(tensor.options().backend()) *
      static_cast<size_t>(at::ScalarType::NumOptions) +
      static_cast<size_t>(tensor.scalar_type());
}

/// 将稠密张量列表展平为一个张量
inline at::Tensor flatten_dense_tensors(at::TensorList tensors) {
  // 调用ATen库的函数，将稠密张量列表展平
  return at::flatten_dense_tensors(tensors);
}

/// 将扁平的张量重新展开为张量列表
inline std::vector<at::Tensor> unflatten_dense_tensors(
    const at::Tensor& flat,
    at::TensorList tensors) {
  // 调用ATen库的函数，将扁平的张量重新展开为张量列表
  return at::unflatten_dense_tensors(flat, tensors);
}

struct TensorGroup {
  std::vector<at::Tensor> tensors;  // 张量列表
  size_t size = 0;  // 大小初始化为0

  /// 返回此张量组的类型ID
  size_t type_id() {
    AT_ASSERT(!tensors.empty());  // 断言：张量列表不为空
    return ::torch::utils::type_id(tensors[0]);  // 调用type_id函数返回第一个张量的类型ID
  }

  /// 返回此张量组的选项
  const at::TensorOptions options() {
    AT_ASSERT(!tensors.empty());  // 断言：张量列表不为空
    return tensors[0].options();  // 返回第一个张量的选项
  }
};

// 辅助函数，将张量列表按大小限制分组，并输出这些张量组。
// 如果输入张量是不同类型的张量，它们也将被分成不同的组。
//
// 用户提供的两种分组选项，
//
// 假设size_limit为256，输入张量列表如下：
// tensor_a(fp16 - 128 bytes),
// tensor_b(fp32 - 256 bytes),
// tensor_c(fp16 - 128 bytes),
//
// 当fine_grained == false时：
// 函数将按顺序读取张量列表，并累积足够多的相同数据类型的张量，直到达到size_limit，因此：
// 输出为：{{tensor_a, tensor_c}, {tensor_b}}
//
// 当fine_grained == true时：
// 函数将按顺序读取张量列表，并累积足够多的所有数据类型的张量，直到达到size_limit，然后按数据类型将累积的张量分成不同组，因此：
// 输出为：{{tensor_a}, {tensor_b}, {tensor_c}}
TORCH_API std::vector<TensorGroup> take_tensors(
    at::TensorList tensors,
    size_t size_limit,
    bool fine_grained = false);

// 根据给定的顺序重新排列张量列表中的张量
TORCH_API void reorder_tensors_like(
    std::vector<at::Tensor>& tensors,
    at::TensorList order);

// 将稀疏张量列表展平为一对张量（索引和值）
TORCH_API std::pair<at::Tensor, at::Tensor> flatten_sparse_tensors(
    at::TensorList tensors);

// 将扁平的稀疏张量（索引和值）重新展开为张量列表
TORCH_API std::vector<at::Tensor> unflatten_sparse_tensors(
    const at::Tensor& flat_indices,
    const at::Tensor& flat_values,
    at::TensorList tensors);

} // namespace torch::utils
```