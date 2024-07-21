# `.\pytorch\torch\csrc\api\include\torch\nested.h`

```
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ATen_fwd.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>
#include <algorithm>

namespace torch {
namespace nested {

/// Nested tensor
///
/// See
/// https://pytorch.org/docs/main/nested.html#torch.nested.nested_tensor
///
/// ```
// implemented on python object to allow torch.nested.nested_tensor to be
// constructed with arbitrarily nested python objects - for now, only arbitrary
// python lists and lists of Tensors
// See torch/csrc/autograd/python_nested_functions_manual.cpp for Python
// implementation
// See here for C++ implementation
// 定义一个内联函数，用于创建嵌套张量，接受一个张量列表和选项参数，返回一个张量
inline at::Tensor nested_tensor(
    at::TensorList nested_tensor_data,
    const at::TensorOptions& options = {}) {
  // 调用底层函数 _nested_tensor_from_tensor_list，从张量列表创建嵌套张量
  auto out = at::_nested_tensor_from_tensor_list(
      nested_tensor_data,
      c10::typeMetaToScalarType(options.dtype()),
      c10::nullopt,
      options.device(),
      options.pinned_memory());
  // 如果选项中要求梯度追踪，则设置输出张量的 requires_grad 属性为 true
  if (options.has_requires_grad() && options.requires_grad()) {
    out.requires_grad_(true);
  }
  // 返回创建的嵌套张量
  return out;
}

// 定义另一个重载函数，接受 TensorDataContainer 数组作为输入
inline at::Tensor nested_tensor(
    at::ArrayRef<detail::TensorDataContainer> nested_tensor_data,
    const at::TensorOptions& options = {}) {
  // 对每个 TensorDataContainer 检查其是否为初始化列表，如果不是则抛出异常
  for (const auto& tdc : nested_tensor_data) {
    TORCH_CHECK(
        tdc.is_init_list(),
        "nested_tensor() not implemented for these parameters");
  }
  // 构造一个 TensorList，使用 nested_tensor_data 中的数据转换为张量
  std::vector<at::Tensor> tensor_list(nested_tensor_data.size());
  std::transform(
      nested_tensor_data.begin(),
      nested_tensor_data.end(),
      tensor_list.begin(),
      [&](const detail::TensorDataContainer& tdc) {
        return tdc.convert_to_tensor(options);
      });
  // 调用底层函数 _nested_tensor_from_tensor_list，从张量列表创建嵌套张量
  auto out = at::_nested_tensor_from_tensor_list(
      tensor_list,
      c10::typeMetaToScalarType(options.dtype()),
      c10::nullopt,
      options.device(),
      options.pinned_memory());
  // 如果选项中要求梯度追踪，则设置输出张量的 requires_grad 属性为 true
  if (options.has_requires_grad() && options.requires_grad()) {
    out.requires_grad_(true);
  }
  // 返回创建的嵌套张量
  return out;
}

/// As Nested Tensor
///
/// See
/// https://pytorch.org/docs/main/nested.html#torch.nested.as_nested_tensor
///
/// ```
// 定义一个内联函数，将输入列表转换为嵌套张量
inline at::Tensor as_nested_tensor(
    at::TensorList list,
    std::optional<at::ScalarType> dtype = c10::nullopt,
    std::optional<at::Device> device = c10::nullopt) {
  // 调用底层函数 _nested_tensor_from_tensor_list，从张量列表创建嵌套张量
  return at::_nested_tensor_from_tensor_list(
      list, dtype, c10::nullopt, device, c10::nullopt);
}

/// Nested to padded tensor
///
/// See
/// https://pytorch.org/docs/main/nested.html#torch.nested.to_padded_tensor
///
/// ```
// 定义一个内联函数，将嵌套张量转换为填充张量
inline at::Tensor to_padded_tensor(
    const at::Tensor& self,
    double padding,
    at::OptionalIntArrayRef output_size = c10::nullopt) {
  // 调用 at::nested_to_padded_tensor 函数，将嵌套张量 self 转换为填充张量
  return at::nested_to_padded_tensor(self, padding, output_size);
}

} // namespace nested
} // namespace torch
```