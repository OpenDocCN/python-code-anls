# `.\pytorch\aten\src\ATen\templates\RegisterFunctionalization.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于限定仅包含方法操作符

// ${generated_comment}
// 生成的注释信息，通常是由代码生成工具自动生成的注释

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/FunctionalInverses.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#include <ATen/NativeFunctions.h>
#else
// needed for the meta tensor calls to get stride info in functionalization
#include <ATen/ops/empty_strided_native.h>
// needed for special handling of copy_().
// See Note [functionalizating copy_() and not preserving strides]
#include <ATen/ops/to_ops.h>
#include <ATen/ops/expand_copy_ops.h>

$ops_headers
#endif

namespace at {
namespace functionalization {

// This keyset is used by functionalization when it calls into meta kernels
// to accurately propagate stride metadata.
// Exclude any modes: the purpose of calling into meta kernels is only as an implementation
// detail to perform shape inference, and we don't want any modal keys to run.
// Specifically, we want to prevent functionalization and Python modes from running.
constexpr auto exclude_keys_for_meta_dispatch =
    c10::functorch_transforms_ks |
    c10::DispatchKeySet({
        c10::DispatchKey::FuncTorchDynamicLayerBackMode,
        c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
        c10::DispatchKey::Python,
        c10::DispatchKey::PreDispatch,

    });

// Helper around at::has_internal_overlap.
// The ATen util is used in hot-path eager mode: it's always fast,
// but might return TOO_HARD sometimes.
// During functionalization, we're ok taking a bit longer
// to detect memory overlap.
inline bool has_internal_overlap_helper(const at::Tensor t) {
  auto has_overlap = at::has_internal_overlap(t);
  if (has_overlap == at::MemOverlap::Yes) return true;
  if (has_overlap == at::MemOverlap::No) return false;
  return false;
}

// Convert a tensor to its meta representation for functionalization purposes.
inline Tensor to_meta(const Tensor& t) {
    if (!t.defined()) return t;
    // Create a meta tensor with symbolic sizes and strides, preserving dtype and layout.
    return at::native::empty_strided_meta_symint(t.sym_sizes(), t.sym_strides(),
        /*dtype=*/c10::make_optional(t.scalar_type()), /*layout=*/c10::make_optional(t.layout()),
        /*device=*/c10::make_optional(c10::Device(kMeta)), /*pin_memory=*/c10::nullopt);
}

// Convert an optional tensor to its meta representation, preserving optionality.
inline std::optional<Tensor> to_meta(const std::optional<Tensor>& t) {
  if (t.has_value()) {
    return c10::make_optional<Tensor>(to_meta(*t));
  }
  return c10::nullopt;
}

// Convert a list of tensors to their meta representations.
inline std::vector<Tensor> to_meta(at::ITensorListRef t_list) {
  std::vector<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto& tensor : t_list) {
    outputs.push_back(to_meta(tensor));
  }
  return outputs;
}

// Convert a C++ list of tensors to their meta representations.
inline c10::List<Tensor> to_meta(const c10::List<Tensor>& t_list) {
  c10::List<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(to_meta(t_list[i]));
  }
  return outputs;
}

} // namespace functionalization
} // namespace at
// 将输入列表中的每个元素转换为元数据形式，并返回转换后的列表
inline c10::List<::std::optional<Tensor>> to_meta(const c10::List<::std::optional<Tensor>>& t_list) {
  // 创建一个空的输出列表，预留与输入列表相同数量的空间
  c10::List<::std::optional<Tensor>> outputs;
  outputs.reserve(t_list.size());
  // 遍历输入列表的每个元素
  for (const auto i : c10::irange(t_list.size())) {
    // 调用 to_meta 函数将当前元素转换为元数据形式，并添加到输出列表中
    outputs.push_back(to_meta(t_list[i]));
  }
  // 返回转换后的输出列表
  return outputs;
}

// 匿名命名空间，用于定义不对外部可见的函数和变量

namespace functionalization {

// Torch 库的函数定义

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  ${func_registrations};
}

}  // namespace functionalization

// 结束匿名命名空间

} // namespace at
```