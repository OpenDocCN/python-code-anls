# `.\pytorch\torch\csrc\autograd\input_metadata.cpp`

```
// 包含 Torch 的自动微分模块中的输入元数据定义头文件
#include <torch/csrc/autograd/input_metadata.h>

// TODO: 可能可以将一些从 input_metadata.h 移动到这里的导入，但 function.h 似乎间接依赖于其中一些导入。

namespace torch {
namespace autograd {

namespace {

// 计算变体形状的元数据形状
MetadataShape compute_variant_shape(const at::Tensor& input) {
  // 如果输入是嵌套的且不是 Python 分发的
  if (input.is_nested() && !input.unsafeGetTensorImpl()->is_python_dispatch()) {
    auto nested_size = input._nested_tensor_size();
    return MetadataShape{std::in_place_type<at::Tensor>, nested_size};
  }
  return MetadataShape{std::in_place_type<SymIntSmallVec>, input.sym_sizes()};
}

// 检查张量是否是 Python 分发的
bool is_python_dispatch(const at::Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->is_python_dispatch();
}

// 检查张量是否是 C++ 嵌套张量
bool is_cpp_nested_tensor(const at::Tensor& tensor) {
  return tensor.is_nested() && !is_python_dispatch(tensor);
}

} // namespace

// 输入元数据类的构造函数，初始化选项、形状、是否是张量子类、是否是嵌套
InputMetadata::InputMetadata(
    const at::TensorOptions& options,
    MetadataShape input_shape,
    bool is_tensor_subclass,
    bool is_nested)
    : options_{options},
      shape_{std::move(input_shape)},
      is_tensor_subclass_{is_tensor_subclass},
      is_nested_{is_nested},
      was_default_constructed_{false} {
  // 获取设备并设置流
  auto device_ = options.device();
  stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
}

// 输入元数据类的构造函数，根据张量 t 初始化
InputMetadata::InputMetadata(const at::Tensor& t)
    : InputMetadata(
          t.options(),
          compute_variant_shape(t),
          is_python_dispatch(t),
          t.is_nested()) {}

// 返回与当前形状相同的零张量
at::Tensor InputMetadata::zeros_like() const {
  TORCH_CHECK(
      !is_nested_, "Zeros is not currently supported for nested tensors.")
  return at::zeros_symint(shape_as_dim_vector(), options_);
}

// 可能根据形状减少梯度张量
at::Tensor InputMetadata::maybe_reduce(
    const size_t i,
    at::Tensor grad,
    const std::function<std::string(const std::string&)>& format_error) const {
  auto fail = [&]() {
    const auto message = incompatible_shape_error_message(i, grad);
    TORCH_CHECK(false, format_error(message.str()));
  };

  // 嵌套张量的处理逻辑
  if (is_nested_ || is_cpp_nested_tensor() || grad.is_nested() ||
      ::torch::autograd::is_cpp_nested_tensor(grad)) {
    if (!is_same_shape(grad)) {
      if (is_expandable_to_shape(grad)) {
        return reduce_grad(grad);
      } else {
        fail();
      }
    } else {
      return grad;
    }
  }

  auto shape = shape_as_dim_vector();
  auto desired = grad.sym_sizes();

  size_t ndim = shape.size();
  size_t target_dim = desired.size();
  if (ndim > target_dim) {
    fail();
  }
  bool needs_reduce = false;
  for (const auto i : c10::irange(ndim)) {
    const auto& size = shape[ndim - i - 1];
    const auto& target = desired[target_dim - i - 1];
    // 这里的条件是精心编写的，以便我们能够推断延迟运行时断言
    if (TORCH_GUARD_SIZE_OBLIVIOUS(size.sym_eq(1))) {
      // 如果 size.sym_eq(1) 是 TORCH_GUARD_SIZE_OBLIVIOUS 的结果
      // 注意：一旦 needs_reduce 为 true，可以提前结束，但是因为减少函数会在任何情况下都保护这一点，所以没有必要。
      if (!c10::definitely_true(size.sym_eq(target), __FILE__, __LINE__)) {
        // 如果 size.sym_eq(target) 不是绝对真的，则将 needs_reduce 设为 true
        needs_reduce = true;
      }
    } else {
      // 如果 size.sym_eq(1) 不是 TORCH_GUARD_SIZE_OBLIVIOUS 的结果
      if (!size.sym_eq(target).expect_true(__FILE__, __LINE__)) {
        // 如果 size.sym_eq(target) 不是预期的真值，则触发失败操作
        fail();
      }
    }
  }
  // 如果 ndim 不等于 target_dim，则需要进行减少操作
  if (ndim != target_dim) {
    needs_reduce = true;
  }

  // 如果需要减少操作，则调用 reduce_grad 函数处理 grad，否则直接返回 grad
  if (needs_reduce) {
    return reduce_grad(grad);
  } else {
    return grad;
  }
} // namespace autograd
} // namespace torch



// 结束 autograd 和 torch 命名空间的定义
```