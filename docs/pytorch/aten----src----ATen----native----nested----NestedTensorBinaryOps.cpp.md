# `.\pytorch\aten\src\ATen\native\nested\NestedTensorBinaryOps.cpp`

```
#include <ATen/native/nested/NestedTensorMath.h>
#include  <ATen/native/nested/NestedTensorBinaryOps.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include <tuple>

namespace at {
namespace native {

// 定义分发函数的分派点，并注册空的 CPU 分发函数
DEFINE_DISPATCH(nested_dense_elementwise_stub);
REGISTER_NO_CPU_DISPATCH(nested_dense_elementwise_stub);

// 函数：获取两个嵌套张量的实现指针
std::pair<NestedTensorImpl*, NestedTensorImpl*>
static get_elementwise_nested_tensor_impl(
    const Tensor& self,
    const Tensor& other,
    const std::string& op_name) {
  // 检查输入张量的嵌套属性
  if (self.is_nested() && !(other.is_nested())) {
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a nested self and non-nested other");
  } else if (!(self.is_nested()) && other.is_nested()) {
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a non-nested self and nested other");
  } else if (!(self.is_nested()) || !(other.is_nested())) {
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a non-nested self and non-nested other");
  }

  // 获取嵌套张量的实现指针
  auto self_ptr = get_nested_tensor_impl(self);
  auto other_ptr = get_nested_tensor_impl(other);

  // 检查张量维度是否匹配
  TORCH_CHECK(
      self.dim() == other.dim(),
      op_name,
      " does not support broadcasting when given a NestedTensor");
  // 检查嵌套张量的大小是否相等
  TORCH_CHECK(
      at::equal(
          self_ptr->get_nested_sizes(),
          other_ptr->get_nested_sizes()),
      op_name,
      " does not support broadcasting when given a NestedTensor");
  // 检查嵌套张量的步长是否相等
  TORCH_CHECK(
      at::equal(
          self_ptr->get_nested_strides(),
          other_ptr->get_nested_strides()),
      op_name,
      " requires strides to match when given NestedTensors");
  // 检查嵌套张量的存储偏移是否相等
  const auto self_offsets = self_ptr->get_storage_offsets();
  int64_t *self_offsets_ptr = self_offsets.data_ptr<int64_t>();
  int64_t *other_offsets_ptr = other_ptr->get_storage_offsets().data_ptr<int64_t>();
  bool offsets_match = true;
  for (auto i = 0; i < self_offsets.size(0); i++) {
    offsets_match = offsets_match && (self_offsets_ptr[i] == other_offsets_ptr[i]);
  }
  TORCH_CHECK(
      offsets_match,
      op_name,
      " requires offsets to match when given NestedTensors");
  // 返回嵌套张量的实现指针对
  return std::make_pair(self_ptr, other_ptr);
}

// 模板函数：应用于嵌套张量和张量之间的元素级操作
template <typename Func>
Tensor NestedTensor_elementwise_Tensor(
    const Tensor& self,
    const Tensor& other,
    const std::string& op_name,
    bool supports_striding,
    Func f) {
  // 获取连续的张量副本
  Tensor self_contiguous = self;
  Tensor other_contiguous = other;
  // 如果 self 是标量
  if (!self.is_nested() && self.dim() == 0 && self.numel() == 1) {
    // 获取其他嵌套张量的实现指针
    auto other_impl = get_nested_tensor_impl(other);
    // 如果 other 是一个嵌套张量并且不是标量（即具有多个维度和元素），则执行以下操作
    return wrap_buffer(
      // 使用自定义函数 f 处理 self 和 other 的不安全存储作为张量的数据
      f(self, other_impl->get_unsafe_storage_as_tensor()),
      // 克隆 other 的嵌套大小（nested sizes）
      other_impl->get_nested_sizes().clone(),
      // 克隆 other 的嵌套步长（nested strides）
      other_impl->get_nested_strides().clone(),
      // 获取 other 的存储偏移量（storage offsets）
      other_impl->get_storage_offsets()
    );
  }
  // 如果 other 是一个标量
  // 且同时具有零维度和一个元素
  if (!other.is_nested() && other.dim() == 0 && other.numel() == 1) {
    auto self_impl = get_nested_tensor_impl(self);
    return wrap_buffer(
      // 使用自定义函数 f 处理 self_impl 的不安全存储作为张量和 other 的标量
      f(self_impl->get_unsafe_storage_as_tensor(), other),
      // 克隆 self_impl 的嵌套大小（nested sizes）
      self_impl->get_nested_sizes().clone(),
      // 克隆 self_impl 的嵌套步长（nested strides）
      self_impl->get_nested_strides().clone(),
      // 获取 self_impl 的存储偏移量（storage offsets）
      self_impl->get_storage_offsets()
    );
  }
  // 当 other 是稠密张量时的特殊情况（目前仅适用于 CUDA）
  if (self.is_nested() && !other.is_nested() && self.is_cuda() && other.is_cuda()) {
    auto self_ptr = get_nested_tensor_impl(self);
    auto other_ = other;
    // 检查 [B, *, D], [B, 1, D] 的情况 -> 使用自定义内核
    // TODO: 这个 if 语句比较丑陋，希望近期内能去除它
    bool is_broadcastable_3d = (
        self_ptr->dim() == 3 &&
        other.dim() == 3 &&
        self_ptr->size(0) == other.size(0) &&
        other.size(1) == 1 &&
        self_ptr->opt_size(2).has_value() &&
        self_ptr->opt_size(2).value() == other.size(2));
    // 检查 [B, *], [B, 1] 的情况 -> 将其视为 3D，即 [B, *, 1], [B, 1, 1]
    bool is_broadcastable_2d = (
        self_ptr->dim() == 2 &&
        other.dim() == 2 &&
        self_ptr->size(0) == other.size(0) &&
        other.size(1) == 1);
    if(is_broadcastable_2d) {
        other_ = other.unsqueeze(-1);
        is_broadcastable_3d = true;
    }

    if (is_broadcastable_3d) {
      // 将 self 转换为连续张量
      self_contiguous = self.contiguous();
      self_ptr = get_nested_tensor_impl(self_contiguous);
      // 获取 self 的缓冲区（buffer）、嵌套大小（nested sizes）
      const auto self_buffer = self_ptr->get_buffer();
      const auto self_sizes = self_ptr->get_nested_sizes();
      // 创建一个与 self_buffer 相似的空张量
      auto result_buffer = at::empty_like(self_buffer);
      // 使用自定义的嵌套稠密元素操作函数，根据操作名操作 self、other_ 并将结果包装为张量
      auto result = wrap_buffer(result_buffer, self_sizes);
      if (op_name == "add") {
        nested_dense_elementwise_stub(self.device().type(), result, self, other_, NESTED_DENSE_OP::ADD);
      } else if (op_name == "mul") {
        nested_dense_elementwise_stub(self.device().type(), result, self, other_, NESTED_DENSE_OP::MUL);
      } else {
        TORCH_CHECK(false, "Unsupported nested dense elementwise op: ", op_name, ".");
      }
      return result;
    }

    // 检查 [B, C, *, *], [C, 1, 1] 的情况
    bool is_broadcastable_4d_3d = (
        self_ptr->dim() == 4 &&
        other.dim() == 3 &&
        self_ptr->opt_size(1).has_value() &&
        self_ptr->size(1) == other.size(0) &&
        other.size(1) == 1 &&
        other.size(2) == 1);
    if (is_broadcastable_4d_3d) {
      // 创建一个张量列表 results，存储对 self 的解绑操作结果与 other 的函数 f 结果
      std::vector<Tensor> results;
      for (auto t : self.unbind()) {
        results.push_back(f(t, other));
      }
      // 将 results 转换为嵌套张量并返回
      return at::_nested_tensor_from_tensor_list(results);
    }
    // 使用 TORCH_CHECK 宏来进行断言检查，如果条件为 false，则输出错误信息并终止程序
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a nested self and non-nested other for op: ",
        op_name,
        ".");
  }

  // 定义两个指向 NestedTensorImpl 的指针，并初始化为 nullptr
  NestedTensorImpl* self_impl = nullptr;
  NestedTensorImpl* other_impl = nullptr;

  // 根据是否支持 stride，选择是否需要对 self 和 other 进行连续性处理，得到连续的张量
  self_contiguous = supports_striding ? self.contiguous() : self;
  other_contiguous = supports_striding ? other.contiguous() : other;

  // 调用 get_elementwise_nested_tensor_impl 函数，获取 self 和 other 的 NestedTensorImpl 对象
  std::tie(self_impl, other_impl) =
      get_elementwise_nested_tensor_impl(self_contiguous, other_contiguous, op_name);
  
  // 断言调试模式下 self_impl 和 other_impl 必须有效，否则会触发内部断言错误
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self_impl);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other_impl);

  // 调用 f 函数对 self_impl 和 other_impl 的存储数据进行操作，将结果封装成 NestedTensor 并返回
  return wrap_buffer(
      f(self_impl->get_unsafe_storage_as_tensor(),
        other_impl->get_unsafe_storage_as_tensor()),
      self_impl->get_nested_sizes(),
      self_impl->get_nested_strides(),
      self_impl->get_storage_offsets());
// 使用Tensor类型的self、other和alpha参数，执行elementwise_Tensor操作，返回操作结果Tensor
Tensor NestedTensor_add_Tensor(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  // 调用NestedTensor_elementwise_Tensor函数，执行add操作，支持跨步(striding)
  return NestedTensor_elementwise_Tensor(
      self, other, "add", true /* supports_striding*/, [alpha](const Tensor& b1, const Tensor& b2) {
        // 执行Tensor的add操作，使用参数alpha作为标量因子
        return at::add(b1, b2, alpha);
      });
}

// 使用Tensor类型的self、other和alpha参数，执行elementwise_Tensor操作，返回操作结果Tensor
Tensor NestedTensor_sub_Tensor(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  // 调用NestedTensor_elementwise_Tensor函数，执行sub操作，支持跨步(striding)
  return NestedTensor_elementwise_Tensor(
      self, other, "sub", true /* supports_striding*/, [alpha](const Tensor& b1, const Tensor& b2) {
        // 执行Tensor的sub操作，使用参数alpha作为标量因子
        return at::sub(b1, b2, alpha);
      });
}

// 使用Tensor类型的self和other参数，执行elementwise_Tensor操作，返回操作结果Tensor
Tensor NestedTensor_mul_Tensor(const Tensor& self, const Tensor& other) {
  // 调用NestedTensor_elementwise_Tensor函数，执行mul操作，不支持跨步(striding)
  return NestedTensor_elementwise_Tensor(
      self, other, "mul", false /* supports_striding*/, [](const Tensor& b1, const Tensor& b2) {
        // 执行Tensor的mul操作
        return at::mul(b1, b2);
      });
}

// 仅用于C++端，对Python传入的标量进行转换为Tensor后执行mul_Tensor操作
Tensor NestedTensor_mul_Scalar(const Tensor& self, const Scalar& other) {
  // 调用wrapped_scalar_tensor函数将标量转换为Tensor，然后执行mul_Tensor操作
  return NestedTensor_mul_Tensor(self, wrapped_scalar_tensor(other));
}

// 使用Tensor类型的self和other参数，执行elementwise_Tensor操作，返回操作结果Tensor
Tensor NestedTensor_div_Tensor(const Tensor& self, const Tensor& other) {
  // 调用NestedTensor_elementwise_Tensor函数，执行div操作，不支持跨步(striding)
  return NestedTensor_elementwise_Tensor(
      self, other, "div", false /* supports_striding*/, [](const Tensor& b1, const Tensor& b2) {
        // 执行Tensor的div操作
        return at::div(b1, b2);
      });
}

// 仅用于C++端，对Python传入的标量进行转换为Tensor后执行div_Tensor操作
Tensor NestedTensor_div_Scalar(const Tensor& self, const Scalar& other) {
  // 调用wrapped_scalar_tensor函数将标量转换为Tensor，然后执行div_Tensor操作
  return NestedTensor_div_Tensor(self, wrapped_scalar_tensor(other));
}

// 使用Tensor类型的self、mask和value参数，执行elementwise_Tensor操作，返回操作结果Tensor
Tensor NestedTensor_masked_fill(
    const Tensor& self,
    const Tensor& mask,
    const Scalar& value) {
  // 调用NestedTensor_elementwise_Tensor函数，执行masked_fill操作，不支持跨步(striding)
  return NestedTensor_elementwise_Tensor(
      self, mask, "masked_fill", false /* supports_striding*/, [value](const Tensor& b1, const Tensor& b2) {
        // 执行Tensor的masked_fill操作，使用参数value作为填充值
        return at::masked_fill(b1, b2, value);
      });
}

// 使用Func类型的参数f，执行elementwise操作，修改Tensor self的内容
template <typename Func>
Tensor& NestedTensor_elementwise__Tensor(
    Tensor& self,
    const Tensor& other,
    const std::string& op_name,
    Func f) {
  // 如果self是标量，且非嵌套（nested）且维度为0且元素数为1，则执行特定操作
  if (!self.is_nested() && self.dim() == 0 && self.numel() == 1) {
    // 获取other的NestedTensor实现，然后调用函数f修改self
    auto other_impl = get_nested_tensor_impl(other);
    f(self, other_impl->get_buffer());
    return self;
  }
  // 如果other是标量，且非嵌套（nested）且维度为0且元素数为1，则执行特定操作
  if (!other.is_nested() && other.dim() == 0 && other.numel() == 1) {
    // 获取self的NestedTensor实现，然后调用函数f修改self
    auto self_impl = get_nested_tensor_impl(self);
    f(self_impl->get_buffer(), other);
    return self;
  }
  // 获取self和other的elementwise NestedTensor实现，然后调用函数f修改self
  NestedTensorImpl* self_impl = nullptr;
  NestedTensorImpl* other_impl = nullptr;
  std::tie(self_impl, other_impl) =
      get_elementwise_nested_tensor_impl(self, other, op_name);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self_impl);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other_impl);
  const auto& nt_self = *self_impl;
  const auto& nt_other = *other_impl;
  f(nt_self.get_buffer().view({-1}), nt_other.get_buffer().view({-1}));
  return self;
}

// 使用Tensor类型的self和other参数，执行elementwise_Tensor操作，修改Tensor self的内容
Tensor& NestedTensor_add__Tensor(
    Tensor& self,
    const Tensor& other,
    // 定义一个函数，实现 NestedTensor 的元素级加法操作 "add_"
    const Scalar& alpha) {
  // 调用 NestedTensor_elementwise__Tensor 函数，传入 self 和 other 张量，并指定操作为 "add_"
  return NestedTensor_elementwise__Tensor(
      self, other, "add_", 
      // 在 lambda 函数中定义元素级加法的具体操作，使用参数 alpha
      [alpha](const Tensor& b1, const Tensor& b2) {
        // 调用 b1 的 in-place 加法函数 add_，传入 b2 和 alpha，并返回结果
        return b1.add_(b2, alpha);
      });
}

// 结束 namespace native
} // namespace native

// 结束 namespace at
} // namespace at


这段代码是 C++ 中的命名空间定义和结束部分。命名空间用于组织代码，防止命名冲突，并提供了作用域限定。在这里，代码定义了两个命名空间：`native` 和 `at`。其中：

- `Native` 命名空间包含了一些与张量操作相关的函数实现，如 `NestedTensor_mul__Tensor`、`NestedTensor_mul__Scalar`、`fill_nested_`、`ge_scalar_nested`、`gt_scalar_nested`、`eq_scalar_nested` 等。
- `at` 命名空间在 `native` 命名空间内部，可能表示其所属的更高级命名空间或库的一部分。

每个命名空间的结束都用 `}` 表示，并且通过注释指出了命名空间的结束。
```