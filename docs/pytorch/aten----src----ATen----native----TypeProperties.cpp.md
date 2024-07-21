# `.\pytorch\aten\src\ATen\native\TypeProperties.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/TypeProperties.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_has_compatible_shallow_copy_type_native.h>
#include <ATen/ops/_is_zerotensor_native.h>
#include <ATen/ops/can_cast_native.h>
#include <ATen/ops/is_complex_native.h>
#include <ATen/ops/is_conj_native.h>
#include <ATen/ops/is_distributed_native.h>
#include <ATen/ops/is_floating_point_native.h>
#include <ATen/ops/is_inference_native.h>
#include <ATen/ops/is_neg_native.h>
#include <ATen/ops/is_signed_native.h>
#include <ATen/ops/promote_types_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/result_type_native.h>
#include <ATen/ops/type_as_native.h>
#endif

namespace at::native {

// 检查张量是否处于分布式状态，总是返回false
bool is_distributed(const Tensor& self) {
  return false;
}

// 检查张量是否为复数类型，调用Tensor的成员函数is_complex()
bool is_complex(const Tensor& self) {
  return self.is_complex();
}

// 检查张量是否为浮点数类型，调用Tensor的成员函数is_floating_point()
bool is_floating_point(const Tensor& self) {
  return self.is_floating_point();
}

// 检查张量是否为推断类型，调用Tensor的成员函数is_inference()
bool is_inference(const Tensor& self) {
  return self.is_inference();
}

// 检查张量是否为有符号类型，调用Tensor的成员函数is_signed()
bool is_signed(const Tensor &self) {
  return self.is_signed();
}

// 检查张量是否为零张量，调用Tensor的成员函数_is_zerotensor()
bool _is_zerotensor(const Tensor& self) {
  return self._is_zerotensor();
}

// 检查张量是否为共轭类型，调用Tensor的成员函数is_conj()
bool is_conj(const Tensor& self) {
  return self.is_conj();
}

// 检查张量是否为负数类型，调用Tensor的成员函数is_neg()
bool is_neg(const Tensor& self) {
  return self.is_neg();
}

// 检查两个张量是否具有兼容的张量类型，以便可以将from的TensorImpl复制到self
bool _has_compatible_shallow_copy_type(const Tensor& self, const Tensor& from) {
  return self.unsafeGetTensorImpl()->has_compatible_shallow_copy_type(
      from.key_set());
}

// 将张量转换为与另一个张量相同类型，调用Tensor的成员函数to()
Tensor type_as(const Tensor& self, const Tensor& other) {
  return self.to(other.options());
}

// 促进类型时跳过未定义类型，返回两个ScalarType中不为Undefined的类型
static inline ScalarType promote_skip_undefined(ScalarType a, ScalarType b) {
  if (a == ScalarType::Undefined) {
    return b;
  }
  if (b == ScalarType::Undefined) {
    return a;
  }
  return promoteTypes(a, b);
}

// 合并类型类别，返回高级和低级ScalarType之间的类型组合
static inline ScalarType combine_categories(ScalarType higher, ScalarType lower) {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if(isComplexType(higher)) {
    return higher;
  } else if (isComplexType(lower)) {
    // 如果高级类型是浮点类型，则保留高级值类型。
    if (isFloatingType(higher)) {
      return toComplexType(higher);
    }
    // 对于整数输入，低级复数优先。
    return lower;
  } else if (isFloatingType(higher)) {
    return higher;
  }
  if (higher == ScalarType::Bool || isFloatingType(lower)) {
    return promote_skip_undefined(higher, lower);
  }
  if (higher != ScalarType::Undefined) {
    return higher;
  }
  return lower;
}

// 更新结果类型状态，根据张量是否已定义来更新状态
ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state) {
  if (!tensor.defined()) {
    return in_state;
  }
  ResultTypeState new_state = in_state;
  ScalarType current = tensor.scalar_type();
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    # 如果当前张量类型为复数类型，则将其转换为默认的复数数据类型
    if(isComplexType(current)) {
      current = typeMetaToScalarType(at::get_default_complex_dtype());
    }
    # 如果当前张量类型为浮点数类型，则将其转换为默认的浮点数数据类型
    else if(isFloatingType(current)) {
      current = typeMetaToScalarType(at::get_default_dtype());
    }
  }
  # 如果张量的维度大于 0
  if ( tensor.dim() > 0 ) {
    # 使用 promote_skip_undefined 函数将当前状态中的 dimResult 提升为 new_state 中的结果，并跳过未定义情况
    new_state.dimResult = promote_skip_undefined(in_state.dimResult, current);
  } else if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    # 如果张量的实现是包装数字类型
    # 使用 promote_skip_undefined 函数将当前状态中的 wrappedResult 提升为 new_state 中的结果，并跳过未定义情况
    new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
  } else {
    # 否则（即张量维度为 0 且未包装数字类型）
    # 使用 promote_skip_undefined 函数将当前状态中的 zeroResult 提升为 new_state 中的结果，并跳过未定义情况
    new_state.zeroResult = promote_skip_undefined(in_state.zeroResult, current);
  }
  # 返回更新后的状态 new_state
  return new_state;
}

// 更新结果类型状态，根据标量和当前状态
ResultTypeState update_result_type_state(const Scalar& scalar, const ResultTypeState& in_state) {
  // 复制输入状态作为新状态
  ResultTypeState new_state = in_state;
  // 获取当前标量的类型
  ScalarType current = scalar.type();
  // 如果当前类型是复数类型
  if (isComplexType(current)) {
    // 将当前类型设置为默认复数数据类型
    current = typeMetaToScalarType(at::get_default_complex_dtype());
  } else if (isFloatingType(current)) {
    // 如果当前类型是浮点类型，将当前类型设置为默认浮点数据类型
    current = typeMetaToScalarType(at::get_default_dtype());
  }
  // 更新状态中的 wrappedResult，使用当前类型进行提升，跳过未定义的类型
  new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
  // 返回更新后的状态
  return new_state;
}

// 计算结果类型，合并各种类别的结果
ScalarType result_type(const ResultTypeState& in_state) {
  // 合并维度结果、零结果和封装结果的类别，返回合并后的标量类型
  return combine_categories(in_state.dimResult, combine_categories(in_state.zeroResult, in_state.wrappedResult));
}

// 计算张量列表的结果类型
ScalarType result_type(ITensorListRef tensors) {
  // 初始化结果类型状态
  ResultTypeState state = {};
  // 遍历张量列表，更新状态
  for (const Tensor& tensor : tensors) {
    state = update_result_type_state(tensor, state);
  }
  // 返回更新后的状态的结果类型
  return result_type(state);
}

// 计算两个张量的结果类型
ScalarType result_type(const Tensor &tensor, const Tensor &other) {
  // 初始化结果类型状态
  ResultTypeState state = {};
  // 分别使用两个张量更新状态
  state = update_result_type_state(tensor, state);
  state = update_result_type_state(other, state);
  // 返回更新后的状态的结果类型
  return result_type(state);
}

// 计算张量和标量的结果类型
ScalarType result_type(const Tensor &tensor, const Scalar& other) {
  // 初始化结果类型状态
  ResultTypeState state = {};
  // 分别使用张量和标量更新状态
  state = update_result_type_state(tensor, state);
  state = update_result_type_state(other, state);
  // 返回更新后的状态的结果类型
  return result_type(state);
}

// 计算标量和张量的结果类型
ScalarType result_type(const Scalar& scalar, const Tensor &tensor) {
  // 调用 ATen 库中的函数计算标量和张量的结果类型
  return at::result_type(tensor, scalar);
}

// 计算两个标量的结果类型
ScalarType result_type(const Scalar& scalar1, const Scalar& scalar2) {
  // 初始化结果类型状态
  ResultTypeState state = {};
  // 分别使用两个标量更新状态
  state = update_result_type_state(scalar1, state);
  state = update_result_type_state(scalar2, state);
  // 返回更新后的状态的结果类型
  return result_type(state);
}

// 判断是否可以将 from_ 类型转换为 to 类型
bool can_cast(const at::ScalarType from_, const at::ScalarType to) {
  // 调用 ATen 库中的函数判断是否可以转换类型
  return at::canCast(from_, to);
}

// 提升两个类型
ScalarType promote_types(ScalarType type1, ScalarType type2) {
  // 使用 ATen 库中的函数提升两个类型
  ScalarType ret = promoteTypes(type1, type2);
  // 检查结果类型不是未定义类型
  TORCH_CHECK(ret != ScalarType::Undefined, "Promotion from ", type1, " and ", type2, " is unsupported.");
  // 返回提升后的类型
  return ret;
}

} // namespace at::native
```