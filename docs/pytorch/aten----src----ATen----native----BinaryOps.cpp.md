# `.\pytorch\aten\src\ATen\native\BinaryOps.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/BinaryOps.h>

#include <type_traits>
#include <utility>

#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorMeta.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_add_relu_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/_test_serialization_subcmul_native.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/and_native.h>
#include <ATen/ops/arctan2_native.h>
#include <ATen/ops/atan2.h>
#include <ATen/ops/atan2_native.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_and_native.h>
#include <ATen/ops/bitwise_left_shift.h>
#include <ATen/ops/bitwise_left_shift_native.h>
#include <ATen/ops/bitwise_or.h>
#include <ATen/ops/bitwise_or_native.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/bitwise_right_shift_native.h>
#include <ATen/ops/bitwise_xor.h>
#include <ATen/ops/bitwise_xor_native.h>
#include <ATen/ops/copysign.h>
#include <ATen/ops/copysign_native.h>
#include <ATen/ops/div.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/divide_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/floor_divide_native.h>
#include <ATen/ops/fmax_native.h>
#include <ATen/ops/fmin_native.h>
#include <ATen/ops/fmod.h>
#include <ATen/ops/fmod_native.h>
#include <ATen/ops/full.h>
#include <ATen/ops/gcd_native.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/greater_equal_native.h>
#include <ATen/ops/greater_native.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/heaviside_native.h>
#include <ATen/ops/hypot_native.h>
#include <ATen/ops/igamma.h>
#include <ATen/ops/igamma_native.h>
#include <ATen/ops/igammac.h>
#include <ATen/ops/igammac_native.h>
#include <ATen/ops/lcm_native.h>
#include <ATen/ops/ldexp.h>
#include <ATen/ops/ldexp_native.h>
#include <ATen/ops/le.h>
#include <ATen/ops/le_native.h>
#include <ATen/ops/less_equal_native.h>
#include <ATen/ops/less_native.h>
#include <ATen/ops/linalg_cross_native.h>
#include <ATen/ops/linalg_cross_ops.h>
#include <ATen/ops/logaddexp2_native.h>
#include <ATen/ops/logaddexp_native.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_and_native.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/logical_or_native.h>
#include <ATen/ops/logical_xor.h>
#include <ATen/ops/logical_xor_native.h>
#include <ATen/ops/logit_backward_native.h>
#include <ATen/ops/lshift_native.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/lt_native.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/maximum_native.h>
#include <ATen/ops/min_native.h>



// 在此文件中导入了大量的 ATen 操作函数头文件，以支持各种张量运算
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义以便只包含方法运算符
#include <ATen/native/BinaryOps.h>
// 包含了 ATen 的二元操作的本地实现

#include <type_traits>
// 引入类型特性库，用于编译时类型信息查询
#include <utility>
// 引入实用工具库，包含了各种常用功能的实现

#include <ATen/core/Tensor.h>
// 引入张量核心定义头文件
#include <ATen/ScalarOps.h>
// 引入标量操作的头文件
#include <ATen/TensorIterator.h>
// 引入张量迭代器的头文件
#include <ATen/TensorOperators.h>
// 引入张量操作的头文件
#include <ATen/TensorMeta.h>
// 引入张量元数据的头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 如果没有按操作符单独分离头文件，则引入通用函数头文件
#include <ATen/NativeFunctions.h>
// 引入原生函数头文件
#else
#include <ATen/ops/_add_relu_native.h>
// 引入特定操作 _add_relu_native 的头文件
#include <ATen/ops/_efficientzerotensor.h>
// 引入特定操作 _efficientzerotensor 的头文件
#include <ATen/ops/_test_serialization_subcmul_native.h>
// 引入特定操作 _test_serialization_subcmul_native 的头文件
#include <ATen/ops/_to_copy.h>
// 引入特定操作 _to_copy 的头文件
#include <ATen/ops/add.h>
// 引入特定操作 add 的头文件
#include <ATen/ops/add_native.h>
// 引入特定操作 add_native 的头文件
#include <ATen/ops/add_ops.h>
// 引入特定操作 add_ops 的头文件
#include <ATen/ops/and_native.h>
// 引入特定操作 and_native 的头文件
#include <ATen/ops/arctan2_native.h>
// 引入特定操作 arctan2_native 的头文件
#include <ATen/ops/atan2.h>
// 引入特定操作 atan2 的头文件
#include <ATen/ops/atan2_native.h>
// 引入特定操作 atan2_native 的头文件
#include <ATen/ops/bitwise_and.h>
// 引入特定操作 bitwise_and 的头文件
#include <ATen/ops/bitwise_and_native.h>
// 引入特定操作 bitwise_and_native 的头文件
#include <ATen/ops/bitwise_left_shift.h>
// 引入特定操作 bitwise_left_shift 的头文件
#include <ATen/ops/bitwise_left_shift_native.h>
// 引入特定操作 bitwise_left_shift_native 的头文件
#include <ATen/ops/bitwise_or.h>
// 引入特定操作 bitwise_or 的头文件
#include <ATen/ops/bitwise_or_native.h>
// 引入特定操作 bitwise_or_native 的头文件
#include <ATen/ops/bitwise_right_shift.h>
// 引入特定操作 bitwise_right_shift 的头文件
#include <ATen/ops/bitwise_right_shift_native.h>
// 引入特定操作 bitwise_right_shift_native 的头文件
#include <ATen/ops/bitwise_xor.h>
// 引入特定操作 bitwise_xor 的头文件
#include <ATen/ops/bitwise_xor_native.h>
// 引入特定操作 bitwise_xor_native 的头文件
#include <ATen/ops/copysign.h>
// 引入特定操作 copysign 的头文件
#include <ATen/ops/copysign_native.h>
// 引入特定操作 copysign_native 的头文件
#include <ATen/ops/div.h>
// 引入特定操作 div 的头文件
#include <ATen/ops/div_native.h>
// 引入特定操作 div_native 的头文件
#include <ATen/ops/div_ops.h>
// 引入特定操作 div_ops 的头文件
#include <ATen/ops/divide_native.h>
// 引入特定操作 divide_native 的头文件
#include <ATen/ops/empty.h>
// 引入特定操作 empty 的头文件
#include <ATen/ops/eq_native.h>
// 引入特定操作 eq_native 的头文件
#include <ATen/ops/floor_divide.h>
// 引入特定操作 floor_divide 的头文件
#include <ATen/ops/floor_divide_native.h>
// 引入特定操作 floor_divide_native 的头文件
#include <ATen/ops/fmax_native.h>
// 引入特定操作 fmax_native 的头
#ifndef ATEN_META_OPERATORS_H
#define ATEN_META_OPERATORS_H

namespace at::meta {

// 定义名为 add 的元函数，接受两个张量和一个标量作为参数
TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  // 调用函数构建二元操作，输出结果可能由外部获取
  build_borrowing_binary_op(maybe_get_output(), self, other);
  // 调用本地函数检查标量 alpha 的数据类型
  native::alpha_check(dtype(), alpha);
}

// 定义名为 sub 的元函数，接受两个张量和一个标量作为参数
TORCH_META_FUNC2(sub, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
// 调用 native 命名空间下的 sub_check 函数，对 self 和 other 参数进行检查
native::sub_check(self, other);
// 调用 build_borrowing_binary_op 函数，生成一个二元操作，操作数是 maybe_get_output()、self 和 other
build_borrowing_binary_op(maybe_get_output(), self, other);
// 调用 native 命名空间下的 alpha_check 函数，检查当前张量的数据类型以及 alpha 参数
native::alpha_check(dtype(), alpha);

TORCH_META_FUNC2(mul, Tensor) (
  const Tensor& self, const Tensor& other
) {
  // 调用 build_borrowing_binary_op 函数，生成一个二元操作，操作数是 maybe_get_output()、self 和 other
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(div, Tensor) (const Tensor& self, const Tensor& other) {
  // 调用 build_borrowing_binary_float_op 函数，生成一个浮点数二元操作，操作数是 maybe_get_output()、self 和 other
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(div, Tensor_mode) (const Tensor& self, const Tensor& other, std::optional<c10::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    // 如果没有指定舍入模式，调用 build_borrowing_binary_float_op 函数生成浮点数二元操作
    build_borrowing_binary_float_op(maybe_get_output(), self, other);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (*rounding_mode == "trunc") {
    // 如果舍入模式是 "trunc"，调用 build_borrowing_binary_op 函数生成一个二元操作
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else if (*rounding_mode == "floor") {
    // 如果舍入模式是 "floor"，调用 build_borrowing_binary_op 函数生成一个二元操作
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else {
    // 如果舍入模式既不是 None、"trunc"，也不是 "floor"，则抛出错误
    TORCH_CHECK(false,
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        "but found '", *rounding_mode, "'");
  }
}

// 下面的函数都调用 build_borrowing_binary_float_op 函数，生成浮点数二元操作，操作数是 maybe_get_output()、self 和 n
TORCH_META_FUNC(special_xlog1py) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(special_zeta) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(special_chebyshev_polynomial_t) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_chebyshev_polynomial_u) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_chebyshev_polynomial_v) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_chebyshev_polynomial_w) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_hermite_polynomial_h) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_hermite_polynomial_he) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_laguerre_polynomial_l) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_legendre_polynomial_p) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_shifted_chebyshev_polynomial_t) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_shifted_chebyshev_polynomial_u) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}
# 定义特殊的 shifted Chebyshev 多项式函数，操作两个张量
TORCH_META_FUNC(special_shifted_chebyshev_polynomial_v) (const Tensor& self, const Tensor& n) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 n 张量
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

# 定义特殊的 shifted Chebyshev 多项式函数，操作两个张量
TORCH_META_FUNC(special_shifted_chebyshev_polynomial_w) (const Tensor& self, const Tensor& n) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 n 张量
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

# 定义 copysign 函数，操作两个张量
TORCH_META_FUNC2(copysign, Tensor) (
  const Tensor& self, const Tensor& other
) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

# 定义 heaviside 函数，操作两个张量
TORCH_META_FUNC(heaviside) (
  const Tensor& self, const Tensor& other
) {
  # 检查 self 和 other 是否为复数，并且确认输出对象的类型不是复数
  TORCH_CHECK(!self.is_complex() && !other.is_complex() &&
              (maybe_get_output().defined() ? !maybe_get_output().is_complex() : true),
              "heaviside is not yet implemented for complex tensors.")
  # 检查 self 和 other 的数据类型是否一致，如果有输出对象，则检查输出对象的数据类型是否与 self 的一致
  TORCH_CHECK(self.dtype() == other.dtype() &&
              (maybe_get_output().defined() ? maybe_get_output().dtype() == self.dtype() : true),
              "heaviside is not yet implemented for tensors with different dtypes.")

  # 调用函数构建一个二元操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_binary_op(maybe_get_output(), self, other);
}

# 定义 atan2 函数，操作两个张量
TORCH_META_FUNC(atan2) (const Tensor& self, const Tensor& other) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

# 定义 remainder 函数，操作两个张量
TORCH_META_FUNC2(remainder, Tensor)(const Tensor& self, const Tensor& other) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

# 定义 bitwise_left_shift 函数，操作两个张量
TORCH_META_FUNC2(bitwise_left_shift, Tensor) (
  const Tensor& self, const Tensor& other
) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

# 定义 bitwise_right_shift 函数，操作两个张量
TORCH_META_FUNC2(bitwise_right_shift, Tensor) (
  const Tensor& self, const Tensor& other
) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

# 定义 bitwise_and 函数，操作两个张量
TORCH_META_FUNC2(bitwise_and, Tensor) (const Tensor& self, const Tensor& other) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

# 定义 bitwise_or 函数，操作两个张量
TORCH_META_FUNC2(bitwise_or, Tensor) (const Tensor& self, const Tensor& other) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

# 定义 bitwise_xor 函数，操作两个张量
TORCH_META_FUNC2(bitwise_xor, Tensor) (const Tensor& self, const Tensor& other) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

# 定义 fmod 函数，操作两个张量
TORCH_META_FUNC2(fmod, Tensor) (const Tensor& self, const Tensor& other) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

# 定义 xlogy 函数，操作两个张量
TORCH_META_FUNC2(xlogy, Tensor) (const Tensor& self, const Tensor& other) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 self 和 other 张量
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

# 定义 logit_backward 函数，操作两个张量
TORCH_META_FUNC(logit_backward) (const Tensor& grad_output, const Tensor& input, std::optional<double> eps) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 grad_output 和 input 张量
  build_borrowing_binary_op(maybe_get_output(), grad_output, input);
}

# 定义 sigmoid_backward 函数，操作两个张量
TORCH_META_FUNC(sigmoid_backward) (const Tensor& grad_output, const Tensor& output) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 grad_output 和 output 张量
  build_borrowing_binary_op(maybe_get_output(), grad_output, output);
}

# 定义 tanh_backward 函数，操作两个张量
TORCH_META_FUNC(tanh_backward) (const Tensor& grad_output, const Tensor& output) {
  # 调用函数构建一个操作，生成一个可能的输出对象，使用输入的 grad_output 和 output 张量
  build_borrowing_binary_op(maybe_get_output(), grad_output, output);
}

# 这些是保持数据类型的普通二元操作
#define CREATE_BINARY_META_FUNC(func)                                 \
  // 定义宏 CREATE_BINARY_META_FUNC(func)，用于生成元函数 func
  TORCH_META_FUNC(func) (const Tensor& self, const Tensor& other) {   \
    // 实现元函数 func 的逻辑，接受两个张量 self 和 other
    build_borrowing_binary_op(maybe_get_output(), self, other);                 \
  }

CREATE_BINARY_META_FUNC(logaddexp);
CREATE_BINARY_META_FUNC(logaddexp2);
CREATE_BINARY_META_FUNC(gcd);
CREATE_BINARY_META_FUNC(lcm);
CREATE_BINARY_META_FUNC(hypot);
CREATE_BINARY_META_FUNC(igamma);
CREATE_BINARY_META_FUNC(igammac);
CREATE_BINARY_META_FUNC(nextafter);

TORCH_META_FUNC(maximum) (const Tensor& self, const Tensor& other) {
  // 实现元函数 maximum，检查不支持复数张量的情况
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "maximum not implemented for complex tensors.");
  // 调用函数构建元函数的二进制操作
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(minimum) (const Tensor& self, const Tensor& other) {
  // 实现元函数 minimum，检查不支持复数张量的情况
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "minimum not implemented for complex tensors.");
  // 调用函数构建元函数的二进制操作
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(fmax) (const Tensor& self, const Tensor& other) {
    // 实现元函数 fmax，检查不支持复数张量的情况
    TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmax not implemented for complex tensors.");
    // 调用函数构建元函数的二进制操作
    build_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(fmin) (const Tensor& self, const Tensor& other) {
    // 实现元函数 fmin，检查不支持复数张量的情况
    TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");
    // 调用函数构建元函数的二进制操作
    build_binary_op(maybe_get_output(), self, other);
}

#define CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(func)                     \
  // 定义宏 CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(func)，用于生成比较函数
  TORCH_META_FUNC2(func, Tensor)(const Tensor& self, const Tensor& other) { \
    // 实现元函数 func 对两个张量进行比较
    const Tensor& result = maybe_get_output();                              \
    // 调用函数构建元函数的比较操作
    build_borrowing_comparison_op(result, self, other);                     \
  }                                                                         \
                                                                            \
  TORCH_META_FUNC2(func, Scalar)(const Tensor& self, const Scalar& other) { \
    // 实现元函数 func 对张量和标量进行比较
    auto other_tensor =                                                     \
        native::wrapped_scalar_tensor(other);                               \
    // 调用函数构建元函数的比较操作（除最后一个参数外）
    build_borrowing_except_last_argument_comparison_op(maybe_get_output(), self, other_tensor);  \
  }

CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(eq);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(ne);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(lt);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(le);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(gt);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(ge);

} // namespace at::meta


namespace at::native {

DEFINE_DISPATCH(add_clamp_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(div_true_stub);
DEFINE_DISPATCH(div_floor_stub);
DEFINE_DISPATCH(div_trunc_stub);
DEFINE_DISPATCH(remainder_stub);
DEFINE_DISPATCH(atan2_stub);
DEFINE_DISPATCH(bitwise_and_stub);
DEFINE_DISPATCH(bitwise_or_stub);
DEFINE_DISPATCH(bitwise_xor_stub);
DEFINE_DISPATCH(lshift_stub);
DEFINE_DISPATCH(rshift_stub);
DEFINE_DISPATCH(logical_and_stub);
DEFINE_DISPATCH(logical_or_stub);
DEFINE_DISPATCH(logical_xor_stub);
DEFINE_DISPATCH(lt_stub);
DEFINE_DISPATCH(le_stub);
DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(ge_stub);
DEFINE_DISPATCH(eq_stub);
DEFINE_DISPATCH(ne_stub);
DEFINE_DISPATCH(sigmoid_backward_stub);
DEFINE_DISPATCH(logit_backward_stub);
DEFINE_DISPATCH(tanh_backward_stub);
DEFINE_DISPATCH(maximum_stub);
DEFINE_DISPATCH(minimum_stub);
DEFINE_DISPATCH(fmax_stub);
DEFINE_DISPATCH(fmin_stub);
DEFINE_DISPATCH(fmod_stub);
DEFINE_DISPATCH(logaddexp_stub);
DEFINE_DISPATCH(logaddexp2_stub);
DEFINE_DISPATCH(gcd_stub);
DEFINE_DISPATCH(lcm_stub);
DEFINE_DISPATCH(hypot_stub);
DEFINE_DISPATCH(igamma_stub);
DEFINE_DISPATCH(igammac_stub);
DEFINE_DISPATCH(nextafter_stub);
DEFINE_DISPATCH(heaviside_stub);
DEFINE_DISPATCH(copysign_stub);
DEFINE_DISPATCH(xlogy_stub);
DEFINE_DISPATCH(xlog1py_stub);
DEFINE_DISPATCH(zeta_stub);
DEFINE_DISPATCH(chebyshev_polynomial_t_stub);
DEFINE_DISPATCH(chebyshev_polynomial_u_stub);
DEFINE_DISPATCH(chebyshev_polynomial_v_stub);
DEFINE_DISPATCH(chebyshev_polynomial_w_stub);
DEFINE_DISPATCH(hermite_polynomial_h_stub);
DEFINE_DISPATCH(hermite_polynomial_he_stub);
DEFINE_DISPATCH(laguerre_polynomial_l_stub);
DEFINE_DISPATCH(legendre_polynomial_p_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_t_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_u_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_v_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_w_stub);

# 实现函数 `sub_out`，计算张量之间的减法并将结果存储在给定的结果张量中
TORCH_IMPL_FUNC(sub_out) (
  const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
) {
  # 调用加法操作的存根函数，将 alpha 乘以 -1，实现减法
  add_stub(device_type(), *this, -alpha);
  # 检查结果张量的数据类型是否与输出的数据类型一致
  TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
}

# 实现函数 `mul_out`，计算张量之间的乘法并将结果存储在给定的结果张量中
TORCH_IMPL_FUNC(mul_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  # 调用乘法操作的存根函数
  mul_stub(device_type(), *this);
}

# 实现函数 `div_out`，计算张量之间的除法并将结果存储在给定的结果张量中
TORCH_IMPL_FUNC(div_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  # 调用真除法操作的存根函数
  div_true_stub(device_type(), *this);
}

# 实现函数 `div_out_mode`，根据舍入模式计算张量之间的除法并将结果存储在给定的结果张量中
TORCH_IMPL_FUNC(div_out_mode) (
  const Tensor& self, const Tensor& other, std::optional<c10::string_view> rounding_mode, const Tensor& result
) {
  if (!rounding_mode.has_value()) {
    # 若未指定舍入模式，则使用真除法操作的存根函数
    div_true_stub(device_type(), *this);
  } else if (*rounding_mode == "trunc") {
    # 若舍入模式为截断，则使用截断除法操作的存根函数
    div_trunc_stub(device_type(), *this);
  } else if (*rounding_mode == "floor") {
    # 若舍入模式为向下取整，则使用向下除法操作的存根函数
    div_floor_stub(device_type(), *this);
  }
}

# 实现函数 `logit_backward_out`，计算逻辑斯蒂回归反向传播的存根函数
TORCH_IMPL_FUNC(logit_backward_out) (const Tensor& grad_output, const Tensor& input, std::optional<double> eps, const Tensor& result) {
  # 调用逻辑斯蒂回归反向传播操作的存根函数，根据是否提供 eps 参数来确定是否使用默认的 -1.0
  logit_backward_stub(device_type(), *this, Scalar(eps ? eps.value() : -1.0));
}

# 实现函数 `sigmoid_backward_out`，计算 sigmoid 函数的反向传播的存根函数
TORCH_IMPL_FUNC(sigmoid_backward_out) (const Tensor& grad_output, const Tensor& output, const Tensor& result) {
  # 调用 sigmoid 函数的反向传播操作的存根函数
  sigmoid_backward_stub(device_type(), *this);
}

# 实现函数 `special_xlog1py_out`，计算特殊 xlog1py 函数的存根函数
TORCH_IMPL_FUNC(special_xlog1py_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  # 调用特殊 xlog1py 函数的存根函数
  xlog1py_stub(device_type(), *this);
}
// 定义特殊函数 special_zeta_out 的实现，接收三个张量参数 self, other 和 result
TORCH_IMPL_FUNC(special_zeta_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  // 调用 zeta_stub 函数处理当前设备类型和当前对象 *this
  zeta_stub(device_type(), *this);
}

// 定义特殊函数 special_chebyshev_polynomial_t_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_chebyshev_polynomial_t_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 chebyshev_polynomial_t_stub 函数处理当前设备类型和当前对象 *this
  chebyshev_polynomial_t_stub(device_type(), *this);
}

// 定义特殊函数 special_chebyshev_polynomial_u_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_chebyshev_polynomial_u_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 chebyshev_polynomial_u_stub 函数处理当前设备类型和当前对象 *this
  chebyshev_polynomial_u_stub(device_type(), *this);
}

// 定义特殊函数 special_chebyshev_polynomial_v_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_chebyshev_polynomial_v_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 chebyshev_polynomial_v_stub 函数处理当前设备类型和当前对象 *this
  chebyshev_polynomial_v_stub(device_type(), *this);
}

// 定义特殊函数 special_chebyshev_polynomial_w_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_chebyshev_polynomial_w_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 chebyshev_polynomial_w_stub 函数处理当前设备类型和当前对象 *this
  chebyshev_polynomial_w_stub(device_type(), *this);
}

// 定义特殊函数 special_hermite_polynomial_h_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_hermite_polynomial_h_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 hermite_polynomial_h_stub 函数处理当前设备类型和当前对象 *this
  hermite_polynomial_h_stub(device_type(), *this);
}

// 定义特殊函数 special_hermite_polynomial_he_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_hermite_polynomial_he_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 hermite_polynomial_he_stub 函数处理当前设备类型和当前对象 *this
  hermite_polynomial_he_stub(device_type(), *this);
}

// 定义特殊函数 special_laguerre_polynomial_l_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_laguerre_polynomial_l_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 laguerre_polynomial_l_stub 函数处理当前设备类型和当前对象 *this
  laguerre_polynomial_l_stub(device_type(), *this);
}

// 定义特殊函数 special_legendre_polynomial_p_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_legendre_polynomial_p_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 legendre_polynomial_p_stub 函数处理当前设备类型和当前对象 *this
  legendre_polynomial_p_stub(device_type(), *this);
}

// 定义特殊函数 special_shifted_chebyshev_polynomial_t_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_shifted_chebyshev_polynomial_t_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 shifted_chebyshev_polynomial_t_stub 函数处理当前设备类型和当前对象 *this
  shifted_chebyshev_polynomial_t_stub(device_type(), *this);
}

// 定义特殊函数 special_shifted_chebyshev_polynomial_u_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_shifted_chebyshev_polynomial_u_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 shifted_chebyshev_polynomial_u_stub 函数处理当前设备类型和当前对象 *this
  shifted_chebyshev_polynomial_u_stub(device_type(), *this);
}

// 定义特殊函数 special_shifted_chebyshev_polynomial_v_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_shifted_chebyshev_polynomial_v_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 shifted_chebyshev_polynomial_v_stub 函数处理当前设备类型和当前对象 *this
  shifted_chebyshev_polynomial_v_stub(device_type(), *this);
}

// 定义特殊函数 special_shifted_chebyshev_polynomial_w_out 的实现，接收三个张量参数 self, n 和 result
TORCH_IMPL_FUNC(special_shifted_chebyshev_polynomial_w_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  // 调用 shifted_chebyshev_polynomial_w_stub 函数处理当前设备类型和当前对象 *this
  shifted_chebyshev_polynomial_w_stub(device_type(), *this);
}

// 定义函数 tanh_backward_out 的实现，接收三个张量参数 grad_output, output 和 result
TORCH_IMPL_FUNC(tanh_backward_out) (const Tensor& grad_output, const Tensor& output, const Tensor& result) {
  // 调用 tanh_backward_stub 函数处理当前设备类型和当前对象 *this
  tanh_backward_stub(device_type(), *this);
}

// 定义宏 CREATE_BINARY_TORCH_IMPL_FUNC，用于简化二元函数定义的模板
#define CREATE_BINARY_TORCH_IMPL_FUNC(func_out, func_stub)                                                    \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& other, const Tensor& result) {  \
  // 调用指定的 func_stub 函数处理当前设备类型和当前对象 *this
  func_stub(device_type(), *this);                                                           \
}

// 使用宏 CREATE_BINARY_TORCH_IMPL_FUNC 定义函数 bitwise_and_out 的实现
CREATE_BINARY_TORCH_IMPL_FUNC(bitwise_and_out, bitwise_and_stub);
// 使用宏 CREATE_BINARY_TORCH_IMPL_FUNC 定义函数 bitwise_or_out 的实现
CREATE_BINARY_TORCH_IMPL_FUNC(bitwise_or_out, bitwise_or_stub);
// 使用宏 CREATE_BINARY_TORCH_IMPL_FUNC 定义函数 bitwise_xor_out 的实现
CREATE_BINARY_TORCH_IMPL_FUNC(bitwise_xor_out, bitwise_xor_stub);
// 使用宏 CREATE_BINARY_TORCH_IMPL_FUNC 定义函数 maximum_out 的实现
CREATE_BINARY_TORCH_IMPL_FUNC(maximum_out, maximum_stub);
// 创建二进制Torch函数的实现，这里分别创建了多个函数，如minimum_out、fmax_out等
CREATE_BINARY_TORCH_IMPL_FUNC(minimum_out, minimum_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(fmax_out, fmax_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(fmin_out, fmin_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(fmod_out, fmod_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp_out, logaddexp_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp2_out, logaddexp2_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(gcd_out, gcd_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(lcm_out, lcm_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(hypot_out, hypot_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(igamma_out, igamma_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(igammac_out, igammac_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(nextafter_out, nextafter_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(remainder_out, remainder_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(xlogy_out, xlogy_stub);

// 定义一个特殊函数，计算 xlog1py，其中 x 是标量，y 是张量
Tensor special_xlog1py(const Scalar& x, const Tensor& y) {
  return at::special_xlog1py(wrapped_scalar_tensor(x), y);
}

// 定义一个特殊函数，计算 xlog1py，其中 x 是张量，y 是标量
Tensor special_xlog1py(const Tensor& x, const Scalar& y) {
  return at::special_xlog1py(x, wrapped_scalar_tensor(y));
}

// 定义一个特殊函数，计算 xlog1py，并将结果保存在 result 张量中，其中 self 是标量，other 是张量
Tensor& special_xlog1py_out(const Scalar& self, const Tensor& other, Tensor& result) {
  return at::special_xlog1py_out(result, wrapped_scalar_tensor(self), other);
}

// 定义一个特殊函数，计算 xlog1py，并将结果保存在 result 张量中，其中 self 是张量，other 是标量
Tensor& special_xlog1py_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::special_xlog1py_out(result, self, wrapped_scalar_tensor(other));
}

// 定义一个特殊函数，计算 Riemann zeta 函数，其中 x 是标量，y 是张量
Tensor special_zeta(const Scalar& x, const Tensor& y) {
  return at::special_zeta(wrapped_scalar_tensor(x), y);
}

// 定义一个特殊函数，计算 Riemann zeta 函数，其中 x 是张量，y 是标量
Tensor special_zeta(const Tensor& x, const Scalar& y) {
  return at::special_zeta(x, wrapped_scalar_tensor(y));
}

// 定义一个特殊函数，计算 Riemann zeta 函数，并将结果保存在 result 张量中，其中 self 是标量，other 是张量
Tensor& special_zeta_out(const Scalar& self, const Tensor& other, Tensor& result) {
  return at::special_zeta_out(result, wrapped_scalar_tensor(self), other);
}

// 定义一个特殊函数，计算 Riemann zeta 函数，并将结果保存在 result 张量中，其中 self 是张量，other 是标量
Tensor& special_zeta_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::special_zeta_out(result, self, wrapped_scalar_tensor(other));
}

// 定义一个特殊函数，计算第一类Chebyshev多项式 T_n(x)，其中 x 是标量，n 是张量
Tensor special_chebyshev_polynomial_t(const Scalar& x, const Tensor& n) {
  return at::special_chebyshev_polynomial_t(wrapped_scalar_tensor(x), n);
}

// 定义一个特殊函数，计算第一类Chebyshev多项式 T_n(x)，其中 x 是张量，n 是标量
Tensor special_chebyshev_polynomial_t(const Tensor& x, const Scalar& n) {
  return at::special_chebyshev_polynomial_t(x, wrapped_scalar_tensor(n));
}

// 定义一个特殊函数，计算第一类Chebyshev多项式 T_n(x)，并将结果保存在 result 张量中，其中 self 是标量，n 是张量
Tensor& special_chebyshev_polynomial_t_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_chebyshev_polynomial_t_out(result, wrapped_scalar_tensor(self), n);
}

// 定义一个特殊函数，计算第一类Chebyshev多项式 T_n(x)，并将结果保存在 result 张量中，其中 self 是张量，n 是标量
Tensor& special_chebyshev_polynomial_t_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_chebyshev_polynomial_t_out(result, self, wrapped_scalar_tensor(n));
}

// 定义一个特殊函数，计算第二类Chebyshev多项式 U_n(x)，其中 x 是标量，n 是张量
Tensor special_chebyshev_polynomial_u(const Scalar& x, const Tensor& n) {
  return at::special_chebyshev_polynomial_u(wrapped_scalar_tensor(x), n);
}

// 定义一个特殊函数，计算第二类Chebyshev多项式 U_n(x)，其中 x 是张量，n 是标量
Tensor special_chebyshev_polynomial_u(const Tensor& x, const Scalar& n) {
  return at::special_chebyshev_polynomial_u(x, wrapped_scalar_tensor(n));
}
// 使用特殊的切比雪夫多项式 U 函数计算，并将结果存储到输出张量 result 中
Tensor& special_chebyshev_polynomial_u_out(const Scalar& self, const Tensor& n, Tensor& result) {
  // 调用 ATen 库的 special_chebyshev_polynomial_u_out 函数，计算 U 函数的结果
  return at::special_chebyshev_polynomial_u_out(result, wrapped_scalar_tensor(self), n);
}

// 使用特殊的切比雪夫多项式 U 函数计算，并将结果存储到输出张量 result 中
Tensor& special_chebyshev_polynomial_u_out(const Tensor& self, const Scalar& n, Tensor& result) {
  // 调用 ATen 库的 special_chebyshev_polynomial_u_out 函数，计算 U 函数的结果
  return at::special_chebyshev_polynomial_u_out(result, self, wrapped_scalar_tensor(n));
}

// 使用特殊的切比雪夫多项式 V 函数计算，返回计算结果张量
Tensor special_chebyshev_polynomial_v(const Scalar& x, const Tensor& n) {
  // 调用 ATen 库的 special_chebyshev_polynomial_v 函数，计算 V 函数的结果
  return at::special_chebyshev_polynomial_v(wrapped_scalar_tensor(x), n);
}

// 使用特殊的切比雪夫多项式 V 函数计算，返回计算结果张量
Tensor special_chebyshev_polynomial_v(const Tensor& x, const Scalar& n) {
  // 调用 ATen 库的 special_chebyshev_polynomial_v 函数，计算 V 函数的结果
  return at::special_chebyshev_polynomial_v(x, wrapped_scalar_tensor(n));
}

// 使用特殊的切比雪夫多项式 V 函数计算，并将结果存储到输出张量 result 中
Tensor& special_chebyshev_polynomial_v_out(const Scalar& self, const Tensor& n, Tensor& result) {
  // 调用 ATen 库的 special_chebyshev_polynomial_v_out 函数，计算 V 函数的结果并存储到 result
  return at::special_chebyshev_polynomial_v_out(result, wrapped_scalar_tensor(self), n);
}

// 使用特殊的切比雪夫多项式 V 函数计算，并将结果存储到输出张量 result 中
Tensor& special_chebyshev_polynomial_v_out(const Tensor& self, const Scalar& n, Tensor& result) {
  // 调用 ATen 库的 special_chebyshev_polynomial_v_out 函数，计算 V 函数的结果并存储到 result
  return at::special_chebyshev_polynomial_v_out(result, self, wrapped_scalar_tensor(n));
}

// 使用特殊的切比雪夫多项式 W 函数计算，返回计算结果张量
Tensor special_chebyshev_polynomial_w(const Scalar& x, const Tensor& n) {
  // 调用 ATen 库的 special_chebyshev_polynomial_w 函数，计算 W 函数的结果
  return at::special_chebyshev_polynomial_w(wrapped_scalar_tensor(x), n);
}

// 使用特殊的切比雪夫多项式 W 函数计算，返回计算结果张量
Tensor special_chebyshev_polynomial_w(const Tensor& x, const Scalar& n) {
  // 调用 ATen 库的 special_chebyshev_polynomial_w 函数，计算 W 函数的结果
  return at::special_chebyshev_polynomial_w(x, wrapped_scalar_tensor(n));
}

// 使用特殊的切比雪夫多项式 W 函数计算，并将结果存储到输出张量 result 中
Tensor& special_chebyshev_polynomial_w_out(const Scalar& self, const Tensor& n, Tensor& result) {
  // 调用 ATen 库的 special_chebyshev_polynomial_w_out 函数，计算 W 函数的结果并存储到 result
  return at::special_chebyshev_polynomial_w_out(result, wrapped_scalar_tensor(self), n);
}

// 使用特殊的切比雪夫多项式 W 函数计算，并将结果存储到输出张量 result 中
Tensor& special_chebyshev_polynomial_w_out(const Tensor& self, const Scalar& n, Tensor& result) {
  // 调用 ATen 库的 special_chebyshev_polynomial_w_out 函数，计算 W 函数的结果并存储到 result
  return at::special_chebyshev_polynomial_w_out(result, self, wrapped_scalar_tensor(n));
}

// 使用特殊的厄米多项式 H 函数计算，返回计算结果张量
Tensor special_hermite_polynomial_h(const Scalar& x, const Tensor& n) {
  // 调用 ATen 库的 special_hermite_polynomial_h 函数，计算 H 函数的结果
  return at::special_hermite_polynomial_h(wrapped_scalar_tensor(x), n);
}

// 使用特殊的厄米多项式 H 函数计算，返回计算结果张量
Tensor special_hermite_polynomial_h(const Tensor& x, const Scalar& n) {
  // 调用 ATen 库的 special_hermite_polynomial_h 函数，计算 H 函数的结果
  return at::special_hermite_polynomial_h(x, wrapped_scalar_tensor(n));
}

// 使用特殊的厄米多项式 H 函数计算，并将结果存储到输出张量 result 中
Tensor& special_hermite_polynomial_h_out(const Scalar& self, const Tensor& n, Tensor& result) {
  // 调用 ATen 库的 special_hermite_polynomial_h_out 函数，计算 H 函数的结果并存储到 result
  return at::special_hermite_polynomial_h_out(result, wrapped_scalar_tensor(self), n);
}

// 使用特殊的厄米多项式 H 函数计算，并将结果存储到输出张量 result 中
Tensor& special_hermite_polynomial_h_out(const Tensor& self, const Scalar& n, Tensor& result) {
  // 调用 ATen 库的 special_hermite_polynomial_h_out 函数，计算 H 函数的结果并存储到 result
  return at::special_hermite_polynomial_h_out(result, self, wrapped_scalar_tensor(n));
}

// 使用特殊的厄米多项式 HE 函数计算，返回计算结果张量
Tensor special_hermite_polynomial_he(const Scalar& x, const Tensor& n) {
  // 调用 ATen 库的 special_hermite_polynomial_he 函数，计算 HE 函数的结果
  return at::special_hermite_polynomial_he(wrapped_scalar_tensor(x), n);
}

// 使用特殊的厄米多项式 HE 函数计算，返回计算结果张量
Tensor special_hermite_polynomial_he(const Tensor& x, const Scalar& n) {
  // 调用 ATen 库的 special_hermite_polynomial_he 函数，计算 HE 函数的结果
  return at::special_hermite_polynomial_he(x, wrapped_scalar_tensor(n));
}

// 使用特殊的厄米多项式 HE 函数计算，并将结果存储到输出张量 result 中
Tensor& special_hermite_polynomial_he_out(const Scalar& self, const Tensor& n, Tensor& result) {
  // 调用 ATen 库的 special_hermite_polynomial_he_out 函数，计算 HE 函数的结果并存储到 result
  return at::special_hermite_polynomial_he_out(result, wrapped_scalar_tensor(self), n);
}
// 调用 ATen 库中的特殊埃尔米特多项式函数，将结果存入输出张量 result 中
Tensor& special_hermite_polynomial_he_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_hermite_polynomial_he_out(result, self, wrapped_scalar_tensor(n));
}

// 调用 ATen 库中的特殊拉盖尔多项式函数，返回结果张量
Tensor special_laguerre_polynomial_l(const Scalar& x, const Tensor& n) {
  return at::special_laguerre_polynomial_l(wrapped_scalar_tensor(x), n);
}

// 调用 ATen 库中的特殊拉盖尔多项式函数，返回结果张量
Tensor special_laguerre_polynomial_l(const Tensor& x, const Scalar& n) {
  return at::special_laguerre_polynomial_l(x, wrapped_scalar_tensor(n));
}

// 调用 ATen 库中的特殊拉盖尔多项式函数，将结果存入输出张量 result 中
Tensor& special_laguerre_polynomial_l_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_laguerre_polynomial_l_out(result, wrapped_scalar_tensor(self), n);
}

// 调用 ATen 库中的特殊拉盖尔多项式函数，将结果存入输出张量 result 中
Tensor& special_laguerre_polynomial_l_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_laguerre_polynomial_l_out(result, self, wrapped_scalar_tensor(n));
}

// 调用 ATen 库中的特殊勒让德多项式函数，返回结果张量
Tensor special_legendre_polynomial_p(const Scalar& x, const Tensor& n) {
  return at::special_legendre_polynomial_p(wrapped_scalar_tensor(x), n);
}

// 调用 ATen 库中的特殊勒让德多项式函数，返回结果张量
Tensor special_legendre_polynomial_p(const Tensor& x, const Scalar& n) {
  return at::special_legendre_polynomial_p(x, wrapped_scalar_tensor(n));
}

// 调用 ATen 库中的特殊勒让德多项式函数，将结果存入输出张量 result 中
Tensor& special_legendre_polynomial_p_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_legendre_polynomial_p_out(result, wrapped_scalar_tensor(self), n);
}

// 调用 ATen 库中的特殊勒让德多项式函数，将结果存入输出张量 result 中
Tensor& special_legendre_polynomial_p_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_legendre_polynomial_p_out(result, self, wrapped_scalar_tensor(n));
}

// 调用 ATen 库中的特殊移位切比雪夫多项式函数，返回结果张量
Tensor special_shifted_chebyshev_polynomial_t(const Scalar& x, const Tensor& n) {
  return at::special_shifted_chebyshev_polynomial_t(wrapped_scalar_tensor(x), n);
}

// 调用 ATen 库中的特殊移位切比雪夫多项式函数，返回结果张量
Tensor special_shifted_chebyshev_polynomial_t(const Tensor& x, const Scalar& n) {
  return at::special_shifted_chebyshev_polynomial_t(x, wrapped_scalar_tensor(n));
}

// 调用 ATen 库中的特殊移位切比雪夫多项式函数，将结果存入输出张量 result 中
Tensor& special_shifted_chebyshev_polynomial_t_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_t_out(result, wrapped_scalar_tensor(self), n);
}

// 调用 ATen 库中的特殊移位切比雪夫多项式函数，将结果存入输出张量 result 中
Tensor& special_shifted_chebyshev_polynomial_t_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_t_out(result, self, wrapped_scalar_tensor(n));
}

// 调用 ATen 库中的特殊移位切比雪夫多项式函数，返回结果张量
Tensor special_shifted_chebyshev_polynomial_u(const Scalar& x, const Tensor& n) {
  return at::special_shifted_chebyshev_polynomial_u(wrapped_scalar_tensor(x), n);
}

// 调用 ATen 库中的特殊移位切比雪夫多项式函数，返回结果张量
Tensor special_shifted_chebyshev_polynomial_u(const Tensor& x, const Scalar& n) {
  return at::special_shifted_chebyshev_polynomial_u(x, wrapped_scalar_tensor(n));
}

// 调用 ATen 库中的特殊移位切比雪夫多项式函数，将结果存入输出张量 result 中
Tensor& special_shifted_chebyshev_polynomial_u_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_u_out(result, wrapped_scalar_tensor(self), n);
}


这些注释为每个函数调用解释了其作用和每个参数的含义，使得代码更易于理解其功能和使用方式。
// 将自身与另一个张量进行按元素加法，并应用 ReLU（整流线性单元），结果存入 result 张量中
static Tensor& add_relu_impl(
    Tensor& result, const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 创建张量迭代器，用于对 self 和 other 执行二元操作，并将结果存入 result 中
  auto iter = TensorIterator::binary_op(result, self, other);
  // 定义最小值和最大值标量，根据 self 的数据类型设置不同的范围
  Scalar min_val;
  Scalar max_val;
  if (self.dtype() == at::kInt) {
    min_val = 0;
    max_val = std::numeric_limits<int32_t>::max();
  } else if (self.dtype() == at::kLong) {
    min_val = 0;
    max_val = std::numeric_limits<int64_t>::max();
  }
    // 如果张量的数据类型是 int64_t，则设置最大值为 int64_t 的最大值，最小值未指定，由后续逻辑决定
    max_val = std::numeric_limits<int64_t>::max();
  } else if (self.dtype() == at::kShort) {
    // 如果张量的数据类型是 int16_t，则设置最小值为 0，最大值为 int16_t 的最大值
    min_val = 0;
    max_val = std::numeric_limits<int16_t>::max();
  } else if (self.dtype() == at::kChar) {
    // 如果张量的数据类型是 int8_t，则设置最小值为 0，最大值为 int8_t 的最大值
    min_val = 0;
    max_val = std::numeric_limits<int8_t>::max();
  } else if (self.dtype() == at::kFloat) {
    // 如果张量的数据类型是 float，则设置最小值为 0.0，最大值为 float 的最大值
    min_val = 0.0;
    max_val = std::numeric_limits<float>::max();
  } else if (self.dtype() == at::kDouble) {
    // 如果张量的数据类型是 double，则设置最小值为 0.0，最大值为 double 的最大值
    min_val = 0.0;
    max_val = std::numeric_limits<double>::max();
  } else {
    // 如果张量的数据类型不受支持，则抛出内部断言错误并显示不支持的数据类型信息
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported datatype for add_relu:", self.dtype().name());
  }

  // 将迭代器的输出保存到 result 中
  result = iter.output();
  // 调用特定的 add_clamp_stub 函数处理迭代器及参数 alpha、min_val、max_val
  add_clamp_stub(iter.device_type(), iter, alpha, min_val, max_val);
  // 返回处理后的 result
  return result;
// 返回一个 Tensor 的引用，通过调用 add_relu_impl 函数实现将 self 和 other 相加，并应用 ReLU 激活函数，结果存储在 result 中
Tensor& add_relu_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& result) {
  return add_relu_impl(result, self, other, alpha);
}

// 返回一个新的 Tensor，通过调用 add_relu_impl 函数实现将 self 和 other 相加，并应用 ReLU 激活函数
Tensor add_relu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  Tensor result;
  return add_relu_impl(result, self, other, alpha);
}

// 返回一个新的 Tensor，通过调用 add_relu 函数实现将 self 和标量 other 相加，并应用 ReLU 激活函数
Tensor add_relu(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return add_relu(self, wrapped_scalar_tensor(other), alpha);
}

// 返回一个 Tensor 的引用，通过调用 add_relu_impl 函数实现将 self 和 other 相加，并应用 ReLU 激活函数，结果存储在 self 中
Tensor& add_relu_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return add_relu_impl(self, self, other, alpha);
}

// 返回一个 Tensor 的引用，通过调用 add_relu_ 函数实现将 self 和标量 other 相加，并应用 ReLU 激活函数
Tensor& add_relu_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return add_relu_(self, wrapped_scalar_tensor(other), alpha);
}

// 实现 copysign_out 的 TORCH_IMPL_FUNC，调用 copysign_stub 实现具体的功能
TORCH_IMPL_FUNC(copysign_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  copysign_stub(device_type(), *this);
}

// 返回一个新的 Tensor，通过调用 at::copysign 函数实现将 self 中的值与标量 other 的符号合并
Tensor copysign(const Tensor& self, const Scalar& other) {
  // redispatch!
  return at::copysign(self, wrapped_scalar_tensor(other));
}

// 返回一个 Tensor 的引用，通过调用 self 的 copysign_ 函数实现将 self 中的值与标量 other 的符号合并
Tensor& copysign_(Tensor& self, const Scalar& other) {
  // redispatch!
  return self.copysign_(wrapped_scalar_tensor(other));
}

// 返回一个 Tensor 的引用，通过调用 at::copysign_out 函数实现将 self 中的值与标量 other 的符号合并，并将结果存储在 result 中
Tensor& copysign_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // redispatch!
  return at::copysign_out(result, self, wrapped_scalar_tensor(other));
}

// 返回一个新的 Tensor，通过调用 self 的 div 函数实现将 self 中的每个元素与标量 other 相除
Tensor div(const Tensor& self, const Scalar& other) {
  return self.div(wrapped_scalar_tensor(other)); // redispatch!
}

// 返回一个 Tensor 的引用，通过调用 self 的 div_ 函数实现将 self 中的每个元素与标量 other 相除
Tensor& div_(Tensor& self, const Scalar& other) {
  return self.div_(wrapped_scalar_tensor(other)); // redispatch!
}

// 返回一个新的 Tensor，通过调用 self 的 div 函数实现将 self 中的每个元素与标量 other 相除，并指定舍入模式
Tensor div(const Tensor& self, const Scalar& other, std::optional<c10::string_view> rounding_mode) {
  return self.div(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

// 返回一个 Tensor 的引用，通过调用 self 的 div_ 函数实现将 self 中的每个元素与标量 other 相除，并指定舍入模式
Tensor& div_(Tensor& self, const Scalar& other, std::optional<c10::string_view> rounding_mode) {
  return self.div_(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

// divide 的别名，返回一个 Tensor 的引用，通过调用 at::div_out 函数实现将 self 与 other 相除，并将结果存储在 result 中
Tensor& divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::div_out(result, self, other);
}

// 返回一个新的 Tensor，通过调用 self 的 div 函数实现将 self 与 other 相除
Tensor divide(const Tensor& self, const Tensor& other) {
  return self.div(other);
}

// 返回一个 Tensor 的引用，通过调用 self 的 div_ 函数实现将 self 与 other 相除
Tensor& divide_(Tensor& self, const Tensor& other) {
  return self.div_(other);
}

// 返回一个新的 Tensor，通过调用 self 的 div 函数实现将 self 与标量 other 相除
Tensor divide(const Tensor& self, const Scalar& other) {
  return self.div(other);
}

// 返回一个 Tensor 的引用，通过调用 self 的 div_ 函数实现将 self 与标量 other 相除
Tensor& divide_(Tensor& self, const Scalar& other) {
  return self.div_(other);
}

// 返回一个 Tensor 的引用，通过调用 at::div_out 函数实现将 self 与 other 相除，并指定舍入模式，结果存储在 result 中
Tensor& divide_out(const Tensor& self, const Tensor& other, std::optional<c10::string_view> rounding_mode, Tensor& result) {
  return at::div_out(result, self, other, std::move(rounding_mode));
}
// 定义一个函数，将两个张量进行除法操作，并返回结果张量
Tensor divide(const Tensor& self, const Tensor& other, std::optional<c10::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

// 定义一个函数，在原地将一个张量除以另一个张量，并返回自身的引用
Tensor& divide_(Tensor& self, const Tensor& other, std::optional<c10::string_view> rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

// 定义一个函数，将一个张量与标量进行除法操作，并返回结果张量
Tensor divide(const Tensor& self, const Scalar& other, std::optional<c10::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

// 定义一个函数，在原地将一个张量除以标量，并返回自身的引用
Tensor& divide_(Tensor& self, const Scalar& other, std::optional<c10::string_view> rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

// true_divide 的别名，使用 div_out 将两个张量进行除法操作，并将结果存入 result
Tensor& true_divide_out(const Tensor& self, const Tensor& divisor, Tensor& result) {
  return at::div_out(result, self, divisor);
}

// 定义一个函数，将一个张量与另一个张量进行除法操作，并返回结果张量
Tensor true_divide(const Tensor& self, const Tensor& divisor) {
  return self.div(divisor);
}

// 定义一个函数，在原地将一个张量除以另一个张量，并返回自身的引用
Tensor& true_divide_(Tensor& self, const Tensor& divisor) {
  return self.div_(divisor);
}

// 定义一个函数，将一个张量与标量进行除法操作，并返回结果张量
Tensor true_divide(const Tensor& self, const Scalar& divisor) {
  return self.div(divisor);
}

// 定义一个函数，在原地将一个张量除以标量，并返回自身的引用
Tensor& true_divide_(Tensor& self, const Scalar& divisor) {
  return self.div_(divisor);
}

// 使用 TensorIterator 执行 floor_divide 操作，将两个张量进行整数除法并返回结果张量
Tensor& floor_divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  div_floor_stub(iter.device_type(), iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

// 定义一个函数，将一个张量与另一个张量进行整数除法操作，并返回结果张量
Tensor floor_divide(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  div_floor_stub(iter.device_type(), iter);
  return iter.output();
}

// 在原地将一个张量与另一个张量进行整数除法操作，并返回自身的引用
Tensor& floor_divide_(Tensor& self, const Tensor& other) {
  return native::floor_divide_out(self, other, self);
}

// TODO: 对于性能回退从 native:: 移除重新实现

// 定义一个函数，将一个张量与标量进行乘法操作，并返回结果张量
Tensor mul(const Tensor& self, const Scalar& other) {
  return at::mul(self, wrapped_scalar_tensor(other)); // 重新调度！
}

// 在原地将一个张量与标量进行乘法操作，并返回自身的引用
Tensor& mul_(Tensor& self, const Scalar& other) {
  return at::mul_out(self, wrapped_scalar_tensor(other), self); // 重新调度！
}

// 在稀疏 CSR 格式张量中，对值进行标量乘法操作，并返回自身的引用
Tensor& mul__scalar_sparse_csr(Tensor& self, const Scalar& other) {
  self.values().mul_(other);
  return self;
}

// 根据输入张量的设备类型，返回正确的输出设备
static Device correct_out_device(const Tensor& self, const Tensor& other) {
  if (self.device() == at::kCPU){
      return other.device();
  } else {
    return self.device();
  }
}

// 将两个张量进行乘法操作，并返回结果张量，使用 TensorIterator 获取广播和类型提升逻辑
Tensor mul_zerotensor(const Tensor& self, const Tensor& other) {
  auto out_device = correct_out_device(self, other);
  // 使用 TensorIterator 来获得正确的广播和类型提升逻辑
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  auto meta_out = at::_ops::mul_Tensor::redispatch(meta_dks, self.to(device_), other.to(device_));
  // 返回一个零张量，其大小和设备从 meta_out 中获取
  return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
}
// 定义函数 div_zerotensor，接受两个 Tensor 类型参数 self 和 other
Tensor div_zerotensor(const Tensor& self, const Tensor& other) {
  // 获取正确的输出设备
  auto out_device = correct_out_device(self, other);
  // 使用 TensorIterator 来获取正确的广播和类型提升逻辑的 hack
  auto device_ = Device(DeviceType::Meta);
  // 定义元设备 DispatchKeySet
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  // 使用 redispatch 方法调用 div_Tensor 操作，返回 meta_out
  auto meta_out = at::_ops::div_Tensor::redispatch(meta_dks, self.to(device_), other.to(device_));

  // 如果 self 是零张量
  if (self._is_zerotensor()) {
    // 如果 other 也是零张量，返回尺寸与 meta_out 相同且元素为 NaN 的全零张量
    if (other._is_zerotensor()) {
      return at::full(meta_out.sizes(), std::numeric_limits<float>::quiet_NaN(), meta_out.options().device(out_device));
    }
    // 如果 other 不是零张量，返回零张量
    else {
      return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
    }
  }
  // 如果 self 不是零张量
  else {
    // 如果 other 是零张量，返回尺寸与 meta_out 相同且元素为 INF 的全零张量
    if (other._is_zerotensor()) {
      return at::full(meta_out.sizes(), std::numeric_limits<float>::infinity(), meta_out.options().device(out_device));
    }
    // 如果 other 不是零张量，返回零张量
    else {
      // 这段代码理论上是无法到达的，参见上文的 TORCH_INTERNAL_ASSERT
      return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
    }
  }
}

// 定义静态函数 maybe_add_maybe_sub，接受三个参数 self、other、alpha
static Tensor maybe_add_maybe_sub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 获取正确的输出设备
  auto out_device = correct_out_device(self, other);
  // 使用 TensorIterator 来获取正确的广播和类型提升逻辑的 hack
  auto device_ = Device(DeviceType::Meta);
  // 定义元设备 DispatchKeySet
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  // 使用 redispatch 方法调用 add_Tensor 操作，返回 meta_out
  auto meta_out = at::_ops::add_Tensor::redispatch(
      meta_dks, self.to(device_), other.to(device_), alpha);

  // 定义 lambda 函数 get_out_like，接受一个参数 tensor，返回对应尺寸的张量
  auto get_out_like = [&] (const Tensor& tensor)
  {
      auto sizes = meta_out.sizes();
      return at::_to_copy(tensor.expand(sizes), meta_out.options().device(out_device));
  };

  // 如果 self 是零张量
  if (self._is_zerotensor()) {
    // 如果 other 是零张量，返回尺寸与 meta_out 相同的全零张量
    if (other._is_zerotensor()) {
      return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
    }
    // 否则，获取与 other 尺寸相同的结果，并根据 alpha 的值做乘法或不变操作
    auto res = get_out_like(other);
    return alpha.equal(1) ? std::move(res) : res.mul(alpha);
  } else {
    // 如果 self 不是零张量，返回与 self 尺寸相同的结果
    return get_out_like(self);
  }
}

// 定义函数 add_zerotensor，接受三个参数 self、other、alpha
Tensor add_zerotensor(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 调用 maybe_add_maybe_sub 函数处理 self、other、alpha 参数
  return maybe_add_maybe_sub(self, other, alpha);
}

// 定义函数 sub_zerotensor，接受三个参数 self、other、alpha
Tensor sub_zerotensor(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 调用 maybe_add_maybe_sub 函数处理 self、other、-alpha 参数
  return maybe_add_maybe_sub(self, other, -alpha);
}

// 定义函数 linalg_cross_zerotensor，接受三个参数 input、other、dim
Tensor linalg_cross_zerotensor(
  const Tensor& input,
  const Tensor& other,
  const int64_t dim)
{
  // 获取正确的输出设备
  auto out_device = correct_out_device(input, other);
  // 使用 TensorIterator 来获取正确的广播和类型提升逻辑的 hack（参见 add_zerotensor）
  auto device = Device(DeviceType::Meta);
  // 使用 redispatch 方法调用 linalg_cross 操作，返回 meta_out
  auto meta_out = at::_ops::linalg_cross::redispatch(
    c10::DispatchKeySet(at::DispatchKey::Meta),
    input.to(device),
    other.to(device),
    dim);

  // 返回尺寸与 meta_out 相同的全零张量
  return at::_efficientzerotensor(
    meta_out.sizes(),
    meta_out.options().device(out_device));
}
// multiply_out函数：计算两个张量的乘法，并将结果存储在给定的结果张量中
Tensor& multiply_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::mul_out(result, self, other);
}

// multiply函数：返回两个张量的乘法结果
Tensor multiply(const Tensor& self, const Tensor& other) {
  return self.mul(other);
}

// multiply_函数：在原地修改张量，将其与另一个张量相乘
Tensor& multiply_(Tensor& self, const Tensor& other) {
  return self.mul_(other);
}

// multiply函数（标量版本）：返回张量与标量的乘法结果
Tensor multiply(const Tensor& self, const Scalar& other) {
  return self.mul(other);
}

// multiply_函数（标量版本）：在原地修改张量，将其与标量相乘
Tensor& multiply_(Tensor& self, const Scalar& other) {
  return self.mul_(other);
}

// sub函数：返回两个张量之间的减法结果，带有标量系数
Tensor sub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::sub(self, wrapped_scalar_tensor(other), alpha); // redispatch!
}

// sub_函数：在原地修改张量，将其与另一个张量相减，带有标量系数
Tensor& sub_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub_(wrapped_scalar_tensor(other), alpha); // redispatch!
}

// subtract_out函数：计算两个张量的减法，并将结果存储在给定的结果张量中
Tensor& subtract_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& result) {
  return at::sub_out(result, self, other, alpha);
}

// subtract函数：返回两个张量之间的减法结果，带有标量系数
Tensor subtract(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self.sub(other, alpha);
}

// subtract_函数：在原地修改张量，将其与另一个张量相减，带有标量系数
Tensor& subtract_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self.sub_(other, alpha);
}

// subtract函数（标量版本）：返回张量与标量之间的减法结果，带有标量系数
Tensor subtract(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub(other, alpha);
}

// subtract_函数（标量版本）：在原地修改张量，将其与标量相减，带有标量系数
Tensor& subtract_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub_(other, alpha);
}

// rsub函数：返回两个张量相反顺序相减的结果，带有标量系数
Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::sub(other, self, alpha); // redispatch!
}

// add函数：返回两个张量之间的加法结果，带有标量系数
Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::add(self, wrapped_scalar_tensor(other), alpha);
}

// add_函数：在原地修改张量，将其与标量相加，带有标量系数
Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.add_(wrapped_scalar_tensor(other), alpha);
}

// remainder函数：返回两个张量之间的取余结果
Tensor remainder(const Tensor& self, const Scalar& other) {
  // redispatch
  return at::remainder(self, wrapped_scalar_tensor(other));
}

// remainder_函数：在原地修改张量，将其与标量取余
Tensor& remainder_(Tensor& self, const Scalar& other) {
  // redispatch
  return self.remainder_(wrapped_scalar_tensor(other));
}

// remainder_out函数：计算两个张量的取余，并将结果存储在给定的结果张量中
Tensor& remainder_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // redispatch
  return at::remainder_out(result, self, wrapped_scalar_tensor(other));
}

// remainder函数（标量版本）：返回标量与张量之间的取余结果
Tensor remainder(const Scalar& self, const Tensor& other) {
  return at::remainder(wrapped_scalar_tensor(self), other);
}

// rsub函数（标量版本）：返回两个张量相反顺序相减的结果，带有标量系数
Tensor rsub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return native::rsub(self, wrapped_scalar_tensor(other), alpha);
}

// bitwise_and_out函数：计算两个张量的按位与，并将结果存储在给定的结果张量中
Tensor& bitwise_and_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::bitwise_and_out(result, self, wrapped_scalar_tensor(other));
}

// bitwise_and函数：返回两个张量的按位与结果
Tensor bitwise_and(const Tensor& self, const Scalar& other) {
  return at::bitwise_and(self, wrapped_scalar_tensor(other));
}
// 对给定的标量和张量进行按位与操作，返回结果张量
Tensor bitwise_and(const Scalar& self, const Tensor& other) {
  // 将标量包装为张量后，调用 ATen 库的按位与函数
  return at::bitwise_and(wrapped_scalar_tensor(self), other);
}

// 对张量自身和给定的标量进行原地按位与操作，返回修改后的自身张量
Tensor& bitwise_and_(Tensor& self, const Scalar& other) {
  // 调用自身张量的原地按位与函数
  return self.bitwise_and_(wrapped_scalar_tensor(other));
}

// 旧版本的按位与接口，作为 bitwise_and* 函数的别名
Tensor __and__(const Tensor& self, const Tensor& other) {
  // 调用 ATen 库的按位与函数
  return at::bitwise_and(self, other);
}

// 对张量和标量进行按位与操作，作为 bitwise_and* 函数的别名
Tensor __and__(const Tensor& self, const Scalar& other) {
  // 调用 ATen 库的按位与函数
  return at::bitwise_and(self, other);
}

// 对自身张量和给定的张量进行原地按位与操作，返回修改后的自身张量
Tensor& __iand__(Tensor& self, const Tensor& other) {
  // 调用自身张量的原地按位与函数
  return self.bitwise_and_(other);
}

// 对自身张量和给定的标量进行原地按位与操作，返回修改后的自身张量
Tensor& __iand__(Tensor& self, const Scalar& other) {
  // 调用自身张量的原地按位与函数
  return self.bitwise_and_(other);
}

// 对给定的标量和张量进行按位或操作，将结果存储在指定的结果张量中，返回结果张量的引用
Tensor& bitwise_or_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // 调用 ATen 库的按位或操作，并将结果存储在指定的结果张量中
  return at::bitwise_or_out(result, self, wrapped_scalar_tensor(other));
}

// 对给定的张量和标量进行按位或操作，返回结果张量
Tensor bitwise_or(const Tensor& self, const Scalar& other) {
  // 调用 ATen 库的按位或函数
  return at::bitwise_or(self, wrapped_scalar_tensor(other));
}

// 对给定的标量和张量进行按位或操作，返回结果张量
Tensor bitwise_or(const Scalar& self, const Tensor& other) {
  // 将标量包装为张量后，调用 ATen 库的按位或函数
  return at::bitwise_or(wrapped_scalar_tensor(self), other);
}

// 对自身张量和给定的标量进行原地按位或操作，返回修改后的自身张量
Tensor& bitwise_or_(Tensor& self, const Scalar& other) {
  // 调用自身张量的原地按位或函数
  return self.bitwise_or_(wrapped_scalar_tensor(other));
}

// 旧版本的按位或接口，作为 bitwise_or* 函数的别名
Tensor __or__(const Tensor& self, const Tensor& other) {
  // 调用 ATen 库的按位或函数
  return at::bitwise_or(self, other);
}

// 对给定的张量和标量进行按位或操作，作为 bitwise_or* 函数的别名
Tensor __or__(const Tensor& self, const Scalar& other) {
  // 调用 ATen 库的按位或函数
  return at::bitwise_or(self, other);
}

// 对自身张量和给定的张量进行原地按位或操作，返回修改后的自身张量
Tensor& __ior__(Tensor& self, const Tensor& other) {
  // 调用自身张量的原地按位或函数
  return self.bitwise_or_(other);
}

// 对自身张量和给定的标量进行原地按位或操作，返回修改后的自身张量
Tensor& __ior__(Tensor& self, const Scalar& other) {
  // 调用自身张量的原地按位或函数
  return self.bitwise_or_(other);
}

// 对给定的标量和张量进行按位异或操作，将结果存储在指定的结果张量中，返回结果张量的引用
Tensor& bitwise_xor_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // 调用 ATen 库的按位异或操作，并将结果存储在指定的结果张量中
  return at::bitwise_xor_out(result, self, wrapped_scalar_tensor(other));
}

// 对给定的张量和标量进行按位异或操作，返回结果张量
Tensor bitwise_xor(const Tensor& self, const Scalar& other) {
  // 调用 ATen 库的按位异或函数
  return at::bitwise_xor(self, wrapped_scalar_tensor(other));
}

// 对给定的标量和张量进行按位异或操作，返回结果张量
Tensor bitwise_xor(const Scalar& self, const Tensor& other) {
  // 将标量包装为张量后，调用 ATen 库的按位异或函数
  return at::bitwise_xor(wrapped_scalar_tensor(self), other);
}

// 对自身张量和给定的标量进行原地按位异或操作，返回修改后的自身张量
Tensor& bitwise_xor_(Tensor& self, const Scalar& other) {
  // 调用自身张量的原地按位异或函数
  return self.bitwise_xor_(wrapped_scalar_tensor(other));
}

// 旧版本的按位异或接口，作为 bitwise_xor* 函数的别名
Tensor __xor__(const Tensor& self, const Tensor& other) {
  // 调用 ATen 库的按位异或函数
  return at::bitwise_xor(self, other);
}

// 对给定的张量和标量进行按位异或操作，作为 bitwise_xor* 函数的别名
Tensor __xor__(const Tensor& self, const Scalar& other) {
  // 调用 ATen 库的按位异或函数
  return at::bitwise_xor(self, other);
}

// 对自身张量和给定的张量进行按位左移操作，返回结果张量
Tensor __lshift__(const Tensor& self, const Tensor& other) {
  // 创建空张量作为结果
  Tensor result;
  // 构建张量迭代器，执行二元操作
  auto iter = TensorIterator::binary_op(result, self, other);
  // 调用具体的左移操作函数
  lshift_stub(iter.device_type(), iter);
  // 返回迭代器的输出张量
  return iter.output();
}
Tensor __lshift__(const Tensor& self, const Scalar& other) {
  // 创建一个空的 Tensor 对象，用于存储结果
  Tensor result;
  // 将标量 other 包装成 Tensor
  auto wrapper = wrapped_scalar_tensor(other);
  // 创建一个 Tensor 迭代器，执行二进制左移操作
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  // 调用底层的左移操作函数
  lshift_stub(iter.device_type(), iter);
  // 返回左移后的结果 Tensor
  return iter.output();
}

Tensor& __ilshift__(Tensor& self, const Tensor& other) {
  // 创建一个 Tensor 迭代器，执行就地赋值的二进制左移操作
  auto iter = TensorIterator::binary_op(self, self, other);
  // 调用底层的左移操作函数
  lshift_stub(iter.device_type(), iter);
  // 返回左移后的自身 Tensor 引用
  return self;
}

Tensor& __ilshift__(Tensor& self, const Scalar& other) {
  // 将标量 other 包装成 Tensor
  auto wrapper = wrapped_scalar_tensor(other);
  // 创建一个 Tensor 迭代器，执行就地赋值的二进制左移操作
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  // 调用底层的左移操作函数
  lshift_stub(iter.device_type(), iter);
  // 返回左移后的自身 Tensor 引用
  return self;
}

TORCH_IMPL_FUNC(bitwise_left_shift_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  // 调用底层的左移操作函数
  lshift_stub(device_type(), *this);
}

Tensor& bitwise_left_shift_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // 调用 at::bitwise_left_shift_out 函数，执行二进制左移操作
  return at::bitwise_left_shift_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_left_shift(const Tensor& self, const Scalar& other) {
  // 调用 at::bitwise_left_shift 函数，执行二进制左移操作
  return at::bitwise_left_shift(self, wrapped_scalar_tensor(other));
}

Tensor& bitwise_left_shift_(Tensor& self, const Scalar& other) {
  // 调用 at::bitwise_left_shift_out 函数，执行就地赋值的二进制左移操作
  return at::bitwise_left_shift_out(self, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_left_shift(const Scalar& self, const Tensor& other) {
  // 将标量 self 包装成 Tensor，然后调用 at::bitwise_left_shift 函数，执行二进制左移操作
  return at::bitwise_left_shift(wrapped_scalar_tensor(self), other);
}

Tensor __rshift__(const Tensor& self, const Tensor& other) {
  // 创建一个空的 Tensor 对象，用于存储结果
  Tensor result;
  // 创建一个 Tensor 迭代器，执行二进制右移操作
  auto iter = TensorIterator::binary_op(result, self, other);
  // 调用底层的右移操作函数
  rshift_stub(iter.device_type(), iter);
  // 返回右移后的结果 Tensor
  return iter.output();
}

Tensor __rshift__(const Tensor& self, const Scalar& other) {
  // 创建一个空的 Tensor 对象，用于存储结果
  Tensor result;
  // 将标量 other 包装成 Tensor
  auto wrapper = wrapped_scalar_tensor(other);
  // 创建一个 Tensor 迭代器，执行二进制右移操作
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  // 调用底层的右移操作函数
  rshift_stub(iter.device_type(), iter);
  // 返回右移后的结果 Tensor
  return iter.output();
}

Tensor& __irshift__(Tensor& self, const Tensor& other) {
  // 创建一个 Tensor 迭代器，执行就地赋值的二进制右移操作
  auto iter = TensorIterator::binary_op(self, self, other);
  // 调用底层的右移操作函数
  rshift_stub(iter.device_type(), iter);
  // 返回右移后的自身 Tensor 引用
  return self;
}

Tensor& __irshift__(Tensor& self, const Scalar& other) {
  // 将标量 other 包装成 Tensor
  auto wrapper = wrapped_scalar_tensor(other);
  // 创建一个 Tensor 迭代器，执行就地赋值的二进制右移操作
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  // 调用底层的右移操作函数
  rshift_stub(iter.device_type(), iter);
  // 返回右移后的自身 Tensor 引用
  return self;
}

TORCH_IMPL_FUNC(bitwise_right_shift_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  // 调用底层的右移操作函数
  rshift_stub(device_type(), *this);
}

Tensor& bitwise_right_shift_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // 调用 at::bitwise_right_shift_out 函数，执行二进制右移操作
  return at::bitwise_right_shift_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_right_shift(const Tensor& self, const Scalar& other) {
  // 调用 at::bitwise_right_shift 函数，执行二进制右移操作
  return at::bitwise_right_shift(self, wrapped_scalar_tensor(other));
}

Tensor& bitwise_right_shift_(Tensor& self, const Scalar& other) {
  // 调用 at::bitwise_right_shift_out 函数，执行就地赋值的二进制右移操作
  return at::bitwise_right_shift_out(self, self, wrapped_scalar_tensor(other));
}
// 返回按位右移操作后的 Tensor 结果
Tensor bitwise_right_shift(const Scalar& self, const Tensor& other) {
  // 调用 wrapped_scalar_tensor 将 self 包装成 Tensor，然后进行按位右移操作
  return at::bitwise_right_shift(wrapped_scalar_tensor(self), other);
}

// 比较操作的通用函数模板，用于处理两个 Tensor 之间的比较
template <typename Stub>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Tensor& other, Stub& stub) {
  // 创建一个 Tensor 迭代器，用于执行比较操作
  auto iter = TensorIterator::comparison_op(result, self, other);
  // 使用 stub 函数处理迭代器，这里假设 stub 是一个可以处理迭代器的函数对象
  stub(iter.device_type(), iter);
  // 返回比较结果的 Tensor
  return result;
}

// 比较操作的模板函数，返回两个 Tensor 之间的比较结果
template <typename OutImpl>
Tensor comparison_op(const Tensor& self, const Tensor& other, OutImpl& out_impl) {
  // 创建一个空的 Tensor 用于存放比较结果，数据类型为 kBool 类型
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  // 调用 out_impl 处理比较操作，并返回结果 Tensor
  return out_impl(result, self, other);
}

// 原位比较操作的模板函数，修改 self Tensor 本身并返回
template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Tensor& other, OutImpl& out_impl) {
  // 调用 out_impl 处理原位比较操作，并返回修改后的 self Tensor
  return out_impl(self, self, other);
}

// 比较操作的模板函数，用于处理 Tensor 和 Scalar 之间的比较
template <typename OutImpl>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  // 将 Scalar 包装成 Tensor 后，调用 out_impl 处理比较操作
  return out_impl(result, self, wrapped_scalar_tensor(other));
}

// 比较操作的模板函数，用于处理 Tensor 和 Scalar 之间的比较
template <typename OutImpl>
Tensor comparison_op(const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  // 调用前面定义的比较操作模板函数，处理 Tensor 和 Scalar 之间的比较
  return comparison_op(self, wrapped_scalar_tensor(other), out_impl);
}

// 原位比较操作的模板函数，用于处理 Tensor 和 Scalar 之间的比较
template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Scalar& other, OutImpl& out_impl) {
  // 调用前面定义的原位比较操作模板函数，处理 Tensor 和 Scalar 之间的比较
  return out_impl(self, self, wrapped_scalar_tensor(other));
}

// OutFunc 是一个类型别名，用于处理 *_out 函数的重载问题
using OutFunc = std::add_const<Tensor&(&)(Tensor&, const Tensor&, const Tensor&)>::type;

// less 函数的别名，用于处理 Tensor 之间的小于比较操作
Tensor& less_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::lt_out(result, self, other); }
// less 函数，用于返回 Tensor 之间的小于比较结果
Tensor less(const Tensor& self, const Tensor& other) { return self.lt(other); }
// less_ 函数，用于修改 self Tensor，进行小于比较操作
Tensor& less_(Tensor& self, const Tensor& other) { return self.lt_(other); }
// less_out 函数，用于处理 Tensor 和 Scalar 之间的小于比较操作
Tensor& less_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::lt_out(result, self, other); }
// less 函数，用于返回 Tensor 和 Scalar 之间的小于比较结果
Tensor less(const Tensor& self, const Scalar& other) { return self.lt(other); }
// less_ 函数，用于修改 self Tensor，进行小于比较操作
Tensor& less_(Tensor& self, const Scalar& other) { return self.lt_(other); }

// less_equal 函数的别名，用于处理 Tensor 之间的小于等于比较操作
Tensor& less_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::le_out(result, self, other); }
// less_equal 函数，用于返回 Tensor 之间的小于等于比较结果
Tensor less_equal(const Tensor& self, const Tensor& other) { return self.le(other); }
// less_equal_ 函数，用于修改 self Tensor，进行小于等于比较操作
Tensor& less_equal_(Tensor& self, const Tensor& other) { return self.le_(other); }
// less_equal_out 函数，用于处理 Tensor 和 Scalar 之间的小于等于比较操作
Tensor& less_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::le_out(result, self, other); }
// less_equal 函数，用于返回 Tensor 和 Scalar 之间的小于等于比较结果
Tensor less_equal(const Tensor& self, const Scalar& other) { return self.le(other); }
// less_equal_ 函数，用于修改 self Tensor，进行小于等于比较操作
Tensor& less_equal_(Tensor& self, const Scalar& other) { return self.le_(other); }

// greater 函数的别名，用于处理 Tensor 之间的大于比较操作
Tensor& greater_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::gt_out(result, self, other); }
// greater 函数，用于返回 Tensor 之间的大于比较结果
Tensor greater(const Tensor& self, const Tensor& other) { return self.gt(other); }
// 用于原地计算张量 self 大于 other 的元素，返回修改后的 self 引用
Tensor& greater_(Tensor& self, const Tensor& other) { return self.gt_(other); }

// 计算张量 self 和标量 other 的大于关系，结果存入 result 张量中，返回修改后的 result 引用
Tensor& greater_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::gt_out(result, self, other); }

// 返回张量 self 中大于标量 other 的元素的新张量
Tensor greater(const Tensor& self, const Scalar& other) { return self.gt(other); }

// 在张量 self 上原地计算大于标量 other 的元素，返回修改后的 self 引用
Tensor& greater_(Tensor& self, const Scalar& other) { return self.gt_(other); }

// greater_equal 的别名，调用 torch.ge_out
Tensor& greater_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::ge_out(result, self, other); }

// 返回张量 self 中大于等于另一张量 other 的元素的新张量
Tensor greater_equal(const Tensor& self, const Tensor& other) { return self.ge(other); }

// 在张量 self 上原地计算大于等于另一张量 other 的元素，返回修改后的 self 引用
Tensor& greater_equal_(Tensor& self, const Tensor& other) { return self.ge_(other); }

// 在张量 self 上原地计算大于等于标量 other 的元素，返回修改后的 self 引用
Tensor& greater_equal_(Tensor& self, const Scalar& other) { return self.ge_(other); }

// 宏定义，用于生成比较标量和张量的函数实现
#define CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(func)             \
  TORCH_IMPL_FUNC(func##_Tensor_out)                                \
  (const Tensor& self, const Tensor& other, const Tensor& result) { \
    func##_stub(device_type(), *this);                              \
  }                                                                 \
                                                                    \
  TORCH_IMPL_FUNC(func##_Scalar_out)                                \
  (const Tensor& self, const Scalar& other, const Tensor& result) { \
    func##_stub(device_type(), *this);                              \
  }

// 创建各种比较操作的标量和张量实现函数
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(eq);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(ne);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(gt);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(ge);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(lt);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(le);

// not_equal 的别名，调用 torch.ne_out
Tensor& not_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::ne_out(result, self, other); }

// 返回张量 self 中不等于另一张量 other 的元素的新张量
Tensor not_equal(const Tensor& self, const Tensor& other) { return self.ne(other); }

// 在张量 self 上原地计算不等于另一张量 other 的元素，返回修改后的 self 引用
Tensor& not_equal_(Tensor& self, const Tensor& other) { return self.ne_(other); }

// 在张量 self 上原地计算不等于标量 other 的元素，返回修改后的 self 引用
Tensor& not_equal_(Tensor& self, const Scalar& other) { return self.ne_(other); }

// 逻辑与运算的输出函数，调用 comparison_op_out
Tensor& logical_and_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, logical_and_stub); }

// 计算张量 self 和另一张量 other 的逻辑与，返回新张量
Tensor logical_and(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out)); }

// 在张量 self 上原地计算与另一张量 other 的逻辑与，返回修改后的 self 引用
Tensor& logical_and_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out)); }
// 返回按逻辑或操作的结果到给定结果张量的引用
Tensor& logical_or_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // 调用比较操作的输出版本，使用逻辑或的函数逻辑
  return comparison_op_out(result, self, other, logical_or_stub);
}

// 返回按逻辑或操作的结果的张量
Tensor logical_or(const Tensor& self, const Tensor& other) {
  // 调用比较操作，使用逻辑或的函数逻辑
  return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out));
}

// 原地执行逻辑或操作，修改自身并返回修改后的结果的引用
Tensor& logical_or_(Tensor& self, const Tensor& other) {
  // 调用比较操作的原地版本，使用逻辑或的函数逻辑
  return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out));
}

// 返回按逻辑异或操作的结果到给定结果张量的引用
Tensor& logical_xor_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // 调用比较操作的输出版本，使用逻辑异或的函数逻辑
  return comparison_op_out(result, self, other, logical_xor_stub);
}

// 返回按逻辑异或操作的结果的张量
Tensor logical_xor(const Tensor& self, const Tensor& other) {
  // 调用比较操作，使用逻辑异或的函数逻辑
  return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out));
}

// 原地执行逻辑异或操作，修改自身并返回修改后的结果的引用
Tensor& logical_xor_(Tensor& self, const Tensor& other) {
  // 调用比较操作的原地版本，使用逻辑异或的函数逻辑
  return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out));
}

// 返回按最大值操作的结果到给定结果张量的引用，别名为 maximum
Tensor& max_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // 调用 at::maximum_out 函数，计算两个张量的最大值
  return at::maximum_out(result, self, other);
}

// 返回两个张量的最大值张量
Tensor max(const Tensor& self, const Tensor& other) {
  // 调用 at::maximum 函数，计算两个张量的最大值
  return at::maximum(self, other);
}

// 返回按最小值操作的结果到给定结果张量的引用，别名为 minimum
Tensor& min_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // 调用 at::minimum_out 函数，计算两个张量的最小值
  return at::minimum_out(result, self, other);
}

// 返回两个张量的最小值张量
Tensor min(const Tensor& self, const Tensor& other) {
  // 调用 at::minimum 函数，计算两个张量的最小值
  return at::minimum(self, other);
}

// 返回张量与标量之间的 floor divide 运算结果
Tensor floor_divide(const Tensor& self, const Scalar& other) {
  // 调用 at::floor_divide 函数，对张量 self 执行 floor divide 运算
  return at::floor_divide(self, wrapped_scalar_tensor(other));
}

// 原地执行张量与标量之间的 floor divide 运算，修改自身并返回修改后的结果的引用
Tensor& floor_divide_(Tensor& self, const Scalar& other) {
  // 调用 at::floor_divide_out 函数，对张量 self 原地执行 floor divide 运算
  return at::floor_divide_out(self, self, wrapped_scalar_tensor(other));
}

// 返回按 fmod 操作的结果到给定结果张量的引用
Tensor& fmod_out(const Tensor& self, const Scalar& other, Tensor & result) {
  // 重新分派调用 at::fmod_out 函数，计算张量 self 和标量 other 的 fmod
  return at::fmod_out(result, self, wrapped_scalar_tensor(other));
}

// 返回按 fmod 操作的结果的张量
Tensor fmod(const Tensor& self, const Scalar& other) {
  // 重新分派调用 at::fmod 函数，计算张量 self 和标量 other 的 fmod
  return at::fmod(self, wrapped_scalar_tensor(other));
}

// 原地执行按 fmod 操作的结果，修改自身并返回修改后的结果的引用
Tensor& fmod_(Tensor& self, const Scalar& other) {
  // 重新分派调用张量对象的 fmod_ 方法，计算自身和标量 other 的 fmod
  return self.fmod_(wrapped_scalar_tensor(other));
}

// 注意：此函数仅用于测试。
// 此函数未记录，并且不应在测试以外的场合使用。
// 返回两个张量的差减去另一个张量乘以标量的结果
Tensor _test_serialization_subcmul(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self - (other * alpha);
}

// TORCH_IMPL_FUNC 宏定义的 heaviside_out 函数，具体实现未提供
// 根据设备类型和当前对象调用 heaviside_stub 函数
TORCH_IMPL_FUNC(heaviside_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  heaviside_stub(device_type(), *this);
}

// 返回按 ldexp 操作的结果到给定结果张量的引用
Tensor& ldexp_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // 调用 at::mul_out 函数，计算张量 self 和 2 的 other 次方的乘积
  return at::mul_out(result, self, at::pow(2.0, other));
}

// 返回按 ldexp 操作的结果的张量
Tensor ldexp(const Tensor& self, const Tensor& other) {
  // 调用 at::mul 函数，计算张量 self 和 2 的 other 次方的乘积
  return at::mul(self, at::pow(2.0, other));
}

// 原地执行按 ldexp 操作的结果，修改自身并返回修改后的结果的引用
Tensor& ldexp_(Tensor& self, const Tensor& other) {
  // 调用 at::ldexp_out 函数，对自身进行 ldexp 运算
  return at::ldexp_out(self, self, other);
}

// 返回按 xlogy 操作的结果到给定结果张量的引用
Tensor& xlogy_out(const Scalar& self, const Tensor& other, Tensor& result) {
  // 调用 at::xlogy_out 函数，计算标量 self 和张量 other 的 xlogy
  return at::xlogy_out(result, wrapped_scalar_tensor(self), other);
}
// 将 self 的每个元素与标量 other 的自然对数相乘，结果存入 result 中，返回 result 引用
Tensor& xlogy_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // 调用 ATen 库的 xlogy_out 函数，传入 result、self 和封装后的标量 other 的张量
  return at::xlogy_out(result, self, wrapped_scalar_tensor(other));
}

// 计算标量 x 与张量 y 的每个元素的自然对数相乘，返回结果张量
Tensor xlogy(const Scalar& x, const Tensor& y) {
  // 调用 ATen 库的 xlogy 函数，传入封装后的标量 x 和张量 y
  return at::xlogy(wrapped_scalar_tensor(x), y);
}

// 计算张量 x 的每个元素与标量 y 的自然对数相乘，返回结果张量
Tensor xlogy(const Tensor& x, const Scalar& y) {
  // 调用 ATen 库的 xlogy 函数，传入张量 x 和封装后的标量 y
  return at::xlogy(x, wrapped_scalar_tensor(y));
}

// 将张量 x 的每个元素与标量 y 的自然对数相乘，结果存回 x 中，返回 x 的引用
Tensor& xlogy_(Tensor& x, const Scalar& y) {
  // 调用 ATen 库的 xlogy_ 函数，传入 x 和封装后的标量 y
  return at::xlogy_(x, wrapped_scalar_tensor(y));
}

// 将 self 的每个元素与 other 的自然对数相乘，结果存入 result 中，返回 result 引用
Tensor& special_xlogy_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // 调用 ATen 库的 xlogy_out 函数，传入 result、self 和 other
  return at::xlogy_out(result, self, other);
}

// 将标量 self 与 other 的每个元素的自然对数相乘，结果存入 result 中，返回 result 引用
Tensor& special_xlogy_out(const Scalar& self, const Tensor& other, Tensor& result) {
  // 调用 ATen 库的 xlogy_out 函数，传入 result、封装后的标量 self 和 other
  return at::xlogy_out(result, self, other);
}

// 将 self 的每个元素与标量 other 的自然对数相乘，结果存入 result 中，返回 result 引用
Tensor& special_xlogy_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // 调用 ATen 库的 xlogy_out 函数，传入 result、self 和封装后的标量 other
  return at::xlogy_out(result, self, other);
}

// 计算两个张量 x 和 y 的每个元素的自然对数相乘，返回结果张量
Tensor special_xlogy(const Tensor& x, const Tensor& y) {
  // 调用 ATen 库的 xlogy 函数，传入张量 x 和 y
  return at::xlogy(x, y);
}

// 计算标量 x 与张量 y 的每个元素的自然对数相乘，返回结果张量
Tensor special_xlogy(const Scalar& x, const Tensor& y) {
  // 调用 ATen 库的 xlogy 函数，传入封装后的标量 x 和张量 y
  return at::xlogy(x, y);
}

// 计算张量 x 的每个元素与标量 y 的自然对数相乘，返回结果张量
Tensor special_xlogy(const Tensor& x, const Scalar& y) {
  // 调用 ATen 库的 xlogy 函数，传入张量 x 和封装后的标量 y
  return at::xlogy(x, y);
}

} // namespace at::native
```