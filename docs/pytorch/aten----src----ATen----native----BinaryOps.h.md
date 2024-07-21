# `.\pytorch\aten\src\ATen\native\BinaryOps.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/core/TensorBase.h>
// 引入 ATen 核心模块中的 TensorBase 类定义

#include <ATen/native/DispatchStub.h>
// 引入 ATen 本地分发存根定义

#include <c10/core/Scalar.h>
// 引入 c10 核心模块中的 Scalar 类定义

#include <c10/util/TypeSafeSignMath.h>
// 引入 c10 工具模块中的 TypeSafeSignMath 头文件

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
}
// 在 at 命名空间中声明 TensorIterator 和 TensorIteratorBase 结构体

namespace at::native {

inline void alpha_check(const ScalarType dtype, const Scalar& alpha) {
  // 检查 alpha 值是否符合特定条件，用于运算时的类型和值域检查
  TORCH_CHECK(! alpha.isBoolean() || dtype == ScalarType::Bool,
              "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype)
              || alpha.isIntegral(true),
              "For integral input tensors, argument alpha must not be a floating point number.");
  TORCH_CHECK(isComplexType(dtype) || !alpha.isComplex(),
              "For non-complex input tensors, argument alpha must not be a complex number.")
}

// 对所有子函数进行基本检查
inline void sub_check(const TensorBase& self, const TensorBase& other) {
  // 检查两个张量是否支持减法操作，并给出相应的错误信息
  TORCH_CHECK(self.scalar_type() != kBool || other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with two bool tensors is not supported. "
              "Use the `^` or `logical_xor()` operator instead.")
  TORCH_CHECK(self.scalar_type() != kBool && other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
}

inline void sub_check(const TensorBase& self, const Scalar& scalar) {
  // 检查张量和标量之间是否支持减法操作，并给出相应的错误信息
  TORCH_CHECK(self.scalar_type() != kBool || !scalar.isBoolean(),
              "Subtraction, the `-` operator, with two bool tensors is not supported. "
              "Use the `^` or `logical_xor()` operator instead.")
  TORCH_CHECK(self.scalar_type() != kBool && !scalar.isBoolean(),
              "Subtraction, the `-` operator, with a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
}

using structured_binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);
// 定义一个函数指针类型，用于接受具有 alpha 参数的结构化二元操作函数

using structured_binary_fn_double = void(*)(TensorIteratorBase&, double);
// 定义一个函数指针类型，用于接受具有 double 类型参数的结构化二元操作函数

using structured_binary_fn = void(*)(TensorIteratorBase&);
// 定义一个函数指针类型，用于接受没有额外参数的结构化二元操作函数

using binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);
// 定义一个函数指针类型，用于接受具有 alpha 参数的二元操作函数

using binary_fn_double = void(*)(TensorIterator&, double);
// 定义一个函数指针类型，用于接受具有 double 类型参数的二元操作函数

using binary_fn = void(*)(TensorIterator&);
// 定义一个函数指针类型，用于接受没有额外参数的二元操作函数

using binary_clamp_fn_alpha =
    void(*)(TensorIterator&, const Scalar& alpha, const Scalar& min_val, const Scalar& max_val);
// 定义一个函数指针类型，用于接受具有 alpha、最小值和最大值参数的二元操作函数

// NB: codegenned
// 注意：此处的函数声明是通过代码生成生成的

DECLARE_DISPATCH(structured_binary_fn_alpha, add_stub);
// 声明一个分发函数，用于结构化二元加法操作

DECLARE_DISPATCH(binary_clamp_fn_alpha, add_clamp_stub);
// 声明一个分发函数，用于带 clamp 参数的二元加法操作

DECLARE_DISPATCH(structured_binary_fn_alpha, sub_stub);
// 声明一个分发函数，用于结构化二元减法操作

DECLARE_DISPATCH(structured_binary_fn, mul_stub);
// 声明一个分发函数，用于结构化二元乘法操作

DECLARE_DISPATCH(structured_binary_fn, div_true_stub);
// 声明一个分发函数，用于结构化真除操作

DECLARE_DISPATCH(structured_binary_fn, div_floor_stub);
// 声明一个分发函数，用于结构化地板除操作

DECLARE_DISPATCH(structured_binary_fn, div_trunc_stub);
// 声明一个分发函数，用于结构化截断除操作

DECLARE_DISPATCH(structured_binary_fn, atan2_stub);
// 声明一个分发函数，用于结构化反正切操作

DECLARE_DISPATCH(structured_binary_fn, remainder_stub);
// 声明一个分发函数，用于结构化求余操作
# 声明并分发结构化二进制函数，具体为按位与操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, bitwise_and_stub);
# 声明并分发结构化二进制函数，具体为按位或操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, bitwise_or_stub);
# 声明并分发结构化二进制函数，具体为按位异或操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, bitwise_xor_stub);
# 声明并分发结构化二进制函数，具体为左移操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, lshift_stub);
# 声明并分发结构化二进制函数，具体为右移操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, rshift_stub);

# 声明并分发二进制函数，具体为逻辑异或操作的存根函数声明
DECLARE_DISPATCH(binary_fn, logical_xor_stub);
# 声明并分发二进制函数，具体为逻辑与操作的存根函数声明
DECLARE_DISPATCH(binary_fn, logical_and_stub);
# 声明并分发二进制函数，具体为逻辑或操作的存根函数声明
DECLARE_DISPATCH(binary_fn, logical_or_stub);

# 声明并分发结构化二进制函数，具体为小于操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, lt_stub);
# 声明并分发结构化二进制函数，具体为小于等于操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, le_stub);
# 声明并分发结构化二进制函数，具体为大于操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, gt_stub);
# 声明并分发结构化二进制函数，具体为大于等于操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, ge_stub);
# 声明并分发结构化二进制函数，具体为等于操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, eq_stub);
# 声明并分发结构化二进制函数，具体为不等于操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, ne_stub);

# 声明并分发二进制函数，具体为元素级最大值操作的存根函数声明
DECLARE_DISPATCH(binary_fn, max_elementwise_stub);
# 声明并分发二进制函数，具体为元素级最小值操作的存根函数声明
DECLARE_DISPATCH(binary_fn, min_elementwise_stub);

# 声明并分发结构化二进制函数，具体为元素级最大值操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, maximum_stub);
# 声明并分发结构化二进制函数，具体为元素级最小值操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, minimum_stub);
# 声明并分发结构化二进制函数，具体为浮点数最大值操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, fmax_stub);
# 声明并分发结构化二进制函数，具体为浮点数最小值操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, fmin_stub);

# 声明并分发双精度结构化二进制函数，具体为平滑 L1 损失函数的存根函数声明
DECLARE_DISPATCH(structured_binary_fn_double, smooth_l1_stub);
# 声明并分发双精度二进制函数，具体为 Huber 损失函数的存根函数声明
DECLARE_DISPATCH(binary_fn_double, huber_stub);

# 声明并分发结构化二进制函数，具体为 sigmoid 反向传播操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, sigmoid_backward_stub);
# 声明并分发带 alpha 的二进制函数，具体为 logistic 反向传播操作的存根函数声明
DECLARE_DISPATCH(binary_fn_alpha, logit_backward_stub);
# 声明并分发结构化二进制函数，具体为 tanh 反向传播操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, tanh_backward_stub);

# 声明并分发结构化二进制函数，具体为均方误差操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, mse_stub);
# 声明并分发结构化二进制函数，具体为取余操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, fmod_stub);
# 声明并分发结构化二进制函数，具体为 logaddexp 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, logaddexp_stub);
# 声明并分发结构化二进制函数，具体为 logaddexp2 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, logaddexp2_stub);

# 声明并分发结构化二进制函数，具体为最大公约数操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, gcd_stub);
# 声明并分发结构化二进制函数，具体为最小公倍数操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, lcm_stub);
# 声明并分发结构化二进制函数，具体为欧几里得距离操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, hypot_stub);

# 声明并分发结构化二进制函数，具体为下一个浮点数操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, nextafter_stub);
# 声明并分发结构化二进制函数，具体为海维赛德函数操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, heaviside_stub);
# 声明并分发结构化二进制函数，具体为符号复制操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, copysign_stub);

# 声明并分发结构化二进制函数，具体为 x*log(y) 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, xlogy_stub);
# 声明并分发结构化二进制函数，具体为 x*log(1+y) 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, xlog1py_stub);

# 声明并分发结构化二进制函数，具体为 Riemann zeta 函数操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, zeta_stub);

# 声明并分发结构化二进制函数，具体为切比雪夫多项式 T 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, chebyshev_polynomial_t_stub);
# 声明并分发结构化二进制函数，具体为切比雪夫多项式 U 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, chebyshev_polynomial_u_stub);
# 声明并分发结构化二进制函数，具体为切比雪夫多项式 V 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, chebyshev_polynomial_v_stub);
# 声明并分发结构化二进制函数，具体为切比雪夫多项式 W 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, chebyshev_polynomial_w_stub);

# 声明并分发结构化二进制函数，具体为 Hermite 多项式 H 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, hermite_polynomial_h_stub);
# 声明并分发结构化二进制函数，具体为 Hermite 多项式 He 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, hermite_polynomial_he_stub);
# 声明并分发结构化二进制函数，具体为 Laguerre 多项式 L 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, laguerre_polynomial_l_stub);
# 声明并分发结构化二进制函数，具体为 Legendre 多项式 P 操作的存根函数声明
DECLARE_DISPATCH(structured_binary_fn, legendre_polynomial_p_stub);

# 声明并分发结构
```