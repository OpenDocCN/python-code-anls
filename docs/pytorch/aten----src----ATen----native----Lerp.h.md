# `.\pytorch\aten\src\ATen\native\Lerp.h`

```
#pragma once

#include <ATen/native/DispatchStub.h>  // 包含分发函数声明的头文件
#include <ATen/OpMathType.h>           // 包含操作数数学类型的头文件
#include <ATen/TensorIterator.h>       // 包含张量迭代器的头文件
#include <c10/core/Scalar.h>           // 包含标量类型的头文件

namespace at::native {

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE bool is_lerp_weight_small(scalar_t weight) {
  return std::abs(weight) < scalar_t(0.5);  // 检查线性插值权重是否小于0.5的绝对值
}
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE bool is_lerp_weight_small(c10::complex<scalar_t> weight) {
  // 避免在 abs(weight) 中使用 sqrt
  return (weight.real() * weight.real() + weight.imag() * weight.imag()) < scalar_t(0.25);
}

template <typename scalar_t, typename weight_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t lerp(scalar_t self_, scalar_t end_, weight_t weight_) {
  using opmath_t = at::opmath_type<scalar_t>;         // 定义操作数的数学类型
  using opmath_weight_t = at::opmath_type<weight_t>; // 定义权重的数学类型

  opmath_t self = self_;
  opmath_t end = end_;
  opmath_weight_t weight = weight_;

  // 条件分支以优化数值计算。此处讨论在 https://github.com/pytorch/pytorch/pull/18871 中有提及
  return is_lerp_weight_small(weight)
      ? self + weight * (end - self)  // 如果权重较小，使用简化的线性插值公式
      : end - (end - self) * (opmath_t(1) - weight);  // 否则，使用权重反向插值公式
}

using lerp_fn_scalar = void (*)(
    at::TensorIteratorBase& iter,
    const Scalar& weight);  // 定义接受标量权重的线性插值函数指针类型

using lerp_fn_tensor = void (*)(
    at::TensorIteratorBase& iter);  // 定义接受张量权重的线性插值函数指针类型

DECLARE_DISPATCH(lerp_fn_scalar, lerp_kernel_scalar_weight);  // 声明标量权重的线性插值分发函数
DECLARE_DISPATCH(lerp_fn_tensor, lerp_kernel_tensor_weight);  // 声明张量权重的线性插值分发函数

} // namespace at::native
```