# `.\pytorch\aten\src\ATen\TensorOperators.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/core/Tensor.h>
// 引入 ATen 库中的 Tensor 类定义

#include <c10/core/Scalar.h>
// 引入 c10 库中的 Scalar 类定义

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif
// 条件编译：根据 AT_PER_OPERATOR_HEADERS 宏的定义选择性包含不同的头文件

namespace at {

#define AT_FORALL_BINARY_OPS(_)                                             \
  _(+, x.add(y), y.add(x))                                                  \
  _(*, x.mul(y), y.mul(x))                                                  \
  _(-,                                                                      \
    x.sub(y),                                                               \
    ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).sub_(y))       \
  _(/,                                                                      \
    x.div(y),                                                               \
    ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).div_(y))       \
  _(%,                                                                      \
    x.remainder(y),                                                         \
    ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).remainder_(y)) \
  _(&, x.bitwise_and(y), y.bitwise_and(x))                                  \
  _(|, x.bitwise_or(y), y.bitwise_or(x))                                    \
  _(^, x.bitwise_xor(y), y.bitwise_xor(x))                                  \
  _(<, x.lt(y), y.gt(x))                                                    \
  _(<=, x.le(y), y.ge(x))                                                   \
  _(>, x.gt(y), y.lt(x))                                                    \
  _(>=, x.ge(y), y.le(x))                                                   \
  _(==, x.eq(y), y.eq(x))                                                   \
  _(!=, x.ne(y), y.ne(x))
// 宏定义 AT_FORALL_BINARY_OPS：定义了一系列二元操作符及其对应的操作，包括加减乘除、位运算、比较操作等

#define DEFINE_OPERATOR(op, body, reverse_scalar_body)                 \
  static inline Tensor operator op(const Tensor& x, const Tensor& y) { \
    return body;                                                       \
  }                                                                    \
  static inline Tensor operator op(const Tensor& x, const Scalar& y) { \
    return body;                                                       \
  }                                                                    \
  static inline Tensor operator op(const Scalar& x, const Tensor& y) { \
    return reverse_scalar_body;                                        \
  }
// 宏定义 DEFINE_OPERATOR：定义了操作符重载函数，分别处理两个 Tensor、一个 Tensor 和一个 Scalar、一个 Scalar 和一个 Tensor 的情况

AT_FORALL_BINARY_OPS(DEFINE_OPERATOR)
// 使用 AT_FORALL_BINARY_OPS 宏展开来定义所有二元操作符的重载函数

#undef DEFINE_OPERATOR
#undef AT_FORALL_BINARY_OPS
// 清除先前定义的宏，避免影响后续代码

} // namespace at
// 命名空间结束
```