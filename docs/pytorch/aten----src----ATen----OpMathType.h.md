# `.\pytorch\aten\src\ATen\OpMathType.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <c10/core/ScalarType.h>
// 引入 C10 库中的 ScalarType 头文件，定义了数据类型

#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
// 引入 C10 库中各种数据类型的头文件

namespace at {

// 命名空间 at，包含了 PyTorch 的核心实现

// For FP16 or BFloat16 inputs, ops should perform internal math in FP32.
// 对于 FP16 或 BFloat16 输入，操作应在 FP32 内部进行数学运算。

template <typename scalar_t>
struct OpMathType {
  using type = scalar_t;
};
// 定义一个结构体模板 OpMathType，用于映射不同数据类型的数学计算类型

template <>
struct OpMathType<at::Half> {
  using type = float;
};
// 对于 Half 类型，数学计算类型为 float

template <>
struct OpMathType<at::BFloat16> {
  using type = float;
};
// 对于 BFloat16 类型，数学计算类型为 float

template <>
struct OpMathType<at::Float8_e5m2> {
  using type = float;
};
// 对于 Float8_e5m2 类型，数学计算类型为 float

template <>
struct OpMathType<at::Float8_e4m3fn> {
  using type = float;
};
// 对于 Float8_e4m3fn 类型，数学计算类型为 float

template <>
struct OpMathType<at::Float8_e5m2fnuz> {
  using type = float;
};
// 对于 Float8_e5m2fnuz 类型，数学计算类型为 float

template <>
struct OpMathType<at::Float8_e4m3fnuz> {
  using type = float;
};
// 对于 Float8_e4m3fnuz 类型，数学计算类型为 float

template <>
struct OpMathType<c10::complex<Half>> {
  using type = c10::complex<float>;
};
// 对于 Half 复数类型，数学计算类型为 float 复数类型

template <typename T>
using opmath_type = typename OpMathType<T>::type;
// 定义模板别名 opmath_type，用于获取特定类型的数学计算类型

namespace {

inline c10::ScalarType toOpMathType(const c10::ScalarType type) {
  // 定义局部匿名命名空间，包含 toOpMathType 函数

  switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum) \
  case ScalarType::TypeNum:            \
    return CppTypeToScalarType<at::opmath_type<scalar_t>>::value;
    // 根据输入的 ScalarType 类型进行切换

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
    // 对于所有标量类型及其复数类型，执行宏定义的切换情况

#undef DEFINE_CASE

    default:
      TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
      // 默认情况下，如果未识别的 ScalarType，触发内部断言错误
  }
}

} // namespace

} // namespace at
// 结束命名空间 at
```