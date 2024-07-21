# `.\pytorch\aten\src\ATen\NumericUtils.h`

```
#pragma once
// 如果正在使用 HIPCC 编译器，包含 HIP 运行时头文件
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

// 包含 C10 库中的宏定义和数据类型头文件
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

// 包含数学运算函数头文件
#include <cmath>
// 包含类型特性判断函数头文件
#include <type_traits>

// 定义在 at 命名空间中
namespace at {

// 对于整数类型 T，定义 _isnan 函数，始终返回 false
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T /*val*/) {
  return false;
}

// 对于浮点数类型 T，定义 _isnan 函数，根据编译器类型调用相应的 std::isnan 函数
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isnan(val); // 使用 GPU 编译器提供的 isnan 函数
#else
  return std::isnan(val); // 使用标准库提供的 isnan 函数
#endif
}

// 对于复数类型 T，定义 _isnan 函数，判断实部或虚部是否为 NaN
template <typename T, std::enable_if_t<c10::is_complex<T>::value, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return std::isnan(val.real()) || std::isnan(val.imag());
}

// 对于 Half 类型，定义 _isnan 函数，将 Half 转换为 float 后调用相应的 _isnan 函数
template <typename T, std::enable_if_t<std::is_same_v<T, at::Half>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return at::_isnan(static_cast<float>(val));
}

// 对于 BFloat16 类型，定义 _isnan 函数，将 BFloat16 转换为 float 后调用相应的 _isnan 函数
template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::BFloat16>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(at::BFloat16 val) {
  return at::_isnan(static_cast<float>(val));
}

// 对于 Float8_e5m2 类型，定义 _isnan 函数，调用其成员函数 isnan 判断是否为 NaN
template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e5m2>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

// 对于 Float8_e4m3fn 类型，定义 _isnan 函数，调用其成员函数 isnan 判断是否为 NaN
template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e4m3fn>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

// 对于 Float8_e5m2fnuz 类型，定义 _isnan 函数，调用其成员函数 isnan 判断是否为 NaN
template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e5m2fnuz>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

// 对于 Float8_e4m3fnuz 类型，定义 _isnan 函数，调用其成员函数 isnan 判断是否为 NaN
template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e4m3fnuz>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

// 对于整数类型 T，定义 _isinf 函数，始终返回 false
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isinf(T /*val*/) {
  return false;
}

// 对于浮点数类型 T，定义 _isinf 函数，根据编译器类型调用相应的 std::isinf 函数
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isinf(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isinf(val); // 使用 GPU 编译器提供的 isinf 函数
#else
  return std::isinf(val); // 使用标准库提供的 isinf 函数
#endif
}

// 对于 Half 类型，定义 _isinf 函数，将 Half 转换为 float 后调用相应的 _isinf 函数
inline C10_HOST_DEVICE bool _isinf(at::Half val) {
  return at::_isinf(static_cast<float>(val));
}

// 对于 BFloat16 类型，定义 _isinf 函数，将 BFloat16 转换为 float 后调用相应的 _isinf 函数
inline C10_HOST_DEVICE bool _isinf(at::BFloat16 val) {
  return at::_isinf(static_cast<float>(val));
}
// 检查给定的 at::Float8_e5m2 类型的值是否为无穷大
inline C10_HOST_DEVICE bool _isinf(at::Float8_e5m2 val) {
  return val.isinf();
}

// 对于 at::Float8_e4m3fn 类型的值，始终返回 false，表示不是无穷大
inline C10_HOST_DEVICE bool _isinf(at::Float8_e4m3fn val) {
  return false;
}

// 对于 at::Float8_e5m2fnuz 类型的值，始终返回 false，表示不是无穷大
inline C10_HOST_DEVICE bool _isinf(at::Float8_e5m2fnuz val) {
  return false;
}

// 对于 at::Float8_e4m3fnuz 类型的值，始终返回 false，表示不是无穷大
inline C10_HOST_DEVICE bool _isinf(at::Float8_e4m3fnuz val) {
  return false;
}

// 计算指数函数 exp(x)，对于双精度类型抛出静态断言
template <typename T>
C10_HOST_DEVICE inline T exp(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // 在 CUDA 或 HIP 架构下使用 __expf 快速近似计算，以提高带宽利用率
  return __expf(x);
#else
  // 在其它架构下使用标准库中的 exp 函数
  return ::exp(x);
#endif
}

// 对于双精度类型的 exp 函数的特化版本，直接使用标准库中的 exp 函数
template <>
C10_HOST_DEVICE inline double exp<double>(double x) {
  return ::exp(x);
}

// 计算自然对数函数 log(x)，对于双精度类型抛出静态断言
template <typename T>
C10_HOST_DEVICE inline T log(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // 在 CUDA 或 HIP 架构下使用 __logf 快速近似计算，以提高带宽利用率
  return __logf(x);
#else
  // 在其它架构下使用标准库中的 log 函数
  return ::log(x);
#endif
}

// 对于双精度类型的 log 函数的特化版本，直接使用标准库中的 log 函数
template <>
C10_HOST_DEVICE inline double log<double>(double x) {
  return ::log(x);
}

// 计算 log(1 + x) 函数 log1p(x)，对于双精度类型抛出静态断言
template <typename T>
C10_HOST_DEVICE inline T log1p(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // 在 CUDA 或 HIP 架构下使用 __logf 快速近似计算，注意这里无法直接使用 __log1pf 函数，损失精度
  return __logf(1.0f + x);
#else
  // 在其它架构下使用标准库中的 log1p 函数
  return ::log1p(x);
#endif
}

// 对于双精度类型的 log1p 函数的特化版本，直接使用标准库中的 log1p 函数
template <>
C10_HOST_DEVICE inline double log1p<double>(double x) {
  return ::log1p(x);
}

// 计算正切函数 tan(x)，对于双精度类型抛出静态断言
template <typename T>
C10_HOST_DEVICE inline T tan(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // 在 CUDA 或 HIP 架构下使用 __tanf 快速近似计算
  return __tanf(x);
#else
  // 在其它架构下使用标准库中的 tan 函数
  return ::tan(x);
#endif
}

// 对于双精度类型的 tan 函数的特化版本，直接使用标准库中的 tan 函数
template <>
C10_HOST_DEVICE inline double tan<double>(double x) {
  return ::tan(x);
}

} // namespace at
```