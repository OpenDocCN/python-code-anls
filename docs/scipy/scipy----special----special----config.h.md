# `D:\src\scipysrc\scipy\scipy\special\special\config.h`

```
#pragma once
// 如果未定义 M_E，定义为自然对数的底数 e
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// 如果未定义 M_LOG2E，定义为以 2 为底的对数 e 的倒数
#ifndef M_LOG2E
#define M_LOG2E 1.44269504088896340736
#endif

// 如果未定义 M_LOG10E，定义为以 10 为底的对数 e 的倒数
#ifndef M_LOG10E
#define M_LOG10E 0.434294481903251827651
#endif

// 如果未定义 M_LN2，定义为自然对数 2 的值
#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

// 如果未定义 M_LN10，定义为自然对数 10 的值
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif

// 如果未定义 M_PI，定义为圆周率 π 的值
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 如果未定义 M_PI_2，定义为圆周率 π 的二分之一
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

// 如果未定义 M_PI_4，定义为圆周率 π 的四分之一
#ifndef M_PI_4
#define M_PI_4 0.785398163397448309616
#endif

// 如果未定义 M_1_PI，定义为圆周率 π 的倒数
#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538
#endif

// 如果未定义 M_2_PI，定义为圆周率 π 的二倍的倒数
#ifndef M_2_PI
#define M_2_PI 0.636619772367581343076
#endif

// 如果未定义 M_2_SQRTPI，定义为 2 乘以根号下的圆周率 π 的倒数
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390
#endif

// 如果未定义 M_SQRT2，定义为根号下的 2
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

// 如果未定义 M_SQRT1_2，定义为根号下的 2 的倒数
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524401
#endif

#ifdef __CUDACC__
// 在 CUDA 编译环境中，定义 SPECFUN_HOST_DEVICE 为 __host__ __device__

#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

// 如果使用 NVRTC JIT 编译器，则将部分函数回退到全局命名空间
#ifdef _LIBCUDACXX_COMPILER_NVRTC
#include <cuda_runtime.h>
#endif

namespace std {

// 在 CUDA 编译环境中，使用 CUDA 标准库的实现

SPECFUN_HOST_DEVICE inline double abs(double num) { return cuda::std::abs(num); }
SPECFUN_HOST_DEVICE inline double exp(double num) { return cuda::std::exp(num); }
SPECFUN_HOST_DEVICE inline double log(double num) { return cuda::std::log(num); }
SPECFUN_HOST_DEVICE inline double sqrt(double num) { return cuda::std::sqrt(num); }
SPECFUN_HOST_DEVICE inline bool isinf(double num) { return cuda::std::isinf(num); }
SPECFUN_HOST_DEVICE inline bool isnan(double num) { return cuda::std::isnan(num); }
SPECFUN_HOST_DEVICE inline bool isfinite(double num) { return cuda::std::isfinite(num); }
SPECFUN_HOST_DEVICE inline double pow(double x, double y) { return cuda::std::pow(x, y); }
SPECFUN_HOST_DEVICE inline double sin(double x) { return cuda::std::sin(x); }
SPECFUN_HOST_DEVICE inline double cos(double x) { return cuda::std::cos(x); }
SPECFUN_HOST_DEVICE inline double tan(double x) { return cuda::std::tan(x); }
SPECFUN_HOST_DEVICE inline double atan(double x) { return cuda::std::atan(x); }
SPECFUN_HOSt_DEVICE inline double acos(double x) { return cuda::std::acos(x); }
SPECFUN_HOST_DEVICE inline double sinh(double x) { return cuda::std::sinh(x); }
SPECFUN_HOST_DEVICE inline double cosh(double x) { return cuda::std::cosh(x); }
SPECFUN_HOST_DEVICE inline double asinh(double x) { return cuda::std::asinh(x); }
SPECFUN_HOST_DEVICE inline bool signbit(double x) { return cuda::std::signbit(x); }

// 除非使用 NVRTC，否则在 CUDA 环境中定义额外的数学函数
#ifndef _LIBCUDACXX_COMPILER_NVRTC
SPECFUN_HOST_DEVICE inline double ceil(double x) { return cuda::std::ceil(x); }
SPECFUN_HOST_DEVICE inline double floor(double x) { return cuda::std::floor(x); }
#endif  // _LIBCUDACXX_COMPILER_NVRTC
// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的四舍五入操作
SPECFUN_HOST_DEVICE inline double round(double x) { return cuda::std::round(x); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的截断操作
SPECFUN_HOST_DEVICE inline double trunc(double x) { return cuda::std::trunc(x); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的浮点数乘加操作
SPECFUN_HOST_DEVICE inline double fma(double x, double y, double z) { return cuda::std::fma(x, y, z); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的符号复制操作
SPECFUN_HOST_DEVICE inline double copysign(double x, double y) { return cuda::std::copysign(x, y); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的浮点数分解整数与小数部分操作
SPECFUN_HOST_DEVICE inline double modf(double value, double *iptr) { return cuda::std::modf(value, iptr); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的最大值操作
SPECFUN_HOST_DEVICE inline double fmax(double x, double y) { return cuda::std::fmax(x, y); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的最小值操作
SPECFUN_HOST_DEVICE inline double fmin(double x, double y) { return cuda::std::fmin(x, y); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的以 10 为底的对数操作
SPECFUN_HOST_DEVICE inline double log10(double num) { return cuda::std::log10(num); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的 log(1 + num) 操作
SPECFUN_HOST_DEVICE inline double log1p(double num) { return cuda::std::log1p(num); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的浮点数分解操作
SPECFUN_HOST_DEVICE inline double frexp(double num, int *exp) { return cuda::std::frexp(num); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的指数乘操作
SPECFUN_HOST_DEVICE inline double ldexp(double num, int *exp) { return cuda::std::ldexp(num); }

// 定义一个内联函数，用于在 CUDA 环境中执行 double 类型的浮点数取模操作
SPECFUN_HOST_DEVICE inline double fmod(double x, double y) { return cuda::std::fmod(x, y); }

#ifdef
// 若非 CUDA 环境，以下函数使用标准库函数实现

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的向上取整操作
SPECFUN_HOST_DEVICE inline double ceil(double x) { return ::ceil(x); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的向下取整操作
SPECFUN_HOST_DEVICE inline double floor(double x) { return ::floor(x); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的四舍五入操作
SPECFUN_HOST_DEVICE inline double round(double x) { return ::round(x); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的截断操作
SPECFUN_HOST_DEVICE inline double trunc(double x) { return ::trunc(x); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的浮点数乘加操作
SPECFUN_HOST_DEVICE inline double fma(double x, double y, double z) { return ::fma(x, y, z); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的符号复制操作
SPECFUN_HOST_DEVICE inline double copysign(double x, double y) { return ::copysign(x, y); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的浮点数分解整数与小数部分操作
SPECFUN_HOST_DEVICE inline double modf(double value, double *iptr) { return ::modf(value, iptr); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的最大值操作
SPECFUN_HOST_DEVICE inline double fmax(double x, double y) { return ::fmax(x, y); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的最小值操作
SPECFUN_HOST_DEVICE inline double fmin(double x, double y) { return ::fmin(x, y); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的以 10 为底的对数操作
SPECFUN_HOST_DEVICE inline double log10(double num) { return ::log10(num); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的 log(1 + num) 操作
SPECFUN_HOST_DEVICE inline double log1p(double num) { return ::log1p(num); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的浮点数分解操作
SPECFUN_HOST_DEVICE inline double frexp(double num, int *exp) { return ::frexp(num); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的指数乘操作
SPECFUN_HOST_DEVICE inline double ldexp(double num, int *exp) { return ::ldexp(num); }

// 定义一个内联函数，用于在非 CUDA 环境中执行 double 类型的浮点数取模操作
SPECFUN_HOST_DEVICE inline double fmod(double x, double y) { return ::fmod(x, y); }
#endif

// 定义一个模板函数，用于在 CUDA 环境中交换两个对象的值
template <typename T>
SPECFUN_HOST_DEVICE void swap(T &a, T &b) {
    cuda::std::swap(a, b);
}

// 定义一个模板函数，用于在 CUDA 环境中将一个值限制在给定范围内
template <typename T>
SPECFUN_HOST_DEVICE const T &clamp(const T &v, const T &lo, const T &hi) {
    return cuda::std::clamp(v, lo, hi);
}

// 定义一个模板别名，用于支持复数类型的数值限制操作
template <typename T>
using numeric_limits = cuda::std::numeric_limits<T>;

// 定义一个模板别名，用于支持复数类型的数学运算
template <typename T>
using complex = thrust::complex<T>;

// 定义一个模板函数，用于在 CUDA 环境中计算复数的绝对值
template <typename T>
SPECFUN_HOST_DEVICE T abs(const complex<T> &z) {
    return thrust::abs(z);
}

// 定义一个模板函数，用于在 CUDA 环境中计算复数的指数函数
template <typename T>
SPECFUN_HOST_DEVICE complex<T> exp(const complex<T> &z) {
    return thrust::exp(z);
}
// 定义一个返回复数 z 的自然对数的函数
SPECFUN_HOST_DEVICE complex<T> log(const complex<T> &z) {
    // 调用 thrust 库中的 log 函数计算复数 z 的对数
    return thrust::log(z);
}

// 定义一个返回复数 z 的模长的函数
template <typename T>
SPECFUN_HOST_DEVICE T norm(const complex<T> &z) {
    // 调用 thrust 库中的 norm 函数计算复数 z 的模长
    return thrust::norm(z);
}

// 定义一个返回复数 z 的平方根的函数
template <typename T>
SPECFUN_HOST_DEVICE complex<T> sqrt(const complex<T> &z) {
    // 调用 thrust 库中的 sqrt 函数计算复数 z 的平方根
    return thrust::sqrt(z);
}

// 定义一个返回复数 z 的共轭的函数
template <typename T>
SPECFUN_HOST_DEVICE complex<T> conj(const complex<T> &z) {
    // 调用 thrust 库中的 conj 函数计算复数 z 的共轭
    return thrust::conj(z);
}

// 定义一个计算复数 x 的复数 y 次方的函数
template <typename T>
SPECFUN_HOST_DEVICE complex<T> pow(const complex<T> &x, const complex<T> &y) {
    // 调用 thrust 库中的 pow 函数计算复数 x 的复数 y 次方
    return thrust::pow(x, y);
}

// 定义一个计算复数 x 的实数 y 次方的函数
template <typename T>
SPECFUN_HOST_DEVICE complex<T> pow(const complex<T> &x, const T &y) {
    // 调用 thrust 库中的 pow 函数计算复数 x 的实数 y 次方
    return thrust::pow(x, y);
}

// 使用 cuda::std 命名空间中的 is_floating_point、pair 和 uint64_t
using cuda::std::is_floating_point;
using cuda::std::pair;
using cuda::std::uint64_t;

// 定义 SPECFUN_ASSERT 宏
#define SPECFUN_ASSERT(a)

// 结束 std 命名空间
} // namespace std

// 如果不在 CUDA 编译环境中
#else

// 定义 SPECFUN_HOST_DEVICE 为空
#define SPECFUN_HOST_DEVICE

// 包含以下标准库头文件
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstddef>
#include <iterator>
#include <limits>
#include <math.h>
#include <type_traits>
#include <utility>

// 如果定义了 DEBUG 宏，则定义 SPECFUN_ASSERT 宏为 assert(a)
#ifdef DEBUG
#define SPECFUN_ASSERT(a) assert(a)
// 否则，定义 SPECFUN_ASSERT 宏为空
#else
#define SPECFUN_ASSERT(a)
#endif

// 结束条件编译块
#endif
```