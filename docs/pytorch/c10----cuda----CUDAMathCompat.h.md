# `.\pytorch\c10\cuda\CUDAMathCompat.h`

```
#pragma once

/* This file defines math functions compatible across different gpu
 * platforms (currently CUDA and HIP).
 */

#if defined(__CUDACC__) || defined(__HIPCC__)
// 包含所需的头文件和异常处理
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#ifdef __HIPCC__
// 如果是 HIP 编译器，则定义数学函数为内联函数，且设备端可调用
#define __MATH_FUNCTIONS_DECL__ inline C10_DEVICE
#else /* __HIPCC__ */
#ifdef __CUDACC_RTC__
// 如果是 CUDA RTC 编译器，则定义数学函数为主机和设备端都可调用的静态内联函数
#define __MATH_FUNCTIONS_DECL__ C10_HOST_DEVICE
#else /* __CUDACC_RTC__ */
// 否则定义数学函数为静态内联函数，只能在主机端和设备端内联调用
#define __MATH_FUNCTIONS_DECL__ static inline C10_HOST_DEVICE
#endif /* __CUDACC_RTC__ */
#endif /* __HIPCC__ */

// 定义命名空间为 c10::cuda::compat
namespace c10::cuda::compat {

// 实现不同类型的绝对值函数
__MATH_FUNCTIONS_DECL__ float abs(float x) {
  return ::fabsf(x);
}
__MATH_FUNCTIONS_DECL__ double abs(double x) {
  return ::fabs(x);
}

// 实现不同类型的指数函数
__MATH_FUNCTIONS_DECL__ float exp(float x) {
  return ::expf(x);
}
__MATH_FUNCTIONS_DECL__ double exp(double x) {
  return ::exp(x);
}

// 实现不同类型的向上取整函数
__MATH_FUNCTIONS_DECL__ float ceil(float x) {
  return ::ceilf(x);
}
__MATH_FUNCTIONS_DECL__ double ceil(double x) {
  return ::ceil(x);
}

// 实现不同类型的符号保持函数
__MATH_FUNCTIONS_DECL__ float copysign(float x, float y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return ::copysignf(x, y);
#else
  // 在主机端未定义该函数的情况下抛出异常
  TORCH_INTERNAL_ASSERT(
      false, "CUDAMathCompat copysign should not run on the CPU");
#endif
}
__MATH_FUNCTIONS_DECL__ double copysign(double x, double y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return ::copysign(x, y);
#else
  // 在主机端未定义该函数的情况下抛出异常
  TORCH_INTERNAL_ASSERT(
      false, "CUDAMathCompat copysign should not run on the CPU");
#endif
}

// 实现不同类型的向下取整函数
__MATH_FUNCTIONS_DECL__ float floor(float x) {
  return ::floorf(x);
}
__MATH_FUNCTIONS_DECL__ double floor(double x) {
  return ::floor(x);
}

// 实现不同类型的对数函数
__MATH_FUNCTIONS_DECL__ float log(float x) {
  return ::logf(x);
}
__MATH_FUNCTIONS_DECL__ double log(double x) {
  return ::log(x);
}

// 实现不同类型的 log(1+x) 函数
__MATH_FUNCTIONS_DECL__ float log1p(float x) {
  return ::log1pf(x);
}
__MATH_FUNCTIONS_DECL__ double log1p(double x) {
  return ::log1p(x);
}

// 实现不同类型的最大值函数
__MATH_FUNCTIONS_DECL__ float max(float x, float y) {
  return ::fmaxf(x, y);
}
__MATH_FUNCTIONS_DECL__ double max(double x, double y) {
  return ::fmax(x, y);
}

// 实现不同类型的最小值函数
__MATH_FUNCTIONS_DECL__ float min(float x, float y) {
  return ::fminf(x, y);
}
__MATH_FUNCTIONS_DECL__ double min(double x, double y) {
  return ::fmin(x, y);
}

// 实现不同类型的幂函数
__MATH_FUNCTIONS_DECL__ float pow(float x, float y) {
  return ::powf(x, y);
}
__MATH_FUNCTIONS_DECL__ double pow(double x, double y) {
  return ::pow(x, y);
}

// 实现不同类型的正弦余弦函数
__MATH_FUNCTIONS_DECL__ void sincos(float x, float* sptr, float* cptr) {
  return ::sincosf(x, sptr, cptr);
}
__MATH_FUNCTIONS_DECL__ void sincos(double x, double* sptr, double* cptr) {
  return ::sincos(x, sptr, cptr);
}

// 实现不同类型的平方根函数
__MATH_FUNCTIONS_DECL__ float sqrt(float x) {
  return ::sqrtf(x);
}
__MATH_FUNCTIONS_DECL__ double sqrt(double x) {
  return ::sqrt(x);
}

} // namespace c10::cuda::compat
# 定义一个函数 rsqrt，计算浮点数的平方根的倒数，声明为 float 类型
__MATH_FUNCTIONS_DECL__ float rsqrt(float x) {
  return ::rsqrtf(x);
}

# 定义一个函数 rsqrt，计算双精度浮点数的平方根的倒数，声明为 double 类型
__MATH_FUNCTIONS_DECL__ double rsqrt(double x) {
  return ::rsqrt(x);
}

# 定义一个函数 tan，计算浮点数的正切，声明为 float 类型
__MATH_FUNCTIONS_DECL__ float tan(float x) {
  return ::tanf(x);
}

# 定义一个函数 tan，计算双精度浮点数的正切，声明为 double 类型
__MATH_FUNCTIONS_DECL__ double tan(double x) {
  return ::tan(x);
}

# 定义一个函数 tanh，计算浮点数的双曲正切，声明为 float 类型
__MATH_FUNCTIONS_DECL__ float tanh(float x) {
  return ::tanhf(x);
}

# 定义一个函数 tanh，计算双精度浮点数的双曲正切，声明为 double 类型
__MATH_FUNCTIONS_DECL__ double tanh(double x) {
  return ::tanh(x);
}

# 定义一个函数 normcdf，计算浮点数的标准正态分布累积分布函数（CDF），声明为 float 类型
__MATH_FUNCTIONS_DECL__ float normcdf(float x) {
  return ::normcdff(x);
}

# 定义一个函数 normcdf，计算双精度浮点数的标准正态分布累积分布函数（CDF），声明为 double 类型
__MATH_FUNCTIONS_DECL__ double normcdf(double x) {
  return ::normcdf(x);
}
```