# `.\pytorch\c10\util\generic_math.h`

```
#pragma once

#include <c10/macros/Macros.h>  // 引入宏定义和声明
#include <c10/util/TypeSafeSignMath.h>  // 引入类型安全的符号数学运算
#include <cmath>  // 引入数学函数

#if defined(__CUDA_ARCH__)
#include <c10/cuda/CUDAMathCompat.h>  // CUDA兼容数学函数
#define C10_COMPAT_COPYSIGN c10::cuda::compat::copysign  // 定义C10_COMPAT_COPYSIGN为CUDA兼容的copysign函数
#elif defined(__HIPCC__)
#include <c10/hip/HIPMathCompat.h>  // HIP兼容数学函数
#define C10_COMPAT_COPYSIGN c10::hip::compat::copysign  // 定义C10_COMPAT_COPYSIGN为HIP兼容的copysign函数
#else
#include <c10/util/copysign.h>  // 引入copysign函数
#define C10_COMPAT_COPYSIGN c10::copysign  // 定义C10_COMPAT_COPYSIGN为通用的copysign函数
#endif

// 该文件中的函数应当全部是头文件函数，因为它们在ABI兼容模式下使用。

namespace c10 {

// 注意：[Python中的Floor Division]
// Python的__floordiv__运算符比简单的floor(a / b)更复杂。
// 它旨在保持这个属性：a == (a // b) * b + remainder(a, b)
// 否则可能由于余数的四舍五入误差而失败。
// 因此，它的计算方式为：a // b = (a - remainder(a, b)) / b
// 还要在结果上进行一些额外的修正。
//
// 参考CPython的实现：
// https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636

template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t div_floor_floating(scalar_t a, scalar_t b)
    __ubsan_ignore_float_divide_by_zero__ {  // 忽略浮点数除以零的UBSan报告
  if (C10_UNLIKELY(b == 0)) {  // 如果除数b为0
    // 除以零：返回标准的IEEE结果
    return a / b;
  }

  auto mod = std::fmod(a, b);  // 计算a除以b的余数
  auto div = (a - mod) / b;  // 计算a除以b的整数部分
  if ((mod != 0) && (b < 0) != (mod < 0)) {  // 如果余数不为零并且符号不一致
    div -= scalar_t(1);  // 整数部分减1
  }

  scalar_t floordiv;
  if (div != 0) {
    floordiv = std::floor(div);  // 对整数部分向下取整
    if (div - floordiv > scalar_t(0.5)) {  // 如果小数部分大于0.5
      floordiv += scalar_t(1.0);  // 向上取整
    }
  } else {
    floordiv = C10_COMPAT_COPYSIGN(scalar_t(0), a / b);  // 如果整数部分为零，则根据a/b的符号返回0或者-0
  }
  return floordiv;  // 返回最终整除的结果
}

template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t div_floor_integer(scalar_t a, scalar_t b) {
  if (c10::signs_differ(a, b)) {  // 如果除数和被除数的符号不同
    // 如果截断除法的结果和余数不为零，则从截断除法的结果中减去一
    const auto quot = a / b;  // 截断除法的结果
    const auto rem = a % b;  // 求余数
    return rem ? quot - 1 : quot;  // 如果余数不为零，则结果减一；否则返回截断除法的结果
  }
  return a / b;  // 如果除数和被除数的符号相同，则返回截断除法的结果
}

} // namespace c10
```