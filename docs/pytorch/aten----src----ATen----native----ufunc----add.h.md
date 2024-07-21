# `.\pytorch\aten\src\ATen\native\ufunc\add.h`

```
#pragma once
// 预处理指令：确保头文件只包含一次

#include <c10/macros/Macros.h>
// 包含 c10 库中的宏定义头文件

#if !defined(__CUDACC__) && !defined(__HIPCC__)
// 如果未定义 __CUDACC__ 和 __HIPCC__ 宏，则包含以下头文件
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
// 包含 ATen 库中的向量化功能和向量类的头文件
#endif

namespace at {
namespace native {
namespace ufunc {

template <typename T>
// 模板函数定义：对于类型 T
C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
  // 函数名：add
  // 参数：self - 第一个操作数，other - 第二个操作数，alpha - 缩放因子
  // 属性：C10_HOST_DEVICE - 指定为主机和设备可调用，C10_ALWAYS_INLINE - 声明为始终内联
  // 效果：返回 self + alpha * other 的结果
  return self + alpha * other;
}

#if !defined(__CUDACC__) && !defined(__HIPCC__)
// 如果未定义 __CUDACC__ 和 __HIPCC__ 宏，则定义以下函数模板
using vec::Vectorized;
template <typename T>
// 模板函数定义：对于类型 T
C10_ALWAYS_INLINE Vectorized<T> add(Vectorized<T> self, Vectorized<T> other, Vectorized<T> alpha) __ubsan_ignore_undefined__ {
  // 函数名：add
  // 参数：self - 第一个向量操作数，other - 第二个向量操作数，alpha - 缩放因子向量
  // 属性：C10_ALWAYS_INLINE - 声明为始终内联
  // 效果：使用 vec 命名空间中的 fmadd 函数对向量进行加法操作和乘法累加，返回结果向量
  return vec::fmadd(other, alpha, self);
}
#endif

}}}  // namespace at::native::ufunc
// 命名空间结束标记
```