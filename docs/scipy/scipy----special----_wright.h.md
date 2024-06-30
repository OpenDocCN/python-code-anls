# `D:\src\scipysrc\scipy\scipy\special\_wright.h`

```
#ifndef WRIGHT_H
#define WRIGHT_H

// 如果 WRIGHT_H 宏未定义，则定义 WRIGHT_H，避免头文件重复包含造成的问题


#ifdef __cplusplus
#define EXTERN_C_START extern "C" {
#define EXTERN_C_END }
#else
#define EXTERN_C_START
#define EXTERN_C_END
#endif

// 如果是 C++ 编译环境，将 EXTERN_C_START 宏定义为 extern "C" {，表示使用 C 语言调用约定
// 如果不是 C++ 编译环境，将 EXTERN_C_START 和 EXTERN_C_END 宏定义为空


#include <Python.h>
#include <complex>

#include "wright.hh"

// 包含 Python.h 头文件，用于与 Python 解释器交互
// 包含 complex 头文件，定义了复数类型和操作
// 包含 wright.hh 头文件，引入 Wright Omega 函数的声明和定义


EXTERN_C_START

#include <numpy/npy_math.h>

npy_cdouble wrightomega(npy_cdouble zp);
double wrightomega_real(double x);

EXTERN_C_END

// 在 extern "C" {...} 块内部声明了以下内容：
// 包含 numpy/npy_math.h 头文件，提供了与数学相关的宏定义和函数声明
// 声明了两个函数 wrightomega 和 wrightomega_real，前者接受 npy_cdouble 类型的参数并返回 npy_cdouble，后者接受 double 类型参数并返回 double


#endif

// 结束条件编译指令，确保本头文件内容只被包含一次，避免重复定义问题
```