# `D:\src\scipysrc\scipy\scipy\special\_faddeeva.h`

```
#ifndef FADDEEVA_H_  // 如果未定义 FADDEEVA_H_ 宏，则开始条件编译防护
#define FADDEEVA_H_

#ifdef __cplusplus  // 如果是 C++ 编译环境，则进行下面的定义
#define EXTERN_C_START extern "C" {  // 使用 extern "C" 包裹以支持 C++ 调用
#define EXTERN_C_END }  // 结束 extern "C" 块
#else
#define EXTERN_C_START  // 如果不是 C++ 编译环境，清空宏定义
#define EXTERN_C_END
#endif

#include <Python.h>  // 引入 Python C API 头文件
#include <complex>   // 引入复数计算库

#include "Faddeeva.hh"  // 引入 Faddeeva.hh 头文件

EXTERN_C_START  // 开始 extern "C" 块，用于支持 C 调用

#include <numpy/npy_math.h>  // 引入 NumPy 数学函数头文件

npy_cdouble faddeeva_w(npy_cdouble zp);  // 声明 Faddeeva 函数 w 的原型
npy_cdouble faddeeva_erf(npy_cdouble zp);  // 声明 Faddeeva 函数 erf 的原型

double faddeeva_erfc(double x);  // 声明 Faddeeva 函数 erfc 的原型
npy_cdouble faddeeva_erfc_complex(npy_cdouble zp);  // 声明 Faddeeva 函数 erfc 的复数版本的原型

double faddeeva_erfcx(double x);  // 声明 Faddeeva 函数 erfcx 的原型
npy_cdouble faddeeva_erfcx_complex(npy_cdouble zp);  // 声明 Faddeeva 函数 erfcx 的复数版本的原型

double faddeeva_erfi(double zp);  // 声明 Faddeeva 函数 erfi 的原型
npy_cdouble faddeeva_erfi_complex(npy_cdouble zp);  // 声明 Faddeeva 函数 erfi 的复数版本的原型

double faddeeva_dawsn(double zp);  // 声明 Faddeeva 函数 dawsn 的原型
npy_cdouble faddeeva_dawsn_complex(npy_cdouble zp);  // 声明 Faddeeva 函数 dawsn 的复数版本的原型

npy_cdouble faddeeva_ndtr(npy_cdouble zp);  // 声明 Faddeeva 函数 ndtr 的原型

double faddeeva_log_ndtr(double x);  // 声明 Faddeeva 函数 log_ndtr 的原型
npy_cdouble faddeeva_log_ndtr_complex(npy_cdouble zp);  // 声明 Faddeeva 函数 log_ndtr 的复数版本的原型

double faddeeva_voigt_profile(double x, double sigma, double gamma);  // 声明 Voigt profile 函数的原型

EXTERN_C_END  // 结束 extern "C" 块

#endif  // 结束条件编译防护，定义结束
```