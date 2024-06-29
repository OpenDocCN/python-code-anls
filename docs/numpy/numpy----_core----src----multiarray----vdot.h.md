# `.\numpy\numpy\_core\src\multiarray\vdot.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_VDOT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_VDOT_H_

// 包含 "common.h" 头文件，这是当前头文件的依赖文件
#include "common.h"

// 声明 CFLOAT_vdot 函数，计算复数浮点数向量的点积
NPY_NO_EXPORT void
CFLOAT_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

// 声明 CDOUBLE_vdot 函数，计算双精度浮点数向量的点积
NPY_NO_EXPORT void
CDOUBLE_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

// 声明 CLONGDOUBLE_vdot 函数，计算长双精度浮点数向量的点积
NPY_NO_EXPORT void
CLONGDOUBLE_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

// 声明 OBJECT_vdot 函数，计算对象类型向量的点积
NPY_NO_EXPORT void
OBJECT_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

// 结束条件：关闭 NUMPY_CORE_SRC_MULTIARRAY_VDOT_H_ 宏定义
#endif  /* NUMPY_CORE_SRC_MULTIARRAY_VDOT_H_ */
```