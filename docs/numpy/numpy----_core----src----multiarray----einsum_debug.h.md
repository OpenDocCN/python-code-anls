# `.\numpy\numpy\_core\src\multiarray\einsum_debug.h`

```py
/*
 * This file provides debug macros used by the other einsum files.
 *
 * Copyright (c) 2011 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.
 */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_EINSUM_DEBUG_H_
#define NUMPY_CORE_SRC_MULTIARRAY_EINSUM_DEBUG_H_

/********** PRINTF DEBUG TRACING **************/
// 定义调试输出级别，0 表示关闭调试输出
#define NPY_EINSUM_DBG_TRACING 0

// 如果开启了调试输出
#if NPY_EINSUM_DBG_TRACING
// 包含标准输出头文件
#include <cstdio>
// 定义输出宏，打印字符串
#define NPY_EINSUM_DBG_PRINT(s) printf("%s", s);
// 定义输出宏，打印带有一个参数的格式化字符串
#define NPY_EINSUM_DBG_PRINT1(s, p1) printf(s, p1);
// 定义输出宏，打印带有两个参数的格式化字符串
#define NPY_EINSUM_DBG_PRINT2(s, p1, p2) printf(s, p1, p2);
// 定义输出宏，打印带有三个参数的格式化字符串
#define NPY_EINSUM_DBG_PRINT3(s, p1, p2, p3) printf(s);
// 如果未开启调试输出，则定义这些宏为空
#else
#define NPY_EINSUM_DBG_PRINT(s)
#define NPY_EINSUM_DBG_PRINT1(s, p1)
#define NPY_EINSUM_DBG_PRINT2(s, p1, p2)
#define NPY_EINSUM_DBG_PRINT3(s, p1, p2, p3)
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_EINSUM_DEBUG_H_ */
```