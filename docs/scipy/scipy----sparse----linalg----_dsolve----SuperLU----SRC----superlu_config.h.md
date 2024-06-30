# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\superlu_config.h`

```
#ifndef SUPERLU_CONFIG_H
#define SUPERLU_CONFIG_H

/* 定义超级LU配置头文件的宏保护 */

/* 启用metis */
/* #undef HAVE_METIS */

/* 启用colamd */
/* #undef HAVE_COLAMD */

/* 启用64位索引模式 */
/* #undef XSDK_INDEX_SIZE */

/* 用于索引稀疏矩阵元数据结构的整数类型 */
#if (XSDK_INDEX_SIZE == 64)
#include <stdint.h>
#define _LONGINT 1
typedef int64_t int_t;
#else
typedef int int_t; /* 默认 */
#endif

#endif /* SUPERLU_CONFIG_H */

/* 结束超级LU配置头文件的宏保护 */
```