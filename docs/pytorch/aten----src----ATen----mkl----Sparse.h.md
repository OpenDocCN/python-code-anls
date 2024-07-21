# `.\pytorch\aten\src\ATen\mkl\Sparse.h`

```py
// 声明编译器指令，指定此处代码仅编译一次，类似于头文件保护
#pragma once

// 包含 ATen 库的配置文件
#include <ATen/Config.h>

// 如果 MKL 在当前平台下启用
#if AT_MKL_ENABLED()
// 定义宏 AT_USE_MKL_SPARSE，使其返回值为 1
#define AT_USE_MKL_SPARSE() 1
// 否则，如果 MKL 在当前平台下未启用
#else
// 定义宏 AT_USE_MKL_SPARSE，使其返回值为 0
#define AT_USE_MKL_SPARSE() 0
// 结束条件编译指令
#endif
```