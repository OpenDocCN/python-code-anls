# `.\pytorch\aten\src\ATen\mkl\Limits.h`

```
#pragma once
// 包含 MKL 库的类型定义头文件
#include <mkl_types.h>

namespace at::native {

  // 定义 MKL_LONG 的最大值常量
  // 因为 MKL_LONG 的大小在不同平台上有所变化（例如 Linux 64 位、Windows 32 位），所以需要程序计算其最大值
  constexpr int64_t MKL_LONG_MAX = ((1LL << (sizeof(MKL_LONG) * 8 - 2)) - 1) * 2 + 1;

} // namespace at::native
```