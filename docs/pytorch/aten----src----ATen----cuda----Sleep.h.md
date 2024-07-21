# `.\pytorch\aten\src\ATen\cuda\Sleep.h`

```py
#pragma once
#include <c10/macros/Export.h>  // 包含 C10 库的导出宏定义
#include <cstdint>  // 包含 C++ 标准库中的cstdint头文件，提供整数类型定义

namespace at::cuda {

// 定义在 TORCH_CUDA_CU_API 下的函数 sleep，用于在 CUDA 设备上执行一个自旋的内核
TORCH_CUDA_CU_API void sleep(int64_t cycles);

// 定义在 TORCH_CUDA_CU_API 下的函数 flush_icache，用于在 ROCm 上刷新指令缓存，在 CUDA 上无操作
TORCH_CUDA_CU_API void flush_icache();

}  // namespace at::cuda
```