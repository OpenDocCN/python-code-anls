# `.\pytorch\aten\src\ATen\cuda\ATenCUDAGeneral.h`

```
#pragma once

// 使用 `#pragma once` 预处理指令，确保头文件只被编译一次，提高编译效率


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 包含 CUDA 标准头文件 `<cuda.h>`, `<cuda_runtime.h>`, `<cuda_fp16.h>`，用于 CUDA 编程


#include <c10/macros/Export.h>

// 包含 `<c10/macros/Export.h>` 头文件，用于导出符号定义，与 C10 库相关


// Use TORCH_CUDA_CPP_API or TORCH_CUDA_CU_API for exports from this folder

// 建议使用 `TORCH_CUDA_CPP_API` 或 `TORCH_CUDA_CU_API` 来导出本文件夹中的符号，与 Torch CUDA 相关的 C++ API 或 CU API 导出设置
```