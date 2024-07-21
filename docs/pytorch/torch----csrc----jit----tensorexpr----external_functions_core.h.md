# `.\pytorch\torch\csrc\jit\tensorexpr\external_functions_core.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/ATen.h>
// 包含 ATen 库的头文件

#include <ATen/Parallel.h>
// 包含 ATen 并行处理的头文件

#include <torch/csrc/Export.h>
// 包含 Torch 导出相关的头文件

#include <cstdint>
// 包含 C++ 标准整数类型的头文件

namespace torch {
namespace jit {
namespace tensorexpr {

#ifdef C10_MOBILE
extern "C" {
#endif
// 如果在移动平台上，使用 C 语言编译

void DispatchParallel(
    int8_t* func,
    int64_t start,
    int64_t stop,
    int8_t* packed_data) noexcept;
// 声明 DispatchParallel 函数，接受函数指针、起始和结束索引以及打包数据，无异常抛出

TORCH_API void nnc_aten_free(int64_t bufs_num, void** ptrs) noexcept;
// 声明 nnc_aten_free 函数，接受缓冲区数量和指向指针数组的指针，无异常抛出

#ifdef C10_MOBILE
} // extern "C"
#endif
// 结束 C 语言编译的外部声明

} // namespace tensorexpr
} // namespace jit
} // namespace torch
// 命名空间封闭
```