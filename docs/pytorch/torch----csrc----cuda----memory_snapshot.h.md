# `.\pytorch\torch\csrc\cuda\memory_snapshot.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/util/Optional.h>
// 包含 c10 库中的 Optional 头文件

#include <torch/csrc/Export.h>
// 包含 torch 库中的 Export 头文件

#include <cstdint>
// 包含标准整数类型头文件

#include <string>
// 包含标准字符串头文件

namespace torch::cuda {

// 命名空间 torch::cuda

// C++-only versions of these, for python use
// those defined in cuda/Module.cpp which also record python state.

TORCH_CUDA_CU_API void _record_memory_history(
    bool enabled,
    // 函数声明：记录内存历史，接受一个布尔值参数 enabled
    bool record_context = true,
    // 布尔值参数 record_context 默认为 true
    int64_t trace_alloc_max_entries = 1,
    // 整数参数 trace_alloc_max_entries 默认为 1
    bool trace_alloc_record_context = false,
    // 布尔值参数 trace_alloc_record_context 默认为 false
    bool record_cpp_context = false);
    // 布尔值参数 record_cpp_context 默认为 false

TORCH_CUDA_CU_API void _record_memory_history(
    std::optional<std::string> enabled = "all",
    // 重载的函数声明：记录内存历史，接受一个可选的字符串参数 enabled，默认为 "all"
    std::optional<std::string> context = "all",
    // 可选的字符串参数 context，默认为 "all"
    const std::string& stacks = "all",
    // 字符串引用参数 stacks，默认为 "all"
    size_t max_entries = SIZE_MAX);
    // 大小类型参数 max_entries，默认为 SIZE_MAX

TORCH_CUDA_CU_API std::string _memory_snapshot_pickled();
// 函数声明：返回内存快照的字符串表示

} // namespace torch::cuda
// 命名空间结束声明：torch::cuda
```