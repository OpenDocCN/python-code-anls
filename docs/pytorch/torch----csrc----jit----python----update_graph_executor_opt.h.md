# `.\pytorch\torch\csrc\jit\python\update_graph_executor_opt.h`

```py
#pragma once
// 包含 Torch C++ API 中的 Export.h 文件，用于导出符号
#include <torch/csrc/Export.h>

// 定义 torch::jit 命名空间，用于包裹 JIT 编译器相关功能
namespace torch::jit {

// 声明一个 TORCH_API 函数，设置图执行优化选项为指定的布尔值
TORCH_API void setGraphExecutorOptimize(bool o);

// 声明一个 TORCH_API 函数，获取当前图执行优化选项的布尔值
TORCH_API bool getGraphExecutorOptimize();

} // namespace torch::jit
```