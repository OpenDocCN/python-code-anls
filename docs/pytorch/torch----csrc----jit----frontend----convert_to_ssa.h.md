# `.\pytorch\torch\csrc\jit\frontend\convert_to_ssa.h`

```py
#pragma once
#include <functional>  // 包含 C++ 标准库中的 functional 头文件
#include <memory>       // 包含 C++ 标准库中的 memory 头文件
#include <string>       // 包含 C++ 标准库中的 string 头文件

#include <torch/csrc/Export.h>  // 包含 Torch 库中的 Export.h 头文件
#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch 库中的 ir.h 头文件

namespace torch {  // 定义 torch 命名空间
namespace jit {    // 定义 jit 命名空间

// 将具有 Loads 和 Stores 的图形转换为 SSA 形式
TORCH_API void ConvertToSSA(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
```