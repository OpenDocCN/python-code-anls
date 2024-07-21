# `.\pytorch\torch\csrc\utils\cpp_stacktraces.h`

```py
#pragma once

# 使用 `#pragma once` 来确保头文件只被包含一次，防止多重包含问题


#include <torch/csrc/Export.h>
#include <torch/csrc/profiler/unwind/unwind.h>

# 包含 Torch 库的导出头文件和展开库的头文件


namespace torch {

# 定义命名空间 `torch`


TORCH_API bool get_cpp_stacktraces_enabled();
TORCH_API torch::unwind::Mode get_symbolize_mode();

# 声明 `torch` 命名空间中的两个函数：`get_cpp_stacktraces_enabled()` 返回布尔值，用于获取 C++ 堆栈跟踪是否启用；`get_symbolize_mode()` 返回 `torch::unwind::Mode` 枚举类型，用于获取符号化模式


} // namespace torch

# 结束 `torch` 命名空间
```