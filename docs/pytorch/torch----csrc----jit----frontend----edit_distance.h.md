# `.\pytorch\torch\csrc\jit\frontend\edit_distance.h`

```
#pragma once

# 使用 `#pragma once` 指令，确保此头文件只被编译一次，避免重复包含


#include <torch/csrc/Export.h>
#include <cstddef>

# 包含 Torch 库中的 Export 头文件和标准库的 cstddef 头文件


namespace torch {
namespace jit {

# 定义命名空间 torch 下的 jit 子命名空间


TORCH_API size_t ComputeEditDistance(
    const char* word1,
    const char* word2,
    size_t maxEditDistance);

# 声明一个函数 ComputeEditDistance，用于计算两个字符串之间的编辑距离，支持设置最大编辑距离


} // namespace jit
} // namespace torch

# 结束 torch 命名空间和 jit 子命名空间
```