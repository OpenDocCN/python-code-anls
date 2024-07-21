# `.\pytorch\torch\csrc\jit\runtime\slice_indices_adjust.h`

```py
#pragma once
// 预处理指令，表示本头文件只包含一次

#include <torch/csrc/Export.h>
// 引入 Torch 的导出头文件

#include <cstddef>
// 引入标准库定义的大小类型，例如 size_t

#include <cstdint>
// 引入标准库定义的整数类型，例如 int64_t

namespace torch::jit {

// 命名空间 torch::jit，包含了下面的函数实现

// 实现了调整切片索引的功能，根据 Python 列表语义调整索引，并返回调整后列表中的元素数量
TORCH_API int64_t slice_indices_adjust(
    int64_t length,    // 原始列表的长度
    int64_t* start,    // 起始索引的指针，会被修改以适应切片逻辑
    int64_t* stop,     // 结束索引的指针，会被修改以适应切片逻辑
    int64_t step       // 步长，用于计算切片间隔
);

} // namespace torch::jit
```