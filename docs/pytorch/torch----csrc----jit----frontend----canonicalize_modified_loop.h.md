# `.\pytorch\torch\csrc\jit\frontend\canonicalize_modified_loop.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <memory>
// 包含内存管理相关的头文件

#include <torch/csrc/Export.h>
// 包含导出相关的头文件，用于定义导出符号

namespace torch {
namespace jit {

struct Graph;
// 声明一个名为 Graph 的结构体，用于表示计算图

// Transforms loops so that they can be represented as python
// for or while loops
// 将循环转换为可以表示为 Python 的 for 或 while 循环

TORCH_API void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph);
// 声明一个函数 CanonicalizeModifiedLoops，接受一个 Graph 的共享指针作为参数，
// 用于将修改过的循环转换为标准形式

} // namespace jit
} // namespace torch
```