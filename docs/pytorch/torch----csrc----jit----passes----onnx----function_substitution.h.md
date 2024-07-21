# `.\pytorch\torch\csrc\jit\passes\onnx\function_substitution.h`

```
#pragma once

// 预处理指令，确保头文件只被包含一次


#include <torch/csrc/jit/ir/ir.h>

// 包含头文件 torch/csrc/jit/ir/ir.h


namespace torch {
namespace jit {

// 命名空间 torch 和 jit 的开始


TORCH_API void ONNXFunctionCallSubstitution(Graph& graph);

// 声明一个名为 ONNXFunctionCallSubstitution 的函数，接受一个 Graph 类型的参数 graph


}
} // namespace torch

// 命名空间 jit 和 torch 的结束
```