# `.\pytorch\torch\csrc\jit\passes\onnx\prepare_division_for_onnx.h`

```
#pragma once
// 使用 `#pragma once` 预处理指令，确保头文件只被包含一次，提高编译效率

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT 模块的 IR 头文件

namespace torch {
namespace jit {

// 为了在导出为 ONNX 格式时准备除法操作，执行下面的优化
//
// 此优化修正以下问题：
//
// - aten::div(int, int) -> float 是 Python 中的真除法运算符，但 ONNX 不支持，
//   所以我们将整数转换为 FloatTensors
//
TORCH_API void PrepareDivisionForONNX(const std::shared_ptr<Graph>& graph);
// 函数声明：为给定的图对象准备除法操作，以便在导出为 ONNX 时使用

} // namespace jit
} // namespace torch
```