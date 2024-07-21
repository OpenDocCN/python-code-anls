# `.\pytorch\torch\csrc\jit\codegen\onednn\layout_propagation.h`

```
#pragma once
// 使用 #pragma once 预处理指令确保头文件只被包含一次，避免重复定义错误

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中用于 JIT 的中间表示(IR)相关的头文件

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

void PropagateLayout(const std::shared_ptr<Graph>& graph);
// 声明了一个函数 PropagateLayout，接受一个 std::shared_ptr<Graph> 类型的参数
// 用于在 OneDNN 加速器上传播布局信息

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
// 定义了命名空间，确保 PropagateLayout 函数位于 torch::jit::fuser::onednn 命名空间下
```