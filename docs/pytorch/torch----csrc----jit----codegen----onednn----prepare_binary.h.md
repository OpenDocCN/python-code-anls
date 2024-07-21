# `.\pytorch\torch\csrc\jit\codegen\onednn\prepare_binary.h`

```
#pragma once
// 预处理二元操作以便用于LLGA

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 准备二元操作以供LLGA使用
//
// 该 pass 执行以下操作：
//
// - 将 aten::add 和 aten::mul 的标量输入转换为具有维度[1]的Float张量
//
// - 当 alpha != 1.0 时，将融合的加法分解为 aten::mul + aten::add
//
// - 消除恒等加法/乘法，即 tensor + 0，tensor * 1
//
void PrepareBinaryForLLGA(const std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```