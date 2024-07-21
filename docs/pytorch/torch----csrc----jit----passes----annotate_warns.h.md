# `.\pytorch\torch\csrc\jit\passes\annotate_warns.h`

```py
#pragma once

# 指令：#pragma once

#include <torch/csrc/jit/ir/ir.h>

# 包含头文件：引入torch库中的ir.h文件

namespace torch {
namespace jit {

# 命名空间：定义了命名空间torch::jit

TORCH_API void AnnotateWarns(const std::shared_ptr<Graph>& graph);

# 函数声明：声明了一个名为AnnotateWarns的函数，接受一个std::shared_ptr类型的Graph指针参数，并且使用了TORCH_API宏修饰该函数

} // namespace jit
} // namespace torch

# 命名空间结束：结束了命名空间torch::jit
```