# `.\pytorch\torch\csrc\autograd\grad_mode.h`

```py
#pragma once
// 使用 #pragma once 指令确保头文件只被包含一次，防止多重包含问题

#include <ATen/core/grad_mode.h>
// 包含 ATen 库中的 grad_mode.h 头文件，用于处理梯度模式相关的功能

#include <torch/csrc/Export.h>
// 包含 Torch 库中的 Export.h 头文件，用于导出符号以支持动态链接

namespace torch::autograd {
// 命名空间 torch::autograd，用于组织 autograd 相关的内容

using GradMode = at::GradMode;
// 将 at::GradMode 别名为 GradMode，用于方便使用梯度模式

using AutoGradMode = at::AutoGradMode;
// 将 at::AutoGradMode 别名为 AutoGradMode，用于方便使用自动梯度模式

} // namespace torch::autograd
// 结束命名空间 torch::autograd
```