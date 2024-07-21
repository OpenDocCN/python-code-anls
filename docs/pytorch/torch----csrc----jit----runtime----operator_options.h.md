# `.\pytorch\torch\csrc\jit\runtime\operator_options.h`

```py
#pragma once
// 预处理指令，表示该头文件只包含一次，防止多次包含同一文件
#include <ATen/core/dispatch/OperatorOptions.h>
// 包含 ATen 库中的 OperatorOptions.h 文件，其中定义了运算符选项相关内容

namespace torch::jit {
// 进入 torch::jit 命名空间

using AliasAnalysisKind = c10::AliasAnalysisKind;
// 定义别名 AliasAnalysisKind，表示 c10::AliasAnalysisKind 类型在 torch::jit 命名空间中也可以用 AliasAnalysisKind 来表示

} // namespace torch::jit
// 退出 torch::jit 命名空间
```