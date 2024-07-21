# `.\pytorch\torch\csrc\autograd\InferenceMode.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/core/InferenceMode.h>
// 包含 c10 库的 InferenceMode 头文件

#include <torch/csrc/Export.h>
// 包含 torch 库的 Export 头文件

namespace torch::autograd {
// 进入 torch::autograd 命名空间

using InferenceMode = c10::InferenceMode;
// 使用 c10 命名空间中的 InferenceMode 类，并起一个别名为 InferenceMode

}
// 结束 torch::autograd 命名空间
```