# `.\pytorch\aten\src\ATen\core\grad_mode.h`

```py
#pragma once
// 使用 pragma once 指令确保头文件只被包含一次，防止多重包含

#include <c10/macros/Macros.h>
// 包含 c10 库中的 Macros.h 头文件，提供宏定义和宏操作相关功能

#include <c10/core/GradMode.h>
// 包含 c10 库中的 GradMode.h 头文件，定义了与梯度模式相关的功能

namespace at {
  using GradMode = c10::GradMode;
  // 在 at 命名空间中使用 GradMode 别名，指向 c10::GradMode 类型

  using AutoGradMode = c10::AutoGradMode;
  // 在 at 命名空间中使用 AutoGradMode 别名，指向 c10::AutoGradMode 类型

  using NoGradGuard = c10::NoGradGuard;
  // 在 at 命名空间中使用 NoGradGuard 别名，指向 c10::NoGradGuard 类型
}
// 定义命名空间 at，通过 using 关键字引入 c10 命名空间中的 GradMode、AutoGradMode 和 NoGradGuard 类型
```