# `.\pytorch\torch\csrc\api\include\torch\torch.h`

```
#pragma once
// 使用#pragma once指令确保头文件只被包含一次，防止重复定义

#include <torch/all.h>
// 包含torch库的所有头文件，这些头文件提供了Torch C++库的全部功能

#ifdef TORCH_API_INCLUDE_EXTENSION_H
// 如果定义了宏TORCH_API_INCLUDE_EXTENSION_H，则执行以下操作，用于包含Torch的扩展功能

#include <torch/extension.h>
// 包含Torch的扩展头文件，这些头文件用于开发Torch的扩展功能

#endif // defined(TORCH_API_INCLUDE_EXTENSION_H)
// 结束条件编译指令块，确保在未定义TORCH_API_INCLUDE_EXTENSION_H时不包含这些扩展功能的头文件
```