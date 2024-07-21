# `.\pytorch\aten\src\ATen\templates\DispatchKeyNativeFunctions.h`

```
#pragma once
// 如果外部后端在其代码树中生成文件，并使用 clang-format 检查整个代码树的源文件，
// 可能会使用不同的配置。因此，禁用 clang-format。
// clang-format off

// ${generated_comment}
// 包含 ATen 库的 Tensor 头文件

#include <ATen/Tensor.h>

// ${namespace_prologue}
// 命名空间的引入和定义

struct ${class_name} {

${dispatch_declarations}
// 分发函数声明

};
// ${namespace_epilogue}
// 命名空间的尾声
```