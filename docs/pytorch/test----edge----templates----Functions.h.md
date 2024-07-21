# `.\pytorch\test\edge\templates\Functions.h`

```py
// clang-format off
// 禁用 clang 格式化器，保持原始格式
#pragma once
// 一次性导入 ATen 库的多个头文件
#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
// ${generated_comment}
// 包含生成的注释（在实际代码中会被替换）
${static_dispatch_extra_headers}

// 命名空间 torch 下的 executor 命名空间
namespace torch {
namespace executor {

// 声明在 executor 命名空间中的函数，具体实现未列出
${Functions_declarations}

} // namespace executor
} // namespace torch
```