# `.\pytorch\aten\src\ATen\templates\NativeFunctions.h`

```py
#pragma once

// ${generated_comment}  // 插入由代码生成的注释内容，具体内容动态生成

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,            \
  meaning the file will need to be re-compiled every time an operator     \
  is changed or added. Consider if your change would be better placed in  \
  another file, or if a more specific header might achieve the same goal. \
  See NOTE: [Tensor vs. TensorBase]
#endif

#if defined(AT_PER_OPERATOR_HEADERS) && defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)
#error This change adds a dependency on all pytorch operators, meaning the      \
  file will need to be re-compiled every time an operator is changed or added.  \
  Consider including a specific operator from <ATen/ops/{my_operator}_native.h> \
  and see NOTE [TORCH_ASSERT_ONLY_METHOD_OPERATORS].
#endif

#include <c10/core/Scalar.h>         // 包含 C10 标量定义
#include <c10/core/Storage.h>        // 包含 C10 存储定义
#include <c10/core/TensorOptions.h>  // 包含 C10 张量选项定义
#include <c10/util/Deprecated.h>     // 包含 C10 废弃功能支持
#include <c10/util/Optional.h>       // 包含 C10 可选值支持
#include <c10/core/QScheme.h>        // 包含 C10 量化方案支持
#include <ATen/core/Reduction.h>     // 包含 ATen 缩减操作定义
#include <ATen/core/Tensor.h>        // 包含 ATen 张量定义
#include <tuple>                     // 包含标准元组库
#include <vector>                    // 包含标准向量库

${NativeFunctions_includes}        // 包含 NativeFunctions 的额外依赖头文件

${NativeFunctions_declarations}    // 声明 NativeFunctions 中的函数声明
```