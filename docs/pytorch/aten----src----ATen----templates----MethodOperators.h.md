# `.\pytorch\aten\src\ATen\templates\MethodOperators.h`

```
#pragma once

// ${generated_comment}  // 插入生成的注释，可能是自动生成的文件头注释

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,             \
  meaning the file will need to be re-compiled every time an operator      \
  is changed or added. Consider if your change would be better placed in   \
  another file, or if a more specific header might achieve the same goal.  \
  See NOTE: [Tensor vs. TensorBase]
#endif

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>  // 包含 ATen 库的前向声明头文件，用于类型的前向声明

${MethodOperators_includes}  // 插入 MethodOperators_includes 的内容，可能是一些方法操作符的头文件包含

namespace at {
namespace _ops {
${MethodOperators_declarations}  // 插入 MethodOperators_declarations 的内容，可能是一些方法操作符的声明
} // namespace _ops
} // namespace at
```