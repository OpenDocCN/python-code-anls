# `.\pytorch\aten\src\ATen\templates\RedispatchFunctions.h`

```py
#pragma once

// ${generated_comment}  // 插入生成注释，通常用于自动生成的代码中

#ifdef TORCH_ASSERT_ONLY_METHOD_OPERATORS
#error This change adds a dependency on all pytorch operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider using the at::_ops::{name}::redispatch() interface by including     \
  the specific operator from <ATen/ops/{my_operator}_ops.h>
#endif

#include <c10/core/Scalar.h>  // 包含标量的定义
#include <ATen/Tensor.h>  // 包含张量的定义
#include <c10/core/Storage.h>  // 包含存储器的定义
#include <ATen/core/Generator.h>  // 包含生成器的定义
#include <c10/util/Deprecated.h>  // 包含已弃用功能的工具
#include <ATen/DeviceGuard.h>  // 包含设备守卫的定义
#include <c10/core/TensorOptions.h>  // 包含张量选项的定义
#include <ATen/core/Reduction.h>  // 包含减少操作的定义
#include <c10/util/Optional.h>  // 包含可选类型的定义
#include <ATen/TensorUtils.h>  // 包含张量工具的定义
#include <ATen/Context.h>  // 包含运行时上下文的定义
#include <ATen/TracerMode.h>  // 包含跟踪模式的定义
#include <ATen/Operators.h>  // 包含操作符的定义

namespace at {

namespace redispatch {
    ${function_redispatch_definitions}  // 插入函数重新分派的定义
} // namespace redispatch

}  // namespace at
```