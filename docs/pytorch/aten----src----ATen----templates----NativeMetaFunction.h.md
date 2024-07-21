# `.\pytorch\aten\src\ATen\templates\NativeMetaFunction.h`

```
#pragma once
// 声明本头文件只被编译一次

// 引入所需的头文件
#include <c10/core/Scalar.h>                 // 包含标量类型相关定义
#include <c10/core/Storage.h>                // 包含存储相关定义
#include <c10/core/TensorOptions.h>          // 包含张量选项相关定义
#include <c10/util/Deprecated.h>             // 包含过时功能相关定义
#include <c10/util/Optional.h>               // 包含可选值相关定义
#include <c10/core/QScheme.h>                // 包含量化方案相关定义
#include <ATen/core/Reduction.h>             // 包含减少操作相关定义
#include <ATen/TensorIterator.h>             // 包含张量迭代器相关定义
#include <ATen/TensorMeta.h>                 // 包含张量元信息相关定义
#include <tuple>                             // 包含元组相关定义
#include <vector>                            // 包含向量相关定义

namespace at {
namespace meta {
// 声明 meta 命名空间下的函数声明

${meta_function_declarations}
// 插入由其它工具生成的函数声明

} // namespace native
} // namespace at
// 结束 native 和 at 命名空间的定义
```